#!/usr/bin/env python3
"""
Uniform layout perturbations for parser-robustness benchmarking.

Perturbations (relative strengths, identical for every dataset):
  block_displacement : shift one text region by up to k * page_width
  line_compression   : shrink every inter-line gap to (1 - k) * original gap

Region / line source (--regions auto|annotations|heuristics):
  annotations : CORD valid_line/group_id, FUNSD form/word boxes
  heuristics  : Otsu ink mask + morphology + row projection
  auto        : annotations if the JSON is found, else heuristics

Design notes
------------
* All perturbations are applied in image space; every parser receives the
  same image. Size is preserved (no padding), so PSNR/SSIM are comparable.
* Region candidates are filtered for *available room*, not for a fixed width
  band. CORD receipts are tightly cropped; a candidate at the page border
  cannot be shifted without clipping, so we choose the direction with the
  most space and clamp the shift to the available room.
* Line compression accumulates fractional shifts and rounds once, then
  clamps *per line* (na = max(0, a - d)). The previous global clamp by
  bands[0][0] zeroed every shift when the first band touched the top
  border, which is the normal case for tight receipts.
* Background is the mode of a 2-pixel border by default. The old non-ink
  median mis-estimates the bg on photographed receipts with uneven
  illumination.
* FSIM is the runtime bottleneck. Use --fsim-strengths to sample it, and
  install pyfftw (`pip install pyfftw`) for a 3-10x speedup. Pass
  --fsim-strengths "" to skip FSIM entirely for a fast pilot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import torch
from piq import fsim


import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# ---------------------------------------------------------------- config
PERTURBATIONS = {
    "block_displacement": [0.01, 0.02, 0.03],
    # 0.02..0.10: lower bound was already chosen to keep SSIM > 0.95.
    # With the max-shift bug fixed, 0.02 now produces a >=1 px total shift
    # on typical receipts; 0.10 is the ceiling before it becomes obvious.
    "line_compression":   [0.02, 0.05, 0.10],
}

DATASET_GLOBS = {
    "FUNSD": ["testing_data/images/*.png", "testing_data/images/*.jpg", "images/*.png"],
    "CORD":  ["test/image/*.png", "test/images/*.png", "test/image/*.jpg", "image/*.png"],
    "SROIE": ["test/images/*.jpg", "test/images/*.png", "images/*.jpg", "images/*.png", "*/*.jpg"],
}

MANIFEST_FIELDS = [
    "dataset", "split", "image_id", "perturbation", "strength",
    "clean_path", "perturbed_path", "psnr", "ssim", "fsim",
    "region_source", "detail",
]

SKIPPED_FIELDS = [
    "dataset", "image_id", "perturbation", "strength", "reason", "detail",
]


# ---------------------------------------------------------------- IO
def imread(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def otsu_ink(gray: np.ndarray) -> np.ndarray:
    _, binv = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binv


def estimate_bg(bgr: np.ndarray, strategy: str = "border") -> np.ndarray:
    """Robust background estimate.

    border : per-channel mode of a 2-pixel border ring (robust to tight
             crops and photographed receipts; text rarely fills the ring)
    nonink : median of all non-ink pixels (old behaviour)
    """
    if strategy == "nonink":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ink = otsu_ink(gray) > 0
        bg_pixels = bgr[~ink]
        if bg_pixels.size == 0:
            return np.array([255, 255, 255], np.uint8)
        return np.median(bg_pixels, axis=0).astype(np.uint8)

    H, W = bgr.shape[:2]
    ring = np.concatenate([
        bgr[0:2, :, :].reshape(-1, 3),
        bgr[-2:, :, :].reshape(-1, 3),
        bgr[:, 0:2, :].reshape(-1, 3),
        bgr[:, -2:, :].reshape(-1, 3),
    ], axis=0)
    if ring.size == 0:
        return np.array([255, 255, 255], np.uint8)

    mode = []
    for c in range(3):
        vals = ring[:, c].astype(np.int32)
        counts = np.bincount(vals, minlength=256)
        mode.append(int(counts.argmax()))
    return np.array(mode, dtype=np.uint8)


# ---------------------------------------------------------------- annotations
def _find_json(img_path: Path, dataset: str) -> Path | None:
    stem = img_path.stem
    cands: list[Path] = []
    if dataset.upper() == "CORD":
        # data/CORD/test/image/x.png -> data/CORD/test/json/x.json
        cands += [
            img_path.parent.parent / "json" / f"{stem}.json",
            img_path.parent / "json" / f"{stem}.json",
            img_path.with_suffix(".json"),
        ]
    elif dataset.upper() == "FUNSD":
        # data/FUNSD/testing_data/images/x.png -> .../annotations/x.json
        # (also accept the misspelled 'anotations' folder)
        cands += [
            img_path.parent.parent / "annotations" / f"{stem}.json",
            img_path.parent.parent / "anotations" / f"{stem}.json",
            img_path.parent / "annotations" / f"{stem}.json",
            img_path.parent / "anotations" / f"{stem}.json",
            img_path.with_suffix(".json"),
        ]
    else:
        cands += [
            img_path.with_suffix(".json"),
            img_path.parent / f"{stem}.json",
        ]
    for p in cands:
        if p.exists():
            return p
    return None


def _quad_to_box(quad: dict) -> tuple[int, int, int, int]:
    xs = [quad[k] for k in ("x1", "x2", "x3", "x4") if k in quad]
    ys = [quad[k] for k in ("y1", "y2", "y3", "y4") if k in quad]
    if not xs or not ys:
        return (0, 0, 0, 0)
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def _union_box(boxes):
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _cluster_word_boxes_into_lines(boxes, gap_frac: float = 0.5):
    """Group word boxes into text-line boxes by y-centre proximity."""
    if not boxes:
        return []
    heights = [b[3] - b[1] + 1 for b in boxes]
    med_h = int(np.median(heights)) if heights else 10
    gap_tol = max(2, int(gap_frac * med_h))

    items = sorted(boxes, key=lambda b: ((b[1] + b[3]) / 2, b[0]))
    lines, cur = [], [items[0]]
    for b in items[1:]:
        cy_prev = (cur[-1][1] + cur[-1][3]) / 2
        cy = (b[1] + b[3]) / 2
        if abs(cy - cy_prev) <= gap_tol:
            cur.append(b)
        else:
            lines.append(_union_box(cur))
            cur = [b]
    lines.append(_union_box(cur))
    return lines


def load_annotations(img_path: Path, dataset: str) -> dict | None:
    """Return {'blocks': [box], 'lines': [box], 'json_path': str} or None.

    Blocks are semantic grouping regions; lines are individual text lines.
    Both lists are sorted top-to-bottom.
    """
    json_path = _find_json(img_path, dataset)
    if json_path is None:
        return None
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    blocks, lines = [], []

    if dataset.upper() == "CORD":
        # CORD image annotation: {"valid_line": [{"words": [...], "group_id": ...}, ...]}
        groups: dict = {}
        for vl in data.get("valid_line", []):
            word_boxes = []
            for w in vl.get("words", []):
                q = w.get("quad")
                if q:
                    word_boxes.append(_quad_to_box(q))
            if not word_boxes:
                continue
            line_box = _union_box(word_boxes)
            lines.append(line_box)
            gid = vl.get("group_id", vl.get("sub_group_id", 0))
            groups.setdefault(gid, []).append(line_box)
        for boxes in groups.values():
            blocks.append(_union_box(boxes))

    elif dataset.upper() == "FUNSD":
        # FUNSD: {"form": [{"box": [x0,y0,x1,y1], "words": [{"box": [...]}, ...]}, ...]}
        word_boxes = []
        for form in data.get("form", []):
            box = form.get("box")
            if isinstance(box, list) and len(box) == 4:
                blocks.append(tuple(int(v) for v in box))
            for w in form.get("words", []):
                wb = w.get("box")
                if isinstance(wb, list) and len(wb) == 4:
                    word_boxes.append(tuple(int(v) for v in wb))
        lines = _cluster_word_boxes_into_lines(word_boxes)

    elif dataset.upper() == "SROIE":
        # SROIE uses .txt files for boxes: x1,y1,x2,y2,x3,y3,x4,y4,text
        # Look for a 'box' or 'boxes' directory next to the images
        txt_cands = [
            img_path.parent.parent / "box" / f"{img_path.stem}.txt",
            img_path.parent.parent / "boxes" / f"{img_path.stem}.txt",
            img_path.parent / f"{img_path.stem}.txt",
        ]
        txt_path = next((p for p in txt_cands if p.exists()), None)

        if txt_path:
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 9:
                        try:
                            coords = list(map(int, parts[:8]))
                            xs, ys = coords[0::2], coords[1::2]
                            box = (min(xs), min(ys), max(xs), max(ys))
                            lines.append(box)
                        except ValueError:
                            continue
            lines = sorted(lines, key=lambda b: (b[1], b[0]))
            # SROIE doesn't have explicit "blocks" (like paragraphs).
            # Treat the entire receipt as one block.
            blocks = []

    if not blocks and not lines:
        return None

    def _sort(lst):
        return sorted(lst, key=lambda b: (b[1], b[0]))

    return {"blocks": _sort(blocks), "lines": _sort(lines),
            "json_path": str(json_path)}


# ---------------------------------------------------------------- heuristics
def detect_lines(gray: np.ndarray, min_ink_frac: float = 0.004,
                 merge_gap: int = 3):
    """Row-projection text lines -> [(y0, y1), ...] top-to-bottom."""
    ink = otsu_ink(gray) > 0
    H, W = ink.shape
    row_ink = ink.sum(axis=1)
    thresh = max(2, int(min_ink_frac * W))
    active = row_ink > thresh

    lines, y = [], 0
    while y < H:
        if not active[y]:
            y += 1
            continue
        y0 = y1 = y
        gap = 0
        while y < H:
            if active[y]:
                y1, gap = y, 0
            else:
                gap += 1
                if gap > merge_gap:
                    break
            y += 1
        if y1 - y0 + 1 >= 3:
            lines.append((y0, y1))
    return lines


def detect_lines_adaptive(gray: np.ndarray, min_lines: int = 3):
    """Try increasing ink thresholds until we find at least `min_lines`."""
    for frac, gap in [(0.004, 3), (0.01, 2), (0.02, 1), (0.03, 1)]:
        lines = detect_lines(gray, min_ink_frac=frac, merge_gap=gap)
        if len(lines) >= min_lines:
            return lines
    return detect_lines(gray)  # best effort


def detect_regions(gray: np.ndarray):
    """Morphology-based text regions -> [(x0, y0, x1, y1), ...]."""
    ink = otsu_ink(gray)
    H, W = ink.shape
    lines = detect_lines_adaptive(gray)
    if not lines:
        return []
    med_h = int(np.median([b - a + 1 for a, b in lines]))
    kx = max(3, med_h // 2)
    ky = max(2, med_h // 6)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    dil = cv2.dilate(ink, kernel, iterations=1)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 0.03 * W or h < 0.5 * med_h:
            continue
        if w > 0.98 * W:          # skip whole-page blocks
            continue
        boxes.append((x, y, x + w - 1, y + h - 1))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


# ---------------------------------------------------------------- helpers
def _paste_clipped(out: np.ndarray, crop: np.ndarray, na: int, xa: int) -> None:
    """Paste crop at (na, xa), clipping at the image borders."""
    H, W = out.shape[:2]
    h, w = crop.shape[:2]
    if na < 0:
        crop = crop[-na:, :]
        na = 0
        h = crop.shape[0]
    if na + h > H:
        crop = crop[:H - na, :]
        h = crop.shape[0]
    if xa < 0:
        crop = crop[:, -xa:]
        xa = 0
        w = crop.shape[1]
    if xa + w > W:
        crop = crop[:, :W - xa]
        w = crop.shape[1]
    if crop.size == 0:
        return
    out[na:na + h, xa:xa + w] = crop


# ---------------------------------------------------------------- perturb 1
def perturb_block_displacement(bgr, strength, rng, bg,
                               regions=None, source="heuristic"):
    """Shift one region horizontally by up to strength*W, choosing the
    direction with room and clamping to the available space."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    if regions is None:
        regions = detect_regions(gray)
    if not regions:
        return None, {"reason": "no_candidate_region",
                      "region_source": source, "n_regions": 0}

    # Keep candidates with reasonable size and at least 1 px of room.
    cands = []
    for (x0, y0, x1, y1) in regions:
        if (y1 - y0 + 1) < 6:
            continue
        if (x1 - x0 + 1) < 4:
            continue
        room_l = x0
        room_r = (W - 1) - x1
        best = max(room_l, room_r)
        if best < 1:
            continue
        cands.append((x0, y0, x1, y1, room_l, room_r, best))

    if not cands:
        return None, {"reason": "all_clipped", "region_source": source,
                      "n_regions": len(regions), "max_room": 0}

    target_px = max(1, int(round(strength * W)))

    # Prefer candidates with room >= target_px; fall back to the roomiest.
    spacious = [c for c in cands if c[6] >= target_px]
    if spacious:
        rng.shuffle(spacious)
        pool = spacious
    else:
        pool = sorted(cands, key=lambda c: -c[6])[:10]
        rng.shuffle(pool)

    for (x0, y0, x1, y1, room_l, room_r, _best) in pool:
        direction = +1 if room_r >= room_l else -1
        room = room_r if direction > 0 else room_l
        dx_mag = min(target_px, room)
        if dx_mag < 1:
            continue
        dx = direction * dx_mag

        out = bgr.copy()
        patch = bgr[y0:y1 + 1, x0:x1 + 1].copy()
        out[y0:y1 + 1, x0:x1 + 1] = bg
        _paste_clipped(out, patch, y0, x0 + dx)

        if np.array_equal(out, bgr):
            continue

        return out, {
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "dx_px": int(dx),
            "dx_frac": float(abs(dx) / W),
            "target_px": int(target_px),
            "room_l": int(room_l), "room_r": int(room_r),
            "n_candidates": len(cands),
            "region_source": source,
        }

    return None, {"reason": "all_clipped", "region_source": source,
                  "n_candidates": len(cands),
                  "max_room": int(max(c[6] for c in cands))}


# ---------------------------------------------------------------- perturb 2
def perturb_line_compression(bgr, strength, bg, margin=1,
                             lines=None, source="heuristic"):
    """Compress every inter-line gap by `strength`.

    `lines` is a list of (x0, y0, x1, y1) boxes, sorted top-to-bottom.
    When x0/x1 are None (heuristic mode), the horizontal ink extent is used.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = otsu_ink(gray) > 0
    H, W = gray.shape

    if lines is None:
        raw = detect_lines_adaptive(gray)
        lines = [(None, y0, None, y1) for (y0, y1) in raw]

    if len(lines) < 3:
        return None, {"reason": "too_few_lines", "n_lines": len(lines),
                      "region_source": source}

    lines = sorted(lines, key=lambda b: (b[1], b[0]))

    # Build bands (a, b, xa, xb) and crops.
    bands, crops = [], []
    for (x0, y0, x1, y1) in lines:
        a = max(0, y0 - margin)
        b = min(H - 1, y1 + margin)
        if x0 is None:
            cols = np.where(ink[a:b + 1].any(axis=0))[0]
            if cols.size:
                xa = max(0, int(cols.min()) - margin)
                xb = min(W - 1, int(cols.max()) + margin)
            else:
                xa, xb = 0, W - 1
        else:
            xa = max(0, x0 - margin)
            xb = min(W - 1, x1 + margin)
        bands.append((a, b, xa, xb))
        crops.append(bgr[a:b + 1, xa:xb + 1].copy())

    # Gaps from *unexpanded* line positions.
    ys = [(y0, y1) for (_, y0, _, y1) in lines]
    deltas_f = [0.0]
    for i in range(1, len(ys)):
        gap = max(0, ys[i][0] - ys[i - 1][1] - 1)
        deltas_f.append(deltas_f[-1] + gap * strength)

    if deltas_f[-1] < 0.5:
        return None, {"reason": "no_effective_shift",
                      "n_lines": len(lines),
                      "raw_shift_f": round(deltas_f[-1], 3),
                      "region_source": source}

    # Round once, then force a >=1 px total shift if we were just under.
    deltas = [int(round(d)) for d in deltas_f]
    if deltas[-1] == 0:
        deltas = [int(np.ceil(d)) for d in deltas_f]

    # Erase all bands, then paste them shifted up with a per-band clamp.
    out = bgr.copy()
    for (a, b, xa, xb) in bands:
        out[a:b + 1, xa:xb + 1] = bg

    for (a, _b, xa, _xb), crop, d in zip(bands, crops, deltas):
        na = max(0, a - d)
        _paste_clipped(out, crop, na, xa)

    if np.array_equal(out, bgr):
        return None, {"reason": "pixel_identical",
                      "total_shift_px": int(deltas[-1]),
                      "region_source": source}

    return out, {
        "n_lines": len(lines),
        "total_shift_px": int(deltas[-1]),
        "strength": float(strength),
        "region_source": source,
    }


# ---------------------------------------------------------------- driver
def find_images(root: Path, name: str):
    for pattern in DATASET_GLOBS.get(name, []):
        hits = sorted(root.glob(pattern))
        if hits:
            return hits
    return sorted(p for p in root.rglob("*")
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg",
                                          ".tif", ".tiff"})


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)


def compute_fsim_gpu(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """FSIM using piq, with automatic GPU acceleration if available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # piq expects shape (B, C, H, W) and data_range=1.0
    a = torch.from_numpy(gray_a).float().unsqueeze(0).unsqueeze(0) / 255.0
    b = torch.from_numpy(gray_b).float().unsqueeze(0).unsqueeze(0) / 255.0
    
    a = a.to(device)
    b = b.to(device)
    
    with torch.no_grad():
        score = fsim(a, b, data_range=1.0, chromatic=False)
    
    return score.item()


def _parse_strengths(s: str) -> set[str]:
    """Parse '0.02,0.05' -> {'0.02','0.05'}. Empty string -> empty set."""
    if not s or not s.strip():
        return set()
    return {f"{float(x.strip()):.2f}" for x in s.split(",") if x.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="FUNSD | CORD | <name>")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--regions", choices=["auto", "annotations", "heuristics"],
                    default="auto")
    ap.add_argument("--bg-strategy", choices=["border", "nonink"],
                    default="border")
    ap.add_argument("--fsim-strengths", default="0.02,0.05,0.10",
                    help="Comma-separated strengths at which to compute FSIM. "
                         'Empty string skips FSIM entirely (fastest pilot). '
                         'Example: --fsim-strengths "0.05" samples middle only.')
    ap.add_argument("--flush-every", type=int, default=20,
                    help="Flush manifest/skipped CSV every N rows "
                         "(0 = end only).")
    args = ap.parse_args()

    fsim_set = _parse_strengths(args.fsim_strengths)
    if not fsim_set:
        print("[info] --fsim-strengths is empty: FSIM will be skipped. "
              "PSNR/SSIM are still computed for every row.")

    args.out.mkdir(parents=True, exist_ok=True)
    images = find_images(args.root, args.dataset)
    if args.limit:
        images = images[:args.limit]
    if not images:
        sys.exit(f"no images found under {args.root} for {args.dataset}")
    print(f"[{args.dataset}] {len(images)} images")

    n_variants = sum(len(v) for v in PERTURBATIONS.values())
    print(f"  variants per image: {n_variants}")
    print(f"  region source: {args.regions}   bg: {args.bg_strategy}")
    print(f"  fsim strengths: {sorted(fsim_set) if fsim_set else '(skipped)'}")

    manifest_rows, skipped = [], []
    total_attempts = 0
    counts = Counter()
    t_fsim_total = 0.0
    t_script_start = time.perf_counter()

    # Region freeze state for block_displacement (one region per image,
    # reused across strengths so PSNR/SSIM/FSIM are monotone in strength).
    chosen_region = None
    chosen_region_key = None

    def flush_partial():
        write_csv(args.out / "manifest.csv", manifest_rows, MANIFEST_FIELDS)
        if skipped:
            write_csv(args.out / "skipped.csv", skipped, SKIPPED_FIELDS)

    for img_idx, img_path in enumerate(images, 1):
        image_id = img_path.stem
        try:
            clean = imread(img_path)
        except IOError as e:
            print(f"\n[skip] cannot read {img_path}: {e}")
            continue
        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        bg = estimate_bg(clean, args.bg_strategy)

        ann = None
        if args.regions != "heuristics":
            ann = load_annotations(img_path, args.dataset)
            if ann is None and args.regions == "annotations":
                skipped.append({
                    "dataset": args.dataset, "image_id": image_id,
                    "perturbation": "*", "strength": "*",
                    "reason": "no_annotations",
                    "detail": str(img_path),
                })
                counts[("*", "no_annotations")] += 1
                continue

        for pert, strengths in PERTURBATIONS.items():
            for strength in strengths:
                total_attempts += 1
                tag = f"{strength:.2f}"
                seed = int(hashlib.md5(
                    f"{args.dataset}|{image_id}|{pert}|{tag}".encode()
                ).hexdigest()[:8], 16)
                rng = random.Random(seed)

                if pert == "block_displacement":
                    if ann is not None:
                        regions = list(ann["blocks"]) + list(ann["lines"])
                        source = "annotations"
                    else:
                        regions, source = None, "heuristics"
                    # Freeze region across strengths: pick once per image at
                    # the max strength, reuse for every strength.
                    if chosen_region_key != image_id:
                        _rng = random.Random(int(hashlib.md5(
                            f"{args.dataset}|{image_id}|region".encode()
                        ).hexdigest()[:8], 16))
                        _, _pre = perturb_block_displacement(
                            clean, max(strengths), _rng, bg,
                            regions=regions, source=source)
                        chosen_region = (tuple(_pre["bbox"])
                                         if _pre.get("reason") is None else None)
                        chosen_region_key = image_id
                    if chosen_region is None:
                        out, meta = None, {"reason": "no_candidate_region",
                                           "region_source": source}
                    else:
                        out, meta = perturb_block_displacement(
                            clean, strength, rng, bg,
                            regions=[chosen_region], source=source)
                else:  # line_compression
                    if ann is not None:
                        lines, source = ann["lines"], "annotations"
                    else:
                        lines, source = None, "heuristics"
                    out, meta = perturb_line_compression(
                        clean, strength, bg, lines=lines, source=source)

                if out is None:
                    skipped.append({
                        "dataset": args.dataset, "image_id": image_id,
                        "perturbation": pert, "strength": tag,
                        "reason": meta.get("reason", "unknown"),
                        "detail": str(meta),
                    })
                    counts[(pert, meta.get("reason", "unknown"))] += 1
                    continue

                out_dir = args.out / args.dataset / pert / tag
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{image_id}.png"
                cv2.imwrite(str(out_path), out)

                out_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                if np.array_equal(gray, out_gray):
                    skipped.append({
                        "dataset": args.dataset, "image_id": image_id,
                        "perturbation": pert, "strength": tag,
                        "reason": "identical_post_write",
                        "detail": str(meta),
                    })
                    counts[(pert, "identical_post_write")] += 1
                    continue

                p = psnr_fn(gray, out_gray, data_range=255)
                s = ssim_fn(gray, out_gray, data_range=255)

                f_val = ""
                if tag in fsim_set:
                    t0 = time.perf_counter()
                    try:
                        f_val = f"{compute_fsim_gpu(gray, out_gray):.6f}"
                    except Exception as e:
                        print(f"\n[warn] FSIM failed for "
                              f"{image_id} {pert} {tag}: {e}")
                        f_val = ""
                    t_fsim_total += time.perf_counter() - t0

                manifest_rows.append({
                    "dataset": args.dataset, "split": args.split,
                    "image_id": image_id, "perturbation": pert,
                    "strength": tag,
                    "clean_path": str(img_path.resolve()),
                    "perturbed_path": str(out_path.resolve()),
                    "psnr": f"{p:.4f}" if np.isfinite(p) else "inf",
                    "ssim": f"{s:.6f}",
                    "fsim": f_val,
                    "region_source": meta.get("region_source", ""),
                    "detail": str(meta),
                })
                counts[(pert, "ok")] += 1

                if args.flush_every and len(manifest_rows) % args.flush_every == 0:
                    flush_partial()

        print(f"  [{img_idx}/{len(images)}] {image_id}: "
              f"written={len(manifest_rows)} skipped={len(skipped)}",
              end="\r", flush=True)

    print()
    flush_partial()  # final flush (covers the tail when not a multiple of N)

    skip_rate = len(skipped) / max(1, total_attempts)
    print(f"\nwrote {len(manifest_rows)} variants -> "
          f"{args.out / 'manifest.csv'}")
    print(f"skipped {len(skipped)}/{total_attempts} attempts "
          f"({skip_rate * 100:.1f}%)")

    if counts:
        print("per (perturbation, outcome):")
        for (pert, outcome), n in sorted(counts.items()):
            print(f"  {pert:22s} {outcome:24s} {n}")

    # Per (perturbation, strength) summary for the report.
    if manifest_rows:
        summary = defaultdict(lambda: {"n": 0, "ssim": [], "psnr": [], "fsim": []})
        for r in manifest_rows:
            k = (r["perturbation"], r["strength"])
            summary[k]["n"] += 1
            if r["psnr"] != "inf":
                summary[k]["psnr"].append(float(r["psnr"]))
            summary[k]["ssim"].append(float(r["ssim"]))
            if r.get("fsim"):
                summary[k]["fsim"].append(float(r["fsim"]))

        with open(args.out / "summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["perturbation", "strength", "n",
                        "mean_psnr", "mean_ssim", "min_ssim",
                        "n_fsim", "mean_fsim", "min_fsim"])
            for (pert, tag), s in sorted(summary.items()):
                w.writerow([
                    pert, tag, s["n"],
                    f"{np.mean(s['psnr']):.4f}" if s["psnr"] else "",
                    f"{np.mean(s['ssim']):.6f}",
                    f"{np.min(s['ssim']):.6f}",
                    len(s["fsim"]),
                    f"{np.mean(s['fsim']):.6f}" if s["fsim"] else "",
                    f"{np.min(s['fsim']):.6f}" if s["fsim"] else "",
                ])
        print(f"summary -> {args.out / 'summary.csv'}")

    # Timing report so you can confirm FSIM is the bottleneck.
    t_total = time.perf_counter() - t_script_start
    if t_total > 0:
        pct = 100.0 * t_fsim_total / t_total
        print(f"\ntime: total={t_total:.1f}s "
              f"fsim={t_fsim_total:.1f}s ({pct:.1f}%)")
        if pct > 50:
            print("      FSIM dominates; install pyfftw "
                  "(`pip install pyfftw`) or use --fsim-strengths "
                  "to sample.")


if __name__ == "__main__":
    main()