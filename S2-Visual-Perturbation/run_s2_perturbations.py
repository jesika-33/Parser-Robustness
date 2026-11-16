"""Generate S2 visual perturbations, metrics.csv, previews; run sanity checks."""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import sys
import traceback
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visual_perturbations import (
    LEVELS,
    PERTURBATION_GRID,
    PERTURBATION_TYPES,
    apply_visual_perturbation,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "perturbed"
PREVIEW = ROOT / "preview"
METRICS_CSV = ROOT / "metrics.csv"
DATASETS = ("cord", "sroie", "funsd")

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def list_images(ds: str) -> list[Path]:
    d = DATA / ds
    if not d.is_dir():
        raise FileNotFoundError(d)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)


def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array, preserving grayscale (2D) vs color (HxWx3)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to decode {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    buf.tofile(str(path))


def to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """HxW or HxWx3/4 uint8 -> HxWx3 uint8 RGB for metric libraries."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported shape {img.shape}")
    return np.ascontiguousarray(img)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_ISM_OK = True
_ISM_ERR = ""
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from image_similarity_measures.quality_metrics import (
            psnr as _ism_psnr,
            ssim as _ism_ssim,
        )
except Exception as e:  # pragma: no cover
    _ISM_OK = False
    _ISM_ERR = str(e)

from fsim_utils import (
    compute_fsim,
    compute_psnr,
    compute_ssim,
    ism_available,
    ism_error,
    make_fsim_scorer,
)


def compute_metrics(clean: np.ndarray, pert: np.ndarray) -> tuple[float, float, float]:
    """Return (psnr, ssim, fsim). See fsim_utils for resolution caps."""
    return compute_psnr(clean, pert), compute_ssim(clean, pert), compute_fsim(clean, pert)


# ---------------------------------------------------------------------------
# Worker: one image -> all 12 perturbations + metrics rows
# ---------------------------------------------------------------------------


def process_image(args: tuple) -> list[dict]:
    ds, clean_path_s, out_root_s = args
    clean_path = Path(clean_path_s)
    out_root = Path(out_root_s)
    fname = clean_path.name
    image_id = clean_path.stem

    try:
        clean = load_image(clean_path)
        fsim_scorer = make_fsim_scorer(clean)
        rows: list[dict] = []
        for ptype in PERTURBATION_TYPES:
            for level in LEVELS:
                pert = apply_visual_perturbation(clean, ptype, level)
                if pert.shape != clean.shape:
                    raise RuntimeError(
                        f"shape change {clean.shape}->{pert.shape} for {ptype} L{level}"
                    )
                dest = out_root / ds / f"s2_{ptype}_l{level}" / fname
                save_image(dest, pert)
                psnr_v = compute_psnr(clean, pert)
                ssim_v = compute_ssim(clean, pert)
                fsim_v = fsim_scorer(pert)
                rows.append(
                    {
                        "dataset": ds,
                        "image_id": image_id,
                        "perturbation": ptype,
                        "level": level,
                        "psnr": psnr_v,
                        "ssim": ssim_v,
                        "fsim": fsim_v,
                    }
                )
        return rows
    except Exception:
        return [
            {
                "dataset": ds,
                "image_id": image_id,
                "perturbation": "ERROR",
                "level": -1,
                "psnr": float("nan"),
                "ssim": float("nan"),
                "fsim": float("nan"),
                "_error": traceback.format_exc(),
            }
        ]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def prepare_dirs(clean_only: bool = False, wipe: bool = True) -> None:
    if not wipe:
        for ds in DATASETS:
            for ptype in PERTURBATION_TYPES:
                for level in LEVELS:
                    (OUT / ds / f"s2_{ptype}_l{level}").mkdir(parents=True, exist_ok=True)
            (OUT / ds / "clean").mkdir(parents=True, exist_ok=True)
        return
    for ds in DATASETS:
        for ptype in PERTURBATION_TYPES:
            for level in LEVELS:
                d = OUT / ds / f"s2_{ptype}_l{level}"
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
        c = OUT / ds / "clean"
        if c.exists():
            shutil.rmtree(c)
        c.mkdir(parents=True, exist_ok=True)


def copy_clean() -> None:
    for ds in DATASETS:
        for src in list_images(ds):
            shutil.copy2(src, OUT / ds / "clean" / src.name)


PARTIAL_METRICS = ROOT / "metrics_partial.jsonl"


def _image_outputs_complete(ds: str, src: Path) -> bool:
    for ptype in PERTURBATION_TYPES:
        for level in LEVELS:
            if not (OUT / ds / f"s2_{ptype}_l{level}" / src.name).is_file():
                return False
    return True


def load_partial_metrics() -> dict[tuple[str, str], list[dict]]:
    """image_id key -> rows. Resume support."""
    done: dict[tuple[str, str], list[dict]] = {}
    if not PARTIAL_METRICS.is_file():
        return done
    with open(PARTIAL_METRICS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                import json

                r = json.loads(line)
            except Exception:
                continue
            key = (r.get("dataset", ""), r.get("image_id", ""))
            # incomplete/error rows should be redone
            if r.get("perturbation") == "ERROR":
                done.pop(key, None)
                continue
            done.setdefault(key, []).append(r)
    # keep only complete 12-row sets
    complete = {}
    for k, rows in done.items():
        if len({(r["perturbation"], r["level"]) for r in rows}) >= 12:
            complete[k] = rows
    return complete


def append_partial_metrics(rows: list[dict]) -> None:
    import json

    with open(PARTIAL_METRICS, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_metrics_and_perturb(workers: int, resume: bool = True) -> list[dict]:
    partial = load_partial_metrics() if resume else {}
    if resume and partial:
        print(f"resume: {len(partial)} images already have full metrics", flush=True)

    tasks = []
    cached_rows: list[dict] = []
    for ds in DATASETS:
        for src in list_images(ds):
            key = (ds, src.stem)
            if resume and key in partial and _image_outputs_complete(ds, src):
                cached_rows.extend(partial[key])
                continue
            tasks.append((ds, str(src), str(OUT)))

    print(f"to process: {len(tasks)} images (cached: {len(partial) if resume else 0})", flush=True)

    all_rows: list[dict] = list(cached_rows)
    if workers <= 1:
        for i, t in enumerate(tasks, 1):
            rows = process_image(t)
            all_rows.extend(rows)
            append_partial_metrics(rows)
            if i % 10 == 0:
                print(f"  processed {i}/{len(tasks)} images...", flush=True)
    else:
        with mp.Pool(workers) as pool:
            for i, rows in enumerate(pool.imap_unordered(process_image, tasks, chunksize=2), 1):
                all_rows.extend(rows)
                append_partial_metrics(rows)
                if i % 10 == 0:
                    print(f"  processed {i}/{len(tasks)} images...", flush=True)
    return all_rows


def write_metrics_csv(rows: list[dict]) -> None:
    fieldnames = ["dataset", "image_id", "perturbation", "level", "psnr", "ssim", "fsim"]
    # dedupe: keep last row per (dataset, image_id, perturbation, level)
    deduped: dict[tuple, dict] = {}
    for r in rows:
        key = (r["dataset"], r["image_id"], r["perturbation"], r["level"])
        deduped[key] = r
    rows = list(deduped.values())
    # stable order: dataset, image_id, perturbation, level
    p_order = {p: i for i, p in enumerate(PERTURBATION_TYPES)}
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            DATASETS.index(r["dataset"]) if r["dataset"] in DATASETS else 99,
            r["image_id"],
            p_order.get(r["perturbation"], 99),
            r["level"],
        ),
    )
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows_sorted:
            w.writerow(
                {
                    "dataset": r["dataset"],
                    "image_id": r["image_id"],
                    "perturbation": r["perturbation"],
                    "level": r["level"],
                    "psnr": "" if r["psnr"] != r["psnr"] else r["psnr"],
                    "ssim": "" if r["ssim"] != r["ssim"] else r["ssim"],
                    "fsim": "" if r["fsim"] != r["fsim"] else r["fsim"],
                }
            )


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def check_counts() -> list[str]:
    problems = []
    for ds in DATASETS:
        clean_n = len(list_images(ds))
        clean_out = list((OUT / ds / "clean").iterdir())
        clean_out_n = sum(1 for p in clean_out if p.is_file())
        if clean_out_n != clean_n:
            problems.append(f"{ds}/clean has {clean_out_n}, expected {clean_n}")
        for ptype in PERTURBATION_TYPES:
            for level in LEVELS:
                d = OUT / ds / f"s2_{ptype}_l{level}"
                n = sum(1 for p in d.iterdir() if p.is_file()) if d.exists() else 0
                if n != clean_n:
                    problems.append(f"{ds}/s2_{ptype}_l{level} has {n}, expected {clean_n}")
    return problems


def load_metrics_rows() -> list[dict]:
    rows = []
    with open(METRICS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("psnr", "ssim", "fsim"):
                if r[k] == "" or r[k] is None:
                    r[k] = float("nan")
                else:
                    r[k] = float(r[k])
            r["level"] = int(r["level"])
            rows.append(r)
    return rows


def summary_table(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Mean metrics per perturbation x level; check SSIM monotonicity L1->L3."""
    from collections import defaultdict

    acc: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
    for r in rows:
        acc[(r["perturbation"], r["level"])].append((r["psnr"], r["ssim"], r["fsim"]))

    table = []
    violations = []
    for ptype in PERTURBATION_TYPES:
        means = {}
        for level in LEVELS:
            vals = acc.get((ptype, level), [])
            if not vals:
                means[level] = (float("nan"),) * 3
                continue
            arr = np.array(vals, dtype=float)
            # PSNR may contain inf (identical); mean: any inf among finite+inf -> keep as mean with inf
            psnr_col = arr[:, 0]
            if psnr_col.size and np.all(np.isinf(psnr_col)):
                m_psnr = float("inf")
            elif psnr_col.size and np.any(np.isinf(psnr_col)):
                # mixed inf/finite: mean of finite only (inf dominates otherwise)
                finite = psnr_col[np.isfinite(psnr_col)]
                m_psnr = float(np.mean(finite)) if finite.size else float("inf")
            else:
                m_psnr = float(np.nanmean(psnr_col)) if psnr_col.size else float("nan")
            m_ssim = float(np.nanmean(arr[:, 1])) if np.any(~np.isnan(arr[:, 1])) else float("nan")
            m_fsim = float(np.nanmean(arr[:, 2])) if np.any(~np.isnan(arr[:, 2])) else float("nan")
            means[level] = (m_psnr, m_ssim, m_fsim)
            table.append(
                {
                    "perturbation": ptype,
                    "level": level,
                    "psnr": m_psnr,
                    "ssim": m_ssim,
                    "fsim": m_fsim,
                    "n": len(vals),
                }
            )
        # monotonicity: mean SSIM must decrease L1 -> L2 -> L3
        s1, s2, s3 = means[1][1], means[2][1], means[3][1]
        if not (s1 > s2 > s3):
            violations.append(
                f"{ptype}: mean SSIM not strictly decreasing L1->L2->L3 "
                f"(got {s1:.6f}, {s2:.6f}, {s3:.6f})"
            )
    return table, violations


def format_summary(table: list[dict]) -> str:
    lines = []
    lines.append(f"{'perturbation':<12} {'level':>5} {'mean_PSNR':>12} {'mean_SSIM':>12} {'mean_FSIM':>12} {'n':>6}")
    lines.append("-" * 62)
    for r in table:
        psnr_s = "inf" if np.isinf(r["psnr"]) else f"{r['psnr']:.4f}"
        lines.append(
            f"{r['perturbation']:<12} {r['level']:>5} {psnr_s:>12} {r['ssim']:>12.6f} {r['fsim']:>12.6f} {r['n']:>6}"
        )
    return "\n".join(lines)


def flag_strong_l1(table: list[dict]) -> list[str]:
    flags = []
    for r in table:
        if r["level"] != 1:
            continue
        reasons = []
        ssim_v = r["ssim"]
        psnr_v = r["psnr"]
        if not (ssim_v > 0.97) and not np.isnan(ssim_v):
            reasons.append(f"L1 mean SSIM {ssim_v:.6f} < 0.97")
        if not np.isnan(psnr_v) and psnr_v < 35:
            reasons.append(f"L1 mean PSNR {psnr_v:.4f} dB < 35")
        if reasons:
            flags.append(f"{r['perturbation']}: " + "; ".join(reasons))
    return flags


# ---------------------------------------------------------------------------
# Preview grids
# ---------------------------------------------------------------------------


def make_previews(n_samples: int = 3) -> list[Path]:
    PREVIEW.mkdir(parents=True, exist_ok=True)
    # clean old previews
    for p in PREVIEW.glob("*.png"):
        p.unlink()

    # layout: for each sample, a row: clean + 12 perturbed (4 types x 3 levels)
    # grid PNG per dataset: 3 sample rows x 13 columns
    paths_out = []
    for ds in DATASETS:
        imgs = list_images(ds)
        if not imgs:
            continue
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(imgs), size=min(n_samples, len(imgs)), replace=False)
        samples = [imgs[i] for i in sorted(idxs)]

        # compute cell size
        cell_w, cell_h = 220, 300
        label_h = 36
        header_h = 40
        cols = 1 + len(PERTURBATION_TYPES) * len(LEVELS)
        rows_n = len(samples)
        canvas_w = cols * cell_w
        canvas_h = header_h + rows_n * cell_h
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            font_sm = ImageFont.truetype("arial.ttf", 11)
        except Exception:
            font = ImageFont.load_default()
            font_sm = font

        # column headers
        headers = ["clean"] + [f"{p}_L{l}" for p in PERTURBATION_TYPES for l in LEVELS]
        for c, h in enumerate(headers):
            x = c * cell_w + cell_w // 2
            draw.text((x, 8), h, fill=(0, 0, 0), font=font, anchor="mt")

        for r_i, spath in enumerate(samples):
            clean = load_image(spath)
            y0 = header_h + r_i * cell_h
            draw.text((4, y0 + 2), f"{r_i}:{spath.name}", fill=(0, 0, 0), font=font_sm)

            cells = [clean]
            for ptype in PERTURBATION_TYPES:
                for level in LEVELS:
                    cells.append(apply_visual_perturbation(clean, ptype, level))

            for c, cell_img in enumerate(cells):
                # to RGB for PIL
                if cell_img.ndim == 2:
                    rgb = cv2.cvtColor(cell_img, cv2.COLOR_GRAY2RGB)
                else:
                    rgb = cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                pil.thumbnail((cell_w - 8, cell_h - 40), Image.Resampling.LANCZOS)
                x = c * cell_w + (cell_w - pil.width) // 2
                y = y0 + label_h + (cell_h - label_h - pil.height) // 2
                canvas.paste(pil, (x, y))
                draw.rectangle(
                    [c * cell_w, y0 + label_h - 2, (c + 1) * cell_w - 1, y0 + cell_h - 1],
                    outline=(200, 200, 200),
                )

        out = PREVIEW / f"{ds}_preview.png"
        canvas.save(out)
        paths_out.append(out)
        print(f"  wrote {out}", flush=True)
    return paths_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    global _ISM_OK
    ap = argparse.ArgumentParser(description="S2 visual perturbations benchmark")
    ap.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 2)))
    ap.add_argument("--skip-preview", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="wipe outputs and start over")
    ap.add_argument("--no-resume", action="store_true", help="ignore partial metrics cache")
    args = ap.parse_args()

    print("=== S2 visual perturbation run ===", flush=True)
    if not ism_available():
        print(f"WARNING: image-similarity-measures unavailable ({ism_error()}); PSNR/SSIM fallback to skimage", flush=True)
    else:
        print("image-similarity-measures: available (PSNR/SSIM)", flush=True)
    print("FSIM: fast_fsim (port of ISM FSIM; resolution-capped — see README)", flush=True)

    # count inputs
    counts = {ds: len(list_images(ds)) for ds in DATASETS}
    total = sum(counts.values())
    print(f"input counts: {counts} total={total}", flush=True)
    if total != 497:
        print(f"WARNING: expected 497 images, found {total}", flush=True)

    wipe = bool(args.fresh)
    if wipe and PARTIAL_METRICS.exists():
        PARTIAL_METRICS.unlink()
    resume = not args.no_resume and not wipe

    print("preparing directories + copying clean originals...", flush=True)
    prepare_dirs(wipe=wipe)
    copy_clean()

    print(f"generating perturbations + metrics (workers={args.workers}, resume={resume})...", flush=True)
    rows = run_metrics_and_perturb(args.workers, resume=resume)
    errors = [r for r in rows if r.get("perturbation") == "ERROR"]
    if errors:
        print(f"ERRORS: {len(errors)}")
        for e in errors[:5]:
            print(e.get("_error", e))
    rows = [r for r in rows if r.get("perturbation") != "ERROR"]

    print(f"writing {METRICS_CSV} ({len(rows)} rows)...")
    write_metrics_csv(rows)

    print("sanity check 1: counts...")
    count_problems = check_counts()
    if count_problems:
        print("COUNT CHECK FAILED:")
        for p in count_problems:
            print("  -", p)
    else:
        print("COUNT CHECK PASSED: all folders match clean counts")

    print("sanity check 2: monotonicity...")
    table, violations = summary_table(rows)
    print(format_summary(table))
    if violations:
        print("MONOTONICITY VIOLATIONS:")
        for v in violations:
            print("  -", v)
    else:
        print("MONOTONICITY PASSED: mean SSIM decreases L1->L2->L3 for all perturbations")

    strong_flags = flag_strong_l1(table)
    if strong_flags:
        print("L1 TOO STRONG FLAGS (not auto-fixed):")
        for s in strong_flags:
            print("  -", s)
    else:
        print("L1 strength OK (SSIM>=0.97 and PSNR>=35 dB for all perturbations)")

    if not args.skip_preview:
        print("sanity check 3: previews...")
        make_previews(3)

    if count_problems or violations:
        return 1
    return 0


if __name__ == "__main__":
    # Windows multiprocessing spawn safety
    if sys.platform.startswith("win"):
        mp.freeze_support()
    sys.exit(main())
