#!/usr/bin/env python3
"""
Paired Tesseract evaluation: clean vs. perturbed, ∆CER / ∆WER.

Ground truth:
    FUNSD    -> JSON["form"][*]["text"]
    CORD v1  -> recursive flatten of JSON["gt_parse"]
    CORD v2  -> JSON["valid_line"][*]["words"][*]["text"]
    fallback -> recursive collection of any "text" field

Usage:
    python eval_tesseract.py \\
        --manifest out/manifest.csv \\
        --tessdata /usr/share/tesseract-ocr/5/tessdata_best \\
        --psm 3 --oem 1 --out out/tesseract_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------- edit distance
try:
    from rapidfuzz.distance import Levenshtein as _Lev

    def edit_distance(a, b):
        return _Lev.distance(a, b)
except ImportError:  # pure-python fallback (works for str and list[Hashable])

    def edit_distance(a, b):
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(
                    min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
                )
            prev = cur
        return prev[-1]


FIELDNAMES = [
    "dataset", "split", "image_id", "perturbation", "strength",
    "clean_cer", "pert_cer", "d_cer",
    "clean_wer", "pert_wer", "d_wer",
    "psnr", "ssim",
]


# ---------------------------------------------------------------- text / metrics
def normalize_text(s: str) -> str:
    """Collapse whitespace so newline placement differences don't inflate CER."""
    return re.sub(r"\s+", " ", s).strip()


def cer(ref: str, hyp: str) -> float:
    return edit_distance(ref, hyp) / max(1, len(ref))


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    return edit_distance(r, h) / max(1, len(r))


# ---------------------------------------------------------------- OCR
def tesseract_ocr(
    img_path: Path,
    tessdata_dir: str,
    psm: str = "3",
    oem: str = "1",
    lang: str = "eng",
) -> str:
    cmd = [
        "tesseract", str(img_path), "stdout",
        "--tessdata-dir", tessdata_dir,
        "-l", lang, "--oem", oem, "--psm", psm,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or "tesseract failed")
    return res.stdout


# ---------------------------------------------------------------- ground truth
def load_gt(json_path: Path) -> str:
    #SROIE 
    if json_path.suffix.lower() == ".txt":
        lines = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 9:
                    # The transcript is everything after the 8th comma
                    text = ",".join(parts[8:]).strip()
                    if text:
                        lines.append(text)
        return "\n".join(lines)
    
    with open(json_path) as f:
        data = json.load(f)

    # FUNSD
    if isinstance(data, dict) and "form" in data:
        return "\n".join(item.get("text", "") for item in data["form"])

    # CORD v1
    if isinstance(data, dict) and "gt_parse" in data:
        def flat(o):
            if isinstance(o, dict):
                return [x for v in o.values() for x in flat(v)]
            if isinstance(o, list):
                return [x for v in o for x in flat(v)]
            return [o] if isinstance(o, str) else []
        return "\n".join(flat(data["gt_parse"]))

    # CORD v2
    if isinstance(data, dict) and "valid_line" in data:
        lines = []
        for line in data["valid_line"]:
            words = [w.get("text", "") for w in line.get("words", [])]
            line_text = " ".join(w for w in words if w)
            if line_text:
                lines.append(line_text)
        return "\n".join(lines)

    # Fallback: collect every "text" field recursively
    def collect_text(o):
        if isinstance(o, dict):
            if "text" in o and isinstance(o["text"], str):
                yield o["text"]
            for v in o.values():
                yield from collect_text(v)
        elif isinstance(o, list):
            for v in o:
                yield from collect_text(v)

    joined = "\n".join(collect_text(data))
    if joined.strip():
        print(f"[warn] {json_path}: unknown schema, used text-field fallback")
        return joined

    raise ValueError(f"unknown GT format: {json_path}")


def find_gt(img_path: Path, dataset: str) -> Path | None:
    stem, parent = img_path.stem, img_path.parent
    root = parent.parent
    
    # SROIE specific search
    if dataset.upper() == "SROIE":
        for sub in ["box", "boxes"]:
            cand = root / sub / f"{stem}.txt"
            if cand.exists():
                return cand

    for sub in ["annotations", "anotations", "json", "jsons", "ground_truth"]:
        cand = root / sub / f"{stem}.json"
        if cand.exists():
            return cand
            
    return None

# ---------------------------------------------------------------- IO helpers
def write_csv(path: Path, rows: list[dict], fieldnames=FIELDNAMES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--tessdata", required=True,
                    help="path to tessdata_best directory")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--psm", default="3")
    ap.add_argument("--oem", default="1")
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--normalize", action="store_true",
                    help="Collapse whitespace in ref/hyp before CER/WER.")
    ap.add_argument("--flush-every", type=int, default=20,
                    help="Write intermediate results every N rows (0 = disable).")
    ap.add_argument("--progress-every", type=int, default=50,
                    help="Print progress every N manifest rows (0 = disable).")
    args = ap.parse_args()

    # Provenance
    try:
        ver_lines = subprocess.run(
            ["tesseract", "--version"], capture_output=True, text=True
        ).stdout.splitlines()
        print(ver_lines[0] if ver_lines else "tesseract (unknown version)")
    except Exception as e:
        print(f"[warn] cannot query tesseract version: {e}")
    print(
        f"oem={args.oem} psm={args.psm} lang={args.lang} "
        f"tessdata={args.tessdata} normalize={args.normalize}"
    )

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("[warn] manifest is empty; writing header-only output")
        write_csv(args.out, [])
        return
    print(f"manifest: {len(rows)} rows from {args.manifest}")

    clean_cache: dict[str, tuple[str, float, float] | None] = {}
    out_rows: list[dict] = []
    skipped = defaultdict(int)   # (dataset, pert, strength) -> count
    total_skipped = 0

    def flush():
        write_csv(args.out, out_rows)

    for idx, row in enumerate(rows, 1):
        ds = row["dataset"]
        iid = row["image_id"]
        pert = row["perturbation"]
        st = row["strength"]
        clean_path = Path(row["clean_path"])
        pert_path = Path(row["perturbed_path"])
        key = str(clean_path)

        # ---- clean pass (cached, one OCR per unique clean image) ----
        if key not in clean_cache:
            try:
                gt_file = find_gt(clean_path, ds)
            except Exception as e:
                print(f"[warn] find_gt failed for {clean_path}: {e}")
                gt_file = None

            if gt_file is None:
                print(f"[warn] no GT for {clean_path}")
                clean_cache[key] = None
            else:
                try:
                    gt = load_gt(gt_file)
                    if args.normalize:
                        gt = normalize_text(gt)
                    hyp = tesseract_ocr(
                        clean_path, args.tessdata, args.psm, args.oem, args.lang
                    )
                    if args.normalize:
                        hyp = normalize_text(hyp)
                    clean_cache[key] = (gt, cer(gt, hyp), wer(gt, hyp))
                except Exception as e:
                    print(f"[warn] clean failed for {clean_path}: {e}")
                    clean_cache[key] = None

        if clean_cache.get(key) is None:
            skipped[(ds, pert, st)] += 1
            total_skipped += 1
            continue

        gt, clean_cer, clean_wer = clean_cache[key]

        # ---- perturbed pass ----
        try:
            hyp = tesseract_ocr(
                pert_path, args.tessdata, args.psm, args.oem, args.lang
            )
            if args.normalize:
                hyp = normalize_text(hyp)
            pc = cer(gt, hyp)
            pw = wer(gt, hyp)
        except Exception as e:
            print(f"[warn] pert failed for {pert_path}: {e}")
            skipped[(ds, pert, st)] += 1
            total_skipped += 1
            continue

        out_rows.append({
            "dataset": ds,
            "split": row["split"],
            "image_id": iid,
            "perturbation": pert,
            "strength": st,
            "clean_cer": f"{clean_cer:.4f}",
            "pert_cer": f"{pc:.4f}",
            "d_cer": f"{pc - clean_cer:+.4f}",
            "clean_wer": f"{clean_wer:.4f}",
            "pert_wer": f"{pw:.4f}",
            "d_wer": f"{pw - clean_wer:+.4f}",
            "psnr": row["psnr"],
            "ssim": row["ssim"],
        })

        if args.flush_every and len(out_rows) % args.flush_every == 0:
            flush()

        if args.progress_every and idx % args.progress_every == 0:
            print(f"  [{idx}/{len(rows)}] kept={len(out_rows)} "
                  f"skipped={total_skipped}")

    flush()
    print(f"\nwrote {len(out_rows)} rows -> {args.out} "
          f"(skipped {total_skipped})")

    # ---- aggregate ----
    if not out_rows:
        print("[warn] no rows produced; nothing to aggregate")
        return

    agg = defaultdict(lambda: {"clean_cer": [], "clean_wer": [], "d_cer": [], "d_wer": [], "psnr": [], "ssim": []})
    for r in out_rows:
        k = (r["dataset"], r["perturbation"], r["strength"])
        agg[k]["clean_cer"].append(float(r["clean_cer"]))
        agg[k]["clean_wer"].append(float(r["clean_wer"]))
        agg[k]["d_cer"].append(float(r["d_cer"]))
        agg[k]["d_wer"].append(float(r["d_wer"]))
        agg[k]["psnr"].append(float(r["psnr"]))
        agg[k]["ssim"].append(float(r["ssim"]))

    # print(
    #     f"\n{'dataset':8} {'perturbation':20} {'str':>5} "
    #     f"{'n':>4} {'skip':>5} {'ΔCER':>8} {'ΔWER':>8} "
    #     f"{'PSNR':>7} {'SSIM':>7}"
    # )
    # for (ds, pert, st), v in sorted(agg.items()):
    #     n = len(v["d_cer"])
    #     ns = skipped.get((ds, pert, st), 0)
    #     print(
    #         f"{ds:8} {pert:20} {st:>5} "
    #         f"{n:>4} {ns:>5} "
    #         f"{sum(v['d_cer']) / n:+8.4f} {sum(v['d_wer']) / n:+8.4f} "
    #         f"{sum(v['psnr']) / n:7.2f} {sum(v['ssim']) / n:7.4f}"
    #     )


    print(
        f"\n{'dataset':8} {'perturbation':20} {'str':>5} "
        f"{'n':>4} {'BaseCER':>8} {'BaseWER':>8} {'ΔCER':>8} {'ΔWER':>8} "
    )
    for (ds, pert, st), v in sorted(agg.items()):
        n = len(v["d_cer"])
        # Calculate means and multiply by 100 for percentage
        mean_bc = (sum(v['clean_cer']) / n) * 100
        mean_bw = (sum(v['clean_wer']) / n) * 100
        mean_dc = (sum(v['d_cer']) / n) * 100
        mean_dw = (sum(v['d_wer']) / n) * 100
        
        print(
            f"{ds:8} {pert:20} {st:>5} {n:>4} "
            f"{mean_bc:7.2f}% {mean_bw:7.2f}% "
            f"{mean_dc:+7.2f}% {mean_dw:+7.2f}%"
        )


if __name__ == "__main__":
    main()
