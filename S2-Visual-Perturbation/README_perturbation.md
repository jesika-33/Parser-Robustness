# S2 Visual Perturbations — Run Instructions

Document-parser robustness benchmark: 4 perturbations × 3 levels applied to 497 document images.

## Inputs

| Dataset | Path | Count |
|---------|------|-------|
| CORD | `data/cord/` | 100 |
| SROIE | `data/sroie/` | 347 |
| FUNSD | `data/funsd/` | 50 |

Originals in `data/` are never modified.

## Requirements

```text
opencv-python-headless
numpy
scikit-image
pillow
image-similarity-measures   # PSNR/SSIM (FSIM deps: phasepack, pyfftw)
phasepack
pyfftw
```

Install:

```bash
pip install opencv-python-headless numpy scikit-image pillow image-similarity-measures phasepack pyfftw
```

## Run

```bash
# full run (idempotent; rerun overwrites cleanly)
python run_s2_perturbations.py --workers 8

# wipe outputs and start over
python run_s2_perturbations.py --workers 8 --fresh

# resume after interruption (default): skips images that already have
# complete outputs + rows in metrics_partial.jsonl
python run_s2_perturbations.py --workers 8
```

## Outputs

```text
perturbed/
  cord/ sroie/ funsd/
    clean/                 # byte-for-byte copies of data/<ds>/
    s2_blur_l1/ ... s2_jpeg_l3/   # 12 perturbation folders each
metrics.csv                # 5964 rows = 497 × 4 × 3
metrics_partial.jsonl      # append-only resume cache
preview/
  cord_preview.png
  sroie_preview.png
  funsd_preview.png
```

### metrics.csv columns

```text
dataset,image_id,perturbation,level,psnr,ssim,fsim
```

## Perturbation grid

| Type | L1 (subtle) | L2 (moderate) | L3 (strong) |
|------|-------------|---------------|-------------|
| blur | σ=0.5, k=3 | σ=1.0, k=5 | σ=1.8, k=7 |
| noise | σ=3 | σ=8 | σ=15 |
| contrast | f=0.90 | f=0.75 | f=0.60 |
| jpeg | q=90 | q=70 | q=50 |

- **blur:** `cv2.GaussianBlur(img, (k,k), sigma)`
- **noise:** zero-mean Gaussian, clipped to [0,255]; RNG seeded from image content hash (reproducible)
- **contrast:** `img*f + mean*(1-f)` (per-channel mean)
- **jpeg:** encode to in-memory JPEG at quality q, decode (output keeps original file extension)

## Contract: `apply_visual_perturbation`

Defined in `visual_perturbations.py` — image-in / image-out, no batch logic (reused by combined scenarios S4–S7):

```python
def apply_visual_perturbation(img: np.ndarray, perturb_type: str, level: int) -> np.ndarray:
    """
    Args:
        img: HxW or HxWx3 uint8 ndarray (OpenCV BGR/gray as loaded).
        perturb_type: "blur" | "noise" | "contrast" | "jpeg"
        level: 1 | 2 | 3
    Returns:
        Perturbed image, same shape/dtype as input. Input is not modified.
    """
```

Also exported: `PERTURBATION_GRID`, `PERTURBATION_TYPES`, `LEVELS`.

## Metrics notes

- **PSNR:** full resolution; `image-similarity-measures` (`max_p=4095` default), skimage fallback.
- **SSIM:** full resolution if `H*W ≤ 400_000`, else isotropic downsample (`cv2.INTER_AREA`) — large receipts would otherwise take minutes/pair.
- **FSIM:** algorithm port in `fast_fsim.py` matching `image-similarity-measures.quality_metrics.fsim` (phasepack PC: `nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978`, T1=0.85, T2=160) with filter-bank caching + clean-image PC cache; inputs capped to `250_000` pixels. Stock library FSIM on ~34 MP images is ~minutes/pair and cannot finish 5,964 pairs in the one-hour CPU budget.
- FSIM failure → `NaN` + warning (does not abort).

## Sanity checks (run automatically)

1. **Counts:** every `s2_*` folder matches its `clean/` count.
2. **Monotonicity:** mean SSIM strictly decreases L1→L2→L3 for each perturbation.
3. **Preview:** one grid PNG per dataset (3 samples × clean + 12 perturbations).

L1 too-strong flags (SSIM&lt;0.97 or PSNR&lt;35 dB) are reported, never auto-fixed.
