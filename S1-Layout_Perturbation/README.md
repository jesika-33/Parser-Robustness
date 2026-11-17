# Benchmark for Parser Robustness to Visually Subtle Perturbations

A reproducible benchmark for studying how small image-space layout changes affect OCR performance on document images. The current implementation generates perturbed document images, records image-similarity measurements, and evaluates clean-versus-perturbed OCR performance with Tesseract.

## Overview

The benchmark is split into two independent command-line stages:

1. **`make_perturbations.py`** generates perturbed images and an image-similarity manifest.
2. **`eval_tesseract.py`** reads that manifest, runs Tesseract on each clean/perturbed pair, and calculates Character Error Rate (CER), Word Error Rate (WER), and their changes.

The stages can be run separately. For example, you can regenerate image-space perturbations and similarity metrics without running OCR, or rerun OCR evaluation using an existing manifest.

## Project data layout

The scripts expect the dataset images and annotations to follow these layouts:

```text
data/
├── CORD/
│   └── test/
│       ├── image/
│       │   └── *.png
│       └── json/
│           └── *.json
│
├── FUNSD/
│   └── testing_data/
│       ├── images/
│       │   └── *.png
│       └── annotations/
│           └── *.json
│
└── SROIE/
    └── test/
        ├── images/
        │   └── *.jpg
        └── box/
            └── *.txt
```

Each image must have a corresponding annotation/ground-truth file with the same filename stem. For example, `image/receipt_001.png` should correspond to `json/receipt_001.json`.

The scripts include several filename and directory fallbacks, but the layouts above are the expected project structure.

## Perturbations

The current implementation applies two image-space layout perturbations. The same configured relative strengths are used regardless of dataset.

| Perturbation         | Description                                                                                                                                                                          | Strengths              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------- |
| `block_displacement` | Selects one text region and shifts it horizontally. The target displacement is based on the image width and the selected strength, subject to available space at the image boundary. | `0.01`, `0.02`, `0.03` |
| `line_compression`   | Reduces the vertical gaps between detected text lines, moving subsequent lines upward.                                                                                               | `0.02`, `0.05`, `0.10` |

### Region sources

The `--regions` option controls how text regions and lines are obtained:

- `auto` (default): use annotations when a supported annotation file is found; otherwise fall back to image-based heuristics.
- `annotations`: require annotations. Images without usable annotations are skipped.
- `heuristics`: detect text lines and regions from the image using thresholding, morphology, and row projections.

Supported annotation sources in the current implementation:

- **CORD:** word quadrilaterals in `valid_line` are used to construct line boxes and grouped regions.
- **FUNSD:** form boxes and word boxes are used; word boxes are clustered into lines.
- **SROIE:** quadrilateral coordinates are read from the `box/*.txt` files and used as line boxes.

The background used to fill the original region is estimated using `--bg-strategy`:

- `border` (default): per-channel mode of a two-pixel border around the image.
- `nonink`: median of pixels classified as non-ink.

The perturbations preserve the original image dimensions; they do not add padding.

## Requirements

- Python 3.10 or later
- Tesseract OCR (including the language data used for evaluation)
- Python packages:
  - `numpy`
  - `opencv-python`
  - `scikit-image`
  - `torch`
  - `piq`
- Optional:
  - `rapidfuzz` — accelerates edit-distance calculations in the evaluation script. A pure-Python fallback is included.
  - `pyfftw` — can speed up FSIM computation when supported by the installed environment.

Install the Python dependencies in your environment:

```bash
python -m pip install numpy opencv-python scikit-image torch piq
```

Optional packages:

```bash
python -m pip install rapidfuzz pyfftw
```

Install Tesseract using the package manager for your operating system. Confirm that it is available on your `PATH`:

```bash
tesseract --version
```

You also need a Tesseract trained-data directory. The evaluation script requires its path through `--tessdata`. For example, on some Linux installations:

```text
/usr/share/tesseract-ocr/5/tessdata_best
```

Use the path that exists on your machine and contains the trained data for the language specified by `--lang`.

## Usage

Run commands from the project root, where the two Python scripts are located.

### 1. Generate perturbations and image-similarity metrics

Example for CORD:

```bash
python make_perturbations.py \
    --dataset CORD \
    --root data/CORD \
    --out out \
    --regions auto
```

For the other datasets, change `--dataset` and `--root`:

```bash
python make_perturbations.py \
    --dataset FUNSD \
    --root data/FUNSD \
    --out out \
    --regions auto

python make_perturbations.py \
    --dataset SROIE \
    --root data/SROIE \
    --out out \
    --regions auto
```

Useful options:

| Option             | Description                                                                               |
| ------------------ | ----------------------------------------------------------------------------------------- |
| `--dataset`        | Dataset identifier, such as `CORD`, `FUNSD`, or `SROIE`. Required.                        |
| `--root`           | Dataset root directory. Required.                                                         |
| `--out`            | Output directory. Required.                                                               |
| `--split`          | Split label written to the manifest. Defaults to `test`.                                  |
| `--limit`          | Maximum number of images to process; `0` means all discovered images.                     |
| `--regions`        | `auto`, `annotations`, or `heuristics`. Defaults to `auto`.                               |
| `--bg-strategy`    | `border` or `nonink`. Defaults to `border`.                                               |
| `--fsim-strengths` | Comma-separated strengths at which FSIM is computed. Defaults to `0.02,0.05,0.10`.        |
| `--flush-every`    | Write intermediate CSV results every N manifest rows; `0` disables intermediate flushing. |

**FSIM runtime note:** PSNR and SSIM are computed for every successfully generated variant. FSIM is computed only at the strengths listed in `--fsim-strengths`, because it can be substantially more expensive. To skip FSIM for a faster pilot run:

```bash
python make_perturbations.py \
    --dataset CORD \
    --root data/CORD \
    --out out \
    --regions auto \
    --fsim-strengths ""
```

You can also sample only one strength, for example `--fsim-strengths "0.05"`.

### 2. Evaluate clean and perturbed images with Tesseract

After generating the manifest, run:

```bash
python eval_tesseract.py \
    --manifest out/manifest.csv \
    --tessdata /usr/share/tesseract-ocr/5/tessdata_best \
    --psm 3 \
    --oem 1 \
    --normalize \
    --out out/tesseract_results.csv
```

Replace the example `--tessdata` path with the trained-data directory on your system.

Useful options:

| Option             | Description                                                                                          |
| ------------------ | ---------------------------------------------------------------------------------------------------- |
| `--manifest`       | Path to the `manifest.csv` generated by the perturbation stage. Required.                            |
| `--tessdata`       | Tesseract trained-data directory. Required.                                                          |
| `--out`            | Output CSV path. Required.                                                                           |
| `--psm`            | Tesseract page segmentation mode. Defaults to `3`.                                                   |
| `--oem`            | Tesseract OCR engine mode. Defaults to `1`.                                                          |
| `--lang`           | OCR language code. Defaults to `eng`.                                                                |
| `--normalize`      | Collapse consecutive whitespace and trim leading/trailing whitespace before calculating CER and WER. |
| `--flush-every`    | Write intermediate results every N successful rows; defaults to `20`. Set to `0` to disable.         |
| `--progress-every` | Print progress every N manifest rows; defaults to `50`. Set to `0` to disable.                       |

The evaluation script caches the clean-image OCR result for each unique clean image path, so a clean image is not OCR-processed again for every perturbation variant.

## Outputs

With `--out out`, the perturbation stage produces:

```text
out/
├── manifest.csv
├── skipped.csv
├── summary.csv
└── <DATASET>/
    ├── block_displacement/
    │   ├── 0.01/
    │   ├── 0.02/
    │   └── 0.03/
    └── line_compression/
        ├── 0.02/
        ├── 0.05/
        └── 0.10/
```

- **`manifest.csv`** records each successfully generated variant, including the dataset, image ID, perturbation, strength, clean and perturbed image paths, PSNR, SSIM, FSIM (when sampled), region source, and perturbation details.
- **`skipped.csv`** records perturbation attempts that could not produce a valid output, along with a reason and diagnostic details. It is written when there are skipped attempts.
- **`summary.csv`** aggregates image-similarity statistics by perturbation and strength, including the number of samples, mean PSNR, mean and minimum SSIM, and FSIM sample counts and statistics.
- **Generated PNG files** are stored under the dataset, perturbation, and strength directories.

The evaluation stage produces the CSV specified by `--out`, for example `out/tesseract_results.csv`. Its columns include:

| Column                           | Meaning                                                                 |
| -------------------------------- | ----------------------------------------------------------------------- |
| `dataset`, `split`, `image_id`   | Dataset and sample identifiers.                                         |
| `perturbation`, `strength`       | Perturbation type and configured strength.                              |
| `clean_cer`, `pert_cer`, `d_cer` | CER on the clean image, CER on the perturbed image, and the difference. |
| `clean_wer`, `pert_wer`, `d_wer` | WER on the clean image, WER on the perturbed image, and the difference. |
| `psnr`, `ssim`                   | Image-similarity values copied from the manifest.                       |

The evaluation output includes PSNR and SSIM, but does not currently copy FSIM into `tesseract_results.csv`; FSIM remains available in `manifest.csv` and `summary.csv`.

## Metrics

### Character Error Rate (CER)

CER is calculated using character-level Levenshtein edit distance:

\[
CER = \frac{S + D + I}{N}
\]

where `S`, `D`, and `I` are substitutions, deletions, and insertions, and `N` is the number of characters in the reference text.

### Word Error Rate (WER)

WER uses the same calculation at the word level:

\[
WER = \frac{S_w + D_w + I_w}{N_w}
\]

where the edit distance is calculated over whitespace-separated tokens.

### Change in error rate

For each clean/perturbed pair:

\[
\Delta CER = CER_{\text{perturbed}} - CER_{\text{clean}}
\]

\[
\Delta WER = WER_{\text{perturbed}} - WER_{\text{clean}}
\]

A positive delta means the perturbed image produced a higher error rate than its clean counterpart. A negative delta means the measured error rate decreased.

When `--normalize` is enabled, repeated whitespace is collapsed and leading/trailing whitespace is removed from both reference and OCR text before CER and WER are computed.

PSNR and SSIM quantify image similarity between the clean and perturbed images. FSIM is an optional, selectively computed metric in the perturbation-generation stage.

## Ground-truth handling

The evaluation script loads ground truth according to the dataset annotation format:

- **FUNSD:** collects the `text` values from entries in `form`.
- **CORD v1:** recursively flattens string values under `gt_parse`.
- **CORD v2:** collects word text from `valid_line[*].words[*].text`.
- **SROIE:** reads the transcript after the first eight comma-separated coordinate fields in each box annotation line.
- **Fallback:** recursively collects string-valued `text` fields from JSON objects if the expected schema is not detected.
