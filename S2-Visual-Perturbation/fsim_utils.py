"""PSNR / SSIM / FSIM helpers with documented caps for large images.

Budget note: full-res phase-congruency FSIM and skimage SSIM on ~34 MP
receipts take minutes per pair; 5,964 pairs cannot finish under one hour.
PSNR is cheap → always full resolution.

Policy (README_perturbation.md):
  - PSNR: full resolution (image-similarity-measures; skimage fallback)
  - SSIM: full res if pixels <= SSIM_MAX_PIXELS, else isotropic downsample
  - FSIM: fast_fsim (validated port of image-similarity-measures FSIM);
    inputs capped to FSIM_MAX_PIXELS; NaN + warning on failure
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

from fast_fsim import FSIMCache, fsim_gray

SSIM_MAX_PIXELS = 400_000
FSIM_MAX_PIXELS = 250_000

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

try:
    from skimage.metrics import peak_signal_noise_ratio as _sk_psnr
    from skimage.metrics import structural_similarity as _sk_ssim
except Exception:  # pragma: no cover
    _sk_psnr = None
    _sk_ssim = None


def ism_available() -> bool:
    return _ISM_OK


def ism_error() -> str:
    return _ISM_ERR


def _n_pixels(img: np.ndarray) -> int:
    return int(img.shape[0]) * int(img.shape[1])


def downsample_to_cap(img: np.ndarray, max_pixels: int) -> np.ndarray:
    n = _n_pixels(img)
    if n <= max_pixels:
        return img
    scale = (max_pixels / n) ** 0.5
    new_w = max(1, int(round(img.shape[1] * scale)))
    new_h = max(1, int(round(img.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported shape {img.shape}")


def to_rgb_u8(img: np.ndarray) -> np.ndarray:
    """BGR/gray uint8 → RGB uint8 HxWx3 for metric libraries."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    raise ValueError(f"Unsupported shape {img.shape}")


def compute_psnr(clean: np.ndarray, pert: np.ndarray) -> float:
    a = to_rgb_u8(clean)
    b = to_rgb_u8(pert)
    if a.shape != b.shape:
        return float("nan")
    # Fast MSE in float32 (ISM psnr casts both to float32 anyway)
    d = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(d * d))
    if mse == 0:
        return float("inf")
    # Match ISM default max_p=4095
    return float(20.0 * np.log10(4095.0 / np.sqrt(mse)))


def compute_ssim(clean: np.ndarray, pert: np.ndarray) -> float:
    try:
        a = to_rgb_u8(clean)
        b = to_rgb_u8(pert)
        if a.shape != b.shape:
            return float("nan")
        if _n_pixels(a) > SSIM_MAX_PIXELS:
            a = downsample_to_cap(a, SSIM_MAX_PIXELS)
            b = downsample_to_cap(b, SSIM_MAX_PIXELS)
            if a.shape != b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        if _ISM_OK:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(_ism_ssim(a, b))
        if _sk_ssim is not None:
            return float(_sk_ssim(a, b, channel_axis=2, data_range=255))
        return float("nan")
    except Exception:
        return float("nan")


def make_fsim_scorer(clean: np.ndarray):
    """Return callable pred -> FSIM, with clean PC/GM cached. Caps resolution."""
    g = to_gray_u8(clean)
    g = downsample_to_cap(g, FSIM_MAX_PIXELS)
    # FSIMCache expects arrays it can convert; pass gray via RGB-compatible path
    cache = FSIMCache(g[:, :, None] if g.ndim == 2 else g)

    def score(pred: np.ndarray) -> float:
        try:
            p = to_gray_u8(pred)
            p = downsample_to_cap(p, FSIM_MAX_PIXELS)
            if p.shape != g.shape:
                p = cv2.resize(p, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
            return cache.compute(p)
        except Exception as e:
            warnings.warn(f"FSIM failed, writing NaN: {e}", stacklevel=2)
            return float("nan")

    return score


def compute_fsim(clean: np.ndarray, pert: np.ndarray) -> float:
    return make_fsim_scorer(clean)(pert)
