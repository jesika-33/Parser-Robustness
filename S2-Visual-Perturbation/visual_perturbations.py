"""S2 visual perturbations for document-parser robustness benchmark.

Image-in / image-out API reused later by combined scenarios S4-S7.
"""

from __future__ import annotations

import hashlib
import io

import cv2
import numpy as np

# Perturbation grid: type -> level -> params
PERTURBATION_GRID: dict[str, dict[int, dict]] = {
    "blur": {
        1: {"sigma": 0.5, "k": 3},
        2: {"sigma": 1.0, "k": 5},
        3: {"sigma": 1.8, "k": 7},
    },
    "noise": {
        1: {"sigma": 3},
        2: {"sigma": 8},
        3: {"sigma": 15},
    },
    "contrast": {
        1: {"factor": 0.90},
        2: {"factor": 0.75},
        3: {"factor": 0.60},
    },
    "jpeg": {
        1: {"quality": 90},
        2: {"quality": 70},
        3: {"quality": 50},
    },
}

PERTURBATION_TYPES = tuple(PERTURBATION_GRID.keys())
LEVELS = (1, 2, 3)

BASE_SEED = 20240517


def _rng_for(img: np.ndarray, perturb_type: str, level: int) -> np.random.Generator:
    """Deterministic per-image RNG derived from content hash + perturb keys."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(img).tobytes())
    h.update(f"{perturb_type}:{level}:{BASE_SEED}".encode())
    seed = int.from_bytes(h.digest()[:8], "little")
    return np.random.default_rng(seed)


def _gaussian_blur(img: np.ndarray, level: int) -> np.ndarray:
    p = PERTURBATION_GRID["blur"][level]
    k = int(p["k"])
    return cv2.GaussianBlur(img, (k, k), sigmaX=float(p["sigma"]), sigmaY=float(p["sigma"]))


def _gaussian_noise(img: np.ndarray, level: int) -> np.ndarray:
    p = PERTURBATION_GRID["noise"][level]
    sigma = float(p["sigma"])
    rng = _rng_for(img, "noise", level)
    # float32 path: much faster/lower memory on multi-MP images than float64
    noise = rng.normal(loc=0.0, scale=sigma, size=img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(img.dtype)


def _contrast(img: np.ndarray, level: int) -> np.ndarray:
    p = PERTURBATION_GRID["contrast"][level]
    f = float(p["factor"])
    mean = img.astype(np.float32).mean(axis=(0, 1), keepdims=True)
    out = img.astype(np.float32) * f + mean * (1.0 - f)
    return np.clip(out, 0, 255).astype(img.dtype)


def _jpeg(img: np.ndarray, level: int) -> np.ndarray:
    p = PERTURBATION_GRID["jpeg"][level]
    q = int(p["quality"])
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError(f"JPEG encode failed at quality={q}")
    # IMREAD_UNCHANGED: preserve grayscale (IMREAD_COLOR would force 3-channel)
    out = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if out is None:
        raise RuntimeError(f"JPEG decode failed at quality={q}")
    # Some JPEG decoders still return BGR for gray sources — restore input rank
    if img.ndim == 2 and out.ndim == 3:
        if out.shape[2] == 3:
            # identical channels → gray; otherwise keep as-is (shouldn't happen)
            if np.array_equal(out[:, :, 0], out[:, :, 1]) and np.array_equal(out[:, :, 1], out[:, :, 2]):
                out = out[:, :, 0]
        elif out.shape[2] == 1:
            out = out[:, :, 0]
    elif img.ndim == 3 and img.shape[2] == 1 and out.ndim == 3 and out.shape[2] == 3:
        out = out[:, :, :1]
    return out


_IMPL = {
    "blur": _gaussian_blur,
    "noise": _gaussian_noise,
    "contrast": _contrast,
    "jpeg": _jpeg,
}


def apply_visual_perturbation(img: np.ndarray, perturb_type: str, level: int) -> np.ndarray:
    """Apply a single S2 visual perturbation.

    Args:
        img: Input image as HxWx3 (or HxW) uint8 ndarray, RGB order.
        perturb_type: One of "blur", "noise", "contrast", "jpeg".
        level: 1 (subtle), 2 (moderate), or 3 (strong).

    Returns:
        Perturbed image, same shape/dtype as input. Input is not modified.
    """
    if perturb_type not in _IMPL:
        raise ValueError(f"Unknown perturb_type {perturb_type!r}; expected one of {PERTURBATION_TYPES}")
    if level not in LEVELS:
        raise ValueError(f"Unknown level {level!r}; expected one of {LEVELS}")
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy ndarray")
    return _IMPL[perturb_type](img, level)


def jpeg_compress_bytes(img: np.ndarray, quality: int) -> bytes:
    """Helper: encode image to JPEG bytes at given quality (for buffer round-trips)."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def jpeg_roundtrip_from_bytes(data: bytes) -> np.ndarray:
    """Helper: decode JPEG bytes back to image array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("JPEG decode failed")
    return img
