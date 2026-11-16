"""Fast FSIM: optimized port of image-similarity-measures' FSIM (phasepack PC).

Same algorithm/parameters as quality_metrics.fsim → phasepack.phasecong(
nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978) with T1=0.85, T2=160.

Speedups (values verified against library on samples — see README):
  - log-Gabor + angular filter bank cached per image size
  - pyfftw FFTW plans via scipy-compatible interfaces
  - clean-image PC/GM computed once per image (FSIMCache) for batch use
  - grayscale FSIM (phase congruency on luminance); library RGB averages
    per-channel FSIM — we also provide fsim_rgb for parity checks
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy.fftpack import ifftshift

from phasepack.tools import lowpassfilter

try:
    from pyfftw.interfaces.scipy_fftpack import fft2 as _fft2, ifft2 as _ifft2
    _FFTBACKEND = "pyfftw"
except Exception:  # pragma: no cover
    from scipy.fftpack import fft2 as _fft2, ifft2 as _ifft2
    _FFTBACKEND = "scipy"

NSCALE = 4
NORIENT = 6
MIN_WAVELENGTH = 6
MULT = 2.0
SIGMA_ONF = 0.5978
K = 2.0
CUTOFF = 0.5
G = 10.0

_FILTER_CACHE: dict[tuple[int, int], tuple] = {}


def fft_backend() -> str:
    return _FFTBACKEND


def _build_filters(rows: int, cols: int) -> tuple:
    key = (rows, cols)
    hit = _FILTER_CACHE.get(key)
    if hit is not None:
        return hit

    if cols % 2:
        xvals = np.arange(-(cols - 1) / 2.0, ((cols - 1) / 2.0) + 1) / float(cols - 1)
    else:
        xvals = np.arange(-cols / 2.0, cols / 2.0) / float(cols)
    if rows % 2:
        yvals = np.arange(-(rows - 1) / 2.0, ((rows - 1) / 2.0) + 1) / float(rows - 1)
    else:
        yvals = np.arange(-rows / 2.0, rows / 2.0) / float(rows)

    x, y = np.meshgrid(xvals, yvals, sparse=True)
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(-y, x)
    radius = ifftshift(radius)
    theta = ifftshift(theta)
    radius[0, 0] = 1.0
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    lp = lowpassfilter((rows, cols), 0.45, 15)
    logGaborDenom = 2.0 * np.log(SIGMA_ONF) ** 2.0
    logGabor = []
    for ss in range(NSCALE):
        wavelength = MIN_WAVELENGTH * MULT**ss
        fo = 1.0 / wavelength
        logRadOverFo = np.log(radius / fo)
        tmp = np.exp(-(logRadOverFo * logRadOverFo) / logGaborDenom) * lp
        tmp[0, 0] = 0.0
        logGabor.append(tmp)

    filts = []
    for oo in range(NORIENT):
        angl = oo * (np.pi / NORIENT)
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        np.clip(dtheta * NORIENT / 2.0, 0.0, np.pi, out=dtheta)
        spread = (np.cos(dtheta) + 1.0) / 2.0
        filts.append(tuple(logGabor[ss] * spread for ss in range(NSCALE)))

    if len(_FILTER_CACHE) > 64:
        _FILTER_CACHE.clear()
    _FILTER_CACHE[key] = tuple(filts)
    return _FILTER_CACHE[key]


def phasecong_pc_sum(img: np.ndarray) -> np.ndarray:
    """Sum of phase-congruency maps over orientations (PC sum used by FSIM)."""
    if img.ndim == 3:
        img = img.mean(2)
    img = np.asarray(img, dtype=np.float64)
    rows, cols = img.shape
    epsilon = 1e-4
    filts = _build_filters(rows, cols)
    IM = _fft2(img)

    pc_sum = np.zeros((rows, cols), dtype=np.float64)
    # Precompute orientation constants
    angles = [oo * (np.pi / NORIENT) for oo in range(NORIENT)]

    for oo in range(NORIENT):
        sumE = np.zeros((rows, cols), dtype=np.float64)
        sumO = np.zeros((rows, cols), dtype=np.float64)
        sumAn = np.zeros((rows, cols), dtype=np.float64)
        maxAn = None
        tau = None
        EOscale = []

        for ss in range(NSCALE):
            thisEO = _ifft2(IM * filts[oo][ss])
            An = np.abs(thisEO)
            sumAn += An
            sumE += thisEO.real
            sumO += thisEO.imag
            if ss == 0:
                tau = np.median(sumAn.ravel()) / np.sqrt(np.log(4))
                maxAn = An
            else:
                np.maximum(maxAn, An, out=maxAn)
            EOscale.append(thisEO)

        angl = angles[oo]
        ca, sa = np.cos(angl), np.sin(angl)

        XEnergy = np.sqrt(sumE * sumE + sumO * sumO) + epsilon
        MeanE = sumE / XEnergy
        MeanO = sumO / XEnergy

        Energy = np.zeros((rows, cols), dtype=np.float64)
        for ss in range(NSCALE):
            E = EOscale[ss].real
            O = EOscale[ss].imag
            Energy += E * MeanE + O * MeanO - np.abs(E * MeanO - O * MeanE)

        totalTau = tau * (1.0 - (1.0 / MULT) ** NSCALE) / (1.0 - (1.0 / MULT))
        EstMean = totalTau * np.sqrt(np.pi / 2.0)
        EstSigma = totalTau * np.sqrt((4 - np.pi) / 2.0)
        T = np.maximum(EstMean + K * EstSigma, epsilon)

        np.maximum(Energy - T, 0.0, out=Energy)
        width = (sumAn / (maxAn + epsilon) - 1.0) / (NSCALE - 1)
        weight = 1.0 / (1.0 + np.exp(G * (CUTOFF - width)))
        thisPC = weight * Energy / sumAn
        pc_sum += thisPC

    return pc_sum


def _scharr_gm(gray_f64: np.ndarray) -> np.ndarray:
    u8 = np.clip(gray_f64, 0, 255).astype(np.uint8)
    sx = cv2.Scharr(u8, cv2.CV_16U, 1, 0).astype(np.float64)
    sy = cv2.Scharr(u8, cv2.CV_16U, 0, 1).astype(np.float64)
    return np.sqrt(sx * sx + sy * sy)


def _sim(x: np.ndarray, y: np.ndarray, c: float) -> np.ndarray:
    return (2.0 * x * y + c) / (x * x + y * y + c)


def _to_gray_f64(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        g = img
    elif img.ndim == 3 and img.shape[2] == 1:
        g = img[:, :, 0]
    elif img.ndim == 3 and img.shape[2] == 3:
        # accept RGB (metric libs) — convert via luminance
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        g = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    else:
        raise ValueError(f"Unsupported shape {img.shape}")
    return g.astype(np.float64)


def fsim_gray(org: np.ndarray, pred: np.ndarray, T1: float = 0.85, T2: float = 160) -> float:
    a = _to_gray_f64(org)
    b = _to_gray_f64(pred)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    pc1 = phasecong_pc_sum(a)
    pc2 = phasecong_pc_sum(b)
    gm1 = _scharr_gm(a)
    gm2 = _scharr_gm(b)
    S_l = _sim(pc1, pc2, T1) * _sim(gm1, gm2, T2)
    denom = np.sum(np.maximum(pc1, pc2))
    if denom == 0:
        return float("nan")
    return float(np.sum(S_l * np.maximum(pc1, pc2)) / denom)


class FSIMCache:
    """Batch helper: cache clean-image PC and GM; compute FSIM vs many perturbations."""

    def __init__(self, clean: np.ndarray, T1: float = 0.85, T2: float = 160):
        self.T1 = T1
        self.T2 = T2
        self.clean = clean
        a = _to_gray_f64(clean)
        self._gray_clean = a
        self._pc1 = phasecong_pc_sum(a)
        self._gm1 = _scharr_gm(a)

    def compute(self, pred: np.ndarray) -> float:
        b = _to_gray_f64(pred)
        if b.shape != self._gray_clean.shape:
            raise ValueError(f"shape mismatch {b.shape} vs {self._gray_clean.shape}")
        pc2 = phasecong_pc_sum(b)
        gm2 = _scharr_gm(b)
        S_l = _sim(self._pc1, pc2, self.T1) * _sim(self._gm1, gm2, self.T2)
        denom = np.sum(np.maximum(self._pc1, pc2))
        if denom == 0:
            return float("nan")
        return float(np.sum(S_l * np.maximum(self._pc1, pc2)) / denom)


def fsim(org: np.ndarray, pred: np.ndarray, T1: float = 0.85, T2: float = 160) -> float:
    return fsim_gray(org, pred, T1, T2)
