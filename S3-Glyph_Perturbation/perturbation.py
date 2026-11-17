import os
import re
import sys
import json
import math
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------------------
# OpenCV import with seamless PIL/NumPy fallback when cv2 is not in current env
# ---------------------------------------------------------------------------
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

    class _CV2Compat:
        COLOR_BGR2GRAY = 6
        COLOR_GRAY2BGR = 8
        MORPH_ELLIPSE = 2
        MORPH_RECT = 0
        INTER_AREA = 3
        INTER_LINEAR = 1
        INTER_CUBIC = 2
        INTER_NEAREST = 0
        LINE_AA = 16
        CV_32F = 5
        DIST_L2 = 2

        @staticmethod
        def imread(path):
            if not os.path.exists(path):
                return None
            try:
                img = Image.open(path).convert('RGB')
                arr = np.array(img, dtype=np.uint8)
                return arr[:, :, ::-1].copy()  # RGB -> BGR
            except Exception:
                return None

        @staticmethod
        def imwrite(path, img):
            parent = os.path.dirname(os.path.abspath(path))
            if parent and not os.path.isdir(parent):
                try:
                    os.makedirs(parent, exist_ok=True)
                except PermissionError:
                    pass
            arr = np.clip(img, 0, 255).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 3:
                pil_img = Image.fromarray(arr[:, :, ::-1], mode='RGB')
            else:
                pil_img = Image.fromarray(arr, mode='L')
            pil_img.save(path)
            return True

        @staticmethod
        def cvtColor(img, code):
            if code == _CV2Compat.COLOR_BGR2GRAY:
                f = img.astype(np.float32)
                g = 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]
                return np.clip(np.round(g), 0, 255).astype(img.dtype) if img.dtype == np.uint8 else g
            elif code == _CV2Compat.COLOR_GRAY2BGR:
                return np.repeat(img[..., np.newaxis], 3, axis=-1)
            return img.copy()

        @staticmethod
        def getStructuringElement(shape, ksize):
            kw, kh = ksize
            y, x = np.ogrid[-kh // 2:kh - kh // 2, -kw // 2:kw - kw // 2]
            if shape == _CV2Compat.MORPH_ELLIPSE:
                rx = max(1.0, kw / 2.0)
                ry = max(1.0, kh / 2.0)
                mask = ((x / rx) ** 2 + (y / ry) ** 2) <= 1.05
                return mask.astype(np.uint8)
            return np.ones((kh, kw), dtype=np.uint8)

        @staticmethod
        def medianBlur(img, ksize):
            if ksize <= 1:
                return img.copy()
            if img.ndim == 2:
                pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode='L')
                return np.array(pil.filter(ImageFilter.MedianFilter(size=ksize)), dtype=img.dtype)
            channels = []
            for c in range(img.shape[2]):
                pil = Image.fromarray(np.clip(img[..., c], 0, 255).astype(np.uint8), mode='L')
                channels.append(np.array(pil.filter(ImageFilter.MedianFilter(size=ksize)), dtype=img.dtype))
            return np.stack(channels, axis=-1)

        @staticmethod
        def _morph_channel(ch, ksize, is_dilate):
            r = ksize // 2
            if r <= 0:
                return ch.copy()
            h, w = ch.shape[:2]
            px = np.pad(ch, ((0, 0), (r, r)), mode='edge')
            op = np.maximum.reduce if is_dilate else np.minimum.reduce
            hx = op([px[:, i:i + w] for i in range(ksize)])
            py = np.pad(hx, ((r, r), (0, 0)), mode='edge')
            return op([py[i:i + h, :] for i in range(ksize)])

        @staticmethod
        def dilate(img, kernel, iterations=1):
            kh, kw = kernel.shape[:2]
            ksize = max(3, kh | 1)
            out = img.copy()
            for _ in range(iterations):
                if out.ndim == 2:
                    out = _CV2Compat._morph_channel(out, ksize, True)
                else:
                    out = np.stack([_CV2Compat._morph_channel(out[..., c], ksize, True) for c in range(out.shape[2])], axis=-1)
            return out

        @staticmethod
        def erode(img, kernel, iterations=1):
            kh, kw = kernel.shape[:2]
            ksize = max(3, kh | 1)
            out = img.copy()
            for _ in range(iterations):
                if out.ndim == 2:
                    out = _CV2Compat._morph_channel(out, ksize, False)
                else:
                    out = np.stack([_CV2Compat._morph_channel(out[..., c], ksize, False) for c in range(out.shape[2])], axis=-1)
            return out

        @staticmethod
        def GaussianBlur(img, ksize, sigmaX, sigmaY=0.0):
            sigma = float(sigmaX)
            if sigma <= 0.0:
                kw, kh = ksize
                sigma = 0.3 * ((max(kw, kh) - 1) * 0.5 - 1) + 0.8
            radius = max(1, int(math.ceil(3.0 * sigma)))
            x = np.arange(-radius, radius + 1, dtype=np.float32)
            kernel_1d = np.exp(-0.5 * (x / max(1e-4, sigma)) ** 2)
            kernel_1d /= kernel_1d.sum()

            def _convolve_2d(arr2d):
                pad_x = np.pad(arr2d, ((0, 0), (radius, radius)), mode='reflect')
                tmp = sum(kernel_1d[i] * pad_x[:, i:i + arr2d.shape[1]] for i in range(len(kernel_1d)))
                pad_y = np.pad(tmp, ((radius, radius), (0, 0)), mode='reflect')
                res = sum(kernel_1d[i] * pad_y[i:i + arr2d.shape[0], :] for i in range(len(kernel_1d)))
                return res

            arr_f = img.astype(np.float32)
            if arr_f.ndim == 2:
                res = _convolve_2d(arr_f)
            else:
                res = np.stack([_convolve_2d(arr_f[..., c]) for c in range(arr_f.shape[2])], axis=-1)
            return res.astype(img.dtype) if img.dtype == np.uint8 else res

        @staticmethod
        def resize(img, dsize, interpolation=3):
            target_w, target_h = dsize
            if target_w <= 0 or target_h <= 0:
                return np.zeros((max(1, target_h), max(1, target_w)), dtype=img.dtype)
            pil_Map = {
                _CV2Compat.INTER_NEAREST: Image.Resampling.NEAREST,
                _CV2Compat.INTER_LINEAR: Image.Resampling.BILINEAR,
                _CV2Compat.INTER_CUBIC: Image.Resampling.BICUBIC,
                _CV2Compat.INTER_AREA: Image.Resampling.BOX,
            }
            resample = pil_Map.get(interpolation, Image.Resampling.BILINEAR)
            if img.ndim == 2:
                pil = Image.fromarray(img.astype(np.float32), mode='F')
                out = np.array(pil.resize((target_w, target_h), resample=resample), dtype=np.float32)
                return np.clip(out, 0, 255).astype(np.uint8) if img.dtype == np.uint8 else out
            channels = []
            for c in range(img.shape[2]):
                pil = Image.fromarray(img[..., c].astype(np.float32), mode='F')
                channels.append(np.array(pil.resize((target_w, target_h), resample=resample), dtype=np.float32))
            out = np.stack(channels, axis=-1)
            return np.clip(out, 0, 255).astype(np.uint8) if img.dtype == np.uint8 else out

        @staticmethod
        def Sobel(src, ddepth, dx, dy, ksize=3):
            f = src.astype(np.float32)
            p = np.pad(f, ((1, 1), (1, 1)), mode='reflect')
            if dx == 1 and dy == 0:
                gx = (p[:-2, 2:] + 2.0 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2.0 * p[1:-1, :-2] + p[2:, :-2])
                return gx
            elif dx == 0 and dy == 1:
                gy = (p[2:, :-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2.0 * p[:-2, 1:-1] + p[:-2, 2:])
                return gy
            return np.zeros_like(f)

        @staticmethod
        def Laplacian(src, ddepth, ksize=1):
            f = src.astype(np.float32)
            p = np.pad(f, ((1, 1), (1, 1)), mode='reflect')
            return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * f)

        @staticmethod
        def distanceTransform(src, distanceType, maskSize):
            # Iterative chamfer distance approximation on binary uint8 mask
            mask = (src > 0)
            if not np.any(mask):
                return np.zeros(src.shape, dtype=np.float32)
            h, w = src.shape
            dist = np.where(mask, 1e4, 0.0).astype(np.float32)
            for y in range(h):
                for x in range(w):
                    if dist[y, x] > 0:
                        v = dist[y, x]
                        if y > 0:
                            v = min(v, dist[y - 1, x] + 1.0)
                            if x > 0:
                                v = min(v, dist[y - 1, x - 1] + 1.4142)
                            if x + 1 < w:
                                v = min(v, dist[y - 1, x + 1] + 1.4142)
                        if x > 0:
                            v = min(v, dist[y, x - 1] + 1.0)
                        dist[y, x] = v
            for y in range(h - 1, -1, -1):
                for x in range(w - 1, -1, -1):
                    if dist[y, x] > 0:
                        v = dist[y, x]
                        if y + 1 < h:
                            v = min(v, dist[y + 1, x] + 1.0)
                            if x + 1 < w:
                                v = min(v, dist[y + 1, x + 1] + 1.4142)
                            if x > 0:
                                v = min(v, dist[y + 1, x - 1] + 1.4142)
                        if x + 1 < w:
                            v = min(v, dist[y, x + 1] + 1.0)
                        dist[y, x] = v
            return dist

        @staticmethod
        def line(img, pt1, pt2, color, thickness=1, lineType=16):
            h, w = img.shape[:2]
            pil = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8), mode='L')
            draw = ImageDraw.Draw(pil)
            val = int(round(float(color) * 255.0)) if float(color) <= 1.05 else int(round(float(color)))
            draw.line([pt1, pt2], fill=val, width=max(1, int(thickness)))
            img[:, :] = np.array(pil, dtype=np.float32) / (255.0 if float(color) <= 1.05 else 1.0)
            return img

        @staticmethod
        def circle(img, center, radius, color, thickness=1, lineType=16):
            cx, cy = center
            r = max(1, int(radius))
            pil = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8), mode='L')
            draw = ImageDraw.Draw(pil)
            val = int(round(float(color) * 255.0)) if float(color) <= 1.05 else int(round(float(color)))
            bbox = [cx - r, cy - r, cx + r, cy + r]
            if thickness < 0:
                draw.ellipse(bbox, fill=val)
            else:
                draw.ellipse(bbox, outline=val, width=max(1, int(thickness)))
            img[:, :] = np.array(pil, dtype=np.float32) / (255.0 if float(color) <= 1.05 else 1.0)
            return img

        @staticmethod
        def ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=1, lineType=16):
            cx, cy = center
            rx, ry = max(1, int(axes[0])), max(1, int(axes[1]))
            pil = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8), mode='L')
            draw = ImageDraw.Draw(pil)
            val = int(round(float(color) * 255.0)) if float(color) <= 1.05 else int(round(float(color)))
            bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
            if thickness < 0:
                draw.ellipse(bbox, fill=val)
            else:
                draw.arc(bbox, start=startAngle, end=endAngle, fill=val, width=max(1, int(thickness)))
            img[:, :] = np.array(pil, dtype=np.float32) / (255.0 if float(color) <= 1.05 else 1.0)
            return img

    cv2 = _CV2Compat()


# ===========================================================================
# 3. Curated Perturbation Set & Spatial Spacing Rules
# ===========================================================================
# Category (a): Visually distinct 1-to-1 homoglyphs (Alphanumeric & Case Lookalikes)
#   - Strictly excludes identical-outline cross-script swaps (e.g., Latin 'a'/'C' -> Cyrillic 'а'/'С')
#     that produce zero visual pixel change when rendered with TrueType fonts.
#   - Strictly excludes jarring symbol morphs (such as 5 -> $ or 9 -> g).
#   - Every homoglyph substitution below guarantees a genuine visual glyph shape change
#     between confusable lookalike symbols: 0 <-> O (and o -> 0), 1 <-> I/l, 5 <-> S (and s -> 5),
#     8 <-> B, and 2 <-> Z (and z -> 2).

CROSS_SCRIPT_HOMOGLYPH_MAP = {}

ALPHANUM_HOMOGLYPH_MAP = {
    'O': ['0'],
    '0': ['O'],
    'o': ['0'],
    'I': ['1'],
    'l': ['1'],
    '1': ['l', 'I'],
    'S': ['5'],
    's': ['5'],
    '5': ['S'],
    'B': ['8'],
    '8': ['B'],
    'Z': ['2'],
    'z': ['2'],
    '2': ['Z'],
}

HOMOGLYPH_MAP = {k: list(v) for k, v in ALPHANUM_HOMOGLYPH_MAP.items()}

# Category (b): Subtle single-mark diacritics across both lowercase and uppercase letters
#   - Lowercase: é, è, ë, à, á, â, ó, ò, ú, ù, í, ì, ç
#   - Uppercase: É, È, Ë, À, Á, Â, Ó, Ò, Ú, Ù, Í, Ì, Ç (ensures ALL-CAPS and Title-Case words
#     receive genuine visual diacritic perturbations rather than no-op cross-script swaps)
DIACRITIC_CHAR_MAP = {
    'e': [('é', 'acute'), ('è', 'grave'), ('ë', 'diaeresis')],
    'a': [('à', 'grave'), ('á', 'acute'), ('â', 'circumflex')],
    'o': [('ó', 'acute'), ('ò', 'grave')],
    'u': [('ú', 'acute'), ('ù', 'grave')],
    'i': [('í', 'acute'), ('ì', 'grave')],
    'c': [('ç', 'cedilla')],
    'E': [('É', 'acute'), ('È', 'grave'), ('Ë', 'diaeresis')],
    'A': [('À', 'grave'), ('Á', 'acute'), ('Â', 'circumflex')],
    'O': [('Ó', 'acute'), ('Ò', 'grave')],
    'U': [('Ú', 'acute'), ('Ù', 'grave')],
    'I': [('Í', 'acute'), ('Ì', 'grave')],
    'C': [('Ç', 'cedilla')],
}

# Keep legacy key view for backwards compatibility
DIACRITIC_MAP = {k: [accent for _, accent in v] for k, v in DIACRITIC_CHAR_MAP.items()}

STRENGTH_LEVELS = {
    'low': 0.10,
    'medium': 0.20,
    'high': 0.35,
}


# ===========================================================================
# 1. Adaptive Font Pool & Geometry / Stroke Matching
# ===========================================================================
FONT_CANDIDATE_SPECS = [
    # Condensed sans-serif (thermal receipts)
    {'name': 'bahnschrift_light_cond', 'file': 'bahnschrift.ttf', 'style': 'condensed_sans', 'variation': 'Light Condensed', 'weight': 300},
    {'name': 'bahnschrift_semilight_cond', 'file': 'bahnschrift.ttf', 'style': 'condensed_sans', 'variation': 'SemiLight Condensed', 'weight': 350},
    {'name': 'arial_narrow', 'file': 'ARIALN.TTF', 'style': 'condensed_sans', 'variation': None, 'weight': 400},
    {'name': 'bahnschrift_cond', 'file': 'bahnschrift.ttf', 'style': 'condensed_sans', 'variation': 'Condensed', 'weight': 400},
    {'name': 'bahnschrift_semicond', 'file': 'bahnschrift.ttf', 'style': 'condensed_sans', 'variation': 'SemiCondensed', 'weight': 400},
    {'name': 'bahnschrift_semibold_cond', 'file': 'bahnschrift.ttf', 'style': 'condensed_sans', 'variation': 'SemiBold Condensed', 'weight': 600},
    {'name': 'bahnschrift_semibold_semicond', 'file': 'bahnschrift.ttf', 'style': 'condensed_sans', 'variation': 'SemiBold SemiCondensed', 'weight': 600},
    {'name': 'arial_narrow_bold', 'file': 'ARIALNB.TTF', 'style': 'condensed_sans', 'variation': None, 'weight': 700},
    {'name': 'bahnschrift_bold_cond', 'file': 'bahnschrift.ttf', 'style': 'condensed_sans', 'variation': 'Bold Condensed', 'weight': 700},
    {'name': 'bahnschrift_bold_semicond', 'file': 'bahnschrift.ttf', 'style': 'condensed_sans', 'variation': 'Bold SemiCondensed', 'weight': 700},

    # Standard sans-serif
    {'name': 'segoe_ui_light', 'file': 'segoeuil.ttf', 'style': 'standard_sans', 'variation': None, 'weight': 300},
    {'name': 'calibri_light', 'file': 'calibril.ttf', 'style': 'standard_sans', 'variation': None, 'weight': 300},
    {'name': 'segoe_ui_semilight', 'file': 'segoeuisl.ttf', 'style': 'standard_sans', 'variation': None, 'weight': 350},
    {'name': 'segoe_ui', 'file': 'segoeui.ttf', 'style': 'standard_sans', 'variation': None, 'weight': 400},
    {'name': 'calibri', 'file': 'calibri.ttf', 'style': 'standard_sans', 'variation': None, 'weight': 400},
    {'name': 'arial', 'file': 'arial.ttf', 'style': 'standard_sans', 'variation': None, 'weight': 400},
    {'name': 'tahoma', 'file': 'tahoma.ttf', 'style': 'standard_sans', 'variation': None, 'weight': 400},
    {'name': 'verdana', 'file': 'verdana.ttf', 'style': 'standard_sans', 'variation': None, 'weight': 400},
    {'name': 'calibri_bold', 'file': 'calibrib.ttf', 'style': 'standard_sans', 'variation': None, 'weight': 700},
    {'name': 'segoe_ui_bold', 'file': 'segoeuib.ttf', 'style': 'standard_sans', 'variation': None, 'weight': 700},
    {'name': 'arial_bold', 'file': 'arialbd.ttf', 'style': 'standard_sans', 'variation': None, 'weight': 700},
    {'name': 'tahoma_bold', 'file': 'tahomabd.ttf', 'style': 'standard_sans', 'variation': None, 'weight': 700},
    {'name': 'verdana_bold', 'file': 'verdanab.ttf', 'style': 'standard_sans', 'variation': None, 'weight': 700},

    # Monospace / dot-matrix style
    {'name': 'courier_new', 'file': 'cour.ttf', 'style': 'monospace', 'variation': None, 'weight': 400},
    {'name': 'consolas', 'file': 'consola.ttf', 'style': 'monospace', 'variation': None, 'weight': 400},
    {'name': 'lucida_console', 'file': 'lucon.ttf', 'style': 'monospace', 'variation': None, 'weight': 400},
    {'name': 'courier_new_bold', 'file': 'courbd.ttf', 'style': 'monospace', 'variation': None, 'weight': 700},
    {'name': 'consolas_bold', 'file': 'consolab.ttf', 'style': 'monospace', 'variation': None, 'weight': 700},

    # Serif (scanned forms)
    {'name': 'times_new_roman', 'file': 'times.ttf', 'style': 'serif', 'variation': None, 'weight': 400},
    {'name': 'georgia', 'file': 'georgia.ttf', 'style': 'serif', 'variation': None, 'weight': 400},
    {'name': 'times_new_roman_bold', 'file': 'timesbd.ttf', 'style': 'serif', 'variation': None, 'weight': 700},
    {'name': 'georgia_bold', 'file': 'georgiab.ttf', 'style': 'serif', 'variation': None, 'weight': 700},
]


class AdaptiveFontPool:
    """Multi-style font pool that characterizes and selects fonts matching local stroke thickness & aspect ratio."""

    def __init__(self, font_dir=r"C:\Windows\Fonts"):
        self.font_dir = font_dir
        self.entries = []
        self.glyph_metrics_cache = {}
        self._init_pool()

    def _load_pil_font(self, spec, size):
        path = os.path.join(self.font_dir, spec['file'])
        if not os.path.exists(path):
            return None
        try:
            font = ImageFont.truetype(path, size)
            if spec.get('variation'):
                try:
                    font.set_variation_by_name(spec['variation'].encode('utf-8'))
                except Exception:
                    pass
            return font
        except Exception:
            return None

    def _init_pool(self):
        probe_chars = "O01lIS5B8Z2aceopxy"
        for spec in FONT_CANDIDATE_SPECS:
            font_64 = self._load_pil_font(spec, 64)
            if font_64 is None:
                continue
            aspects, strokes = [], []
            for ch in probe_chars:
                aspect, stroke_ratio = self._measure_font_glyph(font_64, ch)
                aspects.append(aspect)
                strokes.append(stroke_ratio)
            entry = dict(spec)
            entry['font_64'] = font_64
            entry['font_96'] = self._load_pil_font(spec, 96) or font_64
            entry['mean_aspect'] = float(np.mean(aspects))
            entry['mean_stroke_ratio'] = float(np.mean(strokes))
            self.entries.append(entry)

    @staticmethod
    def _measure_font_glyph(pil_font, ch):
        canvas = Image.new('L', (144, 144), 0)
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 20), ch, fill=255, font=pil_font)
        arr = np.array(canvas, dtype=np.uint8)
        mask = arr > 110
        ys, xs = np.where(mask)
        if len(ys) < 4:
            return 0.60, 0.11
        h = float(ys.max() - ys.min() + 1)
        w = float(xs.max() - xs.min() + 1)
        aspect = w / max(1.0, h)
        perim = float(np.sum(mask[:, 1:] != mask[:, :-1]) + np.sum(mask[1:, :] != mask[:-1, :]))
        stroke_px = (2.0 * float(mask.sum())) / max(1.0, perim)
        stroke_ratio = stroke_px / max(1.0, h)
        return aspect, stroke_ratio

    def get_glyph_metrics(self, entry, ch):
        key = (entry['name'], ch)
        if key not in self.glyph_metrics_cache:
            self.glyph_metrics_cache[key] = self._measure_font_glyph(entry['font_64'], ch)
        return self.glyph_metrics_cache[key]

    def select_font(self, target_char, target_stroke_ratio, target_aspect_ratio,
                    preferred_style=None, allowed_styles=None, target_stroke_px=1.5):
        """Selects the font & weight from the pool best matching local stroke thickness, weight, and aspect ratio."""
        if not self.entries:
            return None

        # Regular-weight threshold: when local stroke thickness/ratio exceeds threshold, select Bold variants
        is_bold_target = (target_stroke_ratio >= 0.125) or (target_stroke_px >= 2.15)
        is_light_target = (target_stroke_ratio <= 0.088) and (target_stroke_px <= 1.35)

        best_entry = None
        best_score = float('inf')

        for entry in self.entries:
            if allowed_styles and entry['style'] not in allowed_styles:
                continue
            # Avoid programmer slashed-zero fonts (Consolas) when rendering '0' as a homoglyph for 'O'
            if target_char == '0' and 'consolas' in entry['name']:
                continue

            g_aspect, g_stroke = self.get_glyph_metrics(entry, target_char)
            if target_char in ('1', 'l', 'I'):
                aspect_err = abs(entry['mean_aspect'] - target_aspect_ratio)
            else:
                aspect_err = 0.65 * abs(g_aspect - target_aspect_ratio) + 0.35 * abs(entry['mean_aspect'] - target_aspect_ratio)

            stroke_err = 0.70 * abs(g_stroke - target_stroke_ratio) + 0.30 * abs(entry['mean_stroke_ratio'] - target_stroke_ratio)

            style_penalty = 0.0
            if preferred_style and entry['style'] != preferred_style:
                style_penalty = 0.06

            # Dynamic font weight matching penalty so bold strokes pick bold fonts and regular strokes pick regular/light fonts
            weight = entry.get('weight', 400)
            weight_penalty = 0.0
            if is_bold_target and weight < 600:
                weight_penalty = 0.12
            elif not is_bold_target and weight >= 600 and target_stroke_ratio < 0.115:
                weight_penalty = 0.08
            elif is_light_target and weight > 400:
                weight_penalty = 0.06

            score = 2.8 * stroke_err + 1.0 * aspect_err + style_penalty + weight_penalty
            if score < best_score:
                best_score = score
                best_entry = entry

        return best_entry


_GLOBAL_FONT_POOL = None


def get_font_pool():
    global _GLOBAL_FONT_POOL
    if _GLOBAL_FONT_POOL is None:
        _GLOBAL_FONT_POOL = AdaptiveFontPool()
    return _GLOBAL_FONT_POOL


def estimate_char_stroke_and_aspect(char_ink_map, orig_w, orig_h):
    """
    Estimates original foreground stroke thickness (via distance transform & scanline run-length
    on the half-max core glyph mask) and character aspect ratio (width-to-height).
    """
    aspect_ratio = float(orig_w) / max(1.0, float(orig_h))
    if char_ink_map.size == 0 or orig_w < 2 or orig_h < 4:
        return 1.35, 0.10, aspect_ratio

    peak_val = float(np.max(char_ink_map))
    thresh = max(0.40, 0.52 * peak_val)
    binary = (char_ink_map >= thresh).astype(np.uint8)
    fg_count = int(binary.sum())
    if fg_count < 3:
        binary = (char_ink_map >= max(0.25, 0.35 * peak_val)).astype(np.uint8)
        fg_count = int(binary.sum())
    if fg_count < 3:
        return max(1.0, min(orig_w, orig_h) * 0.14), 0.10, aspect_ratio

    # 1. Distance transform on half-maximum core mask
    padded = np.pad(binary, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    fg_dist = dist[binary > 0]
    dt_stroke = 2.0 * float(np.percentile(fg_dist, 60)) if len(fg_dist) > 0 else 1.4

    # 2. Scanline contiguous run-length estimation (immune to blur halos and bowl geometry)
    runs = []
    for row in binary:
        if np.any(row):
            diffs = np.diff(np.concatenate(([0], row, [0])))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            for s, e in zip(starts, ends):
                runs.append(float(e - s))
    for col in binary.T:
        if np.any(col):
            diffs = np.diff(np.concatenate(([0], col, [0])))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            for s, e in zip(starts, ends):
                runs.append(float(e - s))

    rl_stroke = float(np.percentile(runs, 40)) if runs else dt_stroke

    # 3. Area-to-perimeter morphological ratio (2 * Area / Perimeter)
    perim = float(np.sum(padded[:, 1:] != padded[:, :-1]) + np.sum(padded[1:, :] != padded[:-1, :]))
    ap_stroke = (2.0 * float(fg_count)) / max(1.0, perim)

    raw_stroke = 0.40 * dt_stroke + 0.35 * rl_stroke + 0.25 * ap_stroke
    max_allowed = max(2.2, min(float(orig_w) * 0.42, float(orig_h) * 0.28))
    stroke_px = float(np.clip(raw_stroke, 1.0, max_allowed))
    stroke_ratio = float(np.clip(stroke_px / max(1.0, float(orig_h)), 0.055, 0.28))
    return stroke_px, stroke_ratio, aspect_ratio


def estimate_word_geometry_profile(word_ink_map, boxes, original_text):
    """
    Aggregates stroke thickness, aspect ratio, dot-matrix texture score, and typographical
    style cues (condensed_sans, standard_sans, monospace, serif) across the word.
    """
    stroke_pxs = []
    stroke_ratios = []
    aspects = []
    slot_widths = []
    serif_scores = []
    dot_matrix_scores = []

    stem_chars = set("Il1HTNMBPRDFELKirnmhpqdbtf")

    for idx, (ix1, ix2, iy1, iy2, sx1, sx2) in enumerate(boxes):
        cw, ch = ix2 - ix1, iy2 - iy1
        if cw < 2 or ch < 5:
            continue
        sub_ink = word_ink_map[iy1:iy2, ix1:ix2]
        s_px, s_rat, asp = estimate_char_stroke_and_aspect(sub_ink, cw, ch)
        stroke_pxs.append(s_px)
        stroke_ratios.append(s_rat)
        ch_str = original_text[idx] if idx < len(original_text) else ''
        if ch_str not in ('1', 'l', 'I', 'i', '.', ',', ':', ';', '-'):
            aspects.append(asp)
            slot_widths.append(float(cw))

        # Check high-frequency dot-matrix modulation along active ink pixels
        if ch >= 8 and cw >= 5:
            active = sub_ink > 0.28
            if np.sum(active) >= 8:
                lap_ink = np.abs(cv2.Laplacian(sub_ink, cv2.CV_32F))
                dot_energy = float(np.mean(lap_ink[active]))
                dot_matrix_scores.append(dot_energy)

        # Check top/bottom horizontal terminal flare ONLY on vertical-stem letters (not round letters c,e,o,s)
        if ch_str in stem_chars and ch >= 9 and cw >= 5:
            top_w = float(np.sum(sub_ink[:max(2, ch // 6), :] > 0.40)) / max(1, ch // 6)
            bot_w = float(np.sum(sub_ink[-max(2, ch // 6):, :] > 0.40)) / max(1, ch // 6)
            mid_w = float(np.sum(sub_ink[ch // 3:2 * ch // 3, :] > 0.40)) / max(1, (2 * ch // 3 - ch // 3))
            if mid_w > 0.6:
                serif_scores.append((0.5 * (top_w + bot_w)) / mid_w)

    med_stroke_px = float(np.median(stroke_pxs)) if stroke_pxs else 1.4
    med_stroke_ratio = float(np.median(stroke_ratios)) if stroke_ratios else 0.10
    med_aspect = float(np.median(aspects)) if aspects else 0.58
    is_dot_matrix = bool(dot_matrix_scores and float(np.median(dot_matrix_scores)) > 0.55 and med_stroke_px < 2.5)

    # Style classification heuristic
    if len(slot_widths) >= 4 and (np.std(slot_widths) / max(1.0, np.mean(slot_widths))) < 0.10 and med_aspect > 0.64:
        preferred_style = 'monospace'
    elif len(serif_scores) >= 2 and float(np.median(serif_scores)) > 1.52 and med_aspect > 0.64:
        preferred_style = 'serif'
    elif med_aspect < 0.56:
        preferred_style = 'condensed_sans'
    else:
        preferred_style = 'standard_sans'

    return {
        'median_stroke_px': med_stroke_px,
        'median_stroke_ratio': med_stroke_ratio,
        'median_aspect': med_aspect,
        'preferred_style': preferred_style,
        'is_dot_matrix': is_dot_matrix,
    }


# ===========================================================================
# Existing Background Erasure / Paper Surface & Segmentation Logic (Preserved)
# ===========================================================================
def estimate_paper_surface(cv_img):
    """
    Estimates an ink-free, unbiased 2D paper color surface across the receipt using normalized convolution.
    Normalizes high-resolution images (>1024px) to a canonical scale during envelope dilation so thick
    ink strokes on 4K photos are completely bridged rather than misclassified as paper.
    """
    orig_h, orig_w = cv_img.shape[:2]
    max_dim = max(orig_h, orig_w)
    if max_dim > 1024:
        scale_f = 1024.0 / float(max_dim)
        work_w = max(16, int(round(orig_w * scale_f)))
        work_h = max(16, int(round(orig_h * scale_f)))
        work_img = cv2.resize(cv_img, (work_w, work_h), interpolation=cv2.INTER_AREA)
    else:
        work_img = cv_img

    img_f = work_img.astype(np.float32)
    gray_u8 = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32)

    # Local upper envelope to detect dark ink strokes
    med_gray = cv2.medianBlur(gray_u8, 3)
    dil_gray = cv2.dilate(med_gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).astype(np.float32)

    # Pure paper mask: 1.0 on paper pixels, 0.0 on ink pixels and their halos
    is_paper = ((dil_gray - gray) < 10.0).astype(np.float32)
    is_paper = cv2.erode(is_paper, np.ones((3, 3), np.float32), iterations=1)
    weights = np.maximum(is_paper, 0.02)
    weights_3d = weights[..., np.newaxis]

    # Normalized Gaussian convolution interpolates pure paper color smoothly under ink strokes
    num = cv2.GaussianBlur(img_f * weights_3d, (21, 21), 5.0)
    den = cv2.GaussianBlur(weights, (21, 21), 5.0)
    if den.ndim == 2:
        den = den[..., np.newaxis]
    paper_surface = num / np.maximum(den, 1e-4)

    if max_dim > 1024:
        paper_surface = cv2.resize(paper_surface, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    paper_gray = cv2.cvtColor(np.clip(paper_surface, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

    return paper_surface, paper_gray


def get_char_boxes_for_word(word_gray, word_paper_gray, n_chars, word_text=None):
    """Finds character bounding boxes inside a word using local ink depth relative to paper surface."""
    h, w = word_gray.shape
    if h < 5 or w < n_chars * 3:
        return None, None

    diff = np.maximum(0.0, word_paper_gray - word_gray)
    peak_diff = float(np.percentile(diff, 97))
    if peak_diff < 12.0:
        return None, None

    ink_map = np.clip((diff - 5.0) / max(8.0, peak_diff - 5.0), 0.0, 1.0)
    col_ink = np.sum(ink_map, axis=0)
    if w >= 5:
        col_ink_smooth = np.convolve(col_ink, np.array([0.25, 0.50, 0.25], dtype=np.float32), mode='same')
    else:
        col_ink_smooth = col_ink

    # Proportional-aware valley-snapped segmentation so narrow characters ('i', 'l', '1', ',')
    # do not shift cut positions into neighboring letters
    if word_text and len(word_text) == n_chars:
        narrow_set = set("il1I!|.,:;'`-")
        wide_set = set("mwMW0OB8GQD")
        weights = [0.56 if c in narrow_set else (1.16 if c in wide_set else 1.0) for c in word_text]
    else:
        weights = [1.0] * n_chars

    total_wt = float(sum(weights))
    cum_wt = np.cumsum([0.0] + weights)
    avg_step = w / float(n_chars)

    cuts = [0]
    for i in range(1, n_chars):
        nominal = int(round(w * (cum_wt[i] / total_wt)))
        search_r = max(2, int(avg_step * 0.42))
        left = max(cuts[-1] + 2, nominal - search_r)
        right = min(w - (n_chars - i) * 2, nominal + search_r)
        if left < right:
            window = col_ink_smooth[left:right + 1]
            # Slight distance regularization toward nominal when valley is flat
            dist_pen = 0.03 * float(np.max(col_ink_smooth) + 1.0) * (np.abs(np.arange(left, right + 1) - nominal) / max(1.0, float(search_r)))
            valley = left + int(np.argmin(window + dist_pen))
            cuts.append(valley)
        else:
            cuts.append(nominal)
    cuts.append(w)

    boxes = []
    for i in range(n_chars):
        sx1, sx2 = cuts[i], cuts[i + 1]
        if sx2 - sx1 < 2:
            boxes.append((sx1, sx2, 0, h, sx1, sx2))
            continue
        sub = ink_map[:, sx1:sx2] > 0.25
        ys, xs = np.where(sub)
        if len(ys) > 0:
            ix1, ix2 = sx1 + int(xs.min()), sx1 + int(xs.max()) + 1
            iy1, iy2 = int(ys.min()), int(ys.max()) + 1
            boxes.append((ix1, ix2, iy1, iy2, sx1, sx2))
        else:
            boxes.append((sx1, sx2, 0, h, sx1, sx2))

    return boxes, ink_map


# ===========================================================================
# 2. Photometric & Capture-Degradation Matching (Ink Color, Blur, Grain)
# ===========================================================================
def check_capture_quality_gate(word_gray, word_paper_gray, ink_map, box):
    """
    Returns True if the character box sits on a clean region suitable for perturbation;
    Returns False if the character box sits on extreme shadows, heavy paper creases, or
    regions where local background/ink estimation has high variance.
    """
    ix1, ix2, iy1, iy2, sx1, sx2 = box
    if ix2 - ix1 < 2 or iy2 - iy1 < 6:
        return False

    char_gray = word_gray[iy1:iy2, ix1:ix2]
    char_paper = word_paper_gray[iy1:iy2, ix1:ix2]
    char_ink = ink_map[iy1:iy2, ix1:ix2]

    # 1. Extreme shadow gate: skip very dark illumination zones
    med_paper = float(np.median(char_paper))
    if med_paper < 75.0:
        return False

    # 2. Heavy paper crease / shadow edge gate: high variance in local paper surface or background
    paper_std = float(np.std(char_paper))
    paper_span = float(np.percentile(char_paper, 95) - np.percentile(char_paper, 5))
    if paper_std > 16.0 or paper_span > 36.0:
        return False

    bg_pixels = char_gray[char_ink < 0.20]
    if len(bg_pixels) >= 4 and float(np.std(bg_pixels)) > 22.0:
        return False

    # 3. Foreground ink contrast and stability gate
    fg_pixels = char_gray[char_ink >= 0.40]
    if len(fg_pixels) < 3:
        return False
    med_ink = float(np.median(fg_pixels))
    contrast = med_paper - med_ink
    if contrast < 16.0:
        return False
    ink_std = float(np.std(fg_pixels))
    if ink_std > max(38.0, 0.85 * contrast):
        return False

    return True


def _extract_core_ink_pixels(sub_bgr, sub_ink, sub_diff):
    """
    Extracts core foreground ink pixels using morphological erosion (when mask is large enough)
    and luminance percentile filtering (darkest 15-20% of pixels within the character mask)
    so light anti-aliased edge pixels never dilute dark ink estimates.
    """
    mask = (sub_ink >= max(0.30, float(np.percentile(sub_ink, 55)))).astype(np.uint8)
    if int(mask.sum()) >= 12 and sub_ink.shape[0] >= 5 and sub_ink.shape[1] >= 4:
        eroded = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1)
        if int(eroded.sum()) >= 3:
            mask = eroded
    bool_mask = mask > 0
    if not np.any(bool_mask):
        return None, None
    cand_bgr = sub_bgr[bool_mask]
    cand_diff = sub_diff[bool_mask]
    luma = 0.114 * cand_bgr[:, 0] + 0.587 * cand_bgr[:, 1] + 0.299 * cand_bgr[:, 2]
    # Keep the darkest 15-20% of pixels inside the mask
    cutoff = float(np.percentile(luma, 18))
    darkest_sel = luma <= cutoff
    if not np.any(darkest_sel):
        darkest_sel = np.ones(len(luma), dtype=bool)
    return cand_bgr[darkest_sel], cand_diff[darkest_sel]


def sample_local_ink_color(cv_img, full_gray, paper_gray, wx1, wy1, wx2, wy2,
                           boxes, ink_map, target_idx, perturbed_indices, line_words_context=None):
    """
    Samples foreground ink BGR color from the darkest 15-20% core pixels of surrounding
    unperturbed characters in the same word/line. Never returns pure black (#000000).
    """
    word_bgr = cv_img[wy1:wy2, wx1:wx2].astype(np.float32)
    word_diff = np.maximum(0.0, paper_gray[wy1:wy2, wx1:wx2] - full_gray[wy1:wy2, wx1:wx2])

    fg_samples = []
    fg_diffs = []

    # Collect core foreground ink pixels from unperturbed characters in the same word
    for idx, (ix1, ix2, iy1, iy2, _, _) in enumerate(boxes):
        if idx in perturbed_indices or idx == target_idx:
            continue
        if ix2 - ix1 < 2 or iy2 - iy1 < 4:
            continue
        sub_ink = ink_map[iy1:iy2, ix1:ix2]
        sub_bgr = word_bgr[iy1:iy2, ix1:ix2]
        sub_diff = word_diff[iy1:iy2, ix1:ix2]
        c_bgr, c_diff = _extract_core_ink_pixels(sub_bgr, sub_ink, sub_diff)
        if c_bgr is not None and len(c_bgr) > 0:
            fg_samples.append(c_bgr)
            fg_diffs.append(c_diff)

    # Fallback to neighboring words in the same line if all other chars in this word were unavailable
    if not fg_samples and line_words_context is not None:
        for l_wx1, l_wy1, l_wx2, l_wy2 in line_words_context:
            if l_wx1 == wx1 and l_wy1 == wy1:
                continue
            sub_bgr = cv_img[l_wy1:l_wy2, l_wx1:l_wx2].astype(np.float32)
            sub_diff = np.maximum(0.0, paper_gray[l_wy1:l_wy2, l_wx1:l_wx2] - full_gray[l_wy1:l_wy2, l_wx1:l_wx2])
            if sub_diff.size > 0:
                p75 = float(np.percentile(sub_diff, 75))
                if p75 > 10.0:
                    mask = sub_diff >= p75
                    cand_bgr = sub_bgr[mask]
                    cand_diff = sub_diff[mask]
                    luma = 0.114 * cand_bgr[:, 0] + 0.587 * cand_bgr[:, 1] + 0.299 * cand_bgr[:, 2]
                    dark_sel = luma <= float(np.percentile(luma, 18))
                    fg_samples.append(cand_bgr[dark_sel] if np.any(dark_sel) else cand_bgr)
                    fg_diffs.append(cand_diff[dark_sel] if np.any(dark_sel) else cand_diff)

    # Final fallback to target character's own ink core if single-word line
    if not fg_samples:
        ix1, ix2, iy1, iy2, _, _ = boxes[target_idx]
        sub_ink = ink_map[iy1:iy2, ix1:ix2]
        sub_bgr = word_bgr[iy1:iy2, ix1:ix2]
        sub_diff = word_diff[iy1:iy2, ix1:ix2]
        c_bgr, c_diff = _extract_core_ink_pixels(sub_bgr, sub_ink, sub_diff)
        if c_bgr is not None and len(c_bgr) > 0:
            fg_samples.append(c_bgr)
            fg_diffs.append(c_diff)

    if not fg_samples:
        # Safe non-black fallback derived from local paper darkened by 72%
        med_paper_bgr = np.median(word_bgr.reshape(-1, 3), axis=0)
        safe_ink = np.clip(med_paper_bgr * 0.28, 14.0, 210.0)
        return {
            'median_bgr': safe_ink,
            'core_bgr': np.clip(safe_ink * 0.88, 12.0, 205.0),
            'edge_bgr': np.clip(safe_ink * 1.10, 18.0, 225.0),
            'peak_alpha': 0.92,
        }

    all_fg = np.concatenate(fg_samples, axis=0)
    all_diff = np.concatenate(fg_diffs, axis=0)

    # Further filter combined samples to the darkest 20% by luminance so dark ink cores are never diluted
    all_luma = 0.114 * all_fg[:, 0] + 0.587 * all_fg[:, 1] + 0.299 * all_fg[:, 2]
    core_sel = all_luma <= float(np.percentile(all_luma, 50))
    dark_core_fg = all_fg[core_sel] if np.any(core_sel) else all_fg

    # Strictly forbid pure black (#000000): clamp minimum channel intensity to >= 10.0
    median_bgr = np.clip(np.median(dark_core_fg, axis=0), 10.0, 225.0)
    core_bgr = np.clip(np.percentile(dark_core_fg, 20, axis=0), 10.0, 220.0)
    edge_bgr = np.clip(np.percentile(all_fg, 75, axis=0), 14.0, 235.0)

    local_paper_luma = float(np.median(paper_gray[wy1:wy2, wx1:wx2]))
    med_ink_luma = float(0.114 * median_bgr[0] + 0.587 * median_bgr[1] + 0.299 * median_bgr[2])
    expected_drop = max(12.0, local_paper_luma - med_ink_luma)
    observed_p90_drop = float(np.percentile(all_diff, 90))
    peak_alpha = float(np.clip(observed_p90_drop / max(12.0, expected_drop * 0.98), 0.86, 0.98))

    return {
        'median_bgr': median_bgr.astype(np.float32),
        'core_bgr': core_bgr.astype(np.float32),
        'edge_bgr': edge_bgr.astype(np.float32),
        'peak_alpha': peak_alpha,
    }


def estimate_local_blur_sigma(word_gray, word_paper_gray, ink_map, boxes, perturbed_indices):
    """
    Measures the edge sharpness of neighboring unperturbed characters using contrast-normalized
    Sobel gradient magnitude and Laplacian variance, returning a proportional Gaussian blur
    sigma in [0.4, 1.5].
    """
    gx = cv2.Sobel(word_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(word_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy) / 4.0  # Normalize 3x3 Sobel scale factor
    lap = np.abs(cv2.Laplacian(word_gray, cv2.CV_32F))

    unperturbed_mask = np.zeros_like(word_gray, dtype=bool)
    for idx, (ix1, ix2, iy1, iy2, _, _) in enumerate(boxes):
        if idx not in perturbed_indices and ix2 > ix1 and iy2 > iy1:
            unperturbed_mask[iy1:iy2, ix1:ix2] = True
    if not np.any(unperturbed_mask):
        unperturbed_mask[:, :] = True

    edge_band = unperturbed_mask & (ink_map >= 0.12) & (ink_map <= 0.88)
    if np.sum(edge_band) < 6:
        edge_band = (ink_map >= 0.10) & (ink_map <= 0.90)

    diff = np.maximum(0.0, word_paper_gray - word_gray)
    local_contrast = max(16.0, float(np.percentile(diff[unperturbed_mask], 95)) if np.any(unperturbed_mask) else float(np.percentile(diff, 95)))

    if np.sum(edge_band) >= 4:
        p85_grad = float(np.percentile(grad_mag[edge_band], 85))
        p85_lap = float(np.percentile(lap[edge_band], 85))
    else:
        p85_grad = float(np.percentile(grad_mag, 85))
        p85_lap = float(np.percentile(lap, 85))

    # Normalized sharpness index: higher means sharper edge transitions relative to contrast
    rel_grad = p85_grad / local_contrast
    rel_lap = p85_lap / local_contrast
    sharpness = 0.72 * rel_grad + 0.28 * rel_lap

    # Map sharpness inversely to Gaussian blur sigma in [0.40, 1.50]
    sigma = 0.42 / max(0.26, sharpness)
    return float(np.clip(sigma, 0.40, 1.50))


def sample_local_high_freq_grain(word_bgr, ink_map, target_shape):
    """
    Extracts high-frequency capture/paper grain from the local patch so rendered strokes
    and inpainted background match local sensor noise and paper fiber texture.
    """
    roi_h, roi_w = target_shape[:2]
    smooth = cv2.GaussianBlur(word_bgr, (5, 5), 1.0)
    hf_res = word_bgr - smooth

    paper_mask = ink_map < 0.20
    if np.sum(paper_mask) >= 10:
        paper_residuals = hf_res[paper_mask]
        channel_std = np.clip(np.std(paper_residuals, axis=0), 0.75, 6.5)
    else:
        channel_std = np.array([1.6, 1.6, 1.6], dtype=np.float32)

    wh, ww = hf_res.shape[:2]
    if wh >= roi_h and ww >= roi_w:
        # Sample a randomly shifted high-frequency residual patch from the word region
        y0 = random.randint(0, max(0, wh - roi_h))
        x0 = random.randint(0, max(0, ww - roi_w))
        sampled_patch = hf_res[y0:y0 + roi_h, x0:x0 + roi_w].copy()
        # Clamp extreme ink-edge spikes in the residual patch
        limit = (2.5 * channel_std)[np.newaxis, np.newaxis, :]
        sampled_patch = np.clip(sampled_patch, -limit, limit)
    else:
        sampled_patch = np.zeros((roi_h, roi_w, 3), dtype=np.float32)

    synth_grain = np.random.normal(0.0, 1.0, (roi_h, roi_w, 3)).astype(np.float32) * channel_std[np.newaxis, np.newaxis, :]
    blended_grain = 0.60 * sampled_patch + 0.40 * synth_grain
    return blended_grain.astype(np.float32), channel_std.astype(np.float32)


def feather_and_texture_alpha(alpha_2d, blur_sigma, local_grain_bgr, peak_alpha, stroke_px=1.5):
    """
    Applies stroke-proportional anti-washout blur matching, soft boundary feathering,
    and subtle high-frequency grain modulation to the glyph alpha mask.
    """
    # Scale blur sigma proportionally to rendered stroke thickness and cap on thin strokes
    # so blurring softens edges without diluting the dark ink core into pale gray
    stroke_cap = 0.52 if stroke_px <= 1.55 else min(1.30, 0.38 * float(stroke_px))
    eff_sigma = float(np.clip(min(blur_sigma, stroke_cap), 0.38, 1.30))

    ksize = max(3, int(math.ceil(eff_sigma * 3.0)) * 2 + 1)
    blurred = cv2.GaussianBlur(alpha_2d, (ksize, ksize), eff_sigma)

    # Preserve solid ink core plateau: normalize by the 90th percentile of active stroke pixels
    # so the interior spine reaches full peak_alpha rather than washing out
    active_vals = blurred[blurred > 0.15 * max(1e-4, float(blurred.max()))]
    if len(active_vals) >= 4:
        norm_ref = max(1e-4, float(np.percentile(active_vals, 85)))
    else:
        norm_ref = max(1e-4, float(blurred.max()))
    feathered = np.clip((blurred / norm_ref) * peak_alpha, 0.0, peak_alpha)

    # Slight edge feathering only in the outer halo (< 22% of peak_alpha)
    feathered = np.where(feathered < 0.22 * peak_alpha, feathered * 0.90, feathered)

    # Modulate stroke interior with subtle high-frequency paper/ink micro-texture
    luma_grain = (0.114 * local_grain_bgr[..., 0] + 0.587 * local_grain_bgr[..., 1] + 0.299 * local_grain_bgr[..., 2])
    grain_norm = np.clip(luma_grain / 14.0, -0.08, 0.08)
    textured = feathered * (1.0 + grain_norm * (feathered > 0.15).astype(np.float32))
    textured[textured < 0.025] = 0.0
    return np.clip(textured, 0.0, 0.99)


# ===========================================================================
# Upgraded Glyph & Diacritic Rendering Stages
# ===========================================================================
def _render_vector_fallback_glyph(char, up_h, up_w, scx, scy, sw, sh, thickness):
    """Procedural supersampled fallback if no TrueType font is available on the host OS."""
    canvas = np.zeros((up_h, up_w), dtype=np.float32)
    y1, y2 = scy - sh // 2, scy + sh // 2
    x1, x2 = scx - sw // 2, scx + sw // 2
    rx = max(thickness + 2, int(round(sw * 0.42)))
    ry = max(thickness + 2, int(round(sh * 0.45)))

    if char in ('O', '0', 'o', '\u043e'):
        cv2.ellipse(canvas, (scx, scy), (rx, ry), 0, 0, 360, 1.0, thickness, cv2.LINE_AA)
    elif char in ('1', 'l', 'I'):
        top_y, bot_y = y1 + thickness // 2, y2 - thickness // 2
        cv2.line(canvas, (scx, top_y), (scx, bot_y), 1.0, thickness, cv2.LINE_AA)
    elif char in ('c', '\u0441'):
        cv2.ellipse(canvas, (scx, scy), (rx, ry), 0, 35, 325, 1.0, thickness, cv2.LINE_AA)
    elif char in ('x', '\u0445'):
        cv2.line(canvas, (x1, y1), (x2, y2), 1.0, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x2, y1), (x1, y2), 1.0, thickness, cv2.LINE_AA)
    else:
        cv2.ellipse(canvas, (scx, scy), (rx, ry), 0, 0, 360, 1.0, thickness, cv2.LINE_AA)
    return canvas


def render_adaptive_homoglyph(char, canvas_hw, target_box_local, orig_w, orig_h,
                              target_stroke_px, target_stroke_ratio, target_aspect_ratio,
                              preferred_style, allowed_styles, blur_sigma, peak_alpha,
                              local_grain_bgr, is_dot_matrix=False, orig_char=None):
    """
    Renders a font- and stroke-matched homoglyph strictly constrained inside the original
    character bounding box and baseline [ix1_l:ix2_l, iy1_l:iy2_l].
    """
    h, w = canvas_hw
    ix1_l, ix2_l, iy1_l, iy2_l = target_box_local
    scale = 8
    up_h, up_w = h * scale, w * scale

    # Apply ~0.85x horizontal width scale factor to 0 -> O substitutions so replaced zeros
    # match narrow receipt digit proportions without touching adjacent characters
    eff_aspect_ratio = target_aspect_ratio * 0.85 if (orig_char == '0' and char in ('O', '\u041e')) else target_aspect_ratio

    pool = get_font_pool()
    font_entry = pool.select_font(
        char,
        target_stroke_ratio=target_stroke_ratio,
        target_aspect_ratio=eff_aspect_ratio,
        preferred_style=preferred_style,
        allowed_styles=allowed_styles,
        target_stroke_px=target_stroke_px,
    )

    up_canvas = np.zeros((up_h, up_w), dtype=np.float32)
    font_name_used = 'vector_fallback'
    font_style_used = preferred_style or 'standard_sans'

    if font_entry is not None:
        font_name_used = font_entry['name']
        font_style_used = font_entry['style']
        pil_font = font_entry['font_96']

        # Render glyph onto a large scratch pad at high resolution
        pad_img = Image.new('L', (220, 220), 0)
        draw = ImageDraw.Draw(pad_img)
        draw.text((32, 32), char, fill=255, font=pil_font)
        pad_arr = np.array(pad_img, dtype=np.float32) / 255.0

        mask = pad_arr > 0.15
        ys, xs = np.where(mask)
        if len(ys) >= 4:
            gy1, gy2 = int(ys.min()), int(ys.max()) + 1
            gx1, gx2 = int(xs.min()), int(xs.max()) + 1
            glyph_crop = pad_arr[gy1:gy2, gx1:gx2]
            gh, gw = glyph_crop.shape
            natural_aspect = float(gw) / max(1.0, float(gh))

            # Strictly constrain rendered glyph inside original character bounding box and baseline
            box_up_h = max(8, int(round(orig_h * scale)))
            box_up_w = max(8, int(round(orig_w * scale)))

            if char in ('1', 'l', 'I'):
                # Preserve slender width for '1', 'l', 'I' while staying inside box_up_w
                natural_up_w = int(round(box_up_h * natural_aspect))
                target_up_w = int(np.clip(natural_up_w, max(6, int(round(target_stroke_px * scale * 1.10))), box_up_w))
            elif orig_char == '0' and char in ('O', '\u041e'):
                # Apply ~0.85x horizontal width scale factor for 0 -> O substitutions
                scaled_w = int(round(box_up_w * 0.85))
                target_up_w = int(np.clip(scaled_w, max(8, int(round(target_stroke_px * scale * 2.2))), box_up_w))
            else:
                # Match original character box width closely (82% orig_w + 18% natural font width, capped at box_up_w)
                natural_up_w = int(round(box_up_h * natural_aspect))
                blended_w = int(round(0.82 * box_up_w + 0.18 * natural_up_w))
                target_up_w = int(np.clip(blended_w, max(8, int(box_up_w * 0.72)), box_up_w))

            target_up_h = box_up_h
            resized_glyph = cv2.resize(glyph_crop, (target_up_w, target_up_h), interpolation=cv2.INTER_AREA)

            # Supersampled stroke-thickness matching (supports bold thickening up to target_stroke_px)
            bw = (resized_glyph > 0.45).astype(np.uint8)
            perim = float(np.sum(bw[:, 1:] != bw[:, :-1]) + np.sum(bw[1:, :] != bw[:-1, :]))
            cur_stroke_up = (2.0 * float(bw.sum())) / max(1.0, perim)
            desired_stroke_up = target_stroke_px * scale
            stroke_delta = desired_stroke_up - cur_stroke_up

            if stroke_delta >= 2.0 and cur_stroke_up < desired_stroke_up * 0.88:
                k = min(7, max(3, int(round(stroke_delta)) | 1))
                resized_glyph = cv2.dilate(resized_glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
            elif stroke_delta <= -2.0 and cur_stroke_up > 9.0:
                resized_glyph = cv2.erode(resized_glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

            # Place anchored to the original baseline (iy2_l) and centered horizontally in [ix1_l, ix2_l]
            up_x_center = int(round(0.5 * (ix1_l + ix2_l) * scale))
            up_y_bottom = int(round(iy2_l * scale))
            px1 = up_x_center - target_up_w // 2
            px2 = px1 + target_up_w
            py2 = up_y_bottom
            py1 = py2 - target_up_h

            # Clip safely to supersampled canvas and original bounding box limits
            cx1, cy1 = max(0, px1), max(0, py1)
            cx2, cy2 = min(up_w, px2), min(up_h, py2)
            if cx2 > cx1 and cy2 > cy1:
                up_canvas[cy1:cy2, cx1:cx2] = resized_glyph[cy1 - py1:cy2 - py1, cx1 - px1:cx2 - px1]

    if up_canvas.max() <= 1e-4:
        scx = int(round(0.5 * (ix1_l + ix2_l) * scale))
        scy = int(round(0.5 * (iy1_l + iy2_l) * scale))
        sh = max(16, int(round(orig_h * scale * 0.94)))
        sw = max(12, int(round(orig_w * scale * 0.92)))
        thickness = max(6, int(round(target_stroke_px * scale)))
        up_canvas = _render_vector_fallback_glyph(char, up_h, up_w, scx, scy, sw, sh, thickness)

    down = cv2.resize(up_canvas, (w, h), interpolation=cv2.INTER_AREA)

    # Apply subtle dot-matrix modulation when surrounding word exhibits dot-matrix thermal print
    if is_dot_matrix:
        yy, xx = np.mgrid[0:h, 0:w]
        dot_grid = 0.74 + 0.26 * np.cos(math.pi * xx) * np.cos(math.pi * yy)
        down = down * dot_grid.astype(np.float32)

    # Strictly constrain within the character slot and taper 1px boundary so neighbors never shift
    confinement = np.zeros((h, w), dtype=np.float32)
    bx1 = max(0, int(math.floor(ix1_l)) - 1)
    bx2 = min(w, int(math.ceil(ix2_l)) + 1)
    by1 = max(0, int(math.floor(iy1_l)) - 1)
    by2 = min(h, int(math.ceil(iy2_l)) + 1)
    confinement[by1:by2, bx1:bx2] = 1.0
    if w >= 5:
        confinement[:, 0] *= 0.25
        confinement[:, -1] *= 0.25
    down *= confinement

    alpha_out = feather_and_texture_alpha(down, blur_sigma, local_grain_bgr, peak_alpha, stroke_px=target_stroke_px)
    return alpha_out, font_name_used, font_style_used


def create_adaptive_diacritic_alpha(char_alpha, accent_type, pad_top, pad_bot,
                                    stroke_px, blur_sigma, peak_alpha, local_grain_bgr):
    """
    Renders a single-mark diacritic (acute, grave, circumflex, diaeresis, cedilla) scaled
    inward so it never collides with adjacent lines or clips outside the text line height,
    matched to local stroke thickness, blur, and grain.
    """
    ch_h, cw = char_alpha.shape
    total_h = pad_top + ch_h + pad_bot
    scale = 8
    up_h, up_w = total_h * scale, cw * scale
    up_canvas = np.zeros((up_h, up_w), dtype=np.float32)

    cx = up_w // 2
    # Scale diacritic stroke slightly finer than main stem (82% of local stroke_px) and inward horizontally
    thickness = max(4, int(round(stroke_px * scale * 0.82)))
    dx = max(6, int(round(cw * scale * 0.44)))  # Strictly <= 44% of char width so it stays well inward

    if accent_type in ('acute', 'grave', 'circumflex', 'diaeresis'):
        avail_up = max(thickness + 4, pad_top * scale)
        # Keep a clear vertical safety margin from top border (at least thickness + 3 supersampled px)
        top_margin = min(avail_up // 3, thickness + 4)
        gap_from_char = max(3, min(avail_up // 4, int(round(ch_h * scale * 0.06))))
        y_base = max(top_margin + 4, (pad_top * scale) - gap_from_char)
        max_dy = max(4, y_base - top_margin - thickness // 2)
        dy = min(max_dy, max(5, int(round(ch_h * scale * 0.15))))

        if accent_type == 'acute':
            cv2.line(up_canvas, (cx - dx // 3, y_base), (cx + dx // 2, y_base - dy), 0.95, thickness, cv2.LINE_AA)
        elif accent_type == 'grave':
            cv2.line(up_canvas, (cx - dx // 2, y_base - dy), (cx + dx // 3, y_base), 0.95, thickness, cv2.LINE_AA)
        elif accent_type == 'circumflex':
            half_dx = max(4, int(dx * 0.52))
            cv2.line(up_canvas, (cx - half_dx, y_base), (cx, y_base - dy), 0.95, thickness, cv2.LINE_AA)
            cv2.line(up_canvas, (cx, y_base - dy), (cx + half_dx, y_base), 0.95, thickness, cv2.LINE_AA)
        elif accent_type == 'diaeresis':
            dot_r = max(3, int(round(thickness * 0.58)))
            sep = max(dot_r * 2 + 3, int(round(cw * scale * 0.22)))
            y_dot = max(dot_r + 3, y_base - dy // 2)
            cv2.circle(up_canvas, (cx - sep // 2, y_dot), dot_r, 0.95, -1, cv2.LINE_AA)
            cv2.circle(up_canvas, (cx + sep // 2, y_dot), dot_r, 0.95, -1, cv2.LINE_AA)

    elif accent_type == 'cedilla':
        avail_bot = max(thickness + 4, pad_bot * scale)
        y_top = (pad_top + ch_h) * scale - 2
        dy = min(max(4, avail_bot - thickness - 3), max(5, int(round(ch_h * scale * 0.16))))
        y_bot = min(up_h - thickness // 2 - 2, y_top + dy)
        hook_w = max(4, int(dx * 0.42))
        cv2.line(up_canvas, (cx, y_top), (cx - hook_w // 3, (y_top + y_bot) // 2), 0.95, thickness, cv2.LINE_AA)
        cv2.line(up_canvas, (cx - hook_w // 3, (y_top + y_bot) // 2), (cx + hook_w // 2, y_bot), 0.95, thickness, cv2.LINE_AA)
        cv2.line(up_canvas, (cx + hook_w // 2, y_bot), (cx - hook_w // 2, y_bot), 0.95, thickness, cv2.LINE_AA)

    down = cv2.resize(up_canvas, (cw, total_h), interpolation=cv2.INTER_AREA)
    alpha_out = feather_and_texture_alpha(down, blur_sigma, local_grain_bgr, peak_alpha * 0.95, stroke_px=stroke_px)
    return alpha_out


def compose_ink_layer(erased_roi, glyph_alpha_2d, ink_profile, local_grain_bgr):
    """
    Fills the new glyph using the local median ink BGR color, core-to-edge contrast range,
    and subtle high-frequency grain so it never renders in #000000 or looks flat.
    """
    alpha_3d = glyph_alpha_2d[..., np.newaxis]
    peak = max(1e-4, float(glyph_alpha_2d.max()))
    rel_core = np.clip(glyph_alpha_2d / peak, 0.0, 1.0)[..., np.newaxis]

    # Smoothly vary ink color across the local contrast range (darker at stroke spine, softer at edges)
    median_bgr = ink_profile['median_bgr'][np.newaxis, np.newaxis, :]
    core_bgr = ink_profile['core_bgr'][np.newaxis, np.newaxis, :]
    edge_bgr = ink_profile['edge_bgr'][np.newaxis, np.newaxis, :]

    stroke_bgr = np.where(
        rel_core >= 0.55,
        median_bgr * (1.0 - 0.45 * (rel_core - 0.55) / 0.45) + core_bgr * (0.45 * (rel_core - 0.55) / 0.45),
        edge_bgr * (1.0 - rel_core / 0.55) + median_bgr * (rel_core / 0.55),
    )

    # Blend subtle local capture grain into the ink fill and enforce non-zero minimum (>= 10)
    ink_with_grain = np.clip(stroke_bgr + 0.55 * local_grain_bgr, 10.0, 245.0)
    composited = (1.0 - alpha_3d) * erased_roi + alpha_3d * ink_with_grain
    return np.clip(composited, 0.0, 255.0)


# ===========================================================================
# 4. Image Similarity Metrics (PSNR, SSIM, FSIM) & Debug Visualizations
# ===========================================================================
def compute_psnr(clean_img, pert_img):
    """Computes Peak Signal-to-Noise Ratio (dB) between clean and perturbed images."""
    diff = clean_img.astype(np.float64) - pert_img.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    if mse <= 1e-10:
        return 100.0
    return float(10.0 * math.log10((255.0 ** 2) / mse))


def compute_ssim(clean_img, pert_img):
    """Computes Gaussian-windowed Structural Similarity Index (SSIM) between clean and perturbed images."""
    x = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY).astype(np.float32) if clean_img.ndim == 3 else clean_img.astype(np.float32)
    y = cv2.cvtColor(pert_img, cv2.COLOR_BGR2GRAY).astype(np.float32) if pert_img.ndim == 3 else pert_img.astype(np.float32)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / np.maximum(den, 1e-8)
    return float(np.mean(ssim_map))


def compute_fsim(clean_img, pert_img):
    """
    Computes Feature Similarity Index (FSIM, Zhang et al. 2011) combining multi-scale local
    phase congruency / quadrature band-pass energy and gradient magnitude similarity.
    """
    x = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY).astype(np.float32) if clean_img.ndim == 3 else clean_img.astype(np.float32)
    y = cv2.cvtColor(pert_img, cv2.COLOR_BGR2GRAY).astype(np.float32) if pert_img.ndim == 3 else pert_img.astype(np.float32)

    # Downsample slightly for large images as specified in standard FSIM protocol (min_dim ~ 256)
    h, w = x.shape
    min_dim = min(h, w)
    if min_dim > 384:
        scale_f = 256.0 / float(min_dim)
        nh, nw = max(32, int(round(h * scale_f))), max(32, int(round(w * scale_f)))
        x = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA)
        y = cv2.resize(y, (nw, nh), interpolation=cv2.INTER_AREA)

    # Gradient Magnitude (GM) via Sobel/Scharr operators
    gx1 = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    gm1 = np.hypot(gx1, gy1)

    gx2 = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    gm2 = np.hypot(gx2, gy2)

    # Multi-scale Phase Congruency (PC) approximation via multi-scale quadrature DoG / Laplacian band-pass energy
    def _phase_congruency(img_f):
        e_sum = np.zeros_like(img_f)
        a_sum = np.zeros_like(img_f) + 1e-4
        for s in (1.0, 2.0, 4.0):
            b1 = cv2.GaussianBlur(img_f, (0, 0), s)
            b2 = cv2.GaussianBlur(img_f, (0, 0), s * 1.6)
            even = np.abs(b1 - b2)
            ox = cv2.Sobel(b1, cv2.CV_32F, 1, 0, ksize=3) * 0.25
            oy = cv2.Sobel(b1, cv2.CV_32F, 0, 1, ksize=3) * 0.25
            odd = np.hypot(ox, oy)
            amp = np.hypot(even, odd)
            e_sum += even
            a_sum += amp
        pc = np.clip(e_sum / a_sum, 0.0, 1.0)
        return pc

    pc1 = _phase_congruency(x)
    pc2 = _phase_congruency(y)

    t1 = 0.85
    t2 = 160.0
    s_pc = (2.0 * pc1 * pc2 + t1) / (pc1 * pc1 + pc2 * pc2 + t1)
    s_g = (2.0 * gm1 * gm2 + t2) / (gm1 * gm1 + gm2 * gm2 + t2)
    s_l = s_pc * s_g
    pc_m = np.maximum(pc1, pc2)
    fsim_val = float(np.sum(s_l * pc_m) / max(1e-6, float(np.sum(pc_m))))
    return fsim_val


def save_debug_word_crop(clean_img, pert_img, wx1, wy1, wx2, wy2, word_record, out_path):
    """
    Saves a zoomed side-by-side comparison crop (Clean | Perturbed | 5x Diff Heatmap)
    with a diagnostic metadata header for visual subtlety verification.
    """
    img_h, img_w = clean_img.shape[:2]
    pad = 8
    x1 = max(0, wx1 - pad)
    y1 = max(0, wy1 - pad)
    x2 = min(img_w, wx2 + pad)
    y2 = min(img_h, wy2 + pad)

    clean_crop = clean_img[y1:y2, x1:x2]
    pert_crop = pert_img[y1:y2, x1:x2]
    ch, cw = clean_crop.shape[:2]
    if ch < 4 or cw < 4:
        return

    zoom = 4
    zw, zh = cw * zoom, ch * zoom
    z_clean = cv2.resize(clean_crop, (zw, zh), interpolation=cv2.INTER_NEAREST)
    z_pert = cv2.resize(pert_crop, (zw, zh), interpolation=cv2.INTER_NEAREST)

    diff_abs = np.abs(clean_crop.astype(np.float32) - pert_crop.astype(np.float32))
    diff_vis = np.clip(diff_abs * 5.0, 0, 255).astype(np.uint8)
    z_diff = cv2.resize(diff_vis, (zw, zh), interpolation=cv2.INTER_NEAREST)

    sep = np.full((zh, 6, 3), 220, dtype=np.uint8)
    panel_row = np.concatenate([z_clean, sep, z_pert, sep, z_diff], axis=1)

    header_h = 44
    total_w = max(360, panel_row.shape[1])
    if panel_row.shape[1] < total_w:
        pad_right = np.full((zh, total_w - panel_row.shape[1], 3), 245, dtype=np.uint8)
        panel_row = np.concatenate([panel_row, pad_right], axis=1)

    canvas = np.full((header_h + zh, total_w, 3), 248, dtype=np.uint8)
    canvas[header_h:, :] = panel_row

    # Annotate header using PIL with a Unicode-capable TrueType font
    pil_vis = Image.fromarray(canvas[:, :, ::-1], mode='RGB')
    draw = ImageDraw.Draw(pil_vis)
    header_font = None
    for candidate_ttf in ("arial.ttf", "segoeui.ttf", "calibri.ttf"):
        ttf_path = os.path.join(r"C:\Windows\Fonts", candidate_ttf)
        if os.path.exists(ttf_path):
            try:
                header_font = ImageFont.truetype(ttf_path, 13)
                break
            except Exception:
                pass
    orig_txt = word_record.get('original_text', '')
    pert_txt = word_record.get('perturbed_text', '')
    edits = word_record.get('edits', [])
    edit_summary = ", ".join(
        f"{e['original_char']}->{e['perturbed_char']} ({e.get('font_style', 'sans')}, s={e.get('stroke_px', 0):.1f}px, blur={e.get('blur_sigma', 0):.2f})"
        for e in edits
    )
    draw.text((8, 4), f"GT: '{orig_txt}' -> Perturbed: '{pert_txt}'   [Clean | Perturbed | 5x Diff]", fill=(20, 20, 20), font=header_font)
    draw.text((8, 22), edit_summary[:105], fill=(70, 70, 120), font=header_font)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    except PermissionError:
        pass
    pil_vis.save(out_path)


# ===========================================================================
# Multi-Dataset Line-to-Word Segmentation & Format Adapters (CORD, FUNSD, SROIE)
# ===========================================================================
def split_line_into_word_quads(line_gray, line_paper_gray, lx1, ly1, lx2, ly2, line_text):
    """
    Segments a multi-word text line box (e.g., SROIE 8-coord line quad) into individual
    whitespace-delimited word bounding quads by snapping nominal typographic word+space
    boundaries to vertical ink-projection valleys and trimming empty margins.
    """
    matches = list(re.finditer(r'\S+', line_text))
    if not matches:
        return []
    if len(matches) == 1:
        tok = matches[0].group(0)
        return [{
            'quad': {'x1': int(lx1), 'y1': int(ly1), 'x2': int(lx2), 'y2': int(ly1),
                     'x3': int(lx2), 'y3': int(ly2), 'x4': int(lx1), 'y4': int(ly2)},
            'text': tok,
        }]

    tokens = [m.group(0) for m in matches]
    space_counts = [max(1, matches[i + 1].start() - matches[i].end()) for i in range(len(matches) - 1)]

    h, w = line_gray.shape
    if h < 5 or w < sum(len(t) for t in tokens) * 2:
        return [{
            'quad': {'x1': int(lx1), 'y1': int(ly1), 'x2': int(lx2), 'y2': int(ly1),
                     'x3': int(lx2), 'y3': int(ly2), 'x4': int(lx1), 'y4': int(ly2)},
            'text': line_text.strip(),
        }]

    diff = np.maximum(0.0, line_paper_gray - line_gray)
    peak_diff = float(np.percentile(diff, 97))
    if peak_diff < 10.0:
        return []

    ink_map = np.clip((diff - 5.0) / max(8.0, peak_diff - 5.0), 0.0, 1.0)
    col_ink = np.sum(ink_map, axis=0)
    col_smooth = np.convolve(col_ink, np.array([0.2, 0.6, 0.2], dtype=np.float32), mode='same') if w >= 5 else col_ink

    narrow_set = set("il1I!|.,:;'`-")
    wide_set = set("mwMW0OB8GQD")

    def _token_wt(tok):
        return sum(0.56 if c in narrow_set else (1.16 if c in wide_set else 1.0) for c in tok)

    tok_wts = [_token_wt(t) for t in tokens]
    space_wt = 0.72
    gap_wts = [sc * space_wt for sc in space_counts]
    total_wt = sum(tok_wts) + sum(gap_wts)

    cuts = [0]
    acc = 0.0
    avg_char_w = w / max(1.0, total_wt)
    for i in range(len(tokens) - 1):
        acc += tok_wts[i] + 0.5 * gap_wts[i]
        nom = int(round(w * (acc / total_wt)))
        acc += 0.5 * gap_wts[i]
        search_r = max(3, int(round(avg_char_w * max(1.5, 0.25 * (len(tokens[i]) + len(tokens[i + 1]) + space_counts[i])))))
        left = max(cuts[-1] + len(tokens[i]) * 2, nom - search_r)
        right = min(w - sum(len(t) * 2 for t in tokens[i + 1:]), nom + search_r)
        if left < right:
            window = col_smooth[left:right + 1]
            dist_pen = 0.02 * float(np.max(col_smooth) + 1.0) * (
                np.abs(np.arange(left, right + 1) - nom) / max(1.0, float(search_r))
            )
            valley = left + int(np.argmin(window + dist_pen))
            cuts.append(valley)
        else:
            cuts.append(nom)
    cuts.append(w)

    word_items = []
    for i, tok in enumerate(tokens):
        sx1, sx2 = cuts[i], cuts[i + 1]
        if sx2 - sx1 < 2:
            continue
        sub_cols = np.where(col_ink[sx1:sx2] > 0.20)[0]
        if len(sub_cols) > 0:
            wx1 = int(lx1 + sx1 + int(sub_cols.min()))
            wx2 = int(lx1 + sx1 + int(sub_cols.max()) + 1)
        else:
            wx1, wx2 = int(lx1 + sx1), int(lx1 + sx2)
        sub_rows = np.where(np.sum(ink_map[:, max(0, wx1 - lx1):min(w, wx2 - lx1)], axis=1) > 0.20)[0]
        if len(sub_rows) > 0:
            wy1 = int(ly1 + int(sub_rows.min()))
            wy2 = int(ly1 + int(sub_rows.max()) + 1)
        else:
            wy1, wy2 = int(ly1), int(ly2)
        word_items.append({
            'quad': {'x1': wx1, 'y1': wy1, 'x2': wx2, 'y2': wy1,
                     'x3': wx2, 'y3': wy2, 'x4': wx1, 'y4': wy2},
            'text': tok,
        })
    return word_items


def load_funsd_annotation(json_path):
    """Loads a FUNSD annotation JSON and normalizes word boxes into 'quad' dictionaries."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data.setdefault('dataset', 'FUNSD')
    valid_lines = []
    for item in data.get('form', []):
        words_list = []
        for w in item.get('words', []):
            txt = w.get('text', '').strip()
            box = w.get('box', [])
            if not txt or len(box) != 4:
                continue
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            w['quad'] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y1, 'x3': x2, 'y3': y2, 'x4': x1, 'y4': y2}
            words_list.append(w)
        if words_list:
            valid_lines.append({
                'category': item.get('label', 'other'),
                'group_id': item.get('id', 0),
                'form_id': item.get('id', 0),
                'words': words_list,
            })
    data['valid_line'] = valid_lines
    return data


def load_sroie_annotation(box_path, entities_path=None):
    """Loads an SROIE 8-coordinate box file and optional ground-truth entities JSON file."""
    try:
        with open(box_path, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()
    except UnicodeDecodeError:
        with open(box_path, 'r', encoding='latin-1') as f:
            raw_lines = f.readlines()

    entities = {}
    if entities_path and os.path.exists(entities_path):
        try:
            with open(entities_path, 'r', encoding='utf-8') as ef:
                entities = json.load(ef)
        except Exception:
            try:
                with open(entities_path, 'r', encoding='latin-1') as ef:
                    entities = json.load(ef)
            except Exception:
                entities = {}

    valid_lines = []
    for line_idx, raw_ln in enumerate(raw_lines):
        ln = raw_ln.strip()
        if not ln:
            continue
        parts = ln.split(',', 8)
        if len(parts) < 9:
            continue
        try:
            coords = [int(round(float(c))) for c in parts[:8]]
        except ValueError:
            continue
        line_text = parts[8].strip()
        if not line_text:
            continue
        line_quad = {
            'x1': coords[0], 'y1': coords[1],
            'x2': coords[2], 'y2': coords[3],
            'x3': coords[4], 'y3': coords[5],
            'x4': coords[6], 'y4': coords[7],
        }
        valid_lines.append({
            'line_id': line_idx,
            'category': 'receipt_line',
            'quad': line_quad,
            'text': line_text,
            'original_text': line_text,
            'words': [],  # Populated during perturb_receipt_document using paper_gray & full_gray
        })

    return {
        'dataset': 'SROIE',
        'entities': entities,
        'valid_line': valid_lines,
    }


# ===========================================================================
# Main Document Perturbation Pipeline
# ===========================================================================
def perturb_receipt_document(cv_img, receipt_data, strength='medium', rng=None):
    """
    Applies adaptive font-matched, photometrically blended glyph & diacritic perturbations
    to a single document image (CORD, FUNSD, or SROIE) and returns
    (perturbed_img, updated_receipt_data, doc_stats).
    """
    if rng is None:
        rng = random

    target_fraction = STRENGTH_LEVELS.get(strength, 0.20)
    orig_img = cv_img.copy()
    out_img = cv_img.copy()
    img_h, img_w = out_img.shape[:2]

    # 1. Compute ink-free smooth paper surface across the entire document once
    paper_surface, paper_gray = estimate_paper_surface(out_img)
    full_gray = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Auto-adapt native FUNSD ('form') if 'valid_line' was not yet populated
    if 'valid_line' not in receipt_data and 'form' in receipt_data:
        valid_lines = []
        for item in receipt_data.get('form', []):
            words_list = []
            for w in item.get('words', []):
                txt = w.get('text', '').strip()
                box = w.get('box', [])
                if not txt or len(box) != 4:
                    continue
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                if 'quad' not in w:
                    w['quad'] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y1, 'x3': x2, 'y3': y2, 'x4': x1, 'y4': y2}
                words_list.append(w)
            if words_list:
                valid_lines.append({
                    'category': item.get('label', 'other'),
                    'group_id': item.get('id', 0),
                    'words': words_list,
                })
        receipt_data['valid_line'] = valid_lines

    # First pass: segment lines/words, estimate word geometry profiles, and identify eligible characters
    word_contexts = []
    total_eligible_chars = 0

    for line_idx, line in enumerate(receipt_data.get('valid_line', [])):
        words_list = line.get('words', [])
        # If this is an SROIE-style line with a line-level 'quad' and 'text' but unpopulated 'words',
        # segment the line quad into tight word-level quads using full_gray and paper_gray
        if not words_list and line.get('quad') and line.get('text'):
            lq = line['quad']
            lx1 = max(0, min(lq['x1'], lq['x2'], lq['x3'], lq['x4']))
            lx2 = min(img_w, max(lq['x1'], lq['x2'], lq['x3'], lq['x4']))
            ly1 = max(0, min(lq['y1'], lq['y2'], lq['y3'], lq['y4']))
            ly2 = min(img_h, max(lq['y1'], lq['y2'], lq['y3'], lq['y4']))
            if lx2 > lx1 and ly2 > ly1:
                words_list = split_line_into_word_quads(
                    full_gray[ly1:ly2, lx1:lx2],
                    paper_gray[ly1:ly2, lx1:lx2],
                    lx1, ly1, lx2, ly2,
                    line['text'],
                )
                line['words'] = words_list

        line_word_bboxes = []
        for wd in words_list:
            if 'quad' not in wd and len(wd.get('box', [])) == 4:
                bx1, by1, bx2, by2 = [int(v) for v in wd['box']]
                wd['quad'] = {'x1': bx1, 'y1': by1, 'x2': bx2, 'y2': by1, 'x3': bx2, 'y3': by2, 'x4': bx1, 'y4': by2}
            q = wd.get('quad', {})
            if q:
                lx1 = max(0, min(q['x1'], q['x2'], q['x3'], q['x4']))
                lx2 = min(img_w, max(q['x1'], q['x2'], q['x3'], q['x4']))
                ly1 = max(0, min(q['y1'], q['y2'], q['y3'], q['y4']))
                ly2 = min(img_h, max(q['y1'], q['y2'], q['y3'], q['y4']))
                if lx2 > lx1 and ly2 > ly1:
                    line_word_bboxes.append((lx1, ly1, lx2, ly2))

        for word_idx, word_data in enumerate(words_list):
            original_text = word_data.get('text', '')
            n_chars = len(original_text)
            if n_chars == 0:
                continue

            quad = word_data['quad']
            wx1 = max(0, min(quad['x1'], quad['x2'], quad['x3'], quad['x4']))
            wx2 = min(img_w, max(quad['x1'], quad['x2'], quad['x3'], quad['x4']))
            wy1 = max(0, min(quad['y1'], quad['y2'], quad['y3'], quad['y4']))
            wy2 = min(img_h, max(quad['y1'], quad['y2'], quad['y3'], quad['y4']))

            if wx2 - wx1 < n_chars * 4 or wy2 - wy1 < 8:
                continue

            word_gray = full_gray[wy1:wy2, wx1:wx2]
            word_paper_gray = paper_gray[wy1:wy2, wx1:wx2]
            boxes, ink_map = get_char_boxes_for_word(word_gray, word_paper_gray, n_chars, word_text=original_text)
            if boxes is None:
                continue

            geom_profile = estimate_word_geometry_profile(ink_map, boxes, original_text)

            eligible_indices = []
            for idx, ch in enumerate(original_text):
                if ch not in HOMOGLYPH_MAP and ch not in DIACRITIC_CHAR_MAP:
                    continue
                if not check_capture_quality_gate(word_gray, word_paper_gray, ink_map, boxes[idx]):
                    continue
                eligible_indices.append(idx)

            if eligible_indices:
                total_eligible_chars += len(eligible_indices)
                word_contexts.append({
                    'line_idx': line_idx,
                    'word_idx': word_idx,
                    'category': line.get('category', ''),
                    'word_data': word_data,
                    'original_text': original_text,
                    'wx1': wx1, 'wy1': wy1, 'wx2': wx2, 'wy2': wy2,
                    'boxes': boxes,
                    'ink_map': ink_map,
                    'geom_profile': geom_profile,
                    'eligible_indices': eligible_indices,
                    'line_word_bboxes': line_word_bboxes,
                    'selected_edits': {},
                })

    # Plan character perturbations across the document to match target_fraction (--strength)
    # while strictly enforcing:
    #   - Never perturb two adjacent characters in the same word (|i - j| > 1)
    #   - Cap diacritics to at most 1 per word
    target_total_edits = int(round(target_fraction * total_eligible_chars))
    if total_eligible_chars > 0 and target_fraction > 0 and target_total_edits == 0:
        target_total_edits = 1

    all_candidates = []
    for w_ctx_idx, w_ctx in enumerate(word_contexts):
        for char_idx in w_ctx['eligible_indices']:
            all_candidates.append((w_ctx_idx, char_idx))
    rng.shuffle(all_candidates)

    planned_count = 0
    for w_ctx_idx, char_idx in all_candidates:
        if planned_count >= target_total_edits:
            break
        w_ctx = word_contexts[w_ctx_idx]
        selected = w_ctx['selected_edits']

        # Rule: Never perturb two adjacent characters in the same word
        if (char_idx - 1) in selected or (char_idx + 1) in selected:
            continue

        ch = w_ctx['original_text'][char_idx]
        has_diacritic_already = any(v['type'] == 'diacritic' for v in selected.values())
        has_digit_edit_already = any(w_ctx['original_text'][k].isdigit() for k in selected.keys())

        # Rule: Cap numeric edits to at most one substituted digit per numeric token
        if ch.isdigit() and has_digit_edit_already:
            continue

        possible_ops = []
        if ch in HOMOGLYPH_MAP:
            for repl_ch in HOMOGLYPH_MAP[ch]:
                sub_cat = 'cross_script_homoglyph' if ch in CROSS_SCRIPT_HOMOGLYPH_MAP else 'alphanumeric_homoglyph'
                possible_ops.append({
                    'type': 'homoglyph',
                    'subtype': sub_cat,
                    'perturbed_char': repl_ch,
                    'op': f"{ch}->{repl_ch}",
                })
        # Rule: Cap diacritics to at most one per word
        if ch in DIACRITIC_CHAR_MAP and not has_diacritic_already:
            for accented_ch, accent_name in DIACRITIC_CHAR_MAP[ch]:
                possible_ops.append({
                    'type': 'diacritic',
                    'subtype': 'single_mark_diacritic',
                    'perturbed_char': accented_ch,
                    'accent_type': accent_name,
                    'op': accent_name,
                })

        if not possible_ops:
            continue

        chosen_op = rng.choice(possible_ops)
        selected[char_idx] = chosen_op
        planned_count += 1

    # Second pass: Render selected perturbations with photometric & geometric matching
    perturbed_words_log = []
    roi_metrics_list = []
    total_edits_applied = 0

    for w_ctx in word_contexts:
        selected = w_ctx['selected_edits']
        if not selected:
            continue

        wx1, wy1, wx2, wy2 = w_ctx['wx1'], w_ctx['wy1'], w_ctx['wx2'], w_ctx['wy2']
        boxes = w_ctx['boxes']
        ink_map = w_ctx['ink_map']
        original_text = w_ctx['original_text']
        geom_profile = w_ctx['geom_profile']
        perturbed_indices = set(selected.keys())

        word_gray = full_gray[wy1:wy2, wx1:wx2]
        word_paper_gray = paper_gray[wy1:wy2, wx1:wx2]
        word_bgr = orig_img[wy1:wy2, wx1:wx2].astype(np.float32)

        # Estimate word-level blur sharpness from unperturbed characters
        blur_sigma = estimate_local_blur_sigma(word_gray, word_paper_gray, ink_map, boxes, perturbed_indices)

        perturbed_chars = list(original_text)
        word_edits_log = []

        for idx in sorted(selected.keys()):
            op_info = selected[idx]
            ch = original_text[idx]
            ix1, ix2, iy1, iy2, sx1, sx2 = boxes[idx]
            orig_w, orig_h = ix2 - ix1, iy2 - iy1
            if orig_w < 2 or orig_h < 6:
                continue

            abs_ix1, abs_ix2 = wx1 + ix1, wx1 + ix2
            abs_iy1, abs_iy2 = wy1 + iy1, wy1 + iy2

            # Measure character-specific stroke thickness & aspect ratio, blended with word median
            char_ink_weights = ink_map[iy1:iy2, ix1:ix2]
            c_stroke_px, c_stroke_ratio, c_aspect = estimate_char_stroke_and_aspect(char_ink_weights, orig_w, orig_h)
            target_stroke_px = 0.60 * c_stroke_px + 0.40 * geom_profile['median_stroke_px']
            target_stroke_ratio = 0.60 * c_stroke_ratio + 0.40 * geom_profile['median_stroke_ratio']
            target_aspect_ratio = c_aspect if ch not in ('1', 'l', 'I') else geom_profile['median_aspect']

            # Sample local foreground ink BGR color & contrast range from surrounding unperturbed characters
            ink_profile = sample_local_ink_color(
                orig_img, full_gray, paper_gray,
                wx1, wy1, wx2, wy2,
                boxes, ink_map, idx, perturbed_indices,
                line_words_context=w_ctx['line_word_bboxes'],
            )

            if op_info['type'] == 'diacritic':
                accent_type = op_info['accent_type']
                nom_pad_top = max(4, int(round(orig_h * 0.38)))
                nom_pad_bot = max(4, int(round(orig_h * 0.32)))

                ey1 = max(0, abs_iy1 - nom_pad_top)
                ey2 = min(img_h, abs_iy2 + nom_pad_bot)

                # Check for upper line ink collision (ignoring 2px anti-aliased halo of current char)
                # and clip pad_top so diacritic never collides with the line above
                if accent_type in ('acute', 'grave', 'circumflex', 'diaeresis') and (abs_iy1 - 2) > ey1:
                    upper_strip_diff = paper_gray[ey1:abs_iy1 - 2, abs_ix1:abs_ix2] - full_gray[ey1:abs_iy1 - 2, abs_ix1:abs_ix2]
                    upper_ink_rows = np.where(np.max(upper_strip_diff, axis=1) > 22.0)[0]
                    if len(upper_ink_rows) > 0:
                        lowest_upper_ink = int(upper_ink_rows.max()) + 1
                        ey1 = min(abs_iy1 - 2, ey1 + lowest_upper_ink + 1)

                real_pad_top = abs_iy1 - ey1
                real_pad_bot = ey2 - abs_iy2

                # If vertical headroom above character is tight, scale base character slightly inward
                # toward its baseline (abs_iy2) so diacritic fits cleanly inside the text line height
                inward_shift = 0
                if accent_type in ('acute', 'grave', 'circumflex', 'diaeresis') and real_pad_top < max(3, int(orig_h * 0.22)):
                    inward_shift = min(max(2, int(round(orig_h * 0.15))), max(1, orig_h - 5))
                    # Ghost-free erase and slightly compress base glyph downward toward baseline abs_iy2
                    char_roi_orig = out_img[abs_iy1:abs_iy2, abs_ix1:abs_ix2].astype(np.float32)
                    char_roi_paper = paper_surface[abs_iy1:abs_iy2, abs_ix1:abs_ix2]
                    char_diff = np.maximum(0.0, paper_gray[abs_iy1:abs_iy2, abs_ix1:abs_ix2] - full_gray[abs_iy1:abs_iy2, abs_ix1:abs_ix2])
                    act_cd = char_diff[char_diff > 5.0]
                    ref_cd = float(np.percentile(act_cd, 85)) if len(act_cd) > 3 else max(10.0, float(np.max(char_diff)))
                    c_erase = np.clip((char_diff - 2.0) / max(5.0, ref_cd * 0.34), 0.0, 1.0)
                    c_erase = cv2.dilate(c_erase, np.ones((3, 3), np.uint8), iterations=1)[..., np.newaxis]
                    c_grain, _ = sample_local_high_freq_grain(word_bgr, ink_map, (orig_h, orig_w))
                    erased_char_bg = (1.0 - c_erase) * char_roi_orig + c_erase * np.clip(char_roi_paper + c_grain, 0, 255)
                    new_ch_h = orig_h - inward_shift
                    compressed_char = cv2.resize(char_roi_orig, (orig_w, new_ch_h), interpolation=cv2.INTER_AREA)
                    erased_char_bg[inward_shift:, :] = compressed_char
                    out_img[abs_iy1:abs_iy2, abs_ix1:abs_ix2] = np.clip(erased_char_bg, 0, 255).astype(np.uint8)
                    real_pad_top += inward_shift
                    eff_char_weights = cv2.resize(char_ink_weights, (orig_w, new_ch_h), interpolation=cv2.INTER_AREA)
                else:
                    eff_char_weights = char_ink_weights

                total_dh = real_pad_top + eff_char_weights.shape[0] + real_pad_bot
                local_grain, _ = sample_local_high_freq_grain(word_bgr, ink_map, (total_dh, orig_w))

                accent_alpha_2d = create_adaptive_diacritic_alpha(
                    eff_char_weights, accent_type, real_pad_top, real_pad_bot,
                    stroke_px=target_stroke_px,
                    blur_sigma=blur_sigma,
                    peak_alpha=ink_profile['peak_alpha'],
                    local_grain_bgr=local_grain,
                )

                target_slice = out_img[ey1:ey2, abs_ix1:abs_ix2].astype(np.float32)
                composited_slice = compose_ink_layer(target_slice, accent_alpha_2d, ink_profile, local_grain)
                out_img[ey1:ey2, abs_ix1:abs_ix2] = composited_slice.astype(np.uint8)

                perturbed_chars[idx] = op_info['perturbed_char']
                word_edits_log.append({
                    'char_index': idx,
                    'original_char': ch,
                    'perturbed_char': op_info['perturbed_char'],
                    'type': 'diacritic',
                    'subtype': op_info['subtype'],
                    'op': op_info['op'],
                    'font_used': 'adaptive_diacritic',
                    'font_style': geom_profile['preferred_style'],
                    'stroke_px': round(float(target_stroke_px), 2),
                    'aspect_ratio': round(float(target_aspect_ratio), 3),
                    'blur_sigma': round(float(blur_sigma), 2),
                    'ink_rgb': [int(round(ink_profile['median_bgr'][2])),
                                int(round(ink_profile['median_bgr'][1])),
                                int(round(ink_profile['median_bgr'][0]))],
                })
                total_edits_applied += 1

            elif op_info['type'] == 'homoglyph':
                new_ch = op_info['perturbed_char']

                # Slot ROI padded by 2px vertically and strictly bounded by valley cuts [sx1, sx2] horizontally
                ey1 = max(0, wy1 - 2)
                ey2 = min(img_h, wy2 + 2)
                ex1 = max(0, wx1 + sx1)
                ex2 = min(img_w, wx1 + sx2)
                roi_h, roi_w = ey2 - ey1, ex2 - ex1
                if roi_w < 3 or roi_h < 6:
                    continue

                roi_orig = out_img[ey1:ey2, ex1:ex2].astype(np.float32)
                roi_paper = paper_surface[ey1:ey2, ex1:ex2]
                roi_diff = np.maximum(0.0, paper_gray[ey1:ey2, ex1:ex2] - full_gray[ey1:ey2, ex1:ex2])

                # Existing proportional ink-only background erasure logic (preserved & ghost-free)
                max_d = float(np.max(roi_diff))
                if max_d < 10.0:
                    continue
                active_d = roi_diff[roi_diff > 5.0]
                ref_d = float(np.percentile(active_d, 85)) if len(active_d) > 4 else max_d
                erase_alpha = np.clip((roi_diff - 2.5) / max(5.5, ref_d * 0.36), 0.0, 1.0)
                erase_dil = cv2.dilate(erase_alpha, np.ones((3, 3), np.uint8), iterations=1)
                erase_alpha = 0.65 * erase_dil + 0.35 * erase_alpha

                # Taper horizontal edges by 1px so adjacent characters are never touched
                h_window = np.ones((roi_h, roi_w), dtype=np.float32)
                if roi_w >= 5:
                    h_window[:, 0] = 0.35
                    h_window[:, -1] = 0.35
                erase_alpha = cv2.GaussianBlur(erase_alpha * h_window, (3, 3), 0.65)[..., np.newaxis]
                erase_alpha = np.clip(erase_alpha * 1.30, 0.0, 1.0)

                # Sample authentic high-frequency grain from the local patch
                local_grain, _ = sample_local_high_freq_grain(word_bgr, ink_map, (roi_h, roi_w))
                paper_with_grain = np.clip(roi_paper + local_grain, 0, 255)
                erased_roi = (1.0 - erase_alpha) * roi_orig + erase_alpha * paper_with_grain

                # Local character bounding box coordinates inside [ex1:ex2, ey1:ey2]
                ix1_local = max(0, abs_ix1 - ex1)
                ix2_local = min(roi_w, abs_ix2 - ex1)
                iy1_local = max(0, abs_iy1 - ey1)
                iy2_local = min(roi_h, abs_iy2 - ey1)

                # Rule 3a: alphanumeric lookalikes rendered in matching sans-serif/monospace style
                if ch in ALPHANUM_HOMOGLYPH_MAP:
                    allowed_styles = {'condensed_sans', 'standard_sans', 'monospace'}
                    pref_style = geom_profile['preferred_style'] if geom_profile['preferred_style'] in allowed_styles else 'standard_sans'
                else:
                    allowed_styles = {'condensed_sans', 'standard_sans', 'monospace', 'serif'}
                    pref_style = geom_profile['preferred_style']

                glyph_alpha_2d, font_name_used, font_style_used = render_adaptive_homoglyph(
                    new_ch,
                    canvas_hw=(roi_h, roi_w),
                    target_box_local=(ix1_local, ix2_local, iy1_local, iy2_local),
                    orig_w=orig_w,
                    orig_h=orig_h,
                    target_stroke_px=target_stroke_px,
                    target_stroke_ratio=target_stroke_ratio,
                    target_aspect_ratio=target_aspect_ratio,
                    preferred_style=pref_style,
                    allowed_styles=allowed_styles,
                    blur_sigma=blur_sigma,
                    peak_alpha=ink_profile['peak_alpha'],
                    local_grain_bgr=local_grain,
                    is_dot_matrix=geom_profile['is_dot_matrix'],
                    orig_char=ch,
                )

                final_roi = compose_ink_layer(erased_roi, glyph_alpha_2d, ink_profile, local_grain)
                out_img[ey1:ey2, ex1:ex2] = final_roi.astype(np.uint8)

                perturbed_chars[idx] = new_ch
                word_edits_log.append({
                    'char_index': idx,
                    'original_char': ch,
                    'perturbed_char': new_ch,
                    'type': 'homoglyph',
                    'subtype': op_info['subtype'],
                    'op': op_info['op'],
                    'font_used': font_name_used,
                    'font_style': font_style_used,
                    'stroke_px': round(float(target_stroke_px), 2),
                    'aspect_ratio': round(float(target_aspect_ratio), 3),
                    'blur_sigma': round(float(blur_sigma), 2),
                    'ink_rgb': [int(round(ink_profile['median_bgr'][2])),
                                int(round(ink_profile['median_bgr'][1])),
                                int(round(ink_profile['median_bgr'][0]))],
                })
                total_edits_applied += 1

        if word_edits_log:
            perturbed_text = "".join(perturbed_chars)
            wd = w_ctx['word_data']
            # Preserve original unperturbed transcription as ground-truth label in 'text' and 'original_text'
            wd['text'] = original_text
            wd['original_text'] = original_text
            wd['perturbed_text'] = perturbed_text
            wd['perturbations'] = word_edits_log

            clean_w = orig_img[wy1:wy2, wx1:wx2]
            pert_w = out_img[wy1:wy2, wx1:wx2]
            if clean_w.shape[0] >= 11 and clean_w.shape[1] >= 11:
                roi_metrics_list.append((
                    compute_psnr(clean_w, pert_w),
                    compute_ssim(clean_w, pert_w),
                    compute_fsim(clean_w, pert_w),
                ))

            perturbed_words_log.append({
                'line_index': w_ctx['line_idx'],
                'word_index': w_ctx['word_idx'],
                'category': w_ctx['category'],
                'ground_truth_text': original_text,
                'original_text': original_text,
                'perturbed_text': perturbed_text,
                'bbox': [int(wx1), int(wy1), int(wx2), int(wy2)],
                'edits': word_edits_log,
            })

    # Synchronize line-level and FUNSD form-level perturbed_text summaries while preserving ground-truth 'text'
    for line in receipt_data.get('valid_line', []):
        words_list = line.get('words', [])
        if words_list and any('perturbed_text' in w for w in words_list):
            if 'text' in line:
                line['original_text'] = line['text']
            line['perturbed_text'] = " ".join(w.get('perturbed_text', w.get('text', '')) for w in words_list)

    if 'form' in receipt_data:
        for item in receipt_data.get('form', []):
            w_list = item.get('words', [])
            if w_list and any('perturbed_text' in w for w in w_list):
                item['original_text'] = item.get('text', '')
                item['perturbed_text'] = " ".join(
                    w.get('perturbed_text', w.get('text', ''))
                    for w in w_list if w.get('text', '').strip()
                )

    doc_stats = {
        'eligible_chars': total_eligible_chars,
        'target_fraction': target_fraction,
        'num_perturbed_words': len(perturbed_words_log),
        'num_character_edits': total_edits_applied,
        'actual_fraction': round(total_edits_applied / max(1, total_eligible_chars), 4) if total_eligible_chars > 0 else 0.0,
        'perturbed_words': perturbed_words_log,
        'roi_metrics_list': roi_metrics_list,
    }
    return out_img, receipt_data, doc_stats


def _parse_max_images(val):
    """Allows integer limits or '0', '-1', 'all', 'none', 'unlimited' for no limit."""
    s = str(val).strip().lower()
    if s in ("all", "none", "unlimited", "inf", "0", "-1"):
        return None
    n = int(s)
    return None if n <= 0 else n


# ===========================================================================
# Multi-Dataset Test-Split Configuration (CORD, FUNSD, SROIE -> 3 Distinct Folders)
# ===========================================================================
DATASET_ORDER = ("cord", "funsd", "sroie")


def get_dataset_configs(workspace_dir):
    """
    Returns the canonical test-split configurations and 3 distinct output folders
    for CORD, FUNSD, and SROIE in this workspace.
    """
    return {
        "cord": {
            "name": "CORD",
            "dataset_key": "cord",
            "split": "test",
            "base_dir": os.path.join(workspace_dir, "CORD", "CORD", "test"),
            "image_dir": os.path.join(workspace_dir, "CORD", "CORD", "test", "image"),
            "annotation_dir": os.path.join(workspace_dir, "CORD", "CORD", "test", "json"),
            "entities_dir": None,
            "default_output_dir": os.path.join(workspace_dir, "perturbed_CORD"),
        },
        "funsd": {
            "name": "FUNSD",
            "dataset_key": "funsd",
            "split": "test",
            "base_dir": os.path.join(workspace_dir, "FUNSD", "testing_data"),
            "image_dir": os.path.join(workspace_dir, "FUNSD", "testing_data", "images"),
            "annotation_dir": os.path.join(workspace_dir, "FUNSD", "testing_data", "annotations"),
            "entities_dir": None,
            "default_output_dir": os.path.join(workspace_dir, "perturbed_FUNSD"),
        },
        "sroie": {
            "name": "SROIE",
            "dataset_key": "sroie",
            "split": "test",
            "base_dir": os.path.join(workspace_dir, "SROIE2019", "test"),
            "image_dir": os.path.join(workspace_dir, "SROIE2019", "test", "img"),
            "annotation_dir": os.path.join(workspace_dir, "SROIE2019", "test", "box"),
            "entities_dir": os.path.join(workspace_dir, "SROIE2019", "test", "entities"),
            "default_output_dir": os.path.join(workspace_dir, "perturbed_SROIE"),
        },
    }


def _resolve_custom_folder_to_spec(folder_arg, split_arg, workspace_dir):
    """
    Resolves a custom --folder argument (e.g. 'CORD', 'FUNSD', 'SROIE2019', 'CORD/CORD/test',
    'FUNSD/testing_data', 'SROIE2019/test', or a custom directory path) to a dataset spec dict.
    Always defaults to the test split across CORD, FUNSD, and SROIE.
    """
    configs = get_dataset_configs(workspace_dir)
    norm_arg = folder_arg.strip().replace("/", "\\").lower()

    if norm_arg in ("cord", "cord\\cord", "cord\\cord\\test", "test"):
        return dict(configs["cord"])
    if norm_arg in ("funsd", "funsd\\testing_data", "testing_data"):
        return dict(configs["funsd"])
    if norm_arg in ("sroie", "sroie2019", "sroie2019\\test", "sroie\\test"):
        return dict(configs["sroie"])

    cand = folder_arg if os.path.isabs(folder_arg) else os.path.join(workspace_dir, folder_arg)
    if os.path.isdir(os.path.join(cand, "images")) and os.path.isdir(os.path.join(cand, "annotations")):
        spec = dict(configs["funsd"])
        spec["base_dir"] = cand
        spec["image_dir"] = os.path.join(cand, "images")
        spec["annotation_dir"] = os.path.join(cand, "annotations")
        return spec
    if os.path.isdir(os.path.join(cand, "img")) and os.path.isdir(os.path.join(cand, "box")):
        spec = dict(configs["sroie"])
        spec["base_dir"] = cand
        spec["image_dir"] = os.path.join(cand, "img")
        spec["annotation_dir"] = os.path.join(cand, "box")
        spec["entities_dir"] = os.path.join(cand, "entities") if os.path.isdir(os.path.join(cand, "entities")) else None
        return spec
    if os.path.isdir(os.path.join(cand, "image")) and os.path.isdir(os.path.join(cand, "json")):
        spec = dict(configs["cord"])
        spec["base_dir"] = cand
        spec["image_dir"] = os.path.join(cand, "image")
        spec["annotation_dir"] = os.path.join(cand, "json")
        spec["split"] = split_arg or os.path.basename(os.path.normpath(cand))
        return spec

    # Check nested test directories inside cand
    if os.path.isdir(os.path.join(cand, "testing_data", "images")):
        return dict(configs["funsd"])
    if os.path.isdir(os.path.join(cand, "test", "img")) and os.path.isdir(os.path.join(cand, "test", "box")):
        return dict(configs["sroie"])
    if os.path.isdir(os.path.join(cand, "CORD", "test", "image")):
        return dict(configs["cord"])

    spec = dict(configs["cord"])
    spec["base_dir"] = cand
    spec["image_dir"] = os.path.join(cand, "image")
    spec["annotation_dir"] = os.path.join(cand, "json")
    return spec


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Adaptive non-generative glyph-level perturbation pipeline across CORD, FUNSD, and SROIE test splits."
    )
    parser.add_argument(
        "--dataset",
        "--datasets",
        "-d",
        dest="dataset",
        default="all",
        help="Dataset(s) to process: 'all' (default: runs CORD, FUNSD, and SROIE test splits into 3 distinct folders), "
             "or 'cord', 'funsd', 'sroie' (or comma-separated list such as 'cord,funsd').",
    )
    parser.add_argument(
        "--strength",
        choices=["low", "medium", "high"],
        default="medium",
        help="Fraction of eligible characters perturbed per document: low (10%%), medium (20%%), high (35%%).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to process (default: test; benchmark evaluates exclusively on test splits).",
    )
    parser.add_argument(
        "--folder",
        "--input-dir",
        "--base-dir",
        dest="base_dir",
        default=None,
        help="Optional explicit input folder override for a single dataset (e.g., CORD/CORD/test, FUNSD/testing_data, SROIE2019/test).",
    )
    parser.add_argument(
        "--output-dir",
        "--out-folder",
        dest="output_dir",
        default=None,
        help="Optional explicit output folder override when running a single dataset. "
             "By default, outputs are written to 3 distinct folders: perturbed_CORD, perturbed_FUNSD, and perturbed_SROIE.",
    )
    parser.add_argument(
        "--max-images",
        "-n",
        type=_parse_max_images,
        default=None,
        help="Maximum number of images to process per dataset (default: None / unlimited). Pass an integer (e.g., 100) to limit.",
    )
    parser.add_argument(
        "--unlimit",
        "--all",
        dest="unlimit",
        action="store_true",
        help="Process all test images in each selected dataset with no file limit (enabled by default).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: save side-by-side zoomed word crops and compute PSNR, SSIM, FSIM metrics.",
    )
    parser.add_argument(
        "--max-debug-crops-per-image",
        type=int,
        default=8,
        help="Maximum number of side-by-side zoomed word crops saved per document in debug mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible perturbations.",
    )
    args = parser.parse_args(argv)
    if args.unlimit:
        args.max_images = None
    return args


def _safe_makedirs(path):
    """Creates directory if needed, safely handling Windows sandbox ACL checks when directory already exists."""
    if not path or os.path.isdir(path):
        return True
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except PermissionError:
        return os.path.isdir(path)


def _find_document_image(image_dir, base_name):
    """Locates the image file (<base_name>.png, .jpg, .jpeg, .tif) in image_dir."""
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".tif", ".tiff"):
        p = os.path.join(image_dir, f"{base_name}{ext}")
        if os.path.exists(p):
            return p
    return None


def run_single_dataset(spec, args, pool):
    """Runs the Channel S3 glyph perturbation pipeline on a single dataset test split into its distinct output folder."""
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    dataset_key = spec["dataset_key"]
    dataset_name = spec["name"]
    split = spec.get("split", "test")
    base_dir = spec["base_dir"]
    image_dir = spec["image_dir"]
    annotation_dir = spec["annotation_dir"]
    entities_dir = spec.get("entities_dir")
    output_img_dir = spec["output_dir"]

    output_json_dir = os.path.join(output_img_dir, "json")
    debug_crop_dir = os.path.join(output_img_dir, "debug_crops")

    _safe_makedirs(output_img_dir)
    _safe_makedirs(output_json_dir)
    if dataset_key == "sroie":
        _safe_makedirs(os.path.join(output_img_dir, "box"))
        _safe_makedirs(os.path.join(output_img_dir, "entities"))

    if args.debug:
        if not _safe_makedirs(debug_crop_dir):
            debug_crop_dir = output_img_dir

    ann_ext = ".txt" if dataset_key == "sroie" else ".json"
    ann_files = sorted(f for f in os.listdir(annotation_dir) if f.endswith(ann_ext))
    max_limit = args.max_images
    limit_label = "unlimited" if max_limit is None else str(max_limit)
    total_target_display = len(ann_files) if max_limit is None else min(max_limit, len(ann_files))

    print(f"\n===========================================================================")
    print(f"[Dataset: {dataset_name}] split={split} | input={base_dir}")
    print(f"[Output]  {output_img_dir}")
    print(f"[Config]  strength={args.strength} ({STRENGTH_LEVELS[args.strength]*100:.0f}%) | "
          f"max_images={limit_label} (target={total_target_display}/{len(ann_files)}) | debug={args.debug}")
    print(f"===========================================================================")

    processed_count = 0
    manifest_records = []
    total_eligible = 0
    total_words_perturbed = 0
    total_char_edits = 0
    full_metrics_accum = []
    roi_metrics_accum = []

    for ann_filename in ann_files:
        if max_limit is not None and processed_count >= max_limit:
            break

        base_name = ann_filename[:-len(ann_ext)]
        img_path = _find_document_image(image_dir, base_name)
        if img_path is None:
            continue

        cv_img = cv2.imread(img_path)
        if cv_img is None:
            continue

        ann_path = os.path.join(annotation_dir, ann_filename)
        if dataset_key == "cord":
            with open(ann_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            doc_data.setdefault('dataset', 'CORD')
        elif dataset_key == "funsd":
            doc_data = load_funsd_annotation(ann_path)
        elif dataset_key == "sroie":
            ent_path = os.path.join(entities_dir, f"{base_name}.txt") if entities_dir else None
            doc_data = load_sroie_annotation(ann_path, ent_path)
        else:
            with open(ann_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)

        pert_img, updated_doc_data, doc_stats = perturb_receipt_document(
            cv_img, doc_data, strength=args.strength, rng=rng
        )

        if doc_stats['num_character_edits'] > 0:
            out_img_name = f"{split}_{base_name}_perturbed.png"
            out_json_name = f"{split}_{base_name}_perturbed.json"
            output_path = os.path.join(output_img_dir, out_img_name)
            output_json_path = os.path.join(output_json_dir, out_json_name)

            cv2.imwrite(output_path, pert_img)
            with open(output_json_path, 'w', encoding='utf-8') as jf:
                json.dump(updated_doc_data, jf, indent=2, ensure_ascii=False)

            # For SROIE, also write native box/ and entities/ files with ground-truth transcriptions preserved
            if dataset_key == "sroie":
                out_box_path = os.path.join(output_img_dir, "box", f"{split}_{base_name}_perturbed.txt")
                with open(out_box_path, 'w', encoding='utf-8') as bf:
                    for ln in updated_doc_data.get('valid_line', []):
                        q = ln.get('quad', {})
                        coords_str = f"{q['x1']},{q['y1']},{q['x2']},{q['y2']},{q['x3']},{q['y3']},{q['x4']},{q['y4']}"
                        bf.write(f"{coords_str},{ln.get('text', '')}\n")
                if updated_doc_data.get('entities'):
                    out_ent_path = os.path.join(output_img_dir, "entities", f"{split}_{base_name}_perturbed.txt")
                    with open(out_ent_path, 'w', encoding='utf-8') as ef:
                        json.dump(updated_doc_data['entities'], ef, indent=2, ensure_ascii=False)

            record = {
                'dataset': dataset_name,
                'document_id': base_name,
                'receipt_id': base_name,
                'image_file': out_img_name,
                'json_file': f"json/{out_json_name}",
                'strength': args.strength,
                'eligible_characters': doc_stats['eligible_chars'],
                'num_perturbed_words': doc_stats['num_perturbed_words'],
                'num_character_edits': doc_stats['num_character_edits'],
                'actual_perturbation_fraction': doc_stats['actual_fraction'],
                'perturbed_words': doc_stats['perturbed_words'],
            }

            doc_psnr = compute_psnr(cv_img, pert_img)
            doc_ssim = compute_ssim(cv_img, pert_img)
            doc_fsim = compute_fsim(cv_img, pert_img)
            full_metrics_accum.append((doc_psnr, doc_ssim, doc_fsim))

            if doc_stats['roi_metrics_list']:
                r_arr = np.array(doc_stats['roi_metrics_list'], dtype=np.float64)
                roi_psnr, roi_ssim, roi_fsim = float(r_arr[:, 0].mean()), float(r_arr[:, 1].mean()), float(r_arr[:, 2].mean())
                roi_metrics_accum.extend(doc_stats['roi_metrics_list'])
            else:
                roi_psnr, roi_ssim, roi_fsim = doc_psnr, doc_ssim, doc_fsim

            record['similarity_metrics'] = {
                'full_image': {
                    'PSNR_dB': round(doc_psnr, 3),
                    'SSIM': round(doc_ssim, 5),
                    'FSIM': round(doc_fsim, 5),
                },
                'perturbed_word_rois': {
                    'PSNR_dB': round(roi_psnr, 3),
                    'SSIM': round(roi_ssim, 5),
                    'FSIM': round(roi_fsim, 5),
                },
            }

            if args.debug:
                for w_rec in doc_stats['perturbed_words'][:args.max_debug_crops_per_image]:
                    wx1, wy1, wx2, wy2 = w_rec['bbox']
                    crop_filename = f"{split}_{base_name}_w{w_rec['line_index']:02d}_{w_rec['word_index']:02d}.png"
                    crop_path = os.path.join(debug_crop_dir, crop_filename)
                    save_debug_word_crop(cv_img, pert_img, wx1, wy1, wx2, wy2, w_rec, crop_path)

            print(
                f"[{dataset_name}] Saved ({processed_count + 1}/{total_target_display}): {output_path} | "
                f"edits={doc_stats['num_character_edits']}/{doc_stats['eligible_chars']} "
                f"({doc_stats['actual_fraction']*100:.1f}%) | "
                f"Full [PSNR={doc_psnr:.2f}dB, SSIM={doc_ssim:.4f}, FSIM={doc_fsim:.4f}] | "
                f"ROI [PSNR={roi_psnr:.2f}dB, SSIM={roi_ssim:.4f}, FSIM={roi_fsim:.4f}]"
            )

            manifest_records.append(record)
            total_eligible += doc_stats['eligible_chars']
            total_words_perturbed += doc_stats['num_perturbed_words']
            total_char_edits += doc_stats['num_character_edits']
            processed_count += 1

    manifest = {
        'dataset': dataset_name,
        'split': split,
        'strength': args.strength,
        'target_fraction': STRENGTH_LEVELS[args.strength],
        'total_images': processed_count,
        'total_eligible_characters': total_eligible,
        'total_perturbed_words': total_words_perturbed,
        'total_character_edits': total_char_edits,
        'overall_perturbation_fraction': round(total_char_edits / max(1, total_eligible), 4) if total_eligible > 0 else 0.0,
    }

    if full_metrics_accum:
        f_arr = np.array(full_metrics_accum, dtype=np.float64)
        r_arr = np.array(roi_metrics_accum, dtype=np.float64) if roi_metrics_accum else f_arr
        manifest['similarity_metrics_summary'] = {
            'full_image_mean': {
                'PSNR_dB': round(float(f_arr[:, 0].mean()), 3),
                'SSIM': round(float(f_arr[:, 1].mean()), 5),
                'FSIM': round(float(f_arr[:, 2].mean()), 5),
            },
            'perturbed_word_rois_mean': {
                'PSNR_dB': round(float(r_arr[:, 0].mean()), 3),
                'SSIM': round(float(r_arr[:, 1].mean()), 5),
                'FSIM': round(float(r_arr[:, 2].mean()), 5),
            },
        }
        print(
            f"\n[{dataset_name} Summary] Mean Full-Image Metrics: "
            f"PSNR={f_arr[:, 0].mean():.2f} dB, SSIM={f_arr[:, 1].mean():.5f}, FSIM={f_arr[:, 2].mean():.5f}"
        )
        print(
            f"[{dataset_name} Summary] Mean Perturbed-Word ROI Metrics: "
            f"PSNR={r_arr[:, 0].mean():.2f} dB, SSIM={r_arr[:, 1].mean():.5f}, FSIM={r_arr[:, 2].mean():.5f}"
        )

    manifest['records'] = manifest_records
    manifest_path = os.path.join(output_img_dir, "perturbation_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)
    print(f"[{dataset_name}] Manifest written to: {manifest_path}")
    return manifest


def run_pipeline(args):
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    configs = get_dataset_configs(workspace_dir)

    # Pre-warm font pool once across all datasets
    pool = get_font_pool()
    print(f"[FontPool] Loaded {len(pool.entries)} font/weight variants across styles: "
          f"{sorted({e['style'] for e in pool.entries})}")

    specs_to_run = []
    if args.base_dir:
        spec = _resolve_custom_folder_to_spec(args.base_dir, args.split, workspace_dir)
        if args.output_dir:
            spec["output_dir"] = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(workspace_dir, args.output_dir)
        elif args.strength != "medium":
            spec["output_dir"] = f"{spec['default_output_dir']}_{args.strength}"
        else:
            spec["output_dir"] = spec["default_output_dir"]
        specs_to_run.append(spec)
    else:
        raw_ds = [d.strip().lower() for d in re.split(r'[,\s]+', str(args.dataset)) if d.strip()]
        if not raw_ds or "all" in raw_ds:
            selected_keys = list(DATASET_ORDER)
        else:
            selected_keys = []
            for token in raw_ds:
                if token in ("cord",):
                    selected_keys.append("cord")
                elif token in ("funsd",):
                    selected_keys.append("funsd")
                elif token in ("sroie", "sroie2019"):
                    selected_keys.append("sroie")
                else:
                    raise ValueError(f"Unknown dataset '{token}'. Choose from: all, cord, funsd, sroie.")

        for ds_key in selected_keys:
            spec = dict(configs[ds_key])
            if args.output_dir and len(selected_keys) == 1:
                spec["output_dir"] = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(workspace_dir, args.output_dir)
            elif args.output_dir and len(selected_keys) > 1:
                base_out = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(workspace_dir, args.output_dir)
                spec["output_dir"] = f"{base_out}_{spec['name']}"
            elif args.strength != "medium":
                spec["output_dir"] = f"{spec['default_output_dir']}_{args.strength}"
            else:
                spec["output_dir"] = spec["default_output_dir"]
            specs_to_run.append(spec)

    results = {}
    for spec in specs_to_run:
        results[spec["name"]] = run_single_dataset(spec, args, pool)

    if len(results) > 1:
        print("\n===========================================================================")
        print("[All Datasets Complete] Summary across 3 distinct test-split output folders:")
        for ds_name, mf in results.items():
            sim = mf.get("similarity_metrics_summary", {})
            fi = sim.get("full_image_mean", {})
            roi = sim.get("perturbed_word_rois_mean", {})
            sim_str = ""
            if fi and roi:
                sim_str = (
                    f" | Full [PSNR={fi['PSNR_dB']:.2f}dB, SSIM={fi['SSIM']:.4f}, FSIM={fi['FSIM']:.4f}]"
                    f" | ROI [PSNR={roi['PSNR_dB']:.2f}dB, SSIM={roi['SSIM']:.4f}, FSIM={roi['FSIM']:.4f}]"
                )
            print(
                f"  - {ds_name:5s}: {mf['total_images']:3d} test docs | "
                f"edits={mf['total_character_edits']:5d}/{mf['total_eligible_characters']:5d} "
                f"({mf['overall_perturbation_fraction']*100:.2f}%){sim_str}"
            )
        print("===========================================================================")
    return results


if __name__ == "__main__":
    cli_args = parse_args()
    run_pipeline(cli_args)
