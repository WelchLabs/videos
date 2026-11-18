"""
Synthetic "exposure bias" drift for the ball clip: clean frame in -> a frame that looks like an autoregressive video
model's output after its errors have compounded (the look of the Minecraft rollouts in the Diffusion Forcing paper:
fine grain everywhere + the picture melting into blobs).

    python p54_drift.py <folder of clean frames> <out folder> [frame numbers ...]

Each output starts from that frame's own clean version; `level` (0 = clean, 1 = chaos) sets how far it has drifted.
Five effects, all growing with level:
  warp    the picture is pushed around by a smooth random flow (the flow at one level contains the flow of the
          levels before it, so shapes keep sliding the same way: errors accumulate)
  blobs   detail is replaced by flat patches (median filter on a shrunken copy)
  mottle  blotchy light / dark patches a few patches wide, with hard edges (the "blobiness")
  colour  saturation / contrast creep up a little and the tint wanders
  grain   per-pixel noise plus a coarser 2 - 3 px speckle
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

PRESETS = {
    "mild":   dict(warp=14, blob=7,  blob_mix=0.75, mottle=0.10, grain=0.09, speckle=0.06, sat=0.15, contrast=0.10,
                   tint=0.015),
    "medium": dict(warp=24, blob=11, blob_mix=0.90, mottle=0.16, grain=0.13, speckle=0.10, sat=0.25, contrast=0.20,
                   tint=0.025),
    "strong": dict(warp=38, blob=15, blob_mix=1.00, mottle=0.24, grain=0.18, speckle=0.14, sat=0.35, contrast=0.30,
                   tint=0.04),
}
LEVELS = (0.0, 0.33, 0.66, 1.0)     # clean, then three steps down


def smooth_field(rng, shape, cells):
    """Smooth random field in [-1, 1]-ish: random values on a coarse grid, blown up with bicubic interpolation."""
    h, w = shape
    coarse = rng.standard_normal((cells, max(2, round(cells * w / h)))).astype(np.float32)
    return np.asarray(Image.fromarray(coarse).resize((w, h), Image.BICUBIC), dtype=np.float32)


def bilinear(img, x, y):
    h, w = img.shape[:2]
    x, y = np.clip(x, 0, w - 1.001), np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def drift(clean, level, p, seed=0):
    """clean: uint8 [H, W, 3]. level: 0 .. 1. p: a PRESETS entry. -> uint8 [H, W, 3]"""
    if level <= 0:
        return clean.copy()
    x = clean.astype(np.float32) / 255
    h, w = x.shape[:2]
    flow_rng = np.random.default_rng(seed)              # same flow fields at every level: the drift keeps its direction
    noise_rng = np.random.default_rng(seed + 1000 + int(level * 1000))     # grain is fresh on every frame

    # warp: a large slow flow + a smaller, tighter one that comes in later
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = smooth_field(flow_rng, (h, w), 4) * level + 0.5 * smooth_field(flow_rng, (h, w), 12) * level ** 2
    dy = smooth_field(flow_rng, (h, w), 4) * level + 0.5 * smooth_field(flow_rng, (h, w), 12) * level ** 2
    x = bilinear(x, xx + p["warp"] * dx, yy + p["warp"] * dy)

    # blobs: median filter on a half-size copy, mixed in more and more
    size = int(round(3 + (p["blob"] - 3) * level)) | 1
    small = Image.fromarray((x * 255).astype(np.uint8)).resize((w // 2, h // 2), Image.BILINEAR)
    small = small.filter(ImageFilter.MedianFilter(size)).filter(ImageFilter.ModeFilter(max(3, size // 2 | 1)))
    blob = np.asarray(small.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255
    mix = min(1.0, p["blob_mix"] * level * 1.3)
    x = x * (1 - mix) + blob * mix

    # mottle: hard-edged light / dark patches at two sizes, mostly brightness with a little colour
    for cells, amount in ((22, 1.0), (48, 0.7)):
        f = smooth_field(noise_rng, (h, w), cells)
        patch = np.sign(f) * np.minimum(np.abs(f) * 3, 1)           # nearly two-level: flat patches, soft-ish edges
        colour = np.stack([smooth_field(noise_rng, (h, w), cells) for _ in range(3)], -1) * 0.35
        x = x + p["mottle"] * level * amount * (patch[..., None] + colour)

    # colour: more saturation and contrast, and a tint that wanders across the picture
    gray = x.mean(-1, keepdims=True)
    x = gray + (x - gray) * (1 + p["sat"] * level)
    x = 0.5 + (x - 0.5) * (1 + p["contrast"] * level)
    tint = np.stack([smooth_field(flow_rng, (h, w), 3) for _ in range(3)], -1)
    x = x + p["tint"] * level * tint

    # grain: per-pixel + a coarser speckle
    mono = noise_rng.standard_normal((h, w, 1)).astype(np.float32)          # mostly brightness, a little colour
    x = x + p["grain"] * level * (0.75 * mono + 0.45 * noise_rng.standard_normal((h, w, 3)).astype(np.float32))
    coarse = noise_rng.standard_normal((h // 3, w // 3, 1)).astype(np.float32).repeat(3, -1)
    coarse = np.asarray(Image.fromarray(((coarse * 40) + 128).clip(0, 255).astype(np.uint8)).resize((w, h), Image.NEAREST),
                        dtype=np.float32)
    x = x + p["speckle"] * level * (coarse - 128) / 40
    return (x.clip(0, 1) * 255).astype(np.uint8)


def main(src, out, frames=(7, 11, 15, 19), levels=LEVELS):
    src, out = Path(src), Path(out)
    clean = [np.asarray(Image.open(src / f"frame_{f:03d}.png").convert("RGB")) for f in frames]
    sheets = []
    for name, p in PRESETS.items():
        (out / name).mkdir(parents=True, exist_ok=True)
        row = []
        for k, (img, level) in enumerate(zip(clean, levels)):
            res = drift(img, level, p, seed=k)
            Image.fromarray(res).save(out / name / f"drift_{k}_frame_{frames[k]:03d}.png")
            row.append(res)
        sheets.append(np.concatenate(row, 1))
    sheet = np.concatenate([np.concatenate(clean, 1)] + sheets, 0)
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2), Image.LANCZOS).save(out / "compare.png")
    print("rows of compare.png: clean,", ", ".join(PRESETS), "->", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], args[1], *([tuple(int(a) for a in args[2:])] if len(args) > 2 else []))
