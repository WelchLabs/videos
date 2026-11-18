"""Patch color averaging used by the VLA token visualizations."""

import colorsys

import numpy as np


def patch_bright_average(img, exponent=2.0):
    patches = img.reshape(16, 14, 16, 14, 3)
    chroma = patches.max(axis=-1, keepdims=True) - patches.min(axis=-1, keepdims=True)
    chroma_weights = chroma**exponent
    totals = chroma_weights.sum(axis=(1, 3), keepdims=True)
    weights = np.divide(chroma_weights, totals, out=np.full_like(chroma_weights, 1 / (14 * 14)), where=totals > 0)
    return (patches * weights).sum(axis=(1, 3))


def boost_colors_hsv(colors, saturation_boost=1.0, min_saturation=0.0, min_value=0.0):
    """
    Adjust colors in HSV space for better visibility in the barcode viz.

    Parameters
    ----------
    colors : ndarray, shape (N, 3)
        RGB colors in [0, 1].
    saturation_boost : float
        Multiplier on S channel. 1.0 = no change, 1.5 = 50% more saturated.
    min_saturation : float in [0, 1]
        Floor for S channel. Prevents fully gray rows.
    min_value : float in [0, 1]
        Floor for V channel. Lifts dark/muddy colors so the hue reads clearly.

    Returns
    -------
    boosted : ndarray, shape (N, 3), dtype float32
    """
    colors = np.asarray(colors, dtype=np.float32)
    out = np.empty_like(colors)
    for i in range(len(colors)):
        h, s, v = colorsys.rgb_to_hsv(*colors[i])
        s = min(1.0, s * saturation_boost)
        s = max(s, min_saturation)
        v = max(v, min_value)
        out[i] = colorsys.hsv_to_rgb(h, s, v)
    return out
