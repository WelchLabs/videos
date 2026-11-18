"""Shared artwork and color scale for the VLA trajectory scenes."""
from welch_paths import welch_path

import os
from pathlib import Path

import manimgx as gx
from matplotlib import colormaps
import numpy as np

from welch_scene import image_at_height


ASSET_ROOT = Path(os.environ.get(
    "WELCH_VLA_ASSETS",
    welch_path('vla'),
)).expanduser()
STORYBOARD_DIR = ASSET_ROOT / "graphics" / "p50_52_to_manim"
TRAJECTORY_PATH = ASSET_ROOT / "hackin" / "p51" / "all_trajectories.npy"
HEATMAP_DIR = ASSET_ROOT / "hackin" / "p51b"
DIFFUSION_DIR = ASSET_ROOT / "hackin" / "mar_17_2"


def load_panels():
    return tuple(
        gx.SVGMobject(STORYBOARD_DIR / f"p50_52_to_manim-{number:02d}.svg")[1:end]
        for number, end in ((2, None), (3, None), (4, None), (5, -1))
    )


def _viridis_pixels(values):
    colors = np.asarray(colormaps["viridis"](values))
    return (colors * 255).astype(np.uint8)


def color_scale():
    pixels = _viridis_pixels(np.linspace(1, 0, 512))
    return image_at_height(np.repeat(pixels[:, None, :], 32, axis=1))


def load_trajectory_heatmap(row):
    trajectory = np.load(TRAJECTORY_PATH)[-1]
    heatmap = trajectory[:, :14].T
    low, high = heatmap.min(), heatmap.max()
    normalized = (heatmap - low) / (high - low) if high > low else np.zeros_like(heatmap)
    pixels = _viridis_pixels(normalized)
    pixels = pixels.repeat(20, axis=0).repeat(20, axis=1)
    rgb = _viridis_pixels(normalized[row].mean())[:3]
    row_color = "#{:02x}{:02x}{:02x}".format(*rgb)
    return image_at_height(pixels), trajectory[:, row], heatmap.shape[0], row_color
