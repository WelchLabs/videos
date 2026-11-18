"""Sampled and textured surfaces shared by Welch Labs scenes."""

import manimgx as m
import numpy as np
from PIL import Image


def _surface_options(resolution, style):
    counts = (resolution, resolution) if isinstance(resolution, int) else tuple(resolution)
    intervals = tuple(max(1, count - 1) for count in counts)
    color = style.pop("color", None)
    opacity = style.pop("opacity", None)
    shading = style.pop("shading", None)
    if color is not None:
        style.setdefault("fill_color", color)
    if opacity is not None:
        style.setdefault("fill_opacity", opacity)
    style.setdefault("stroke_width", 0)
    style.setdefault("checkerboard_colors", False)
    return intervals, shading


def parametric_surface(func, u_range=(0, 1), v_range=(0, 1), resolution=(101, 101), **style):
    """Sample a single-color parametric surface at the specified vertex counts."""
    intervals, shading = _surface_options(resolution, style)
    surface = m.Surface(func, u_range=u_range, v_range=v_range, resolution=intervals, **style)
    if shading is not None:
        set_surface_shading(surface, *shading)
    return surface


def textured_surface(surface, image_file, **style):
    """Apply an RGBA image across a sampled surface's existing u/v grid."""
    if surface.grid is None:
        raise ValueError("Texture placement requires a parametric surface grid")
    nu, nv = surface.grid
    u, v = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(1, 0, nv + 1), indexing="ij")
    with Image.open(image_file) as image:
        pixels = np.asarray(image.convert("RGBA"))
    style.setdefault("fill_opacity", surface.get_fill_opacity())
    style.setdefault("shade_in_3d", surface.paint.shade_in_3d)
    return m.MeshMobject(surface.points, surface.triangles,
                         uvs=np.column_stack((u.ravel(), v.ravel())), texture=pixels, **style)


def set_surface_shading(mobject, reflectiveness=None, gloss=None, shadow=None):
    """Enable GX lighting when the scene requests any directional shading."""
    values = [value for value in (reflectiveness, gloss, shadow) if value is not None]
    if values:
        enabled = any(value > 0 for value in values)
        # GX exposes set_shade_in_3d on vector paths, not on meshes.
        for member in mobject.get_family():
            member.paint = member.paint.but(shade_in_3d=enabled)
    return mobject


class SampledSurface(m.Surface):
    """Sample a surface defined by a subclass's uv_func method."""

    def __init__(self, resolution=(101, 101), **style):
        intervals, shading = _surface_options(resolution, style)
        super().__init__(self.uv_func, resolution=intervals, **style)
        if shading is not None:
            set_surface_shading(self, *shading)
