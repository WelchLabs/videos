"""ManimGX meshes and grid lines for the CartPole policy visualizations."""

import manimgx as m
import numpy as np
from PIL import Image


def height_field(axes, function, *, u_range, v_range, resolution, opacity=1.0):
    """Sample a probability surface, with its grid available for mesh overlays."""
    n_u, n_v = resolution
    u = np.linspace(*u_range, n_u)
    v = np.linspace(*v_range, n_v)
    u_grid, v_grid = np.meshgrid(u, v, indexing='ij')

    def points(x, y):
        return np.moveaxis(axes.c2p(x, y, function(x, y)), 0, -1)

    vertices = points(u_grid, v_grid).reshape(-1, 3)
    lower = np.arange(n_u * n_v).reshape(n_u, n_v)[:-1, :-1].ravel()
    triangles = np.concatenate((
        np.column_stack((lower, lower + n_v, lower + 1)),
        np.column_stack((lower + 1, lower + n_v, lower + n_v + 1)),
    ))
    s = (u_grid - u[0]) / (u[-1] - u[0])
    t = (v[-1] - v_grid) / (v[-1] - v[0])
    uvs = np.stack((s, t), axis=-1).reshape(-1, 2)
    surface = m.MeshMobject(vertices, triangles, uvs=uvs, fill_opacity=opacity)
    surface.sample_points = points
    surface.u_range = u_range
    surface.v_range = v_range
    return surface


def texture_surface(surface, path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('RGBA'))
    surface.paint = surface.paint.but(texture=pixels)
    return surface


def color_surface_vertices(surface, colors):
    rgba = np.array([m.ManimColor(color).to_rgba() for color in colors])
    surface.paint = surface.paint.but(fill=rgba)
    return surface


def surface_wireframe(surface, *, resolution, **style):
    u, v = np.meshgrid(
        np.linspace(*surface.u_range, resolution[0]),
        np.linspace(*surface.v_range, resolution[1]),
        indexing='ij',
    )
    points = surface.sample_points(u, v) + 0.001 * m.OUT
    return m.VGroup(*(
        m.VMobject(**style).set_points_as_corners(line)
        for lines in (points, np.swapaxes(points, 0, 1))
        for line in lines
    ))
