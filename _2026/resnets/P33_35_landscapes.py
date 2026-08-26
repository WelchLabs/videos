from manimlib import *
import numpy as np
from pathlib import Path

data_dir = Path('/Volumes/hot_1/Stephencwelch Dropbox/welch_labs/resnet/hackin/P33_landscapes_v7/hires_option1')

# ---- tunables ----
MAX_HEIGHT = 1.75          # every landscape is rescaled so its tallest (clipped) point sits at this z
LOSS_CAP = 256           # matches the clip used when the textures were saved
CANVAS_EXTENT = 2.5        # every surface spans u, v in [-2.5, 2.5] on canvas regardless of true alpha/beta extent
NUM_GRIDLINES = 64
IMG_IDX = 20907

# cfg -> (dir_seed, grid, extent), straight from the option3 FAVORITES list
LANDSCAPES = {
    'plain8_first3':   (2, 512, 0.75),
    'plain8_last3':    (2, 512, 0.75),
    'plain26_first4':  (25, 512, 1.5),
    'plain74_first8':  (28, 400, 2.5),
    'resnet74_first8': (27, 400, 2.5),
    'plain74_last8':   (25, 400, 2.5),
    'resnet74_last8':  (27, 400, 2.5),
    'plain74_first4':  (37, 512, 1.5),
    'resnet74_first4':    (39, 512, 1.5),
}


def landscape_files(cfg):
    seed, grid, extent = LANDSCAPES[cfg]
    stem = f'{cfg}_img{IMG_IDX:06d}_dir{seed:04d}_grid{grid}'
    if extent != 2.5:
        stem += f'_ext{extent:g}'
    d = data_dir / cfg
    return d / f'{stem}.npy', d / f'{stem}_tex.png'


class LossLandscape:
    """Holds one normalized landscape and maps canvas (u, v) -> surface points.

    Z[i, j] = loss at alpha = lin[j], beta = lin[i]; u indexes alpha (columns), v indexes beta (rows),
    same convention as param_surface_1 in P32. Bilinear interpolation so the surface's epsilon
    shifts give real normals; at grid nodes it's identical to nearest-neighbor.
    """

    def __init__(self, cfg, max_height=MAX_HEIGHT):
        npy, tex = landscape_files(cfg)
        Z = np.clip(np.load(npy), None, LOSS_CAP)
        self.z = Z * (max_height / Z.max())
        self.n = Z.shape[0]
        self.texture = str(tex)

    def _interp(self, u, v):
        f = (self.n - 1) / (2 * CANVAS_EXTENT)
        x = np.clip((np.asarray(u, dtype=float) + CANVAS_EXTENT) * f, 0, self.n - 1)
        y = np.clip((np.asarray(v, dtype=float) + CANVAS_EXTENT) * f, 0, self.n - 1)
        x0 = np.clip(np.floor(x).astype(int), 0, self.n - 2)
        y0 = np.clip(np.floor(y).astype(int), 0, self.n - 2)
        tx, ty = x - x0, y - y0
        z = ((1 - tx) * (1 - ty) * self.z[y0, x0] + tx * (1 - ty) * self.z[y0, x0 + 1]
             + (1 - tx) * ty * self.z[y0 + 1, x0] + tx * ty * self.z[y0 + 1, x0 + 1])
        return z

    def point(self, u, v):
        return np.array([u, v, float(self._interp(u, v))])

    def points(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        return np.stack([u, v, self._interp(u, v)], axis=-1)


class P33_plain8_last3(InteractiveScene):

    cfg = 'plain8_last3'
    max_height = 2.5
    fold_view = (43, 56, 0, (np.float32(-0.38), np.float32(0.17), np.float32(0.52)), 6.61)
    gridline_view = (43, 53, 0, (np.float32(-0.4), np.float32(0.15), np.float32(0.53)), 6.01)
    final_view = (133, 58, 0, (np.float32(-0.18), np.float32(-0.28), np.float32(0.57)), 6.08)

    def construct(self):
        land = LossLandscape(self.cfg, self.max_height)
        n = land.n

        surface = ParametricSurface(
            land.point,
            u_range=[-CANVAS_EXTENT, CANVAS_EXTENT],
            v_range=[-CANVAS_EXTENT, CANVAS_EXTENT],
            resolution=(n, n),
        )
        ts = TexturedSurface(surface, land.texture)
        ts.set_shading(0.0, 0.1, 0)

        line_values = np.linspace(-CANVAS_EXTENT, CANVAS_EXTENT, NUM_GRIDLINES)
        sweep = np.linspace(-CANVAS_EXTENT, CANVAS_EXTENT, n)
        u_gridlines = VGroup()
        v_gridlines = VGroup()
        for u in line_values:
            line = VMobject()
            line.set_points_smoothly(land.points(u, sweep))
            line.set_stroke(width=1, color=WHITE, opacity=0.15)
            u_gridlines.add(line)
        for v in line_values:
            line = VMobject()
            line.set_points_smoothly(land.points(sweep, v))
            line.set_stroke(width=1, color=WHITE, opacity=0.15)
            v_gridlines.add(line)

        self.frame.reorient(*self.fold_view)
        self.wait(0)

        self.play(ShowCreation(u_gridlines),
                  ShowCreation(v_gridlines),
                  self.frame.animate.reorient(*self.gridline_view),
                  run_time=4.0)
        self.wait()

        ts.set_opacity(0.0)
        self.add(ts)
        self.add(u_gridlines, v_gridlines)
        self.play(ts.animate.set_opacity(1.0),
                  self.frame.animate.reorient(*self.final_view),
                  run_time=5.0)
        self.wait(2)

        self.wait(20)
        self.embed()

class P33_plain8_first3(InteractiveScene):

    cfg = 'plain8_first3'
    max_height = 2.5
    fold_view = (-131, 60, 0, (np.float32(0.0), np.float32(-0.07), np.float32(0.61)), 6.96)
    gridline_view = (-131, 56, 0, (np.float32(-0.03), np.float32(-0.08), np.float32(0.58)), 6.38)
    final_view = (-47, 54, 0, (np.float32(-0.08), np.float32(-0.03), np.float32(0.12)), 6.13)

    def construct(self):
        land = LossLandscape(self.cfg, self.max_height)
        n = land.n

        surface = ParametricSurface(
            land.point,
            u_range=[-CANVAS_EXTENT, CANVAS_EXTENT],
            v_range=[-CANVAS_EXTENT, CANVAS_EXTENT],
            resolution=(n, n),
        )
        ts = TexturedSurface(surface, land.texture)
        ts.set_shading(0.0, 0.1, 0)

        line_values = np.linspace(-CANVAS_EXTENT, CANVAS_EXTENT, NUM_GRIDLINES)
        sweep = np.linspace(-CANVAS_EXTENT, CANVAS_EXTENT, n)
        u_gridlines = VGroup()
        v_gridlines = VGroup()
        for u in line_values:
            line = VMobject()
            line.set_points_smoothly(land.points(u, sweep))
            line.set_stroke(width=1, color=WHITE, opacity=0.15)
            u_gridlines.add(line)
        for v in line_values:
            line = VMobject()
            line.set_points_smoothly(land.points(sweep, v))
            line.set_stroke(width=1, color=WHITE, opacity=0.15)
            v_gridlines.add(line)

        self.frame.reorient(*self.fold_view)
        self.wait(0)

        self.play(ShowCreation(u_gridlines),
                  ShowCreation(v_gridlines),
                  self.frame.animate.reorient(*self.gridline_view),
                  run_time=4.0)
        self.wait()

        ts.set_opacity(0.0)
        self.add(ts)
        self.add(u_gridlines, v_gridlines)
        self.play(ts.animate.set_opacity(1.0),
                  self.frame.animate.reorient(*self.final_view),
                  run_time=5.0)
        self.wait(2)

        self.wait(20)
        self.embed()


class P34_plain26_first3(InteractiveScene):

    cfg = 'plain26_first4'
    max_height = 2.5
    fold_view = (-44, 51, 0, (np.float32(-0.1), np.float32(-0.02), np.float32(0.56)), 7.39)
    gridline_view = (-44, 46, 0, (np.float32(-0.09), np.float32(0.02), np.float32(0.6)), 6.70)
    final_view = (46, 48, 0, (np.float32(-0.21), np.float32(0.12), np.float32(0.59)), 6.70)

    def construct(self):
        land = LossLandscape(self.cfg, self.max_height)
        n = land.n

        surface = ParametricSurface(
            land.point,
            u_range=[-CANVAS_EXTENT, CANVAS_EXTENT],
            v_range=[-CANVAS_EXTENT, CANVAS_EXTENT],
            resolution=(n, n),
        )
        ts = TexturedSurface(surface, land.texture)
        ts.set_shading(0.0, 0.1, 0)

        line_values = np.linspace(-CANVAS_EXTENT, CANVAS_EXTENT, NUM_GRIDLINES)
        sweep = np.linspace(-CANVAS_EXTENT, CANVAS_EXTENT, n)
        u_gridlines = VGroup()
        v_gridlines = VGroup()
        for u in line_values:
            line = VMobject()
            line.set_points_smoothly(land.points(u, sweep))
            line.set_stroke(width=1, color=WHITE, opacity=0.15)
            u_gridlines.add(line)
        for v in line_values:
            line = VMobject()
            line.set_points_smoothly(land.points(sweep, v))
            line.set_stroke(width=1, color=WHITE, opacity=0.15)
            v_gridlines.add(line)

        self.frame.reorient(*self.fold_view)
        self.wait(0)

        self.play(ShowCreation(u_gridlines),
                  ShowCreation(v_gridlines),
                  self.frame.animate.reorient(*self.gridline_view),
                  run_time=4.0)
        self.wait()

        ts.set_opacity(0.0)
        self.add(ts)
        self.add(u_gridlines, v_gridlines)
        self.play(ts.animate.set_opacity(1.0),
                  self.frame.animate.reorient(*self.final_view),
                  run_time=5.0)
        self.wait(2)

        self.wait(20)
        self.embed()


# class P33_plain8_first3(LandscapeScene):
#     cfg = 'plain8_first3'


# class P33_plain8_last3(LandscapeScene):
#     cfg = 'plain8_last3'


# class P33_plain26_first4(LandscapeScene):
#     cfg = 'plain26_first4'


# class P33_plain74_first8(LandscapeScene):
#     cfg = 'plain74_first8'


# class P33_resnet74_first8(LandscapeScene):
#     cfg = 'resnet74_first8'


# class P33_plain74_last8(LandscapeScene):
#     cfg = 'plain74_last8'


# class P33_resnet74_last8(LandscapeScene):
#     cfg = 'resnet74_last8'