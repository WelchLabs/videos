from manimlib import *
import numpy as np
from pathlib import Path
 
data_dir = Path('/Volumes/hot_1/Stephencwelch Dropbox/welch_labs/resnet/hackin/P33_landscapes_v7/hires_option1')
grad_field_file = Path('/Volumes/hot_1/Stephencwelch Dropbox/welch_labs/resnet/hackin/p35_gradient_field/plain74_first4/'
                       'plain74_first4_img020907_dir0037_grid512_ext1.5_gradfield.npz')
 
# ---- landscape tunables ----
MAX_HEIGHT = 1.75          # every landscape is rescaled so its tallest (clipped) point sits at this z
LOSS_CAP = 256             # matches the clip used when the textures were saved
CANVAS_EXTENT = 2.5        # every surface spans u, v in [-2.5, 2.5] on canvas regardless of true alpha/beta extent
NUM_GRIDLINES = 64
IMG_IDX = 20907
 
# ---- gradient field tunables ----
DOWNSAMPLE = 16             # 512 / 4 = 128 arrows per side (16k arrows). 8 -> 64/side, 16 -> 32/side (the notebook preview)
DOWNSAMPLE_MODE = 'mean'   # 'mean' block-averages the gradient over each cell; 'stride' just picks every k-th sample
DESCENT = True             # arrows point downhill (-grad). False -> raw ascent gradient
LENGTH_MODE = 'unit'       # 'unit' = direction only (all arrows same length); 'log' / 'linear' = length encodes |grad|
ARROW_FILL = 0.8           # arrow length as a fraction of the arrow spacing (the longest arrow in log/linear modes)
MIN_LENGTH_FRAC = 0.15     # shortest arrow relative to the longest (log / linear modes only)
ARROW_PIVOT = 'mid'        # 'mid' centers the arrow on its grid point (quiver pivot='mid'); 'tail' starts it there
ARROW_Z = 0.01             # lift above the flattened surface
HEAD_FRAC = 0.4            # arrowhead length as a fraction of total arrow length
HEAD_WIDTH_FRAC = 0.8      # arrowhead base width relative to arrowhead length (1.0 -> roughly equilateral)
SHAFT_STROKE_WIDTH = 1.5   # try ~1 at 128/side, ~2.5 at 32/side
ARROW_OPACITY = 0.9
 
SWEEP_DIRECTION = (1, 1)   # canvas (u, v) direction the reveal travels in; (1, 0) = left to right, (1, 1) = diagonal
SWEEP_BANDS = 64           # how many slices the field is split into along the sweep direction
SWEEP_LAG = 0.08           # LaggedStart lag ratio between bands (smaller = more overlap, wider "wavefront")
SWEEP_TIME = 4.0
 
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
    'plain74_last4':   (37, 512, 1.5),
    'resnet74_first4': (39, 512, 1.5),
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
 
 
class GradientField:
    """Loads the P35 gradfield .npz, downsamples it, and maps it onto canvas (u, v).
 
    Layout in the file: Ga[i, j] = dL/dalpha, Gb[i, j] = dL/dbeta at alpha = alphas[j], beta = betas[i]
    (same as the P33 landscapes). Canvas u <-> alpha, v <-> beta, and the alpha->u map is a uniform
    scale (CANVAS_EXTENT / true extent), so the arrow directions carry over unchanged.
    Gradients in the file are raw ascent; DESCENT negates them.
    """
 
    def __init__(self, npz_path, downsample=DOWNSAMPLE, mode=DOWNSAMPLE_MODE):
        d = np.load(npz_path)
        alphas, betas, Ga, Gb = d['alphas'], d['betas'], d['Ga'], d['Gb']
        n, k = len(alphas), int(downsample)
        m = n // k
        c = m * k                      # drop any ragged remainder (512 is divisible by 4, 8, 16, so nothing is dropped)
        if mode == 'mean':
            Ga = Ga[:c, :c].reshape(m, k, m, k).mean(axis=(1, 3))
            Gb = Gb[:c, :c].reshape(m, k, m, k).mean(axis=(1, 3))
            a = alphas[:c].reshape(m, k).mean(axis=1)
            b = betas[:c].reshape(m, k).mean(axis=1)
        elif mode == 'stride':
            Ga, Gb = Ga[:c:k, :c:k], Gb[:c:k, :c:k]
            a, b = alphas[:c:k], betas[:c:k]
        else:
            raise ValueError(mode)
 
        scale = CANVAS_EXTENT / alphas.max()          # true alpha/beta -> canvas u/v
        self.m = m
        self.u, self.v = a * scale, b * scale
        self.U, self.V = np.meshgrid(self.u, self.v)  # U[i, j] = u[j], V[i, j] = v[i]
        self.Ga, self.Gb = Ga, Gb
        self.mag = np.hypot(Ga, Gb)
        self.spacing = float(self.u[1] - self.u[0])
        self.positions = np.stack([self.U.ravel(), self.V.ravel()], axis=-1)   # (m*m, 2), row-major i, j
 
    def directions(self, descent=DESCENT):
        s = -1.0 if descent else 1.0
        return s * self.Ga / (self.mag + 1e-12), s * self.Gb / (self.mag + 1e-12)
 
    def lengths(self, mode=LENGTH_MODE, fill=ARROW_FILL, min_frac=MIN_LENGTH_FRAC):
        L = fill * self.spacing
        if mode == 'unit':
            return np.full_like(self.mag, L)
        if mode == 'log':
            w = np.log10(self.mag + 1e-12)
        elif mode == 'linear':
            w = self.mag
        else:
            raise ValueError(mode)
        w = (w - w.min()) / (w.max() - w.min() + 1e-12)
        return L * (min_frac + (1 - min_frac) * w)
 
 
def line_arrow(p, direction, length, pivot=ARROW_PIVOT, head_frac=HEAD_FRAC, head_width_frac=HEAD_WIDTH_FRAC,
               z=ARROW_Z, landscape=None):
    """One arrow = VGroup(shaft, head): a stroked line plus a filled triangle sitting on its end.
 
    The shaft stops at the triangle's base so it never pokes through the tip. Cheaper than Arrow()
    and version-proof. If a LossLandscape is passed, every point is lifted to the surface height.
    """
    dx, dy = direction
    d = np.array([dx, dy, 0.0])
    perp = np.array([-dy, dx, 0.0])
    tail = p - 0.5 * length * d if pivot == 'mid' else p.copy()
    head = tail + length * d
    base = head - head_frac * length * d
    hw = 0.5 * head_width_frac * head_frac * length
    pts = np.array([tail, base, base + hw * perp, head, base - hw * perp])
    if landscape is not None:
        pts[:, 2] = landscape._interp(pts[:, 0], pts[:, 1]) + z
    else:
        pts[:, 2] = z
    shaft = VMobject()
    shaft.set_points_as_corners(pts[:2])
    tri = Polygon(*pts[2:])
    return VGroup(shaft, tri)
 
 
def build_gradient_arrows(field, color=WHITE, stroke_width=SHAFT_STROKE_WIDTH, opacity=ARROW_OPACITY,
                          landscape=None, **arrow_kwargs):
    """Flat VGroup of arrows (each a VGroup(shaft, head)), one per downsampled grid cell, row-major like field.positions."""
    dx, dy = field.directions()
    L = field.lengths()
    arrows = VGroup()
    for i in range(field.m):
        for j in range(field.m):
            p = np.array([field.U[i, j], field.V[i, j], 0.0])
            arrows.add(line_arrow(p, (dx[i, j], dy[i, j]), L[i, j], landscape=landscape, **arrow_kwargs))
    VGroup(*[a[0] for a in arrows]).set_stroke(color, width=stroke_width, opacity=opacity)
    VGroup(*[a[1] for a in arrows]).set_fill(color, opacity=opacity).set_stroke(width=0)
    return arrows
 
 
def sweep_bands(arrows, positions, direction=SWEEP_DIRECTION, n_bands=SWEEP_BANDS):
    """Split the arrows into VGroups of slices perpendicular to `direction`, ordered along it."""
    d = np.array(direction, dtype=float)
    d /= np.linalg.norm(d)
    proj = positions @ d
    edges = np.linspace(proj.min(), proj.max() + 1e-9, n_bands + 1)
    idx = np.clip(np.digitize(proj, edges) - 1, 0, n_bands - 1)
    return [VGroup(*[arrows[k] for k in np.flatnonzero(idx == b)]) for b in range(n_bands) if np.any(idx == b)]
 
def build_gridlines(land, n_lines=NUM_GRIDLINES, color=WHITE, width=1, opacity=0.15):
    line_values = np.linspace(-CANVAS_EXTENT, CANVAS_EXTENT, n_lines)
    sweep = np.linspace(-CANVAS_EXTENT, CANVAS_EXTENT, land.n)
    u_lines, v_lines = VGroup(), VGroup()
    for x in line_values:
        lu = VMobject(); lu.set_points_smoothly(land.points(x, sweep))
        lv = VMobject(); lv.set_points_smoothly(land.points(sweep, x))
        u_lines.add(lu); v_lines.add(lv)
    for g in (u_lines, v_lines):
        g.set_stroke(width=width, color=color, opacity=opacity)
    return u_lines, v_lines


def build_textured_surface(land, texture, n):
    surface = ParametricSurface(
        land.point,
        u_range=[-CANVAS_EXTENT, CANVAS_EXTENT],
        v_range=[-CANVAS_EXTENT, CANVAS_EXTENT],
        resolution=(n, n),
    )
    ts = TexturedSurface(surface, texture)
    ts.set_shading(0.0, 0.1, 0)
    return ts
 
 
class P47_48_landscape_1(InteractiveScene):
 
    cfg = 'plain74_first4'
    max_height = 2.5
    fold_view = (-45, 51, 0, (np.float32(0.03), np.float32(-0.15), np.float32(0.52)), 7.21)
    gridline_view = (-42, 46, 0, (np.float32(0.01), np.float32(-0.2), np.float32(0.45)), 6.70)
    final_view = (46, 45, 0, (np.float32(-0.07), np.float32(-0.01), np.float32(0.42)))
    overhead_view = (90, 0, 0, (np.float32(0.03), np.float32(-0.02), np.float32(0.35)), 6.0)
 
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
 
        # self.frame.reorient(*self.fold_view)
        # self.wait(0)

        field = GradientField(grad_field_file)
        arrows = build_gradient_arrows(field)          # pass landscape=land here to drape them on the 3D surface instead
        arrows.set_color(WHITE)

        # keep the 3D versions
        ts_3d = ts.copy()
        u_grid_3d = u_gridlines.copy()
        v_grid_3d = v_gridlines.copy()

        # pre-flatten the live ones
        ts.stretch(0, 2, about_point=ORIGIN)
        u_gridlines.stretch(0, 2, about_point=ORIGIN)
        v_gridlines.stretch(0, 2, about_point=ORIGIN)
        self.frame.reorient(*overhead_view)

        self.add(ts, arrows)
        self.wait(1.0)

        self.play(FadeOut(arrows), FadeIn(u_gridlines), FadeIn(v_gridlines), run_time=1.5)
        self.play(
            Transform(ts, ts_3d),
            Transform(u_gridlines, u_grid_3d),
            Transform(v_gridlines, v_grid_3d),
            self.frame.animate.reorient(*final_view),
            run_time=5.0,
        )

        self.wait(1)










        # #Reverse move from last time
        # self.play(FadeOut(arrows), FadeIn(arrows), run_time=1.5)
        # self.play(
        #     ts.animate.stretch(2, 0,  about_point=ORIGIN),
        #     u_gridlines.animate.stretch(2, 0,  about_point=ORIGIN),
        #     v_gridlines.animate.stretch(2, 0,  about_point=ORIGIN),
        #     self.frame.animate.reorient(*self.final_view),
        #     run_time=5.0,
        # )
        # self.wait()




        # self.play(ShowCreation(u_gridlines),
        #           ShowCreation(v_gridlines),
        #           self.frame.animate.reorient(*self.gridline_view),
        #           run_time=4.0)
        # self.wait()
 
        # ts.set_opacity(0.0)
        # self.add(ts)
        # self.add(u_gridlines, v_gridlines)
        # self.play(ts.animate.set_opacity(1.0),
        #           self.frame.animate.reorient(*self.final_view),
        #           run_time=5.0)
        # self.wait(2)
 
        # # --- go overhead and flatten ---
        # self.play(
        #     ts.animate.stretch(0, 2, about_point=ORIGIN),
        #     u_gridlines.animate.stretch(0, 2, about_point=ORIGIN),
        #     v_gridlines.animate.stretch(0, 2, about_point=ORIGIN),
        #     self.frame.animate.reorient(*self.overhead_view),
        #     run_time=5.0,
        # )
        # self.wait()



        # self.wait(1)
        # self.play(FadeIn(arrows), FadeOut(u_gridlines), FadeOut(v_gridlines), ts.animate.set_opacity(0.65), run_time=5.0)
        # self.add(arrows) 
        # self.remove(u_gridlines, v_gridlines)
        # ts.set_opacity(0.75)

        # bands = sweep_bands(arrows, field.positions)
        # self.play(LaggedStart(*[ShowCreation(b, lag_ratio=0) for b in bands],
        #                       lag_ratio=SWEEP_LAG, run_time=SWEEP_TIME))
        # self.remove(*bands)
                                      # single flat handle for whatever comes next
        self.wait(2)

        self.wait(20)
        self.embed()


        #overhead_view = (90, 0, 0, (np.float32(-0.0), np.float32(0.09), np.float32(0.35)), 5.16)




