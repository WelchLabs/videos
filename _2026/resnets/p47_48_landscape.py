import manimgx as gx
from resnet_assets import LANDSCAPE_ROOT
from welch_surface import parametric_surface, textured_surface
from welch_scene import WelchScene
import numpy as np
from pathlib import Path
 
from resnet_landscapes import CANVAS_EXTENT, NUM_GRIDLINES, SWEEP_LAG, SWEEP_TIME, GradientField, LossLandscape, build_gradient_arrows, grad_field_file, sweep_bands


def build_gridlines(land, n_lines=NUM_GRIDLINES, color=gx.WHITE, width=1, opacity=0.15):
    line_values = np.linspace(-CANVAS_EXTENT, CANVAS_EXTENT, n_lines)
    sweep = np.linspace(-CANVAS_EXTENT, CANVAS_EXTENT, land.n)
    u_lines, v_lines = gx.VGroup(), gx.VGroup()
    for x in line_values:
        lu = gx.VMobject(); lu.set_points_smoothly(land.points(x, sweep))
        lv = gx.VMobject(); lv.set_points_smoothly(land.points(sweep, x))
        u_lines.add(lu); v_lines.add(lv)
    for g in (u_lines, v_lines):
        g.set_stroke(width=width, color=color, opacity=opacity)
    return u_lines, v_lines


def build_textured_surface(land, texture, n):
    surface = parametric_surface(
        land.point,
        u_range=[-CANVAS_EXTENT, CANVAS_EXTENT],
        v_range=[-CANVAS_EXTENT, CANVAS_EXTENT],
        resolution=(n, n),
    )
    ts = textured_surface(surface, texture, shade_in_3d=True)
    return ts

class P47_48_landscape_1(WelchScene):

    cfg = 'plain74_first4'
    cfg2 = 'resnet74_first4'
    max_height = 2.5
    fold_view = (-45, 51, 0, (np.float32(0.03), np.float32(-0.15), np.float32(0.52)), 7.21)
    gridline_view = (-42, 46, 0, (np.float32(0.01), np.float32(-0.2), np.float32(0.45)), 6.70)
    final_view = (46, 45, 0, (np.float32(-0.07), np.float32(-0.01), np.float32(0.42)), 9)
    overhead_view = (90, 0, 0, (np.float32(-0.0), np.float32(0.09), np.float32(0.35)), 6)

    def construct(self):
        # ---------- build plain74 (this replaces the old inline surface + gridline loops) ----------
        land = LossLandscape(self.cfg, self.max_height)
        n = land.n

        ts = build_textured_surface(land, land.texture, n)
        u_gridlines, v_gridlines = build_gridlines(land)

        field = GradientField(grad_field_file)
        arrows = build_gradient_arrows(field)
        arrows.set_color(gx.WHITE)

        # ---------- keep 3D copies, then flatten the live objects ----------
        ts_3d = ts.copy()
        u_grid_3d = u_gridlines.copy()
        v_grid_3d = v_gridlines.copy()

        ts.stretch(0, 2, about_point=gx.ORIGIN)
        u_gridlines.stretch(0, 2, about_point=gx.ORIGIN)
        v_gridlines.stretch(0, 2, about_point=gx.ORIGIN)
        self.set_view(*self.overhead_view)

        self.add(ts)
        self.add(arrows)
        self.wait(1.0)

        # ---------- unfold: flat -> 3D, camera to self.final_view, arrows fade out ----------
        self.play(gx.FadeOut(arrows), gx.FadeIn(u_gridlines), gx.FadeIn(v_gridlines), run_time=1.5)
        self.play(
            gx.Transform(ts, ts_3d),
            gx.Transform(u_gridlines, u_grid_3d),
            gx.Transform(v_gridlines, v_grid_3d),
            # FadeOut(arrows),
            self.animate_view(*self.final_view),
            run_time=5.0,
        )
        self.wait()

        # At this point:
        #   ts          = plain74 geometry, plain74 texture, opacity 1
        #   u/v_gridlines = plain74 gridlines
        #   camera      = self.final_view, and it stays there for the morph

        # ---------- morph plain74 -> resnet74 ----------
        MORPH_TIME = 5.0
        LIFT = 0.003 * gx.OUT      # keeps the stacked surface from z-fighting its twin

        land2 = LossLandscape(self.cfg2, self.max_height)
        assert land2.n == n     # both 512-grid, so Transform is point-for-point

        # plain-textured surface: morph geometry to resnet while fading out
        ts_plain_target = build_textured_surface(land2, land.texture, n)
        ts_plain_target.set_opacity(0.0)

        # resnet-textured surface: starts with plain geometry at opacity 0, morphs to resnet at opacity 1
        ts_res = build_textured_surface(land, land2.texture, n).shift(LIFT)
        ts_res.set_opacity(0.0)
        ts_res_target = build_textured_surface(land2, land2.texture, n).shift(LIFT)

        u_grid2, v_grid2 = build_gridlines(land2)

        self.wait(1)
        # self.add(ts_res)   # added after ts, so draw order is plain-behind, resnet-in-front
        self.add(ts_res, u_gridlines, v_gridlines)

        self.play(
            gx.Transform(ts, ts_plain_target),
            gx.Transform(ts_res, ts_res_target),
            gx.Transform(u_gridlines, u_grid2),
            gx.Transform(v_gridlines, v_grid2),
            run_time=MORPH_TIME,
        )
        self.remove(ts)    # fully transparent now; ts_res is the live surface from here on
        self.wait(2)


        #Fun littel camera move just in case we want it.
        self.play(self.animate_view(-133, 43, 0, (np.float32(-0.07), np.float32(-0.01), np.float32(0.42)), 9.00),
                 run_time=10.0)

        self.wait(20)
 
#     cfg = 'plain74_first4'
#     max_height = 2.5
#     fold_view = (-45, 51, 0, (np.float32(0.03), np.float32(-0.15), np.float32(0.52)), 7.21)
#     gridline_view = (-42, 46, 0, (np.float32(0.01), np.float32(-0.2), np.float32(0.45)), 6.70)
#     final_view = (46, 45, 0, (np.float32(-0.07), np.float32(-0.01), np.float32(0.42)))
#     overhead_view = (90, 0, 0, (np.float32(0.03), np.float32(-0.02), np.float32(0.35)), 6.0)
 
#     def construct(self):
#         land = LossLandscape(self.cfg, self.max_height)
#         n = land.n
 
#         surface = ParametricSurface(
#             land.point,
#             u_range=[-CANVAS_EXTENT, CANVAS_EXTENT],
#             v_range=[-CANVAS_EXTENT, CANVAS_EXTENT],
#             resolution=(n, n),
#         )
#         ts = textured_surface(surface, land.texture, shade_in_3d=True)
#         ts.set_shading(0.0, 0.1, 0)
 
#         line_values = np.linspace(-CANVAS_EXTENT, CANVAS_EXTENT, NUM_GRIDLINES)
#         sweep = np.linspace(-CANVAS_EXTENT, CANVAS_EXTENT, n)
#         u_gridlines = VGroup()
#         v_gridlines = VGroup()
#         for u in line_values:
#             line = VMobject()
#             line.set_points_smoothly(land.points(u, sweep))
#             line.set_stroke(width=1, color=WHITE, opacity=0.15)
#             u_gridlines.add(line)
#         for v in line_values:
#             line = VMobject()
#             line.set_points_smoothly(land.points(sweep, v))
#             line.set_stroke(width=1, color=WHITE, opacity=0.15)
#             v_gridlines.add(line)
 
#         # self.set_view(*self.fold_view)
#         # self.wait(0)

#         field = GradientField(grad_field_file)
#         arrows = build_gradient_arrows(field)          # pass landscape=land here to drape them on the 3D surface instead
#         arrows.set_color(WHITE)

#         # keep the 3D versions
#         ts_3d = ts.copy()
#         u_grid_3d = u_gridlines.copy()
#         v_grid_3d = v_gridlines.copy()

#         # pre-flatten the live ones
#         ts.stretch(0, 2, about_point=ORIGIN)
#         u_gridlines.stretch(0, 2, about_point=ORIGIN)
#         v_gridlines.stretch(0, 2, about_point=ORIGIN)
#         self.set_view(*overhead_view)

#         self.add(ts, arrows)
#         self.wait(1.0)

#         self.play(FadeOut(arrows), FadeIn(u_gridlines), FadeIn(v_gridlines), run_time=1.5)
#         self.play(
#             Transform(ts, ts_3d),
#             Transform(u_gridlines, u_grid_3d),
#             Transform(v_gridlines, v_grid_3d),
#             self.animate_view(*final_view),
#             run_time=5.0,
#         )

#         self.wait(1)


#         # --- morph plain74 -> resnet74 ---
#         MORPH_TIME = 5.0
#         LIFT = 0.003 * OUT      # keeps the stacked surface from z-fighting its twin

#         land2 = LossLandscape('resnet74_first4', self.max_height)
#         assert land2.n == n     # both 512-grid, so Transform is point-for-point

#         # geometry: plain -> resnet, keeping plain texture, fading out
#         ts_plain_target = build_textured_surface(land2, land.texture, n)
#         ts_plain_target.set_opacity(0.0)

#         # geometry: plain -> resnet, resnet texture, fading in (lifted a hair)
#         ts_res = build_textured_surface(land, land2.texture, n).shift(LIFT)
#         ts_res.set_opacity(0.0)
#         ts_res_target = build_textured_surface(land2, land2.texture, n).shift(LIFT)

#         u_grid2, v_grid2 = build_gridlines(land2)

#         self.add(ts_res)   # added after ts so draw order is plain-behind, resnet-in-front

#         self.play(
#             Transform(ts, ts_plain_target),
#             Transform(ts_res, ts_res_target),
#             Transform(u_gridlines, u_grid2),
#             Transform(v_gridlines, v_grid2),
#             run_time=MORPH_TIME,
#         )
#         self.remove(ts)    # fully transparent now; ts_res is the live surface from here on
#         self.wait(2)


        # #Reverse move from last time
        # self.play(FadeOut(arrows), FadeIn(arrows), run_time=1.5)
        # self.play(
        #     ts.animate.stretch(2, 0,  about_point=ORIGIN),
        #     u_gridlines.animate.stretch(2, 0,  about_point=ORIGIN),
        #     v_gridlines.animate.stretch(2, 0,  about_point=ORIGIN),
        #     self.animate_view(*self.final_view),
        #     run_time=5.0,
        # )
        # self.wait()


        # self.play(ShowCreation(u_gridlines),
        #           ShowCreation(v_gridlines),
        #           self.animate_view(*self.gridline_view),
        #           run_time=4.0)
        # self.wait()
 
        # ts.set_opacity(0.0)
        # self.add(ts)
        # self.add(u_gridlines, v_gridlines)
        # self.play(ts.animate.set_opacity(1.0),
        #           self.animate_view(*self.final_view),
        #           run_time=5.0)
        # self.wait(2)
 
        # # --- go overhead and flatten ---
        # self.play(
        #     ts.animate.stretch(0, 2, about_point=ORIGIN),
        #     u_gridlines.animate.stretch(0, 2, about_point=ORIGIN),
        #     v_gridlines.animate.stretch(0, 2, about_point=ORIGIN),
        #     self.animate_view(*self.overhead_view),
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


