import manimgx as gx
from resnet_assets import LANDSCAPE_ROOT
from welch_surface import parametric_surface, textured_surface
from welch_scene import WelchScene
import numpy as np
from pathlib import Path
 
from resnet_landscapes import CANVAS_EXTENT, NUM_GRIDLINES, SWEEP_LAG, SWEEP_TIME, GradientField, LossLandscape, build_gradient_arrows, grad_field_file, sweep_bands


class P35_Gradient_Field(WelchScene):
 
    cfg = 'plain74_first4'
    max_height = 2.5
    fold_view = (-45, 51, 0, (np.float32(0.03), np.float32(-0.15), np.float32(0.52)), 7.21)
    gridline_view = (-42, 46, 0, (np.float32(0.01), np.float32(-0.2), np.float32(0.45)), 6.70)
    final_view = (45, 41, 0, (np.float32(0.07), np.float32(0.03), np.float32(0.35)), 6.92)
    overhead_view = (90, 0, 0, (np.float32(-0.0), np.float32(0.09), np.float32(0.35)), 5.16)
 
    def construct(self):
        land = LossLandscape(self.cfg, self.max_height)
        n = land.n
 
        surface = parametric_surface(
            land.point,
            u_range=[-CANVAS_EXTENT, CANVAS_EXTENT],
            v_range=[-CANVAS_EXTENT, CANVAS_EXTENT],
            resolution=(n, n),
        )
        ts = textured_surface(surface, land.texture, shade_in_3d=True)
 
        line_values = np.linspace(-CANVAS_EXTENT, CANVAS_EXTENT, NUM_GRIDLINES)
        sweep = np.linspace(-CANVAS_EXTENT, CANVAS_EXTENT, n)
        u_gridlines = gx.VGroup()
        v_gridlines = gx.VGroup()
        for u in line_values:
            line = gx.VMobject()
            line.set_points_smoothly(land.points(u, sweep))
            line.set_stroke(width=1, color=gx.WHITE, opacity=0.15)
            u_gridlines.add(line)
        for v in line_values:
            line = gx.VMobject()
            line.set_points_smoothly(land.points(sweep, v))
            line.set_stroke(width=1, color=gx.WHITE, opacity=0.15)
            v_gridlines.add(line)
 
        self.set_view(*self.fold_view)
        self.wait(0)
 
        self.play(gx.Create(u_gridlines),
                  gx.Create(v_gridlines),
                  self.animate_view(*self.gridline_view),
                  run_time=4.0)
        self.wait()
 
        ts.set_opacity(0.0)
        self.add(ts)
        self.add(u_gridlines, v_gridlines)
        self.play(ts.animate.set_opacity(1.0),
                  self.animate_view(*self.final_view),
                  run_time=5.0)
        self.wait(2)
 
        # --- go overhead and flatten ---
        self.play(
            ts.animate.stretch(0, 2, about_point=gx.ORIGIN),
            u_gridlines.animate.stretch(0, 2, about_point=gx.ORIGIN),
            v_gridlines.animate.stretch(0, 2, about_point=gx.ORIGIN),
            self.animate_view(*self.overhead_view),
            run_time=5.0,
        )
        self.wait()

        # --- gradient field sweeps in ---
        field = GradientField(grad_field_file)
        arrows = build_gradient_arrows(field)          # pass landscape=land here to drape them on the 3D surface instead
        arrows.set_color(gx.WHITE)

        self.wait(1)
        self.play(gx.FadeIn(arrows), gx.FadeOut(u_gridlines), gx.FadeOut(v_gridlines), ts.animate.set_opacity(0.65), run_time=5.0)
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


