import manimgx as gx
from resnet_assets import LANDSCAPE_ROOT
from welch_surface import parametric_surface, textured_surface
from welch_scene import WelchScene
import numpy as np
from pathlib import Path
 
from resnet_landscapes import CANVAS_EXTENT, NUM_GRIDLINES, LossLandscape


class P48_final_layers_landscape(WelchScene):
 
    cfg = 'resnet74_last4'
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

        self.wait(20)
