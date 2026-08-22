from manimlib import *
import sys
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CUSTOM = str(REPO / 'custom')
sys.path.append(CUSTOM)
import welch_axes
welch_axes.WELCH_ASSET_PATH = CUSTOM
from welch_axes import *

data_dir = Path(__file__).resolve().parent / 'data'

CHILL_BROWN = '#948979'
CYAN = '#00FFFF'
MAGENTA = '#FF00FF'

alphas_1 = np.linspace(-2.5, 2.5, 512)
loss_2d_1 = np.load(data_dir / 'loss_2d_1.npy')
loss_texture = str(data_dir / 'loss_2d_1.png')

SURFACE_RESOLUTION = (512, 512)
NUM_GRIDLINES = 64
GRIDLINE_POINTS = 512

two_panel_view = (0, 89, 0, (0.0, 0.0, -1.85), 7.60)
fold_view = (36, 64, 0, (-0.07, 0.22, 0.23), 6.77)
gridline_view = (42, 58, 0, (-0.03, 0.06, 0.02), 6.19)
book_view = (137, 41, 0, (0.14, -0.04, -0.09), 6.81)


def param_surface_1(u, v):
    u_idx = np.abs(alphas_1 - u).argmin()
    v_idx = np.abs(alphas_1 - v).argmin()
    try:
        z = 0.07 * loss_2d_1[v_idx, u_idx]
    except IndexError:
        z = 0
    return np.array([u, v, z])


def get_pivot_and_scale(axis_min, axis_max, axis_end):
    return axis_min, axis_end / (axis_max - axis_min)


def make_panel(loss_values, color, x_label_tex=r'\alpha'):
    x_axis = WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0, -1.0, 0, 1.0, 2.0], x_tick_height=0.15,
                        x_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=5)
    y_axis = WelchYAxis(y_min=0, y_max=25, y_ticks=[0, 5, 10, 15, 20], y_tick_width=0.15,
                        y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)

    x_label = Tex(x_label_tex, font_size=28).set_color(CHILL_BROWN)
    y_label = Tex('Loss', font_size=22).set_color(CHILL_BROWN)
    x_label.next_to(x_axis, RIGHT, buff=0.05)
    y_label.next_to(y_axis, UP, buff=0.08)

    mapped_x = x_axis.map_to_canvas(alphas_1)
    mapped_y = y_axis.map_to_canvas(loss_values)

    curve = VMobject()
    curve.set_points_smoothly(np.vstack((mapped_x, mapped_y, np.zeros_like(mapped_x))).T)
    curve.set_stroke(width=4, color=color, opacity=1.0)

    panel = VGroup(x_axis, y_axis, x_label, y_label, curve)
    panel.rotate(90 * DEGREES, [1, 0, 0], about_point=ORIGIN)
    return panel


class P32(InteractiveScene):
    def construct(self):
        surface = ParametricSurface(
            param_surface_1,
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=SURFACE_RESOLUTION,
        )
        ts = TexturedSurface(surface, loss_texture)
        ts.set_shading(0.0, 0.1, 0)

        u_gridlines = VGroup()
        v_gridlines = VGroup()
        line_values = np.linspace(-2.5, 2.5, NUM_GRIDLINES)
        sweep = np.linspace(-2.5, 2.5, GRIDLINE_POINTS)
        for u in line_values:
            line = VMobject()
            line.set_points_smoothly([param_surface_1(u, v) for v in sweep])
            line.set_stroke(width=1, color=WHITE, opacity=0.15)
            u_gridlines.add(line)
        for v in line_values:
            line = VMobject()
            line.set_points_smoothly([param_surface_1(u, v) for u in sweep])
            line.set_stroke(width=1, color=WHITE, opacity=0.15)
            v_gridlines.add(line)

        panel_a = make_panel(loss_2d_1[255, :], CYAN, r'\alpha')
        panel_b = make_panel(loss_2d_1[:, 255], MAGENTA, r'\beta')
        panel_a.move_to([0, 0, 0])
        panel_b.move_to([0, 0, -3.75])

        x_axis_a, y_axis_a, x_label_a, y_label_a, curve_a = panel_a
        x_axis_b, y_axis_b, x_label_b, y_label_b, curve_b = panel_b

        self.frame.reorient(*two_panel_view)
        self.add(x_axis_a, y_axis_a, x_label_a, y_label_a)
        self.add(x_axis_b, y_axis_b, x_label_b, y_label_b)
        self.wait(0)

        self.play(ShowCreation(curve_a), ShowCreation(curve_b), run_time=4.0)
        self.wait()

        pivot_x, scale_x = get_pivot_and_scale(x_axis_a.x_min, x_axis_a.x_max, x_axis_a.axis_length_on_canvas)
        pivot_y, scale_y = get_pivot_and_scale(y_axis_a.y_min, y_axis_a.y_max, y_axis_a.axis_length_on_canvas)

        self.remove(y_axis_a, y_label_a, y_axis_b, y_label_b, x_label_a, x_label_b)

        rescale = [1 / scale_x, 1 / scale_x, 0.07 / scale_y]
        self.play(x_axis_b[-1][2].animate.set_opacity(0),
                  curve_a.animate.scale(rescale).move_to([0, 0, 0.72]),
                  curve_b.animate.scale(rescale).move_to([0, 0, 0.65]).rotate(90 * DEGREES, axis=[0, 0, 1]),
                  x_axis_a.animate.move_to([0, 0, -0.2]),
                  x_axis_b.animate.move_to([0, 0, -0.2]).rotate(90 * DEGREES, axis=[0, 0, 1]),
                  self.frame.animate.reorient(*fold_view),
                  run_time=5.0)
        self.wait()

        self.play(ShowCreation(u_gridlines),
                  ShowCreation(v_gridlines),
                  self.frame.animate.reorient(*gridline_view),
                  run_time=4.0)
        self.wait()

        ts.set_opacity(0.0)
        self.add(ts)
        self.add(u_gridlines, v_gridlines)
        self.add(curve_a, curve_b)
        clear_early = squish_rate_func(smooth, 0.0, 0.33)
        self.play(ts.animate.set_opacity(1.0),
                  curve_a.animate(rate_func=clear_early).set_opacity(0.0),
                  curve_b.animate(rate_func=clear_early).set_opacity(0.0),
                  x_axis_a.animate(rate_func=clear_early).set_opacity(0.0),
                  x_axis_b.animate(rate_func=clear_early).set_opacity(0.0),
                  self.frame.animate.reorient(*book_view),
                  run_time=5.0)
        self.remove(curve_a, curve_b, x_axis_a, x_axis_b)
        self.wait(2)
