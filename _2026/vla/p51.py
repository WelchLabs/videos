import manimgx as gx
import numpy as np

from _2026.vla.trajectory_storyboard import color_scale, load_panels, load_trajectory_heatmap
from welch_scene import WelchScene


CHILL_BROWN = '#948979'
ROW_IDX = 8


def get_index_labels(svg, color=gx.YELLOW):
    """Returns a VGroup of index numbers centered on each part of an SVGMobject.
    Font size scales with the height of each part."""
    labels = gx.VGroup()
    for i, part in enumerate(svg):
        font_size = svg.height * 0.5
        label = gx.Integer(i, font_size=font_size, color=color)
        label.next_to(part, gx.LEFT, buff=0).align_to(part, gx.LEFT)
        labels.add(label)
    return labels


class P51(WelchScene):
    def construct(self):
        svg_02, svg_03, svg_04, svg_05 = load_panels()
        colorbar_10, col_vals, row_count, row_color = load_trajectory_heatmap(ROW_IDX)
        colorbar_vert = color_scale()

        gx.VGroup(svg_02, svg_03, svg_04, svg_05).scale_to_fit_width(self.camera.frame.width-1).move_to(gx.ORIGIN)

        arrow_02_03 = gx.Arrow(svg_02.get_center()+(gx.RIGHT*svg_02.width/2), svg_02.get_center()+(gx.RIGHT*svg_02.width/2)+gx.RIGHT * 0.5, buff=0.1, stroke_width=1).set_color(CHILL_BROWN)
        arrow_05_left = gx.Arrow(svg_05.get_center()+(gx.LEFT*svg_05.width/2)+gx.UP*0.3, svg_05.get_center()+(gx.LEFT*svg_05.width/2)+gx.LEFT * 0.5+gx.UP*0.3, buff=0.1, stroke_width=1).set_color('#8B0000')

        colorbar_frame = svg_03[13]
        colorbar_10.stretch_to_fit_width(colorbar_frame.width * (966.16 / 972.16))
        colorbar_10.stretch_to_fit_height(colorbar_frame.height * (297.25 / 303.25))
        colorbar_10.move_to(colorbar_frame.get_center())

        tick_span = gx.VGroup(svg_04[4], svg_04[6])
        colorbar_vert.stretch_to_fit_height(tick_span.height)
        colorbar_vert.next_to(tick_span, gx.LEFT, buff=0)
        colorbar_vert.shift(gx.LEFT * 0.15)
        colorbar_vert_outline = gx.SurroundingRectangle(colorbar_vert, buff=0, color=CHILL_BROWN, stroke_width=1)

        timestep_label_grp = gx.VGroup(*svg_05[0:8])

        row_h = colorbar_10.height / row_count
        col_rect = gx.Rectangle(width=colorbar_10.width, height=row_h)
        col_rect.set_fill(row_color, opacity=1)
        col_rect.set_stroke(width=0)
        col_rect.move_to([
            colorbar_10.get_center()[0],
            colorbar_10.get_top()[1] - row_h * (ROW_IDX + 0.5),
            0,
        ])

        line_width  = 5.5
        line_height = line_width * (750 / 3000)
        line_cx = timestep_label_grp.get_center()[0]
        line_cy = timestep_label_grp.get_top()[1] + 0.25 + line_height / 2
        line_left, line_right  = line_cx - line_width / 2, line_cx + line_width / 2
        line_bottom, line_top  = line_cy - line_height / 2, line_cy + line_height / 2

        x_pts = np.linspace(line_left, line_right, len(col_vals))
        y_pts = np.interp(col_vals, [col_vals.min(), col_vals.max()], [line_bottom, line_top])
        shoulder_line = gx.VMobject().set_points_as_corners(
            np.column_stack([x_pts, y_pts, np.zeros(len(col_vals))])
        )
        shoulder_line.set_color('#FFC107').set_stroke(width=2)

        self.play(gx.FadeIn(svg_02))
        self.play(gx.GrowArrow(arrow_02_03))
        self.play(gx.Create(svg_03))
        self.play(gx.FadeIn(colorbar_10))
        self.add(col_rect)
        self.play(gx.Create(svg_04))
        self.play(gx.FadeIn(colorbar_vert), gx.Create(colorbar_vert_outline))

        self.play(gx.Create(svg_05))
        self.play(gx.GrowArrow(arrow_05_left))

        self.play(col_rect.animate.move_to([line_cx, line_cy, 0]), run_time=1.0)
        self.play(gx.Transform(col_rect, shoulder_line), run_time=0.8)


class P51v2(WelchScene):
    def construct(self):
        svg_02, svg_03, svg_04, svg_05 = load_panels()
        colorbar_10, col_vals, row_count, row_color = load_trajectory_heatmap(ROW_IDX)
        colorbar_vert = color_scale()

        gx.VGroup(svg_02, svg_03, svg_04, svg_05).scale_to_fit_width(self.camera.frame.width-1).move_to(gx.ORIGIN)

        _ts = gx.VGroup(*svg_03[0:8])
        _ts_orig_h = _ts.height
        _ts.scale(1.5, about_point=_ts.get_center())
        _ts.shift(gx.DOWN * (_ts.height - _ts_orig_h))

        _ae_text = gx.VGroup(*svg_02[1:])
        _ae_text.scale(1.5, about_point=_ae_text.get_center())

        _joint = gx.VGroup(*svg_03[8:13])
        _joint_orig_w = _joint.width
        _joint.scale(1.5, about_point=_joint.get_center())
        svg_02.shift(gx.LEFT * (_joint.width - _joint_orig_w))

        _ts5 = gx.VGroup(*svg_05[0:8])
        _ts5_orig_h = _ts5.height
        _ts5.scale(1.5, about_point=_ts5.get_center())
        _ts5.shift(gx.DOWN * (_ts5.height - _ts5_orig_h))

        _jp = gx.VGroup(*svg_05[8:21])
        _jp_orig_w = _jp.width
        _jp.scale(1.5, about_point=_jp.get_center())
        _jp.shift(gx.LEFT * (_jp.width - _jp_orig_w))

        arrow_02_03 = gx.Arrow(svg_02.get_center()+(gx.RIGHT*svg_02.width/2), svg_02.get_center()+(gx.RIGHT*svg_02.width/2)+gx.RIGHT * 0.5, buff=0.1, stroke_width=1).set_color(CHILL_BROWN)
        arrow_05_left = gx.Arrow(_jp.get_left()+gx.UP*0.3, _jp.get_left()+gx.LEFT*1.1+gx.UP*0.3, buff=0.1, stroke_width=3).set_color('#8B0000')

        colorbar_frame = svg_03[13]
        colorbar_10.stretch_to_fit_width(colorbar_frame.width * (966.16 / 972.16))
        colorbar_10.stretch_to_fit_height(colorbar_frame.height * (297.25 / 303.25))
        colorbar_10.move_to(colorbar_frame.get_center())

        tick_span = gx.VGroup(svg_04[4], svg_04[6])
        colorbar_vert.stretch_to_fit_height(tick_span.height)
        colorbar_vert.next_to(tick_span, gx.LEFT, buff=0)
        colorbar_vert.shift(gx.LEFT * 0.15)
        colorbar_vert_outline = gx.SurroundingRectangle(colorbar_vert, buff=0, color=CHILL_BROWN, stroke_width=1)

        timestep_label_grp = gx.VGroup(*svg_05[0:8])

        row_h = colorbar_10.height / row_count
        col_rect = gx.Rectangle(width=colorbar_10.width, height=row_h)
        col_rect.set_fill(row_color, opacity=1)
        col_rect.set_stroke(width=0)
        col_rect.move_to([
            colorbar_10.get_center()[0],
            colorbar_10.get_top()[1] - row_h * (ROW_IDX + 0.5),
            0,
        ])

        line_width  = colorbar_10.width * 0.93
        line_height = line_width * (750 / 3000)
        line_right  = colorbar_10.get_right()[0]
        line_left   = line_right - line_width
        line_cx     = (line_left + line_right) / 2
        line_cy     = timestep_label_grp.get_top()[1] + 0.5 + line_height / 2
        line_bottom, line_top = line_cy - line_height / 2, line_cy + line_height / 2

        x_pts = np.linspace(line_left, line_right, len(col_vals))

        y_data_min, y_data_max = 0.2, 0.6
        y_pts = line_bottom + (col_vals - y_data_min) / (y_data_max - y_data_min) * line_height

        shoulder_line = gx.VMobject().set_points_as_corners(
            np.column_stack([x_pts, y_pts, np.zeros(len(col_vals))])
        ).set_color('#FFC107').set_stroke(width=2)

        ax_color = CHILL_BROWN
        ax_stroke = 1
        graph_box = gx.Rectangle(width=line_width, height=line_height)
        graph_box.set_stroke(color=ax_color, width=ax_stroke).set_fill(opacity=0)
        graph_box.move_to([line_cx, line_cy, 0])

        y_axis = gx.Line([line_left, line_bottom, 0], [line_left, line_top, 0]).set_stroke(color=ax_color, width=ax_stroke)
        y_ticks = gx.VGroup()
        y_labels = gx.VGroup()
        for yv in [0.2, 0.3, 0.4, 0.5, 0.6]:
            ty = line_bottom + (yv - y_data_min) / (y_data_max - y_data_min) * line_height
            tick = gx.Line([line_left - 0.08, ty, 0], [line_left, ty, 0]).set_stroke(color=ax_color, width=ax_stroke)
            lbl = gx.DecimalNumber(yv, num_decimal_places=1, font_size=22).set_color(ax_color)
            lbl.next_to(tick, gx.LEFT, buff=0.05)
            y_ticks.add(tick)
            y_labels.add(lbl)

        x_ticks = gx.VGroup()
        x_labels = gx.VGroup()
        for ji in range(0, len(col_vals), 10):
            tx = x_pts[ji]
            tick = gx.Line([tx, line_bottom, 0], [tx, line_bottom - 0.06, 0]).set_stroke(color=ax_color, width=ax_stroke)
            lbl = gx.Integer(ji, font_size=20).set_color(ax_color)
            lbl.next_to(tick, gx.DOWN, buff=0.04)
            x_ticks.add(tick)
            x_labels.add(lbl)

        axes_grp = gx.VGroup(graph_box, y_axis, y_ticks, y_labels, x_ticks, x_labels)

        self.play(gx.FadeIn(svg_02))
        self.play(gx.GrowArrow(arrow_02_03), gx.FadeIn(svg_03), run_time=0.8)
        for hide_idx in [14, 15, 16, 17, 18, 19]:
            svg_03[hide_idx].set_opacity(0)
        self.play(gx.FadeIn(colorbar_10), gx.Create(svg_04), run_time=0.8)
        self.add(col_rect)
        self.play(gx.FadeIn(colorbar_vert), gx.Create(colorbar_vert_outline), run_time=0.6)
        self.play(gx.FadeIn(svg_05), run_time=0.8)
        self.play(gx.FadeIn(axes_grp), run_time=1.2)

        pin_w = line_width / len(col_vals)
        pins = gx.VGroup()
        for i in range(len(col_vals)):
            pin = gx.Rectangle(width=pin_w * 0.85, height=row_h)
            pin.set_fill(row_color, opacity=1).set_stroke(width=0)
            pin.move_to([x_pts[i], colorbar_10.get_top()[1] - row_h * (ROW_IDX + 0.5), 0])
            pins.add(pin)
        self.remove(col_rect)
        self.add(pins)

        pin_anims = gx.LaggedStart(
            *[pin.animate.move_to([x_pts[i], y_pts[i], 0]).scale_to_fit_height(0.06)
              for i, pin in enumerate(pins)],
            lag_ratio=0.04,
        )
        self.play(
            gx.LaggedStart(pin_anims, gx.Create(shoulder_line), lag_ratio=0.6),
            run_time=5.5,
        )

        self.wait(0.5)
        self.play(gx.FadeOut(pins), run_time=1.0)
        self.play(gx.GrowArrow(arrow_05_left), run_time=1.5)
        self.wait(2.0)
