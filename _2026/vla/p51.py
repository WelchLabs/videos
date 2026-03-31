from manimlib import *
import numpy as np
import matplotlib.cm as mpl_cm
from PIL import Image as PILImage


SVG_DIR = '/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/vla/graphics/p50_52_to_manim'
HACKIN_DIR = '/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/vla/hackin/p51'
HACKIN_DIR_B = '/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/vla/hackin/p51b'

all_trajectories = np.load(f'{HACKIN_DIR}/all_trajectories.npy')

ROW_IDX = 8

_heatmap_data = all_trajectories[-1][:, :14].T
_vmin_h, _vmax_h = _heatmap_data.min(), _heatmap_data.max()
_normed_h = (_heatmap_data - _vmin_h) / (_vmax_h - _vmin_h)
_heatmap_rgba = (mpl_cm.viridis(_normed_h) * 255).astype(np.uint8)
_imshow_rgba = np.repeat(np.repeat(_heatmap_rgba, 20, axis=0), 20, axis=1)
_imshow_path = f'{HACKIN_DIR}/gen_imshow.png'
PILImage.fromarray(_imshow_rgba, 'RGBA').save(_imshow_path)

_row8_norm_vals = _normed_h[ROW_IDX]
_row8_mean_norm = float(_row8_norm_vals.mean())
_row8_rgb = (np.array(mpl_cm.viridis(_row8_mean_norm)[:3]) * 255).astype(np.uint8)
ROW_COLOR = '#{:02X}{:02X}{:02X}'.format(*_row8_rgb)

_data_full = all_trajectories[-1]

_grad = np.linspace(1, 0, 512)
_vert_rgba = (mpl_cm.viridis(_grad) * 255).astype(np.uint8)
_vert_path = f'{HACKIN_DIR}/gen_colorbar_vert.png'
PILImage.fromarray(np.tile(_vert_rgba[:, np.newaxis, :], (1, 32, 1)), 'RGBA').save(_vert_path)

colorbar_10   = ImageMobject(_imshow_path)
colorbar_vert = ImageMobject(_vert_path)
CHILL_BROWN = '#948979'

svg_01 = SVGMobject(f'{SVG_DIR}/p50_52_to_manim-01.svg')[1:]
svg_02 = SVGMobject(f'{SVG_DIR}/p50_52_to_manim-02.svg')[1:]
svg_03 = SVGMobject(f'{SVG_DIR}/p50_52_to_manim-03.svg')[1:]
svg_04 = SVGMobject(f'{SVG_DIR}/p50_52_to_manim-04.svg')[1:]
svg_05 = SVGMobject(f'{SVG_DIR}/p50_52_to_manim-05.svg')[1:-1]


def get_index_labels(svg, color=YELLOW):
    """Returns a VGroup of index numbers centered on each part of an SVGMobject.
    Font size scales with the height of each part."""
    labels = VGroup()
    for i, part in enumerate(svg):
        font_size = svg.get_height() * 0.5
        label = Integer(i, font_size=font_size, color=color)
        label.next_to(part, LEFT, buff=0).align_to(part, LEFT)
        labels.add(label)
    return labels


class P51(InteractiveScene):
    def construct(self):
        VGroup(svg_02, svg_03, svg_04, svg_05).set_width(self.camera.get_frame_width()-1).move_to(ORIGIN)

        arrow_02_03 = Arrow(svg_02.get_center()+(RIGHT*svg_02.get_width()/2), svg_02.get_center()+(RIGHT*svg_02.get_width()/2)+RIGHT * 0.5, buff=0.1, thickness=1).set_color(CHILL_BROWN)
        arrow_05_left = Arrow(svg_05.get_center()+(LEFT*svg_05.get_width()/2)+UP*0.3, svg_05.get_center()+(LEFT*svg_05.get_width()/2)+LEFT * 0.5+UP*0.3, buff=0.1, thickness=1).set_color('#8B0000')

        colorbar_frame = svg_03[13]
        colorbar_10.stretch_to_fit_width(colorbar_frame.get_width() * (966.16 / 972.16))
        colorbar_10.stretch_to_fit_height(colorbar_frame.get_height() * (297.25 / 303.25))
        colorbar_10.move_to(colorbar_frame.get_center())

        tick_span = VGroup(svg_04[4], svg_04[6])
        colorbar_vert.stretch_to_fit_height(tick_span.get_height())
        colorbar_vert.next_to(tick_span, LEFT, buff=0)
        colorbar_vert.shift(LEFT * 0.15)
        colorbar_vert_outline = SurroundingRectangle(colorbar_vert, buff=0, color=CHILL_BROWN, stroke_width=1)

        timestep_label_grp = VGroup(*svg_05[0:8])

        row_h = colorbar_10.get_height() / _heatmap_data.shape[0]
        col_rect = Rectangle(width=colorbar_10.get_width(), height=row_h)
        col_rect.set_fill(ROW_COLOR, opacity=1)
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

        col_vals = _data_full[:, 8]
        x_pts = np.linspace(line_left, line_right, len(col_vals))
        y_pts = np.interp(col_vals, [col_vals.min(), col_vals.max()], [line_bottom, line_top])
        shoulder_line = VMobject().set_points_as_corners(
            np.column_stack([x_pts, y_pts, np.zeros(len(col_vals))])
        )
        shoulder_line.set_color('#FFC107').set_stroke(width=2)

        self.play(FadeIn(svg_02))
        self.play(GrowArrow(arrow_02_03))
        self.play(ShowCreation(svg_03))
        self.play(FadeIn(colorbar_10))
        self.add(col_rect)
        self.play(ShowCreation(svg_04))
        self.play(FadeIn(colorbar_vert), ShowCreation(colorbar_vert_outline))

        self.play(ShowCreation(svg_05))
        self.play(GrowArrow(arrow_05_left))

        self.play(col_rect.animate.move_to([line_cx, line_cy, 0]), run_time=1.0)
        self.play(Transform(col_rect, shoulder_line), run_time=0.8)
        self.embed()


class P51v2(InteractiveScene):
    def construct(self):
        VGroup(svg_02, svg_03, svg_04, svg_05).set_width(self.camera.get_frame_width()-1).move_to(ORIGIN)

        _ts = VGroup(*svg_03[0:8])
        _ts_orig_h = _ts.get_height()
        _ts.scale(1.5, about_point=_ts.get_center())
        _ts.shift(DOWN * (_ts.get_height() - _ts_orig_h))

        _ae_text = VGroup(*svg_02[1:])
        _ae_text.scale(1.5, about_point=_ae_text.get_center())

        _joint = VGroup(*svg_03[8:13])
        _joint_orig_w = _joint.get_width()
        _joint.scale(1.5, about_point=_joint.get_center())
        svg_02.shift(LEFT * (_joint.get_width() - _joint_orig_w))

        _ts5 = VGroup(*svg_05[0:8])
        _ts5_orig_h = _ts5.get_height()
        _ts5.scale(1.5, about_point=_ts5.get_center())
        _ts5.shift(DOWN * (_ts5.get_height() - _ts5_orig_h))

        _jp = VGroup(*svg_05[8:21])
        _jp_orig_w = _jp.get_width()
        _jp.scale(1.5, about_point=_jp.get_center())
        _jp.shift(LEFT * (_jp.get_width() - _jp_orig_w))

        arrow_02_03 = Arrow(svg_02.get_center()+(RIGHT*svg_02.get_width()/2), svg_02.get_center()+(RIGHT*svg_02.get_width()/2)+RIGHT * 0.5, buff=0.1, thickness=1).set_color(CHILL_BROWN)
        arrow_05_left = Arrow(_jp.get_left()+UP*0.3, _jp.get_left()+LEFT*1.1+UP*0.3, buff=0.1, thickness=3).set_color('#8B0000')

        colorbar_frame = svg_03[13]
        colorbar_10.stretch_to_fit_width(colorbar_frame.get_width() * (966.16 / 972.16))
        colorbar_10.stretch_to_fit_height(colorbar_frame.get_height() * (297.25 / 303.25))
        colorbar_10.move_to(colorbar_frame.get_center())

        tick_span = VGroup(svg_04[4], svg_04[6])
        colorbar_vert.stretch_to_fit_height(tick_span.get_height())
        colorbar_vert.next_to(tick_span, LEFT, buff=0)
        colorbar_vert.shift(LEFT * 0.15)
        colorbar_vert_outline = SurroundingRectangle(colorbar_vert, buff=0, color=CHILL_BROWN, stroke_width=1)

        timestep_label_grp = VGroup(*svg_05[0:8])

        row_h = colorbar_10.get_height() / _heatmap_data.shape[0]
        col_rect = Rectangle(width=colorbar_10.get_width(), height=row_h)
        col_rect.set_fill(ROW_COLOR, opacity=1)
        col_rect.set_stroke(width=0)
        col_rect.move_to([
            colorbar_10.get_center()[0],
            colorbar_10.get_top()[1] - row_h * (ROW_IDX + 0.5),
            0,
        ])

        line_width  = colorbar_10.get_width() * 0.93
        line_height = line_width * (750 / 3000)
        line_right  = colorbar_10.get_right()[0]
        line_left   = line_right - line_width
        line_cx     = (line_left + line_right) / 2
        line_cy     = timestep_label_grp.get_top()[1] + 0.5 + line_height / 2
        line_bottom, line_top = line_cy - line_height / 2, line_cy + line_height / 2

        col_vals = _data_full[:, 8]
        x_pts = np.linspace(line_left, line_right, len(col_vals))

        y_data_min, y_data_max = 0.2, 0.6
        y_pts = line_bottom + (col_vals - y_data_min) / (y_data_max - y_data_min) * line_height

        shoulder_line = VMobject().set_points_as_corners(
            np.column_stack([x_pts, y_pts, np.zeros(len(col_vals))])
        ).set_color('#FFC107').set_stroke(width=2)

        ax_color = CHILL_BROWN
        ax_stroke = 1
        graph_box = Rectangle(width=line_width, height=line_height)
        graph_box.set_stroke(color=ax_color, width=ax_stroke).set_fill(opacity=0)
        graph_box.move_to([line_cx, line_cy, 0])

        y_axis = Line([line_left, line_bottom, 0], [line_left, line_top, 0]).set_stroke(color=ax_color, width=ax_stroke)
        y_ticks = VGroup()
        y_labels = VGroup()
        for yv in [0.2, 0.3, 0.4, 0.5, 0.6]:
            ty = line_bottom + (yv - y_data_min) / (y_data_max - y_data_min) * line_height
            tick = Line([line_left - 0.08, ty, 0], [line_left, ty, 0]).set_stroke(color=ax_color, width=ax_stroke)
            lbl = DecimalNumber(yv, num_decimal_places=1, font_size=22).set_color(ax_color)
            lbl.next_to(tick, LEFT, buff=0.05)
            y_ticks.add(tick)
            y_labels.add(lbl)

        x_ticks = VGroup()
        x_labels = VGroup()
        for ji in range(0, len(col_vals), 10):
            tx = x_pts[ji]
            tick = Line([tx, line_bottom, 0], [tx, line_bottom - 0.06, 0]).set_stroke(color=ax_color, width=ax_stroke)
            lbl = Integer(ji, font_size=20).set_color(ax_color)
            lbl.next_to(tick, DOWN, buff=0.04)
            x_ticks.add(tick)
            x_labels.add(lbl)

        axes_grp = VGroup(graph_box, y_axis, y_ticks, y_labels, x_ticks, x_labels)

        self.play(FadeIn(svg_02))
        self.play(GrowArrow(arrow_02_03), FadeIn(svg_03), run_time=0.8)
        for hide_idx in [14, 15, 16, 17, 18, 19]:
            svg_03[hide_idx].set_opacity(0)
        self.play(FadeIn(colorbar_10), ShowCreation(svg_04), run_time=0.8)
        self.add(col_rect)
        self.play(FadeIn(colorbar_vert), ShowCreation(colorbar_vert_outline), run_time=0.6)
        self.play(FadeIn(svg_05), run_time=0.8)
        self.play(FadeIn(axes_grp), run_time=1.2)

        pin_w = line_width / len(col_vals)
        pins = VGroup()
        for i in range(len(col_vals)):
            pin = Rectangle(width=pin_w * 0.85, height=row_h)
            pin.set_fill(ROW_COLOR, opacity=1).set_stroke(width=0)
            pin.move_to([x_pts[i], colorbar_10.get_top()[1] - row_h * (ROW_IDX + 0.5), 0])
            pins.add(pin)
        self.remove(col_rect)
        self.add(pins)

        pin_anims = LaggedStart(
            *[pin.animate.move_to([x_pts[i], y_pts[i], 0]).set_height(0.06)
              for i, pin in enumerate(pins)],
            lag_ratio=0.04,
        )
        self.play(
            LaggedStart(pin_anims, ShowCreation(shoulder_line), lag_ratio=0.6),
            run_time=5.5,
        )

        self.wait(0.5)
        self.play(FadeOut(pins), run_time=1.0)
        self.play(GrowArrow(arrow_05_left), run_time=1.5)
        self.wait(2.0)

        self.embed()
