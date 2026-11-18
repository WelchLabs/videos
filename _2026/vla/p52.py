import manimgx as gx
import numpy as np

from _2026.vla.trajectory_storyboard import (
    ASSET_ROOT, DIFFUSION_DIR, HEATMAP_DIR, color_scale, load_panels,
)
from welch_scene import WelchScene, image_at_height


CHILL_BROWN = '#948979'


class P52Part1(WelchScene):
    def construct(self):
        svg_02, svg_03, svg_04, svg_05 = load_panels()
        colorbar_10 = image_at_height(HEATMAP_DIR / "00.png")
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

        arrow_02_03 = gx.Arrow(
            svg_02.get_center() + gx.RIGHT * svg_02.width/2,
            svg_02.get_center() + gx.RIGHT * svg_02.width/2 + gx.RIGHT * 0.5,
            buff=0.1, stroke_width=1
        ).set_color(CHILL_BROWN)

        colorbar_frame = svg_03[13]
        colorbar_10.stretch_to_fit_width(colorbar_frame.width * (966.16 / 972.16))
        colorbar_10.stretch_to_fit_height(colorbar_frame.height * (297.25 / 303.25))
        colorbar_10.move_to(colorbar_frame.get_center())

        tick_span = gx.VGroup(svg_04[4], svg_04[6])
        colorbar_vert.stretch_to_fit_height(tick_span.height)
        colorbar_vert.next_to(tick_span, gx.LEFT, buff=0)
        colorbar_vert.shift(gx.LEFT * 0.15)
        colorbar_vert_outline = gx.SurroundingRectangle(colorbar_vert, buff=0, color=CHILL_BROWN, stroke_width=1)

        upper_grp = gx.Group(svg_02, arrow_02_03, svg_03, colorbar_10, svg_04, colorbar_vert, colorbar_vert_outline)
        upper_grp.shift(gx.UP * 1.0)

        timestep_label_grp = gx.VGroup(*svg_05[0:8])
        cat_img_height = 3.5
        cat_center_x = timestep_label_grp.get_center()[0]
        cat_center_y = svg_03.get_bottom()[1] - 0.3 - cat_img_height / 2
        cat_center = np.array([cat_center_x, cat_center_y, 0])

        cat_images = []
        for idx in range(100):
            path = DIFFUSION_DIR / f"step_{idx:03d}.png"
            img = image_at_height(path)
            img.scale_to_fit_height(cat_img_height)
            img.move_to(cat_center)
            cat_images.append(img)

        heatmap_frames = [colorbar_10]
        for idx in range(1, 11):
            path = HEATMAP_DIR / f"{idx:02d}.png"
            img = image_at_height(path)
            img.stretch_to_fit_width(colorbar_10.width)
            img.stretch_to_fit_height(colorbar_10.height)
            img.move_to(colorbar_10.get_center())
            heatmap_frames.append(img)

        self.play(gx.FadeIn(svg_02))
        self.play(gx.GrowArrow(arrow_02_03), gx.FadeIn(svg_03), run_time=0.8)
        for hide_idx in [14, 15, 16, 17, 18, 19]:
            svg_03[hide_idx].set_opacity(0)
        self.play(gx.FadeIn(heatmap_frames[0]), gx.Create(svg_04), run_time=0.8)
        self.play(gx.FadeIn(colorbar_vert), gx.Create(colorbar_vert_outline), run_time=0.6)

        self.play(gx.FadeIn(cat_images[0]))

        num_cat = 100
        num_heat = 11
        current_heat = 0
        for i in range(1, num_cat):
            self.remove(cat_images[i - 1])
            self.add(cat_images[i])

            target_heat = min(int(i / (num_cat - 1) * (num_heat - 1)), num_heat - 1)
            if target_heat != current_heat:
                self.remove(heatmap_frames[current_heat])
                self.add(heatmap_frames[target_heat])
                current_heat = target_heat

            self.wait(1/30)

        self.wait()


P52_ASSETS = ASSET_ROOT / "hackin" / "pranav_mar_27" / "p52_assets"
NUM_GRID = 16


class P52Part2(WelchScene):
    def construct(self):
        frame_w = self.camera.frame.width
        frame_h = self.camera.frame.height
        half_w = frame_w / 2

        div_line = gx.Line(
            [0, frame_h/2, 0], [0, -frame_h/2, 0],
            stroke_width=1
        ).set_color(CHILL_BROWN)

        rows, cols = 4, 4
        cell_w = half_w / cols
        cell_h = frame_h / rows

        cat_seq_dirs = [f'{P52_ASSETS}/cat_seq{i}' for i in range(NUM_GRID)]
        robot_ep_dirs = [f'{P52_ASSETS}/robot_ep{i}' for i in range(NUM_GRID)]

        num_cat_steps = 100
        num_robot_frames = 300

        cat_grid_cells = []
        for r in range(rows):
            for c in range(cols):
                cx = -half_w / 2 + (c - (cols - 1) / 2) * cell_w
                cy = (rows - 1) / 2 * cell_h - r * cell_h
                cat_grid_cells.append(np.array([cx, cy, 0]))

        robot_grid_cells = []
        for r in range(rows):
            for c in range(cols):
                cx = half_w / 2 + (c - (cols - 1) / 2) * cell_w
                cy = (rows - 1) / 2 * cell_h - r * cell_h
                robot_grid_cells.append(np.array([cx, cy, 0]))

        cat_grid_mobjects = []
        for cell_idx, pos in enumerate(cat_grid_cells):
            img = image_at_height(f'{cat_seq_dirs[cell_idx]}/step_000.png')
            img.scale_to_fit_height(cell_h)
            img.stretch_to_fit_width(cell_w)
            img.move_to(pos)
            cat_grid_mobjects.append(img)

        robot_grid_mobjects = []
        for cell_idx, pos in enumerate(robot_grid_cells):
            img = image_at_height(f'{robot_ep_dirs[cell_idx]}/000.png')
            img.scale_to_fit_height(cell_h)
            img.stretch_to_fit_width(cell_w)
            img.move_to(pos)
            robot_grid_mobjects.append(img)

        self.play(gx.Create(div_line))
        self.play(
            *[gx.FadeIn(img) for img in cat_grid_mobjects],
            *[gx.FadeIn(img) for img in robot_grid_mobjects],
            run_time=1.5,
        )
        self.wait(0.5)

        current_cat_idx = 0
        for frame_i in range(1, num_robot_frames):
            target_cat_idx = min(int(frame_i / (num_robot_frames - 1) * (num_cat_steps - 1)), num_cat_steps - 1)

            if target_cat_idx != current_cat_idx:
                current_cat_idx = target_cat_idx
                for cell_idx, pos in enumerate(cat_grid_cells):
                    old = cat_grid_mobjects[cell_idx]
                    new_img = image_at_height(f'{cat_seq_dirs[cell_idx]}/step_{current_cat_idx:03d}.png')
                    new_img.scale_to_fit_height(cell_h)
                    new_img.stretch_to_fit_width(cell_w)
                    new_img.move_to(pos)
                    self.remove(old)
                    self.add(new_img)
                    cat_grid_mobjects[cell_idx] = new_img

            for cell_idx, pos in enumerate(robot_grid_cells):
                old = robot_grid_mobjects[cell_idx]
                new_img = image_at_height(f'{robot_ep_dirs[cell_idx]}/{frame_i:03d}.png')
                new_img.scale_to_fit_height(cell_h)
                new_img.stretch_to_fit_width(cell_w)
                new_img.move_to(pos)
                self.remove(old)
                self.add(new_img)
                robot_grid_mobjects[cell_idx] = new_img

            self.wait(1/30)

        self.wait()
