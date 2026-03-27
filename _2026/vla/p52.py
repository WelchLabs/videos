from manimlib import *
import numpy as np
import matplotlib.cm as mpl_cm
from PIL import Image as PILImage


SVG_DIR = '/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/vla/graphics/p50_52_to_manim'
HACKIN_DIR = '/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/vla/hackin/p51'
HACKIN_DIR_B = '/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/vla/hackin/p51b'
CAT_DIR = '/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/vla/hackin/mar_17_2'
ROBOT_VID_DIR_1 = '/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/vla/hackin/mar_17_4'
ROBOT_VID_DIR_2 = '/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/vla/hackin/mar_17_5'

CHILL_BROWN = '#948979'

all_trajectories = np.load(f'{HACKIN_DIR}/all_trajectories.npy')

_heatmap_data = all_trajectories[-1][:, :14].T
_vmin_h, _vmax_h = _heatmap_data.min(), _heatmap_data.max()
_normed_h = (_heatmap_data - _vmin_h) / (_vmax_h - _vmin_h)
_heatmap_rgba = (mpl_cm.viridis(_normed_h) * 255).astype(np.uint8)
_imshow_rgba = np.repeat(np.repeat(_heatmap_rgba, 20, axis=0), 20, axis=1)
_imshow_path = f'{HACKIN_DIR}/gen_imshow.png'
PILImage.fromarray(_imshow_rgba, 'RGBA').save(_imshow_path)

_grad = np.linspace(1, 0, 512)
_vert_rgba = (mpl_cm.viridis(_grad) * 255).astype(np.uint8)
_vert_path = f'{HACKIN_DIR}/gen_colorbar_vert.png'
PILImage.fromarray(np.tile(_vert_rgba[:, np.newaxis, :], (1, 32, 1)), 'RGBA').save(_vert_path)

colorbar_10   = ImageMobject(_imshow_path)
colorbar_vert = ImageMobject(_vert_path)

svg_02 = SVGMobject(f'{SVG_DIR}/p50_52_to_manim-02.svg')[1:]
svg_03 = SVGMobject(f'{SVG_DIR}/p50_52_to_manim-03.svg')[1:]
svg_04 = SVGMobject(f'{SVG_DIR}/p50_52_to_manim-04.svg')[1:]
svg_05 = SVGMobject(f'{SVG_DIR}/p50_52_to_manim-05.svg')[1:-1]


class P52Part1(InteractiveScene):
    def construct(self):
        VGroup(svg_02, svg_03, svg_04, svg_05).set_width(self.camera.get_frame_width()-1).move_to(ORIGIN)

        arrow_02_03 = Arrow(
            svg_02.get_center() + RIGHT * svg_02.get_width()/2,
            svg_02.get_center() + RIGHT * svg_02.get_width()/2 + RIGHT * 0.5,
            buff=0.1, thickness=1
        ).set_color(CHILL_BROWN)

        colorbar_frame = svg_03[13]
        colorbar_10.stretch_to_fit_width(colorbar_frame.get_width() * (966.16 / 972.16))
        colorbar_10.stretch_to_fit_height(colorbar_frame.get_height() * (297.25 / 303.25))
        colorbar_10.move_to(colorbar_frame.get_center())

        tick_span = VGroup(svg_04[4], svg_04[6])
        colorbar_vert.stretch_to_fit_height(tick_span.get_height())
        colorbar_vert.next_to(tick_span, LEFT, buff=0)
        colorbar_vert.shift(LEFT * 0.15)
        colorbar_vert_outline = SurroundingRectangle(colorbar_vert, buff=0, color=CHILL_BROWN, stroke_width=1)

        upper_grp = Group(svg_02, arrow_02_03, svg_03, colorbar_10, svg_04, colorbar_vert, colorbar_vert_outline)
        upper_grp.shift(UP * 1.0)

        timestep_label_grp = VGroup(*svg_05[0:8])
        cat_img_height = 3.5
        cat_center_x = timestep_label_grp.get_center()[0]
        cat_center_y = svg_03.get_bottom()[1] - 0.3 - cat_img_height / 2
        cat_center = np.array([cat_center_x, cat_center_y, 0])

        cat_images = []
        for idx in range(100):
            path = f'{CAT_DIR}/step_{idx:03d}.png'
            img = ImageMobject(path)
            img.set_height(cat_img_height)
            img.move_to(cat_center)
            cat_images.append(img)

        heatmap_frames = []
        for idx in range(11):
            path = f'{HACKIN_DIR_B}/{idx:02d}.png'
            img = ImageMobject(path)
            img.stretch_to_fit_width(colorbar_10.get_width())
            img.stretch_to_fit_height(colorbar_10.get_height())
            img.move_to(colorbar_10.get_center())
            heatmap_frames.append(img)

        self.play(FadeIn(svg_02))
        self.play(GrowArrow(arrow_02_03))
        self.play(FadeIn(svg_03))
        self.add(heatmap_frames[0])
        self.play(FadeIn(heatmap_frames[0]))
        self.play(ShowCreation(svg_04))
        self.play(FadeIn(colorbar_vert), ShowCreation(colorbar_vert_outline))

        self.play(FadeIn(cat_images[0]))

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
        self.embed()


ROBOT_CROP_DIR = '/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/vla/hackin/p52_robot_cropped'


class P52Part2(InteractiveScene):
    def construct(self):
        import os
        os.makedirs(ROBOT_CROP_DIR, exist_ok=True)

        robot_src_dirs = [ROBOT_VID_DIR_1, ROBOT_VID_DIR_2]
        for di, src_dir in enumerate(robot_src_dirs):
            out_dir = f'{ROBOT_CROP_DIR}/ep{di}'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
                for fi in range(300):
                    img = PILImage.open(f'{src_dir}/{fi:03d}.png')
                    lower = img.crop((0, img.height // 2, img.width, img.height))
                    lower.save(f'{out_dir}/{fi:03d}.png')

        frame_w = self.camera.get_frame_width()
        frame_h = self.camera.get_frame_height()
        half_w = frame_w / 2

        div_line = Line(
            [0, frame_h/2, 0], [0, -frame_h/2, 0],
            stroke_width=1
        ).set_color(CHILL_BROWN)

        rows, cols = 4, 4
        cat_grid_w = half_w
        cat_grid_h = frame_h
        cell_w = cat_grid_w / cols
        cell_h = cat_grid_h / rows

        cat_step_indices = list(range(0, 100))
        cat_step_paths = [f'{CAT_DIR}/step_{idx:03d}.png' for idx in cat_step_indices]

        cat_grid_cells = []
        for r in range(rows):
            for c in range(cols):
                cx = -half_w / 2 + (c - (cols - 1) / 2) * cell_w
                cy = (rows - 1) / 2 * cell_h - r * cell_h
                cat_grid_cells.append(np.array([cx, cy, 0]))

        cat_grid_mobjects = []
        for pos in cat_grid_cells:
            img = ImageMobject(cat_step_paths[0])
            img.set_height(cell_h)
            img.set_width(cell_w, stretch=True)
            img.move_to(pos)
            cat_grid_mobjects.append(img)

        robot_frame_indices = list(range(0, 300))
        num_episodes = len(robot_src_dirs)
        robot_ep_dirs = [f'{ROBOT_CROP_DIR}/ep{i}' for i in range(num_episodes)]

        robot_grid_cells = []
        robot_ep_assignments = []
        for r in range(rows):
            for c in range(cols):
                cx = half_w / 2 + (c - (cols - 1) / 2) * cell_w
                cy = (rows - 1) / 2 * cell_h - r * cell_h
                robot_grid_cells.append(np.array([cx, cy, 0]))
                robot_ep_assignments.append((r * cols + c) % num_episodes)

        robot_grid_mobjects = []
        for idx, pos in enumerate(robot_grid_cells):
            ep = robot_ep_assignments[idx]
            img = ImageMobject(f'{robot_ep_dirs[ep]}/{robot_frame_indices[0]:03d}.png')
            img.set_height(cell_h)
            img.set_width(cell_w, stretch=True)
            img.move_to(pos)
            robot_grid_mobjects.append(img)

        self.play(ShowCreation(div_line))

        self.play(
            *[FadeIn(img) for img in cat_grid_mobjects],
            *[FadeIn(img) for img in robot_grid_mobjects],
            run_time=1.5,
        )
        self.wait(0.5)

        num_cat = len(cat_step_indices)
        num_robot = len(robot_frame_indices)
        current_cat_idx = 0

        for frame_i in range(1, num_robot):
            target_cat_idx = min(int(frame_i / (num_robot - 1) * (num_cat - 1)), num_cat - 1)

            if target_cat_idx != current_cat_idx:
                current_cat_idx = target_cat_idx
                new_cat_path = cat_step_paths[current_cat_idx]
                for cell_idx, pos in enumerate(cat_grid_cells):
                    old = cat_grid_mobjects[cell_idx]
                    new_img = ImageMobject(new_cat_path)
                    new_img.set_height(cell_h)
                    new_img.set_width(cell_w, stretch=True)
                    new_img.move_to(pos)
                    self.remove(old)
                    self.add(new_img)
                    cat_grid_mobjects[cell_idx] = new_img

            for cell_idx, pos in enumerate(robot_grid_cells):
                ep = robot_ep_assignments[cell_idx]
                fi = robot_frame_indices[frame_i]
                old = robot_grid_mobjects[cell_idx]
                new_img = ImageMobject(f'{robot_ep_dirs[ep]}/{fi:03d}.png')
                new_img.set_height(cell_h)
                new_img.set_width(cell_w, stretch=True)
                new_img.move_to(pos)
                self.remove(old)
                self.add(new_img)
                robot_grid_mobjects[cell_idx] = new_img

            self.wait(1/30)

        self.wait()
        self.embed()
