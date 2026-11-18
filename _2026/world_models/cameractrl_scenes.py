"""
CameraCtrl training / inference scenes (ManimGX).

    uv run manimgx render cameractrl_scenes.py TrainingExample
    uv run manimgx render cameractrl_scenes.py TrainingGrid
    uv run manimgx render cameractrl_scenes.py InferencePaths

Set CAMERACTRL_ROOT to the CameraCtrl checkout and WM_GRAPHICS to the shared artwork folder.

Data it uses
  training : data/re10k_samples/<clip_id>/  (from tools/download_re10k_samples.py) if present,
             otherwise the 10 clips bundled in assets/reference_videos + assets/pose_files
  inference: outputs/synth_*.mp4 and outputs/re10k_*.mp4 from cameractrl_explore.ipynb
             (all generated from stephen.jpg, seed 42); the synthetic camera paths are
             re-created here with the same formulas as the notebook's make_synthetic()

mp4s are unpacked to PNG frames once (needs ffmpeg on PATH) and cached in manim/_frames/.

Conventions
  RealEstate10K / CameraCtrl poses are OpenCV style (x right, y down, z forward), made relative
  to frame 0 exactly as the model sees them. They are mapped to manim's z-up world with
  (x, y, z) -> (x, z, -y): the first camera sits at the origin looking down +y.
  Pyramid shape comes from the real intrinsics: half-width at depth d = d * 0.5 / fx.
"""

import os

import manimgx as m
import numpy as np

from camera_gx import (
    CHILL_BROWN,
    PATH_COLORS,
    WL_YELLOW,
    CameraCtrlAssets,
    CameraRig,
    ClipFrames,
    Frustum,
    Trajectory,
    VideoPanel,
    VideoScreen,
    camera_to_world,
)
from welch_scene import camera_view

SHOW_LABELS = os.environ.get("CAMCTRL_NO_TEXT") is None


class TrainingExample(m.ThreeDScene):
    """One training pair: the clip (flat panel + on the pyramid's image plane) and its 25 camera poses."""
    camera_ctrl_root = None
    clip_id = None      # None -> pick by clip_index
    clip_index = 10
    run_time = 7         # seconds for the camera to traverse the 25 frames
    n_loops = 2
    extent = 3.5         # manim units the path is scaled to
    depth = 0.9          # pyramid length
    frame_height = 5.5   # camera zoom (smaller = closer)
    color = WL_YELLOW

    def construct(self):
        assets = CameraCtrlAssets(self.camera_ctrl_root)
        ids = assets.available_training_examples()
        if not ids and self.clip_id is None:
            return
        clip_id = self.clip_id or ids[self.clip_index % len(ids)]
        c2w, fxfy, frames = assets.load_training_example(clip_id)
        traj = Trajectory(c2w, fxfy, extent=self.extent)
        rig = CameraRig(traj, ClipFrames(frames), self.color, self.depth)

        mid = (traj.pos.min(axis=0) + traj.pos.max(axis=0)) / 2
        self.set_camera_orientation(**camera_view(
            -28, 68, center=mid + 0.4 * m.UP, height=self.frame_height,
        ))
        self.begin_ambient_camera_rotation(rate=1.5 * m.DEGREES)

        grid = m.NumberPlane(x_range=(-8, 8, 1), y_range=(-8, 8, 1), x_length=16, y_length=16, background_line_style=dict(stroke_color=CHILL_BROWN, stroke_width=1, stroke_opacity=0.35),
                           axis_config=dict(stroke_color=CHILL_BROWN, stroke_width=1, stroke_opacity=0.35))
        grid.shift(traj.pos[:, 2].min() * m.OUT - 0.6 * m.OUT)

        flat = VideoPanel(rig.frames, height=2.0, color=self.color, border_px=3.0).follow(rig.frame_index)
        flat.to_corner(m.UL, buff=0.35)
        self.add(grid, rig.mobjects)
        self.add_fixed_in_frame_mobjects(flat)
        if SHOW_LABELS:
            label = m.Text(f"RealEstate10K  {clip_id}", font_size=22).set_color(CHILL_BROWN)
            label.next_to(flat, m.DOWN, buff=0.12, aligned_edge=m.LEFT)
            self.add_fixed_in_frame_mobjects(label)

        self.wait(0.5)
        for _ in range(self.n_loops):
            rig.t.set_value(0)
            self.play(rig.t.animate.set_value(traj.n - 1), run_time=self.run_time, rate_func=m.linear)
            self.wait(1.0)


class TrainingGrid(m.ThreeDScene):
    """Several training pairs at once: a wall of little worlds, each with its own clip and camera path."""
    camera_ctrl_root = None
    clip_ids = None      # None -> first rows*cols available examples
    rows, cols = 2, 3
    spacing = (5.0, 3.6)  # (x, z) distance between worlds
    extent = 2.2
    depth = 0.8
    run_time = 7
    n_loops = 2

    def construct(self):
        assets = CameraCtrlAssets(self.camera_ctrl_root)
        ids = (self.clip_ids or assets.available_training_examples())[: self.rows * self.cols]
        if not ids:
            return
        rigs = []
        for k, clip_id in enumerate(ids):
            r, c = divmod(k, self.cols)
            offset = np.array([(c - (self.cols - 1) / 2) * self.spacing[0], 0, ((self.rows - 1) / 2 - r) * self.spacing[1]])
            c2w, fxfy, frames = assets.load_training_example(clip_id)
            traj = Trajectory(c2w, fxfy, extent=self.extent, max_scale=2.0, offset=offset)
            rigs.append(CameraRig(traj, ClipFrames(frames), PATH_COLORS[k % len(PATH_COLORS)], self.depth, ghost_every=2))

        self.set_camera_orientation(**camera_view(-18, 74, center=0.8 * m.UP, height=8.0))
        self.add(*[rig.mobjects for rig in rigs])
        self.wait(0.5)
        for _ in range(self.n_loops):
            for rig in rigs:
                rig.t.set_value(0)
            self.play(*[rig.t.animate.set_value(rig.n - 1) for rig in rigs], run_time=self.run_time, rate_func=m.linear)
            self.wait(1.0)

        self.wait(20)


class InferencePaths(m.ThreeDScene):
    """One starting image, several requested camera paths.

    For each path: the 25 target pyramids are laid out first (the path is the *input*), then the
    camera flies it while the generated frames appear on the image plane and in that path's panel.
    Finale: every path replays together from the shared first frame.
    """
    camera_ctrl_root = None
    runs = ["synth:dolly_in", "synth:truck_right", "synth:orbit_right", "synth:pan_right", "synth:dolly_out", "synth:zoom_in"]
    scale = 1.7          # manim units per pose unit - shared so the paths are comparable
    depth = 0.9
    layout_time = 1.5    # seconds to lay out the ghost pyramids
    run_time = 5         # seconds to fly a path
    finale_loops = 2
    frame_height = 6.0   # camera zoom (smaller = closer)

    def construct(self):
        assets = CameraCtrlAssets(self.camera_ctrl_root)
        rigs, names = [], []
        for name in self.runs:
            run = assets.load_inference_run(name)
            if run is None:
                print(f"[InferencePaths] no video for {name} in outputs/ - skipping")
                continue
            c2w, fxfy, frames = run
            traj = Trajectory(c2w, fxfy, scale=self.scale)
            rig = CameraRig(traj, ClipFrames(frames), PATH_COLORS[len(rigs) % len(PATH_COLORS)], self.depth,
                            ghost_every=2, follow_ghosts=False)
            self.set_flying(rig, False)
            rigs.append(rig)
            names.append(name.split(":")[1])

        if not rigs:
            return

        # the shared starting camera + condition image (frame 0 of the first run). It steps aside
        # while a camera is flying so it doesn't block the view of the generated frames.
        pose0 = rigs[0].traj.pose(0)
        start = Frustum(m.WHITE, self.depth).set_pose(pose0)
        start_screen = VideoScreen(rigs[0].frames, self.depth)
        start_screen.add_updater(lambda screen: screen.show(0, pose0, visible=not any(r.screen_visible for r in rigs)))
        start_screen.update()
        for rig in rigs:  # a flying camera shows its own frames from frame 1 on
            rig.screen.clear_updaters()
            rig.screen.add_updater(lambda screen, rig=rig: screen.show(rig.frame_index(), rig.traj.pose(rig.t.get_value()),
                                                             rig.screen_visible and rig.frame_index() >= 1))
            rig.screen.update()

        panels = m.Group()
        for rig, name in zip(rigs, names):
            panel = VideoPanel(rig.frames, height=1.05, color=rig.color, border_px=3.0).follow(rig.frame_index)
            if SHOW_LABELS:
                panel.add(m.Text(name.replace("_", " "), font_size=20).set_color(rig.color)
                          .next_to(panel.border, m.UP, buff=0.08))
            panels.add(panel)
        panels.arrange(m.RIGHT, buff=0.2)
        panels.scale_to_fit_width(min(panels.width, m.config.frame_width - 0.6))
        panels.to_edge(m.DOWN, buff=0.25)

        self.set_camera_orientation(**camera_view(
            -24, 66, center=0.2 * m.RIGHT + 1.1 * m.UP, height=self.frame_height,
        ))
        self.camera.frame.shift(camera_to_world(self.camera) @ (0.45 * m.DOWN))
        self.begin_ambient_camera_rotation(rate=m.DEGREES)
        self.add(*[rig.mobjects for rig in rigs], start_screen, start)
        self.wait(1.5)

        for rig, panel in zip(rigs, panels):
            self.play(rig.ghost.animate.set_value(rig.n), run_time=self.layout_time, rate_func=m.linear)
            self.add_fixed_in_frame_mobjects(panel)
            self.set_flying(rig, True)
            self.play(rig.t.animate.set_value(rig.n - 1), run_time=self.run_time, rate_func=m.linear)
            self.wait(0.8)
            self.set_flying(rig, False)   # leaves the ghost trail + the panel; start image comes back
            rig.t.set_value(0)

        # finale: all paths together
        for _ in range(self.finale_loops):
            self.wait(1.0)
            for rig in rigs:
                rig.t.set_value(0)
                self.set_flying(rig, True)
            self.play(*[rig.t.animate.set_value(rig.n - 1) for rig in rigs], run_time=self.run_time, rate_func=m.linear)
            self.wait(1.0)
            for rig in rigs:
                self.set_flying(rig, False)
        self.wait(1)

    @staticmethod
    def set_flying(rig, flying):
        rig.screen_visible = flying
        rig.frustum_visible = flying
        rig.frustum.set_stroke(opacity=1.0 if flying else 0.0)
