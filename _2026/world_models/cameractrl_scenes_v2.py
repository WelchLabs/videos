"""
CameraCtrl training / inference scenes (ManimGL) - v2, adds the denoising process.

    manimgl <path to this file> TrainingExample
    manimgl <path to this file> TrainingGrid
    manimgl <path to this file> InferenceDenoise     # one path: lay out cameras -> frames resolve from noise -> fly it
    manimgl <path to this file> InferencePaths       # several paths from one image (denoises each one that has data)

Everything is found relative to this file (it lives in <CameraCtrl repo>/manim/), so it
runs from ~/videos on the Mac through the Dropbox copy of the repo.

Data it uses
  training : data/re10k_samples/<clip_id>/  (from tools/download_re10k_samples.py) if present,
             otherwise the 10 clips bundled in assets/reference_videos + assets/pose_files
  inference: outputs/synth_*.mp4 and outputs/re10k_*.mp4 from cameractrl_explore.ipynb
             (all generated from stephen.jpg, seed 42); the synthetic camera paths are
             re-created here with the same formulas as the notebook's make_synthetic()
  denoising: outputs/denoise/<run>/ written by section 9 of cameractrl_explore_v3.ipynb
             (xt = the noisy latents the model is fed, x0 = its running guess of the clean video,
              both decoded to images for every denoising step and every frame)

mp4s are unpacked to PNG frames once (needs ffmpeg on PATH) and cached in manim/_frames/.

Conventions
  RealEstate10K / CameraCtrl poses are OpenCV style (x right, y down, z forward), made relative
  to frame 0 exactly as the model sees them. They are mapped to manim's z-up world with
  (x, y, z) -> (x, z, -y): the first camera sits at the origin looking down +y.
  Pyramid shape comes from the real intrinsics: half-width at depth d = d * 0.5 / fx.
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from manimlib import *
from scipy.spatial.transform import Rotation, Slerp

# ----------------------------------------------------------------------------- config
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FRAME_CACHE = HERE / "_frames"
SAMPLES = REPO / "data" / "re10k_samples"

CHILL_BROWN = "#948979"
WL_YELLOW = "#ffd35a"
WL_BLUE = "#65c8d0"
WL_GREEN = "#00a14b"
WL_RED = "#e2584d"
WL_PURPLE = "#b38cdb"
WL_ORANGE = "#f39c4f"
PATH_COLORS = [WL_YELLOW, WL_BLUE, WL_RED, WL_GREEN, WL_PURPLE, WL_ORANGE]

SHOW_LABELS = os.environ.get("CAMCTRL_NO_TEXT") is None

CV2M = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # OpenCV world -> manim world


# ----------------------------------------------------------------------------- data
def ensure_frames(mp4):
    """Unpack an mp4 to PNGs (cached). Returns sorted list of frame paths."""
    mp4 = Path(mp4)
    out = FRAME_CACHE / mp4.stem
    if not out.exists() or not any(out.glob("*.png")):
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp4), "-vsync", "0",
                        "-start_number", "0", str(out / "%03d.png")], check=True)
    return sorted(out.glob("*.png"))


def load_pose_txt(path):
    """RealEstate10K txt -> (c2w relative to frame 0 [N,4,4], intrinsics [N,2] normalized fx, fy)."""
    rows = [[float(x) for x in l.split()] for l in Path(path).read_text().strip().splitlines()[1:]]
    rows = np.array(rows)
    w2c = np.tile(np.eye(4), (len(rows), 1, 1))
    w2c[:, :3, :] = rows[:, 7:19].reshape(-1, 3, 4)
    c2w = w2c[0][None] @ np.linalg.inv(w2c)
    return c2w, rows[:, 1:3]


def load_training_example(clip_id):
    """-> c2w [25,4,4], fxfy [25,2], frame paths [25]"""
    d = SAMPLES / clip_id
    if (d / "poses.npz").exists():
        z = np.load(d / "poses.npz")
        return z["c2w_sample"], z["intrinsics_sample"][:, :2], sorted((d / "frames").glob("*.png"))
    c2w, fxfy = load_pose_txt(REPO / "assets" / "pose_files" / f"{clip_id}_svdxt.txt")
    return c2w, fxfy, ensure_frames(REPO / "assets" / "reference_videos" / f"{clip_id}_svdxt.mp4")


def available_training_examples():
    if SAMPLES.exists():
        ids = sorted(p.parent.name for p in SAMPLES.glob("*/poses.npz"))
        if ids:
            return ids
    return sorted(p.name.replace("_svdxt.txt", "") for p in (REPO / "assets" / "pose_files").glob("*_svdxt.txt"))


REF_FX, REF_FY = 0.50765091, 0.90249053  # intrinsics the notebook's synthetic paths borrow


def synthetic_path(kind, n=25, amount=1.2, angle_deg=35):
    """Same formulas as make_synthetic() in cameractrl_explore.ipynb."""
    def rot_x(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    def rot_y(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    c2ws, fxfy = [], []
    for k in range(n):
        s = k / (n - 1)
        s = 0.5 - 0.5 * np.cos(np.pi * s)
        a = np.deg2rad(angle_deg) * s
        R, t, fx, fy = np.eye(3), np.zeros(3), REF_FX, REF_FY
        if kind == "dolly_in": t = [0, 0, amount * s]
        elif kind == "dolly_out": t = [0, 0, -amount * s]
        elif kind == "truck_right": t = [amount * s, 0, 0]
        elif kind == "pedestal_up": t = [0, -amount * s, 0]
        elif kind == "pan_right": R = rot_y(a)
        elif kind == "tilt_up": R = rot_x(-a)
        elif kind == "orbit_right":
            R = rot_y(-a); pivot = np.array([0, 0, 2.0 * amount]); t = pivot - R @ pivot
        elif kind == "zoom_in": fx, fy = REF_FX * (1 + 0.6 * s), REF_FY * (1 + 0.6 * s)
        elif kind == "spiral":
            R = rot_y(a * 0.5); t = [amount * np.sin(np.pi * s) * 0.6, -0.3 * amount * s, 0.5 * amount * s]
        else: raise ValueError(kind)
        m = np.eye(4); m[:3, :3] = R; m[:3, 3] = t
        c2ws.append(m); fxfy.append([fx, fy])
    return np.array(c2ws), np.array(fxfy)


def load_inference_run(name):
    """name like 'synth:orbit_right' or 're10k:0f47577ab3441480' -> c2w, fxfy, frames (or None if no video yet)."""
    kind, key = name.split(":")
    mp4 = REPO / "outputs" / f"{kind}_{key}.mp4"
    if not mp4.exists():
        return None
    if kind == "synth":
        c2w, fxfy = synthetic_path(key)
    else:
        c2w, fxfy = load_pose_txt(REPO / "assets" / "pose_files" / f"{key}_svdxt.txt")
    return c2w, fxfy, ensure_frames(mp4)


def load_denoise_run(name, view="xt"):
    """Intermediate denoising states saved by the notebook, or None if this run has none.

    -> dict(c2w, fxfy, states, frames, sigmas, paths) with paths[state][frame] -> image file.
    states[-1] is always the finished video."""
    d = REPO / "outputs" / "denoise" / name.replace(":", "_")
    if not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text())
    if view not in meta["kinds"]:
        return None
    z = np.load(d / "poses.npz")
    ext = meta.get("ext", "jpg")
    paths = {s: {f: d / view / f"s{s:02d}_f{f:03d}.{ext}" for f in meta["frames"]} for s in meta["states"]}
    return dict(c2w=z["c2w"], fxfy=z["fxfy"], states=meta["states"], frames=meta["frames"],
                sigmas=meta["sigmas"], n_frames=meta["n_frames"], paths=paths)


# ----------------------------------------------------------------------------- geometry
class Trajectory:
    """Camera path in manim coordinates with smooth interpolation between the sampled poses."""

    def __init__(self, c2w, fxfy, scale=None, extent=3.0, max_scale=3.0, offset=ORIGIN):
        self.n = len(c2w)
        pos = c2w[:, :3, 3]
        if scale is None:  # fit this path into `extent` manim units (RE10K scene scale is arbitrary per clip)
            reach = np.linalg.norm(pos - pos[0], axis=1).max()
            scale = min(extent / max(reach, 1e-6), max_scale)
        self.scale = scale
        self.pos = (pos * scale) @ CV2M.T + np.array(offset)
        self.fxfy = np.asarray(fxfy, dtype=float)
        self.slerp = Slerp(np.arange(self.n), Rotation.from_matrix(c2w[:, :3, :3]))
        self.has_translation = np.linalg.norm(self.pos - self.pos[0], axis=1).max() > 1e-3

    def pose(self, t):
        """t in [0, n-1] (fractional ok) -> dict(pos, right, up, fwd, fx, fy)"""
        t = float(np.clip(t, 0, self.n - 1))
        i = min(int(np.floor(t)), self.n - 2)
        a = t - i
        B = CV2M @ self.slerp([t])[0].as_matrix()
        fx, fy = (1 - a) * self.fxfy[i] + a * self.fxfy[i + 1]
        return dict(pos=(1 - a) * self.pos[i] + a * self.pos[i + 1],
                    right=B[:, 0], up=-B[:, 1], fwd=B[:, 2], fx=fx, fy=fy)


def image_plane(pose, depth):
    """-> center, half-width, half-height of the image plane `depth` in front of the camera."""
    return pose["pos"] + depth * pose["fwd"], depth * 0.5 / pose["fx"], depth * 0.5 / pose["fy"]


class Frustum(VGroup):
    """Wireframe camera pyramid: 4 edges from the apex + the image-plane rectangle."""

    def __init__(self, color=WL_YELLOW, depth=0.9, stroke_width=3, stroke_opacity=1.0):
        self.depth = depth
        self.edges = VGroup(*[Line(ORIGIN, RIGHT) for _ in range(4)])
        self.rect = VMobject()
        super().__init__(*self.edges, self.rect)
        self.set_stroke(color, stroke_width, stroke_opacity)
        self.set_fill(opacity=0)
        self.apply_depth_test()

    def set_pose(self, pose, grow=1.0):
        d = self.depth * max(grow, 1e-3)
        c, hw, hh = image_plane(pose, d)
        r, u = pose["right"], pose["up"]
        corners = [c - hw * r + hh * u, c + hw * r + hh * u, c + hw * r - hh * u, c - hw * r - hh * u]
        for line, corner in zip(self.edges, corners):
            line.set_points_as_corners([pose["pos"], corner])
        self.rect.set_points_as_corners([*corners, corners[0]])
        return self


class VideoScreen(Group):
    """Video frames that live on a frustum's image plane (one ImageMobject per frame; only one is shown)."""

    def __init__(self, frame_paths, depth=0.9, inset=0.985):
        self.depth = depth * inset  # sit just inside the wire rectangle to avoid z-fighting
        self.images = [ImageMobject(str(p)) for p in frame_paths]
        for im in self.images:
            im.signs = np.sign(im.get_points()[:, :2]).copy()
            im.apply_depth_test()
        super().__init__(*self.images)

    def show(self, idx, pose, visible=True):
        c, hw, hh = image_plane(pose, self.depth)
        for i, im in enumerate(self.images):
            if visible and i == idx:
                pts = c + im.signs[:, :1] * hw * pose["right"] + im.signs[:, 1:] * hh * pose["up"]
            else:  # collapse instead of fading: an invisible quad would still write depth
                pts = np.tile(c, (len(im.signs), 1))
            im.set_points(pts)
        return self


class FlatVideo(Group):
    """The same frames as a flat, fixed-in-frame panel with a border."""

    def __init__(self, frame_paths, height=1.6, color=WHITE):
        self.images = [ImageMobject(str(p), height=height) for p in frame_paths]
        self.border = SurroundingRectangle(self.images[0], buff=0).set_stroke(color, 3)
        super().__init__(*self.images, self.border)
        self.show(0)

    def show(self, idx):
        for i, im in enumerate(self.images):
            im.set_opacity(1.0 if i == idx else 0.0)
        return self


class CameraRig:
    """Moving frustum + video on its image plane + ghost pyramids at every sampled pose + path line.

    Driven by two ValueTrackers:
      t     camera position along the path, 0 .. n-1 (frame index)
      ghost how many ghost pyramids have been revealed, 0 .. n
    """

    def __init__(self, traj, frame_paths, color=WL_YELLOW, depth=0.9, ghost_every=1,
                 ghost_opacity=0.35, ghost_width=1.5, follow_ghosts=True):
        self.traj, self.n = traj, traj.n
        self.frame_paths = list(frame_paths)
        self.color = color
        self.t = ValueTracker(0)
        self.ghost = ValueTracker(0)
        self.screen_visible = True
        self.follow_ghosts = follow_ghosts  # True: ghosts appear as the camera passes (training)

        self.frustum = Frustum(color, depth)
        self.screen = VideoScreen(frame_paths, depth)
        self.ghost_ids = list(range(0, self.n, ghost_every))
        self.ghost_opacity = ghost_opacity
        self.ghosts = VGroup(*[Frustum(color, depth, ghost_width, ghost_opacity) for _ in self.ghost_ids])
        self.path = VMobject().set_stroke(color, 2.5, 0.9)
        self.path.apply_depth_test()

        self.frustum.add_updater(lambda m: m.set_pose(traj.pose(self.t.get_value())))
        self.screen.add_updater(lambda m: m.show(self.frame_index(), traj.pose(self.t.get_value()), self.screen_visible))
        self.ghosts.add_updater(self.update_ghosts)
        self.path.add_updater(self.update_path)
        self.mobjects = Group(self.path, self.ghosts, self.screen, self.frustum)
        self.mobjects.update()

    def frame_index(self):
        return int(np.clip(round(self.t.get_value()), 0, self.n - 1))

    def progress(self):
        return self.t.get_value() + 1 if self.follow_ghosts else self.ghost.get_value()

    def update_ghosts(self, ghosts):
        p = self.progress()
        for g, i in zip(ghosts, self.ghost_ids):
            grow = float(np.clip(p - i, 0, 1))
            g.set_pose(self.traj.pose(i), grow=grow)
            g.set_stroke(opacity=self.ghost_opacity if grow > 0 else 0)

    def update_path(self, path):
        end = float(np.clip(self.progress() - 1, 0, self.n - 1))
        if not self.traj.has_translation or end < 0.05:
            path.set_points_as_corners([self.traj.pos[0], self.traj.pos[0] + 1e-4 * UP])
            return
        ts = np.append(np.arange(0, end, 0.25), end)
        path.set_points_as_corners([self.traj.pose(t)["pos"] for t in ts])


def set_flying(rig, flying):
    """Show / hide a rig's moving camera and the video on its image plane (ghost trail is unaffected)."""
    rig.screen_visible = flying
    rig.frustum.set_stroke(opacity=1.0 if flying else 0.0)


class DenoiseKeyframes:
    """A few of the path's cameras, each holding its frame as it is denoised.

    One ValueTracker `step` (0 .. number of saved states - 1) drives every keyframe at once: the model
    denoises all frames of the video jointly, so they all resolve together.
    """

    def __init__(self, traj, run, key_frames=(0, 6, 12, 18, 24), color=WL_YELLOW, depth=0.6):
        self.traj, self.states = traj, run["states"]
        avail = np.array(run["frames"])
        self.key_frames = sorted(set(int(avail[np.abs(avail - f).argmin()]) for f in key_frames))
        self.paths = {f: [run["paths"][s][f] for s in self.states] for f in self.key_frames}
        self.step = ValueTracker(0)
        self.visible = False

        self.frusta = VGroup(*[Frustum(color, depth, 2.5, 1.0) for _ in self.key_frames])
        self.screens = Group(*[VideoScreen(self.paths[f], depth) for f in self.key_frames])
        self.frusta.add_updater(self.update_frusta)
        self.screens.add_updater(self.update_screens)
        self.mobjects = Group(self.screens, self.frusta)
        self.mobjects.update()

    def state_index(self):
        return int(np.clip(round(self.step.get_value()), 0, len(self.states) - 1))

    def update_frusta(self, frusta):
        for fr, f in zip(frusta, self.key_frames):
            fr.set_pose(self.traj.pose(f), grow=1.0 if self.visible else 1e-3)
            fr.set_stroke(opacity=1.0 if self.visible else 0.0)

    def update_screens(self, screens):
        for screen, f in zip(screens, self.key_frames):
            screen.show(self.state_index(), self.traj.pose(f), self.visible)

    def filmstrip(self, height=1.2, color=WHITE, buff=0.15):
        """The same keyframes as a flat row (not yet fixed in frame)."""
        strip = Group()
        for f in self.key_frames:
            flat = FlatVideo(self.paths[f], height=height, color=color)
            flat.add_updater(lambda m: m.show(self.state_index()))
            strip.add(flat)
        strip.arrange(RIGHT, buff=buff)
        return strip

def link_flat_video(flat, rig):
    flat.add_updater(lambda m: m.show(rig.frame_index()))
    return flat


def slow_orbit(frame, deg_per_sec=1.5):
    frame.add_updater(lambda m, dt: m.increment_theta(dt * deg_per_sec * DEGREES))


# ----------------------------------------------------------------------------- scenes
class TrainingExample(InteractiveScene):
    """One training pair: the clip (flat panel + on the pyramid's image plane) and its 25 camera poses."""
    clip_id = None       # None -> first available example
    run_time = 7         # seconds for the camera to traverse the 25 frames
    n_loops = 2
    extent = 3.5         # manim units the path is scaled to
    depth = 0.9          # pyramid length
    frame_height = 5.5   # camera zoom (smaller = closer)
    color = WL_YELLOW

    def construct(self):
        clip_id = self.clip_id or available_training_examples()[0]
        c2w, fxfy, frames = load_training_example(clip_id)
        traj = Trajectory(c2w, fxfy, extent=self.extent)
        rig = CameraRig(traj, frames, self.color, self.depth)

        mid = (traj.pos.min(axis=0) + traj.pos.max(axis=0)) / 2
        self.frame.reorient(-28, 68, 0, mid + 0.4 * UP, self.frame_height)
        slow_orbit(self.frame)

        grid = NumberPlane((-8, 8), (-8, 8), background_line_style=dict(stroke_color=CHILL_BROWN, stroke_width=1, stroke_opacity=0.35),
                           axis_config=dict(stroke_color=CHILL_BROWN, stroke_width=1, stroke_opacity=0.35))
        grid.shift(traj.pos[:, 2].min() * OUT - 0.6 * OUT)
        grid.apply_depth_test()

        flat = link_flat_video(FlatVideo(frames, height=2.0, color=self.color), rig)
        flat.to_corner(UL, buff=0.35)
        flat.fix_in_frame()

        self.add(grid, rig.mobjects, flat)
        if SHOW_LABELS:
            label = Text(f"RealEstate10K  {clip_id}", font_size=22).set_color(CHILL_BROWN)
            label.next_to(flat, DOWN, buff=0.12, aligned_edge=LEFT)
            label.fix_in_frame()
            self.add(label)

        self.wait(0.5)
        for _ in range(self.n_loops):
            rig.t.set_value(0)
            self.play(rig.t.animate.set_value(traj.n - 1), run_time=self.run_time, rate_func=linear)
            self.wait(1.0)


class TrainingGrid(InteractiveScene):
    """Several training pairs at once: a wall of little worlds, each with its own clip and camera path."""
    clip_ids = None      # None -> first rows*cols available examples
    rows, cols = 2, 3
    spacing = (5.0, 3.6)  # (x, z) distance between worlds
    extent = 2.2
    depth = 0.8
    run_time = 7
    n_loops = 2

    def construct(self):
        ids = (self.clip_ids or available_training_examples())[: self.rows * self.cols]
        rigs = []
        for k, clip_id in enumerate(ids):
            r, c = divmod(k, self.cols)
            offset = np.array([(c - (self.cols - 1) / 2) * self.spacing[0], 0, ((self.rows - 1) / 2 - r) * self.spacing[1]])
            c2w, fxfy, frames = load_training_example(clip_id)
            traj = Trajectory(c2w, fxfy, extent=self.extent, max_scale=2.0, offset=offset)
            rigs.append(CameraRig(traj, frames, PATH_COLORS[k % len(PATH_COLORS)], self.depth, ghost_every=2))

        self.frame.reorient(-18, 74, 0, 0.8 * UP, 8.0)
        self.add(*[rig.mobjects for rig in rigs])
        self.wait(0.5)
        for _ in range(self.n_loops):
            for rig in rigs:
                rig.t.set_value(0)
            self.play(*[rig.t.animate.set_value(rig.n - 1) for rig in rigs], run_time=self.run_time, rate_func=linear)
            self.wait(1.0)


class InferencePaths(InteractiveScene):
    """One starting image, several requested camera paths.

    For each path: the 25 target pyramids are laid out first (the path is the *input*), then the
    camera flies it while the generated frames appear on the image plane and in that path's panel.
    Finale: every path replays together from the shared first frame.
    """
    runs = ["synth:dolly_in", "synth:truck_right", "synth:orbit_right", "synth:pan_right", "synth:dolly_out", "synth:zoom_in"]
    scale = 1.7          # manim units per pose unit - shared so the paths are comparable
    depth = 0.9
    layout_time = 1.5    # seconds to lay out the ghost pyramids
    run_time = 5         # seconds to fly a path
    finale_loops = 2
    frame_height = 6.0   # camera zoom (smaller = closer)
    denoise = True       # resolve each path's keyframes from noise before flying it (needs outputs/denoise/<run>/)
    denoise_view = "xt"  # "xt": noisy model input, "x0": the model's running guess of the clean video
    denoise_time = 3
    key_frames = (0, 6, 12, 18, 24)
    key_depth = 0.6

    def construct(self):
        rigs, names = [], []
        for name in self.runs:
            run = load_inference_run(name)
            if run is None:
                print(f"[InferencePaths] no video for {name} in outputs/ - skipping")
                continue
            c2w, fxfy, frames = run
            traj = Trajectory(c2w, fxfy, scale=self.scale)
            rig = CameraRig(traj, frames, PATH_COLORS[len(rigs) % len(PATH_COLORS)], self.depth,
                            ghost_every=2, follow_ghosts=False)
            set_flying(rig, False)
            dn = load_denoise_run(name, self.denoise_view) if self.denoise else None
            rig.keyframes = DenoiseKeyframes(traj, dn, self.key_frames, rig.color, self.key_depth) if dn else None
            rigs.append(rig); names.append(name.split(":")[1])

        # the shared starting camera + condition image (frame 0 of the first run). It steps aside
        # while a camera is flying so it doesn't block the view of the generated frames.
        pose0 = rigs[0].traj.pose(0)
        start = Frustum(WHITE, self.depth).set_pose(pose0)
        start_screen = VideoScreen(rigs[0].frame_paths[:1], self.depth)
        start_screen.add_updater(lambda m: m.show(0, pose0, visible=not any(
            r.screen_visible or (r.keyframes is not None and r.keyframes.visible) for r in rigs)))
        start_screen.update()
        for rig in rigs:  # a flying camera shows its own frames from frame 1 on
            rig.screen.clear_updaters()
            rig.screen.add_updater(lambda m, rig=rig: m.show(rig.frame_index(), rig.traj.pose(rig.t.get_value()),
                                                             rig.screen_visible and rig.frame_index() >= 1))
            rig.screen.update()

        panels = Group()
        for rig, name in zip(rigs, names):
            panel = link_flat_video(FlatVideo(rig.frame_paths, height=1.05, color=rig.color), rig)
            if SHOW_LABELS:
                panel.add(Text(name.replace("_", " "), font_size=20).set_color(rig.color)
                          .next_to(panel.border, UP, buff=0.08))
            panels.add(panel)
        panels.arrange(RIGHT, buff=0.2)
        panels.set_width(min(panels.get_width(), FRAME_WIDTH - 0.6))
        panels.to_edge(DOWN, buff=0.25)
        panels.fix_in_frame()

        self.frame.reorient(-24, 66, 0, 0.2 * RIGHT + 1.1 * UP, self.frame_height)
        self.frame.shift(0.45 * self.frame.get_orientation().apply(DOWN))  # leave room for the panel row
        slow_orbit(self.frame, 1.0)
        self.add(*[rig.mobjects for rig in rigs], start_screen, start)
        self.add(*[rig.keyframes.mobjects for rig in rigs if rig.keyframes])
        self.wait(1.5)

        for rig, panel in zip(rigs, panels):
            self.play(rig.ghost.animate.set_value(rig.n), run_time=self.layout_time, rate_func=linear)
            kf = rig.keyframes
            if kf:   # the frames along this path resolve from noise, all together
                kf.visible = True
                self.play(kf.step.animate.set_value(len(kf.states) - 1), run_time=self.denoise_time, rate_func=linear)
                self.wait(0.6)
                kf.visible = False
            self.add(panel)
            set_flying(rig, True)
            self.play(rig.t.animate.set_value(rig.n - 1), run_time=self.run_time, rate_func=linear)
            self.wait(0.8)
            set_flying(rig, False)   # leaves the ghost trail + the panel; start image comes back
            rig.t.set_value(0)

        # finale: all paths together
        for _ in range(self.finale_loops):
            self.wait(1.0)
            for rig in rigs:
                rig.t.set_value(0)
                set_flying(rig, True)
            self.play(*[rig.t.animate.set_value(rig.n - 1) for rig in rigs], run_time=self.run_time, rate_func=linear)
            self.wait(1.0)
            for rig in rigs:
                set_flying(rig, False)
        self.wait(1)


class InferenceDenoise(InteractiveScene):
    """One camera path, start to finish, with the denoising made visible.

    1. the requested path is laid out as ghost pyramids (it is the input)
    2. a few keyframe cameras get their image planes, starting as noise; all of them resolve together over
       the denoising steps (also shown flat in the filmstrip, with a step progress bar)
    3. the finished video is flown along the path
    """
    run_name = "synth:orbit_right"
    view = "xt"          # "xt": noisy model input (stays pure noise until late - the sigma schedule starts at 700),
                         # "x0": the model's running guess of the clean video (blurry -> sharp from step 0)
    key_frames = (0, 6, 12, 18, 24)
    scale = 1.7
    depth = 0.9          # ghost / flying pyramid length
    key_depth = 0.6      # keyframe pyramid length (smaller so the image planes overlap less)
    layout_time = 1.5
    denoise_time = 8
    fly_time = 5
    fly_loops = 2
    frame_height = 4.6   # camera zoom (smaller = closer)
    color = WL_YELLOW

    def construct(self):
        dn = load_denoise_run(self.run_name, self.view)
        assert dn is not None, f"no outputs/denoise data for {self.run_name} ({self.view}) - run section 9 of cameractrl_explore_v3.ipynb"
        traj = Trajectory(dn["c2w"], dn["fxfy"], scale=self.scale)
        if len(dn["frames"]) == dn["n_frames"]:      # every frame was decoded: fly the final state
            final_frames = [dn["paths"][dn["states"][-1]][f] for f in dn["frames"]]
        else:                                        # only keyframes were decoded: use the notebook's mp4
            final_frames = load_inference_run(self.run_name)[2]

        rig = CameraRig(traj, final_frames, self.color, self.depth, ghost_every=2, follow_ghosts=False)
        set_flying(rig, False)
        kf = DenoiseKeyframes(traj, dn, self.key_frames, self.color, self.key_depth)

        pose0 = traj.pose(0)
        start = Frustum(WHITE, self.depth).set_pose(pose0)
        start_screen = VideoScreen(final_frames[:1], self.depth)
        start_screen.add_updater(lambda m: m.show(0, pose0, visible=not (rig.screen_visible or kf.visible)))
        start_screen.update()
        rig.screen.clear_updaters()
        rig.screen.add_updater(lambda m: m.show(rig.frame_index(), traj.pose(rig.t.get_value()),
                                                rig.screen_visible and rig.frame_index() >= 1))
        rig.screen.update()

        # flat filmstrip of the keyframes + denoising progress bar
        strip = kf.filmstrip(height=1.15, color=self.color)
        strip.set_width(min(strip.get_width(), FRAME_WIDTH - 1.0))
        strip.to_edge(DOWN, buff=0.45)
        bar_bg = Rectangle(width=strip.get_width(), height=0.07).set_stroke(CHILL_BROWN, 1).set_fill(CHILL_BROWN, 0.25)
        bar_bg.next_to(strip, DOWN, buff=0.12)
        bar = Rectangle(width=strip.get_width(), height=0.07).set_stroke(width=0).set_fill(self.color, 1)
        n_steps = len(kf.states) - 1

        def update_bar(m):
            frac = max(kf.step.get_value() / n_steps, 1e-3)
            m.set_width(frac * bar_bg.get_width(), stretch=True)
            m.move_to(bar_bg, LEFT)
        bar.add_updater(update_bar)
        hud = Group(strip, bar_bg, bar)
        if SHOW_LABELS:
            hud.add(Text("denoising", font_size=20).set_color(CHILL_BROWN).next_to(bar_bg, LEFT, buff=0.15))
            for flat, f in zip(strip, kf.key_frames):
                hud.add(Text(f"frame {f}", font_size=18).set_color(CHILL_BROWN).next_to(flat, UP, buff=0.06))
        hud.fix_in_frame()

        mid = (traj.pos.min(axis=0) + traj.pos.max(axis=0)) / 2
        self.frame.reorient(-24, 66, 0, mid + 0.5 * UP, self.frame_height)
        self.frame.shift(0.5 * self.frame.get_orientation().apply(DOWN))   # leave room for the filmstrip
        slow_orbit(self.frame, 1.0)

        self.add(rig.mobjects, kf.mobjects, start_screen, start)
        self.wait(1.5)
        # 1. the path
        self.play(rig.ghost.animate.set_value(rig.n), run_time=self.layout_time, rate_func=linear)
        self.wait(0.5)
        # 2. denoise
        kf.visible = True
        self.add(hud)
        self.wait(1.0)
        self.play(kf.step.animate.set_value(n_steps), run_time=self.denoise_time, rate_func=linear)
        self.wait(1.5)
        # 3. fly the finished video
        kf.visible = False
        set_flying(rig, True)
        for _ in range(self.fly_loops):
            rig.t.set_value(0)
            self.play(rig.t.animate.set_value(rig.n - 1), run_time=self.fly_time, rate_func=linear)
            self.wait(1.0)
