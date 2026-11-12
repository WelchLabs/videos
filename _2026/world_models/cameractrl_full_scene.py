"""
CameraCtrl - the full p39-40 scene (ManimGL), one continuous take:

    manimgl <path to this file> CameraCtrlFull

  beat 1  Stable Video Diffusion: the model block draws in, with an example input image and output video
  beat 2  CameraCtrl: camera icon + conditioning arrows into every block, title
  beat 3  TRAINING: the model block shrinks; the camera icon is replaced by a live 3D camera path (upper left),
          conditioning frame on the left, ground-truth clip on the right, cycling through RealEstate10K examples
  beat 4  INFERENCE: same layout, one conditioning image after the other (outputs/ then outputs_empty_office/),
          each rendered along several requested camera paths

All 2D artwork is redrawn from the p39_40-0x.svg mockups using their exact coordinates (1920x1080 viewBox mapped
onto manim's 8-unit-tall frame), so proportions match the mockups. Everything 2D is fixed in frame; only the
camera pyramids live in 3D, and the scene camera is steered so that 3D world lands in the upper-left corner.

Toggles are class attributes on CameraCtrlFull (show_text, video_on_pyramid, which clips / paths, timings).
"""
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from manimlib import *
from scipy.spatial.transform import Rotation, Slerp

# ----------------------------------------------------------------------------- config
HERE = Path(__file__).resolve().parent
REPO = Path('/Users/stephen/Library/CloudStorage/Dropbox-Stephencwelch/welch_labs/world_models/hackin/CameraCtrl')
if not REPO.exists():          # e.g. on the Linux box: fall back to the repo this file sits in
    REPO = HERE.parent
FRAME_CACHE = REPO / "manim/_frames"
SAMPLES = REPO / "data" / "re10k_samples"
GRAPHICS = REPO.parent.parent / "graphics"      # .../welch_labs/world_models/graphics

# Stable Video Diffusion example (beat 1 + 2). If the named files are missing, any other
# "stable_video_diffusion_all_examples copy*.png" in GRAPHICS is used: first = input image, next three = video.
SVD_IMAGE = "stable_video_diffusion_all_examples copy.png"
SVD_VIDEO = ["stable_video_diffusion_all_examples copy 2.png",
             "stable_video_diffusion_all_examples copy 3.png",
             "stable_video_diffusion_all_examples copy 4.png"]

CHILL_BROWN = "#948979"
LINE_COLOR = "#948678"     # stroke / text color in the mockups
WL_YELLOW = "#ffd35a"
WL_BLUE = "#65c8d0"
WL_GREEN = "#00a14b"
WL_RED = "#e2584d"
WL_PURPLE = "#b38cdb"
WL_ORANGE = "#f39c4f"
PATH_COLORS = [WL_YELLOW, WL_BLUE, WL_RED, WL_GREEN, WL_PURPLE, WL_ORANGE]

FONT = "Open Sans"

CV2M = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # OpenCV world -> manim world

# ----------------------------------------------------------------------------- mockup coordinates
PX = FRAME_HEIGHT / 1080            # manim units per mockup pixel
STROKE_PER_PX = 1 / 2.03            # manim stroke_width that renders 1 mockup pixel wide (measured at 1080p)


def P(x, y):
    """Mockup pixel (1920x1080, y down) -> manim point."""
    return np.array([(x - 960) * PX, (540 - y) * PX, 0.0])


def svg_path(d, stroke_px=2.0, color=LINE_COLOR):
    """Tiny SVG path reader (M L H V C Z, absolute or relative) -> VMobject in manim coordinates."""
    tokens = re.findall(r"[MmLlHhVvCcZz]|-?(?:\d+\.?\d*|\.\d+)", d)
    mob = VMobject()
    pos, start, cmd, i = np.zeros(2), np.zeros(2), None, 0

    def nums(n):
        nonlocal i
        vals = [float(t) for t in tokens[i:i + n]]
        i += n
        return vals

    while i < len(tokens):
        if re.match(r"[A-Za-z]", tokens[i]):
            cmd = tokens[i]; i += 1
            if cmd in "Zz":
                mob.add_line_to(P(*start)); pos = start.copy()
                continue
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            x, y = nums(2); pos = pos + [x, y] if rel else np.array([x, y]); start = pos.copy()
            mob.start_new_path(P(*pos)); cmd = "l" if rel else "L"
        elif c == "L":
            x, y = nums(2); pos = pos + [x, y] if rel else np.array([x, y]); mob.add_line_to(P(*pos))
        elif c == "H":
            x, = nums(1); pos = np.array([pos[0] + x if rel else x, pos[1]]); mob.add_line_to(P(*pos))
        elif c == "V":
            y, = nums(1); pos = np.array([pos[0], pos[1] + y if rel else y]); mob.add_line_to(P(*pos))
        elif c == "C":
            v = np.array(nums(6)).reshape(3, 2)
            pts = v + pos if rel else v
            mob.add_cubic_bezier_curve_to(P(*pts[0]), P(*pts[1]), P(*pts[2])); pos = pts[2]
    mob.set_stroke(color, stroke_px * STROKE_PER_PX).set_fill(opacity=0)
    return mob


def svg_polygon(points, color=LINE_COLOR):
    """Filled polygon from an SVG points string (used for the arrowheads)."""
    v = np.array([float(t) for t in points.replace(",", " ").split()]).reshape(-1, 2)
    return Polygon(*[P(*p) for p in v]).set_fill(color, 1).set_stroke(color, 0)


def svg_rect(x, y, w, h, r, stroke_px=3.0, color=LINE_COLOR):
    rect = RoundedRectangle(width=w * PX, height=h * PX, corner_radius=r * PX)
    rect.move_to(P(x + w / 2, y + h / 2))
    return rect.set_stroke(color, stroke_px * STROKE_PER_PX).set_fill(opacity=0)


_CAP_REF = {}


def svg_text(string, size_px, bold=True, color=LINE_COLOR):
    """Text set so its cap height matches Open Sans at `size_px` in the mockup (cap height = 0.714 em)."""
    weight = BOLD if bold else NORMAL
    if weight not in _CAP_REF:
        _CAP_REF[weight] = Text("H", font=FONT, weight=weight, font_size=48).get_height()
    t = Text(string, font=FONT, weight=weight, font_size=48, alignment="LEFT")
    t.scale(0.714 * size_px * PX / _CAP_REF[weight])
    return t.set_color(color)


def at_baseline(mob, x, y, descent_px=0.0):
    """Put a text's left edge at x and its baseline at y (mockup pixels). descent_px: how far glyphs dip below it."""
    return mob.move_to(P(x, y + descent_px), aligned_edge=DL)


def load_image(path, **kwargs):
    """ImageMobject that doesn't keep its file handle open (matters with a few hundred frames)."""
    im = ImageMobject(str(path), **kwargs)
    try:
        im.image.load()
    except Exception:
        pass
    return im


# ----------------------------------------------------------------------------- data
# def ensure_frames(mp4):
#     """Unpack an mp4 to PNGs (cached). Returns sorted list of frame paths."""
#     mp4 = Path(mp4)
#     out = FRAME_CACHE / f"{mp4.parent.name}__{mp4.stem}"   # folder in the key: output sets share file names
#     if not out.exists() or not any(out.glob("*.png")):
#         out.mkdir(parents=True, exist_ok=True)
#         subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp4), "-vsync", "0",
#                         "-start_number", "0", str(out / "%03d.png")], check=True)
#     return sorted(out.glob("*.png"))

def ensure_frames(mp4):
    """Unpack an mp4 to PNGs (cached). Returns sorted list of frame paths."""
    mp4 = Path(mp4)
    out = FRAME_CACHE / f"{mp4.parent.name}__{mp4.stem}"   # folder in the key: output sets share file names
    if not out.exists() or not any(out.glob("*.png")):
        out.mkdir(parents=True, exist_ok=True)
        # "one output frame per input frame" is spelled -fps_mode on current ffmpeg and -vsync on old ones
        for sync in (["-fps_mode", "passthrough"], ["-vsync", "0"], []):
            for stale in out.glob("*.png"):
                stale.unlink()
            r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp4), *sync,
                                "-start_number", "0", str(out / "%03d.png")], capture_output=True, text=True)
            if r.returncode == 0 and any(out.glob("*.png")):
                break
        else:
            raise RuntimeError(f"ffmpeg could not unpack {mp4}:\n{r.stderr}")
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


def load_inference_run(name, outputs_dir="outputs"):
    """name like 'synth:orbit_right' or 're10k:0f47577ab3441480' -> c2w, fxfy, frames (or None if no video yet)."""
    kind, key = name.split(":")
    mp4 = REPO / outputs_dir / f"{kind}_{key}.mp4"
    if not mp4.exists():
        return None
    if kind == "synth":
        c2w, fxfy = synthetic_path(key)
    else:
        c2w, fxfy = load_pose_txt(REPO / "assets" / "pose_files" / f"{key}_svdxt.txt")
    return c2w, fxfy, ensure_frames(mp4)


def svd_example_paths():
    """-> (input image path or None, [up to 3 output frame paths])"""
    image = GRAPHICS / SVD_IMAGE
    video = [GRAPHICS / v for v in SVD_VIDEO]
    if image.exists() and all(v.exists() for v in video):
        return image, video
    found = sorted(GRAPHICS.glob("stable_video_diffusion_all_examples copy*.png"), key=lambda p: (len(p.name), p.name))
    if not found:
        print(f"[CameraCtrlFull] no SVD example images found in {GRAPHICS} - drawing placeholders")
        return None, []
    return found[0], found[1:4]


# ----------------------------------------------------------------------------- 3D geometry
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

    def center(self, depth=0.0):
        """Middle of everything the path draws (camera positions + image planes `depth` ahead)."""
        pts = [self.pos] + [np.array([self.pose(i)["pos"] + depth * self.pose(i)["fwd"] for i in range(self.n)])]
        pts = np.concatenate(pts)
        return (pts.min(axis=0) + pts.max(axis=0)) / 2


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
        self.images = [load_image(p) for p in frame_paths]
        for im in self.images:
            im.signs = np.sign(im.get_points()[:, :2]).copy()
            im.apply_depth_test()
        super().__init__(*self.images)

    def show(self, idx, pose, visible=True, opacity=1.0):
        c, hw, hh = image_plane(pose, self.depth)
        for i, im in enumerate(self.images):
            if visible and i == idx:
                pts = c + im.signs[:, :1] * hw * pose["right"] + im.signs[:, 1:] * hh * pose["up"]
                im.set_opacity(opacity)
            else:  # collapse instead of fading: an invisible quad would still write depth
                pts = np.tile(c, (len(im.signs), 1))
            im.set_points(pts)
        return self


class CameraRig:
    """Moving frustum + (optionally) video on its image plane + ghost pyramids at the sampled poses + path line.

    ValueTrackers:
      t      camera position along the path, 0 .. n-1 (frame index)
      ghost  how many ghost pyramids have been revealed, 0 .. n   (only used when follow_ghosts=False)
      alpha  overall opacity of the rig, for fading it in and out
    """

    def __init__(self, traj, frame_paths, color=WL_YELLOW, depth=0.9, ghost_every=1, ghost_opacity=0.35,
                 ghost_width=1.5, follow_ghosts=True, show_video=True):
        self.traj, self.n = traj, traj.n
        self.color = color
        self.t = ValueTracker(0)
        self.ghost = ValueTracker(0)
        self.alpha = ValueTracker(1)
        self.follow_ghosts = follow_ghosts  # True: ghosts appear as the camera passes (training)
        self.ghost_opacity = ghost_opacity

        self.frustum = Frustum(color, depth)
        self.ghost_ids = list(range(0, self.n, ghost_every))
        self.ghosts = VGroup(*[Frustum(color, depth, ghost_width, ghost_opacity) for _ in self.ghost_ids])
        self.path = VMobject().set_stroke(color, 2.5, 0.9)
        self.path.apply_depth_test()
        parts = [self.path, self.ghosts]
        self.screen = None
        if show_video:
            self.screen = VideoScreen(frame_paths, depth)
            self.screen.add_updater(lambda m: m.show(self.frame_index(), traj.pose(self.t.get_value()),
                                                     self.alpha.get_value() > 0.02, self.alpha.get_value()))
            parts.append(self.screen)
        parts.append(self.frustum)

        self.frustum.add_updater(lambda m: m.set_pose(traj.pose(self.t.get_value())).set_stroke(opacity=self.alpha.get_value()))
        self.ghosts.add_updater(self.update_ghosts)
        self.path.add_updater(self.update_path)
        self.mobjects = Group(*parts)
        self.mobjects.update()

    def frame_index(self):
        return int(np.clip(round(self.t.get_value()), 0, self.n - 1))

    def progress(self):
        return self.t.get_value() + 1 if self.follow_ghosts else self.ghost.get_value()

    def update_ghosts(self, ghosts):
        p, a = self.progress(), self.alpha.get_value()
        for g, i in zip(ghosts, self.ghost_ids):
            grow = float(np.clip(p - i, 0, 1))
            g.set_pose(self.traj.pose(i), grow=grow)
            g.set_stroke(opacity=self.ghost_opacity * a if grow > 0 else 0)

    def update_path(self, path):
        path.set_stroke(opacity=0.9 * self.alpha.get_value())
        end = float(np.clip(self.progress() - 1, 0, self.n - 1))
        if not self.traj.has_translation or end < 0.05:
            path.set_points_as_corners([self.traj.pos[0], self.traj.pos[0] + 1e-4 * UP])
            return
        ts = np.append(np.arange(0, end, 0.25), end)
        path.set_points_as_corners([self.traj.pose(t)["pos"] for t in ts])


class CornerView:
    """Steers the scene camera so a 3D point stays at a fixed spot on screen (while slowly orbiting it).

    Everything 2D in the scene is fixed in frame, so moving the camera only affects the 3D pyramids:
    this is what puts the little 3D world in the upper-left corner."""

    def __init__(self, frame, screen_point, height=11.0, theta=-28, phi=68, orbit_deg_per_sec=1.5):
        self.frame, self.screen_point = frame, np.array(screen_point, dtype=float)
        self.height, self.theta, self.phi, self.rate = height, theta, phi, orbit_deg_per_sec
        self.target = np.zeros(3)
        frame.add_updater(self.update)

    def look_at(self, target):
        self.target = np.array(target, dtype=float)
        self.frame.reorient(self.theta, self.phi, 0, ORIGIN, self.height)
        self.update(self.frame, 0)

    def update(self, frame, dt):
        frame.increment_theta(dt * self.rate * DEGREES)
        offset = frame.get_orientation().apply(self.screen_point * self.height / FRAME_HEIGHT)
        frame.move_to(self.target - offset)


# ----------------------------------------------------------------------------- 2D pieces
class VideoPanel(Group):
    """Flat, fixed-in-frame video: stacked frames (one visible) + border. `alpha` fades the whole panel."""

    def __init__(self, frame_paths, width, center, color=LINE_COLOR, border_px=2.0):
        self.images = [load_image(p) for p in frame_paths]
        for im in self.images:
            im.set_width(width).move_to(center)
        self.border = SurroundingRectangle(self.images[0], buff=0).set_stroke(color, border_px * STROKE_PER_PX)
        self.border.set_fill(opacity=0)
        self.alpha = ValueTracker(1)
        self.index = 0
        super().__init__(*self.images, self.border)
        self.fix_in_frame()
        self.add_updater(lambda m: m.refresh())
        self.refresh()

    def show(self, idx):
        self.index = int(np.clip(idx, 0, len(self.images) - 1))
        return self

    def refresh(self):
        a = self.alpha.get_value()
        for i, im in enumerate(self.images):
            im.set_opacity(a if i == self.index else 0.0)
        self.border.set_stroke(opacity=a)
        return self


# Mockup p39_40-02.svg (the "CameraCtrl (2024)" frame) is the master for the big diagram; -03 / -04 for the small one.
BOXES = [(724.67, 402.89, 284.99), (815.12, 428.24, 236.58), (905.57, 452.39, 186.01),
         (996.02, 452.39, 186.01), (1086.48, 428.24, 236.58), (1176.93, 402.89, 284.99)]   # x, y, height (width 63.98)
BOX_W, BOX_R = 63.98, 6.62
BUS_DROPS = [  # conditioning arrows, left to right: (path, arrowhead)
    ("M699.29,239.73h46.13c6.21,0,11.24,5.03,11.24,11.24v122.3", "756.65 383.13 750.98 369.24 756.65 372.53 762.33 369.24 756.65 383.13", 2.15),
    ("M756.65,239.73h79.21c6.21,0,11.24,5.03,11.24,11.24v142.75", "847.11 402.89 841.82 389.96 847.11 393.03 852.39 389.96 847.11 402.89", 2.0),
    ("M699.29,239.73h227.03c6.21,0,11.24,5.03,11.24,11.24v168.09", "937.56 428.24 932.28 415.3 937.56 418.37 942.84 415.3 937.56 428.24", 2.0),
    ("M915.51,239.73h101.26c6.21,0,11.24,5.03,11.24,11.24v168.09", "1028.01 428.24 1022.73 415.3 1028.01 418.37 1033.3 415.3 1028.01 428.24", 2.0),
    ("M1007.24,239.73h99.98c6.21,0,11.24,5.03,11.24,11.24v142.75", "1118.47 402.89 1113.18 389.96 1118.47 393.03 1123.75 389.96 1118.47 402.89", 2.0),
    ("M1094.63,239.73h103.05c6.21,0,11.24,5.03,11.24,11.24v122.99", "1208.92 383.13 1203.63 370.2 1208.92 373.27 1214.2 370.2 1208.92 383.13", 2.0),
]
SMALL_SCALE = 50.47 / 63.98                      # block width in -03 / block width in -02
BIG_CENTER, SMALL_CENTER = (982.79, 545.385), (971.54, 546.54)   # center of the six blocks, big / small


def build_diagram(show_text=True):
    """-> dict of fixed-in-frame pieces at their big (mockup -02) positions."""
    d = {}
    d["boxes"] = VGroup(*[svg_rect(x, y, BOX_W, h, BOX_R) for x, y, h in BOXES])
    d["arrow_in"] = VGroup(Line(P(656.52, 546.53), P(700.09, 546.53)).set_stroke(LINE_COLOR, 2 * STROKE_PER_PX),
                           svg_polygon("709.27 546.53 696.33 551.81 699.4 546.53 696.33 541.24"))
    d["arrow_out"] = VGroup(Line(P(1252.29, 545.39), P(1295.86, 545.39)).set_stroke(LINE_COLOR, 2 * STROKE_PER_PX),
                            svg_polygon("1305.04 545.39 1292.1 550.67 1295.17 545.39 1292.1 540.1"))
    d["bus"] = VGroup(*[VGroup(svg_path(path, px), svg_polygon(head)) for path, head, px in BUS_DROPS])
    # camera icon: body is the mockup's rotated rect (x 574.66-652.36, y 218.16-261.29), lens is its path
    d["camera"] = VGroup(svg_rect(574.66, 218.16, 77.7, 43.13, 2.84),
                         svg_path("M652.36,224.02l24.56-7.83c2.96-.94,5.98,1.27,5.98,4.37v38.33c0,3.14-3.08,5.35-6.05,4.35l-24.4-8.23", 3.0))
    if show_text:
        d["model_label"] = at_baseline(svg_text("STABLE VIDEO DIFFUSION MODEL", 28), 738.65, 763.53)
        d["image_label"] = at_baseline(svg_text("IMAGE", 28), 320.64, 728.92)
        d["video_label"] = at_baseline(svg_text("VIDEO", 28), 1566.82, 743.85)
        d["title_cameractrl"] = svg_text("CameraCtrl (2024)", 42)
        d["title_cameractrl"].move_to(P(960, 94.01 - 0.26 * 42))
        d["title_training"] = at_baseline(svg_text("TRAINING", 42), 856.06, 90.74)
        d["title_inference"] = at_baseline(svg_text("INFERENCE", 42), 844.85, 90.74)
        d["cite_svd"] = svg_text('Blattmann, Andreas, et al. "Stable video diffusion:\nScaling latent video diffusion models to large\n'
                                 'datasets." arXiv preprint arXiv:2311.15127 (2023).', 16, bold=False)
        d["cite_svd"].move_to(P(1494.32, 1007.43 + 3.27 - 0.76 * 16), aligned_edge=UL)
        d["cite_cameractrl"] = svg_text('He, Hao, et al. "Cameractrl: Enabling camera control for\n'
                                        'text-to-video generation." arXiv preprint arXiv:2404.02101 (2024).', 16, bold=False)
        d["cite_cameractrl"].move_to(P(1401.01, 1022.86 - 0.76 * 16), aligned_edge=UL)
    for m in d.values():
        m.fix_in_frame()
    return d


def placed_image(path, x, y, w, h, scale, placeholder_color=CHILL_BROWN):
    """Image at the mockup's <image> placement (top-left x, y; native w x h; scale). Falls back to a grey card."""
    if path is not None:
        im = load_image(path)
        im.set_height(h * scale * PX)
    else:
        im = Rectangle(width=w * scale * PX, height=h * scale * PX).set_fill(placeholder_color, 0.35).set_stroke(LINE_COLOR, 1)
    im.move_to(P(x, y), aligned_edge=UL)
    im.fix_in_frame()
    return im


# ----------------------------------------------------------------------------- the scene
class CameraCtrlFull(InteractiveScene):
    # --- toggles -------------------------------------------------------------------------------------------
    show_text = True             # titles, labels and citations as manim Text (Open Sans). False: add them in post
    video_on_pyramid = True      # play the clip on the moving pyramid's image plane
    # --- training beat -------------------------------------------------------------------------------------
    train_clip_indices = [10, 3, 17, 22, 31, 45]   # indices into available_training_examples()
    train_clip_ids = None        # or name clips directly, e.g. ["015cc846a79bd3bc", ...] (overrides the indices)
    train_run_time = 4.5         # seconds per clip
    train_extent = 3.5           # manim units each training path is scaled to
    # --- inference beat ------------------------------------------------------------------------------------
    inference_sets = ["outputs", "outputs_empty_office"]        # one folder of notebook outputs per conditioning image
    inference_runs = ["synth:orbit_right", "synth:truck_right", "synth:dolly_in"]
    inference_scale = 1.7        # manim units per pose unit (shared, so the paths are comparable)
    layout_time = 1.2            # seconds to lay out the requested path as ghost pyramids
    inference_run_time = 4.5     # seconds to fly a path
    # --- 3D corner view ------------------------------------------------------------------------------------
    path_screen_px = (400, 300)  # where the middle of the 3D path sits, in mockup pixels
    view_height = 12.0           # 3D zoom: bigger = smaller pyramids
    pyramid_depth = 0.9
    ghost_every = 2              # a ghost pyramid at every Nth of the 25 poses
    rig_color = WL_YELLOW
    # --- flat panels ---------------------------------------------------------------------------------------
    panel_width_px = 520
    panel_gap_px = 24            # gap between the panels and the top of the (higher) VIDEO label
    panel_center_x_px = (373, 1612)   # left (conditioning frame), right (video)
    hold = 0.6                   # pause after each clip / path

    def construct(self):
        self.d = build_diagram(self.show_text)
        self.beat_svd()
        self.beat_cameractrl()
        self.beat_to_training_layout()
        self.beat_training()
        self.beat_inference()
        self.wait(20)

    # small helpers so every beat also works with show_text = False
    def has(self, key):
        return key in self.d

    def anims(self, cls, *keys, **kwargs):
        return [cls(self.d[k], **kwargs) for k in keys if k in self.d]

    # --------------------------------------------------------------------------------------------- beat 1
    def beat_svd(self):
        d = self.d
        image_path, video_paths = svd_example_paths()
        video_paths = list(video_paths) + [None] * (3 - len(video_paths))
        self.svd_image = placed_image(image_path, 128.51, 404.39, 515, 291, 0.95)
        stack = [(1337.07, 368.13, 519), (1373.28, 408.77, 517), (1412.75, 460.23, 516)]     # back to front
        self.svd_video = Group(*[placed_image(p, x, y, w, 291, 0.8132) for p, (x, y, w) in zip(video_paths, stack)])

        self.wait(0.5)
        self.play(LaggedStart(*[ShowCreation(b) for b in d["boxes"]], lag_ratio=0.15), run_time=2.0)
        self.play(*self.anims(Write, "model_label"), *self.anims(FadeIn, "cite_svd"), run_time=1.0)
        self.play(FadeIn(self.svd_image, shift=0.2 * UP), *self.anims(FadeIn, "image_label"), run_time=0.8)
        self.play(ShowCreation(d["arrow_in"][0]), FadeIn(d["arrow_in"][1]), run_time=0.5)
        self.play(ShowCreation(d["arrow_out"][0]), FadeIn(d["arrow_out"][1]), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(im, shift=0.15 * UP) for im in self.svd_video], lag_ratio=0.3),
                  *self.anims(FadeIn, "video_label"), run_time=1.2)
        self.wait(1.5)

    # --------------------------------------------------------------------------------------------- beat 2
    def beat_cameractrl(self):
        d = self.d
        self.play(*self.anims(FadeIn, "title_cameractrl", shift=0.2 * DOWN),
                  *self.anims(FadeOut, "cite_svd"), *self.anims(FadeIn, "cite_cameractrl"),
                  ShowCreation(d["camera"]), run_time=1.0)
        self.play(LaggedStart(*[AnimationGroup(ShowCreation(drop[0]), FadeIn(drop[1], rate_func=squish_rate_func(smooth, 0.7, 1)))
                                for drop in d["bus"]], lag_ratio=0.12), run_time=2.0)
        self.wait(1.5)

    # --------------------------------------------------------------------------------------------- beat 2 -> 3
    def beat_to_training_layout(self):
        """Shrink the model block to its mockup -03 size, clear the SVD example, swap the camera icon for 3D."""
        d = self.d
        block = Group(d["boxes"], d["arrow_in"], d["arrow_out"], d["bus"], *[d[k] for k in ["model_label"] if k in d])
        target = block.copy()
        target.scale(SMALL_SCALE, about_point=P(*BIG_CENTER)).shift(P(*SMALL_CENTER) - P(*BIG_CENTER))
        for sm in target.family_members_with_points():
            if isinstance(sm, VMobject) and sm.get_stroke_width() > 0:
                sm.set_stroke(width=float(sm.get_stroke_width()) * SMALL_SCALE)
        moves = [Transform(block, target)]
        if self.has("image_label"):
            moves += [d["image_label"].animate.move_to(P(320.46, 889.66), aligned_edge=DL),
                      d["video_label"].animate.move_to(P(1566.65, 877.82), aligned_edge=DL)]
        fades = [FadeOut(self.svd_image), FadeOut(self.svd_video), FadeOut(d["camera"])]
        if self.has("title_cameractrl"):
            fades += [FadeOut(d["title_cameractrl"], shift=0.2 * UP), FadeIn(d["title_training"], shift=0.2 * UP)]
        self.play(*moves, *fades, run_time=1.5)

        # panel geometry for the training / inference beats (each panel sits just above its label)
        w, h = self.panel_width_px, self.panel_width_px * 320 / 576
        label_tops = (889.66 - 0.714 * 28, 877.82 - 0.714 * 28)
        self.panel_width = w * PX
        center_y = min(label_tops) - self.panel_gap_px - h / 2        # same height for both panels
        self.panel_centers = [P(cx, center_y) for cx in self.panel_center_x_px]

        sx, sy = self.path_screen_px
        self.view = CornerView(self.frame, P(sx, sy), self.view_height)

    # --------------------------------------------------------------------------------------------- shared
    def still_panel(self, frame_path):
        panel = VideoPanel([frame_path], self.panel_width, self.panel_centers[0])
        return panel

    def run_path(self, traj, frames, left_panel=None, layout_first=False, run_time=4.5):
        """One camera path: 3D rig in the corner + the clip on the right. Fades itself in and out.

        left_panel: a still to fade in/out with this path (training); None = the caller manages the left panel."""
        rig = CameraRig(traj, frames, self.rig_color, self.pyramid_depth, self.ghost_every,
                        follow_ghosts=not layout_first, show_video=self.video_on_pyramid)
        right = VideoPanel(frames, self.panel_width, self.panel_centers[1])
        right.add_updater(lambda m: m.show(rig.frame_index()))
        faders = [rig.alpha, right.alpha] + ([left_panel.alpha] if left_panel is not None else [])
        for a in faders:
            a.set_value(0)
        self.view.look_at(traj.center(self.pyramid_depth))
        self.add(rig.mobjects, right, *([left_panel] if left_panel is not None else []))

        self.play(*[a.animate.set_value(1) for a in faders], run_time=0.5)
        if layout_first:      # inference: the requested path comes first, as empty pyramids
            self.play(rig.ghost.animate.set_value(rig.n), run_time=self.layout_time, rate_func=linear)
        self.play(rig.t.animate.set_value(rig.n - 1), run_time=run_time, rate_func=linear)
        self.wait(self.hold)
        self.play(*[a.animate.set_value(0) for a in faders], run_time=0.4)
        self.remove(rig.mobjects, right, *([left_panel] if left_panel is not None else []))
        for m in [*rig.mobjects, rig.mobjects, right]:
            m.clear_updaters()

    # --------------------------------------------------------------------------------------------- beat 3
    def beat_training(self):
        ids = self.train_clip_ids
        if ids is None:
            avail = available_training_examples()
            ids = [avail[i % len(avail)] for i in self.train_clip_indices]
        for clip_id in ids:
            c2w, fxfy, frames = load_training_example(clip_id)
            traj = Trajectory(c2w, fxfy, extent=self.train_extent)
            self.run_path(traj, frames, left_panel=self.still_panel(frames[0]), run_time=self.train_run_time)

    # --------------------------------------------------------------------------------------------- beat 4
    def beat_inference(self):
        d = self.d
        if self.has("title_training"):
            self.play(FadeOut(d["title_training"], shift=0.2 * UP), FadeIn(d["title_inference"], shift=0.2 * UP), run_time=0.8)
        for outputs_dir in self.inference_sets:
            runs = [(name, load_inference_run(name, outputs_dir)) for name in self.inference_runs]
            runs = [(name, run) for name, run in runs if run is not None]
            if not runs:
                print(f"[CameraCtrlFull] nothing to show from {outputs_dir}/ - skipping")
                continue
            # the conditioning image stays on the left for all of this set's paths (frame 0 of the first run)
            left = self.still_panel(runs[0][1][2][0])
            left.alpha.set_value(0)
            self.add(left)
            self.play(left.alpha.animate.set_value(1), run_time=0.6)
            for name, (c2w, fxfy, frames) in runs:
                traj = Trajectory(c2w, fxfy, scale=self.inference_scale)
                self.run_path(traj, frames, layout_first=True, run_time=self.inference_run_time)
            self.play(left.alpha.animate.set_value(0), run_time=0.5)
            self.remove(left)
            left.clear_updaters()
