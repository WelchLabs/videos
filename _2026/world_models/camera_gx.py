"""Shared CameraCtrl assets, trajectories, and ManimGX scene elements."""
from welch_paths import welch_path

import os
from pathlib import Path
import subprocess

import manimgx as m
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from assets import graphics_root
from welch_scene import camera_view, image_plane_corners, place_image_plane, set_image_pixels
from welch_storyboard import CAP_HEIGHT, LABEL_FONT as FONT, PX, STROKE_PER_PX, px_box, px_line, px_text
from welch_storyboard import storyboard_point as P, svg_path as parse_svg_path
from welch_style import CHILL_BROWN, YELLOW as WL_YELLOW, BLUE as WL_BLUE, GREEN as WL_GREEN

LINE_COLOR = "#948678"
PATH_COLORS = [WL_YELLOW, WL_BLUE, "#e2584d", WL_GREEN, "#b38cdb", "#f39c4f"]
CV2M = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])

SVD_IMAGE = "stable_video_diffusion_all_examples copy.png"
SVD_VIDEO = [f"stable_video_diffusion_all_examples copy {i}.png" for i in range(2, 5)]


def svg_path(path, stroke_px=2.0, color=LINE_COLOR):
    return parse_svg_path(path, P).set_stroke(color, stroke_px * STROKE_PER_PX).set_fill(opacity=0)


def frame_cache_dir(mp4, cache):
    return cache / f"{mp4.parent.name}__{mp4.stem}"


def cached_frames(mp4, cache):
    candidates = [frame_cache_dir(mp4, cache)]
    if mp4.parent.name in {"outputs", "reference_videos"}:
        candidates.append(cache / mp4.stem)
    for folder in candidates:
        frames = sorted(folder.glob("*.png"))
        if frames:
            return frames
    return []


def ensure_frames(mp4, cache):
    """Return cached PNGs, extracting a source MP4 when the cache is missing."""
    mp4 = Path(mp4)
    frames = cached_frames(mp4, cache)
    if frames:
        return frames
    out = frame_cache_dir(mp4, cache)
    if not mp4.is_file():
        raise FileNotFoundError(f"Missing CameraCtrl video: {mp4}")

    out.mkdir(parents=True, exist_ok=True)
    # Keep one PNG per input frame with both new and older versions of ffmpeg.
    for sync in (["-fps_mode", "passthrough"], ["-vsync", "0"]):
        result = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(mp4), *sync,
             "-start_number", "0", str(out / "%03d.png")],
            capture_output=True,
            text=True,
        )
        frames = sorted(out.glob("*.png"))
        if result.returncode == 0 and frames:
            return frames
        for partial in frames:
            partial.unlink()
    raise RuntimeError(f"ffmpeg could not unpack {mp4}:\n{result.stderr}")


def load_pose_txt(path):
    """RealEstate10K txt -> (c2w relative to frame 0 [N,4,4], intrinsics [N,2] normalized fx, fy)."""
    rows = [[float(x) for x in l.split()] for l in Path(path).read_text().strip().splitlines()[1:]]
    rows = np.array(rows)
    w2c = np.tile(np.eye(4), (len(rows), 1, 1))
    w2c[:, :3, :] = rows[:, 7:19].reshape(-1, 3, 4)
    c2w = w2c[0][None] @ np.linalg.inv(w2c)
    return c2w, rows[:, 1:3]


REF_FX, REF_FY = 0.50765091, 0.90249053  # intrinsics the notebook's synthetic paths borrow


def synthetic_path(kind, n=25, amount=1.2, angle_deg=35):
    """Use the same camera motions as make_synthetic in the CameraCtrl notebook."""
    c2ws, fxfy = [], []
    for progress in np.linspace(0.0, 1.0, n):
        s = 0.5 - 0.5 * np.cos(np.pi * progress)
        a = angle_deg * s
        rotation = np.eye(3)
        translation = np.zeros(3)
        fx, fy = REF_FX, REF_FY
        if kind == "dolly_in":
            translation[2] = amount * s
        elif kind == "dolly_out":
            translation[2] = -amount * s
        elif kind == "truck_right":
            translation[0] = amount * s
        elif kind == "pedestal_up":
            translation[1] = -amount * s
        elif kind == "pan_right":
            rotation = Rotation.from_euler("y", a, degrees=True).as_matrix()
        elif kind == "tilt_up":
            rotation = Rotation.from_euler("x", -a, degrees=True).as_matrix()
        elif kind == "orbit_right":
            rotation = Rotation.from_euler("y", -a, degrees=True).as_matrix()
            pivot = np.array([0, 0, 2.0 * amount])
            translation = pivot - rotation @ pivot
        elif kind == "zoom_in":
            fx, fy = REF_FX * (1 + 0.6 * s), REF_FY * (1 + 0.6 * s)
        elif kind == "spiral":
            rotation = Rotation.from_euler("y", a * 0.5, degrees=True).as_matrix()
            translation = np.array([amount * np.sin(np.pi * s) * 0.6, -0.3 * amount * s, 0.5 * amount * s])
        else:
            raise ValueError(f"Unknown CameraCtrl path: {kind}")

        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        c2ws.append(pose)
        fxfy.append([fx, fy])
    return np.array(c2ws), np.array(fxfy)


class CameraCtrlAssets:
    """Files shared by both CameraCtrl scenes and the individual training and inference demos."""

    def __init__(self, repo=None):
        self.repo = Path(
            repo or os.environ.get("CAMERACTRL_ROOT") or welch_path("world_models/hackin/CameraCtrl")
        ).expanduser()
        self.samples = self.repo / "data" / "re10k_samples"
        self.cache = self.repo / "manim" / "_frames"
        self.graphics = graphics_root(self.repo.parent.parent / "graphics")

    def load_training_example(self, clip_id):
        """-> c2w [25,4,4], fxfy [25,2], frame paths [25]"""
        d = self.samples / clip_id
        frames = sorted((d / "frames").glob("*.png"))
        if frames and (d / "poses.npz").is_file():
            with np.load(d / "poses.npz") as z:
                return z["c2w_sample"], z["intrinsics_sample"][:, :2], frames
        c2w, fxfy = load_pose_txt(self.repo / "assets" / "pose_files" / f"{clip_id}_svdxt.txt")
        return c2w, fxfy, ensure_frames(self.repo / "assets" / "reference_videos" / f"{clip_id}_svdxt.mp4", self.cache)

    def available_training_examples(self):
        if self.samples.exists():
            ids = sorted(p.parent.name for p in self.samples.glob("*/poses.npz")
                         if any((p.parent / "frames").glob("*.png")))
            if ids:
                return ids
        return sorted(p.name.replace("_svdxt.txt", "") for p in (self.repo / "assets" / "pose_files").glob("*_svdxt.txt"))

    def load_inference_run(self, name, outputs_dir="outputs"):
        """name like 'synth:orbit_right' or 're10k:0f47577ab3441480' -> c2w, fxfy, frames (or None if no video yet)."""
        kind, key = name.split(":")
        mp4 = self.repo / outputs_dir / f"{kind}_{key}.mp4"
        frames = cached_frames(mp4, self.cache)
        if not frames and not mp4.is_file():
            return None
        if kind == "synth":
            c2w, fxfy = synthetic_path(key)
        else:
            c2w, fxfy = load_pose_txt(self.repo / "assets" / "pose_files" / f"{key}_svdxt.txt")
        return c2w, fxfy, frames or ensure_frames(mp4, self.cache)

    def svd_example_paths(self):
        """Return an input path and up to three output frame paths."""
        image = self.graphics / SVD_IMAGE
        video = [self.graphics / v for v in SVD_VIDEO]
        if image.exists() and all(v.exists() for v in video):
            return image, video
        found = sorted(self.graphics.glob("stable_video_diffusion_all_examples copy*.png"), key=lambda p: (len(p.name), p.name))
        if not found:
            print(f"[CameraCtrl] no SVD example images found in {self.graphics} - drawing placeholders")
            return None, []
        return found[0], found[1:4]


class Trajectory:
    """Camera path in manim coordinates with smooth interpolation between the sampled poses."""

    def __init__(self, c2w, fxfy, scale=None, extent=3.0, max_scale=3.0, offset=m.ORIGIN):
        self.n = len(c2w)
        pos = c2w[:, :3, 3]
        if scale is None:  # fit this path into `extent` manim units (RE10K scene scale is arbitrary per clip)
            reach = np.linalg.norm(pos - pos[0], axis=1).max()
            scale = min(extent / max(reach, 1e-6), max_scale)
        self.scale = scale
        self.pos = (pos * scale) @ CV2M.T + np.array(offset)
        self.fxfy = np.asarray(fxfy, dtype=float)
        self.rotations = Rotation.from_matrix(c2w[:, :3, :3])
        self.slerp = Slerp(np.arange(self.n), self.rotations) if self.n > 1 else None
        self.has_translation = np.linalg.norm(self.pos - self.pos[0], axis=1).max() > 1e-3

    def pose(self, t):
        """t in [0, n-1] (fractional ok) -> dict(pos, right, up, fwd, fx, fy)"""
        t = float(np.clip(t, 0, self.n - 1))
        i = int(np.floor(t))
        j = min(i + 1, self.n - 1)
        a = t - i
        orientation = self.rotations[0] if self.slerp is None else self.slerp(t)
        B = CV2M @ orientation.as_matrix()
        fx, fy = (1 - a) * self.fxfy[i] + a * self.fxfy[j]
        return dict(pos=(1 - a) * self.pos[i] + a * self.pos[j],
                    right=B[:, 0], up=-B[:, 1], fwd=B[:, 2], fx=fx, fy=fy)

    def center(self, depth=0.0):
        """Middle of everything the path draws (camera positions + image planes `depth` ahead)."""
        poses = (self.pose(i) for i in range(self.n))
        plane_centers = np.array([p["pos"] + depth * p["fwd"] for p in poses])
        pts = np.concatenate((self.pos, plane_centers))
        return (pts.min(axis=0) + pts.max(axis=0)) / 2


def image_plane(pose, depth):
    """-> center, half-width, half-height of the image plane `depth` in front of the camera."""
    return pose["pos"] + depth * pose["fwd"], depth * 0.5 / pose["fx"], depth * 0.5 / pose["fy"]


class Frustum(m.VGroup):
    """Wireframe camera pyramid: 4 edges from the apex + the image-plane rectangle."""

    def __init__(self, color=WL_YELLOW, depth=0.9, stroke_width=3, stroke_opacity=1.0):
        self.depth = depth
        self.edges = m.VGroup(*[m.Line(m.ORIGIN, m.RIGHT) for _ in range(4)])
        self.rect = m.VMobject()
        super().__init__(*self.edges, self.rect)
        self.set_stroke(color, stroke_width, stroke_opacity)
        self.set_fill(opacity=0)

    def set_pose(self, pose, grow=1.0):
        d = self.depth * max(grow, 1e-3)
        c, hw, hh = image_plane(pose, d)
        corners = image_plane_corners(c, pose["right"], pose["up"], 2 * hw, 2 * hh)[[0, 1, 3, 2]]
        for line, corner in zip(self.edges, corners):
            line.set_points_as_corners([pose["pos"], corner])
        self.rect.set_points_as_corners([*corners, corners[0]])
        return self


class ClipFrames:
    """Decode each PNG once and share its pixels between the panel and the moving camera."""

    def __init__(self, paths):
        self.paths = tuple(paths)
        if not self.paths:
            raise ValueError("A CameraCtrl clip must have at least one frame")
        self.pixels = {}

    def __len__(self):
        return len(self.paths)

    def texture(self, index):
        index = int(np.clip(index, 0, len(self) - 1))
        if index not in self.pixels:
            with Image.open(self.paths[index]) as photo:
                pixels = np.array(photo.convert("RGBA"))
            pixels.flags.writeable = False
            self.pixels[index] = pixels
        return self.pixels[index]

    def show_on(self, image, index):
        pixels = self.texture(index)
        if image.pixel_array is not pixels:
            set_image_pixels(image, pixels)


class VideoScreen(m.Group):
    """A textured quad on the moving camera's image plane."""

    def __init__(self, frames, depth=0.9, inset=0.985):
        self.depth = depth * inset
        self.frames = frames
        self.image = m.ImageMobject(frames.texture(0))
        super().__init__(self.image)

    def show(self, idx, pose, visible=True, opacity=1.0):
        center, half_width, half_height = image_plane(pose, self.depth)
        if visible:
            self.frames.show_on(self.image, idx)
            place_image_plane(
                self.image, center, pose["right"], pose["up"], 2 * half_width, 2 * half_height,
            )
            self.image.set_opacity(opacity)
        else:
            self.image.set_points(np.tile(center, (4, 1)))
            self.image.set_opacity(0)
        return self


class CameraRig:
    """Moving frustum + (optionally) video on its image plane + ghost pyramids at the sampled poses + path line.

    ValueTrackers:
      t      camera position along the path, 0 .. n-1 (frame index)
      ghost  how many ghost pyramids have been revealed, 0 .. n   (only used when follow_ghosts=False)
      alpha  overall opacity of the rig, for fading it in and out
    """

    def __init__(self, traj, frames, color=WL_YELLOW, depth=0.9, ghost_every=1, ghost_opacity=0.35,
                 ghost_width=1.5, follow_ghosts=True, show_video=True):
        self.traj, self.n = traj, traj.n
        self.frames = frames
        self.color = color
        self.t = m.ValueTracker(0)
        self.ghost = m.ValueTracker(0)
        self.alpha = m.ValueTracker(1)
        self.follow_ghosts = follow_ghosts  # True: ghosts appear as the camera passes
        self.ghost_opacity = ghost_opacity
        self.screen_visible = True

        self.frustum_visible = True
        self.frustum = Frustum(color, depth)
        self.ghost_ids = list(range(0, self.n, ghost_every))
        self.ghosts = m.VGroup(*[Frustum(color, depth, ghost_width, ghost_opacity) for _ in self.ghost_ids])
        self.path = m.VMobject().set_stroke(color, 2.5, 0.9)
        parts = [self.path, self.ghosts]
        self.screen = None
        if show_video:
            self.screen = VideoScreen(frames, depth)
            self.screen.add_updater(self.update_screen)
            parts.append(self.screen)
        parts.append(self.frustum)

        self.frustum.add_updater(self.update_frustum)
        self.ghosts.add_updater(self.update_ghosts)
        self.path.add_updater(self.update_path)
        self.mobjects = m.Group(*parts)
        self.mobjects.update()

    def update_screen(self, screen):
        opacity = self.alpha.get_value()
        screen.show(
            self.frame_index(), self.traj.pose(self.t.get_value()),
            visible=self.screen_visible and opacity > 0.02, opacity=opacity,
        )

    def update_frustum(self, frustum):
        frustum.set_pose(self.traj.pose(self.t.get_value()))
        frustum.set_stroke(opacity=self.alpha.get_value() if self.frustum_visible else 0)

    def frame_index(self):
        return self.frame_at(self.t.get_value())

    def frame_at(self, position):
        progress = float(np.clip(position / max(1, self.n - 1), 0, 1))
        return round(progress * (len(self.frames) - 1))

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
            path.set_points_as_corners([self.traj.pos[0], self.traj.pos[0] + 1e-4 * m.UP])
            return
        ts = np.append(np.arange(0, end, 0.25), end)
        path.set_points_as_corners([self.traj.pose(t)["pos"] for t in ts])


class CornerView:
    """Keep a world point at a fixed screen position while the camera orbits it."""

    def __init__(self, scene, screen_point, height=11.0, theta=-28, phi=68, orbit_deg_per_sec=1.5):
        self.screen_point = np.array(screen_point, dtype=float)
        self.target = np.zeros(3)
        self.camera = scene.camera
        scene.set_camera_orientation(**camera_view(theta, phi, height=height))
        scene.begin_ambient_camera_rotation(rate=orbit_deg_per_sec * m.DEGREES)
        self.camera.frame.add_updater(self.update)
        scene.add(self.camera.frame)

    def look_at(self, target):
        self.target = np.array(target, dtype=float)
        self.update(self.camera.frame)

    def update(self, frame):
        offset = self.screen_point / self.camera.get_zoom()
        world_offset = camera_to_world(self.camera) @ offset
        frame.move_to(self.target - world_offset)


def camera_to_world(camera):
    # GX orbits about z, then tilts about the camera's local x axis.
    return Rotation.from_euler("ZXZ", [
        camera.get_theta() + np.pi / 2, camera.get_phi(), -camera.get_gamma(),
    ]).as_matrix()


class VideoPanel(m.Group):
    """Fixed-in-frame video with an independently animated opacity."""

    def __init__(self, frames, width=None, center=m.ORIGIN, color=LINE_COLOR, border_px=2.0, *, height=None):
        self.frames = frames
        self.image = m.ImageMobject(frames.texture(0))
        if height is not None:
            self.image.scale_to_fit_height(height)
        else:
            self.image.scale_to_fit_width(width)
        self.image.move_to(center)
        self.border = m.SurroundingRectangle(self.image, buff=0)
        self.border.set_stroke(color, border_px * STROKE_PER_PX).set_fill(opacity=0)
        self.alpha = m.ValueTracker(1)
        super().__init__(self.image, self.border)
        self.add_updater(lambda panel: panel.show(0))

    def show(self, idx):
        self.frames.show_on(self.image, idx)
        opacity = self.alpha.get_value()
        self.image.set_opacity(opacity)
        self.border.set_stroke(opacity=opacity)

    def pin(self, scene):
        # GX retains registered mobjects for the scene lifetime. Register only
        # the visible parts so retired panels can release their frame caches.
        scene.camera.add_fixed_in_frame_mobjects(self.image, self.border)
        scene.add(self)

    def follow(self, frame_index):
        self.clear_updaters()
        self.add_updater(lambda panel: panel.show(frame_index()))
        return self


def svg_polygon(points, color=LINE_COLOR):
    """Filled polygon from an SVG points string (used for the arrowheads)."""
    v = np.array([float(t) for t in points.replace(",", " ").split()]).reshape(-1, 2)
    return m.Polygon(*[P(*p) for p in v]).set_fill(color, 1).set_stroke(color, 0)


def svg_rect(x, y, w, h, r, stroke_px=3.0, color=LINE_COLOR):
    return px_box(x + w / 2, y + h / 2, w, h, r, stroke_px, color)


_CAP_REF = {}


def svg_text(string, size_px, bold=True, color=LINE_COLOR):
    """Text set so its cap height matches Open Sans at `size_px` in the mockup (cap height = 0.714 em)."""
    weight = m.BOLD if bold else m.NORMAL
    if weight not in _CAP_REF:
        _CAP_REF[weight] = m.Text("H", font=FONT, weight=weight, font_size=48).height
    if "\n" in string:
        t = m.Paragraph(*string.splitlines(), font=FONT, weight=weight, font_size=48, alignment="left")
    else:
        t = m.Text(string, font=FONT, weight=weight, font_size=48)
    t.scale(CAP_HEIGHT[FONT] * size_px * PX / _CAP_REF[weight])
    return t.set_color(color)


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
    d["boxes"] = m.VGroup(*[svg_rect(x, y, BOX_W, h, BOX_R) for x, y, h in BOXES])
    d["arrow_in"] = m.VGroup(px_line((656.52, 546.53), (700.09, 546.53), 2, LINE_COLOR),
                           svg_polygon("709.27 546.53 696.33 551.81 699.4 546.53 696.33 541.24"))
    d["arrow_out"] = m.VGroup(px_line((1252.29, 545.39), (1295.86, 545.39), 2, LINE_COLOR),
                            svg_polygon("1305.04 545.39 1292.1 550.67 1295.17 545.39 1292.1 540.1"))
    d["bus"] = m.VGroup(*[m.VGroup(svg_path(path, px), svg_polygon(head)) for path, head, px in BUS_DROPS])
    # camera icon: body is the mockup's rotated rect (x 574.66-652.36, y 218.16-261.29), lens is its path
    d["camera"] = m.VGroup(svg_rect(574.66, 218.16, 77.7, 43.13, 2.84),
                         svg_path("M652.36,224.02l24.56-7.83c2.96-.94,5.98,1.27,5.98,4.37v38.33c0,3.14-3.08,5.35-6.05,4.35l-24.4-8.23", 3.0))
    if show_text:
        d["model_label"] = px_text("STABLE VIDEO DIFFUSION MODEL", 28, 738.65, 763.53, bold=True, color=LINE_COLOR)
        d["image_label"] = px_text("IMAGE", 28, 320.64, 728.92, bold=True, color=LINE_COLOR)
        d["video_label"] = px_text("VIDEO", 28, 1566.82, 743.85, bold=True, color=LINE_COLOR)
        d["title_cameractrl"] = svg_text("CameraCtrl (2024)", 42)
        d["title_cameractrl"].move_to(P(960, 94.01 - 0.26 * 42))
        d["title_training"] = px_text("TRAINING", 42, 856.06, 90.74, bold=True, color=LINE_COLOR)
        d["title_inference"] = px_text("INFERENCE", 42, 844.85, 90.74, bold=True, color=LINE_COLOR)
        d["cite_svd"] = svg_text('Blattmann, Andreas, et al. "Stable video diffusion:\nScaling latent video diffusion models to large\n'
                                 'datasets." arXiv preprint arXiv:2311.15127 (2023).', 16, bold=False)
        d["cite_svd"].move_to(P(1494.32, 1007.43 + 3.27 - 0.76 * 16), aligned_edge=m.UL)
        d["cite_cameractrl"] = svg_text('He, Hao, et al. "Cameractrl: Enabling camera control for\n'
                                        'text-to-video generation." arXiv preprint arXiv:2404.02101 (2024).', 16, bold=False)
        d["cite_cameractrl"].move_to(P(1401.01, 1022.86 - 0.76 * 16), aligned_edge=m.UL)
    return d


def placed_image(path, x, y, w, h, scale, placeholder_color=CHILL_BROWN):
    """Image at the mockup's <image> placement (top-left x, y; native w x h; scale). Falls back to a grey card."""
    if path is not None:
        im = m.ImageMobject(path)
        im.scale_to_fit_height(h * scale * PX)
    else:
        im = m.Rectangle(width=w * scale * PX, height=h * scale * PX).set_fill(placeholder_color, 0.35).set_stroke(LINE_COLOR, 1)
    im.move_to(P(x, y), aligned_edge=m.UL)
    return im


class CameraCtrlScene(m.ThreeDScene):
    # --- toggles -------------------------------------------------------------------------------------------
    show_text = True             # titles, labels and citations as manim m.Text (Open Sans). False: add them in post
    video_on_pyramid = True      # play the clip on the moving pyramid's image plane
    # --- training beat -------------------------------------------------------------------------------------
    train_clip_indices = [10, 3, 17, 22, 31, 45]   # indices into the available examples
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

    camera_ctrl_root = None
    bus_run_time = 3.0
    training_transition_time = 2.5
    end_hold = 20
    inference_path_first = False
    train_path_first = False
    path_time = 3.0
    path_hold = 0.5
    refly_with_video = False

    def construct(self):
        self.assets = CameraCtrlAssets(self.camera_ctrl_root)
        self.d = build_diagram(self.show_text)
        self.camera.add_fixed_in_frame_mobjects(*self.d.values())
        self.beat_svd()
        self.beat_cameractrl()
        self.beat_to_training_layout()
        self.beat_training()
        self.beat_inference()
        self.wait(self.end_hold)

    # small helpers so every beat also works with show_text = False
    def has(self, key):
        return key in self.d

    def anims(self, cls, *keys, **kwargs):
        return [cls(self.d[k], **kwargs) for k in keys if k in self.d]

    # --------------------------------------------------------------------------------------------- beat 1
    def beat_svd(self):
        d = self.d
        image_path, video_paths = self.assets.svd_example_paths()
        video_paths = list(video_paths) + [None] * (3 - len(video_paths))
        self.svd_image = placed_image(image_path, 128.51, 404.39, 515, 291, 0.95)
        stack = [(1337.07, 368.13, 519), (1373.28, 408.77, 517), (1412.75, 460.23, 516)]     # back to front
        self.svd_video = m.Group(*[placed_image(p, x, y, w, 291, 0.8132) for p, (x, y, w) in zip(video_paths, stack)])

        self.camera.add_fixed_in_frame_mobjects(self.svd_image, self.svd_video)
        self.wait(0.5)
        self.play(m.LaggedStart(*[m.Create(b) for b in d["boxes"]], lag_ratio=0.15), run_time=2.0)
        if self.show_text:
            self.play(*self.anims(m.Write, "model_label"), *self.anims(m.FadeIn, "cite_svd"), run_time=1.0)
        else:
            self.wait(1.0)
        self.play(m.FadeIn(self.svd_image, shift=0.2 * m.UP), *self.anims(m.FadeIn, "image_label"), run_time=0.8)
        self.play(m.Create(d["arrow_in"][0]), m.FadeIn(d["arrow_in"][1]), run_time=0.5)
        self.play(m.Create(d["arrow_out"][0]), m.FadeIn(d["arrow_out"][1]), run_time=0.5)
        self.play(m.LaggedStart(*[m.FadeIn(im, shift=0.15 * m.UP) for im in self.svd_video], lag_ratio=0.3),
                  *self.anims(m.FadeIn, "video_label"), run_time=1.2)
        self.wait(1.5)

    # --------------------------------------------------------------------------------------------- beat 2
    def beat_cameractrl(self):
        d = self.d
        self.play(*self.anims(m.FadeIn, "title_cameractrl", shift=0.2 * m.DOWN),
                  *self.anims(m.FadeOut, "cite_svd"), *self.anims(m.FadeIn, "cite_cameractrl"),
                  m.Create(d["camera"]), run_time=1.0)
        self.play(m.LaggedStart(*[m.AnimationGroup(m.Create(drop[0]), m.FadeIn(drop[1], rate_func=m.squish_rate_func(m.smooth, 0.7, 1)))
                                for drop in d["bus"]], lag_ratio=0.12), run_time=self.bus_run_time)
        self.wait(1.5)

    # --------------------------------------------------------------------------------------------- beat 2 -> 3
    def beat_to_training_layout(self):
        """Shrink the model block to its mockup -03 size, clear the SVD example, swap the camera icon for 3D."""
        d = self.d
        block = m.Group(d["boxes"], d["arrow_in"], d["arrow_out"], d["bus"], *[d[k] for k in ["model_label"] if k in d])
        target = block.copy()
        target.scale(SMALL_SCALE, about_point=P(*BIG_CENTER)).shift(P(*SMALL_CENTER) - P(*BIG_CENTER))
        for sm in target.family_members_with_points():
            if isinstance(sm, m.VMobject) and sm.get_stroke_width() > 0:
                sm.set_stroke(width=float(sm.get_stroke_width()) * SMALL_SCALE)
        moves = [m.Transform(block, target)]
        if self.has("image_label"):
            moves += [d["image_label"].animate.move_to(P(320.46, 889.66), aligned_edge=m.DL),
                      d["video_label"].animate.move_to(P(1566.65, 877.82), aligned_edge=m.DL)]
        fades = [m.FadeOut(self.svd_image), m.FadeOut(self.svd_video), m.FadeOut(d["camera"])]
        if self.has("title_cameractrl"):
            fades += [m.FadeOut(d["title_cameractrl"], shift=0.2 * m.UP), m.FadeIn(d["title_training"], shift=0.2 * m.UP)]
        self.play(*moves, *fades, run_time=self.training_transition_time)

        # panel geometry for the training / inference beats (each panel sits just above its label)
        w, h = self.panel_width_px, self.panel_width_px * 320 / 576
        label_tops = (889.66 - 0.714 * 28, 877.82 - 0.714 * 28)
        self.panel_width = w * PX
        center_y = min(label_tops) - self.panel_gap_px - h / 2        # same height for both panels
        self.panel_centers = [P(cx, center_y) for cx in self.panel_center_x_px]

        sx, sy = self.path_screen_px
        self.view = CornerView(self, P(sx, sy), self.view_height)

    # --------------------------------------------------------------------------------------------- shared
    def still_panel(self, frame_path):
        return VideoPanel(ClipFrames([frame_path]), self.panel_width, self.panel_centers[0])

    def run_path(self, traj, frames, left_panel=None, layout_first=False, run_time=4.5, *, path_first=False):
        """One camera path: 3D rig in the corner + the clip on the right. Fades itself in and out.

        path_first=False: pyramid and clip run together.
        path_first=True : the pyramid flies the whole path first (wireframe, no video anywhere yet);
                          then the clip fades in on the right and plays.
        left_panel: a still to fade in/out with this path (training); None = the caller manages the left panel."""
        frames = ClipFrames(frames)
        show_video = self.video_on_pyramid and (not path_first or self.refly_with_video)
        rig = CameraRig(traj, frames, self.rig_color, self.pyramid_depth, self.ghost_every,
                        follow_ghosts=not (path_first or layout_first), show_video=show_video)
        video_t = rig.t if not path_first else m.ValueTracker(0)      # which frame the right-hand clip shows
        right = VideoPanel(frames, self.panel_width, self.panel_centers[1])
        right.follow(lambda: rig.frame_at(video_t.get_value()))
        left = [left_panel] if left_panel is not None else []
        faders = [rig.alpha, right.alpha] + [p.alpha for p in left]
        for a in faders:
            a.set_value(0)
        self.view.look_at(traj.center(self.pyramid_depth))
        self.add(rig.mobjects)
        for panel in [right, *left]:
            panel.pin(self)
        last = rig.n - 1

        if not path_first:
            self.play(*[a.animate.set_value(1) for a in faders], run_time=0.5)
            if layout_first:
                self.play(rig.ghost.animate.set_value(rig.n), run_time=self.layout_time, rate_func=m.linear)
            self.play(rig.t.animate.set_value(last), run_time=run_time, rate_func=m.linear)
        else:
            # 1. the path: pyramid flies to the end, leaving its ghost trail. No video yet.
            rig.screen_visible = False
            rig.ghost.set_value(1)
            self.play(rig.alpha.animate.set_value(1), *[p.alpha.animate.set_value(1) for p in left], run_time=0.5)
            self.play(rig.t.animate.set_value(last), rig.ghost.animate.set_value(rig.n),
                      run_time=self.path_time, rate_func=m.linear)
            self.wait(self.path_hold)
            # 2. the rendered video
            if self.refly_with_video:
                rig.t.set_value(0)
                rig.screen_visible = True
            self.play(right.alpha.animate.set_value(1), run_time=0.5)
            moves = [video_t.animate.set_value(last)] + ([rig.t.animate.set_value(last)] if self.refly_with_video else [])
            self.play(*moves, run_time=run_time, rate_func=m.linear)

        self.wait(self.hold)
        self.play(*[a.animate.set_value(0) for a in faders], run_time=0.4)
        self.remove(rig.mobjects, right, *left)
        rig.mobjects.clear_updaters()
        right.clear_updaters()
        for panel in left:
            panel.clear_updaters()

    # --------------------------------------------------------------------------------------------- beat 3
    def beat_training(self):
        ids = self.train_clip_ids
        if ids is None:
            avail = self.assets.available_training_examples()
            if not avail:
                print(f"[{type(self).__name__}] No training clips in {self.assets.repo}")
                return
            ids = [avail[i % len(avail)] for i in self.train_clip_indices]
        for clip_id in ids:
            c2w, fxfy, frames = self.assets.load_training_example(clip_id)
            traj = Trajectory(c2w, fxfy, extent=self.train_extent)
            self.run_path(traj, frames, left_panel=self.still_panel(frames[0]), path_first=self.train_path_first, run_time=self.train_run_time)

    # --------------------------------------------------------------------------------------------- beat 4
    def beat_inference(self):
        d = self.d
        if self.has("title_training"):
            self.play(m.FadeOut(d["title_training"], shift=0.2 * m.UP), m.FadeIn(d["title_inference"], shift=0.2 * m.UP), run_time=0.8)
        for outputs_dir in self.inference_sets:
            runs = [(name, self.assets.load_inference_run(name, outputs_dir)) for name in self.inference_runs]
            runs = [(name, run) for name, run in runs if run is not None]
            if not runs:
                print(f"[{type(self).__name__}] nothing to show from {outputs_dir}/ - skipping")
                continue
            # the conditioning image stays on the left for all of this set's paths (frame 0 of the first run)
            left = self.still_panel(runs[0][1][2][0])
            left.alpha.set_value(0)
            left.pin(self)
            self.play(left.alpha.animate.set_value(1), run_time=0.6)
            for name, (c2w, fxfy, frames) in runs:
                traj = Trajectory(c2w, fxfy, scale=self.inference_scale)
                self.run_path(traj, frames, path_first=self.inference_path_first, run_time=self.inference_run_time)
            self.play(left.alpha.animate.set_value(0), run_time=0.5)
            self.remove(left)
            left.clear_updaters()
