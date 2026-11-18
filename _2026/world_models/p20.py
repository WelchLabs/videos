import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import manimgx as m
from welch_style import *

IMG_DIR = Path(os.environ.get(
    'NERF_PICS',
    '/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/world_models/hackin/nerf_pics',
))
N_PLACEHOLDERS = 242
THUMB_HEIGHT = 200
HERO_HEIGHT = 1080
N_3D = 96
SEED = 20
SPLIT_FILL_TIME = 10
POSES_FILE = Path(__file__).with_name('p20_colmap.npz')
POSE_SCALE = 0.8
MAX_CLOUD_POINTS = 60000
CLOUD_DIM = 0.35
NEW_VIEW_FOCAL = 0.6
NEW_VIEW_DISTANCE = 2.0
NEW_VIEW_FOV = 65 * np.pi / 180
NERF_FRAMES = Path(__file__).with_name('p20_nerf_frames')
NERF_VIEW_T = 195 / 239
NERF_VIEW_CROP = 0.85
NERF_FRAME_HOLD = 2
VIEW_FOV = 2 * np.arctan(NERF_VIEW_CROP * np.tan(NEW_VIEW_FOV / 2))
NEW_VIEW_HEIGHT = 2 * NEW_VIEW_FOCAL * np.tan(VIEW_FOV / 2)


def capture_time(path):
    exif = Image.open(path).getexif().get_ifd(0x8769)
    return (str(exif.get(36867, '')), str(exif.get(37521, '')), path.name)


def photo_paths():
    if not IMG_DIR.is_dir():
        return []
    exts = {'.jpg', '.jpeg', '.png'}
    return sorted((p for p in IMG_DIR.iterdir() if p.suffix.lower() in exts),
                  key=capture_time)


def load_photo(path, height):
    img = Image.open(path)
    img.draft('RGB', (height * 4, height * 4))
    img = ImageOps.exif_transpose(img).convert('RGB')
    width = round(img.width * height / img.height)
    return np.asarray(img.resize((width, height), Image.LANCZOS))


def placeholder_photo(i, n, height):
    width = round(height * 4 / 3)
    t = i / max(n - 1, 1)
    a = np.array([0x3d, 0x5c, 0x6f]) * (1 - t) + np.array([0x94, 0x89, 0x79]) * t
    ramp = np.linspace(0.55, 1.0, height)[:, None, None]
    arr = np.clip(a[None, None, :] * ramp * np.ones((1, width, 1)), 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.text((width / 2, height / 2), str(i), fill=FRESH_TAN, anchor='mm',
              font_size=height * 0.3)
    return np.asarray(img)


def load_photos(height=THUMB_HEIGHT, indices=None):
    paths = photo_paths()
    n = len(paths) or N_PLACEHOLDERS
    indices = range(n) if indices is None else indices
    if paths:
        return [load_photo(paths[i], height) for i in indices]
    return [placeholder_photo(i, n, height) for i in indices]


def photo_count():
    return len(photo_paths()) or N_PLACEHOLDERS


def tile(array, height):
    img = m.ImageMobject(array)
    img.scale_to_fit_height(height)
    return img


def grid_shape(n, tile_aspect, frame_aspect=16 / 9):
    cols = int(np.ceil(np.sqrt(n * frame_aspect / tile_aspect)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def grid_positions(n, tile_w, tile_h, gap, rows, cols):
    dx, dy = tile_w + gap, tile_h + gap
    out = []
    for i in range(n):
        r, c = divmod(i, cols)
        out.append(np.array([(c - (cols - 1) / 2) * dx, ((rows - 1) / 2 - r) * dy, 0.0]))
    return out


def fit_grid(n, tile_aspect, box_w, box_h, gap_frac=0.06):
    rows, cols = grid_shape(n, tile_aspect, box_w / box_h)
    tile_h = min(box_h / (rows * (1 + gap_frac)),
                 box_w / (cols * (tile_aspect + gap_frac)))
    tile_w = tile_h * tile_aspect
    positions = grid_positions(n, tile_w, tile_h, gap_frac * tile_h, rows, cols)
    return positions, tile_w, tile_h, rows, cols


def synthetic_poses(n, rng):
    poses = []
    for i in range(n):
        ring = i % 3
        angle = m.TAU * i / n + rng.normal(0, 0.04)
        radius = [4.6, 4.0, 3.2][ring] + rng.normal(0, 0.15)
        z = [0.5, 1.7, 2.9][ring] + rng.normal(0, 0.12)
        pos = np.array([radius * np.cos(angle), radius * np.sin(angle), z])
        look_at = np.array([*rng.normal(0, 0.35, 2), 0.7])
        poses.append((pos, look_at))
    return poses


def colmap_poses(paths):
    if not POSES_FILE.is_file() or not paths:
        return None
    data = np.load(POSES_FILE)
    by_stem = {Path(str(name)).stem: i for i, name in enumerate(data['names'])}
    poses = {}
    for i, path in enumerate(paths):
        j = by_stem.get(path.stem)
        if j is not None:
            rot = np.column_stack([data['right'][j], data['up'][j], -data['forward'][j]])
            poses[i] = (data['centers'][j] * POSE_SCALE, rot)
    return poses


def colmap_point_cloud(rng):
    data = np.load(POSES_FILE)
    points, colors = data['points'] * POSE_SCALE, data['colors']
    if len(points) > MAX_CLOUD_POINTS:
        keep = rng.choice(len(points), MAX_CLOUD_POINTS, replace=False)
        points, colors = points[keep], colors[keep]
    rgbas = np.column_stack([colors / 255, np.full(len(colors), 0.9)])
    return m.PMobject(stroke_width=2.5).add_points(points, rgbas=rgbas)


def new_view_pose(t):
    angle = np.deg2rad(200 + 120 * t)
    pos = np.array([1.0 * np.cos(angle), -0.5 + 1.0 * np.sin(angle), 2.3 - 0.3 * t])
    forward = m.normalize(np.array([np.cos(angle), np.sin(angle), -0.12]))
    return pos, forward


def look_at_matrix(pos, look_at):
    forward = m.normalize(look_at - pos)
    right = m.normalize(np.cross(forward, m.OUT))
    up = np.cross(right, forward)
    return np.column_stack([right, up, -forward])


def scene_to_colmap(data):
    align, origin, scale = data['align'], data['origin'], data['scale']

    def point(p):
        return (np.asarray(p) / POSE_SCALE / scale + origin) @ align

    def direction(d):
        return np.asarray(d) @ align

    return point, direction


def write_camera_path(out_path, fps=60, seconds=4.0):
    data = np.load(POSES_FILE)
    point, direction = scene_to_colmap(data)
    poses = []
    for t in np.linspace(0, 1, round(fps * seconds)):
        pos, forward = new_view_pose(t)
        rot = look_at_matrix(pos, pos + forward)
        c2w = np.eye(4)
        c2w[:3, 0] = direction(rot[:, 0])
        c2w[:3, 1] = direction(-rot[:, 1])
        c2w[:3, 2] = direction(-rot[:, 2])
        c2w[:3, 3] = point(pos)
        poses.append(c2w)
    np.savez(out_path, c2w=np.array(poses), fov_y_degrees=np.degrees(NEW_VIEW_FOV),
             width=1920, height=1080, fps=fps)
    print(f'{len(poses)} poses -> {out_path}')


def frustum(apex, img_center, rot, w, h, **style):
    right, up = rot[:, 0], rot[:, 1]
    corners = [img_center + sx * right * w / 2 + sy * up * h / 2
               for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    lines = m.VGroup(*[m.Line(apex, c) for c in corners])
    lines.add(m.Polygon(*corners))
    lines.set_stroke(**style)
    lines.set_fill(opacity=0)
    return lines


def room_point_cloud(rng, n=6000):
    def box(n, lo, hi):
        return rng.uniform(lo, hi, size=(n, 3))

    parts = [
        (box(n // 2, [-2.2, -2.2, 0.0], [2.2, 2.2, 0.02]), CHILL_BROWN),
        (box(n // 5, [-1.3, -0.6, 0.85], [1.3, 0.6, 0.95]), FRESH_TAN),
        (box(n // 10, [-1.25, -0.55, 0.0], [-1.15, 0.55, 0.85]), FRESH_TAN),
        (box(n // 10, [1.15, -0.55, 0.0], [1.25, 0.55, 0.85]), FRESH_TAN),
        (box(n // 6, [-0.6, 0.25, 1.05], [0.6, 0.3, 1.75]), BLUE),
        (box(n // 12, [-0.4, -1.3, 0.0], [0.4, -0.9, 1.1]), CHILL_GREEN),
    ]
    cloud = m.PMobject(stroke_width=4)
    for pts, color in parts:
        cloud.add_points(pts, color=color, alpha=0.9)
    return cloud


class P20a_PhotosTo3D(m.ThreeDScene):
    title_text = 'Neural Radiance Fields (NeRF)'

    def construct(self):
        rng = np.random.default_rng(SEED)
        total = photo_count()
        real = colmap_poses(photo_paths())
        pool = sorted(real) if real else list(range(total))
        n = min(N_3D, len(pool))
        indices = [pool[i] for i in np.linspace(0, len(pool) - 1, n).round().astype(int)]
        arrays = load_photos(THUMB_HEIGHT, indices)
        aspect = arrays[0].shape[1] / arrays[0].shape[0]

        title = m.VGroup()
        grid_h, grid_shift = 7.0, 0.0
        if self.title_text:
            title = m.Text(self.title_text, font_size=44, color=FRESH_TAN)
            title.to_edge(m.UP, buff=0.3)
            grid_h, grid_shift = 6.3, 0.45

        positions, tile_w, tile_h, _, _ = fit_grid(n, aspect, 13.4, grid_h, gap_frac=0.08)
        photos = m.Group(*[tile(a, tile_h).move_to(p + grid_shift * m.DOWN)
                           for a, p in zip(arrays, positions)])
        self.add_fixed_in_frame_mobjects(title)
        self.remove(title)

        self.set_camera_orientation(phi=0, theta=-90 * m.DEGREES)
        self.play(m.FadeIn(title),
                  m.LaggedStart(*[m.FadeIn(p, shift=0.15 * m.UP) for p in photos],
                                lag_ratio=0.015, run_time=2.5))
        self.wait()

        if real:
            poses = [real[i] for i in indices]
            pose_h, focal = 0.42, 0.38
        else:
            poses = [(pos, look_at_matrix(pos, look_at))
                     for pos, look_at in synthetic_poses(n, rng)]
            pose_h, focal = 0.62, 0.55
        pose_w = pose_h * aspect
        frustums = m.VGroup()
        for photo, (apex, rot) in zip(photos, poses):
            center = apex - focal * rot[:, 2]
            photo.generate_target()
            photo.target.scale_to_fit_height(pose_h)
            photo.target.apply_matrix(rot)
            photo.target.move_to(center)
            frustums.add(frustum(apex, center, rot, pose_w, pose_h,
                                 color=CHILL_BROWN, width=1.5, opacity=0.8))

        self.move_camera(
            phi=62 * m.DEGREES, theta=-55 * m.DEGREES, zoom=1.15 if real else 1.05,
            frame_center=[0, 0, 2.2 if real else 1.5],
            added_anims=[m.FadeOut(title, rate_func=m.squish_rate_func(m.smooth, 0, 0.3)),
                         m.LaggedStart(*[m.MoveToTarget(p) for p in photos],
                                       lag_ratio=0.012)],
            run_time=5,
        )
        self.play(m.LaggedStart(*[m.Create(f) for f in frustums],
                                lag_ratio=0.01, run_time=2))

        cloud = colmap_point_cloud(rng) if real else room_point_cloud(rng)
        self.move_camera(theta=-35 * m.DEGREES, added_anims=[m.FadeIn(cloud)],
                         run_time=3, rate_func=m.linear)

        progress = m.ValueTracker(0)

        def new_view():
            pos, forward = new_view_pose(progress.get_value())
            rot = look_at_matrix(pos, pos + forward)
            return frustum(pos, pos + NEW_VIEW_FOCAL * forward, rot,
                           NEW_VIEW_HEIGHT * 16 / 9, NEW_VIEW_HEIGHT,
                           color=YELLOW, width=4, opacity=1)

        camera_icon = m.always_redraw(new_view)
        self.move_camera(theta=-20 * m.DEGREES,
                         added_anims=[m.FadeIn(camera_icon),
                                      cloud.animate.set_opacity(CLOUD_DIM)],
                         run_time=2, rate_func=m.linear)
        self.move_camera(theta=5 * m.DEGREES,
                         added_anims=[progress.animate.set_value(NERF_VIEW_T)],
                         run_time=4, rate_func=m.linear)

        pos, forward = new_view_pose(NERF_VIEW_T)
        rot = look_at_matrix(pos, pos + forward)
        center = pos + NEW_VIEW_FOCAL * forward

        def nerf_frame(path):
            frame = tile(np.asarray(Image.open(path).convert('RGB')), NEW_VIEW_HEIGHT)
            frame.apply_matrix(rot)
            return frame.move_to(center)

        frame_paths = sorted(NERF_FRAMES.glob('*.jpg'))
        render = nerf_frame(frame_paths[0])
        behind = np.arctan2(-forward[1], -forward[0])
        self.move_camera(phi=75 * m.DEGREES, theta=behind - 30 * m.DEGREES,
                         zoom=m.config.frame_height / 1.7, focal_distance=1.6,
                         frame_center=center,
                         added_anims=[m.FadeIn(render)], run_time=3)
        self.wait(0.5)

        fov_height = 2 * NEW_VIEW_DISTANCE * np.tan(VIEW_FOV / 2)
        self.move_camera(
            phi=np.arccos(-forward[2]),
            theta=behind,
            zoom=m.config.frame_height / fov_height,
            focal_distance=NEW_VIEW_DISTANCE,
            frame_center=pos + NEW_VIEW_DISTANCE * forward,
            added_anims=[m.FadeOut(camera_icon, run_time=2.5)],
            run_time=4,
        )
        self.wait(0.5)
        for path in frame_paths[1:]:
            frame = nerf_frame(path)
            self.add(frame)
            self.remove(render)
            render = frame
            self.wait(NERF_FRAME_HOLD / m.config.frame_rate)
        self.wait(1.5)


class P20a_PhotosTo3DNoTitle(P20a_PhotosTo3D):
    title_text = None


class P20b_ZoomOut(m.Scene):
    def construct(self):
        total = photo_count()
        arrays = load_photos(THUMB_HEIGHT)
        aspect = arrays[0].shape[1] / arrays[0].shape[0]

        frame_w, frame_h = m.config.frame_width, m.config.frame_height
        positions, tile_w, tile_h, rows, cols = fit_grid(
            total, aspect, frame_w * 0.98, frame_h * 0.96)

        hero_i = min((rows // 2) * cols + cols // 2, total - 1)
        arrays[hero_i] = load_photos(HERO_HEIGHT, [hero_i])[0]

        grid = m.Group(*[tile(a, tile_h).move_to(p) for a, p in zip(arrays, positions)])
        hero_pos = positions[hero_i]

        frame = self.camera.frame
        s0 = min(tile_h / frame_h, tile_w / frame_w)
        pivot = hero_pos / (1 - s0)

        def set_zoom(frame, alpha):
            s = s0 ** (1 - alpha)
            frame.scale_to_fit_height(frame_h * s)
            frame.move_to(pivot * (1 - s))

        set_zoom(frame, 0)
        self.add(grid)
        self.wait(1.5)
        self.play(m.UpdateFromAlphaFunc(frame, set_zoom, rate_func=m.smooth, run_time=7))
        self.wait(2)


class P20c_SplitScreen(m.Scene):
    def construct(self):
        total = photo_count()
        arrays = load_photos(THUMB_HEIGHT)
        aspect = arrays[0].shape[1] / arrays[0].shape[0]

        frame_w, frame_h = m.config.frame_width, m.config.frame_height
        half_w = frame_w / 2
        positions, tile_w, tile_h, _, _ = fit_grid(
            total, aspect, half_w * 0.92, frame_h * 0.94)
        grid = m.Group(*[tile(a, tile_h).move_to(p) for a, p in zip(arrays, positions)])
        grid.shift(half_w / 2 * m.LEFT)

        divider = m.Line(frame_h / 2 * m.UP, frame_h / 2 * m.DOWN,
                         color=CHILL_BROWN, stroke_width=THIN_STROKE)
        self.add(divider)

        self.play(m.LaggedStart(*[m.FadeIn(p, scale=1.4) for p in grid],
                                lag_ratio=0.25, run_time=SPLIT_FILL_TIME,
                                rate_func=m.linear))
        self.wait(4)


if __name__ == '__main__':
    write_camera_path(sys.argv[1])
