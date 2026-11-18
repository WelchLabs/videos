"""
P29 - mock NeRF: four camera pyramids in 3D with my office photos on their image planes.

    uv run manimgx render p29.py P29

The photos were taken from the middle of the office, turning ~45 degrees between shots.
Cameras 1, 2 and 4 are white (the "training" views), camera 3 is yellow (the "new view").

Layout: the apexes sit on a small ring around the room center, each looking straight outward,
so the pyramids keep a realistic field of view without running into each other.
World is manim's z-up: floor is the xy-plane, the fan of cameras is centered on +y.
The scene camera sits behind and above the room center, so the photos read un-mirrored.
"""
from pathlib import Path

import numpy as np
import manimgx as m
from assets import GRAPHICS
from welch_scene import cache_upright_photo, camera_view, frustum_wire, place_image_plane
from welch_storyboard import STROKE_PER_PX
from welch_style import CHILL_BROWN, YELLOW

# ----------------------------------------------------------------------------- config
ASSETS = GRAPHICS / "p29_nerf_assets"

CACHE = Path(__file__).resolve().parent / "_p29_cache"
MAX_IMAGE_PX = 2048          # longest side of the cached copies
STROKE_SCALE = 2.03 * STROKE_PER_PX  # preserve the original pixel widths at 1080p


def find_image(i):
    """p29_<i>.jpeg (any image extension)."""
    for ext in ("jpeg", "jpg", "png", "JPEG", "JPG", "PNG"):
        p = ASSETS / f"p29_{i}.{ext}"
        if p.exists():
            return p
    cached = CACHE / f"p29_{i}.png"
    if cached.exists():
        return cached
    raise FileNotFoundError(f"no p29_{i}.* in {ASSETS} or {CACHE}")


def upright_copy(path):
    return cache_upright_photo(path, CACHE / f"{path.stem}.png", MAX_IMAGE_PX)


def camera_pose(yaw_deg, ring_radius, height, pitch_deg=0.0, roll_deg=0.0, offset=m.ORIGIN):
    """Camera on the ring, looking outward. yaw is measured counterclockwise from +x, seen from above;
    pitch tilts the view up, roll spins it about the view axis, offset nudges the apex."""
    a = np.deg2rad(yaw_deg)
    fwd = np.array([np.cos(a), np.sin(a), 0.0])
    right = np.array([np.sin(a), -np.cos(a), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    pos = ring_radius * fwd + height * up + np.array(offset)

    p = np.deg2rad(pitch_deg)
    fwd, up = np.cos(p) * fwd + np.sin(p) * up, np.cos(p) * up - np.sin(p) * fwd
    r = np.deg2rad(roll_deg)
    right, up = np.cos(r) * right + np.sin(r) * up, np.cos(r) * up - np.sin(r) * right
    return dict(pos=pos, fwd=fwd, right=right, up=up)


class CameraPyramid(m.Group):
    """Wireframe pyramid (4 edges from the apex + image-plane rectangle) with a photo on the image plane."""

    def __init__(self, image_path, pose, color=m.WHITE, depth=1.5, hfov_deg=65.0,
                 stroke_width=3.0, image_opacity=1.0, inset=0.985):
        self.pose = pose
        self.image = m.ImageMobject(str(image_path))
        aspect = self.image.width / self.image.height

        hw = depth * np.tan(np.deg2rad(hfov_deg) / 2)
        hh = hw / aspect
        pos, fwd, r, u = pose["pos"], pose["fwd"], pose["right"], pose["up"]

        self.wire = frustum_wire(pos, pos + depth * fwd, r, u, 2 * hw, 2 * hh,
                                 color=color, width=stroke_width * STROKE_SCALE)

        # Keep the photo just inside the wire rectangle to avoid z-fighting.
        place_image_plane(self.image, pos + depth * inset * fwd, r, u,
                          2 * hw * inset, 2 * hh * inset)
        self.image.set_opacity(image_opacity)

        super().__init__(self.image, self.wire)


def floor_grid(extent=6.0, step=1.0, color=CHILL_BROWN, stroke_width=1.0, opacity=0.35):
    ticks = np.arange(-extent, extent + 1e-6, step)
    lines = m.VGroup()
    for t in ticks:
        lines.add(m.Line([t, -extent, 0], [t, extent, 0]))
        lines.add(m.Line([-extent, t, 0], [extent, t, 0]))
    lines.set_stroke(color, stroke_width * STROKE_SCALE, opacity)
    return lines


class P29(m.ThreeDScene):
    # --- cameras
    new_view = 3                 # this image's pyramid is yellow
    angle_step = 45.0            # degrees between consecutive photos
    turn = +1                    # +1: photos 1 -> 4 turn left (counterclockwise from above); -1 to flip
    hfov = 65.0                  # horizontal field of view of the pyramids (iPhone main camera-ish)
    ring_radius = 0.8            # apex distance from the room center (smaller = pyramids closer together)

    # --- randomness (uniform in +/- each amount; change the seed for a different arrangement, 0 amounts = regular fan)
    seed = 3
    pos_jitter = 0.25            # apex position, in the floor plane
    height_jitter = 0.15         # apex height
    yaw_jitter = 7.0             # degrees
    pitch_jitter = 5.0           # degrees
    roll_jitter = 4.0            # degrees
    pyramid_depth = 1.5          # apex -> image plane
    camera_height = 1.0          # apex height above the floor grid
    stroke_width = 3.0
    base_color = m.WHITE
    new_color = YELLOW

    # --- floor
    show_grid = True
    grid_extent = 6.0
    grid_opacity = 0.35

    # --- scene camera (fixed 3/4 view): theta, phi, gamma, look-at point, height
    # view = (-22, 62, 0, (0.0, 1.5, 0.7), 8.5)
    view=(-21, 61, 0, (np.float32(0.34), np.float32(1.46), np.float32(0.81)), 6.76)

    hold_time = 5.0

    def construct(self):
        self.set_camera_orientation(**camera_view(*self.view))

        if self.show_grid:
            self.grid = floor_grid(self.grid_extent, opacity=self.grid_opacity)
            self.add(self.grid)

        n = 4
        rng = np.random.default_rng(self.seed)
        self.pyramids = m.Group()
        for i in range(1, n + 1):
            yaw = 90 + self.turn * (i - (n + 1) / 2) * self.angle_step   # fan centered on +y
            j = rng.uniform(-1, 1, 6)
            pose = camera_pose(yaw + j[0] * self.yaw_jitter, self.ring_radius, self.camera_height,
                               pitch_deg=j[1] * self.pitch_jitter, roll_deg=j[2] * self.roll_jitter,
                               offset=[j[3] * self.pos_jitter, j[4] * self.pos_jitter, j[5] * self.height_jitter])
            color = self.new_color if i == self.new_view else self.base_color
            self.pyramids.add(CameraPyramid(upright_copy(find_image(i)), pose, color, self.pyramid_depth,
                                            self.hfov, self.stroke_width))
        self.add(self.pyramids)

        self.wait(self.hold_time)


