"""Shared ManimGX camera geometry and image helpers for Welch Labs scenes."""

import manimgx as m
import numpy as np
from PIL import Image, ImageOps


def camera_view(theta=None, phi=None, gamma=None, center=None, height=None, *,
                fov=np.pi / 4, frame_height=None):
    """Convert the storyboard's degree angles and frame height to GX camera settings."""
    settings = {}
    if theta is not None:
        settings["theta"] = (theta - 90) * m.DEGREES
    if phi is not None:
        settings["phi"] = phi * m.DEGREES
    if gamma is not None:
        settings["gamma"] = -gamma * m.DEGREES
    if center is not None:
        settings["frame_center"] = center
    if height is not None:
        if frame_height is None:
            frame_height = m.config.frame_height
        settings["zoom"] = frame_height / height
        settings["focal_distance"] = height / (2 * np.tan(fov / 2))
    return settings


def image_plane_corners(center, right, up, width, height):
    """Return vertices in the order used by ManimGX's image mesh and UVs."""
    center = np.asarray(center)
    x = np.asarray(right) * (width / 2)
    y = np.asarray(up) * (height / 2)
    return np.array([center - x + y, center + x + y, center - x - y, center + x - y])


def place_image_plane(image, center, right, up, width, height):
    return image.set_points(image_plane_corners(center, right, up, width, height))


def frustum_wire(apex, center, right, up, width, height, **stroke):
    corners = image_plane_corners(center, right, up, width, height)[[0, 1, 3, 2]]
    wire = m.VGroup(*(m.Line(apex, corner) for corner in corners), m.Polygon(*corners))
    return wire.set_stroke(**stroke).set_fill(opacity=0)


def cache_upright_photo(source, target, max_px, **save_options):
    """Cache an RGB copy with EXIF orientation baked in and a bounded long edge."""
    if source != target and (not target.exists() or target.stat().st_mtime < source.stat().st_mtime):
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as photo:
            upright = ImageOps.exif_transpose(photo).convert("RGB")
        upright.thumbnail((max_px, max_px))
        upright.save(target, **save_options)
    return target


def set_image_pixels(image, pixels):
    """Swap a GX image's texture while retaining its mesh and style."""
    rgba = np.asarray(pixels, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB or RGBA image array")
    if rgba.shape[2] == 3:
        alpha = np.full((*rgba.shape[:2], 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgba, alpha], axis=2)
    elif rgba.flags.writeable:
        rgba = rgba.copy()
    rgba.flags.writeable = False
    image.paint = image.paint.but(texture=rgba)
    return image


class _CameraMotion(m.AnimationGroup):
    """Keep the camera frame out of the scene after temporarily animating it."""

    def __init__(self, *animations, frame, **kwargs):
        self.frame = frame
        super().__init__(*animations, **kwargs)

    def _setup_scene(self, scene):
        super()._setup_scene(scene)
        self._frame_was_present = any(mob is self.frame for mob in scene.get_mobject_family_members())

    def clean_up_from_scene(self, scene):
        super().clean_up_from_scene(scene)
        if not self._frame_was_present:
            scene.remove(self.frame)


class WelchScene(m.ThreeDScene):
    """Three-dimensional scene with the framing conventions used in Welch storyboards."""

    _default_view_fov = np.pi / 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.view_fov = self._default_view_fov

    def set_view(self, theta=None, phi=None, gamma=None, center=None, height=None):
        self.set_camera_orientation(**self._view_settings(theta, phi, gamma, center, height))

    def animate_view(self, theta=None, phi=None, gamma=None, center=None, height=None, **kwargs):
        settings = self._view_settings(theta, phi, gamma, center, height)
        return self._animate_camera(settings, **kwargs)

    def _view_settings(self, theta, phi, gamma, center, height):
        return camera_view(theta, phi, gamma, center, height, fov=self.view_fov,
                           frame_height=self.camera.frame_height)

    def _animate_camera(self, settings, **kwargs):
        animations = []
        frame = None
        for key, value in settings.items():
            if key == "frame_center":
                frame = self.camera.frame
                # Defer adding the frame until _CameraMotion records whether it was already in the scene.
                animations.append(frame.animate(introducer=True).move_to(value))
            else:
                animations.append(getattr(self.camera, f"{key}_tracker").animate.set_value(value))
        if frame is None:
            return m.AnimationGroup(*animations, **kwargs)
        return _CameraMotion(*animations, frame=frame, **kwargs)

    @property
    def view_angles(self):
        """Current theta, phi and roll in the storyboard's degree convention."""
        camera = self.camera
        return np.degrees([camera.get_theta() + np.pi / 2, camera.get_phi(), -camera.get_gamma()])

    def save_view(self):
        camera = self.camera
        self._saved_view = dict(
            theta=camera.get_theta(), phi=camera.get_phi(), gamma=camera.get_gamma(),
            zoom=camera.get_zoom(), focal_distance=camera.get_focal_distance(),
            frame_center=camera.frame.get_center(),
        )
        return self

    def restore_view(self):
        self.set_camera_orientation(**self._saved_view)
        return self

    def animate_restore_view(self, **kwargs):
        return self._animate_camera(self._saved_view, **kwargs)

    @property
    def view_fov(self):
        camera = self.camera
        height = camera.frame_height / camera.get_zoom()
        return 2 * np.arctan2(height / 2, camera.get_focal_distance())

    @view_fov.setter
    def view_fov(self, angle):
        height = self.camera.frame_height / self.camera.get_zoom()
        self.set_camera_orientation(focal_distance=height / (2 * np.tan(angle / 2)))


def image_at_height(source, height=4.0, **style):
    """Load an image sized in scene units, independent of its pixel resolution."""
    return m.ImageMobject(source, **style).scale_to_fit_height(height)


def copy_view_positioning(scene, *, clipboard=True):
    """Format a WelchScene's camera position as a reusable set_view call."""
    theta, phi, gamma = (float(value) for value in scene.view_angles)
    center = tuple(float(value) for value in scene.camera.frame.get_center())
    height = scene.camera.frame.height / scene.camera.get_zoom()
    call = f"self.set_view({theta:.3f}, {phi:.3f}, {gamma:.3f}, {center!r}, {height:.3f})"
    if clipboard:
        import pyperclip

        pyperclip.copy(call)
    return call


def manual_camera_interpolation(start_orientation, end_orientation, num_steps):
    """Sample the authored view angles, look-at point, and height between two poses."""
    result = []
    for alpha in np.linspace(0, 1, num_steps):
        values = []
        for index, (start, end) in enumerate(zip(start_orientation, end_orientation, strict=True)):
            if index == 3:
                values.append(tuple(a + alpha * (b - a) for a, b in zip(start, end, strict=True)))
            else:
                values.append(start + alpha * (end - start))
        result.append(values)
    return result
