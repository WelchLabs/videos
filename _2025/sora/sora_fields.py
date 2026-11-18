"""Batched ManimGX arrows for the Sora score and guidance scenes."""

import manimgx as m
import numpy as np


class TrackerControlledVectorField(m.MeshMobject):
    """Recompute one mesh when the tracked noise level changes."""

    def __init__(self, time_tracker, *, func, coordinate_system, density=2.0,
                 max_radius=2.0, min_opacity=0.1, max_opacity=0.7, color=m.WHITE,
                 stroke_width=3, stroke_opacity=1.0, tip_width_ratio=4,
                 tip_len_to_width=0.01, max_vect_len=None, max_vect_len_to_step_size=0.8):
        if density <= 0 or max_radius <= 0:
            raise ValueError("density and max_radius must be positive")
        self.tracker = time_tracker
        self.func = func
        self.coordinates = coordinate_system
        self.max_radius = max_radius
        self.min_opacity = min_opacity
        self.max_opacity = max_opacity
        self.stroke_opacity = stroke_opacity
        self.arrow_width = stroke_width / 100
        self.tip_width = tip_width_ratio * self.arrow_width
        self.tip_length = tip_len_to_width * tip_width_ratio * stroke_width
        output_axes = (coordinate_system.x_range, coordinate_system.y_range)
        samples = [np.arange(lo, hi + step / (2 * density), step / density)
                   for lo, hi, step, *_ in output_axes]
        xx, yy = np.meshgrid(*samples, indexing="ij")
        self.sample_coords = np.column_stack((xx.ravel(), yy.ravel()))
        self.sample_points = self._scene_points(self.sample_coords)
        scale = np.linalg.norm(coordinate_system.c2p(1, 0) - coordinate_system.c2p(0, 0))
        step_size = np.linalg.norm(self.sample_points[1] - self.sample_points[0])
        self.max_length = max_vect_len_to_step_size * step_size if max_vect_len is None else max_vect_len * scale

        # One rectangle and one triangle per arrow. The index buffer is fixed through playback.
        indices = np.arange(len(self.sample_coords))[:, None, None] * 7
        triangles = (indices + np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6]])).reshape(-1, 3)
        self._last_time = time_tracker.get_value()
        super().__init__(self._vertices(), triangles, fill_color=color)
        self.apply_radial_opacity()
        self.add_updater(self.update_from_tracker)

    def _scene_points(self, coords):
        return self.coordinates.c2p(*np.asarray(coords).T).T

    def _vertices(self):
        outputs = np.asarray(self.func(self.sample_coords), dtype=float)
        direction = self._scene_points(outputs) - self.coordinates.c2p(0, 0)
        lengths = np.linalg.norm(direction, axis=1, keepdims=True)
        unit = np.divide(direction, lengths, out=np.zeros_like(direction), where=lengths > 0)
        drawn = self.max_length * np.tanh(lengths / self.max_length) if np.isfinite(self.max_length) else lengths
        ortho = np.column_stack((-unit[:, 1], unit[:, 0], np.zeros(len(unit))))
        starts = self.sample_points
        ends = starts + drawn * unit
        shoulders = ends - np.minimum(drawn, self.tip_length) * unit
        half_width = ortho * np.minimum(drawn, self.arrow_width) / 2
        head_half_width = ortho * np.minimum(drawn, self.tip_width) / 2
        return np.stack((starts + half_width, starts - half_width,
                         shoulders + half_width, shoulders - half_width,
                         shoulders + head_half_width, shoulders - head_half_width, ends), axis=1).reshape(-1, 3)

    def update_from_tracker(self, mob):
        time = self.tracker.get_value()
        if abs(time - self._last_time) > 0.01:
            self._last_time = time
            self.set_points(self._vertices())

    def apply_radial_opacity(self):
        distances = np.linalg.norm(self.sample_points[:, :2], axis=1)
        fade = np.clip(1 - distances / self.max_radius, 0, 1)
        alpha = self.stroke_opacity * (self.min_opacity + (self.max_opacity - self.min_opacity) * fade)
        fill = np.repeat(self.paint.fill[:1], len(alpha) * 7, axis=0)
        fill[:, 3] = np.repeat(alpha, 7)
        self.paint = self.paint.but(fill=fill)

    def set_opacity_range(self, min_opacity, max_opacity):
        self.min_opacity = min_opacity
        self.max_opacity = max_opacity
        self.apply_radial_opacity()
        return self

    def set_max_radius(self, max_radius):
        self.max_radius = max_radius
        self.apply_radial_opacity()
        return self
