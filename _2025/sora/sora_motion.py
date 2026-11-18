"""Motion and visual helpers for the Sora scenes."""

import manimgx as gx
import numpy as np
from welch_style import YELLOW


def manual_camera_interpolation(start_orientation, end_orientation, num_steps):
    """
    Linearly interpolate between two camera orientations.

    Parameters:
    - start_orientation: List containing camera parameters with a tuple at index 3
    - end_orientation: List containing camera parameters with a tuple at index 3
    - num_steps: Number of interpolation steps (including start and end)

    Returns:
    - List of interpolated orientations
    """
    result = []

    for step in range(num_steps):
        # Calculate interpolation factor (0 to 1)
        t = step / (num_steps - 1) if num_steps > 1 else 0

        # Create a new orientation for this step
        interpolated = []

        for i in range(len(start_orientation)):
            if i == 3:  # Handle the tuple at position 3
                start_tuple = start_orientation[i]
                end_tuple = end_orientation[i]

                # Interpolate each element of the tuple
                interpolated_tuple = tuple(
                    start_tuple[j] + t * (end_tuple[j] - start_tuple[j])
                    for j in range(len(start_tuple))
                )

                interpolated.append(interpolated_tuple)
            else:  # Handle regular numeric values
                start_val = start_orientation[i]
                end_val = end_orientation[i]
                interpolated_val = start_val + t * (end_val - start_val)
                interpolated.append(interpolated_val)

        result.append(interpolated)

    return result


def get_color_wheel_colors(n_colors, saturation=1.0, value=1.0, start_hue=0.0):
    """
    Generate N evenly spaced colors from the color wheel.

    Args:
        n_colors: Number of colors to generate
        saturation: Color saturation (0.0 to 1.0)
        value: Color brightness/value (0.0 to 1.0)
        start_hue: Starting hue position (0.0 to 1.0)

    Returns:
        List of Manim-compatible hex color strings
    """
    import colorsys
    colors = []
    for i in range(n_colors):
        hue = (start_hue + i / n_colors) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors


class CustomTracedPath(gx.VMobject):
    """
    A custom traced path that supports:
    - Reverse playback with segment removal
    - Variable opacity based on distance from end
    - Manual control over path segments
    """
    def __init__(
        self,
        traced_point_func,
        stroke_width=2.0,
        stroke_color=YELLOW,
        opacity_range=(0.1, 0.8),  # (min_opacity, max_opacity)
        fade_length=20,  # Number of segments to fade over
        **kwargs
    ):
        super().__init__(**kwargs)
        self.traced_point_func = traced_point_func
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.opacity_range = opacity_range
        self.fade_length = fade_length

        # Store path segments as individual VMobjects
        self.segments = gx.VGroup()
        self.traced_points = []
        self.is_tracing = True

        # Add updater for forward tracing
        self.add_updater(lambda m, dt: m.update_path(dt))

    def update_path(self, dt=0):
        """Update path during forward animation"""
        if not self.is_tracing or dt == 0:
            return

        point = self.traced_point_func()
        self.traced_points.append(point.copy())

        if len(self.traced_points) >= 2:
            # Create a new segment
            segment = gx.Line(
                self.traced_points[-2],
                self.traced_points[-1],
                stroke_width=self.stroke_width,
                stroke_color=self.stroke_color
            )

            # Apply opacity gradient
            self.segments.add(segment)
            self.update_segment_opacities()
            self.add(segment)

    def update_segment_opacities(self):
        """Update opacity of all segments based on their position"""
        n_segments = len(self.segments)
        if n_segments == 0:
            return

        min_op, max_op = self.opacity_range

        for i, segment in enumerate(self.segments):
            if i >= n_segments - self.fade_length:
                # Calculate fade based on distance from end
                fade_progress = (i - (n_segments - self.fade_length)) / self.fade_length
                opacity = min_op + (max_op - min_op) * fade_progress
            else:
                opacity = min_op
            segment.set_opacity(opacity)

    def remove_last_segment(self):
        """Remove the last segment (for reverse playback)
        Kinda hacky but just run 2x to fix bug
        """
        if len(self.segments) > 0:
            last_segment = self.segments[-1]
            self.segments.remove(last_segment)
            self.remove(last_segment)
            if len(self.traced_points) > 0:
                self.traced_points.pop()
            # self.update_segment_opacities()

        if len(self.segments) > 0:
            last_segment = self.segments[-1]
            self.segments.remove(last_segment)
            self.remove(last_segment)
            if len(self.traced_points) > 0:
                self.traced_points.pop()

        self.update_segment_opacities()

    def stop_tracing(self):
        """Stop the automatic tracing updater"""
        self.is_tracing = False

    def start_tracing(self):
        """Resume automatic tracing"""
        self.is_tracing = True

    def get_num_segments(self):
        """Get the current number of segments"""
        return len(self.segments)


def create_noisy_arrow_animation(self, start_point, end_point, target_point, num_steps=100, noise_level=0.1, overshoot_factor=0.3):
    """
    Creates a sequence of arrow end positions that converge from end_point to target_point
    with parameterizable noise and overshoot past the target direction.
    """

    # Calculate initial and target directions
    initial_direction = np.array(end_point) - np.array(start_point)
    target_direction = np.array(target_point) - np.array(start_point)

    # Calculate the constant length
    arrow_length = np.linalg.norm(initial_direction)

    # Calculate angles
    initial_angle = np.arctan2(initial_direction[1], initial_direction[0])
    target_angle = np.arctan2(target_direction[1], target_direction[0])

    # Handle angle wrapping (choose the shorter path)
    angle_diff = target_angle - initial_angle
    if angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2 * np.pi

    # Create interpolation parameter
    t_values = np.linspace(0, 1, num_steps)

    # Generate noise that decreases over time
    np.random.seed(42)
    noise_decay = np.exp(-3 * t_values)
    angle_noise = noise_level * noise_decay * np.random.randn(num_steps)

    # Generate overshoot in angle space - this will make it swing past the target angle
    overshoot_frequency = 3.0
    overshoot_decay = np.exp(-2 * t_values)
    overshoot_oscillation = overshoot_factor * overshoot_decay * np.sin(overshoot_frequency * np.pi * t_values)

    # The key: let the angle interpolation overshoot past the target
    t_effective = t_values + overshoot_oscillation
    # Ensure final angle is exactly the target
    t_effective[-1] = 1.0

    arrow_positions = []

    for i, t_eff in enumerate(t_effective):
        # Interpolate angle - this is where the overshoot happens
        current_angle = initial_angle + t_eff * angle_diff

        # Add angular noise (but not on final step)
        if i < len(t_effective) - 1:
            current_angle += angle_noise[i]

        # Convert back to cartesian coordinates
        end_x = np.array(start_point)[0] + arrow_length * np.cos(current_angle)
        end_y = np.array(start_point)[1] + arrow_length * np.sin(current_angle)

        arrow_positions.append([end_x, end_y, 0])

    return arrow_positions
