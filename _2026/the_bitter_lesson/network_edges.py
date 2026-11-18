"""Shared weighted edges for the neural-network diagrams."""

import manimgx as gx
import matplotlib.pyplot as plt
import numpy as np

NETWORK_INK = "#655d53"


def _circle_occlusion_interval(p0, p1, center, radius):
    """Returns the (t_enter, t_exit) portion of segment p0->p1 that falls
    inside the given circle, clipped to [0, 1], or None if it never enters."""
    d = p1 - p0
    a = np.dot(d, d)
    if a < 1e-9:
        return None
    f = p0 - center
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    t_enter, t_exit = max(min(t1, t2), 0.0), min(max(t1, t2), 1.0)
    if t_enter >= t_exit:
        return None
    return (t_enter, t_exit)


def _visible_segments(p0, p1, obstacles):
    """Splits segment p0->p1 into the sub-segments (as t-ranges) that don't
    pass through any of the given obstacle nodes' circles."""
    intervals = []
    for obstacle in obstacles:
        interval = _circle_occlusion_interval(p0, p1, obstacle.get_center(), obstacle.node_radius)
        if interval is not None:
            intervals.append(interval)

    if not intervals:
        return [(0.0, 1.0)]

    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    segments = []
    prev = 0.0
    for start, end in merged:
        if start > prev:
            segments.append((prev, start))
        prev = max(prev, end)
    if prev < 1.0:
        segments.append((prev, 1.0))
    return segments


class Arrow(gx.VGroup):
    """A weighted edge with an arrowhead and optional gaps around other nodes."""

    def __init__(
        self,
        start_obj,
        end_obj,
        stroke_width=2,
        value=0.5,
        weight=1.0,
        stroke_color=NETWORK_INK,
        dashed=False,
        buff=0.0,
        obstacles=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.start_obj = start_obj
        self.end_obj = end_obj
        self.stroke_width = abs(stroke_width * weight)
        self.weight = weight
        self.stroke_color = stroke_color
        self.dashed = dashed
        self.buff = buff
        self.obstacles = tuple(obstacles) if obstacles is not None else ()
        self.value = value
        self.stroke_opacity = float(np.clip(value, 0.1, 1.0))
        self.segments = gx.VGroup()
        self.add(self.segments)
        self.build()

    @staticmethod
    def _center(obj):
        if hasattr(obj, "get_center"):
            return np.asarray(obj.get_center(), dtype=float)
        if hasattr(obj, "get_position"):
            return np.asarray(obj.get_position(), dtype=float)
        return np.asarray(obj, dtype=float)

    def _endpoint(self, obj, target):
        center = self._center(obj)
        target_center = self._center(target)
        if hasattr(obj, "get_connection_point"):
            return obj.get_connection_point(target_center)
        if hasattr(obj, "node_radius"):
            return center + gx.normalize(target_center - center) * obj.node_radius
        return center

    def build(self):
        start = self._endpoint(self.start_obj, self.end_obj)
        end = self._endpoint(self.end_obj, self.start_obj)
        direction = end - start
        segments = _visible_segments(start, end, self.obstacles)
        style = dict(
            stroke_color=self.stroke_color,
            stroke_width=self.stroke_width,
            stroke_opacity=self.stroke_opacity,
            buff=self.buff,
        )
        self.segments.remove(*self.segments.submobjects)
        self.line = gx.Line(start, end, **style)
        for index, (t0, t1) in enumerate(segments):
            p0, p1 = start + t0 * direction, start + t1 * direction
            is_last = index == len(segments) - 1
            if self.dashed:
                segment = gx.DashedLine(p0, p1, dash_length=0.1, dashed_ratio=0.5, **style)
                if is_last:
                    segment.add_tip()
            else:
                segment = (gx.Arrow if is_last else gx.Line)(p0, p1, **style)
            self.line = segment
            self.segments.add(segment)
        return self

    def update_positions(self):
        return self.build()

    def set_value(self, new_value):
        self.value = new_value
        self.stroke_opacity = float(np.clip(new_value, 0.1, 1.0))
        self.segments.set_stroke(color=self.stroke_color, width=self.stroke_width, opacity=self.stroke_opacity)
        return self

    def pulse(self, duration=1.0, scale=1.5):
        return gx.Succession(
            self.line.animate(run_time=duration / 2).set_stroke(width=self.stroke_width * scale),
            self.line.animate(run_time=duration / 2).set_stroke(width=self.stroke_width),
        )

    def animate_flow(self, duration=1.0):
        if self.dashed:
            return gx.MoveAlongPath(gx.Dot(color=self.stroke_color), self.line.copy(), run_time=duration)
        line_copy = self.line.copy().set_stroke(width=self.stroke_width * 1.5)
        return gx.ShowPassingFlash(line_copy, time_width=0.5, run_time=duration)


class NetworkLayer(gx.VMobject):
    node_type = None

    def __init__(
        self,
        values=None,
        labels=None,
        max_display=16,
        node_radius=0.1,
        node_spacing=0.3,
        node_stroke_color=NETWORK_INK,
        node_stroke_width=6,
        colormap=plt.get_cmap("viridis"),
        position=gx.ORIGIN,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.values = values if values is not None else [0.5, 0.7, 0.3, 0.9, 0.1]
        self.labels = labels if labels is not None else ["A"] * len(self.values)
        self.max_display = max_display
        self.node_radius = node_radius
        self.node_spacing = node_spacing
        self.node_stroke_color = node_stroke_color
        self.node_stroke_width = node_stroke_width
        self.colormap = colormap
        self.position = position

        self.ellipsis_size = self.node_radius * 0.1  # Smaller ellipses
        self.ellipsis_spacing = self.node_radius * 0.5  # Closer together

        self.nodes = gx.VGroup()
        self.ellipsis_dots = gx.VGroup()

        self.build()
        self.move_to(position)

    def build(self):
        self.clear()
        self.nodes = gx.VGroup()
        self.ellipsis_dots = gx.VGroup()

        total_nodes = len(self.values)
        if total_nodes <= self.max_display:
            self._create_simple_layer()
        else:
            self._create_truncated_layer()

        self.ellipsis_dots.set_color(NETWORK_INK)

        self.add(self.nodes, self.ellipsis_dots)

    def _create_simple_layer(self):
        total_nodes = len(self.values)
        total_height = self.node_spacing * (total_nodes - 1)

        for i, value in enumerate(self.values):
            y = total_height/2 - i * self.node_spacing
            node = self.node_type(
                node_radius=self.node_radius,
                node_stroke_color=self.node_stroke_color,
                node_stroke_width=self.node_stroke_width,
                value=value,
                label=self.labels[i]
            ).move_to(gx.UP * y)
            self.nodes.add(node)

    def _create_truncated_layer(self):
        visible_per_side = self.max_display // 2  # Show equal nodes on top and bottom

        # Calculate spacing more clearly
        # Top nodes: visible_per_side nodes with (visible_per_side-1) gaps
        top_section = (visible_per_side - 1) * self.node_spacing
        # Gap before ellipses (1/2 of normal spacing)
        gap_before = self.node_spacing / 2
        # Ellipses: 3 dots with 2 gaps
        ellipsis_section = 2 * self.ellipsis_spacing
        # Gap after ellipses (1/2 of normal spacing)
        gap_after = self.node_spacing / 2
        # Bottom nodes: visible_per_side nodes with (visible_per_side-1) gaps
        bottom_section = (visible_per_side - 1) * self.node_spacing

        total_height = top_section + gap_before + ellipsis_section + gap_after + bottom_section
        start_y = total_height / 2

        # Create top nodes
        for i in range(visible_per_side):
            y = start_y - i * self.node_spacing
            node = self.node_type(
                node_radius=self.node_radius,
                node_stroke_color=self.node_stroke_color,
                node_stroke_width=self.node_stroke_width,
                value=self.values[i],
                label=self.labels[i]
            ).move_to(gx.UP * y)
            self.nodes.add(node)

        last_top_node_y = start_y - (visible_per_side - 1) * self.node_spacing
        first_ellipsis_y = last_top_node_y - gap_before

        # Create ellipsis dots
        for j in range(3):
            y_offset = first_ellipsis_y - j * self.ellipsis_spacing
            dot = gx.Dot(radius=self.ellipsis_size, color=self.node_stroke_color).move_to(gx.UP * y_offset)
            self.ellipsis_dots.add(dot)

        last_ellipsis_y = first_ellipsis_y - 2 * self.ellipsis_spacing
        first_bottom_node_y = last_ellipsis_y - gap_after

        # Create bottom nodes
        for i in range(visible_per_side):
            idx = len(self.values) - visible_per_side + i
            y = first_bottom_node_y - i * self.node_spacing
            node = self.node_type(
                node_radius=self.node_radius,
                node_stroke_color=self.node_stroke_color,
                node_stroke_width=self.node_stroke_width,
                value=self.values[idx],
                label=self.labels[idx]
            ).move_to(gx.UP * y)
            self.nodes.add(node)

    def get_node(self, index):
        if index >= len(self.values):
            return None

        if len(self.values) <= self.max_display:
            return self.nodes[index]

        visible_per_side = self.max_display // 2  # Match the formula in _create_truncated_layer
        if index < visible_per_side:
            return self.nodes[index]
        elif index >= len(self.values) - visible_per_side:
            local_index = visible_per_side + (index - (len(self.values) - visible_per_side))
            return self.nodes[local_index]
        return None

    def update_values(self, new_values):
        self.values = new_values
        self.build()

    def set_value(self, index, value):
        if 0 <= index < len(self.values):
            self.values[index] = value
            node = self.get_node(index)
            if node:
                node.value = value
                node.build()

    def highlight_node(self, index, color=gx.YELLOW):
        node = self.get_node(index)
        if node:
            original_stroke = node.node_stroke_color
            node.node_stroke_color = color
            node.build()
            return original_stroke
        return None

    def reset_highlight(self, index, original_color=None):
        if original_color is None:
            original_color = self.node_stroke_color

        node = self.get_node(index)
        if node:
            node.node_stroke_color = original_color
            node.build()

    def get_all_nodes(self):
        return self.nodes
