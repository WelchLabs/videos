from welch_paths import welch_path
import manimgx as gx
from welch_scene import WelchScene
from welch_text import math_tex

from functools import partial
import numpy as np
import sys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import glob

CHILL_BROWN = "#948979"
FRESH_TAN = "#efe5d1"
YELLOW = "#ffd35a"
BLUE = "#65c8d0"
GREEN = "#00a14b"
THUNDER = "#1b1619"

ASSET_PATH = welch_path('double_descent/hackin')


def get_edge_points(circle1, circle2, neuron_radius):
    direction = circle2.get_center() - circle1.get_center()
    unit_vector = direction / np.linalg.norm(direction)

    start_point = circle1.get_center() + unit_vector * neuron_radius
    end_point = circle2.get_center() - unit_vector * neuron_radius

    return start_point, end_point


viridis_colormap = plt.get_cmap("viridis")
blues_colormap = plt.get_cmap("Blues")
custom_cmap_tans = mcolors.LinearSegmentedColormap.from_list(
    "custom", ["#000000", "#dfd0b9"], N=256
)
custom_cmap_cyan = mcolors.LinearSegmentedColormap.from_list(
    "custom", ["#000000", "#00FFFF"], N=256
)


def get_neuron_color(value, vmax=0.95):
    value_clipped = np.clip(np.abs(value) / vmax, 0, 1)
    rgba = custom_cmap_tans(value_clipped)
    return gx.rgb_to_color(rgba[:3])


def get_grad_color(value):
    value_clipped = np.clip(np.abs(value), 0, 1)
    rgba = custom_cmap_cyan(value_clipped)
    return gx.rgb_to_color(rgba[:3])


def line_circle_intersection(line_start, line_end, circle_center, radius):
    """
    Check if a line segment intersects with a circle and return intersection points.
    Returns a list of t values (0 to 1) where intersections occur along the line.
    """
    direction = line_end - line_start
    to_center = line_start - circle_center

    a = np.dot(direction, direction)
    b = 2 * np.dot(to_center, direction)
    c = np.dot(to_center, to_center) - radius * radius

    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return []

    discriminant_sqrt = np.sqrt(discriminant)
    t1 = (-b - discriminant_sqrt) / (2 * a)
    t2 = (-b + discriminant_sqrt) / (2 * a)

    intersections = []
    if 0 < t1 < 1:
        intersections.append(t1)
    if 0 < t2 < 1:
        intersections.append(t2)

    return sorted(intersections)


def create_split_line(neuron1, neuron2, all_neurons, neuron_radius, exclude_neurons=None):
    """
    Create a line that splits around neurons it passes through.
    Returns a VGroup of line segments that avoid neurons.
    The line segments touch the edges of the source and destination neurons.
    """
    if exclude_neurons is None:
        exclude_neurons = set()

    center1 = neuron1.get_center()
    center2 = neuron2.get_center()
    
    direction = center2 - center1
    unit_vector = direction / np.linalg.norm(direction)
    
    full_start = center1 + unit_vector * neuron_radius
    full_end = center2 - unit_vector * neuron_radius

    t_values = [0.0]

    for neuron in all_neurons:
        if neuron in exclude_neurons:
            continue

        intersections = line_circle_intersection(
            full_start, full_end, neuron.get_center(), neuron_radius
        )
        t_values.extend(intersections)

    t_values.append(1.0)
    t_values = sorted(set(t_values))

    line_segments = gx.VGroup()
    for i in range(len(t_values) - 1):
        t_start = t_values[i]
        t_end = t_values[i + 1]
        t_mid = (t_start + t_end) / 2

        mid_point = full_start + t_mid * (full_end - full_start)

        is_inside_neuron = False
        for neuron in all_neurons:
            if neuron in exclude_neurons:
                continue
            dist = np.linalg.norm(mid_point - neuron.get_center())
            if dist < neuron_radius * 0.95:
                is_inside_neuron = True
                break

        if not is_inside_neuron:
            seg_start = full_start + t_start * (full_end - full_start)
            seg_end = full_start + t_end * (full_end - full_start)
            segment = gx.Line(seg_start, seg_end)
            line_segments.add(segment)

    return line_segments


class AttentionPattern(gx.VMobject):
    def __init__(
        self,
        matrix,
        square_size=0.3,
        min_opacity=0.0,
        max_opacity=1.0,
        stroke_width=1.0,
        viz_scaling_factor=2.5,
        stroke_color=FRESH_TAN,
        colormap=custom_cmap_tans,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.matrix = np.array(matrix)
        self.n_rows, self.n_cols = self.matrix.shape
        self.square_size = square_size
        self.min_opacity = min_opacity
        self.max_opacity = np.max(self.matrix)
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self._colormap = colormap
        self.viz_scaling_factor = viz_scaling_factor

        self.build()

    def map_value_to_style(self, val):
        val_scaled = np.clip(self.viz_scaling_factor * val / self.max_opacity, 0, 1)
        rgba = self._colormap(val_scaled)
        color = gx.rgb_to_color(rgba[:3])
        opacity = 1.0
        return {"color": color, "opacity": opacity}

    def build(self):
        self.clear()
        squares = gx.VGroup()
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                val = self.matrix[i, j]
                style = self.map_value_to_style(val)

                square = gx.Square(side_length=self.square_size)
                square.set_fill(style["color"], opacity=style["opacity"])
                square.set_stroke(self.stroke_color, width=self.stroke_width)

                pos = gx.RIGHT * j * self.square_size + gx.DOWN * i * self.square_size
                square.move_to(pos)
                squares.add(square)

        squares.move_to(gx.ORIGIN)
        self.add(squares)


def get_mlp(
    w1,
    w2,
    neuron_fills=None,
    grads_1=None,
    grads_2=None,
    line_weight=1.0,
    line_opacity=0.5,
    neuron_stroke_width=1.0,
    neuron_stroke_color="#dfd0b9",
    line_stroke_color="#948979",
    connection_display_thresh=0.4,
):

    INPUT_NEURONS = w1.shape[0]
    HIDDEN_NEURONS = w1.shape[1]
    OUTPUT_NEURONS = w1.shape[0]
    NEURON_RADIUS = 0.06
    LAYER_SPACING = 0.23
    VERTICAL_SPACING = 0.18
    DOTS_SCALE = 0.5

    input_layer = gx.VGroup()
    hidden_layer = gx.VGroup()
    output_layer = gx.VGroup()
    dots = gx.VGroup()

    neuron_count = 0
    for i in range(INPUT_NEURONS):
        if i == w1.shape[0] // 2:
            dot = (
                math_tex("...")
                .rotate(gx.PI / 2, gx.OUT)
                .scale(DOTS_SCALE)
                .move_to(
                    gx.LEFT * LAYER_SPACING
                    + gx.UP * ((INPUT_NEURONS // 2 - i) * VERTICAL_SPACING)
                )
            )
            dot.set_color(neuron_stroke_color)
            dots.add(dot)
        else:
            neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
            neuron.set_stroke(width=neuron_stroke_width)
            if neuron_fills is None:
                neuron.set_fill(color="#000000", opacity=1.0)
            else:
                neuron.set_fill(
                    color=get_neuron_color(
                        neuron_fills[0][neuron_count],
                        vmax=np.abs(neuron_fills[0]).max(),
                    ),
                    opacity=1.0,
                )
            neuron.move_to(
                gx.LEFT * LAYER_SPACING
                + gx.UP * ((INPUT_NEURONS // 2 - i) * VERTICAL_SPACING)
            )
            input_layer.add(neuron)
            neuron_count += 1

    neuron_count = 0
    for i in range(HIDDEN_NEURONS):
        if i == w1.shape[1] // 2:
            dot = (
                math_tex("...")
                .rotate(gx.PI / 2, gx.OUT)
                .scale(DOTS_SCALE)
                .move_to(gx.UP * ((HIDDEN_NEURONS // 2 - i) * VERTICAL_SPACING))
            )
            dot.set_color(neuron_stroke_color)
            dots.add(dot)
        else:
            neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
            neuron.set_stroke(width=neuron_stroke_width)
            if neuron_fills is None:
                neuron.set_fill(color="#000000", opacity=1.0)
            else:
                neuron.set_fill(
                    color=get_neuron_color(
                        neuron_fills[1][neuron_count],
                        vmax=np.abs(neuron_fills[1]).max(),
                    ),
                    opacity=1.0,
                )
            neuron.move_to(gx.UP * ((HIDDEN_NEURONS // 2 - i) * VERTICAL_SPACING))
            hidden_layer.add(neuron)
            neuron_count += 1

    neuron_count = 0
    for i in range(OUTPUT_NEURONS):
        if i == w1.shape[0] // 2:
            dot = (
                math_tex("...")
                .rotate(gx.PI / 2, gx.OUT)
                .scale(DOTS_SCALE)
                .move_to(
                    gx.RIGHT * LAYER_SPACING
                    + gx.UP * ((OUTPUT_NEURONS // 2 - i) * VERTICAL_SPACING)
                )
            )
            dot.set_color(neuron_stroke_color)
            dots.add(dot)
        else:
            neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
            neuron.set_stroke(width=neuron_stroke_width)
            if neuron_fills is None:
                neuron.set_fill(color="#000000", opacity=1.0)
            else:
                neuron.set_fill(
                    color=get_neuron_color(
                        neuron_fills[2][neuron_count],
                        vmax=np.abs(neuron_fills[2]).max(),
                    ),
                    opacity=1.0,
                )
            neuron.move_to(
                gx.RIGHT * LAYER_SPACING
                + gx.UP * ((OUTPUT_NEURONS // 2 - i) * VERTICAL_SPACING)
            )
            output_layer.add(neuron)
            neuron_count += 1

    connections = gx.VGroup()
    w1_abs = np.abs(w1)
    w1_scaled = w1_abs / np.percentile(w1_abs, 99)
    for i, in_neuron in enumerate(input_layer):
        for j, hidden_neuron in enumerate(hidden_layer):
            if np.abs(w1_scaled[i, j]) < 0.75:
                continue
            if abs(i - j) > 6:
                continue
            start_point, end_point = get_edge_points(
                in_neuron, hidden_neuron, NEURON_RADIUS
            )
            line = gx.Line(start_point, end_point)

            line.set_stroke(
                opacity=np.clip(w1_scaled[i, j], 0, 1), width=1.0 * w1_scaled[i, j]
            )

            line.set_color(line_stroke_color)
            connections.add(line)

    w2_abs = np.abs(w2)
    w2_scaled = w2_abs / np.percentile(w2_abs, 99)
    for i, hidden_neuron in enumerate(hidden_layer):
        for j, out_neuron in enumerate(output_layer):
            if np.abs(w2_scaled[i, j]) < 0.45:
                continue
            if abs(i - j) > 6:
                continue
            start_point, end_point = get_edge_points(
                hidden_neuron, out_neuron, NEURON_RADIUS
            )
            line = gx.Line(start_point, end_point)
            line.set_stroke(
                opacity=np.clip(w2_scaled[i, j], 0, 1), width=1.0 * w2_scaled[i, j]
            )
            line.set_color(line_stroke_color)
            connections.add(line)

    grad_conections = gx.VGroup()
    if grads_1 is not None:
        grads_1_abs = np.abs(grads_1)
        grads_1_scaled = grads_1_abs / np.percentile(grads_1_abs, 95)
        for i, in_neuron in enumerate(input_layer):
            for j, hidden_neuron in enumerate(hidden_layer):
                if np.abs(grads_1_scaled[i, j]) < 0.5:
                    continue
                if abs(i - j) > 6:
                    continue
                start_point, end_point = get_edge_points(
                    in_neuron, hidden_neuron, NEURON_RADIUS
                )
                line_grad = gx.Line(start_point, end_point)
                line_grad.set_stroke(
                    opacity=np.clip(grads_1_scaled[i, j], 0, 1),
                    width=np.clip(2.0 * grads_1_scaled[i, j], 0, 3),
                )
                line_grad.set_color(get_grad_color(grads_1_scaled[i, j]))
                grad_conections.add(line_grad)

    if grads_2 is not None:
        grads_2_abs = np.abs(grads_2)
        grads_2_scaled = grads_2_abs / np.percentile(grads_2_abs, 97)
        for i, hidden_neuron in enumerate(hidden_layer):
            for j, out_neuron in enumerate(output_layer):
                if np.abs(grads_2_scaled[i, j]) < 0.5:
                    continue
                if abs(i - j) > 6:
                    continue
                start_point, end_point = get_edge_points(
                    hidden_neuron, out_neuron, NEURON_RADIUS
                )
                line_grad = gx.Line(start_point, end_point)
                line_grad.set_stroke(
                    opacity=np.clip(grads_2_scaled[i, j], 0, 1),
                    width=np.clip(1.0 * grads_2_scaled[i, j], 0, 3),
                )
                line_grad.set_color(get_grad_color(grads_2_scaled[i, j]))
                grad_conections.add(line_grad)

    return gx.VGroup(
        connections, grad_conections, input_layer, hidden_layer, output_layer, dots
    )


def get_attention_layer(attn_patterns):
    num_attention_pattern_slots = len(attn_patterns) + 1
    attention_pattern_spacing = 0.51

    attention_border = gx.RoundedRectangle(width=0.59, height=5.4, corner_radius=0.1)
    attention_border.set_stroke(width=1.0, color=FRESH_TAN)

    attention_patterns = gx.VGroup()
    connection_points_left = gx.VGroup()
    connection_points_right = gx.VGroup()

    attn_pattern_count = 0
    for i in range(num_attention_pattern_slots):
        if i == num_attention_pattern_slots // 2:
            dot = (
                math_tex("...")
                .rotate(gx.PI / 2, gx.OUT)
                .scale(0.5)
                .move_to(
                    [
                        0,
                        num_attention_pattern_slots * attention_pattern_spacing / 2
                        - attention_pattern_spacing * (i + 0.5),
                        0,
                    ]
                )
            )
            dot.set_color(FRESH_TAN)
            attention_patterns.add(dot)
        else:
            if i > num_attention_pattern_slots // 2:
                offset = 0.15
            else:
                offset = -0.15
            attn_pattern = AttentionPattern(
                matrix=attn_patterns[attn_pattern_count],
                square_size=0.07,
                stroke_width=0.5,
            )
            attn_pattern.move_to(
                [
                    0,
                    num_attention_pattern_slots * attention_pattern_spacing / 2
                    + offset
                    - attention_pattern_spacing * (i + 0.5),
                    0,
                ]
            )
            attention_patterns.add(attn_pattern)

            connection_point_left = gx.Circle(radius=0)
            connection_point_left.move_to(
                [
                    -0.59 / 2.0,
                    num_attention_pattern_slots * attention_pattern_spacing / 2
                    + offset
                    - attention_pattern_spacing * (i + 0.5),
                    0,
                ]
            )
            connection_points_left.add(connection_point_left)

            connection_point_right = gx.Circle(radius=0)
            connection_point_right.move_to(
                [
                    0.59 / 2.0,
                    num_attention_pattern_slots * attention_pattern_spacing / 2
                    + offset
                    - attention_pattern_spacing * (i + 0.5),
                    0,
                ]
            )
            connection_points_right.add(connection_point_right)
            attn_pattern_count += 1

    attention_layer = gx.VGroup(
        attention_patterns,
        attention_border,
        connection_points_left,
        connection_points_right,
    )
    return attention_layer


def get_mlp_connections_left(
    attention_connections_left,
    mlp_out,
    connection_points_left,
    attention_connections_left_grad=None,
):
    connections_left = gx.VGroup()
    attention_connections_left_abs = np.abs(attention_connections_left)
    attention_connections_left_scaled = attention_connections_left_abs / np.max(
        attention_connections_left_abs
    )  # np.percentile(attention_connections_left_abs, 99)
    for i, mlp_out_neuron in enumerate(mlp_out):
        for j, attention_neuron in enumerate(connection_points_left):
            if np.abs(attention_connections_left_scaled[i, j]) < 0.5:
                continue
            if abs(i / 4 - j) > 3:
                continue  # Need to dial this up or lost it probably, but it is helpful!
            start_point, end_point = get_edge_points(
                mlp_out_neuron, attention_neuron, 0.06
            )
            line = gx.Line(start_point, attention_neuron.get_center())
            # line.set_stroke(width=1, opacity=0.3)
            # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
            line.set_stroke(
                opacity=np.clip(attention_connections_left_scaled[i, j], 0, 1),
                width=np.clip(1.0 * attention_connections_left_scaled[i, j], 0, 3),
            )
            line.set_color(FRESH_TAN)
            connections_left.add(line)

    connections_left_grads = gx.VGroup()
    if attention_connections_left_grad is not None:
        attention_connections_left_grad_abs = np.abs(attention_connections_left_grad)
        attention_connections_left_grad_scaled = (
            attention_connections_left_grad_abs
            / np.percentile(attention_connections_left_grad_abs, 98)
        )
        for i, mlp_out_neuron in enumerate(mlp_out):
            for j, attention_neuron in enumerate(connection_points_left):
                if np.abs(attention_connections_left_grad_scaled[i, j]) < 0.5:
                    continue
                if abs(i / 4 - j) > 3:
                    continue
                start_point, end_point = get_edge_points(
                    mlp_out_neuron, attention_neuron, 0.06
                )
                line = gx.Line(start_point, attention_neuron.get_center())
                line.set_stroke(
                    opacity=np.clip(attention_connections_left_grad_scaled[i, j], 0, 1),
                    width=np.clip(
                        1.0 * attention_connections_left_grad_scaled[i, j], 0, 2
                    ),
                )
                line.set_color(
                    get_grad_color(attention_connections_left_grad_scaled[i, j])
                )
                connections_left_grads.add(line)
    return connections_left, connections_left_grads


def get_mlp_connections_right(
    attention_connections_right,
    mlp_in,
    connection_points_right,
    attention_connections_right_grad=None,
):
    connections_right = gx.VGroup()
    attention_connections_right_abs = np.abs(attention_connections_right)
    attention_connections_right_scaled = (
        attention_connections_right_abs
        / np.percentile(attention_connections_right_abs, 99)
    )
    for i, attention_neuron in enumerate(connection_points_right):
        for j, mlp_in_neuron in enumerate(mlp_in):
            if np.abs(attention_connections_right_scaled[i, j]) < 0.6:
                continue
            if abs(j / 4 - i) > 3:
                continue  # Need to dial this up or lost it probably, but it is helpful!
            start_point, end_point = get_edge_points(
                mlp_in_neuron, attention_neuron, 0.06
            )
            line = gx.Line(start_point, attention_neuron.get_center())
            # line.set_stroke(width=1, opacity=0.3)
            # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
            line.set_stroke(
                opacity=np.clip(attention_connections_right_scaled[i, j], 0, 1),
                width=np.clip(1.0 * attention_connections_right_scaled[i, j], 0, 3),
            )
            line.set_color(FRESH_TAN)
            connections_right.add(line)

    connections_right_grads = gx.VGroup()
    if attention_connections_right_grad is not None:
        attention_connections_right_grad_abs = np.abs(attention_connections_right_grad)
        attention_connections_right_grad_scaled = (
            attention_connections_right_grad_abs
            / np.percentile(attention_connections_right_grad_abs, 98)
        )
        for i, attention_neuron in enumerate(connection_points_right):
            for j, mlp_in_neuron in enumerate(mlp_in):
                if np.abs(attention_connections_right_grad_scaled[i, j]) < 0.5:
                    continue
                if abs(j / 4 - i) > 3:
                    continue
                start_point, end_point = get_edge_points(
                    mlp_in_neuron, attention_neuron, 0.06
                )
                line = gx.Line(start_point, attention_neuron.get_center())
                line.set_stroke(
                    opacity=np.clip(attention_connections_right_grad_scaled[i, j], 0, 1),
                    width=np.clip(
                        1.0 * attention_connections_right_grad_scaled[i, j], 0, 3
                    ),
                )
                line.set_color(
                    get_grad_color(attention_connections_right_grad_scaled[i, j])
                )
                connections_right_grads.add(line)
    return connections_right, connections_right_grads


def get_input_layer(prompt_neuron_indices, snapshot, num_input_neurons=36):
    input_layer_neurons = gx.VGroup()
    input_layer_text = gx.VGroup()
    vertical_spacing = 0.18
    neuron_radius = 0.06
    neuron_stroke_color = "#dfd0b9"
    neuron_stroke_width = 1.0
    words_to_nudge = {" capital": -0.02}

    prompt_token_count = 0
    neuron_count = 0
    for i in range(num_input_neurons):
        if i == num_input_neurons // 2:
            dot = (
                math_tex("...")
                .rotate(gx.PI / 2, gx.OUT)
                .scale(0.4)
                .move_to(gx.UP * ((num_input_neurons // 2 - i) * vertical_spacing))
            )
            dot.set_color(neuron_stroke_color)
        else:
            neuron = gx.Circle(radius=neuron_radius, stroke_color=neuron_stroke_color)
            neuron.set_stroke(width=neuron_stroke_width)
            if neuron_count in prompt_neuron_indices:
                neuron.set_fill(color="#dfd0b9", opacity=1.0)
                t = gx.Text(
                    snapshot["prompt.tokens"][prompt_token_count],
                    font_size=24,
                    font="myriad-pro",
                )
                t.set_color(neuron_stroke_color)
                t.move_to(
                    (0.2 + t.get_right()[0]) * gx.LEFT
                    + gx.UP
                    * (
                        (-t.get_bottom() + num_input_neurons // 2 - i)
                        * vertical_spacing
                    )
                )
                token = snapshot["prompt.tokens"][prompt_token_count]
                if token in words_to_nudge:
                    t.shift([0, words_to_nudge[token], 0])

                input_layer_text.add(t)
                prompt_token_count += 1
            else:
                neuron.set_fill(color="#000000", opacity=1.0)

            neuron.move_to(gx.UP * ((num_input_neurons // 2 - i) * vertical_spacing))
            input_layer_neurons.add(neuron)
            neuron_count += 1

    input_layer = gx.VGroup(input_layer_neurons, dot, input_layer_text)
    return input_layer


def get_output_layer(snapshot, empty=False):
    output_layer_neurons = gx.VGroup()
    output_layer_text = gx.VGroup()
    num_output_neurons = 36
    vertical_spacing = 0.18
    neuron_radius = 0.06
    neuron_stroke_color = "#dfd0b9"
    neuron_stroke_width = 1.0

    neuron_count = 0
    for i in range(num_output_neurons):
        if i == num_output_neurons // 2:
            dot = (
                math_tex("...")
                .rotate(gx.PI / 2, gx.OUT)
                .scale(0.4)
                .move_to(gx.UP * ((num_output_neurons // 2 - i) * vertical_spacing))
            )
            dot.set_color(neuron_stroke_color)
        else:
            n = gx.Circle(radius=neuron_radius, stroke_color=neuron_stroke_color)
            n.set_stroke(width=neuron_stroke_width)
            if not empty:
                n.set_fill(
                    color=get_neuron_color(
                        snapshot["topk.probs"][neuron_count],
                        vmax=np.max(snapshot["topk.probs"]),
                    ),
                    opacity=1.0,
                )
                if neuron_count == 0:
                    font_size = 22
                elif neuron_count < 4:
                    font_size = 16
                else:
                    font_size = 12
                t = gx.Text(
                    snapshot["topk.tokens"][neuron_count],
                    font_size=font_size,
                    font="myriad-pro",
                )
                text_color = get_neuron_color(
                    np.clip(snapshot["topk.probs"][neuron_count], 0.1, 1.0),
                    vmax=np.max(snapshot["topk.probs"]),
                )
                t.set_color(text_color)
                t.set_opacity(np.clip(snapshot["topk.probs"][neuron_count], 0.3, 1.0))
                t.move_to(
                    (0.2 + t.get_right()[0]) * gx.RIGHT
                    + gx.UP
                    * (
                        (-t.get_bottom() + num_output_neurons // 2 - i)
                        * vertical_spacing
                    )
                )
                output_layer_text.add(t)

            else:
                n.set_fill(color="#000000", opacity=1.0)

            n.move_to(gx.UP * ((num_output_neurons // 2 - i) * vertical_spacing))
            output_layer_neurons.add(n)
            neuron_count += 1
    output_layer = gx.VGroup(output_layer_neurons, dot, output_layer_text)
    return output_layer

class P18(WelchScene):
    def construct(self):
        layer_spacing = 0.22
        neuron_radius = 0.06
        vertical_spacing = 0.14
        dots_scale = 0.25
        visible_neurons = 8

        neuron_layers = gx.VGroup()
        dots = gx.VGroup()

        def get_vertical_positions():
            half = visible_neurons // 2
            top_y = [vertical_spacing * (half - i) for i in range(half)]
            bottom_y = [vertical_spacing * (-(i + 1)) for i in range(half)]
            return top_y, bottom_y

        for layer_idx in range(5):
            group = gx.VGroup()
            x_pos = layer_idx * layer_spacing
            if layer_idx < 4:
                top_y, bottom_y = get_vertical_positions()
                for y in top_y:
                    neuron = gx.Circle(radius=neuron_radius, stroke_color=FRESH_TAN)
                    neuron.set_stroke(width=2)
                    neuron.set_fill(THUNDER, 1)
                    neuron.set_stroke(gx.WHITE, width=2)
                    neuron.move_to(gx.RIGHT * x_pos + gx.UP * y)
                    group.add(neuron)
                dot = (
                    math_tex("...")
                    .rotate(gx.PI / 2, gx.OUT)
                    .scale(dots_scale)
                    .set_color(FRESH_TAN)
                ).set_opacity(0.5)
                dot.move_to(gx.RIGHT * x_pos)
                dots.add(dot)
                for y in bottom_y:
                    neuron = gx.Circle(radius=neuron_radius, stroke_color=FRESH_TAN)
                    neuron.set_stroke(width=2)
                    neuron.set_fill(THUNDER, 1)
                    neuron.set_stroke(gx.WHITE, width=2)
                    neuron.move_to(gx.RIGHT * x_pos + gx.UP * y)
                    group.add(neuron)
            else:
                top_y, bottom_y = get_vertical_positions()
                for y in top_y[2:]:
                    neuron = gx.Circle(radius=neuron_radius, stroke_color=FRESH_TAN)
                    neuron.set_stroke(width=2)
                    neuron.set_fill(THUNDER, 1)
                    neuron.set_stroke(gx.WHITE, width=2)
                    neuron.move_to(gx.RIGHT * x_pos + gx.UP * y)
                    group.add(neuron)
                dot = (
                    math_tex("...")
                    .rotate(gx.PI / 2, gx.OUT)
                    .scale(dots_scale)
                    .set_color(FRESH_TAN)
                ).set_opacity(0.5)
                dot.move_to(gx.RIGHT * x_pos)
                dots.add(dot)
                for y in bottom_y[:-2]:
                    neuron = gx.Circle(radius=neuron_radius, stroke_color=FRESH_TAN)
                    neuron.set_stroke(width=2)
                    neuron.set_fill(THUNDER, 1)
                    neuron.set_stroke(gx.WHITE, width=2)
                    neuron.move_to(gx.RIGHT * x_pos + gx.UP * y)
                    group.add(neuron)
            neuron_layers.add(group)

        all_neurons = gx.VGroup()
        for layer in neuron_layers:
            all_neurons.add(*layer)

        all_lines = gx.VGroup()
        neuron_to_lines = {}
        
        for neuron in all_neurons:
            neuron_to_lines[neuron] = gx.VGroup()

        for layer_idx, layer in enumerate(neuron_layers):
            if layer_idx < len(neuron_layers) - 1:
                lines = gx.VGroup()
                current_layer = neuron_layers[layer_idx]
                next_layer = neuron_layers[layer_idx + 1]

                for neuron1 in current_layer:
                    for neuron2 in next_layer:
                        line_segments = create_split_line(
                            neuron1,
                            neuron2,
                            all_neurons,
                            neuron_radius,
                            exclude_neurons={neuron1, neuron2}
                        )

                        for segment in line_segments:
                            segment.set_stroke(FRESH_TAN, width=2.0, opacity=0.6)
                            neuron_to_lines[neuron1].add(segment)
                            neuron_to_lines[neuron2].add(segment)

                        lines.add(line_segments)

                all_lines.add(lines)
                
        network = gx.VGroup(all_lines, neuron_layers, dots)
        network.scale(4).move_to(gx.ORIGIN)

        self.add(network)
        self.wait(2)

        first_four_layers_neurons = gx.VGroup()
        for layer_idx in range(4):
            first_four_layers_neurons.add(*neuron_layers[layer_idx])

        for iteration in range(3):
            neurons_list = list(first_four_layers_neurons)
            total_neurons = len(neurons_list)
            num_to_disable = int(total_neurons * 0.3)
            indices_to_disable = np.random.choice(
                total_neurons, 
                size=num_to_disable, 
                replace=False
            )
            neurons_to_disable = [neurons_list[i] for i in indices_to_disable]

            disable_animations = []
            for neuron in neurons_to_disable:
                disable_animations.append(neuron.animate.set_opacity(0.2))
                for line_segment in neuron_to_lines[neuron]:
                    disable_animations.append(line_segment.animate.set_opacity(0.2))

            self.play(*disable_animations, run_time=1.0)
            self.wait(1)

            enable_animations = []
            for neuron in neurons_to_disable:
                enable_animations.append(neuron.animate.set_opacity(1.0))
                for line_segment in neuron_to_lines[neuron]:
                    enable_animations.append(line_segment.animate.set_opacity(1.0))

            self.play(*enable_animations, run_time=1.0)
            self.wait(1)

        # Scene authoring stops here in the interactive editor.

class P18_Long(WelchScene):
    def construct(self):
        layer_spacing = 0.22
        neuron_radius = 0.06
        vertical_spacing = 0.14
        dots_scale = 0.25
        visible_neurons = 8

        neuron_layers = gx.VGroup()
        dots = gx.VGroup()

        def get_vertical_positions():
            half = visible_neurons // 2
            top_y = [vertical_spacing * (half - i) for i in range(half)]
            bottom_y = [vertical_spacing * (-(i + 1)) for i in range(half)]
            return top_y, bottom_y

        for layer_idx in range(5):
            group = gx.VGroup()
            x_pos = layer_idx * layer_spacing
            if layer_idx < 4:
                top_y, bottom_y = get_vertical_positions()
                for y in top_y:
                    neuron = gx.Circle(radius=neuron_radius, stroke_color=FRESH_TAN)
                    neuron.set_stroke(width=2)
                    neuron.set_fill(THUNDER, 1)
                    neuron.set_stroke(gx.WHITE, width=2)
                    neuron.move_to(gx.RIGHT * x_pos + gx.UP * y)
                    group.add(neuron)
                dot = (
                    math_tex("...")
                    .rotate(gx.PI / 2, gx.OUT)
                    .scale(dots_scale)
                    .set_color(FRESH_TAN)
                ).set_opacity(0.5)
                dot.move_to(gx.RIGHT * x_pos)
                dots.add(dot)
                for y in bottom_y:
                    neuron = gx.Circle(radius=neuron_radius, stroke_color=FRESH_TAN)
                    neuron.set_stroke(width=2)
                    neuron.set_fill(THUNDER, 1)
                    neuron.set_stroke(gx.WHITE, width=2)
                    neuron.move_to(gx.RIGHT * x_pos + gx.UP * y)
                    group.add(neuron)
            else:
                top_y, bottom_y = get_vertical_positions()
                for y in top_y[2:]:
                    neuron = gx.Circle(radius=neuron_radius, stroke_color=FRESH_TAN)
                    neuron.set_stroke(width=2)
                    neuron.set_fill(THUNDER, 1)
                    neuron.set_stroke(gx.WHITE, width=2)
                    neuron.move_to(gx.RIGHT * x_pos + gx.UP * y)
                    group.add(neuron)
                dot = (
                    math_tex("...")
                    .rotate(gx.PI / 2, gx.OUT)
                    .scale(dots_scale)
                    .set_color(FRESH_TAN)
                ).set_opacity(0.5)
                dot.move_to(gx.RIGHT * x_pos)
                dots.add(dot)
                for y in bottom_y[:-2]:
                    neuron = gx.Circle(radius=neuron_radius, stroke_color=FRESH_TAN)
                    neuron.set_stroke(width=2)
                    neuron.set_fill(THUNDER, 1)
                    neuron.set_stroke(gx.WHITE, width=2)
                    neuron.move_to(gx.RIGHT * x_pos + gx.UP * y)
                    group.add(neuron)
            neuron_layers.add(group)

        all_neurons = gx.VGroup()
        for layer in neuron_layers:
            all_neurons.add(*layer)

        all_lines = gx.VGroup()
        neuron_to_lines = {}
        
        for neuron in all_neurons:
            neuron_to_lines[neuron] = gx.VGroup()

        for layer_idx, layer in enumerate(neuron_layers):
            if layer_idx < len(neuron_layers) - 1:
                lines = gx.VGroup()
                current_layer = neuron_layers[layer_idx]
                next_layer = neuron_layers[layer_idx + 1]

                for neuron1 in current_layer:
                    for neuron2 in next_layer:
                        line_segments = create_split_line(
                            neuron1,
                            neuron2,
                            all_neurons,
                            neuron_radius,
                            exclude_neurons={neuron1, neuron2}
                        )

                        for segment in line_segments:
                            segment.set_stroke(FRESH_TAN, width=2.0, opacity=0.6)
                            neuron_to_lines[neuron1].add(segment)
                            neuron_to_lines[neuron2].add(segment)

                        lines.add(line_segments)

                all_lines.add(lines)
                
        network = gx.VGroup(all_lines, neuron_layers, dots)
        network.scale(4).move_to(gx.ORIGIN)

        self.add(network)
        self.wait(2)

        first_four_layers_neurons = gx.VGroup()
        for layer_idx in range(4):
            first_four_layers_neurons.add(*neuron_layers[layer_idx])

        for iteration in range(20):
            neurons_list = list(first_four_layers_neurons)
            total_neurons = len(neurons_list)
            num_to_disable = int(total_neurons * 0.3)
            indices_to_disable = np.random.choice(
                total_neurons, 
                size=num_to_disable, 
                replace=False
            )
            neurons_to_disable = [neurons_list[i] for i in indices_to_disable]

            disable_animations = []
            for neuron in neurons_to_disable:
                disable_animations.append(neuron.animate.set_opacity(0.2))
                for line_segment in neuron_to_lines[neuron]:
                    disable_animations.append(line_segment.animate.set_opacity(0.2))

            self.play(*disable_animations, run_time=1.0)
            self.wait(1)

            enable_animations = []
            for neuron in neurons_to_disable:
                enable_animations.append(neuron.animate.set_opacity(1.0))
                for line_segment in neuron_to_lines[neuron]:
                    enable_animations.append(line_segment.animate.set_opacity(1.0))

            self.play(*enable_animations, run_time=1.0)
            self.wait(1)

        # Scene authoring stops here in the interactive editor.
