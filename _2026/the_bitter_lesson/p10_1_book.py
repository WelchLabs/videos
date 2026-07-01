from manimlib import *
from manimlib import Arrow as ManimArrow
import numpy as np
import random
import matplotlib.pyplot as plt

# CHILL_BROWN='#cabba6'
CHILL_BROWN='#655d53' #Going darker

class Node(VMobject):
    def __init__(
        self,
        node_radius=0.1,
        node_stroke_color=CHILL_BROWN,
        node_stroke_width=3,
        value=0.5,
        label="A",
        font_size=24,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.node_radius = node_radius
        self.node_stroke_color = node_stroke_color
        self.node_stroke_width = node_stroke_width
        self.value = value
        self.label = label
        self.font_size = font_size

        self.build()


    def build(self):
        self.clear()
        circle = Circle(
            radius=self.node_radius,
            stroke_color=self.node_stroke_color,
            stroke_width=self.node_stroke_width,
            fill_opacity=0.0,
        )
        circle.move_to(ORIGIN)
        circle.set_z_index(1)

        text = Text(self.label, font="American Typewriter", font_size=self.font_size, color=self.node_stroke_color)
        text.set_color(self.node_stroke_color)
        text.move_to(circle.get_center())
        text.set_z_index(2)

        self.add(circle, text)

    def get_connection_point(self, target_point):
        center = self.get_center()
        direction = normalize(target_point - center)
        return center + direction * self.node_radius


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


class Arrow(VMobject):
    def __init__(
        self,
        start_obj,
        end_obj,
        stroke_width=2,
        value=0.5,
        weight=1.0,
        stroke_color=CHILL_BROWN,
        dashed=False,
        buff=0.0,
        obstacles=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.start_obj = start_obj
        self.end_obj = end_obj
        self.stroke_width = abs(stroke_width * weight)
        self.weight = weight
        self.stroke_color = stroke_color
        self.dashed = dashed
        self.buff = buff
        self.value = value
        self.obstacles = obstacles if obstacles is not None else []

        # Opacity based on value/strength
        self.stroke_opacity = np.clip(value, 0.1, 1.0)

        self.build()

    def build(self):
        start_point = self._get_point_from_object(self.start_obj, True)
        end_point = self._get_point_from_object(self.end_obj, False)

        # Cut the line into pieces so it never draws through an
        # intervening node's circle, instead of relying on draw order.
        d = end_point - start_point
        segments = _visible_segments(start_point, end_point, self.obstacles)

        self.segments = VGroup()
        for i, (t0, t1) in enumerate(segments):
            seg_start = start_point + t0 * d
            seg_end = start_point + t1 * d
            is_last = (i == len(segments) - 1)

            # manimlib's Arrow renders its body as a filled shape driven by
            # `thickness`, not `stroke_width` (which we keep at 0) - so the
            # gap segments must use the same filled-Arrow body (just with no
            # tip) rather than a plain Line, or they render invisibly.
            seg = ManimArrow(
                start=seg_start,
                end=seg_end,
                stroke_color=self.stroke_color,
                stroke_width=self.stroke_width,
                stroke_opacity=self.stroke_opacity,
                fill_color=self.stroke_color,
                fill_opacity=self.stroke_opacity,
                thickness=1.5,
                tip_width_ratio=(5 if is_last else 1e-6),
                # max_tip_length_to_length_ratio=0.1,
                buff=0
            )
            if is_last:
                self.line = seg

            if self.dashed:
                seg.set_stroke(
                    color=self.stroke_color,
                    width=self.stroke_width,
                    opacity=self.stroke_opacity,
                    dash_length=0.1,
                    dash_spacing=0.1,
                )

            self.segments.add(seg)

        self.add(self.segments)

    def _get_point_from_object(self, obj, is_start):
        if hasattr(obj, "get_center"):
            base_point = obj.get_center()
        elif hasattr(obj, "get_position"):
            base_point = obj.get_position()
        elif isinstance(obj, np.ndarray) and obj.shape == (3,):
            return obj
        else:
            try:
                base_point = obj.get_center()
            except:
                raise ValueError(f"Could not get position from object: {type(obj)}")

        other_obj = self.end_obj if is_start else self.start_obj
        other_center = None

        if hasattr(other_obj, "get_center"):
            other_center = other_obj.get_center()
        elif hasattr(other_obj, "get_position"):
            other_center = other_obj.get_position()
        elif isinstance(other_obj, np.ndarray) and other_obj.shape == (3,):
            other_center = other_obj
        else:
            try:
                other_center = other_obj.get_center()
            except:
                other_center = base_point

        # Prefer get_connection_point method for precise edge calculation
        if hasattr(obj, "get_connection_point"):
            return obj.get_connection_point(other_center)

        # Fallback to manual radius calculation
        direction = other_center - base_point
        unit = normalize(direction)

        if hasattr(obj, "node_radius"):
            radius = obj.node_radius
            return base_point + unit * radius

        return base_point

    def update_positions(self):
        # Node positions may have changed, so the occlusion gaps need
        # recomputing too - just rebuild the segments from scratch.
        self.build()

    def set_value(self, new_value):
        self.value = new_value
        self.stroke_opacity = np.clip(new_value, 0.1, 1.0)
        for seg in self.segments:
            seg.set_stroke(
                color=self.stroke_color,
                width=self.stroke_width,
                opacity=self.stroke_opacity
            )

    def pulse(self, duration=1.0, scale=1.5):
        original_width = self.line.get_stroke_width()
        return Succession(
            ApplyMethod(self.line.set_stroke_width, original_width * scale, run_time=duration/2),
            ApplyMethod(self.line.set_stroke_width, original_width, run_time=duration/2)
        )

    def animate_flow(self, duration=1.0):
        if self.dashed:
            return MoveAlongPath(
                Dot(color=self.stroke_color),
                self.line.copy(),
                run_time=duration
            )
        else:
            line_copy = self.line.copy()
            line_copy.set_stroke(width=self.stroke_width * 1.5)
            return ShowPassingFlash(
                line_copy,
                time_width=0.5,
                run_time=duration
            )
    
class Layer(VMobject):
    def __init__(
        self,
        values=None,
        labels=None,
        max_display=16,
        node_radius=0.1,
        node_spacing=0.3,
        node_stroke_color=CHILL_BROWN,
        node_stroke_width=6,
        colormap=plt.get_cmap("viridis"),
        position=ORIGIN,
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

        self.nodes = VGroup()
        self.ellipsis_dots = VGroup()

        self.build()
        self.move_to(position)

    def build(self):
        self.clear()
        self.nodes = VGroup()
        self.ellipsis_dots = VGroup()

        total_nodes = len(self.values)
        if total_nodes <= self.max_display:
            self._create_simple_layer()
        else:
            self._create_truncated_layer()
            
        self.ellipsis_dots.set_color(CHILL_BROWN)

        self.add(self.nodes, self.ellipsis_dots)

    def _create_simple_layer(self):
        total_nodes = len(self.values)
        total_height = self.node_spacing * (total_nodes - 1)

        for i, value in enumerate(self.values):
            y = total_height/2 - i * self.node_spacing
            node = Node(
                node_radius=self.node_radius,
                node_stroke_color=self.node_stroke_color,
                node_stroke_width=self.node_stroke_width,
                value=value,
                label=self.labels[i]
            ).move_to(UP * y)
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
            node = Node(
                node_radius=self.node_radius,
                node_stroke_color=self.node_stroke_color,
                node_stroke_width=self.node_stroke_width,
                value=self.values[i],
                label=self.labels[i]
            ).move_to(UP * y)
            self.nodes.add(node)

        last_top_node_y = start_y - (visible_per_side - 1) * self.node_spacing
        first_ellipsis_y = last_top_node_y - gap_before

        # Create ellipsis dots
        for j in range(3):
            y_offset = first_ellipsis_y - j * self.ellipsis_spacing
            dot = Dot(radius=self.ellipsis_size, color=self.node_stroke_color).move_to(UP * y_offset)
            self.ellipsis_dots.add(dot)

        last_ellipsis_y = first_ellipsis_y - 2 * self.ellipsis_spacing
        first_bottom_node_y = last_ellipsis_y - gap_after

        # Create bottom nodes
        for i in range(visible_per_side):
            idx = len(self.values) - visible_per_side + i
            y = first_bottom_node_y - i * self.node_spacing
            node = Node(
                node_radius=self.node_radius,
                node_stroke_color=self.node_stroke_color,
                node_stroke_width=self.node_stroke_width,
                value=self.values[idx],
                label=self.labels[idx]
            ).move_to(UP * y)
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

    def highlight_node(self, index, color=YELLOW):
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
    
class Graph(VMobject):
    def __init__(
        self,
        weight_matrices,
        layer_labels=None,
        layer_spacing=2.5,
        max_display=10,
        node_radius=0.1,
        node_spacing=0.3,
        node_stroke_color=CHILL_BROWN,
        node_stroke_width=6,
        arrow_stroke_width=2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.weight_matrices = weight_matrices
        self.layer_labels = layer_labels
        self.layer_spacing = layer_spacing
        self.max_display = max_display
        self.node_radius = node_radius
        self.node_spacing = node_spacing
        self.node_stroke_color = node_stroke_color
        self.node_stroke_width = node_stroke_width
        self.arrow_stroke_width = arrow_stroke_width

        self.layers = VGroup()
        self.arrows = VGroup()

        self.build()

    def build(self):
        self.clear()
        self.layers = VGroup()
        self.arrows = VGroup()

        layer_sizes = [self.weight_matrices[0].shape[0]]  # input layer
        for matrix in self.weight_matrices:
            layer_sizes.append(matrix.shape[1])

        for i, size in enumerate(layer_sizes):
            values = [0.5] * size
            labels = self.layer_labels if self.layer_labels is not None else ["A"] * size
            layer = Layer(
                values=values,
                labels=labels,
                max_display=self.max_display,
                node_radius=self.node_radius,
                node_spacing=self.node_spacing,
                node_stroke_color=self.node_stroke_color,
                node_stroke_width=self.node_stroke_width,
                colormap=plt.get_cmap("viridis")
            )
            layer.move_to(RIGHT * i * self.layer_spacing)
            self.layers.add(layer)

        for i, matrix in enumerate(self.weight_matrices):
            layer_from = self.layers[i]
            layer_to = self.layers[i + 1]

            # An edge can only pass close to other nodes that live in
            # its own two columns, since it never leaves that x-range.
            column_nodes = list(layer_from.get_all_nodes()) + list(layer_to.get_all_nodes())

            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    node_start = layer_from.get_node(r)
                    node_end = layer_to.get_node(c)
                    if node_start is None or node_end is None:
                        continue

                    obstacles = [n for n in column_nodes if n is not node_start and n is not node_end]

                    weight = matrix[r, c]
                    arrow = Arrow(
                        node_start,
                        node_end,
                        stroke_width=self.arrow_stroke_width,
                        value=np.clip(abs(weight), 0, 1),
                        weight=weight,
                        stroke_color=CHILL_BROWN,
                        obstacles=obstacles,
                    )
                    self.arrows.add(arrow)

        self.add(self.arrows)
        self.add(self.layers)
        self.move_to(ORIGIN)

    def update_weights(self, new_matrices):
        self.weight_matrices = new_matrices
        self.build()

    def get_layer(self, index):
        if 0 <= index < len(self.layers):
            return self.layers[index]
        return None


class P10(Scene):
    def construct(self):
        # 6 layers, each with 7 nodes (showing only 6)
        # Phoneme labels for each layer (same across all layers)
        phoneme_labels = ["T", "AH", "EL", "", "G", "I", "V"]  # Index 3 is hidden

        # Create 5 weight matrices to connect the 6 layers
        W1 = np.random.uniform(0.3, 1.0, (7, 7))  # layer 1 -> layer 2
        W2 = np.random.uniform(0.3, 1.0, (7, 7))  # layer 2 -> layer 3
        W3 = np.random.uniform(0.3, 1.0, (7, 7))  # layer 3 -> layer 4
        W4 = np.random.uniform(0.3, 1.0, (7, 7))  # layer 4 -> layer 5
        W5 = np.random.uniform(0.3, 1.0, (7, 7))  # layer 5 -> layer 6
        W6 = np.random.uniform(0.3, 1.0, (7, 7))
        W7 = np.random.uniform(0.3, 1.0, (7, 7))
        W8 = np.random.uniform(0.3, 1.0, (7, 7))


        graph = Graph(
            weight_matrices=[W1, W2, W3, W4, W5, W6, W7, W8],
            layer_labels=phoneme_labels,
            layer_spacing=1.2,
            max_display=6,
            node_radius=0.25,
            node_spacing=0.9,
            node_stroke_color=CHILL_BROWN,
            node_stroke_width=3.0,
            arrow_stroke_width=0,
        )
        self.add(graph)

        # self.frame.

        self.embed()
        
class P10v2(Scene):
    def construct(self):
        phoneme_labels = ["T", "AH", "EL", "", "G", "I", "V"]   # Index 3 is hidden

        W1 = np.random.uniform(0.3, 1.0, (7, 7))  # layer 1 -> layer 2
        W2 = np.random.uniform(0.3, 1.0, (7, 7))  # layer 2 -> layer 3
        W3 = np.random.uniform(0.3, 1.0, (7, 7))  # layer 3 -> layer 4
        W4 = np.random.uniform(0.3, 1.0, (7, 7))  # layer 4 -> layer 5
        W5 = np.random.uniform(0.3, 1.0, (7, 7))  # layer 5 -> layer 6
        W6 = np.random.uniform(0.3, 1.0, (7, 7))
        W7 = np.random.uniform(0.3, 1.0, (7, 7))
        W8 = np.random.uniform(0.3, 1.0, (7, 7))


        graph = Graph(
            weight_matrices=[W1, W2, W3, W4, W5, W6, W7, W8],
            layer_labels=phoneme_labels,
            layer_spacing=1.2,
            max_display=6,
            node_radius=0.25,
            node_spacing=0.9,
            node_stroke_color=CHILL_BROWN,
            node_stroke_width=3.0,
            arrow_stroke_width=0,
        )

        # Zoom out to see the entire graph
        # self.camera.frame.scale()

        first_layer = graph.layers[0]

        node_anims = []
        for node in first_layer.nodes:
            node_anims.append(ShowCreation(node[0]))
            node_anims.append(Write(node[1]))

        ellipsis_anims = [FadeIn(dot) for dot in first_layer.ellipsis_dots]

        self.play(
            LaggedStart(*node_anims, *ellipsis_anims, lag_ratio=0.1),
            run_time=1.5
        )
        self.wait(0.5)

        for layer_idx in range(len(graph.layers) - 1):
            arrows_to_show = []
            for arrow in graph.arrows:
                from_layer_idx = None
                to_layer_idx = None

                for i, layer in enumerate(graph.layers):
                    if arrow.start_obj in layer.nodes:
                        from_layer_idx = i
                    if arrow.end_obj in layer.nodes:
                        to_layer_idx = i

                if from_layer_idx == layer_idx and to_layer_idx == layer_idx + 1:
                    arrows_to_show.append(arrow)

            next_layer = graph.layers[layer_idx + 1]

            arrow_anims = [ShowCreation(arrow) for arrow in arrows_to_show]

            next_node_anims = []
            for node in next_layer.nodes:
                next_node_anims.append(ShowCreation(node[0]))
                next_node_anims.append(Write(node[1]))

            next_ellipsis_anims = [FadeIn(dot) for dot in next_layer.ellipsis_dots]

            self.play(
                LaggedStart(*arrow_anims, lag_ratio=0.05),
                LaggedStart(*next_node_anims, *next_ellipsis_anims, lag_ratio=0.1),
                run_time=1.5
            )

        self.embed()


class P10v3(Scene):
    def construct(self):
        phoneme_labels = ["T", "AH", "EL", "", "G", "I", "V"]   # Index 3 is hidden

        W1 = np.random.uniform(0.3, 1.0, (7, 7))  # layer 1 -> layer 2
        W2 = np.random.uniform(0.3, 1.0, (7, 7))  # layer 2 -> layer 3
        W3 = np.random.uniform(0.3, 1.0, (7, 7))  # layer 3 -> layer 4
        W4 = np.random.uniform(0.3, 1.0, (7, 7))  # layer 4 -> layer 5
        W5 = np.random.uniform(0.3, 1.0, (7, 7))  # layer 5 -> layer 6
        W6 = np.random.uniform(0.3, 1.0, (7, 7))
        W7 = np.random.uniform(0.3, 1.0, (7, 7))
        W8 = np.random.uniform(0.3, 1.0, (7, 7))


        graph = Graph(
            weight_matrices=[W1, W2, W3, W4, W5, W6, W7, W8],
            layer_labels=phoneme_labels,
            layer_spacing=1.2,
            max_display=6,
            node_radius=0.25,
            node_spacing=0.9,
            node_stroke_color=CHILL_BROWN,
            node_stroke_width=3.0,
            arrow_stroke_width=0,
        )

        # Zoom out to see the entire graph
        self.camera.frame.scale(0.8)

        first_layer = graph.layers[0]

        node_anims = []
        for node in first_layer.nodes:
            node_anims.append(ShowCreation(node[0]))
            node_anims.append(Write(node[1]))

        ellipsis_anims = [FadeIn(dot) for dot in first_layer.ellipsis_dots]

        self.wait()
        self.play(
            LaggedStart(*node_anims, *ellipsis_anims, lag_ratio=0.1),
            run_time=3.0
        )
        self.wait(0.5)


        # Collect all animations with their x-positions for sorting
        all_anims_with_pos = []

        # Add arrows
        for arrow in graph.arrows:
            # Use the midpoint or start x-position of the arrow
            x_pos = arrow.line.get_center()[0]
            all_anims_with_pos.append((x_pos, ShowCreation(arrow)))

        # Add nodes from all layers except the first (assuming it's already shown)
        for layer_idx, layer in enumerate(graph.layers[1:], start=1):
            for node in layer.nodes:
                x_pos = node.get_center()[0]
                all_anims_with_pos.append((x_pos, ShowCreation(node[0])))
                all_anims_with_pos.append((x_pos, Write(node[1])))
            for dot in layer.ellipsis_dots:
                x_pos = dot.get_center()[0]
                all_anims_with_pos.append((x_pos, FadeIn(dot)))

        # Sort by x-position
        all_anims_with_pos.sort(key=lambda x: x[0])

        # Extract just the animations in sorted order
        sorted_anims = [anim for _, anim in all_anims_with_pos]

        # Play everything in one smooth left-to-right sweep
        self.wait()
        self.play(
            LaggedStart(*sorted_anims, lag_ratio=0.02),
            run_time=20
        )

        # for layer_idx in range(len(graph.layers) - 1):
        #     arrows_to_show = []
        #     for arrow in graph.arrows:
        #         from_layer_idx = None
        #         to_layer_idx = None

        #         for i, layer in enumerate(graph.layers):
        #             if arrow.start_obj in layer.nodes:
        #                 from_layer_idx = i
        #             if arrow.end_obj in layer.nodes:
        #                 to_layer_idx = i

        #         if from_layer_idx == layer_idx and to_layer_idx == layer_idx + 1:
        #             arrows_to_show.append(arrow)

        #     next_layer = graph.layers[layer_idx + 1]

        #     arrow_anims = [GrowArrow(arrow.line) for arrow in arrows_to_show]

        #     next_node_anims = []
        #     for node in next_layer.nodes:
        #         next_node_anims.append(ShowCreation(node[0]))
        #         next_node_anims.append(Write(node[1]))

        #     next_ellipsis_anims = [FadeIn(dot) for dot in next_layer.ellipsis_dots]

        #     self.play(
        #         LaggedStart(*arrow_anims, lag_ratio=0.05),
        #         LaggedStart(*next_node_anims, *next_ellipsis_anims, lag_ratio=0.1),
        #         run_time=1.5
        #     )

        self.wait(20)
        self.embed()


class P10v4(Scene):
    def construct(self):
        phoneme_labels = ["T", "AH", "EL", "", "G", "I", "V"]
        W1 = np.random.uniform(0.3, 1.0, (7, 7))
        W2 = np.random.uniform(0.3, 1.0, (7, 7))
        W3 = np.random.uniform(0.3, 1.0, (7, 7))
        W4 = np.random.uniform(0.3, 1.0, (7, 7))
        W5 = np.random.uniform(0.3, 1.0, (7, 7))
        W6 = np.random.uniform(0.3, 1.0, (7, 7))
        W7 = np.random.uniform(0.3, 1.0, (7, 7))
        W8 = np.random.uniform(0.3, 1.0, (7, 7))
        graph = Graph(
            weight_matrices=[W1, W2, W3, W4, W5, W6, W7, W8],
            layer_labels=phoneme_labels,
            layer_spacing=1.2,
            max_display=6,
            node_radius=0.25,
            node_spacing=0.9,
            node_stroke_color=CHILL_BROWN,
            node_stroke_width=3.0,
            arrow_stroke_width=0,
        )
        self.camera.frame.scale(0.8)

        # Collect all animations with their x-positions for sorting
        all_anims_with_pos = []

        # Add ALL layers' nodes (including first layer)
        for layer in graph.layers:
            for node in layer.nodes:
                x_pos = node.get_center()[0]
                all_anims_with_pos.append((x_pos, ShowCreation(node[0])))
                all_anims_with_pos.append((x_pos, Write(node[1])))
            for dot in layer.ellipsis_dots:
                x_pos = dot.get_center()[0]
                all_anims_with_pos.append((x_pos, FadeIn(dot)))

        # Add arrows
        for arrow in graph.arrows:
            x_pos = arrow.line.get_center()[0]
            all_anims_with_pos.append((x_pos, ShowCreation(arrow)))

        # Sort by x-position
        all_anims_with_pos.sort(key=lambda x: x[0])

        # Extract just the animations in sorted order
        sorted_anims = [anim for _, anim in all_anims_with_pos]

        # Play everything in one smooth left-to-right sweep
        self.wait()
        self.play(
            LaggedStart(*sorted_anims, lag_ratio=0.02),
            run_time=20
        )


        self.wait(20)
        self.embed()