import manimgx as gx
from _2026.the_bitter_lesson.network_edges import Arrow, NetworkLayer
import numpy as np
import random
import matplotlib.pyplot as plt

# CHILL_BROWN='#cabba6'
CHILL_BROWN='#655d53' #Going darker

class Node(gx.VMobject):
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
        circle = gx.Circle(
            radius=self.node_radius,
            stroke_color=self.node_stroke_color,
            stroke_width=self.node_stroke_width,
            fill_color=gx.WHITE,
            fill_opacity=1.0,
        )
        circle.move_to(gx.ORIGIN)

        text = gx.Text(self.label, font_size=self.font_size, color=self.node_stroke_color)
        text.set_color(gx.BLACK)
        text.move_to(circle.get_center())

        self.add(circle, text)

    def get_connection_point(self, target_point):
        center = self.get_center()
        direction = gx.normalize(target_point - center)
        return center + direction * self.node_radius   

class Layer(NetworkLayer):
    node_type = Node
    
class Graph(gx.VMobject):
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

        self.layers = gx.VGroup()
        self.arrows = gx.VGroup()

        self.build()

    def build(self):
        self.clear()
        self.layers = gx.VGroup()
        self.arrows = gx.VGroup()

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
            layer.move_to(gx.RIGHT * i * self.layer_spacing)
            self.layers.add(layer)

        for i, matrix in enumerate(self.weight_matrices):
            layer_from = self.layers[i]
            layer_to = self.layers[i + 1]

            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    node_start = layer_from.get_node(r)
                    node_end = layer_to.get_node(c)
                    if node_start is None or node_end is None:
                        continue

                    weight = matrix[r, c]
                    arrow = Arrow(
                        node_start,
                        node_end,
                        stroke_width=self.arrow_stroke_width,
                        value=np.clip(abs(weight), 0, 1),
                        weight=weight,
                        stroke_color=CHILL_BROWN
                    )
                    self.arrows.add(arrow)

        self.add(self.arrows)
        self.add(self.layers)
        self.move_to(gx.ORIGIN)

    def update_weights(self, new_matrices):
        self.weight_matrices = new_matrices
        self.build()

    def get_layer(self, index):
        if 0 <= index < len(self.layers):
            return self.layers[index]
        return None


class P10(gx.Scene):
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
            node_stroke_color=gx.BLACK,
            node_stroke_width=3.0,
            arrow_stroke_width=0,
        )
        self.add(graph)


        self.remove(graph[1]); self.add(graph[1])

        # self.camera.frame.
        
        
class P10v2(gx.Scene):
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
            node_stroke_color=gx.BLACK,
            node_stroke_width=3.0,
            arrow_stroke_width=0,
        )

        # Zoom out to see the entire graph
        # self.camera.frame.scale()

        first_layer = graph.layers[0]

        node_anims = []
        for node in first_layer.nodes:
            node_anims.append(gx.Create(node[0]))
            node_anims.append(gx.Write(node[1]))

        ellipsis_anims = [gx.FadeIn(dot) for dot in first_layer.ellipsis_dots]

        self.play(
            gx.LaggedStart(*node_anims, *ellipsis_anims, lag_ratio=0.1),
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

            arrow_anims = [gx.GrowArrow(arrow.line) for arrow in arrows_to_show]

            next_node_anims = []
            for node in next_layer.nodes:
                next_node_anims.append(gx.Create(node[0]))
                next_node_anims.append(gx.Write(node[1]))

            next_ellipsis_anims = [gx.FadeIn(dot) for dot in next_layer.ellipsis_dots]

            self.play(
                gx.LaggedStart(*arrow_anims, lag_ratio=0.05),
                gx.LaggedStart(*next_node_anims, *next_ellipsis_anims, lag_ratio=0.1),
                run_time=1.5
            )



class P10v3(gx.Scene):
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
            node_stroke_color=gx.BLACK,
            node_stroke_width=3.0,
            arrow_stroke_width=0,
        )

        # Zoom out to see the entire graph
        self.camera.frame.scale(0.8)

        first_layer = graph.layers[0]

        node_anims = []
        for node in first_layer.nodes:
            node_anims.append(gx.Create(node[0]))
            node_anims.append(gx.Write(node[1]))

        ellipsis_anims = [gx.FadeIn(dot) for dot in first_layer.ellipsis_dots]

        self.wait()
        self.play(
            gx.LaggedStart(*node_anims, *ellipsis_anims, lag_ratio=0.1),
            run_time=3.0
        )
        self.wait(0.5)


        # Collect all animations with their x-positions for sorting
        all_anims_with_pos = []

        # Add arrows
        for arrow in graph.arrows:
            # Use the midpoint or start x-position of the arrow
            x_pos = arrow.line.get_center()[0]
            all_anims_with_pos.append((x_pos, gx.GrowArrow(arrow.line)))

        # Add nodes from all layers except the first (assuming it's already shown)
        for layer_idx, layer in enumerate(graph.layers[1:], start=1):
            for node in layer.nodes:
                x_pos = node.get_center()[0]
                all_anims_with_pos.append((x_pos, gx.Create(node[0])))
                all_anims_with_pos.append((x_pos, gx.Write(node[1])))
            for dot in layer.ellipsis_dots:
                x_pos = dot.get_center()[0]
                all_anims_with_pos.append((x_pos, gx.FadeIn(dot)))

        # Sort by x-position
        all_anims_with_pos.sort(key=lambda x: x[0])

        # Extract just the animations in sorted order
        sorted_anims = [anim for _, anim in all_anims_with_pos]

        # Play everything in one smooth left-to-right sweep
        self.wait()
        self.play(
            gx.LaggedStart(*sorted_anims, lag_ratio=0.02),
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


class P10v4(gx.Scene):
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
            node_stroke_color=gx.BLACK,
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
                all_anims_with_pos.append((x_pos, gx.Create(node[0])))
                all_anims_with_pos.append((x_pos, gx.Write(node[1])))
            for dot in layer.ellipsis_dots:
                x_pos = dot.get_center()[0]
                all_anims_with_pos.append((x_pos, gx.FadeIn(dot)))

        # Add arrows
        for arrow in graph.arrows:
            x_pos = arrow.line.get_center()[0]
            all_anims_with_pos.append((x_pos, gx.GrowArrow(arrow.line)))

        # Sort by x-position
        all_anims_with_pos.sort(key=lambda x: x[0])

        # Extract just the animations in sorted order
        sorted_anims = [anim for _, anim in all_anims_with_pos]

        # Play everything in one smooth left-to-right sweep
        self.wait()
        self.play(
            gx.LaggedStart(*sorted_anims, lag_ratio=0.02),
            run_time=20
        )


        self.wait(20)
