from manimlib import *
import json

class Node(VGroup):
    def __init__(self, node_data, radius=0.05, font_size=2, **kwargs):
        super().__init__(**kwargs)

        self.node_id = node_data["id"]
        self.phoneme = node_data["phoneme"]
        self.connects_from = node_data["connects_from"]

        if self.phoneme in ["<START>", "<END>"]:
            actual_radius = radius * 1.5
        else:
            actual_radius = radius

        self.circle = Circle(radius=actual_radius, color=WHITE, stroke_color=WHITE)

        self.text = Text(self.phoneme, font_size=font_size, color=WHITE)

        self.text.move_to(self.circle.get_center())

        self.add(self.circle, self.text)


class Connection(VGroup):
    def __init__(self, node_from, node_to, arc_amount=0, **kwargs):
        super().__init__(**kwargs)

        self.node_from = node_from
        self.node_to = node_to

        from_radius = node_from.circle.get_width() / 2
        to_radius = node_to.circle.get_width() / 2
        buff = from_radius + 0.005

        start = node_from.get_center()
        end = node_to.get_center()
        direction = end - start
        length = np.linalg.norm(direction)

        if length > 2 * buff + 0.01:
            unit = direction / length
            start = start + unit * buff
            end = end - unit * buff

            self.arrow = Arrow(
                start,
                end,
                buff=0,
                thickness=0.5,
                fill_color=WHITE,
            )
        else:
            self.arrow = Line(start, end, stroke_width=0.1, color=WHITE)

        self.add(self.arrow)


class TestNode(Scene):
    def construct(self):
        node_data = {
            "id": 0,
            "phoneme": "T",
            "connects_from": []
        }
        node = Node(node_data)
        self.add(node)
        self.wait()
        
        self.embed()


class Network(VGroup):
    def __init__(self, nodes_data, layer_spacing=2.5, node_spacing=1.0, **kwargs):
        super().__init__(**kwargs)

        self.nodes_data = nodes_data
        self.layer_spacing = layer_spacing
        self.node_spacing = node_spacing

        self.nodes = {data["id"]: Node(data) for data in nodes_data}

        self.layers = self._organize_into_layers()

        self._position_nodes()

        self.connections = []
        self._create_connections()

        for connection in self.connections:
            self.add(connection)

        for node in self.nodes.values():
            self.add(node)

    def _organize_into_layers(self):
        layers = []
        visited = set()

        current_layer = [node_id for node_id, node in self.nodes.items()
                        if not node.connects_from]

        while current_layer:
            layers.append(current_layer)
            visited.update(current_layer)

            next_layer = []
            for node_id, node in self.nodes.items():
                if node_id not in visited:
                    if all(parent_id in visited for parent_id in node.connects_from):
                        next_layer.append(node_id)

            current_layer = next_layer

        return layers

    def _position_nodes(self):
        node_positions = {}

        for node_id in self.layers[0]:
            node_positions[node_id] = np.array([0.0, 0.0, 0.0])

        for layer in self.layers[1:]:
            for node_id in layer:
                node_data = self.nodes_data[node_id]
                rel_coords = node_data.get("relative_coordinates", {})

                if rel_coords:
                    parent_id = list(rel_coords.keys())[0]
                    parent_id_int = int(parent_id)

                    if parent_id_int in node_positions:
                        parent_pos = node_positions[parent_id_int]
                        rel_x = rel_coords[parent_id]["rel_x"] / 100.0
                        rel_y = rel_coords[parent_id]["rel_y"] / 100.0

                        node_positions[node_id] = parent_pos + np.array([rel_x, rel_y, 0.0])
                    else:
                        node_positions[node_id] = np.array([0.0, 0.0, 0.0])
                else:
                    node_positions[node_id] = np.array([0.0, 0.0, 0.0])

        for node_id, node in self.nodes.items():
            if node_id in node_positions:
                node.move_to(node_positions[node_id])

    def _create_connections(self):
        for node_id, node in self.nodes.items():
            for parent_id in node.connects_from:
                parent_node = self.nodes[parent_id]
                target_node = node

                connection = Connection(parent_node, target_node, arc_amount=0)
                self.connections.append(connection)


class TestConnection(Scene):
    def construct(self):
        node1_data = {
            "id": 0,
            "phoneme": "T",
            "connects_from": []
        }
        node2_data = {
            "id": 1,
            "phoneme": "E",
            "connects_from": [0]
        }

        node1 = Node(node1_data).shift(LEFT * 2)
        node2 = Node(node2_data).shift(RIGHT * 2)
        connection = Connection(node1, node2)

        self.add(node1, node2, connection)
        self.wait()


class TestNetwork(Scene):
    def construct(self):
        with open("phone_dag_with_connections.json", "r") as f:
            nodes_data = json.load(f)

        network = Network(nodes_data, layer_spacing=2.5, node_spacing=3.0)

        self.add(network)
        self.wait()


class P4(Scene):
    def construct(self):
        with open("phone_dag_with_connections.json", "r") as f:
            nodes_data = json.load(f)

        network = Network(nodes_data, layer_spacing=2.5, node_spacing=3.0)

        # Pre-compute layer info for connections
        node_to_layer = {}
        for l_idx, layer in enumerate(network.layers):
            for node_id in layer:
                node_to_layer[node_id] = l_idx

        # Build layer groups (in order)
        layer_groups = []
        shown_nodes = set()

        for layer_idx, layer in enumerate(network.layers):
            layer_mobjects = []

            for node_id in layer:
                if node_id not in shown_nodes:
                    node = network.nodes[node_id]
                    layer_mobjects.extend([node.circle, node.text])
                    shown_nodes.add(node_id)

            for connection in network.connections:
                from_id = connection.node_from.node_id
                to_id = connection.node_to.node_id
                if node_to_layer.get(from_id) == layer_idx and node_to_layer.get(to_id) == layer_idx + 1:
                    layer_mobjects.append(connection.arrow)

            if layer_mobjects:
                layer_groups.append(VGroup(*layer_mobjects))

        # Pre-compute camera keyframes for each cumulative state
        # (what camera should be when layers 0..i are fully visible)
        padding = 0.5
        aspect = self.camera.frame.get_width() / self.camera.frame.get_height()
        camera_keyframes = []  # List of (width, center_x, center_y)

        cumulative_mobjects = []
        for group in layer_groups:
            cumulative_mobjects.extend(group.submobjects)
            cumulative_group = VGroup(*cumulative_mobjects)
            w = cumulative_group.get_width() + padding
            h = cumulative_group.get_height() + padding
            needed_width = max(w, h * aspect)
            center = cumulative_group.get_center()
            camera_keyframes.append((needed_width, center[0], center[1]))

        # Start with all mobjects invisible
        for group in layer_groups:
            group.set_opacity(0)
            self.add(group)

        # Set initial camera
        init_w, init_cx, init_cy = camera_keyframes[0]
        self.camera.frame.set_width(init_w)
        self.camera.frame.move_to([init_cx, init_cy, 0])

        # Animation settings
        total_duration = 20.0
        lag_ratio = 0.3
        num_layers = len(layer_groups)

        # Bezier ease-in-out
        def bezier_rate(t):
            return t * t * (3 - 2 * t)

        # Progress tracker
        progress = ValueTracker(0)

        def update_scene(_):
            t = progress.get_value()

            # Set layer opacities based on LaggedStart timing
            for i, group in enumerate(layer_groups):
                if num_layers > 1:
                    layer_start = i * lag_ratio / (num_layers - 1 + lag_ratio)
                    layer_end = layer_start + 1.0 / (num_layers - 1 + lag_ratio)
                else:
                    layer_start = 0
                    layer_end = 1

                if t >= layer_end:
                    group.set_opacity(1)
                elif t >= layer_start:
                    layer_progress = (t - layer_start) / (layer_end - layer_start)
                    group.set_opacity(layer_progress)
                else:
                    group.set_opacity(0)

            # Smoothly interpolate camera between keyframes
            # Map t to a continuous "layer index" (can be fractional)
            layer_float = t * (num_layers - 1)
            layer_low = int(layer_float)
            layer_high = min(layer_low + 1, num_layers - 1)
            frac = layer_float - layer_low

            # Interpolate between keyframes
            w0, cx0, cy0 = camera_keyframes[layer_low]
            w1, cx1, cy1 = camera_keyframes[layer_high]

            interp_w = w0 + (w1 - w0) * frac
            interp_cx = cx0 + (cx1 - cx0) * frac
            interp_cy = cy0 + (cy1 - cy0) * frac

            self.camera.frame.set_width(interp_w)
            self.camera.frame.move_to([interp_cx, interp_cy, 0])

        progress.add_updater(update_scene)
        self.add(progress)

        # Animate progress from 0 to 1 with bezier timing
        self.play(
            progress.animate.set_value(1),
            run_time=total_duration,
            rate_func=bezier_rate
        )

        progress.remove_updater(update_scene)
        self.wait(0.5)
        
class AnimateNetwork(Scene):
    def construct(self):
        with open("phone_dag_with_connections.json", "r") as f:
            nodes_data = json.load(f)

        network = Network(nodes_data, layer_spacing=2.5, node_spacing=3.0)

        shown_nodes = set()
        shown_mobjects = []

        for layer_idx, layer in enumerate(network.layers):
            circles_to_show = []
            texts_to_show = []
            for node_id in layer:
                if node_id not in shown_nodes:
                    node = network.nodes[node_id]
                    circles_to_show.append(node.circle)
                    texts_to_show.append(node.text)
                    shown_nodes.add(node_id)
                    shown_mobjects.append(node.circle)
                    shown_mobjects.append(node.text)

            if circles_to_show:
                self.play(*[ShowCreation(circle) for circle in circles_to_show])
                self.play(*[Write(text) for text in texts_to_show])

            arrows_to_show = []
            for connection in network.connections:
                from_node_id = connection.node_from.node_id
                to_node_id = connection.node_to.node_id

                from_layer = None
                to_layer = None
                for l_idx, l in enumerate(network.layers):
                    if from_node_id in l:
                        from_layer = l_idx
                    if to_node_id in l:
                        to_layer = l_idx

                if from_layer == layer_idx and to_layer == layer_idx + 1:
                    arrows_to_show.append(connection.arrow)
                    shown_mobjects.append(connection.arrow)

            if arrows_to_show and shown_mobjects:
                group = VGroup(*shown_mobjects)

                frame_width = group.get_width() + 4
                frame_height = group.get_height() + 4

                current_aspect = self.camera.frame.get_width() / self.camera.frame.get_height()
                needed_width = max(frame_width, frame_height * current_aspect)

                self.play(
                    *[GrowArrow(arrow) for arrow in arrows_to_show],
                    self.camera.frame.animate.set_width(needed_width).move_to(group.get_center())
                )

        self.wait()

class TestArrow(InteractiveScene):
    def construct(self):
        with open("phone_dag_claude_v2.json", "r") as f:
            nodes_data = json.load(f)

        network = Network(nodes_data, layer_spacing=2.5, node_spacing=3.0)
        
        self.add(network)