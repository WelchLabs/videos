from manimlib import *
import json
from collections import defaultdict

CHILL_BROWN = '#948979'
SCALE_FACTOR = 1  # Scale down the coordinates from phone_dag.json


class Node(VGroup):
    """Node for phone_dag.json format: [phoneme, id, x, y]"""
    def __init__(self, phoneme, node_id, x, y, buff=0.2, scale_factor=SCALE_FACTOR, **kwargs):
        super().__init__(**kwargs)

        self.node_id = node_id
        self.phoneme = phoneme
        self.x = x
        self.y = y

        self.text = Text(phoneme, font="American Typewriter", color=CHILL_BROWN)

        self.box = RoundedRectangle(
            width=self.text.get_width() + 2 * buff,
            height=self.text.get_height() + 2 * buff,
            corner_radius=0.15,
            color=CHILL_BROWN,
        )

        self.text.move_to(self.box.get_center())
        self.add(self.box, self.text)

        # Position the node
        self.move_to([x * scale_factor, y * scale_factor, 0])


class Connection(VGroup):
    def __init__(self, node_from, node_to, **kwargs):
        super().__init__(**kwargs)

        self.node_from = node_from
        self.node_to = node_to

        self.arrow = Arrow(
            node_from.get_center(),
            node_to.get_center(),
            buff=0.5,
            thickness=0.5,
            fill_color=CHILL_BROWN,
        )
        self.arrow.set_color(CHILL_BROWN)

        self.add(self.arrow)


class Network(VGroup):
    """Network for phone_dag.json format with nodes and edges"""
    def __init__(self, data, layer_spacing=2.5, node_spacing=3.0, scale_factor=SCALE_FACTOR, **kwargs):
        super().__init__(**kwargs)

        nodes_list = data["nodes"]  # [[phoneme, id, x, y], ...]
        edges_list = data["edges"]  # [[from_id, to_id, bool], ...]

        # Create nodes
        self.nodes = {}
        nodes_list = data["nodes"]
        if isinstance(nodes_list[0], list):
            # old format: [phoneme, id, x, y]
            for phoneme, node_id, x, y in nodes_list:
                node = Node(phoneme, node_id, x, y, scale_factor=scale_factor)
                self.nodes[node_id] = node
        else:
            # new format: {"id":, "phoneme":, "x":, "y":}
            for node in nodes_list:
                phoneme = node["phoneme"]
                node_id = node["id"]
                x = node["x"]
                y = node["y"]
                node_obj = Node(phoneme, node_id, x, y, scale_factor=scale_factor)
                self.nodes[node_id] = node_obj

        # Group nodes by layer (assuming x is layer index)
        layers_dict = defaultdict(list)
        for node_id, node in self.nodes.items():
            layers_dict[node.x].append(node_id)
        self.layers = [layers_dict[i] for i in sorted(layers_dict.keys())]

        # Position nodes in layers
        for layer_idx, layer in enumerate(self.layers):
            y_pos = layer_idx * layer_spacing
            num_nodes = len(layer)
            for i, node_id in enumerate(layer):
                node = self.nodes[node_id]
                x_pos = (i - (num_nodes - 1) / 2) * node_spacing
                node.move_to([x_pos, y_pos, 0])

        # Create connections
        self.connections = []
        edges_list = data["edges"]
        if isinstance(edges_list[0], list):
            # old format: [from_id, to_id, bool]
            for from_id, to_id, _ in edges_list:
                if from_id in self.nodes and to_id in self.nodes:
                    connection = Connection(self.nodes[from_id], self.nodes[to_id])
                    self.connections.append(connection)
        else:
            # new format: {"source":, "target":, ...}
            for edge in edges_list:
                from_id = edge["source"]
                to_id = edge["target"]
                if from_id in self.nodes and to_id in self.nodes:
                    connection = Connection(self.nodes[from_id], self.nodes[to_id])
                    self.connections.append(connection)

        # Add to VGroup (arrows first, then nodes on top)
        for connection in self.connections:
            self.add(connection)
        for node in self.nodes.values():
            self.add(node)


class TestNode(Scene):
    def construct(self):
        node = Node("T", 0, 0, 0)
        self.add(node)
        self.wait()
        self.embed()


class TestConnection(Scene):
    def construct(self):
        node1 = Node("T", 0, -2, 0, scale_factor=1)
        node2 = Node("E", 1, 2, 0, scale_factor=1)
        connection = Connection(node1, node2)

        self.add(node1, node2, connection)
        self.wait()


class TestNetwork(Scene):
    def construct(self):
        with open("phone_dag.json", "r") as f:
            data = json.load(f)

        network = Network(data)
        network.center()

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
                    layer_mobjects.extend([node.box, node.text])
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
        with open("phone_dag_v2.json", "r") as f:
            nodes_data = json.load(f)

        network = Network(nodes_data, layer_spacing=2.5, node_spacing=3.0)

        shown_nodes = set()
        shown_mobjects = []

        for layer_idx, layer in enumerate(network.layers):
            boxes_to_show = []
            texts_to_show = []
            for node_id in layer:
                if node_id not in shown_nodes:
                    node = network.nodes[node_id]
                    boxes_to_show.append(node.box)
                    texts_to_show.append(node.text)
                    shown_nodes.add(node_id)
                    shown_mobjects.append(node.box)
                    shown_mobjects.append(node.text)

            if boxes_to_show:
                self.play(*[ShowCreation(box) for box in boxes_to_show])
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


class RenderNetworkV2(Scene):
    def construct(self):
        with open("phone_dag_v2.json", "r") as f:
            data = json.load(f)

        # Use raw coordinates from JSON (x = horizontal, y = vertical for branching)
        scale = 0.01
        nodes = {}
        node_group = VGroup()

        for node_data in data["nodes"]:
            node_id = node_data["id"]
            phoneme = node_data["phoneme"]
            x = node_data["x"] * scale
            y = node_data["y"] * scale

            text = Text(phoneme, font="American Typewriter", color=CHILL_BROWN)
            box = RoundedRectangle(
                width=text.get_width() + 0.4,
                height=text.get_height() + 0.4,
                corner_radius=0.15,
                color=CHILL_BROWN,
            )
            text.move_to(box.get_center())
            node = VGroup(box, text)
            node.move_to([x, y, 0])
            nodes[node_id] = node
            node_group.add(node)

        # Create edges
        edge_group = VGroup()
        for edge_data in data["edges"]:
            source_id = edge_data["source"]
            target_id = edge_data["target"]
            if source_id in nodes and target_id in nodes:
                arrow = Arrow(
                    nodes[source_id].get_center(),
                    nodes[target_id].get_center(),
                    buff=0.5,
                    thickness=0.5,
                    fill_color=CHILL_BROWN,
                )
                arrow.set_color(CHILL_BROWN)
                edge_group.add(arrow)

        network = VGroup(edge_group, node_group)
        network.center()

        # Fit camera to network
        padding = 1.0
        aspect = self.camera.frame.get_width() / self.camera.frame.get_height()
        w = network.get_width() + padding
        h = network.get_height() + padding
        needed_width = max(w, h * aspect)

        self.camera.frame.set_width(needed_width)
        self.camera.frame.move_to(network.get_center())

        self.add(network)
        self.wait()