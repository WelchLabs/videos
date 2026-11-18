import manimgx as m
import numpy as np
import random
import matplotlib.pyplot as plt

CHILL_BROWN='#cabba6'

class Neuron(m.VMobject):
    def __init__(
        self,
        node_radius=0.1,
        node_stroke_color=CHILL_BROWN,
        node_stroke_width=3,
        value=0.5,
        colormap=plt.get_cmap("viridis"),
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.node_radius = node_radius
        self.node_stroke_color = node_stroke_color
        self.node_stroke_width = node_stroke_width
        self.value = value
        self.colormap = colormap
        self.style = self.map_value_to_style(value)
        self.node_fill_color = self.style["color"]
        self.node_fill_opacity = self.style["opacity"]
        
        self.build()

        
    def build(self):
        circle = m.Circle(
            radius=self.node_radius,
            stroke_color=self.node_stroke_color,
            stroke_width=self.node_stroke_width,
            fill_color=self.node_fill_color,
            fill_opacity=self.node_fill_opacity,
        )
        circle.move_to(m.ORIGIN)
        self.add(circle)
        
    def map_value_to_style(self, value):
        value_clipped = np.clip(value, 0, 1)
        rgba = self.colormap(value_clipped)
        color = m.rgb_to_color(rgba[:3])
        opacity = value_clipped 
        return {"color": color, "opacity": opacity}
    
    def get_connection_point(self, target_point):
        center = self.get_center()
        direction = m.normalize(target_point - center)
        return center + direction * self.node_radius   

class AttentionPattern(m.VMobject):
    def __init__(
        self,
        matrix,
        square_size=0.3,
        min_opacity=0.2,
        max_opacity=1.0,
        stroke_width=1.0,
        stroke_color=CHILL_BROWN,
        colormap=plt.get_cmap("viridis"),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.matrix = np.array(matrix)
        self.n_rows, self.n_cols = self.matrix.shape
        self.square_size = square_size
        self.min_opacity = min_opacity
        self.max_opacity = max_opacity
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self._colormap = colormap

        self.build()

    def map_value_to_style(self, val):
        val_clipped = np.clip(val, 0, 1)
        rgba = self._colormap(val_clipped)
        color = m.rgb_to_color(rgba[:3])
        opacity = self.min_opacity + val_clipped * (self.max_opacity - self.min_opacity)
        return {"color": color, "opacity": opacity}

    def build(self):
        self.clear()
        squares = m.VGroup()
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                val = self.matrix[i, j]
                style = self.map_value_to_style(val)

                square = m.Square(side_length=self.square_size)
                square.set_fill(style["color"], opacity=style["opacity"])
                square.set_stroke(self.stroke_color, width=self.stroke_width)

                pos = m.RIGHT * j * self.square_size + m.DOWN * i * self.square_size
                square.move_to(pos)
                squares.add(square)

        squares.move_to(m.ORIGIN)
        self.add(squares)
   
class Connection(m.VMobject):
    def __init__(
        self,
        start_obj,
        end_obj,
        stroke_width=2,
        value=0.5,
        stroke_opacity=0.8,
        weight=1.0,
        colormap=plt.get_cmap("viridis"),
        arrow=False,
        dashed=False,
        buff=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.start_obj = start_obj
        self.end_obj = end_obj
        self.stroke_width = abs(stroke_width * weight)
        self.stroke_opacity = stroke_opacity
        self.weight = weight
        self.arrow = arrow
        self.dashed = dashed
        self.buff = buff
        self.value = value
        self.colormap = colormap
        
        self.style = self.map_value_to_style(value)
        self.stroke_color = self.style["color"]
        
        self.build()
    
    def build(self):
        start_point = self._get_point_from_object(self.start_obj, True)
        end_point = self._get_point_from_object(self.end_obj, False)
        
        if self.dashed:
            line = m.DashedLine(
                start_point,
                end_point,
                dash_length=0.1,
                dashed_ratio=0.5,
                stroke_color=self.stroke_color,
                stroke_width=self.stroke_width,
                stroke_opacity=self.stroke_opacity,
            )
            if self.arrow:
                line.add_tip(tip_length=min(line.tip_length, line.get_length() / 4))
        elif self.arrow:
            line = m.Arrow(
                start=start_point,
                end=end_point,
                stroke_color=self.stroke_color,
                stroke_width=self.stroke_width,
                stroke_opacity=self.stroke_opacity,
            )
        else:
            line = m.Line(
                start_point,
                end_point,
                stroke_color=self.stroke_color,
                stroke_width=self.stroke_width,
                stroke_opacity=self.stroke_opacity,
            )
            
        self.add(line)
        self.line = line
        
    
    def map_value_to_style(self, value):
        value_clipped = np.clip(value, 0, 1)
        rgba = self.colormap(value_clipped)
        color = m.rgb_to_color(rgba[:3])
        opacity = value_clipped 
        return {"color": color, "opacity": opacity}
    
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
        
        direction = other_center - base_point
        unit = m.normalize(direction)
        
        if hasattr(obj, "node_radius"):
            radius = obj.node_radius
            return base_point + unit * radius
        elif hasattr(obj, "get_connection_point"):
            return obj.get_connection_point(other_center)
        
        return base_point
        
    def update_positions(self):
        start_point = self._get_point_from_object(self.start_obj, True)
        end_point = self._get_point_from_object(self.end_obj, False)
        
        if hasattr(self.line, "put_start_and_end_on"):
            self.line.put_start_and_end_on(start_point, end_point)
        else:
            self.line.set_points_by_ends(start_point, end_point)
            
    def set_value(self, new_value):
        self.value = new_value
        self.style = self.map_value_to_style(new_value)
        self.stroke_color = self.style["color"]
        self.line.set_stroke(
            color=self.stroke_color,
            width=self.stroke_width,
            opacity=self.stroke_opacity
        )
            
    def pulse(self, duration=1.0, scale=1.5):
        original_width = self.line.get_stroke_width()
        return m.Succession(
            self.line.animate(run_time=duration / 2).set_stroke(width=original_width * scale),
            self.line.animate(run_time=duration / 2).set_stroke(width=original_width)
        )
        
    def animate_flow(self, duration=1.0):
        if self.dashed:
            return m.MoveAlongPath(
                m.Dot(color=self.stroke_color),
                self.line.copy(),
                run_time=duration
            )
        else:
            line_copy = self.line.copy()
            line_copy.set_stroke(width=self.stroke_width * 1.5)
            return m.ShowPassingFlash(
                line_copy,
                time_width=0.5,
                run_time=duration
            )
    
class Layer(m.VMobject):
    def __init__(
        self,
        values=None,
        max_display=16,
        node_radius=0.1,
        node_spacing=0.3,
        node_stroke_color=CHILL_BROWN,
        node_stroke_width=6,
        colormap=plt.get_cmap("viridis"),
        position=m.ORIGIN,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.values = values if values is not None else [0.5, 0.7, 0.3, 0.9, 0.1]
        self.max_display = max_display
        self.node_radius = node_radius
        self.node_spacing = node_spacing
        self.node_stroke_color = node_stroke_color
        self.node_stroke_width = node_stroke_width
        self.colormap = colormap
        self.position = position
        
        self.ellipsis_size = self.node_radius * 0.6
        self.ellipsis_spacing = self.node_radius * 2.5
        
        self.neurons = m.VGroup()
        self.ellipsis_dots = m.VGroup()
        
        self.build()
        self.move_to(position)
        
    def build(self):
        self.clear()
        self.neurons = m.VGroup()
        self.ellipsis_dots = m.VGroup()
        
        total_neurons = len(self.values)
        if total_neurons <= self.max_display:
            self._create_simple_layer()
        else:
            self._create_truncated_layer()
            
        self.ellipsis_dots.set_color(self.node_stroke_color)
            
        self.add(self.neurons, self.ellipsis_dots)
        
    def _create_simple_layer(self):
        total_neurons = len(self.values)
        total_height = self.node_spacing * (total_neurons - 1)
        
        for i, value in enumerate(self.values):
            y = total_height/2 - i * self.node_spacing
            neuron = Neuron(
                node_radius=self.node_radius,
                node_stroke_color=self.node_stroke_color,
                node_stroke_width=self.node_stroke_width,
                value=value,
                colormap=self.colormap
            ).move_to(m.UP * y)
            self.neurons.add(neuron)
            
    def _create_truncated_layer(self):
        visible_per_side = (self.max_display - 1) // 2
        ellipsis_vertical_space = 3 * (2 * self.ellipsis_size) + 2 * self.ellipsis_spacing
        
        total_height = (self.max_display - 1) * self.node_spacing + (ellipsis_vertical_space - self.node_spacing)
        start_y = (total_height / 2) - self.ellipsis_spacing
        
        for i in range(visible_per_side):
            y = start_y - i * self.node_spacing
            neuron = Neuron(
                node_radius=self.node_radius,
                node_stroke_color=self.node_stroke_color,
                node_stroke_width=self.node_stroke_width,
                value=self.values[i],
                colormap=self.colormap
            ).move_to(m.UP * y)
            self.neurons.add(neuron)
            
        last_top_node_y = start_y - (visible_per_side - 1) * self.node_spacing
        first_ellipsis_y = last_top_node_y - self.ellipsis_spacing
        
        for j in range(3):
            y_offset = first_ellipsis_y - j * self.ellipsis_spacing
            dot = m.Dot(radius=self.ellipsis_size, color=self.node_stroke_color).move_to(m.UP * y_offset)
            self.ellipsis_dots.add(dot)
            
        last_ellipsis_y = first_ellipsis_y - 2 * self.ellipsis_spacing
        first_bottom_node_y = last_ellipsis_y - self.ellipsis_spacing
        
        for i in range(visible_per_side):
            idx = len(self.values) - visible_per_side + i
            y = first_bottom_node_y - i * self.node_spacing
            neuron = Neuron(
                node_radius=self.node_radius,
                node_stroke_color=self.node_stroke_color,
                node_stroke_width=self.node_stroke_width,
                value=self.values[idx],
                colormap=self.colormap
            ).move_to(m.UP * y)
            self.neurons.add(neuron)
            
    def get_neuron(self, index):
        if index >= len(self.values):
            return None
            
        if len(self.values) <= self.max_display:
            return self.neurons[index]
            
        visible_per_side = (self.max_display - 1) // 2
        if index < visible_per_side:
            return self.neurons[index]
        elif index >= len(self.values) - visible_per_side:
            local_index = visible_per_side + (index - (len(self.values) - visible_per_side))
            return self.neurons[local_index]
        return None
        
    def update_values(self, new_values):
        self.values = new_values
        self.build()
        
    def set_value(self, index, value):
        if 0 <= index < len(self.values):
            self.values[index] = value
            neuron = self.get_neuron(index)
            if neuron:
                neuron.value = value
                style = neuron.map_value_to_style(value)
                neuron.node_fill_color = style["color"]
                neuron.node_fill_opacity = style["opacity"]
                neuron.build()
                
    def highlight_neuron(self, index, color=m.YELLOW):
        neuron = self.get_neuron(index)
        if neuron:
            original_stroke = neuron.node_stroke_color
            neuron.node_stroke_color = color
            neuron.build()
            return original_stroke
        return None
        
    def reset_highlight(self, index, original_color=None):
        if original_color is None:
            original_color = self.node_stroke_color
            
        neuron = self.get_neuron(index)
        if neuron:
            neuron.node_stroke_color = original_color
            neuron.build()
            
    def get_all_neurons(self):
        return self.neurons
    
class AttentionPatternLayer(m.VMobject):
    def __init__(
        self,
        matrices,
        max_display=5,
        spacing=0.3,
        square_size=0.3,
        dot_radius=0.075,
        min_opacity=0.2,
        max_opacity=1.0,
        stroke_width=1.0,
        stroke_color=CHILL_BROWN,
        frame_padding=0.2,
        dot_spacing=0.2, 
        frame_corner_radius=0.2,
        colormap=plt.get_cmap("viridis"),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.matrices = matrices
        self.max_display = max_display
        self.spacing = spacing
        self.square_size = square_size
        self.min_opacity = min_opacity
        self.max_opacity = max_opacity
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.frame_padding = frame_padding
        self.dot_spacing = dot_spacing
        self.dot_radius = dot_radius
        self.frame_corner_radius = frame_corner_radius
        self.colormap = colormap

        self.patterns = m.VGroup()
        self.ellipsis_dots = m.VGroup()
        self.camera.frame = None

        self.build()

    def build(self):
        self.clear()
        self.patterns = m.VGroup()
        self.ellipsis_dots = m.VGroup()

        if self.max_display >= len(self.matrices):
            self._create_simple_layer()
        else:
            self._create_truncated_layer()

        self.patterns.move_to(m.ORIGIN)
        self.ellipsis_dots.move_to(m.ORIGIN)
        self.ellipsis_dots.set_color(CHILL_BROWN)

        self.add(self.patterns, self.ellipsis_dots)

        content = m.VGroup(self.patterns, self.ellipsis_dots)
        content_center = content.get_center()
        content_width = content.width + 2 * self.frame_padding
        content_height = content.height + 2 * self.frame_padding

        self.camera.frame = m.RoundedRectangle(
            width=content_width,
            height=content_height,
            corner_radius=self.frame_corner_radius,
            stroke_color=self.stroke_color
        )
        self.camera.frame.move_to(content_center)
        self.add(self.camera.frame)


    def _create_simple_layer(self):
        total_height = self.spacing * (len(self.matrices) - 1)

        for i, mat in enumerate(self.matrices):
            y = total_height / 2 - i * self.spacing
            pattern = self._make_pattern(mat).move_to(m.UP * y)
            self.patterns.add(pattern)

    def _create_truncated_layer(self):
        total_to_show = self.max_display
        visible_per_side = total_to_show // 2
        extra_top = 1 if total_to_show % 2 else 0

        top_count = visible_per_side + extra_top
        bottom_count = visible_per_side

        top_count = min(top_count, len(self.matrices))
        bottom_count = min(bottom_count, len(self.matrices) - top_count)

        top_matrices = self.matrices[:top_count]
        bottom_matrices = self.matrices[-bottom_count:]

        top_patterns = m.VGroup(*[self._make_pattern(m) for m in top_matrices])
        bottom_patterns = m.VGroup(*[self._make_pattern(m) for m in bottom_matrices])

        pattern_height = top_patterns[0].height if top_patterns else 1.0
        actual_spacing = self.spacing * pattern_height
        top_patterns.arrange(m.DOWN, buff=actual_spacing)
        bottom_patterns.arrange(m.DOWN, buff=actual_spacing)

        gap_between_clusters = self.dot_spacing * 4
        bottom_patterns.next_to(top_patterns, m.DOWN, buff=gap_between_clusters)

        y_top = top_patterns.get_bottom()[1]
        y_bottom = bottom_patterns.get_top()[1]
        center_y = (y_top + y_bottom) / 2

        for j in range(3):
            dot = m.Dot(radius=self.dot_radius, color=self.stroke_color)
            dot.move_to(m.UP * (center_y - j * self.dot_spacing + self.dot_spacing))
            self.ellipsis_dots.add(dot)

        self.patterns.add(*top_patterns, *bottom_patterns)


    def _make_pattern(self, mat):
        return AttentionPattern(
            matrix=mat,
            square_size=self.square_size,
            min_opacity=self.min_opacity,
            max_opacity=self.max_opacity,
            stroke_width=self.stroke_width,
            stroke_color=self.stroke_color,
            colormap=self.colormap
        )        
    
class Network(m.VMobject):
    def __init__(
        self,
        weight_matrices, 
        layer_spacing=2.5,
        max_display=10,
        node_radius=0.1,
        node_spacing=0.3,
        node_stroke_color=CHILL_BROWN,
        node_stroke_width=6,
        colormap=plt.get_cmap("viridis"),
        connection_stroke_width=2,
        connection_opacity=0.8,
        connection_colormap=plt.get_cmap("viridis"),
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.weight_matrices = weight_matrices
        self.layer_spacing = layer_spacing
        self.max_display = max_display
        self.node_radius = node_radius
        self.node_spacing = node_spacing
        self.node_stroke_color = node_stroke_color
        self.node_stroke_width = node_stroke_width
        self.colormap = colormap
        self.connection_stroke_width = connection_stroke_width
        self.connection_opacity = connection_opacity
        self.connection_colormap = connection_colormap

        self.layers = m.VGroup()
        self.connections = m.VGroup()

        self.build()

    def build(self):
        self.clear()
        self.layers = m.VGroup()
        self.connections = m.VGroup()

        layer_sizes = [self.weight_matrices[0].shape[0]]  # input layer
        for matrix in self.weight_matrices:
            layer_sizes.append(matrix.shape[1])

        for i, size in enumerate(layer_sizes):
            values = [0.5] * size
            layer = Layer(
                values=values,
                max_display=self.max_display,
                node_radius=self.node_radius,
                node_spacing=self.node_spacing,
                node_stroke_color=self.node_stroke_color,
                node_stroke_width=self.node_stroke_width,
                colormap=self.colormap
            )
            layer.move_to(m.RIGHT * i * self.layer_spacing)
            self.layers.add(layer)

        self.add(self.layers)

        for i, matrix in enumerate(self.weight_matrices):
            layer_from = self.layers[i]
            layer_to = self.layers[i + 1]

            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    neuron_start = layer_from.get_neuron(r)
                    neuron_end = layer_to.get_neuron(c)
                    if neuron_start is None or neuron_end is None:
                        continue
                    
                    weight = matrix[r, c]
                    conn = Connection(
                        neuron_start,
                        neuron_end,
                        stroke_width=self.connection_stroke_width,
                        stroke_opacity=self.connection_opacity,
                        value=np.clip(abs(weight), 0, 1), 
                        weight=weight,
                        colormap=self.connection_colormap
                    )
                    self.connections.add(conn)

        self.add(self.connections)
        self.move_to(m.ORIGIN)

    def update_weights(self, new_matrices):
        self.weight_matrices = new_matrices
        self.build()

    def get_layer(self, index):
        if 0 <= index < len(self.layers):
            return self.layers[index]
        return None

        
class NeuronExample(m.Scene):
    def construct(self):
        neuron = Neuron()
        self.add(neuron)
        
        pass
        
class AttentionPatternExample(m.Scene):
    def construct(self):
        matrix = np.random.rand(4, 4)
        attn_pattern = AttentionPattern(matrix=matrix)
        
        self.add(attn_pattern)
        
        pass
               
class ConnectionExample(m.Scene):
    def construct(self):
        neuron1 = Neuron().move_to(m.LEFT)
        neuron2 = Neuron().move_to(m.RIGHT)
        
        connection = Connection(neuron1, neuron2)
        
        self.add(neuron1, neuron2, connection)
        
        pass
        
class LayerExample(m.Scene):
    def construct(self):
        values = [round(random.uniform(0, 1), 2) for _ in range(2048)]
        layer = Layer(values=values)

        self.add(layer)
        pass

class AttentionPatternLayerExample(m.Scene):
    def construct(self):
        matrices = [np.random.rand(5, 8) for _ in range(10)]
        attn_layer = AttentionPatternLayer(matrices=matrices, max_display=8)
        self.play(m.Create(attn_layer))
        pass

class NetworkExample(m.Scene):
    def construct(self):
        W1 = np.random.uniform(-1, 1, (4, 6))  # 4 -> 6
        W2 = np.random.uniform(-1, 1, (6, 3))  # 6 -> 3
        W3 = np.random.uniform(-1, 1, (3, 2))  # 3 -> 2

        net = Network([W1, W2, W3])
        self.add(net)
        pass

