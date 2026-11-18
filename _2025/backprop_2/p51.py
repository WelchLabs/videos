from welch_paths import welch_path
import manimgx as m
import pickle
from welch_scene import WelchScene
from welch_text import markup_text, math_tex
from functools import partial
import numpy as np
import torch
import sys
sys.path.append('_2025/backprop_2')
# from network_pranav_pr_1 import *
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import glob

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'
GREEN='#00a14b'

svg_path=welch_path('backprop2/graphics/to_manim')

# Helper function to get edge points between two circles
def get_edge_points(circle1, circle2, neuron_radius):
    # Get direction vector from circle1 to circle2
    direction = circle2.get_center() - circle1.get_center()
    unit_vector = direction / np.linalg.norm(direction)
    
    # Calculate start and end points
    start_point = circle1.get_center() + unit_vector * neuron_radius
    end_point = circle2.get_center() - unit_vector * neuron_radius
    
    return start_point, end_point


viridis_colormap=plt.get_cmap("viridis")
blues_colormap=plt.get_cmap("Blues")
custom_cmap_tans = mcolors.LinearSegmentedColormap.from_list('custom', ['#000000', '#dfd0b9'], N=256)
custom_cmap_cyan = mcolors.LinearSegmentedColormap.from_list('custom', ['#000000', '#00FFFF'], N=256)

# def get_nueron_color(value, vmin=-2, vmax=2):        
#         value_clipped = np.clip((value - vmin) / (vmax - vmin), 0, 1)
#         rgba = custom_cmap_tans(value_clipped) #Would also like to try a monochrome tan option
#         return Color(rgba[:3])

def get_nueron_color(value, vmax=0.95):        
    '''Uses abs, a little reductive'''
    value_clipped = np.clip(np.abs(value)/vmax, 0, 1)
    rgba = custom_cmap_tans(value_clipped) #Would also like to try a monochrome tan option
    return m.rgb_to_color(rgba[:3])

def get_grad_color(value): #, vmin=-2, vmax=2):        
    # value_clipped = np.clip((value - vmin) / (vmax - vmin), 0, 1)
    value_clipped = np.clip(np.abs(value), 0, 1)
    rgba = custom_cmap_cyan(value_clipped) #Would also like to try a monochrome tan option
    return m.rgb_to_color(rgba[:3])

class AttentionPattern(m.VMobject):
    def __init__(
        self,
        matrix,
        square_size=0.3,
        min_opacity=0.0,
        max_opacity=1.0,
        stroke_width=1.0,
        viz_scaling_factor=2.5, 
        stroke_color=CHILL_BROWN,
        colormap=custom_cmap_tans,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.matrix = np.array(matrix)
        self.n_rows, self.n_cols = self.matrix.shape
        self.square_size = square_size
        self.min_opacity = min_opacity
        # self.max_opacity = max_opacity
        self.max_opacity = np.max(self.matrix)
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self._colormap = colormap
        self.viz_scaling_factor=viz_scaling_factor

        self.build()

    def map_value_to_style(self, val):
        # val_clipped = np.clip(val, 0, 1)
        val_scaled=np.clip(self.viz_scaling_factor*val/self.max_opacity,0, 1)
        rgba = self._colormap(val_scaled)
        color = m.rgb_to_color(rgba[:3])
        # opacity = self.min_opacity + val_clipped * (self.max_opacity - self.min_opacity)
        # opacity=val_scaled
        opacity=1.0
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


def get_mlp(w1, 
            w2,
            neuron_fills=None, #Black if None
            grads_1=None,
            grads_2=None,
            line_weight=1.0, 
            line_opacity=0.5, 
            neuron_stroke_width=1.0, 
            neuron_stroke_color='#dfd0b9', 
            line_stroke_color='#948979', 
            connection_display_thresh=0.4):

    INPUT_NEURONS = w1.shape[0]
    HIDDEN_NEURONS = w1.shape[1]
    OUTPUT_NEURONS = w1.shape[0]
    NEURON_RADIUS = 0.06
    LAYER_SPACING = 0.23
    VERTICAL_SPACING = 0.18
    DOTS_SCALE=0.5
    
    # Create layers
    input_layer = m.VGroup()
    hidden_layer = m.VGroup()
    output_layer = m.VGroup()
    dots = m.VGroup()
    
    # Input layer
    neuron_count=0
    for i in range(INPUT_NEURONS):
        if i == w1.shape[0]//2:  # Middle position for ellipsis
            dot = math_tex("...").rotate(m.PI/2, m.OUT).scale(DOTS_SCALE).move_to(m.LEFT * LAYER_SPACING + m.UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            dot.set_color(neuron_stroke_color)
            dots.add(dot)
        else:
            neuron = m.Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
            neuron.set_stroke(width=neuron_stroke_width)
            if neuron_fills is None: 
                neuron.set_fill(color='#000000', opacity=1.0)
            else: 
                neuron.set_fill(color=get_nueron_color(neuron_fills[0][neuron_count], vmax=np.abs(neuron_fills[0]).max()), opacity=1.0)
            neuron.move_to(m.LEFT * LAYER_SPACING + m.UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            input_layer.add(neuron)
            neuron_count+=1
            
    # Hidden layer
    neuron_count=0
    for i in range(HIDDEN_NEURONS):
        if i == w1.shape[1]//2:  # Middle position for ellipsis
            dot = math_tex("...").rotate(m.PI/2, m.OUT).scale(DOTS_SCALE).move_to(m.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
            dot.set_color(neuron_stroke_color)
            dots.add(dot)
        else:
            neuron = m.Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
            neuron.set_stroke(width=neuron_stroke_width)
            if neuron_fills is None: 
                neuron.set_fill(color='#000000', opacity=1.0)
            else: 
                neuron.set_fill(color=get_nueron_color(neuron_fills[1][neuron_count], vmax=np.abs(neuron_fills[1]).max()), opacity=1.0)
            neuron.move_to(m.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
            hidden_layer.add(neuron)
            neuron_count+=1
            
    # Output layer
    neuron_count=0
    for i in range(OUTPUT_NEURONS):
        if i == w1.shape[0]//2:  # Middle position for ellipsis
            dot = math_tex("...").rotate(m.PI/2, m.OUT).scale(DOTS_SCALE).move_to(m.RIGHT * LAYER_SPACING + m.UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            dot.set_color(neuron_stroke_color)
            dots.add(dot)
        else:
            neuron = m.Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
            neuron.set_stroke(width=neuron_stroke_width)
            if neuron_fills is None: 
                neuron.set_fill(color='#000000', opacity=1.0)
            else: 
                neuron.set_fill(color=get_nueron_color(neuron_fills[2][neuron_count], vmax=np.abs(neuron_fills[2]).max()), opacity=1.0)
            neuron.move_to(m.RIGHT * LAYER_SPACING + m.UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            output_layer.add(neuron)
            neuron_count+=1
            
    # Create connections with edge points
    connections = m.VGroup()
    w1_abs=np.abs(w1)
    w1_scaled=w1_abs/np.percentile(w1_abs, 99)
    # w1_scaled=(w1-w1.min())/(w1.max()-w1.min())
    for i, in_neuron in enumerate(input_layer):
        for j, hidden_neuron in enumerate(hidden_layer):
            if np.abs(w1_scaled[i, j])<0.75: continue
            if abs(i-j)>6: continue #Let's try just drawing local ones. 
            start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
            line = m.Line(start_point, end_point)

            line.set_stroke(opacity=np.clip(w1_scaled[i,j], 0, 1), width=1.0*w1_scaled[i,j])
            # line.set_stroke(opacity=np.clip(0.8*(np.abs(w1[i, j])-connection_display_thresh), 0.1, 1), width=line_weight)
            
            line.set_color(line_stroke_color)
            connections.add(line)

    w2_abs=np.abs(w2)
    w2_scaled=w2_abs/np.percentile(w2_abs, 99)
    for i, hidden_neuron in enumerate(hidden_layer):
        for j, out_neuron in enumerate(output_layer):
            if np.abs(w2_scaled[i, j])<0.45: continue
            if abs(i-j)>6: continue #Let's try just drawing local ones.
            start_point, end_point = get_edge_points(hidden_neuron, out_neuron, NEURON_RADIUS)
            line = m.Line(start_point, end_point) #, stroke_opacity=line_opacity, stroke_width=line_weight)
            # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-connection_display_thresh), 0.1, 1), width=line_weight)
            line.set_stroke(opacity=np.clip(w2_scaled[i,j], 0, 1), width=1.0*w2_scaled[i,j])
            line.set_color(line_stroke_color)
            connections.add(line)

    grad_conections=m.VGroup()
    if grads_1 is not None:
        grads_1_abs=np.abs(grads_1)
        grads_1_scaled=grads_1_abs/np.percentile(grads_1_abs, 95)
        for i, in_neuron in enumerate(input_layer):
            for j, hidden_neuron in enumerate(hidden_layer):
                if np.abs(grads_1_scaled[i, j])<0.5: continue
                if abs(i-j)>6: continue #Need to dial this up or lost it probably, but it is helpful!
                start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
                line_grad = m.Line(start_point, end_point)
                # line_grad.set_stroke(opacity=np.clip(0.8*(np.abs(grads_1[i, j])-grad_display_thresh), 0, 1), 
                #                     width=np.abs(grads_1[i, j]))
                line_grad.set_stroke(opacity=np.clip(grads_1_scaled[i,j], 0, 1), width=np.clip(2.0*grads_1_scaled[i,j], 0, 3)) #width=1)
                # line.set_stroke(opacity=np.clip(grads_1_scaled[i,j], 0, 1), width=1.0) #0.1*grads_1_scaled[i,j])
                # print(np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1))
                line_grad.set_color(get_grad_color(grads_1_scaled[i, j]))
                grad_conections.add(line_grad)

            
    if grads_2 is not None:
        grads_2_abs=np.abs(grads_2)
        grads_2_scaled=grads_2_abs/np.percentile(grads_2_abs, 97)
        for i, hidden_neuron in enumerate(hidden_layer):
            for j, out_neuron in enumerate(output_layer):
                if np.abs(grads_2_scaled[i, j])<0.5: continue
                if abs(i-j)>6: continue #Need to dial this up or lost it probably, but it is helpful!
                start_point, end_point = get_edge_points(hidden_neuron, out_neuron, NEURON_RADIUS)
                line_grad = m.Line(start_point, end_point)
                # line_grad.set_stroke(opacity=np.clip(0.8*(np.abs(grads_2[i, j])-grad_display_thresh), 0, 1), 
                #                     width=np.abs(grads_2[i, j]))
                # line_grad.set_stroke(opacity=0.8, width=2)
                line_grad.set_stroke(opacity=np.clip(grads_2_scaled[i,j], 0, 1), width=np.clip(1.0*grads_2_scaled[i,j], 0, 3))
                # print(np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1))
                line_grad.set_color(get_grad_color(grads_2_scaled[i, j]))
                grad_conections.add(line_grad)

                
    return m.VGroup(connections, grad_conections, input_layer, hidden_layer, output_layer, dots)


def get_attention_layer(attn_patterns):
    num_attention_pattern_slots=len(attn_patterns)+1
    attention_pattern_spacing=0.51

    attention_border=m.RoundedRectangle(width=0.59, height=5.4, corner_radius=0.1)
    attention_border.set_stroke(width=1.0, color=CHILL_BROWN)


    attention_patterns=m.VGroup()
    connection_points_left=m.VGroup()
    connection_points_right=m.VGroup()

    attn_pattern_count=0
    for i in range(num_attention_pattern_slots):
        if i==num_attention_pattern_slots//2:
            dot = math_tex("...").rotate(m.PI/2, m.OUT).scale(0.5).move_to([0, num_attention_pattern_slots*attention_pattern_spacing/2 - attention_pattern_spacing*(i+0.5), 0])
            dot.set_color(CHILL_BROWN)
            attention_patterns.add(dot) #Just add here?
        else:
            if i>num_attention_pattern_slots//2: offset=0.15
            else: offset=-0.15 
            # matrix = np.random.rand(6, 6)
            attn_pattern = AttentionPattern(matrix=attn_patterns[attn_pattern_count], square_size=0.07, stroke_width=0.5)
            attn_pattern.move_to([0, num_attention_pattern_slots*attention_pattern_spacing/2+offset - attention_pattern_spacing*(i+0.5), 0])
            attention_patterns.add(attn_pattern)

            connection_point_left=m.Circle(radius=0)
            connection_point_left.move_to([-0.59/2.0, num_attention_pattern_slots*attention_pattern_spacing/2+offset - attention_pattern_spacing*(i+0.5), 0])
            connection_points_left.add(connection_point_left)

            connection_point_right=m.Circle(radius=0)
            connection_point_right.move_to([0.59/2.0, num_attention_pattern_slots*attention_pattern_spacing/2+offset - attention_pattern_spacing*(i+0.5), 0])
            connection_points_right.add(connection_point_right)
            attn_pattern_count+=1

    attention_layer=m.VGroup(attention_patterns, attention_border, connection_points_left, connection_points_right)
    return attention_layer

def get_mlp_connections_left(attention_connections_left, mlp_out, connection_points_left, attention_connections_left_grad=None):
    connections_left=m.VGroup()
    attention_connections_left_abs=np.abs(attention_connections_left)
    attention_connections_left_scaled=attention_connections_left_abs/np.max(attention_connections_left_abs) #np.percentile(attention_connections_left_abs, 99)
    for i, mlp_out_neuron in enumerate(mlp_out):
        for j, attention_neuron in enumerate(connection_points_left):
            if np.abs(attention_connections_left_scaled[i, j])<0.5: continue
            if abs(i/4-j)>3: continue #Need to dial this up or lost it probably, but it is helpful!
            start_point, end_point = get_edge_points(mlp_out_neuron, attention_neuron, 0.06)
            line = m.Line(start_point, attention_neuron.get_center())
            # line.set_stroke(width=1, opacity=0.3)
            # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
            line.set_stroke(opacity=np.clip(attention_connections_left_scaled[i,j], 0, 1), width=np.clip(1.0*attention_connections_left_scaled[i,j],0,3))
            line.set_color(CHILL_BROWN)
            connections_left.add(line)


    connections_left_grads=m.VGroup()
    if attention_connections_left_grad is not None: 
        attention_connections_left_grad_abs=np.abs(attention_connections_left_grad)
        attention_connections_left_grad_scaled=attention_connections_left_grad_abs/np.percentile(attention_connections_left_grad_abs, 98) #np.percentile(attention_connections_left_abs, 99)
        for i, mlp_out_neuron in enumerate(mlp_out):
            for j, attention_neuron in enumerate(connection_points_left):
                if np.abs(attention_connections_left_grad_scaled[i, j])<0.5: continue
                if abs(i/4-j)>3: continue #Need to dial this up or lost it probably, but it is helpful!
                start_point, end_point = get_edge_points(mlp_out_neuron, attention_neuron, 0.06)
                line = m.Line(start_point, attention_neuron.get_center())
                # line.set_stroke(width=1, opacity=0.3)
                # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
                line.set_stroke(opacity=np.clip(attention_connections_left_grad_scaled[i,j], 0, 1), width=np.clip(1.0*attention_connections_left_grad_scaled[i,j],0,2))
                line.set_color(get_grad_color(attention_connections_left_grad_scaled[i,j]))
                connections_left_grads.add(line)
    return connections_left, connections_left_grads



def get_mlp_connections_right(attention_connections_right, mlp_in, connection_points_right, attention_connections_right_grad=None):
    connections_right=m.VGroup()
    attention_connections_right_abs=np.abs(attention_connections_right)
    attention_connections_right_scaled=attention_connections_right_abs/np.percentile(attention_connections_right_abs, 99)
    for i, attention_neuron in enumerate(connection_points_right):
        for j, mlp_in_neuron in enumerate(mlp_in):
            if np.abs(attention_connections_right_scaled[i, j])<0.6: continue
            if abs(j/4-i)>3: continue #Need to dial this up or lost it probably, but it is helpful!
            start_point, end_point = get_edge_points(mlp_in_neuron, attention_neuron, 0.06)
            line = m.Line(start_point, attention_neuron.get_center())
            # line.set_stroke(width=1, opacity=0.3)
            # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
            line.set_stroke(opacity=np.clip(attention_connections_right_scaled[i,j], 0, 1), width=np.clip(1.0*attention_connections_right_scaled[i,j],0,3))
            line.set_color(CHILL_BROWN)
            connections_right.add(line)

    connections_right_grads=m.VGroup()
    if attention_connections_right_grad is not None: 
        attention_connections_right_grad_abs=np.abs(attention_connections_right_grad)
        attention_connections_right_grad_scaled=attention_connections_right_grad_abs/np.percentile(attention_connections_right_grad_abs, 98)
        for i, attention_neuron in enumerate(connection_points_right):
            for j, mlp_in_neuron in enumerate(mlp_in):
                if np.abs(attention_connections_right_grad_scaled[i, j])<0.5: continue
                if abs(j/4-i)>3: continue #Need to dial this up or lost it probably, but it is helpful!
                start_point, end_point = get_edge_points(mlp_in_neuron, attention_neuron, 0.06)
                line = m.Line(start_point, attention_neuron.get_center())
                # line.set_stroke(width=1, opacity=0.3)
                # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
                line.set_stroke(opacity=np.clip(attention_connections_right_grad_scaled[i,j], 0, 1), width=np.clip(1.0*attention_connections_right_grad_scaled[i,j],0,3))
                line.set_color(get_grad_color(attention_connections_right_grad_scaled[i,j]))
                connections_right_grads.add(line)
    return connections_right, connections_right_grads

def get_input_layer(prompt_neuron_indices, snapshot, num_input_neurons=36):
    input_layer_nuerons=m.VGroup()
    input_layer_text=m.VGroup()
    vertical_spacing = 0.18
    neuron_radius = 0.06
    neuron_stroke_color='#dfd0b9'
    neuron_stroke_width= 1.0
    words_to_nudge={' capital':-0.02}

    prompt_token_count=0
    neuron_count=0
    for i in range(num_input_neurons):
        if i == num_input_neurons//2:  # Middle position for ellipsis
            dot = math_tex("...").rotate(m.PI/2, m.OUT).scale(0.4).move_to(m.UP * ((num_input_neurons//2 - i) * vertical_spacing))
            dot.set_color(neuron_stroke_color)
        else:
            neuron = m.Circle(radius=neuron_radius, stroke_color=neuron_stroke_color)
            neuron.set_stroke(width=neuron_stroke_width)
            if neuron_count in prompt_neuron_indices:
                neuron.set_fill(color='#dfd0b9', opacity=1.0)
                t=m.Text(snapshot['prompt.tokens'][prompt_token_count], font_size=24, font='myriad-pro')
                t.set_color(neuron_stroke_color)
                # print(t.get_center())
                t.move_to((0.2+t.get_right()[0])*m.LEFT+m.UP * ((-t.get_bottom()+num_input_neurons//2 - i) * vertical_spacing))
                if snapshot['prompt.tokens'][prompt_token_count] in words_to_nudge.keys():
                    t.shift([0, words_to_nudge[snapshot['prompt.tokens'][prompt_token_count]], 0])

                input_layer_text.add(t)
                prompt_token_count+=1 
            else:
                neuron.set_fill(color='#000000', opacity=1.0)

            neuron.move_to(m.UP * ((num_input_neurons//2 - i) * vertical_spacing))
            input_layer_nuerons.add(neuron)
            neuron_count+=1

    input_layer=m.VGroup(input_layer_nuerons, dot, input_layer_text)
    return input_layer


def get_output_layer(snapshot, empty=False):
    output_layer_nuerons=m.VGroup()
    output_layer_text=m.VGroup()
    num_output_neurons=36   
    vertical_spacing = 0.18
    neuron_radius = 0.06
    neuron_stroke_color='#dfd0b9'
    neuron_stroke_width= 1.0

    neuron_count=0
    for i in range(num_output_neurons):
        if i == num_output_neurons//2:  # Middle position for ellipsis
            dot = math_tex("...").rotate(m.PI/2, m.OUT).scale(0.4).move_to(m.UP * ((num_output_neurons//2 - i) * vertical_spacing))
            dot.set_color(neuron_stroke_color)
        else:
            n = m.Circle(radius=neuron_radius, stroke_color=neuron_stroke_color)
            n.set_stroke(width=neuron_stroke_width)
            if not empty: 
                n.set_fill(color=get_nueron_color(snapshot['topk.probs'][neuron_count],vmax=np.max(snapshot['topk.probs'])), opacity=1.0)
                if neuron_count==0: font_size=22
                elif neuron_count<4: font_size=16
                else: font_size=12 
                t=m.Text(snapshot['topk.tokens'][neuron_count], font_size=font_size, font='myriad-pro')
                # t.set_color(neuron_stroke_color)
                # get_nueron_color(neuron_fills[0][neuron_count], vmax=np.abs(neuron_fills[0]).max())
                text_color=get_nueron_color(np.clip(snapshot['topk.probs'][neuron_count],0.1, 1.0),vmax=np.max(snapshot['topk.probs'])),
                t.set_color(text_color)
                t.set_opacity(np.clip(snapshot['topk.probs'][neuron_count], 0.3, 1.0))
                t.move_to((0.2+t.get_right()[0])*m.RIGHT+ m.UP* ((-t.get_bottom()+num_output_neurons//2 - i) * vertical_spacing))
                output_layer_text.add(t)

            else: 
                n.set_fill(color='#000000', opacity=1.0)

            #I like the idea of having probs on here, but I think it's too much right now, mayb in part 3
            # if neuron_count<5:
            #     t2=Text(f"{snapshot['topk.probs'][neuron_count]:.4f}", font_size=12)
            #     t2.set_color(neuron_stroke_color)
            #     t2.set_opacity(np.clip(snapshot['topk.probs'][neuron_count], 0.4, 0.7))
            #     t2.move_to(t.get_right()+np.array([0.2, 0, 0]))
            #     output_layer_text.add(t2)

            n.move_to(m.UP * ((num_output_neurons//2 - i) * vertical_spacing))
            output_layer_nuerons.add(n)
            neuron_count+=1
    output_layer=m.VGroup(output_layer_nuerons, dot, output_layer_text)
    return output_layer


class P51v2(WelchScene):
    def construct(self):
        '''
        Getting close! Next hurdle is to bring in different examples. 
        '''


        pickle_path=welch_path('backprop2/hackin/jun_3_1')
        snapshots=[]
        for p in sorted(glob.glob(pickle_path+'/*.p')):
            with open(p, 'rb') as f:
                snapshots.append(pickle.load(f))

        #Ok i should maybe refactor - but not sure if it's worth it - I need to capture these things from each snapshot I look at: 
        # backward_pass=VGroup(we_connections_grad, *all_grads, wu_connections_grad)
        #forward_pass=VGroup(input_layer[:-1], *all_activations, output_layer[:-1]) 
        all_backward_passes=[]
        all_forward_passes=[]

        random_seeds=[25, 26, 27, 28, 29, 30, 31, 32, 33, 34] #For ordering input neurons
        for snapshot_count, snapshot_index in enumerate([0, 1, 2]):
            snapshot=snapshots[snapshot_index]

            all_weights=m.VGroup()
            all_activations=m.VGroup()
            all_activations_empty=m.VGroup()
            all_grads=m.VGroup()
            random_background_stuff=m.VGroup()


            mlps=[]
            attns=[]
            start_x=-4.0
            # for layer_count, layer_num in enumerate([0, 1, 2, 3, 12, 13, 14, 15]):
            # for layer_count, layer_num in enumerate([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]):
            for layer_count, layer_num in enumerate([0, 1, 2, 11, 12, 13, 14, 15]):

                #Kinda clunky interface but meh
                neuron_fills=[snapshot['blocks.'+str(layer_num)+'.hook_resid_mid'],
                              snapshot['blocks.'+str(layer_num)+'.mlp.hook_post'],
                              snapshot['blocks.'+str(layer_num)+'.hook_mlp_out']]
                w1=snapshot['blocks.'+str(layer_num)+'.mlp.W_in']
                w2=snapshot['blocks.'+str(layer_num)+'.mlp.W_out']
                grads_1=snapshot['blocks.'+str(layer_num)+'.mlp.W_in.grad']
                grads_2=snapshot['blocks.'+str(layer_num)+'.mlp.W_out.grad']
                all_attn_patterns=snapshot['blocks.'+str(layer_num)+'.attn.hook_pattern']
                wO_full=snapshot['blocks.'+str(layer_num)+'.attn.W_O']
                wq_full=snapshot['blocks.'+str(layer_num)+'.attn.W_Q']
                wO_full_grad=snapshot['blocks.'+str(layer_num)+'.attn.W_O.grad']
                wq_full_grad=snapshot['blocks.'+str(layer_num)+'.attn.W_Q.grad']

                attn_patterns=[]
                wos=[]; wqs=[]
                wosg=[]; wqsg=[]
                for i in range(0, 30, 3): #Just take every thrid pattern for now. 
                    attn_patterns.append(all_attn_patterns[0][i][1:,1:]) #Ignore BOS token
                    wos.append(wO_full[i, 0])
                    wqs.append(wq_full[i, :, 0])
                    wosg.append(wO_full_grad[i, 0])
                    wqsg.append(wq_full_grad[i, :, 0])
                wos=np.array(wos); wqs=np.array(wqs)
                wosg=np.array(wosg); wqsg=np.array(wqsg)
                attention_connections_left=wqs.T #Queries
                attention_connections_right=wos
                attention_connections_left_grad=wqsg.T #Queries
                attention_connections_right_grad=wosg

                attn=get_attention_layer(attn_patterns)
                attn.move_to([start_x+layer_count*1.6, 0, 0]) 
                attns.append(attn)
                all_activations.add(attn[0])
                random_background_stuff.add(attn[1])

                mlp=get_mlp(w1, w2, neuron_fills, grads_1=grads_1, grads_2=grads_2)
                mlp.move_to([start_x+0.8+layer_count*1.6, 0, 0])
                mlps.append(mlp)
                # all_activations.add(*mlp[2:-1]) #Skip weights and connections
                all_activations.add(mlp[2:-1]) #Try as a single block, might actually animate better?
                random_background_stuff.add(mlp[-1])

                attn_empty=get_attention_layer([np.zeros_like(all_attn_patterns[0][0][1:,1:]) for i in range(len(attn_patterns))])
                attn_empty.move_to([start_x+layer_count*1.6, 0, 0]) 
                all_activations_empty.add(attn_empty[0])

                mlp_empty=get_mlp(w1, w2)
                mlp_empty.move_to([start_x+0.8+layer_count*1.6, 0, 0])
                all_activations_empty.add(*mlp_empty[2:-1]) #Skip weights and connections


                connections_right, connections_right_grads=get_mlp_connections_right(attention_connections_right=attention_connections_right, 
                                                                                   mlp_in=mlp[2],
                                                                                   connection_points_right=attn[3],
                                                                                   attention_connections_right_grad=attention_connections_right_grad)


                if len(mlps)>1:
                    connections_left, connections_left_grads=get_mlp_connections_left(attention_connections_left=attention_connections_left, 
                                                                                  mlp_out=mlps[-2][4],
                                                                                  connection_points_left=attn[2],
                                                                                  attention_connections_left_grad=attention_connections_left_grad)
                    # self.add(connections_left)
                    all_weights.add(connections_left)
                    # self.add(connections_left_grads)
                    all_grads.add(connections_left_grads)

                # self.add(connections_right)
                all_weights.add(connections_right)
                # self.add(connections_right_grads)
                all_grads.add(connections_right_grads)

                all_weights.add(mlp[0])
                all_grads.add(mlp[1])

            #Inputs 
            num_input_neurons = 36
            np.random.seed(random_seeds[snapshot_count]) #Need to figure out how to add variety withotu moving the same token like "The" around
            prompt_neuron_indices=np.random.choice(np.arange(36), len(snapshot['prompt.tokens'])-1) #Don't include last token

            input_layer = get_input_layer(prompt_neuron_indices, snapshot, num_input_neurons=num_input_neurons)
            input_layer.move_to([-4.7, 0, 0], aligned_edge=m.RIGHT)

            input_layer_empty = get_input_layer([], snapshot, num_input_neurons=num_input_neurons)
            input_layer_empty.move_to([-4.7, 0, 0], aligned_edge=m.RIGHT)


            # Okie dokie -> Let me add intput/first attention layer connections - this will need to be a separate function
            # Ok right and I need to bring two matrices together here -> hmm. 
            
            all_embeddings=[]; all_embeddings_grad=[]; prompt_token_embeddings=[]; prompt_token_embeddings_grad=[]
            for i in range(0, 30, 3):
                all_embeddings.append(snapshot['embed.W_E'][0, :num_input_neurons, i])
                all_embeddings_grad.append(snapshot['embed.W_E.grad'][0, :num_input_neurons, i])
                prompt_token_embeddings.append(snapshot['prompt.embed.W_E'][:, 0, i])
                prompt_token_embeddings_grad.append(snapshot['prompt.embed.W_E.grad'][:, 0, i])
            all_embeddings=np.array(all_embeddings).T
            all_embeddings_grad=np.array(all_embeddings_grad).T
            prompt_token_embeddings=np.array(prompt_token_embeddings).T
            prompt_token_embeddings_grad=np.array(prompt_token_embeddings_grad).T

            for count, i in enumerate(prompt_neuron_indices):
                all_embeddings[i,:]=prompt_token_embeddings[count, :]
                all_embeddings_grad[i,:]=prompt_token_embeddings_grad[count, :]

            we_connections=m.VGroup()
            all_embeddings_abs=np.abs(all_embeddings)
            all_embeddings_scaled=all_embeddings_abs/np.percentile(all_embeddings_abs, 95)
            for i, n1 in enumerate(input_layer[0]):
                for j, n2 in enumerate(attns[0][2]):
                    # if np.abs(all_embeddings_scaled[i, j])<0.1: continue
                    if abs(j-i/4)>3: continue #Need to dial this up or lost it probably, but it is helpful!
                    start_point, end_point = get_edge_points(n1, n2, 0.06)
                    line = m.Line(start_point, n2.get_center())
                    # line.set_stroke(width=1, opacity=0.3)
                    # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
                    line.set_stroke(opacity=np.clip(all_embeddings_scaled[i,j], 0.4, 1), width=np.clip(1.0*all_embeddings_scaled[i,j],0.5,1.7))
                    line.set_color(CHILL_BROWN)
                    we_connections.add(line)


            we_connections_grad=m.VGroup()
            all_embeddings_grad_abs=np.abs(all_embeddings_grad)
            all_embeddings_grad_scaled=all_embeddings_grad_abs/np.percentile(all_embeddings_grad_abs, 95)
            for i, n1 in enumerate(input_layer[0]):
                for j, n2 in enumerate(attns[0][2]):
                    # if np.abs(all_embeddings_grad_scaled[i, j])<0.1: continue
                    if abs(j-i/4)>4: continue #Need to dial this up or lost it probably, but it is helpful!
                    start_point, end_point = get_edge_points(n1, n2, 0.06)
                    line = m.Line(start_point, n2.get_center())
                    # line.set_stroke(width=1, opacity=0.3)
                    # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
                    # line.set_stroke(opacity=np.clip(all_embeddings_grad_scaled[i,j], 0.4, 1), width=np.clip(1.0*all_embeddings_grad_scaled[i,j],0.5,1.7))
                    # line.set_color(CHILL_BROWN)
                    line.set_stroke(opacity=np.clip(all_embeddings_grad_scaled[i,j], 0, 1), width=np.clip(1.0*all_embeddings_grad_scaled[i,j],0,3))
                    line.set_color(get_grad_color(all_embeddings_grad_scaled[i,j]))
                    we_connections_grad.add(line)


            #Ok I should probably go ahead and wrap up input stuff but I don't really want to -> 
            #'topk.indices', 'topk.tokens', 'topk.probs', 'topk.unembed.W_U', 'topk.unembed.W_U.grad'

            output_layer=get_output_layer(snapshot)
            output_layer.move_to([mlps[-1].get_right()[0]+0.36, -3.21, 0], aligned_edge=m.LEFT+(m.config.frame_height / 2 * m.DOWN)) #Was 5.45

            output_layer_empty=get_output_layer(snapshot, empty=True)
            output_layer_empty.move_to([mlps[-1].get_right()[0]+0.36, -3.21, 0], aligned_edge=m.LEFT+(m.config.frame_height / 2 * m.DOWN))


            wu_connections=m.VGroup()
            unembed_abs=np.abs(snapshot['topk.unembed.W_U'][:,0,:].T)
            unembed_scaled=unembed_abs/np.percentile(unembed_abs, 98)
            for i, n1 in enumerate(mlps[-1][4]):
                for j, n2 in enumerate(output_layer[0]):
                    if np.abs(unembed_scaled[i, j])<0.5: continue
                    if abs(j-i)>8: continue #Need to dial this up or lost it probably, but it is helpful!
                    start_point, end_point = get_edge_points(n1, n2, 0.06)
                    line = m.Line(start_point, n2.get_center())
                    # line.set_stroke(width=1, opacity=0.3)
                    # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
                    line.set_stroke(opacity=np.clip(unembed_scaled[i,j], 0.4, 1), width=np.clip(1.0*unembed_scaled[i,j],0.5,1.7))
                    line.set_color(CHILL_BROWN)
                    wu_connections.add(line)

            wu_connections_grad=m.VGroup()
            unembed_grad_abs=np.abs(snapshot['topk.unembed.W_U.grad'][:,0,:].T)
            unembed_scaled_grad=unembed_grad_abs/np.percentile(unembed_grad_abs, 99)
            for i, n1 in enumerate(mlps[-1][4]):
                for j, n2 in enumerate(output_layer[0]):
                    if np.abs(unembed_scaled_grad[i, j])<0.5: continue
                    if abs(j-i)>8: continue #Need to dial this up or lost it probably, but it is helpful!
                    start_point, end_point = get_edge_points(n1, n2, 0.06)
                    line = m.Line(start_point, n2.get_center())
                    # line.set_stroke(width=1, opacity=0.3)
                    # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
                    # line.set_stroke(opacity=np.clip(unembed_scaled_grad[i,j], 0.4, 1), width=np.clip(1.0*unembed_scaled_grad[i,j],0.5,1.7))
                    # line.set_color(CHILL_BROWN)
                    line.set_stroke(opacity=np.clip(unembed_scaled_grad[i,j], 0, 1), width=np.clip(0.7*unembed_scaled_grad[i,j],0,3))
                    line.set_color(get_grad_color(unembed_scaled_grad[i,j]))
                    wu_connections_grad.add(line)

            all_backward_passes.append(m.VGroup(we_connections_grad, *all_grads, wu_connections_grad))
            # all_forward_passes.append(VGroup(input_layer[:-1], *all_activations, output_layer[:-1])) #Leaving out input output text for now. 
            all_forward_passes.append(m.VGroup(input_layer, *all_activations, output_layer))

        ## -- end big ole loop
        self.wait()

        # self.set_view(0, 0, 0, (2.04, -0.57, 0.0), 8.63)
        # self.set_view(0, 0, 0, (1.99, -0.04, 0.0), 8.32)
        # self.set_view(0, 0, 0, (2.06, -0.06, 0.0), 9.36)
        self.set_view(0, 0, 0, (2.05, -0.07, 0.0), 9.16)
        self.add(random_background_stuff)
        self.add(we_connections, all_weights, wu_connections)

        for backward_pass in all_backward_passes:
            self.add(backward_pass)
            backward_pass.set_opacity(0.0)

        self.add(input_layer_empty, all_activations_empty, output_layer_empty)
        for a in all_activations_empty: #Walk through and correct occlusions
            if len(a)>0: 
                self.remove(a[1])
                self.add(a[1])
        self.remove(input_layer_empty[0]); self.add(input_layer_empty[0])

        for forward_pass in all_forward_passes:
            self.add(forward_pass)
            forward_pass.set_opacity(0.0)

        self.wait()

        #Show some forward prop:
        for i in range(9):
            all_forward_passes[0][i].set_opacity(1.0)
            self.wait(0.1)
        self.wait()

        #Fade out half of model and go 3d, then residual stream animates out!
        # self.remove(random_background_stuff[8:])
        # self.remove(wu_connections)
        # self.remove(all_weights[11:])
        # self.remove(all_activations_empty[16:])
        # self.remove(output_layer_empty)
        # self.set_view(43, 52, 0, (2.85, -0.15, -0.84), 9.16)

        self.play(m.FadeOut(random_background_stuff[8:]), m.FadeOut(wu_connections), m.FadeOut(all_weights[11:]),
                  m.FadeOut(all_activations_empty[16:]), m.FadeOut(all_activations_empty[16:]), m.FadeOut(output_layer_empty),
                  self.animate_view(43, 52, 0, (2.85, -0.15, -0.84), 9.16), run_time=5)

        residual_stream_matrix_entries_1 = [
            ["0.09", "-0.25", "0.43", "\\ldots", "-0.17", "0.27", "-0.10"],
            ["-0.18", "0.15", "-0.42", "\\ldots", "0.33", "-0.07", "0.18"],
            ["0.28", "-0.04", "0.19", "\\ldots", "-0.31", "0.06", "-0.15"],
            ["-0.12", "0.37", "-0.29", "\\ldots", "0.14", "-0.23", "0.41"],
            ["0.35", "-0.08", "0.52", "\\ldots", "-0.19", "0.11", "-0.36"]
        ]
        residual_stream_matrix_entries_2 = [
            ["-0.18", "0.72", "-0.35", "\\ldots", "0.41", "-0.56", "0.29"],
            ["0.63", "-0.24", "0.47", "\\ldots", "-0.38", "0.15", "-0.69"],
            ["-0.52", "0.31", "-0.74", "\\ldots", "0.26", "-0.43", "0.68"],
            ["0.19", "-0.61", "0.36", "\\ldots", "-0.57", "0.42", "-0.28"],
            ["-0.45", "0.58", "-0.21", "\\ldots", "0.34", "-0.67", "0.53"]
        ]
        residual_stream_matrix_entries_3 = [
            ["0.84", "-0.13", "0.56", "\\ldots", "-0.39", "0.27", "-0.61"],
            ["-0.76", "0.48", "-0.32", "\\ldots", "0.55", "-0.18", "0.43"],
            ["0.29", "-0.64", "0.77", "\\ldots", "-0.41", "0.16", "-0.52"],
            ["-0.35", "0.59", "-0.24", "\\ldots", "0.68", "-0.46", "0.33"],
            ["0.72", "-0.28", "0.45", "\\ldots", "-0.57", "0.38", "-0.66"]
        ]
        residual_stream_matrix_entries_4 = [
            ["-0.47", "0.26", "-0.68", "\\ldots", "0.35", "-0.54", "0.41"],
            ["0.59", "-0.33", "0.18", "\\ldots", "-0.62", "0.29", "-0.45"],
            ["-0.71", "0.44", "-0.27", "\\ldots", "0.48", "-0.36", "0.63"],
            ["0.23", "-0.56", "0.74", "\\ldots", "-0.31", "0.67", "-0.52"],
            ["-0.38", "0.65", "-0.19", "\\ldots", "0.42", "-0.58", "0.34"]
        ]
        residual_stream_matrix_entries_5 = [
            ["0.32", "-0.67", "0.44", "\\ldots", "-0.25", "0.59", "-0.37"],
            ["-0.48", "0.23", "-0.71", "\\ldots", "0.36", "-0.54", "0.68"],
            ["0.61", "-0.29", "0.15", "\\ldots", "-0.46", "0.73", "-0.34"],
            ["-0.57", "0.41", "-0.63", "\\ldots", "0.28", "-0.49", "0.52"],
            ["0.74", "-0.16", "0.39", "\\ldots", "-0.65", "0.33", "-0.58"]
        ]
        residual_stream_matrix_entries_6 = [
            ["-0.63", "0.38", "-0.51", "\\ldots", "0.47", "-0.29", "0.64"],
            ["0.26", "-0.72", "0.45", "\\ldots", "-0.34", "0.58", "-0.43"],
            ["-0.49", "0.67", "-0.23", "\\ldots", "0.31", "-0.65", "0.52"],
            ["0.74", "-0.18", "0.36", "\\ldots", "-0.59", "0.42", "-0.68"],
            ["-0.37", "0.53", "-0.66", "\\ldots", "0.28", "-0.44", "0.71"]
        ]

        residual_stream = m.Matrix(residual_stream_matrix_entries_1)
        box=m.SurroundingRectangle(residual_stream[28:35], buff=0.25).set_color(YELLOW).set_stroke(width=3)
        matrix_and_box=m.VGroup(residual_stream, box)

        matrix_and_box.scale(0.5)
        matrix_and_box.rotate(90*m.DEGREES, [1,0,0])
        matrix_and_box.rotate(90*m.DEGREES, [0,0,1])
        matrix_and_box.move_to([1.8, 0.2, -1.4])

        #Pretty sure this could be a one liner - eh
        residual_stream_matrics=m.VGroup()
        residual_stream_matrics.add(residual_stream)
        for r in [residual_stream_matrix_entries_2, residual_stream_matrix_entries_3, residual_stream_matrix_entries_4, 
                    residual_stream_matrix_entries_5, residual_stream_matrix_entries_6]:
            rm=m.Matrix(r)
            rm.scale(0.5)
            rm.rotate(90*m.DEGREES, [1,0,0])
            rm.rotate(90*m.DEGREES, [0,0,1])
            rm.move_to([1.8, 0.2, -1.4])
            residual_stream_matrics.add(rm)


        # len( all_forward_passes[0][8][2]) 
        self.wait()
        self.play(m.ReplacementTransform(all_forward_passes[0][8][2].copy(), residual_stream[6::-1]),
                  m.ReplacementTransform(all_forward_passes[0][8][2].copy(), residual_stream[13:6:-1]),
                  m.ReplacementTransform(all_forward_passes[0][8][2].copy(), residual_stream[20:13:-1]),
                  m.ReplacementTransform(all_forward_passes[0][8][2].copy(), residual_stream[27:20:-1]),
                  m.ReplacementTransform(all_forward_passes[0][8][2], residual_stream[34:27:-1])
                  )
        self.add(residual_stream)
        self.wait()

        #Now can I have a box around the final row??
        self.play(m.Create(box))
        self.wait()


        # Ok now setup for big data passing through animation - not sure how much of this I want to do in manim vs not. 
        # I think first thing will be to zoom out to make room for everything, fade out input text. 
        # I think I can "preload" other patterns basically that I can flip through here. 
        self.play(
                 # FadeOut(box), ## Actually keep box?
                   all_forward_passes[0][:9].animate.set_opacity(0.0), 
                  self.animate_view(52, 57, 0, (2.16, -1.28, -1.97), 11.31), run_time=4
                  )
        self.wait()


        matrix_entries_2 = [
            ["-0.45", "0.58", "-0.21", "\\ldots", "0.34", "-0.67", "0.53"],
            ["0.72", "-0.28", "0.45", "\\ldots", "-0.57", "0.38", "-0.66"],
            ["-0.38", "0.65", "-0.19", "\\ldots", "0.42", "-0.58", "0.34"],
            ["0.74", "-0.16", "0.39", "\\ldots", "-0.65", "0.33", "-0.58"],
            ["-0.37", "0.53", "-0.66", "\\ldots", "0.28", "-0.44", "0.71"],
            ["0.35", "-0.08", "0.52", "\\ldots", "-0.19", "0.11", "-0.36"]
        ]
        results_matrix = m.Matrix(matrix_entries_2)

        #Ok last little thing I want here is little colored city labels next to each row. 
        ##ok i was thinking do labels in manim, but I think Illustrator is the way to go! 3d sucks for this!
        #Ok illustrator is going to be hard actually due to camera moves. What if i add the labels before moving this puppy and move 
        #them as a group??
        paris_label=markup_text('PARIS', font_size=36, font='myriad-pro', weight='semi-bold').set_color(YELLOW)
        paris_label_2=paris_label.copy()
        berlin_label=markup_text('BERLIN', font_size=36, font='myriad-pro', weight='semi-bold').set_color(GREEN)
        berlin_label_2=berlin_label.copy()
        madrid_label=markup_text('MADRID', font_size=36, font='myriad-pro', weight='semi-bold').set_color("#00FFFF")
        madrid_label_2=madrid_label.copy()
        paris_label.move_to([-7.5, 2.1, 0])
        paris_label_2.next_to(paris_label, m.DOWN, buff=0.57)
        berlin_label.next_to(paris_label_2, m.DOWN, buff=0.57)
        berlin_label_2.next_to(berlin_label, m.DOWN, buff=0.57)
        madrid_label.next_to(berlin_label_2, m.DOWN, buff=0.57)
        madrid_label_2.next_to(madrid_label, m.DOWN, buff=0.57)
        results_and_labels=m.VGroup(results_matrix, paris_label, paris_label_2, berlin_label, berlin_label_2, madrid_label, madrid_label_2)
        # results_and_labels_co[u]

        # self.add(results_matrix)
        # self.set_view(0, 0, 0, (-0.15, -0.17, 0.0), 10.08)
        # self.add(results_and_labels)
        # self.remove(results_and_labels)


        results_and_labels.rotate(52 * m.DEGREES, axis=m.OUT)    # theta rotation
        local_right = np.array([np.cos(52 * m.DEGREES), np.sin(52 * m.DEGREES), 0])
        results_and_labels.rotate(57 * m.DEGREES, axis=local_right)
        results_and_labels.scale(0.3)
        results_and_labels.move_to([8.9, -3, 0])
        # self.add(results_and_labels) #ok cool I can build this up using replacement transform, and add labels. 

        # self.remove(results_matrix)
        # Alright now I need 6 wikitext examples.

        #Paris
        t1_string="... It ended on September 3, 1783 \n when Britain accepted American \n independence in the Treaty of ..."
        t2_string="... Gershwin did not particularly like \n Walter Damrosch's interpretation at \n the world premiere of An American in ..."
        
        #Berlin
        t3_string="... As Schopenhauer was preparing \n to escape from ..."
        t4_string="... The Berolina statue had already been \n removed in 1944 and probably melted \n down for use in arms production. During \n the Battle of ..."

        #Madrid
        t5_string="... Agassi's US Open finish, along \n with his Masters Series victories in \n Key Biscayne, Rome and ..."
        t6_string="... Within a few days after Canovas \n del Castillo took power as Premier, \n the new king, proclaimed on 29 \n December 1874, arrived at ..."

        texts=m.VGroup()
        for t in [t1_string, t2_string, t3_string, t4_string, t5_string, t6_string]:
            t1=markup_text(t, font_size=18, alignment='left')
            t1.rotate(52 * m.DEGREES, axis=m.OUT)    # theta rotation
            local_right = np.array([np.cos(52 * m.DEGREES), np.sin(52 * m.DEGREES), 0])
            t1.rotate(57 * m.DEGREES, axis=local_right)
            texts.add(t1)

        #Position tuning
        texts[0].move_to([5.0, -7.5, 0.6], aligned_edge=m.LEFT)
        texts[1].move_to([5.0, -7.5, 0.6], aligned_edge=m.LEFT)
        texts[2].move_to([5.0, -7.5, 0.6], aligned_edge=m.LEFT)
        texts[3].move_to([5.0, -7.5, 0.6], aligned_edge=m.LEFT)
        texts[4].move_to([5.0, -7.5, 0.6], aligned_edge=m.LEFT)
        texts[5].move_to([5.0, -7.5, 0.6], aligned_edge=m.LEFT)

        
        # self.add(texts[5])
        # self.remove(texts[5])

        # Debergsing
        # i=0
        # self.remove(residual_stream_matrics[i])
        # self.add(residual_stream_matrics[(i+1)%len(residual_stream_matrics)])
        # self.play(ReplacementTransform(residual_stream_matrics[(i+1)%len(residual_stream_matrics)][28:35].copy(), results_matrix[i*7:(i+1)*7]))
        # i=0
        # t=texts[i]
        # if i>0: self.remove(texts[i-1])
        # self.add(t)
        # self.wait()
        # self.play(ReplacementTransform(t.copy(), forward_pass[0][0][::-1])) 
        # for j in range(1, 9):
        #     all_forward_passes[i%len(all_forward_passes)][j].set_opacity(1.0)
        #     self.wait(0.1)
        # self.remove(residual_stream_matrics[i]); self.add(residual_stream_matrics[(i+1)%len(residual_stream_matrics)])

        # self.remove(residual_stream_matrics[i+1]);
        # self.add(residual_stream_matrics[i])
        # self.remove(residual_stream_matrics[i])
        # all_forward_passes[0][8][2].set_opacity(0.0)

        # self.play(ReplacementTransform(residual_stream_matrics[(i+1)%len(residual_stream_matrics)][28:35].copy(), results_matrix[i*7:(i+1)*7]))
        # self.wait()
        # for j in range(1, 9):
        #     all_forward_passes[i%len(all_forward_passes)][j].set_opacity(0.0)

        self.wait()
        for i, t in enumerate(texts):
            if i>0: self.remove(texts[i-1])
            self.add(t)
            self.wait()
            self.play(m.ReplacementTransform(t.copy(), forward_pass[0][0][::-1]))
            for j in range(1, 8):
                all_forward_passes[i%len(all_forward_passes)][j].set_opacity(1.0)
                self.wait(0.1)
            all_forward_passes[i%len(all_forward_passes)][8].set_opacity(1.0) #Bring last step out here so it happens w/ matrix change

            self.remove(residual_stream_matrics[i]); self.add(residual_stream_matrics[(i+1)%len(residual_stream_matrics)])
            all_forward_passes[0][8][2].set_opacity(0.0) #Kinda random cleanup i have to do I guess. 

            self.play(m.ReplacementTransform(residual_stream_matrics[(i+1)%len(residual_stream_matrics)][28:35].copy(), results_matrix[i*7:(i+1)*7]))
            self.add(results_and_labels[i+1])
            self.wait()
            for j in range(1, 9):
                all_forward_passes[i%len(all_forward_passes)][j].set_opacity(0.0)

        self.add(results_matrix); self.remove(t)

        #Ok so let's do a nice side by side with this matrix and map coordinate -> maybe move back to plane/2d deal?

        self.play(m.FadeOut(box), m.FadeOut(residual_stream_matrics[0]), m.FadeOut(random_background_stuff[:8]),
                  m.FadeOut(we_connections), m.FadeOut(all_weights[:11]), m.FadeOut(input_layer_empty), m.FadeOut(all_activations_empty[:16]),
                  self.animate_view(52, 57, 0, (5.8, 1.76, -3.51), 10.46), run_time=5.0)
        self.wait()
        #Ok, let's do the comparison to map coordinates in illustrator, I think that will be quicker/easier. 

        # Ok ok ok so final thing here is going to be mapping each row to a scatter point on the cool plot!!
        # Oh yeah and then adding some text to that plot, right. That might be a pretty reasonable thing to do in illustrator
        # Ok, so next i need to finalize the umap plot in matplotlib, export to svg, and import into manim! Option to clean it up 
        # in illustrator if i need to. 

        # Hmm feels like i need to get back to flat here?
        #Ok there's going to be annoying jump here i"ll have to handle in premiere
        results_and_labels.rotate(-57 * m.DEGREES, axis=local_right)
        local_right = np.array([np.cos(52 * m.DEGREES), np.sin(52 * m.DEGREES), 0])
        results_and_labels.rotate(-52 * m.DEGREES, axis=m.OUT)    # theta rotation
        results_and_labels.move_to(m.ORIGIN)
        self.set_view(0, 0, 0, (2.22, -0.01, 0.0), 5.55)


        plot_axes=m.SVGMobject(svg_path+'/p51/p51-03.svg')[1:]
        scatter_pts=m.SVGMobject(svg_path+'/p51/p51-04.svg')[1:]
        plot_labels=m.SVGMobject(svg_path+'/p51/p51-05.svg')[1:]
        all_plot_elements=m.VGroup(plot_axes, scatter_pts, plot_labels)
        all_plot_elements.scale(2.5)

        scatter_pts.set_opacity(1.0) #Eh?

        all_plot_elements.move_to([4.8, 0, 0])

        self.play(m.FadeIn(plot_axes), m.FadeIn(plot_labels))
        self.wait()

        self.play(m.ReplacementTransform(results_matrix[:7],scatter_pts[68]),
                  m.ReplacementTransform(results_matrix[7:14],scatter_pts[151]),
                  m.ReplacementTransform(results_matrix[14:21],scatter_pts[650]),
                  m.ReplacementTransform(results_matrix[21:28],scatter_pts[700]),
                  m.ReplacementTransform(results_matrix[28:35],scatter_pts[400]),
                  m.ReplacementTransform(results_matrix[35:42],scatter_pts[450]),
                  m.FadeOut(results_matrix[42:]),
                  m.FadeOut(results_and_labels[1:]),
                  self.animate_view(0, 0, 0, (4.56, -0.02, 0.0), 4.10),
                   run_time=5)

        excluded = {68, 151, 650, 700, 400, 450}
        # numbers = [i for i in range(750) if i not in excluded]
        remaining_points=m.VGroup(*[scatter_pts[i] for i in range(750) if i not in excluded])

        self.play(m.FadeIn(remaining_points), run_time=2) #I think all in one move fade other stuff and move camera.
        self.wait()

        #Ok now just labels right? yeah and I think we do those in premiere!

        #Drop labels & legend,for illustrator overlay
        self.play(m.FadeOut(plot_labels), run_time=2)
        self.wait()










        self.wait(20)
        pass

