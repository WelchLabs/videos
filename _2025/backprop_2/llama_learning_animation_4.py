from welch_paths import welch_path
import manimgx as m
import pickle
from welch_scene import WelchScene
from welch_text import math_tex
from functools import partial
import numpy as np
import torch
import sys
sys.path.append('_2025/backprop_2')
# from network_pranav_pr_1 import *
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'
GREEN='#00a14b'

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



class LlamaLearningSketchOne(WelchScene):
    def construct(self):

        pickle_path=welch_path('backprop2/hackin/jun_2_1/snapshot_2.p')
        with open(pickle_path, 'rb') as f:
            snapshot = pickle.load(f)

        mlps=[]
        attns=[]
        start_x=-4.0
        for layer_count, layer_num in enumerate([0, 1, 2, 13, 14, 15]):

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

            mlp=get_mlp(w1, w2, neuron_fills, grads_1=grads_1, grads_2=grads_2)
            mlp.move_to([start_x+0.8+layer_count*1.6, 0, 0])
            mlps.append(mlp)

            connections_right, connections_right_grads=get_mlp_connections_right(attention_connections_right=attention_connections_right, 
                                                                               mlp_in=mlp[2],
                                                                               connection_points_right=attn[3],
                                                                               attention_connections_right_grad=attention_connections_right_grad)


            if len(mlps)>1:
                connections_left, connections_left_grads=get_mlp_connections_left(attention_connections_left=attention_connections_left, 
                                                                              mlp_out=mlps[-2][4],
                                                                              connection_points_left=attn[2],
                                                                              attention_connections_left_grad=attention_connections_left_grad)
                self.add(connections_left)
                self.add(connections_left_grads)

            self.add(connections_right)
            self.add(connections_right_grads)
            self.add(attn)
            self.add(mlp)

            if len(mlps)>1:
                self.remove(mlps[-2][3][1]); self.add(mlps[-2][3][1]) #Ok seems like I'm just exploiting a bug, but this fixes layering. 
                self.remove(mlps[-2][2][1]); self.add(mlps[-2][2][1])
                # self.remove(mlps[-2][4][1]); self.add(mlps[-2][4][1])
                self.remove(mlps[-2][4]); self.add(mlps[-2][4]) #Don't ask me how I did it, I just did it, it as confusing.

        self.remove(mlps[-1][3][1]); self.add(mlps[-1][3][1]) #Ok seems like I'm just exploiting a bug, but this fixes layering. 
        self.remove(mlps[-1][2][1]); self.add(mlps[-1][2][1])
        self.remove(mlps[-1][4]); self.add(mlps[-1][4]) 

        # Ok now inputs/prompt
        # Maybe active input circles are fully colored in and everything else is black? That could look cool I think. 


        input_layer_nuerons=m.VGroup()
        input_layer_text=m.VGroup()
        num_input_neurons=36
        vertical_spacing = 0.18
        neuron_radius = 0.06
        neuron_stroke_color='#dfd0b9'
        neuron_stroke_width= 1.0

        np.random.seed(25) #Need to figure out how to add variety withotu moving the same token like "The" around
        prompt_neuron_indices=np.random.choice(np.arange(36), len(snapshot['prompt.tokens'])-1) #Don't include last token
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
        input_layer.move_to([-5.2, 0, 0])


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


        self.add(we_connections_grad)
        self.add(we_connections)
        self.add(input_layer)
        self.wait()
        
        self.remove(input_layer[0])
        self.add(input_layer[0])


        #Ok I should probably go ahead and wrap up input stuff but I don't really want to -> 
        #'topk.indices', 'topk.tokens', 'topk.probs', 'topk.unembed.W_U', 'topk.unembed.W_U.grad'

        output_layer_nuerons=m.VGroup()
        output_layer_text=m.VGroup()
        num_output_neurons=36   

        neuron_count=0
        for i in range(num_output_neurons):
            if i == num_output_neurons//2:  # Middle position for ellipsis
                dot = math_tex("...").rotate(m.PI/2, m.OUT).scale(0.4).move_to(m.UP * ((num_input_neurons//2 - i) * vertical_spacing))
                dot.set_color(neuron_stroke_color)
            else:
                n = m.Circle(radius=neuron_radius, stroke_color=neuron_stroke_color)
                n.set_stroke(width=neuron_stroke_width)
                n.set_fill(color=get_nueron_color(snapshot['topk.probs'][neuron_count],vmax=np.max(snapshot['topk.probs'])), opacity=1.0)


                if neuron_count<4: font_size=22
                else: font_size=12 
                t=m.Text(snapshot['topk.tokens'][neuron_count], font_size=font_size, font='myriad-pro')
                t.set_color(neuron_stroke_color)
                t.set_opacity(np.clip(snapshot['topk.probs'][neuron_count], 0.3, 1.0))
                t.move_to((0.2+t.get_right()[0])*m.RIGHT+ m.UP* ((-t.get_bottom()+num_input_neurons//2 - i) * vertical_spacing))
                output_layer_text.add(t)

                #I like the idea of having probs on here, but I think it's too much right now, mayb in part 3
                # if neuron_count<5:
                #     t2=Text(f"{snapshot['topk.probs'][neuron_count]:.4f}", font_size=12)
                #     t2.set_color(neuron_stroke_color)
                #     t2.set_opacity(np.clip(snapshot['topk.probs'][neuron_count], 0.4, 0.7))
                #     t2.move_to(t.get_right()+np.array([0.2, 0, 0]))
                #     output_layer_text.add(t2)

                n.move_to(m.UP * ((num_input_neurons//2 - i) * vertical_spacing))
                output_layer_nuerons.add(n)
                neuron_count+=1


        output_layer=m.VGroup(output_layer_nuerons, dot, output_layer_text)
        output_layer.move_to([5.5, 0, 0], aligned_edge=m.LEFT)

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
        unembed_scaled_grad=unembed_grad_abs/np.percentile(unembed_grad_abs, 98)
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
                line.set_stroke(opacity=np.clip(unembed_scaled_grad[i,j], 0, 1), width=np.clip(1.0*unembed_scaled_grad[i,j],0,3))
                line.set_color(get_grad_color(unembed_scaled_grad[i,j]))
                wu_connections_grad.add(line)



        self.add(wu_connections)
        self.add(wu_connections_grad)
        self.add(output_layer)
        self.wait()

        self.remove(output_layer[0][1])
        self.add(output_layer[0][1])



        self.wait()





       #  vertical_spacing = 0.18
       #  neuron_radius = 0.06
       #  neuron_stroke_color='#dfd0b9'
       #  neuron_stroke_width= 1.0

       #  np.random.seed(25) #Need to figure out how to add variety withotu moving the same token like "The" around
       #  prompt_neuron_indices=np.random.choice(np.arange(36), len(snapshot['prompt.tokens'])-1) #Don't include last token
       #  words_to_nudge={' capital':-0.02}

       #  prompt_token_count=0
       #  neuron_count=0
       #  for i in range(num_input_neurons):
       #      if i == num_input_neurons//2:  # Middle position for ellipsis
       #          dot = Tex("...").rotate(PI/2, OUT).scale(0.4).move_to(UP * ((num_input_neurons//2 - i) * vertical_spacing))
       #          dot.set_color(neuron_stroke_color)
       #      else:
       #          neuron = Circle(radius=neuron_radius, stroke_color=neuron_stroke_color)
       #          neuron.set_stroke(width=neuron_stroke_width)
       #          if neuron_count in prompt_neuron_indices:
       #              neuron.set_fill(color='#dfd0b9', opacity=1.0)
       #              t=Text(snapshot['prompt.tokens'][prompt_token_count], font_size=24, font='myriad-pro')
       #              t.set_color(neuron_stroke_color)
       #              # print(t.get_center())
       #              t.move_to((0.2+t.get_right()[0])*LEFT+UP * ((-t.get_bottom()+num_input_neurons//2 - i) * vertical_spacing))
       #              if snapshot['prompt.tokens'][prompt_token_count] in words_to_nudge.keys():
       #                  t.shift([0, words_to_nudge[snapshot['prompt.tokens'][prompt_token_count]], 0])

       #              input_layer_text.add(t)
       #              prompt_token_count+=1 

       #          neuron.move_to(UP * ((num_input_neurons//2 - i) * vertical_spacing))
       #          input_layer_nuerons.add(neuron)
       #          neuron_count+=1

       #  input_layer=VGroup(input_layer_nuerons, dot, input_layer_text)
       #  input_layer.move_to([-5.2, 0, 0])




        self.wait()







        self.wait()
        pass



        # self.remove(mlps[1][4])
        # self.add(mlps[1][4])













































