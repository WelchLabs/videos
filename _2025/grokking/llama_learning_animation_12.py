from manimlib import *
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
#         return Color(rgb=rgba[:3])

def get_nueron_color(value, vmax=0.95):        
    '''Uses abs, a little reductive'''
    value_clipped = np.clip(np.abs(value)/vmax, 0, 1)
    rgba = custom_cmap_tans(value_clipped) #Would also like to try a monochrome tan option
    return Color(rgb=rgba[:3])

def get_grad_color(value): #, vmin=-2, vmax=2):        
    # value_clipped = np.clip((value - vmin) / (vmax - vmin), 0, 1)
    value_clipped = np.clip(np.abs(value), 0, 1)
    rgba = custom_cmap_cyan(value_clipped) #Would also like to try a monochrome tan option
    return Color(rgb=rgba[:3])

class AttentionPattern(VMobject):
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
        color = Color(rgb=rgba[:3])
        # opacity = self.min_opacity + val_clipped * (self.max_opacity - self.min_opacity)
        # opacity=val_scaled
        opacity=1.0
        return {"color": color, "opacity": opacity}

    def build(self):
        self.clear()
        squares = VGroup()
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                val = self.matrix[i, j]
                style = self.map_value_to_style(val)

                square = Square(side_length=self.square_size)
                square.set_fill(style["color"], opacity=style["opacity"])
                square.set_stroke(self.stroke_color, width=self.stroke_width)

                pos = RIGHT * j * self.square_size + DOWN * i * self.square_size
                square.move_to(pos)
                squares.add(square)

        squares.move_to(ORIGIN)
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
    input_layer = VGroup()
    hidden_layer = VGroup()
    output_layer = VGroup()
    dots = VGroup()
    
    # Input layer
    neuron_count=0
    for i in range(INPUT_NEURONS):
        if i == w1.shape[0]//2:  # Middle position for ellipsis
            dot = Tex("...").rotate(PI/2, OUT).scale(DOTS_SCALE).move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            dot.set_color(neuron_stroke_color)
            dots.add(dot)
        else:
            neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
            neuron.set_stroke(width=neuron_stroke_width)
            if neuron_fills is None: 
                neuron.set_fill(color='#000000', opacity=1.0)
            else: 
                neuron.set_fill(color=get_nueron_color(neuron_fills[0][neuron_count], vmax=np.abs(neuron_fills[0]).max()), opacity=1.0)
            neuron.move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            input_layer.add(neuron)
            neuron_count+=1
            
    # Hidden layer
    neuron_count=0
    for i in range(HIDDEN_NEURONS):
        if i == w1.shape[1]//2:  # Middle position for ellipsis
            dot = Tex("...").rotate(PI/2, OUT).scale(DOTS_SCALE).move_to(UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
            dot.set_color(neuron_stroke_color)
            dots.add(dot)
        else:
            neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
            neuron.set_stroke(width=neuron_stroke_width)
            if neuron_fills is None: 
                neuron.set_fill(color='#000000', opacity=1.0)
            else: 
                neuron.set_fill(color=get_nueron_color(neuron_fills[1][neuron_count], vmax=np.abs(neuron_fills[1]).max()), opacity=1.0)
            neuron.move_to(UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
            hidden_layer.add(neuron)
            neuron_count+=1
            
    # Output layer
    neuron_count=0
    for i in range(OUTPUT_NEURONS):
        if i == w1.shape[0]//2:  # Middle position for ellipsis
            dot = Tex("...").rotate(PI/2, OUT).scale(DOTS_SCALE).move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            dot.set_color(neuron_stroke_color)
            dots.add(dot)
        else:
            neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
            neuron.set_stroke(width=neuron_stroke_width)
            if neuron_fills is None: 
                neuron.set_fill(color='#000000', opacity=1.0)
            else: 
                neuron.set_fill(color=get_nueron_color(neuron_fills[2][neuron_count], vmax=np.abs(neuron_fills[2]).max()), opacity=1.0)
            neuron.move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            output_layer.add(neuron)
            neuron_count+=1
            
    # Create connections with edge points
    connections = VGroup()
    w1_abs=np.abs(w1)
    w1_scaled=w1_abs/np.percentile(w1_abs, 99)
    # w1_scaled=(w1-w1.min())/(w1.max()-w1.min())
    for i, in_neuron in enumerate(input_layer):
        for j, hidden_neuron in enumerate(hidden_layer):
            if np.abs(w1_scaled[i, j])<0.75: continue
            if abs(i-j)>6: continue #Let's try just drawing local ones. 
            start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
            line = Line(start_point, end_point)

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
            line = Line(start_point, end_point) #, stroke_opacity=line_opacity, stroke_width=line_weight)
            # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-connection_display_thresh), 0.1, 1), width=line_weight)
            line.set_stroke(opacity=np.clip(w2_scaled[i,j], 0, 1), width=1.0*w2_scaled[i,j])
            line.set_color(line_stroke_color)
            connections.add(line)

    grad_conections=VGroup()
    if grads_1 is not None:
        grads_1_abs=np.abs(grads_1)
        grads_1_scaled=grads_1_abs/np.percentile(grads_1_abs, 95)
        for i, in_neuron in enumerate(input_layer):
            for j, hidden_neuron in enumerate(hidden_layer):
                if np.abs(grads_1_scaled[i, j])<0.5: continue
                if abs(i-j)>6: continue #Need to dial this up or lost it probably, but it is helpful!
                start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
                line_grad = Line(start_point, end_point)
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
                line_grad = Line(start_point, end_point)
                # line_grad.set_stroke(opacity=np.clip(0.8*(np.abs(grads_2[i, j])-grad_display_thresh), 0, 1), 
                #                     width=np.abs(grads_2[i, j]))
                # line_grad.set_stroke(opacity=0.8, width=2)
                line_grad.set_stroke(opacity=np.clip(grads_2_scaled[i,j], 0, 1), width=np.clip(1.0*grads_2_scaled[i,j], 0, 3))
                # print(np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1))
                line_grad.set_color(get_grad_color(grads_2_scaled[i, j]))
                grad_conections.add(line_grad)

                
    return VGroup(connections, grad_conections, input_layer, hidden_layer, output_layer, dots)


def get_attention_layer(attn_patterns):
    num_attention_pattern_slots=len(attn_patterns)+1
    attention_pattern_spacing=0.51

    attention_border=RoundedRectangle(width=0.59, height=5.4, corner_radius=0.1)
    attention_border.set_stroke(width=1.0, color=CHILL_BROWN)


    attention_patterns=VGroup()
    connection_points_left=VGroup()
    connection_points_right=VGroup()

    attn_pattern_count=0
    for i in range(num_attention_pattern_slots):
        if i==num_attention_pattern_slots//2:
            dot = Tex("...").rotate(PI/2, OUT).scale(0.5).move_to([0, num_attention_pattern_slots*attention_pattern_spacing/2 - attention_pattern_spacing*(i+0.5), 0])
            dot.set_color(CHILL_BROWN)
            attention_patterns.add(dot) #Just add here?
        else:
            if i>num_attention_pattern_slots//2: offset=0.15
            else: offset=-0.15 
            # matrix = np.random.rand(6, 6)
            attn_pattern = AttentionPattern(matrix=attn_patterns[attn_pattern_count], square_size=0.08, stroke_width=0.5) #square_size=0.07
            attn_pattern.move_to([0, num_attention_pattern_slots*attention_pattern_spacing/2+offset - attention_pattern_spacing*(i+0.5), 0])
            attention_patterns.add(attn_pattern)

            connection_point_left=Circle(radius=0)
            connection_point_left.move_to([-0.59/2.0, num_attention_pattern_slots*attention_pattern_spacing/2+offset - attention_pattern_spacing*(i+0.5), 0])
            connection_points_left.add(connection_point_left)

            connection_point_right=Circle(radius=0)
            connection_point_right.move_to([0.59/2.0, num_attention_pattern_slots*attention_pattern_spacing/2+offset - attention_pattern_spacing*(i+0.5), 0])
            connection_points_right.add(connection_point_right)
            attn_pattern_count+=1

    attention_layer=VGroup(attention_patterns, attention_border, connection_points_left, connection_points_right)
    return attention_layer

def get_mlp_connections_left(attention_connections_left, mlp_out, connection_points_left, attention_connections_left_grad=None):
    connections_left=VGroup()
    attention_connections_left_abs=np.abs(attention_connections_left)
    attention_connections_left_scaled=attention_connections_left_abs/np.max(attention_connections_left_abs) #np.percentile(attention_connections_left_abs, 99)
    for i, mlp_out_neuron in enumerate(mlp_out):
        for j, attention_neuron in enumerate(connection_points_left):
            if np.abs(attention_connections_left_scaled[i, j])<0.5: continue
            if abs(i/4-j)>3: continue #Need to dial this up or lost it probably, but it is helpful!
            start_point, end_point = get_edge_points(mlp_out_neuron, attention_neuron, 0.06)
            line = Line(start_point, attention_neuron.get_center())
            # line.set_stroke(width=1, opacity=0.3)
            # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
            line.set_stroke(opacity=np.clip(attention_connections_left_scaled[i,j], 0, 1), width=np.clip(1.0*attention_connections_left_scaled[i,j],0,3))
            line.set_color(CHILL_BROWN)
            connections_left.add(line)


    connections_left_grads=VGroup()
    if attention_connections_left_grad is not None: 
        attention_connections_left_grad_abs=np.abs(attention_connections_left_grad)
        attention_connections_left_grad_scaled=attention_connections_left_grad_abs/np.percentile(attention_connections_left_grad_abs, 98) #np.percentile(attention_connections_left_abs, 99)
        for i, mlp_out_neuron in enumerate(mlp_out):
            for j, attention_neuron in enumerate(connection_points_left):
                if np.abs(attention_connections_left_grad_scaled[i, j])<0.5: continue
                if abs(i/4-j)>3: continue #Need to dial this up or lost it probably, but it is helpful!
                start_point, end_point = get_edge_points(mlp_out_neuron, attention_neuron, 0.06)
                line = Line(start_point, attention_neuron.get_center())
                # line.set_stroke(width=1, opacity=0.3)
                # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
                line.set_stroke(opacity=np.clip(attention_connections_left_grad_scaled[i,j], 0, 1), width=np.clip(1.0*attention_connections_left_grad_scaled[i,j],0,2))
                line.set_color(get_grad_color(attention_connections_left_grad_scaled[i,j]))
                connections_left_grads.add(line)
    return connections_left, connections_left_grads



def get_mlp_connections_right(attention_connections_right, mlp_in, connection_points_right, attention_connections_right_grad=None):
    connections_right=VGroup()
    attention_connections_right_abs=np.abs(attention_connections_right)
    attention_connections_right_scaled=attention_connections_right_abs/np.percentile(attention_connections_right_abs, 99)
    for i, attention_neuron in enumerate(connection_points_right):
        for j, mlp_in_neuron in enumerate(mlp_in):
            if np.abs(attention_connections_right_scaled[i, j])<0.6: continue
            if abs(j/4-i)>3: continue #Need to dial this up or lost it probably, but it is helpful!
            start_point, end_point = get_edge_points(mlp_in_neuron, attention_neuron, 0.06)
            line = Line(start_point, attention_neuron.get_center())
            # line.set_stroke(width=1, opacity=0.3)
            # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
            line.set_stroke(opacity=np.clip(attention_connections_right_scaled[i,j], 0, 1), width=np.clip(1.0*attention_connections_right_scaled[i,j],0,3))
            line.set_color(CHILL_BROWN)
            connections_right.add(line)

    connections_right_grads=VGroup()
    if attention_connections_right_grad is not None: 
        attention_connections_right_grad_abs=np.abs(attention_connections_right_grad)
        attention_connections_right_grad_scaled=attention_connections_right_grad_abs/np.percentile(attention_connections_right_grad_abs, 98)
        for i, attention_neuron in enumerate(connection_points_right):
            for j, mlp_in_neuron in enumerate(mlp_in):
                if np.abs(attention_connections_right_grad_scaled[i, j])<0.5: continue
                if abs(j/4-i)>3: continue #Need to dial this up or lost it probably, but it is helpful!
                start_point, end_point = get_edge_points(mlp_in_neuron, attention_neuron, 0.06)
                line = Line(start_point, attention_neuron.get_center())
                # line.set_stroke(width=1, opacity=0.3)
                # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
                line.set_stroke(opacity=np.clip(attention_connections_right_grad_scaled[i,j], 0, 1), width=np.clip(1.0*attention_connections_right_grad_scaled[i,j],0,3))
                line.set_color(get_grad_color(attention_connections_right_grad_scaled[i,j]))
                connections_right_grads.add(line)
    return connections_right, connections_right_grads

def get_input_layer(prompt_neuron_indices, snapshot, num_input_neurons=36):
    input_layer_nuerons=VGroup()
    input_layer_text=VGroup()
    vertical_spacing = 0.18
    neuron_radius = 0.06
    neuron_stroke_color='#dfd0b9'
    neuron_stroke_width= 1.0
    words_to_nudge={' capital':-0.02}

    prompt_token_count=0
    neuron_count=0
    for i in range(num_input_neurons):
        if i == num_input_neurons//2:  # Middle position for ellipsis
            dot = Tex("...").rotate(PI/2, OUT).scale(0.4).move_to(UP * ((num_input_neurons//2 - i) * vertical_spacing))
            dot.set_color(neuron_stroke_color)
        else:
            neuron = Circle(radius=neuron_radius, stroke_color=neuron_stroke_color)
            neuron.set_stroke(width=neuron_stroke_width)
            if neuron_count in prompt_neuron_indices:
                neuron.set_fill(color='#dfd0b9', opacity=1.0)
                t=Text(snapshot['prompt.tokens'][prompt_token_count], font_size=24, font='myriad-pro')
                t.set_color(neuron_stroke_color)
                # print(t.get_center())
                t.move_to((0.2+t.get_right()[0])*LEFT+UP * ((-t.get_bottom()+num_input_neurons//2 - i) * vertical_spacing))
                if snapshot['prompt.tokens'][prompt_token_count] in words_to_nudge.keys():
                    t.shift([0, words_to_nudge[snapshot['prompt.tokens'][prompt_token_count]], 0])

                input_layer_text.add(t)
                prompt_token_count+=1 
            else:
                neuron.set_fill(color='#000000', opacity=1.0)

            neuron.move_to(UP * ((num_input_neurons//2 - i) * vertical_spacing))
            input_layer_nuerons.add(neuron)
            neuron_count+=1

    input_layer=VGroup(input_layer_nuerons, dot, input_layer_text)
    return input_layer


def get_output_layer(snapshot, empty=False):
    output_layer_nuerons=VGroup()
    output_layer_text=VGroup()
    num_output_neurons=36   
    vertical_spacing = 0.18
    neuron_radius = 0.06
    neuron_stroke_color='#dfd0b9'
    neuron_stroke_width= 1.0

    neuron_count=0
    for i in range(num_output_neurons):
        if i == num_output_neurons//2:  # Middle position for ellipsis
            dot = Tex("...").rotate(PI/2, OUT).scale(0.4).move_to(UP * ((num_output_neurons//2 - i) * vertical_spacing))
            dot.set_color(neuron_stroke_color)
        else:
            n = Circle(radius=neuron_radius, stroke_color=neuron_stroke_color)
            n.set_stroke(width=neuron_stroke_width)
            if not empty: 
                n.set_fill(color=get_nueron_color(snapshot['topk.probs'][neuron_count],vmax=np.max(snapshot['topk.probs'])), opacity=1.0)
                if neuron_count==0: font_size=22
                elif neuron_count<4: font_size=16
                else: font_size=12 
                t=Text(snapshot['topk.tokens'][neuron_count], font_size=font_size, font='myriad-pro')
                # t.set_color(neuron_stroke_color)
                # get_nueron_color(neuron_fills[0][neuron_count], vmax=np.abs(neuron_fills[0]).max())
                text_color=get_nueron_color(np.clip(snapshot['topk.probs'][neuron_count],0.1, 1.0),vmax=np.max(snapshot['topk.probs'])),
                t.set_color(text_color)
                t.set_opacity(np.clip(snapshot['topk.probs'][neuron_count], 0.3, 1.0))
                t.move_to((0.2+t.get_right()[0])*RIGHT+ UP* ((-t.get_bottom()+num_output_neurons//2 - i) * vertical_spacing))
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

            n.move_to(UP * ((num_output_neurons//2 - i) * vertical_spacing))
            output_layer_nuerons.add(n)
            neuron_count+=1
    output_layer=VGroup(output_layer_nuerons, dot, output_layer_text)
    return output_layer


class LlamaLearningTwelveF(InteractiveScene):
    def construct(self):
        '''
        Getting close! Next hurdle is to bring in different examples. 
        '''


        pickle_path='/Users/stephen/Library/CloudStorage/Dropbox-Stephencwelch/welch_labs/ai_book_vol_2/2_grokking/hackin/jun_3_1'
        snapshots=[]
        for p in sorted(glob.glob(pickle_path+'/*.p')):
            with open(p, 'rb') as f:
                snapshots.append(pickle.load(f))

        #Ok i should maybe refactor - but not sure if it's worth it - I need to capture these things from each snapshot I look at: 
        # backward_pass=VGroup(we_connections_grad, *all_grads, wu_connections_grad)
        #forward_pass=VGroup(input_layer[:-1], *all_activations, output_layer[:-1]) 
        all_backward_passes=[]
        all_forward_passes=[]

        # random_seeds=[25, 26, 27, 28, 29, 30, 31, 32, 33, 34] #For ordering input neurons
        random_seeds=[25, 25, 25, 25, 25, 25, 25, 25, 25, 25]
        for snapshot_count, snapshot_index in enumerate([0,0,0]):
            snapshot=snapshots[snapshot_index]

            all_weights=VGroup()
            all_activations=VGroup()
            all_activations_empty=VGroup()
            all_grads=VGroup()
            random_background_stuff=VGroup()


            mlps=[]
            attns=[]
            start_x=-4.0
            # for layer_count, layer_num in enumerate([0, 1, 2, 3, 12, 13, 14, 15]):
            # for layer_count, layer_num in enumerate([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]):
            # for layer_count, layer_num in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]):
            # for layer_count, layer_num in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 14, 15]):
            for layer_count, layer_num in enumerate([0, 1, 6, 7, 8, 9, 14, 15]):

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
                for i in range(1, 31, 3): #Just take every thrid pattern for now. ### for i in range(0, 30, 3):
                    attn_patterns.append(all_attn_patterns[0][i][1:-1,1:-1]) #Ignore BOS token and final "answer" token
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

                attn_empty=get_attention_layer([np.zeros_like(all_attn_patterns[0][0][1:-1,1:-1]) for i in range(len(attn_patterns))])
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
            input_layer.move_to([-4.7, 0, 0], aligned_edge=RIGHT)

            input_layer_empty = get_input_layer([], snapshot, num_input_neurons=num_input_neurons)
            input_layer_empty.move_to([-4.7, 0, 0], aligned_edge=RIGHT)


            # Okie dokie -> Let me add intput/first attention layer connections - this will need to be a separate function
            # Ok right and I need to bring two matrices together here -> hmm. 
            
            all_embeddings=[]; all_embeddings_grad=[]; prompt_token_embeddings=[]; prompt_token_embeddings_grad=[]
            for i in range(1, 31, 3):
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

            we_connections=VGroup()
            all_embeddings_abs=np.abs(all_embeddings)
            all_embeddings_scaled=all_embeddings_abs/np.percentile(all_embeddings_abs, 95)
            for i, n1 in enumerate(input_layer[0]):
                for j, n2 in enumerate(attns[0][2]):
                    # if np.abs(all_embeddings_scaled[i, j])<0.1: continue
                    if abs(j-i/4)>3: continue #Need to dial this up or lost it probably, but it is helpful!
                    start_point, end_point = get_edge_points(n1, n2, 0.06)
                    line = Line(start_point, n2.get_center())
                    # line.set_stroke(width=1, opacity=0.3)
                    # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
                    line.set_stroke(opacity=np.clip(all_embeddings_scaled[i,j], 0.4, 1), width=np.clip(1.0*all_embeddings_scaled[i,j],0.5,1.7))
                    line.set_color(CHILL_BROWN)
                    we_connections.add(line)


            we_connections_grad=VGroup()
            all_embeddings_grad_abs=np.abs(all_embeddings_grad)
            all_embeddings_grad_scaled=all_embeddings_grad_abs/np.percentile(all_embeddings_grad_abs, 95)
            for i, n1 in enumerate(input_layer[0]):
                for j, n2 in enumerate(attns[0][2]):
                    # if np.abs(all_embeddings_grad_scaled[i, j])<0.1: continue
                    if abs(j-i/4)>4: continue #Need to dial this up or lost it probably, but it is helpful!
                    start_point, end_point = get_edge_points(n1, n2, 0.06)
                    line = Line(start_point, n2.get_center())
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
            output_layer.move_to([mlps[-1].get_right()[0]+0.36, -3.21, 0], aligned_edge=LEFT+BOTTOM) #Was 5.45

            output_layer_empty=get_output_layer(snapshot, empty=True)
            output_layer_empty.move_to([mlps[-1].get_right()[0]+0.36, -3.21, 0], aligned_edge=LEFT+BOTTOM)


            wu_connections=VGroup()
            unembed_abs=np.abs(snapshot['topk.unembed.W_U'][:,0,:].T)
            unembed_scaled=unembed_abs/np.percentile(unembed_abs, 98)
            for i, n1 in enumerate(mlps[-1][4]):
                for j, n2 in enumerate(output_layer[0]):
                    if np.abs(unembed_scaled[i, j])<0.5: continue
                    if abs(j-i)>8: continue #Need to dial this up or lost it probably, but it is helpful!
                    start_point, end_point = get_edge_points(n1, n2, 0.06)
                    line = Line(start_point, n2.get_center())
                    # line.set_stroke(width=1, opacity=0.3)
                    # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
                    line.set_stroke(opacity=np.clip(unembed_scaled[i,j], 0.4, 1), width=np.clip(1.0*unembed_scaled[i,j],0.5,1.7))
                    line.set_color(CHILL_BROWN)
                    wu_connections.add(line)

            wu_connections_grad=VGroup()
            unembed_grad_abs=np.abs(snapshot['topk.unembed.W_U.grad'][:,0,:].T)
            unembed_scaled_grad=unembed_grad_abs/np.percentile(unembed_grad_abs, 99)
            for i, n1 in enumerate(mlps[-1][4]):
                for j, n2 in enumerate(output_layer[0]):
                    if np.abs(unembed_scaled_grad[i, j])<0.5: continue
                    if abs(j-i)>8: continue #Need to dial this up or lost it probably, but it is helpful!
                    start_point, end_point = get_edge_points(n1, n2, 0.06)
                    line = Line(start_point, n2.get_center())
                    # line.set_stroke(width=1, opacity=0.3)
                    # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
                    # line.set_stroke(opacity=np.clip(unembed_scaled_grad[i,j], 0.4, 1), width=np.clip(1.0*unembed_scaled_grad[i,j],0.5,1.7))
                    # line.set_color(CHILL_BROWN)
                    line.set_stroke(opacity=np.clip(unembed_scaled_grad[i,j], 0, 1), width=np.clip(0.7*unembed_scaled_grad[i,j],0,3))
                    line.set_color(get_grad_color(unembed_scaled_grad[i,j]))
                    wu_connections_grad.add(line)

            all_backward_passes.append(VGroup(we_connections_grad, *all_grads, wu_connections_grad))
            # all_forward_passes.append(VGroup(input_layer[:-1], *all_activations, output_layer[:-1])) #Leaving out input output text for now. 
            all_forward_passes.append(VGroup(input_layer, *all_activations, output_layer))

        ## -- end big ole loop
        self.wait()

        # self.frame.reorient(0, 0, 0, (2.04, -0.57, 0.0), 8.63)
        # self.frame.reorient(0, 0, 0, (1.99, -0.04, 0.0), 8.32)
        self.frame.reorient(0, 0, 0, (2.06, -0.06, 0.0), 9.36)
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

        all_forward_passes[1].set_opacity(1.0)
        self.wait(1.0)
        all_backward_passes[1].set_opacity(1.0)
        self.wait(1.0)
        all_forward_passes[1].set_opacity(0.0)
        all_backward_passes[1].set_opacity(0)
        self.wait()
        # all_forward_passes[1].set_opacity(1.0)
        # all_forward_passes[2].set_opacity(1.0)
        # all_forward_passes[0].set_opacity(0.0)
        # all_forward_passes[1].set_opacity(0.0)
        # all_forward_passes[2].set_opacity(0.0)

        # all_backward_passes[0].set_opacity(1.0)
        # all_backward_passes[1].set_opacity(1.0)
        # all_backward_passes[0].set_opacity(0.0)
        # all_backward_passes[1].set_opacity(0.0)

        #Ok nothing has broken so far lol. 

        def create_multi_snapshot_animation(self, all_forward_passes, all_backward_passes, 
                                           individual_time=1.0, lag_ratio=0.5, 
                                           start_opacity=0.0, end_opacity=1.0,
                                           fade_out_time=1.0, pause_between_snapshots=0.5):
            """
            Animates multiple snapshots with forward and backward passes.
            Each snapshot fades in, plays, then fades out before the next one starts.
            
            Args:
                self: The scene object
                all_forward_passes: List of forward pass VGroups for each snapshot
                all_backward_passes: List of backward pass VGroups for each snapshot
                individual_time: Time for each object to fully animate
                lag_ratio: Overlap ratio between consecutive objects
                start_opacity: Starting opacity for fade-in
                end_opacity: Maximum opacity
                fade_out_time: Time to fade out each snapshot
                pause_between_snapshots: Pause time between snapshots
            """
            
            def create_snapshot_animation(forward_pass, backward_pass, time_offset=0, is_final=False):
                """Creates animation for a single snapshot"""
                objects = list(forward_pass) + list(backward_pass[::-1])
                num_objects = len(objects)
                
                if num_objects == 0:
                    return None, 0
                
                lag_time = lag_ratio * individual_time
                animation_time = individual_time + (num_objects - 1) * lag_time
                
                def update_opacity(obj, dt, index, start_time):
                    current_time = time_tracker.get_value()
                    
                    # Check if we're in the animation phase for this snapshot
                    if current_time < start_time:
                        obj.set_opacity(start_opacity)
                        return
                    
                    snapshot_time = current_time - start_time
                    
                    # Fade in phase
                    obj_start_time = index * lag_time
                    obj_end_time = obj_start_time + individual_time
                    
                    if snapshot_time < obj_start_time:
                        obj.set_opacity(start_opacity)
                    elif snapshot_time <= obj_end_time:
                        progress = (snapshot_time - obj_start_time) / individual_time
                        opacity = start_opacity + (end_opacity - start_opacity) * progress
                        obj.set_opacity(opacity)
                    elif snapshot_time <= animation_time:
                        obj.set_opacity(end_opacity)
                    # Fade out phase (skip for final snapshot)
                    elif not is_final and snapshot_time <= animation_time + fade_out_time:
                        fade_progress = (snapshot_time - animation_time) / fade_out_time
                        opacity = end_opacity * (1 - fade_progress)
                        obj.set_opacity(opacity)
                    elif not is_final:
                        obj.set_opacity(0)
                    else:
                        # Keep final snapshot visible
                        obj.set_opacity(end_opacity)
                
                return objects, animation_time, update_opacity
            
            # Calculate total animation time
            total_time = 0
            snapshot_data = []
            
            for i, (forward, backward) in enumerate(zip(all_forward_passes, all_backward_passes)):
                is_final = (i == len(all_forward_passes) - 1)
                objects, anim_time, updater_func = create_snapshot_animation(forward, backward, total_time, is_final)
                
                if objects:
                    snapshot_data.append({
                        'objects': objects,
                        'start_time': total_time,
                        'animation_time': anim_time,
                        'updater_func': updater_func,
                        'is_final': is_final
                    })
                    
                    # Don't add fade_out_time or pause for the final snapshot
                    if not is_final:
                        total_time += anim_time + fade_out_time + pause_between_snapshots
                    else:
                        total_time += anim_time
            
            # Remove the final pause
            total_time -= pause_between_snapshots
            
            # Create the main time tracker
            time_tracker = ValueTracker(0)
            
            # Add updaters to all objects
            all_updaters = []
            for snapshot in snapshot_data:
                for idx, obj in enumerate(snapshot['objects']):
                    updater = lambda o, dt, i=idx, start=snapshot['start_time'], func=snapshot['updater_func']: func(o, dt, i, start)
                    obj.add_updater(updater)
                    all_updaters.append((obj, updater))
            
            # Play the animation
            # self.play(time_tracker.animate.set_value(total_time), run_time=total_time)
            
            # # Clean up updaters
            # for obj, updater in all_updaters:
            #     obj.remove_updater(updater)
            
            # # Ensure non-final objects are at 0 opacity and final objects stay visible
            # for snapshot in snapshot_data:
            #     for obj in snapshot['objects']:
            #         if snapshot['is_final']:
            #             obj.set_opacity(end_opacity)
            #         else:
            #             obj.set_opacity(0)

       

            return time_tracker, total_time

        #Forward and backward pass
        time_tracker, total_time = create_multi_snapshot_animation(self, all_forward_passes, all_backward_passes, 
                                                   individual_time=1.0, lag_ratio=0.5, 
                                                   start_opacity=0.0, end_opacity=1.0,
                                                   fade_out_time=7.0, pause_between_snapshots=0.5)


        self.wait()

        # Play the animation
        self.play(time_tracker.animate.set_value(total_time), run_time=total_time)
        self.wait()

        #Ok now zoom in on that one pattern!
        # self.play(self.frame.animate.reorient(0, 0, 0, (0.8, -1.89, 0.0), 1.00), run_time=7)
        # self.wait()

        # self.play(self.frame.animate.reorient(0, 0, 0, (2.06, -0.06, 0.0), 9.36), run_time=7)
        # self.wait()



        
        # # Clean up updaters
        # for obj, updater in all_updaters:
        #     obj.remove_updater(updater)
        
        # # Ensure all objects are at 0 opacity at the end
        # for snapshot in snapshot_data:
        #     for obj in snapshot['objects']:
        #         obj.set_opacity(0)










        # def create_lag_animation(objects, individual_time=1.5, lag_ratio=0.2, start_opacity=0.0, end_opacity=1.0):
        #     num_objects = len(objects)
        #     if num_objects == 0:
        #         return

        #     lag_time = lag_ratio * individual_time
        #     actual_total_time = individual_time + (num_objects - 1) * lag_time
        #     time_tracker = ValueTracker(0)
            
        #     def update_opacity(obj, dt, index):
        #         current_time = time_tracker.get_value()
        #         start_time = index * lag_time
        #         end_time = start_time + individual_time
                
        #         if current_time < start_time:
        #             obj.set_opacity(start_opacity)
        #         elif current_time > end_time:
        #             obj.set_opacity(end_opacity)
        #         else:
        #             progress = (current_time - start_time) / individual_time
        #             opacity = start_opacity + (end_opacity - start_opacity) * progress
        #             obj.set_opacity(opacity)
            
        #     for i, obj in enumerate(objects):
        #         obj.add_updater(lambda o, dt, idx=i: update_opacity(o, dt, idx))

        #     return time_tracker, actual_total_time, objects

        # # Usage:
        # self.wait()
        # # create_lag_animation(self, list(forward_pass)+list(backward_pass[::-1]), individual_time=1.0, lag_ratio=0.5, start_opacity=0.0, end_opacity=1.0)
        # # create_lag_animation(self, list(backward_pass[::-1]), individual_time=1.5, lag_ratio=0.2, start_opacity=0.0, end_opacity=1.0)
        # time_tracker, actual_total_time, objects=create_lag_animation(list(all_forward_passes[0])+list(all_backward_passes[0][::-1]), 
        #                                                     individual_time=1.0, lag_ratio=0.5, start_opacity=0.0, end_opacity=1.0)

        # self.wait()
        # self.play(time_tracker.animate.set_value(actual_total_time), run_time=actual_total_time)


        # Remove updaters
        # for obj in objects:
        #     obj.clear_updaters()

        # num_loops=5,
        # individual_time=1.0, 
        # lag_ratio=0.5, 
        # start_opacity=0.0, 
        # end_opacity=1.0,
        # fade_out_time=7.0






















        self.wait(20)
        self.embed()
