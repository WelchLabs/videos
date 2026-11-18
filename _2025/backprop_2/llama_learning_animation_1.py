from welch_paths import welch_path
import manimgx as m
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

def get_grad_color(value, vmin=-2, vmax=2):        
    value_clipped = np.clip((value - vmin) / (vmax - vmin), 0, 1)
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
            connection_display_thresh=0.4,
            grad_display_thresh=0.5):

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
                neuron.set_fill(color=get_nueron_color(neuron_fills[0][i]), opacity=1.0)
            neuron.move_to(m.LEFT * LAYER_SPACING + m.UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            input_layer.add(neuron)
            
    # Hidden layer
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
                neuron.set_fill(color=get_nueron_color(neuron_fills[1][i]), opacity=1.0)
            neuron.move_to(m.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
            hidden_layer.add(neuron)
            
    # Output layer
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
                neuron.set_fill(color=get_nueron_color(neuron_fills[2][i]), opacity=1.0)
            neuron.move_to(m.RIGHT * LAYER_SPACING + m.UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            output_layer.add(neuron)
            
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
        for i, in_neuron in enumerate(input_layer):
            for j, hidden_neuron in enumerate(hidden_layer):
                if np.abs(grads_1[i, j])<grad_display_thresh: continue
                if abs(i-j)>6: continue #Need to dial this up or lost it probably, but it is helpful!
                start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
                line_grad = m.Line(start_point, end_point)
                line_grad.set_stroke(opacity=np.clip(0.8*(np.abs(grads_1[i, j])-grad_display_thresh), 0, 1), 
                                    width=np.abs(grads_1[i, j]))
                # line_grad.set_stroke(opacity=0.8, width=2)
                # print(np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1))
                line_grad.set_color(get_grad_color(grads_1[i, j]))
                grad_conections.add(line_grad)

            
    # Connect hidden to output layer
    if grads_2 is not None:
        for i, hidden_neuron in enumerate(hidden_layer):
            for j, out_neuron in enumerate(output_layer):
                if np.abs(grads_2[i, j])<grad_display_thresh: continue
                if abs(i-j)>6: continue #Need to dial this up or lost it probably, but it is helpful!
                start_point, end_point = get_edge_points(hidden_neuron, out_neuron, NEURON_RADIUS)
                line_grad = m.Line(start_point, end_point)
                line_grad.set_stroke(opacity=np.clip(0.8*(np.abs(grads_2[i, j])-grad_display_thresh), 0, 1), 
                                    width=np.abs(grads_2[i, j]))
                # line_grad.set_stroke(opacity=0.8, width=2)
                # print(np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1))
                line_grad.set_color(get_grad_color(grads_2[i, j]))
                grad_conections.add(line_grad)

                
    return m.VGroup(connections, grad_conections, input_layer, hidden_layer, output_layer, dots)
    # return VGroup(grad_conections, input_layer, hidden_layer, output_layer, dots)


class LlamaLearningSketchOne(WelchScene):
    def construct(self):

        # w1 = np.random.randn(32, 34) 
        # w2 = np.random.randn(34, 32)  
        grads_1 = np.random.randn(32, 34) #Do to -> replace with da real nums
        grads_2 = np.random.randn(34, 32) 
        # neuron_fills=[np.random.randn(32), np.random.randn(34), np.random.randn(32)]

        data_dir=welch_path('backprop2/hackin/may_31_1')
        layer_num=8
        neuron_fills=[
            np.load(data_dir + '/blocks.'+str(layer_num)+'.hook_resid_mid.npy'),
            np.load(data_dir + '/blocks.'+str(layer_num)+'.mlp.hook_post.npy'),
            np.load(data_dir + '/blocks.'+str(layer_num)+'.hook_mlp_out.npy')
        ]


        w1=np.load(data_dir + '/blocks.'+str(layer_num)+'.mlp.W_in'+'.npy')
        w2=np.load(data_dir + '/blocks.'+str(layer_num)+'.mlp.W_out'+'.npy')

        mlp=get_mlp(w1, w2, neuron_fills) #, grads_1=grads_1, grads_2=grads_2)
        mlp.move_to([-4, 0, 0])

        #Will do this in a more modular way in next iteration:
        neuron_fills=[
            np.load(data_dir + '/blocks.'+str(layer_num+1)+'.hook_resid_mid.npy'),
            np.load(data_dir + '/blocks.'+str(layer_num+1)+'.mlp.hook_post.npy'),
            np.load(data_dir + '/blocks.'+str(layer_num+1)+'.hook_mlp_out.npy')
        ]


        w1=np.load(data_dir + '/blocks.'+str(layer_num+1)+'.mlp.W_in'+'.npy')
        w2=np.load(data_dir + '/blocks.'+str(layer_num+1)+'.mlp.W_out'+'.npy')

        mlp2=get_mlp(w1, w2, neuron_fills) #, grads_1=grads_1, grads_2=grads_2)
        mlp2.move_to([-2.4, 0, 0])

        #Probably wrap this up.
        # def get_attention_layer() 
        # attention_connections_left=np.random.randn(32, 10)
        # attention_connections_right=np.random.randn(10, 32)
        attention_connection_display_thresh=0.5

        # Not quite sure how I want to pick attention patterns just jet
        # Some kinda activation or gradient based ranking probably?
        # Either way to do want that my incoming and outgoing connections match, ya know?
        # Maybe just take every thrid for now?
        all_attn_patterns=np.load(data_dir+'/blocks.'+str(layer_num)+'.attn.hook_pattern.npy')
        wO_full=np.load(data_dir+'/blocks.'+str(layer_num)+'.attn.W_O.npy')
        wq_full=np.load(data_dir+'/blocks.'+str(layer_num)+'.attn.W_Q.npy')

        attn_patterns=[]
        wos=[]
        wqs=[]
        for i in range(0, 30, 3): #Just take every thrid pattern for now. 
            attn_patterns.append(all_attn_patterns[0][i][1:,1:]) #Ignore BOS token
            wos.append(wO_full[i, 0])
            wqs.append(wq_full[i, :, 0])
        wos=np.array(wos)
        wqs=np.array(wqs)
        attention_connections_left=wqs.T #Queries
        attention_connections_right=wos


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
        attention_layer.move_to([-3.2, 0, 0])

        connections_left=m.VGroup()
        attention_connections_left_abs=np.abs(attention_connections_left)
        attention_connections_left_scaled=attention_connections_left_abs/np.max(attention_connections_left_abs) #np.percentile(attention_connections_left_abs, 99)
        for i, mlp_out_neuron in enumerate(mlp[4]):
            for j, attention_neuron in enumerate(connection_points_left):
                if np.abs(attention_connections_left_scaled[i, j])<0.5: continue
                if abs(i/4-j)>3: continue #Need to dial this up or lost it probably, but it is helpful!
                start_point, end_point = get_edge_points(mlp_out_neuron, attention_neuron, 0.06)
                line = m.Line(start_point, attention_neuron.get_center())
                # line.set_stroke(width=1, opacity=0.3)
                # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
                line.set_stroke(opacity=np.clip(attention_connections_left_scaled[i,j], 0, 1), width=1.0*attention_connections_left_scaled[i,j])
                line.set_color(CHILL_BROWN)
                connections_left.add(line)

        connections_right=m.VGroup()
        attention_connections_right_abs=np.abs(attention_connections_right)
        attention_connections_right_scaled=attention_connections_right_abs/np.percentile(attention_connections_left_abs, 99)
        for i, attention_neuron in enumerate(connection_points_right):
            for j, mlp_in_neuron in enumerate(mlp2[2]):
                if np.abs(attention_connections_right_scaled[i, j])<0.6: continue
                if abs(j/4-i)>3: continue #Need to dial this up or lost it probably, but it is helpful!
                start_point, end_point = get_edge_points(mlp_in_neuron, attention_neuron, 0.06)
                line = m.Line(start_point, attention_neuron.get_center())
                # line.set_stroke(width=1, opacity=0.3)
                # line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-attention_connection_display_thresh), 0.1, 1), width=1)
                line.set_stroke(opacity=np.clip(attention_connections_right_scaled[i,j], 0, 1), width=1.0*attention_connections_right_scaled[i,j])
                line.set_color(CHILL_BROWN)
                connections_right.add(line)





        #Order i add stuff here matters for occlusions. 
        self.add(connections_left)
        self.add(connections_right)
        self.add(mlp)
        self.add(attention_layer)
        self.add(mlp2)


        self.remove(mlp[3][1]); self.add(mlp[3][1]) #Ok seems like I'm just exploiting a bug, but this fixes layering. 
        self.remove(mlp[2][1]); self.add(mlp[2][1])
        self.remove(mlp[4][1]); self.add(mlp[4][1])
        
        self.remove(mlp2[3][1]); self.add(mlp2[3][1]) #Ok seems like I'm just exploiting a bug, but this fixes layering. 
        self.remove(mlp2[2][1]); self.add(mlp2[2][1])
        self.remove(mlp2[4][1]); self.add(mlp2[4][1])


        self.wait()


        # Man this is going to be cool when I can hook it up with predcitions and labels and everything
        # For input layer I think show words in order with other neurons in between? That's a little 
        # handwavy but seems ok to me
        # For the output -> i think we show connections for topk outputs, and topk words and probs
        # man I keep thinking of the poster lol. 


        # Ok Connection from mlps to next attention block is not terrible.
        # Now, I need to hack on attention to next MLP connections
        # Also, gradients
        # And of course, wrapping everything a bit more modularly. 
        # Oh and the the input and output deals layers -> right
        # Ok, I kinda think let me tackle that next mlp layer next and connections to it
        # Then gradients
        # Then probably wrap stuff up 
        # Then input output?

        #Ok, now for gradients I need to hop on a runpod for a bit right?


        #Kinda thinking that before I get stuff to grapped up here I should bring in some real data!
        #Hmm real weight values arr move finicky than I expected - need to noodle with them more!




        
        # self.remove(mlp[4])
        # self.add(mlp[4])
        
        

        # Ok so far so good, I'll wrap this up into a class shortly -> first though how I make connections to mlp??
        





        # self.remove(mlp[0])
        # self.add(mlp[1])

        self.wait() 


        #Ok making progress
        # I think Grads could still use some tuning - not terrible there
        # The only local connections thing is really helpful visuall -> you do lose a lot though hmmm
        #

        # man I hope i get to use this animation a few more times, it's takings some thinking!
        # Ok, so making progress. I've added more mlp neurons and it's getting a bit rat nesty again
        # For the weights, I kinda wonder if just connecting each neuron to a few neighbors might be the move
        # Ok nice yeah that's much less rats nesty - mayb i could even do that prbabilisticlaly or something...




        # Ok, making some progress here - I'm going to need to turn off some portion of the connections I think 
        # for it not be become a wall of fill. I think let's try just not drawing neuronss below some threshold?

        # Alright so I have an MLP layer that I don't hate -> I think at this scale basically just turning off
        # most of the connectiosn is the move. And I think that will look ok/fine. 
        # And obviously I'll turn them on/off based on the actually values -> that will be nice!
        # Ok, now there's a few different fun/intersting options I think. 
        # Could have the circles and and attention patterns fill up left to right for forward prop -> that woudl probably be cool?
        # Could be nice/interesting in monochrome brown opacities, or maybe viridis, we'll seee. 
        # I of course could show forward passes along the synapses, we'll see. 
        # Now backward passes I think are definitely along the synapses. 
        # For extra credit I could use glowdots to actually traverse the gaps -> not sure if this is 
        # visually helpful or not, wil keep that as a maybe option. 
        # Ok one more visual option I think it to bring in the bubbles and/or line colors not all at once
        # Could do min to max or randomly -> but having them kinda bubble in could be cool I think.
        # Man i guess activation patterns are more like activations that weights - I guess that's whey thier called activation patterns lol
        # For the monochrome colormap it might be cool to do that nice light tan as the brightest color. 
        #hmm not sure I like viridis fills -> looks kinda meh.
        # Ok monochrome is a little better, but I think the real problem with both is that you kidna want 
        # most of them/the average to be close to black, ya know? 
        # Can i just add an absolute value? Does that lose too much?
        # Ok yeah the absolute value does help visually a bit i think. 
        # Ok that will take some noodling, but I think it's not terrible. 
        # I think it could be cool/interesting to have the activations in monochrom, and the grads in color. 
        #
        # Ok, again for the grads, viridis is kinda meh -> leaning towards monochrom blues or maybe yellows
        # This opens up the possility of showing grads from different examples in different colors - which could be 
        # pretty cool! 
        # I dont' quite want the standard Blues colormaps, ideally i want a grad of zero to be transparent, not white
        # Let me look into that next. If I can't do it with color I should be able to go from black to my color and use 
        # opacity to acheve something similar
        # Not sure yet if I want to show weights and grads or just grads!
        # Ok cool yeah I'm not hating the cyan on top of the weights!
        # Ok cool let's hack on attention patterns a bit now? I'm a bit fuzzy on how I'm going to handle exports - 
        # will be a good challenge for the morning probably - let me see if I can get some basica attention patterns going -
        # I'll start with what Pranav did, it look pretty nice!







        self.wait()
        pass






# def get_mlp(w1, 
#             w2,
#             neuron_fills=None, #Black if None
#             grads_1=None,
#             grads_2=None,
#             line_weight=1.0, 
#             line_opacity=0.5, 
#             neuron_stroke_width=2.0, 
#             neuron_stroke_color='#948979', 
#             line_stroke_color='#948979', 
#             connection_display_thresh=1.1,
#             grad_display_thresh=0.5):

#     INPUT_NEURONS = w1.shape[0]
#     HIDDEN_NEURONS = w1.shape[1]
#     OUTPUT_NEURONS = w1.shape[0]
#     NEURON_RADIUS = 0.065
#     LAYER_SPACING = 0.28
#     VERTICAL_SPACING = 0.2
#     DOTS_SCALE=0.5
    
#     # Create layers
#     input_layer = VGroup()
#     hidden_layer = VGroup()
#     output_layer = VGroup()
#     dots = VGroup()
    
#     # Input layer
#     for i in range(INPUT_NEURONS):
#         if i == w1.shape[0]//2:  # Middle position for ellipsis
#             dot = Tex("...").rotate(PI/2, OUT).scale(DOTS_SCALE).move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#             dot.set_color(neuron_stroke_color)
#             dots.add(dot)
#         else:
#             neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
#             neuron.set_stroke(width=neuron_stroke_width)
#             if neuron_fills is None: 
#                 neuron.set_fill(color='#000000', opacity=1.0)
#             else: 
#                 neuron.set_fill(color=get_nueron_color(neuron_fills[0][i]), opacity=1.0)
#             neuron.move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#             input_layer.add(neuron)
            
#     # Hidden layer
#     for i in range(HIDDEN_NEURONS):
#         if i == w1.shape[1]//2:  # Middle position for ellipsis
#             dot = Tex("...").rotate(PI/2, OUT).scale(DOTS_SCALE).move_to(UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
#             dot.set_color(neuron_stroke_color)
#             dots.add(dot)
#         else:
#             neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
#             neuron.set_stroke(width=neuron_stroke_width)
#             if neuron_fills is None: 
#                 neuron.set_fill(color='#000000', opacity=1.0)
#             else: 
#                 neuron.set_fill(color=get_nueron_color(neuron_fills[1][i]), opacity=1.0)
#             neuron.move_to(UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
#             hidden_layer.add(neuron)
            
#     # Output layer
#     for i in range(OUTPUT_NEURONS):
#         if i == w1.shape[0]//2:  # Middle position for ellipsis
#             dot = Tex("...").rotate(PI/2, OUT).scale(DOTS_SCALE).move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#             dot.set_color(neuron_stroke_color)
#             dots.add(dot)
#         else:
#             neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
#             neuron.set_stroke(width=neuron_stroke_width)
#             if neuron_fills is None: 
#                 neuron.set_fill(color='#000000', opacity=1.0)
#             else: 
#                 neuron.set_fill(color=get_nueron_color(neuron_fills[2][i]), opacity=1.0)
#             neuron.move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#             output_layer.add(neuron)
            
#     # Create connections with edge points
#     connections = VGroup()
#     grad_conections=VGroup()
    
#     # Connect input to hidden layer
#     for i, in_neuron in enumerate(input_layer):
#         for j, hidden_neuron in enumerate(hidden_layer):
#             if np.abs(w1[i, j])<connection_display_thresh: continue
#             start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
#             line = Line(start_point, end_point)
#             line.set_stroke(opacity=np.clip(0.8*(np.abs(w1[i, j])-connection_display_thresh), 0, 1), width=line_weight)
#             # print(np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1))
#             line.set_color(line_stroke_color)
#             connections.add(line)
#             if grads_1 is not None:
#                 if np.abs(grads_1[i, j])<grad_display_thresh: continue
#                 line_grad = Line(start_point, end_point)
#                 line_grad.set_stroke(opacity=np.clip(0.8*(np.abs(grads_1[i, j])-grad_display_thresh), 0, 1), 
#                                     width=np.abs(grads_1[i, j]))
#                 # line_grad.set_stroke(opacity=0.8, width=2)
#                 # print(np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1))

#                 line_grad.set_color(get_grad_color(grads_1[i, j]))
#                 grad_conections.add(line_grad)

            
#     # Connect hidden to output layer
#     for i, hidden_neuron in enumerate(hidden_layer):
#         for j, out_neuron in enumerate(output_layer):
#             if np.abs(w2[i, j])<connection_display_thresh: continue
#             start_point, end_point = get_edge_points(hidden_neuron, out_neuron, NEURON_RADIUS)
#             line = Line(start_point, end_point) #, stroke_opacity=line_opacity, stroke_width=line_weight)
#             line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-connection_display_thresh), 0, 1), width=line_weight)

#             line.set_color(line_stroke_color)
#             connections.add(line)
#             if grads_2 is not None:
#                 if np.abs(grads_2[i, j])<grad_display_thresh: continue
#                 line_grad = Line(start_point, end_point)
#                 line_grad.set_stroke(opacity=np.clip(0.8*(np.abs(grads_2[i, j])-grad_display_thresh), 0, 1), 
#                                     width=np.abs(grads_2[i, j]))
#                 # line_grad.set_stroke(opacity=0.8, width=2)
#                 # print(np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1))

#                 line_grad.set_color(get_grad_color(grads_2[i, j]))
#                 grad_conections.add(line_grad)

                
#     return VGroup(connections, grad_conections, input_layer, hidden_layer, output_layer, dots)
#     # return VGroup(grad_conections, input_layer, hidden_layer, output_layer, dots)





# def get_mlp(w1, 
#             w2,
#             line_weight=1.0, 
#             line_opacity=0.5, 
#             neuron_stroke_width=2.0, 
#             neuron_stroke_color='#948979', 
#             line_stroke_color='#948979', 
#             connection_display_thresh=1.1):
#     INPUT_NEURONS = w1.shape[0]
#     HIDDEN_NEURONS = w1.shape[1]
#     OUTPUT_NEURONS = w1.shape[0]
#     NEURON_RADIUS = 0.08
#     LAYER_SPACING = 0.25
#     VERTICAL_SPACING = 0.25
#     DOTS_SCALE=0.5
    
#     # Create layers
#     input_layer = VGroup()
#     hidden_layer = VGroup()
#     output_layer = VGroup()
#     dots = VGroup()
    
#     # Input layer
#     for i in range(INPUT_NEURONS):
#         if i == w1.shape[0]//2:  # Middle position for ellipsis
#             dot = Tex("...").rotate(PI/2, OUT).scale(DOTS_SCALE).move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#             dot.set_color(neuron_stroke_color)
#             dots.add(dot)
#         else:
#             neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
#             neuron.set_stroke(width=neuron_stroke_width)
#             neuron.set_fill(color='#000000', opacity=1.0)
#             neuron.move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#             input_layer.add(neuron)
            
#     # Hidden layer
#     for i in range(HIDDEN_NEURONS):
#         if i == w1.shape[1]//2:  # Middle position for ellipsis
#             dot = Tex("...").rotate(PI/2, OUT).scale(DOTS_SCALE).move_to(UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
#             dot.set_color(neuron_stroke_color)
#             dots.add(dot)
#         else:
#             neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
#             neuron.set_stroke(width=neuron_stroke_width)
#             neuron.set_fill(color='#000000', opacity=1.0)
#             neuron.move_to(UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
#             hidden_layer.add(neuron)
            
#     # Output layer
#     for i in range(OUTPUT_NEURONS):
#         if i == w1.shape[0]//2:  # Middle position for ellipsis
#             dot = Tex("...").rotate(PI/2, OUT).scale(DOTS_SCALE).move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#             dot.set_color(neuron_stroke_color)
#             dots.add(dot)
#         else:
#             neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
#             neuron.set_stroke(width=neuron_stroke_width)
#             neuron.set_fill(color='#000000', opacity=1.0)
#             neuron.move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#             output_layer.add(neuron)
            
#     # Create connections with edge points
#     connections = VGroup()
    
#     # Connect input to hidden layer
#     for i, in_neuron in enumerate(input_layer):
#         for j, hidden_neuron in enumerate(hidden_layer):
#             if np.abs(w1[i, j])<connection_display_thresh: continue
#             start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
#             line = Line(start_point, end_point)
#             line.set_stroke(opacity=np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1), width=line_weight)
#             # print(np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1))
#             line.set_color(line_stroke_color)
#             connections.add(line)
            
#     # Connect hidden to output layer
#     for i, hidden_neuron in enumerate(hidden_layer):
#         for j, out_neuron in enumerate(output_layer):
#             if np.abs(w2[i, j])<connection_display_thresh: continue
#             start_point, end_point = get_edge_points(hidden_neuron, out_neuron, NEURON_RADIUS)
#             line = Line(start_point, end_point) #, stroke_opacity=line_opacity, stroke_width=line_weight)
#             line.set_stroke(opacity=np.clip(1.0*(np.abs(w2[i, j])-connection_display_thresh), 0, 1), width=line_weight)

#             line.set_color(line_stroke_color)
#             connections.add(line)

                
#     return VGroup(connections, input_layer, hidden_layer, output_layer, dots)











