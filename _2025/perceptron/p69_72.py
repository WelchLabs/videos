from welch_paths import welch_path
import manimgx as gx
from welch_geometry import diagram_arrow
import random
from welch_scene import WelchScene
from welch_text import math_tex

from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append(welch_path('perceptron/animations/videos'))
from helpers import circle_plus, create_prism
#
CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'


# Helper function to get edge points between two circles
def get_edge_points(circle1, circle2, neuron_radius):
    # Get direction vector from circle1 to circle2
    direction = circle2.get_center() - circle1.get_center()
    unit_vector = direction / np.linalg.norm(direction)
    
    # Calculate start and end points
    start_point = circle1.get_center() + unit_vector * neuron_radius
    end_point = circle2.get_center() - unit_vector * neuron_radius
    
    return start_point, end_point



def get_mlp(line_weight=2.0, line_opacity=0.3):
    INPUT_NEURONS = 5
    HIDDEN_NEURONS = 7
    OUTPUT_NEURONS = 5
    NEURON_RADIUS = 0.2
    LAYER_SPACING = 1.5
    VERTICAL_SPACING = 0.5
    
    # Create layers
    input_layer = gx.VGroup()
    hidden_layer = gx.VGroup()
    output_layer = gx.VGroup()
    dots = gx.VGroup()
    
    # Input layer
    for i in range(INPUT_NEURONS):
        if i == 2:  # Middle position for ellipsis
            dot = math_tex("...").rotate(gx.PI/2, gx.OUT).move_to(gx.LEFT * LAYER_SPACING + gx.UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            dots.add(dot)
        else:
            neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=gx.WHITE)
            neuron.move_to(gx.LEFT * LAYER_SPACING + gx.UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            input_layer.add(neuron)
            
    # Hidden layer
    for i in range(HIDDEN_NEURONS):
        if i == 4:  # Middle position for ellipsis
            dot = math_tex("...").rotate(gx.PI/2, gx.OUT).move_to(gx.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
            dots.add(dot)
        else:
            neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=gx.WHITE)
            neuron.move_to(gx.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
            hidden_layer.add(neuron)
            
    # Output layer
    for i in range(OUTPUT_NEURONS):
        if i == 2:  # Middle position for ellipsis
            dot = math_tex("...").rotate(gx.PI/2, gx.OUT).move_to(gx.RIGHT * LAYER_SPACING + gx.UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            dots.add(dot)
        else:
            neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=gx.WHITE)
            neuron.move_to(gx.RIGHT * LAYER_SPACING + gx.UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            output_layer.add(neuron)
            
    # Create connections with edge points
    connections = gx.VGroup()
    
    # # Helper function to get edge points between two circles
    # def get_edge_points(circle1, circle2):
    #     # Get direction vector from circle1 to circle2
    #     direction = circle2.get_center() - circle1.get_center()
    #     unit_vector = direction / np.linalg.norm(direction)
        
    #     # Calculate start and end points
    #     start_point = circle1.get_center() + unit_vector * NEURON_RADIUS
    #     end_point = circle2.get_center() - unit_vector * NEURON_RADIUS
        
    #     return start_point, end_point
    
    # Connect input to hidden layer
    for in_neuron in input_layer:
        for hidden_neuron in hidden_layer:
            start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
            line = gx.Line(start_point, end_point, stroke_opacity=line_opacity, stroke_width=line_weight)
            connections.add(line)
            
    # Connect hidden to output layer
    for hidden_neuron in hidden_layer:
        for out_neuron in output_layer:
            start_point, end_point = get_edge_points(hidden_neuron, out_neuron, NEURON_RADIUS)
            line = gx.Line(start_point, end_point, stroke_opacity=line_opacity, stroke_width=line_weight)
            connections.add(line)

    #could do this later I guess
    # input_layer.set_opacity(0.9)
    # hidden_layer.set_opacity(0.9)
    # output_layer.set_opacity(0.9)
    # dots.set_opacity(0.9)

                
    return gx.VGroup(connections, input_layer, hidden_layer, output_layer, dots)


def get_transformer_block_gpt(attention_block_width=8, attention_block_height=5, attention_block_depth=1, mlp_block_width=8, mlp_block_height=5, 
                          mlp_block_depth=1, block_orig=np.array([0,0,0]), line_padding=0.3, residual_compute_block_spacing=11, line_thickness=6, 
                          circle_stroke_width=3, plus_stroke_width=3, residual_to_attention_line_spacing=5.0):

        attention_block_1=create_prism(center=block_orig+np.array([11,-3-attention_block_depth/2,0]), height=attention_block_depth, width=attention_block_width, depth=attention_block_height, 
                                    face_colors=BLUE, opacity=0.2, label_text='Attention', 
                                    label_size=80, label_opacity=1.0, label_face="back")
        mlp_block_1=create_prism(center=block_orig+np.array([11,-12-mlp_block_depth,0]), height=mlp_block_depth, width=mlp_block_width, depth=mlp_block_height, 
                            face_colors=gx.GREEN, opacity=0.2, label_text='Multilayer Perceptron',
                            label_size=80, label_opacity=1.0, label_face="back")

        #Input to Attention
        a0=diagram_arrow(start=block_orig+np.array([residual_compute_block_spacing+line_padding,0,0]),
                  end=block_orig+np.array([0,0,0]), #Special offset from matrix here, not sure how to handle this generally yet
                  fill_color=gx.WHITE, thickness=line_thickness, tip_width_ratio=0)
        a1=diagram_arrow(start=block_orig+np.array([residual_compute_block_spacing, 0.3,0]),
                 end=block_orig+np.array([residual_compute_block_spacing, -3.0,0]),fill_color=gx.WHITE, thickness=line_thickness)

        #Attention to Residual
        a2=diagram_arrow(start=block_orig+np.array([residual_compute_block_spacing, -3.0-attention_block_depth, 0]),
                   end=block_orig+np.array([residual_compute_block_spacing, -4.3-attention_block_depth, 0]),fill_color=gx.WHITE, thickness=line_thickness, tip_width_ratio=0)
        a3=diagram_arrow(start=block_orig+np.array([residual_compute_block_spacing+line_padding,-4-attention_block_depth,0]),
                   end=block_orig+np.array([0.5,-4-attention_block_depth,0]),fill_color=gx.WHITE, thickness=line_thickness)

        #First Addition
        c=circle_plus(circle_stroke_width=circle_stroke_width, plus_stroke_width=plus_stroke_width, overall_scale=0.6, position=block_orig+np.array([0,-4-attention_block_depth,0]))

        #Input to First Addition
        a4=diagram_arrow(start=block_orig+np.array([0,0,0]), #-0.75
                   end=block_orig+np.array([0,-3.5-attention_block_depth,0]),fill_color=gx.WHITE, thickness=line_thickness)
        #Input to Second Addition
        a5=diagram_arrow(start=block_orig+np.array([0,-4.5-attention_block_depth,0]),
                   end=block_orig+np.array([0,-8.5-attention_block_depth-mlp_block_depth,0]),fill_color=gx.WHITE, thickness=line_thickness)

        #Residual to MLP
        a6=diagram_arrow(start=block_orig+np.array([residual_compute_block_spacing+line_padding,-6.5-attention_block_depth,0]), #-7.5,0])
                   end=block_orig+np.array([-0.3,-6.5-attention_block_depth,0]), fill_color=gx.WHITE, thickness=line_thickness, tip_width_ratio=0)
        a7=diagram_arrow(start=block_orig+np.array([residual_compute_block_spacing, -6.2-attention_block_depth,0]),
                   end=block_orig+np.array([residual_compute_block_spacing, -8-attention_block_depth,0]), fill_color=gx.WHITE, thickness=line_thickness)

        #MLP to residual
        a8=diagram_arrow(start=block_orig+np.array([residual_compute_block_spacing, -8.0-attention_block_depth-mlp_block_depth,0]), #Shave 3.0
                   end=block_orig+np.array([residual_compute_block_spacing, -9.3-attention_block_depth-mlp_block_depth,0]), fill_color=gx.WHITE, thickness=line_thickness, tip_width_ratio=0)
        a9=diagram_arrow(start=block_orig+np.array([residual_compute_block_spacing+line_padding,-9.0-attention_block_depth-mlp_block_depth,0]),
                   end=block_orig+np.array([0.5,-9.0-attention_block_depth-mlp_block_depth,0]), fill_color=gx.WHITE, thickness=line_thickness)

        #Second Addition
        c2=circle_plus(circle_stroke_width=circle_stroke_width, plus_stroke_width=plus_stroke_width, overall_scale=0.6, position=block_orig+np.array([0,-9-attention_block_depth-mlp_block_depth,0]))

        #Out Arrow
        a10=diagram_arrow(start=block_orig+np.array([0,-9.5-attention_block_depth-mlp_block_depth,0]),
                   end=block_orig+np.array([0,-11.5-attention_block_depth-mlp_block_depth,0]),fill_color=gx.WHITE, thickness=line_thickness)
        return gx.VGroup(attention_block_1, mlp_block_1, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, c, c2),  gx.VGroup(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, c, c2)

def get_attention_first_layer():
     # Configuration
    INPUT_NEURONS = 5
    HIDDEN_NEURONS = 5
    OUTPUT_NEURONS = 5
    NEURON_RADIUS = 0.2
    LAYER_SPACING = 1.5
    VERTICAL_SPACING = 0.5

    # Create layers
    input_layer = gx.VGroup()
    hidden_layer = gx.VGroup()
    query_layer = gx.VGroup()
    key_layer = gx.VGroup()
    output_layer = gx.VGroup()
    dots = gx.VGroup()
    hidden_dots = gx.VGroup()
    query_dots = gx.VGroup()
    key_dots = gx.VGroup()


    # Input layer
    for i in range(INPUT_NEURONS):
        if i == 2:  # Middle position for ellipsis
            dot = math_tex("...").rotate(gx.PI/2, gx.OUT).move_to(gx.LEFT * LAYER_SPACING + gx.UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            dots.add(dot)
        else:
            neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=gx.WHITE)
            neuron.move_to(gx.LEFT * LAYER_SPACING + gx.UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            input_layer.add(neuron)

    # Hidden layer
    for i in range(HIDDEN_NEURONS):
        if i == 2:  # Middle position for ellipsis
            dot = math_tex("...").rotate(gx.PI/2, gx.OUT).move_to(gx.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
            hidden_dots.add(dot)
        else:
            neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=gx.WHITE)
            neuron.move_to(gx.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
            hidden_layer.add(neuron)

    # Queries 
    query_shift=3
    for i in range(HIDDEN_NEURONS):
        if i == 2:  # Middle position for ellipsis
            dot = math_tex("...").rotate(gx.PI/2, gx.OUT).move_to(gx.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
            dot.shift(query_shift*gx.UP)
            dot.set_color(YELLOW)
            query_dots.add(dot)
        else:
            neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=YELLOW)
            neuron.move_to(gx.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
            neuron.shift(query_shift*gx.UP)
            query_layer.add(neuron)

    # Keys 
    key_shift=-3
    for i in range(HIDDEN_NEURONS):
        if i == 2:  # Middle position for ellipsis
            dot = math_tex("...").rotate(gx.PI/2, gx.OUT).move_to(gx.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
            dot.set_color(BLUE)
            dot.shift(key_shift*gx.UP)
            key_dots.add(dot)
        else:
            neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=BLUE)
            neuron.move_to(gx.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
            neuron.shift(key_shift*gx.UP)
            key_layer.add(neuron)

    connections = gx.VGroup()
    # Connect input to hidden layer
    for in_neuron in input_layer:
        for hidden_neuron in hidden_layer:
            start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
            line = gx.Line(start_point, end_point, stroke_opacity=0.3, stroke_width=2.0)
            connections.add(line)

        for hidden_neuron in query_layer:
            start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
            line = gx.Line(start_point, end_point, stroke_opacity=0.3, stroke_width=2.0)
            line.set_color(YELLOW)
            connections.add(line)

        for hidden_neuron in key_layer:
            start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
            line = gx.Line(start_point, end_point, stroke_opacity=0.3, stroke_width=2.0)
            line.set_color(BLUE)
            connections.add(line)

    #Would like to make sure my hidden to output colored neurons do not violate causality!

    return gx.VGroup(input_layer, hidden_layer, hidden_dots, key_layer, query_layer, dots, key_dots, query_dots, connections)



def random_color_between():
    # Convert hex to RGB
    start = tuple(int(x, 16) for x in ('65', 'c8', 'd0'))
    end = tuple(int(x, 16) for x in ('ff', 'd3', '5a'))
    
    # Generate random point between the colors
    random_color = tuple(
        int(start[i] + (end[i] - start[i]) * random.random())
        for i in range(3)
    )
    
    # Convert back to hex
    return '#{:02x}{:02x}{:02x}'.format(*random_color)


def get_attention():
    NEURON_RADIUS = 0.2

    first_layers=[]
    hidden_layers=[]
    hidden_layers_dots=[]
    for i in range(7):
        a=get_attention_first_layer()
        a.shift([0, 0, 5*i/11-5/2])
        first_layers.append(a)
        hidden_layers.append(a[1])
        hidden_layers_dots.append(a[2])

    # for a in first_layers:
    #     self.add(a)
    # self.wait()

    output_layers=[o.copy() for o in hidden_layers]
    output_dots = [o.copy() for o in hidden_layers_dots]
    for o in output_layers:
        o.shift([3, 0, 0])

    for o in output_dots: ## yeah yeah yeah it's ugly calm down everyone
        o.shift([3, 0, 0])

    # for o in output_layers+output_dots:
    #     self.add(o)

    #Ok dope, now i want a masked set of randomly BLUE/YELLOW colored lines between the hidden layer and output.
    output_connections=gx.VGroup()
    for i, h in enumerate(hidden_layers):
        for j, o in enumerate(output_layers):
            if i >= j: 
                for in_neuron in h:
                    for hidden_neuron in o:
                        start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
                        line = gx.Line(start_point, end_point, stroke_opacity=0.3, stroke_width=2.0)
                        line.set_color(random_color_between())
                        output_connections.add(line)

    # self.add(output_connections)

    return gx.VGroup(gx.VGroup(first_layers), gx.VGroup(output_layers), gx.VGroup(output_dots), output_connections)


class P69_72v3(WelchScene):
    def construct(self):
        #Lets go!

        attention_block_width=12
        attention_block_height=5
        attention_block_depth=8
        mlp_block_width=8
        mlp_block_height=5
        mlp_block_depth=8
        block_orig=np.array([0,0,0])
        line_padding=0.3
        residual_compute_block_spacing=11
        line_thickness=6
        circle_stroke_width=3
        plus_stroke_width=3

        main_tb, main_tb_lines=get_transformer_block_gpt(block_orig=np.array([0,0,0]), attention_block_depth=attention_block_depth, mlp_block_depth=mlp_block_depth,
                            attention_block_width=attention_block_width)

        non_main_tbs=gx.VGroup()
        non_main_tb_lines=gx.VGroup()

        for i in range(-7,0):
            tb, tb_lines=get_transformer_block_gpt(block_orig=np.array([0,-27.5*i,0]), attention_block_depth=attention_block_depth, mlp_block_depth=mlp_block_depth,
                                        attention_block_width=attention_block_width)
            non_main_tbs.add(tb)
            non_main_tb_lines.add(tb_lines)

        for i in range(1, 8):
            tb, tb_lines=get_transformer_block_gpt(block_orig=np.array([0,-27.5*i,0]), attention_block_depth=attention_block_depth, mlp_block_depth=mlp_block_depth,
                                        attention_block_width=attention_block_width)
            non_main_tbs.add(tb)
            non_main_tb_lines.add(tb_lines)

        ## Add big ole ellipses in the center
        for o in [-2, -7, -12]:
            d=gx.Dot([6,-27.5*8+o,0], radius=1)
            d.set_color(CHILL_BROWN)
            self.add(d)

        for i in range(9, 17):
            tb, tb_lines=get_transformer_block_gpt(block_orig=np.array([0,-27.5*i+12,0]), attention_block_depth=attention_block_depth, mlp_block_depth=mlp_block_depth,
                                        attention_block_width=attention_block_width)
            non_main_tbs.add(tb)
            non_main_tb_lines.add(tb_lines)

        self.add(main_tb)
        self.add(main_tb_lines)

        self.add(non_main_tbs)
        self.add(non_main_tb_lines)

        # self.set_view(-54, 61, 0, (85.42, -259.86, -42.71), 312.62)
        self.set_view(-144, 68, 0, (3.42, 6.35, 4.32), 37.33)
        self.wait()

        #Get more accurate camera position
        # center = self.camera.frame.get_center()
        # height = self.camera.frame.height
        # angles = self.view_angles() * gx.DEGREES

        # call = f"reorient("
        # theta, phi, gamma = (angles / DEGREES) #.astype(int)
        # call += f"{theta}, {phi}, {gamma}"
        # if any(center != 0):
        #     call += f", {tuple(np.round(center, 4))}"
        # if height != FRAME_HEIGHT:
        #     call += ", {:.4f}".format(height)
        # call += ")"
        # print(call)
        # pyperclip.copy(call)


        self.play(self.animate_view(-90.18440887882343, 58.711012775757, 0.0, (83.48, -134.0099, -39.85), 457.0400) , run_time=6.0)
        self.wait()

        # self.play(self.animate_view(-69, 27, 0, (8.85, -14.37, 0.94), 31.50), run_time=6)
        self.play(self.animate_view(-90, 27, 0, (9.09, -14.74, 0.81), 28.92), run_time=6)
        self.wait()

        #Now fade out all but main tb
        self.play(gx.FadeOut(non_main_tbs), gx.FadeOut(non_main_tb_lines), run_time=3)
        self.wait()

        #Fade in MLPs while turning down opacity on everyting else and moving camera?
        n_stacked_mlps=7
        mlps=gx.VGroup()
        for i in range(n_stacked_mlps):
            if i<6:
                mlp=get_mlp(line_weight=2.0, line_opacity=0.3)
                mlp.set_opacity(0.3)
            else:
                mlp=get_mlp(line_weight=3.0, line_opacity=1.0) #Top one
                # mlp.set_opacity(1.0)
            mlp.rotate(gx.PI/2, gx.IN).scale(2.0)
            mlp.move_to(main_tb[1]) #MLP block
            mlp.shift([0, 0, mlp_block_height*i/float(n_stacked_mlps)-mlp_block_height/2])
            mlps.add(mlp)

        self.wait()

        self.play(main_tb.animate.set_opacity(0.1), gx.FadeIn(mlps), self.animate_view(-90, 19, 0, (10.67, -20.4, 1.78), 15.11), run_time=3)
        self.wait()


        #P71, move over the attention
        self.play(main_tb.animate.set_opacity(1.0), mlps.animate.set_opacity(0.3), 
            self.animate_view(-90, 25, 0, (10.58, -7.19, 0.46), 20.73), run_time=3)
        self.wait()        


        # Now add in attention network

        att=get_attention()        
        att.rotate(gx.PI/2, gx.IN).scale(1.5)
        att.move_to(main_tb[0])
        self.wait()
        self.play(main_tb.animate.set_opacity(0.1), gx.FadeIn(att),
            self.animate_view(-90, 32, 0, (10.52, -6.99, 0.45), 14.98), run_time=3)
        self.wait()


        self.play(self.animate_view(-90, 87, 0, (10.51, -6.99, 0.09), 14.98), run_time=3)
        self.wait()

        self.play(gx.FadeIn(non_main_tbs), gx.FadeIn(non_main_tb_lines), main_tb.animate.set_opacity(0.6), run_time=1)
        self.play(self.animate_view(-90.18440887882343, 58.711012775757, 0.0, (83.48, -134.0099, -39.85), 457.0400), run_time=6)
        self.wait()


        self.wait(20)


class OpeningHacking(WelchScene):
    def construct(self):
        #Lets go!

        attention_block_width=12
        attention_block_height=5
        attention_block_depth=8
        mlp_block_width=8
        mlp_block_height=5
        mlp_block_depth=8
        block_orig=np.array([0,0,0])
        line_padding=0.3
        residual_compute_block_spacing=11
        line_thickness=6
        circle_stroke_width=3
        plus_stroke_width=3

        main_tb, main_tb_lines=get_transformer_block_gpt(block_orig=np.array([0,0,0]), attention_block_depth=attention_block_depth, mlp_block_depth=mlp_block_depth,
                            attention_block_width=attention_block_width)

        non_main_tbs=gx.VGroup()
        non_main_tb_lines=gx.VGroup()

        for i in range(-7,0):
            tb, tb_lines=get_transformer_block_gpt(block_orig=np.array([0,-27.5*i,0]), attention_block_depth=attention_block_depth, mlp_block_depth=mlp_block_depth,
                                        attention_block_width=attention_block_width)
            non_main_tbs.add(tb)
            non_main_tb_lines.add(tb_lines)

        for i in range(1, 8):
            tb, tb_lines=get_transformer_block_gpt(block_orig=np.array([0,-27.5*i,0]), attention_block_depth=attention_block_depth, mlp_block_depth=mlp_block_depth,
                                        attention_block_width=attention_block_width)
            non_main_tbs.add(tb)
            non_main_tb_lines.add(tb_lines)

        ## Add big ole ellipses in the center
        for o in [-2, -7, -12]:
            d=gx.Dot([6,-27.5*8+o,0], radius=1)
            d.set_color(CHILL_BROWN)
            self.add(d)

        for i in range(9, 17):
            tb, tb_lines=get_transformer_block_gpt(block_orig=np.array([0,-27.5*i+12,0]), attention_block_depth=attention_block_depth, mlp_block_depth=mlp_block_depth,
                                        attention_block_width=attention_block_width)
            non_main_tbs.add(tb)
            non_main_tb_lines.add(tb_lines)

        self.add(main_tb)
        self.add(main_tb_lines)

        self.add(non_main_tbs)
        self.add(non_main_tb_lines)

        # self.set_view(-54, 61, 0, (85.42, -259.86, -42.71), 312.62)
        self.set_view(-144, 68, 0, (3.42, 6.35, 4.32), 37.33)
        self.wait()

        #Get more accurate camera position
        # center = self.camera.frame.get_center()
        # height = self.camera.frame.height
        # angles = self.view_angles() * gx.DEGREES

        # call = f"reorient("
        # theta, phi, gamma = (angles / DEGREES) #.astype(int)
        # call += f"{theta}, {phi}, {gamma}"
        # if any(center != 0):
        #     call += f", {tuple(np.round(center, 4))}"
        # if height != FRAME_HEIGHT:
        #     call += ", {:.4f}".format(height)
        # call += ")"
        # print(call)
        # pyperclip.copy(call)


        self.play(self.animate_view(-90.18440887882343, 58.711012775757, 0.0, (83.48, -134.0099, -39.85), 457.0400) , run_time=6.0)
        self.wait()

        # self.play(self.animate_view(-69, 27, 0, (8.85, -14.37, 0.94), 31.50), run_time=6)
        self.play(self.animate_view(-90, 27, 0, (9.09, -14.74, 0.81), 28.92), run_time=6)
        self.wait()

        #Now fade out all but main tb
        self.play(gx.FadeOut(non_main_tbs), gx.FadeOut(non_main_tb_lines), run_time=3)
        self.wait()

        #Fade in MLPs while turning down opacity on everyting else and moving camera?
        n_stacked_mlps=7
        mlps=gx.VGroup()
        for i in range(n_stacked_mlps):
            if i<6:
                mlp=get_mlp(line_weight=2.0, line_opacity=0.3)
                mlp.set_opacity(0.3)
            else:
                mlp=get_mlp(line_weight=3.0, line_opacity=1.0) #Top one
                # mlp.set_opacity(1.0)
            mlp.rotate(gx.PI/2, gx.IN).scale(2.0)
            mlp.move_to(main_tb[1]) #MLP block
            mlp.shift([0, 0, mlp_block_height*i/float(n_stacked_mlps)-mlp_block_height/2])
            mlps.add(mlp)

        self.wait()

        self.play(main_tb.animate.set_opacity(0.1), gx.FadeIn(mlps), self.animate_view(-90, 19, 0, (10.67, -20.4, 1.78), 15.11), run_time=3)
        self.wait()


        #P71, move over the attention
        self.play(main_tb.animate.set_opacity(1.0), #mlps.animate.set_opacity(0.3), 
            self.animate_view(-90, 25, 0, (10.58, -7.19, 0.46), 20.73), run_time=3)
        self.wait()        


        # Now add in attention network

        att=get_attention()        
        att.rotate(gx.PI/2, gx.IN).scale(1.5)
        att.move_to(main_tb[0])
        self.wait()
        self.play(main_tb.animate.set_opacity(0.1), gx.FadeIn(att),
            self.animate_view(-90, 32, 0, (10.52, -6.99, 0.45), 14.98), run_time=3)
        self.wait()


        self.set_view(-55, 62, 0, (10.11, -12.14, -1.59), 22.56)
        self.wait()
        self.play(self.animate_view(-137, 46, 0, (10.11, -12.14, -1.59), 22.56), run_time=5)

        #self.play(self.animate_view(-90, 87, 0, (10.51, -6.99, 0.09), 14.98), run_time=3)
        #self.wait()

        #self.play(FadeIn(non_main_tbs), FadeIn(non_main_tb_lines), main_tb.animate.set_opacity(0.6), run_time=1)
        #self.play(self.animate_view(-90.18440887882343, 58.711012775757, 0.0, (83.48, -134.0099, -39.85), 457.0400), run_time=6)
        #self.wait()


        self.wait(20)


class P70Hackin(WelchScene):
    def construct(self):

        attention_block_width=12
        attention_block_height=5
        attention_block_depth=8
        mlp_block_width=8
        mlp_block_height=5
        mlp_block_depth=8
        block_orig=np.array([0,0,0])
        line_padding=0.3
        residual_compute_block_spacing=11
        line_thickness=6
        circle_stroke_width=3
        plus_stroke_width=3

        tb, tb_lines=get_transformer_block_gpt(block_orig=np.array([0,0,0]), attention_block_depth=attention_block_depth, mlp_block_depth=mlp_block_depth,
                                                attention_block_width=attention_block_width)

        #Ok, so yeah I'll have to mess with the spacing, shoulnd't be terrible though. 

        self.add(tb)
        self.add(tb_lines)

        self.set_view(-90, 1, 0, (7.32, -12.4, -0.31), 23.42) #Something like this for overhead view
        self.wait()

        self.set_view(-60, 56, 0, (8.33, -12.23, 0.91), 23.42) #Something like this for isometric view


        # Ok, new scaling looks pretty good - not let me add stacked MLPs to the MLP block
        # Then I can figure out what the hell to do with attention
        # And then replicate this bugger. 

        n_stacked_mlps=7
        mlps=gx.VGroup()
        for i in range(n_stacked_mlps):
            mlp=get_mlp()
            mlp.rotate(gx.PI/2, gx.IN).scale(2.0)
            mlp.move_to(tb[1]) #MLP block
            mlp.shift([0, 0, mlp_block_height*i/float(n_stacked_mlps)-mlp_block_height/2])
            mlps.add(mlp)

        self.add(mlps)
        self.wait()

        # Ok dope -> I can work with that!
        # Will tweak/adjust as I work through script 
        # But before that - let me go ahead and dive into how I want to do attention, and make sure I can replicate a bunch of these in a row with a ... in the middle
        # Alright - What I believe to be the core core problem, how do I want to handle attention?

        att=get_attention()
        self.wait()
        
        att.rotate(gx.PI/2, gx.IN).scale(1.5)
        att.move_to(tb[0])
        self.add(att) #Ok looks pretty dope!

        #Now last sanity check before I go do the real thing here -> adding a bunch of blocks with an ellipses in the center...
        for i in range(-7,0):
            tb, tb_lines=get_transformer_block_gpt(block_orig=np.array([0,-27.5*i,0]), attention_block_depth=attention_block_depth, mlp_block_depth=mlp_block_depth,
                                        attention_block_width=attention_block_width)
            self.add(tb, tb_lines)

        for i in range(8):
            tb, tb_lines=get_transformer_block_gpt(block_orig=np.array([0,-27.5*i,0]), attention_block_depth=attention_block_depth, mlp_block_depth=mlp_block_depth,
                                        attention_block_width=attention_block_width)
            self.add(tb, tb_lines)

        ## Add big ole ellipses in the center
        for o in [-2, -7, -12]:
            d=gx.Dot([6,-27.5*8+o,0], radius=1)
            d.set_color(CHILL_BROWN)
            self.add(d)


        for i in range(9, 17):
            tb, tb_lines=get_transformer_block_gpt(block_orig=np.array([0,-27.5*i+12,0]), attention_block_depth=attention_block_depth, mlp_block_depth=mlp_block_depth,
                                        attention_block_width=attention_block_width)
            self.add(tb, tb_lines)
        self.set_view(-54, 61, 0, (85.42, -259.86, -42.71), 312.62)
        self.wait()







class AttentionHackin2(WelchScene):
    def construct(self):
        NEURON_RADIUS = 0.2

        first_layers=[]
        hidden_layers=[]
        hidden_layers_dots=[]
        for i in range(7):
            a=get_attention_first_layer()
            a.shift([0, 0, 5*i/7-5/2])
            first_layers.append(a)
            hidden_layers.append(a[1])
            hidden_layers_dots.append(a[2])

        for a in first_layers:
            self.add(a)
        self.wait()

        output_layers=[o.copy() for o in hidden_layers]
        output_dots = [o.copy() for o in hidden_layers_dots]
        for o in output_layers:
            o.shift([3, 0, 0])

        for o in output_dots: ## yeah yeah yeah it's ugly calm down everyone
            o.shift([3, 0, 0])

        for o in output_layers+output_dots:
            self.add(o)

        #Ok dope, now i want a masked set of randomly BLUE/YELLOW colored lines between the hidden layer and output.
        output_connections=gx.VGroup()
        for i, h in enumerate(hidden_layers):
            for j, o in enumerate(output_layers):
                if i >= j: 
                    for in_neuron in h:
                        for hidden_neuron in o:
                            start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
                            line = gx.Line(start_point, end_point, stroke_opacity=0.3, stroke_width=2.0)
                            line.set_color(random_color_between())
                            output_connections.add(line)

        self.add(output_connections)



        # self.set_view(0, 0, 0, (1.78, 2.79, 0.0), 10.05)
        self.set_view(0, 0, 0, (2.31, -0.03, 0.0), 9.07)
        self.wait()







class AttentionHacking(WelchScene):
    def construct(self):
        # Ok ok ok ok ok ok I have an idea -> it's kinda messy, but might not be terrible. 
        # Start with a single set of inputs, just like the mlp block
        # Have 3 sets of hidden layers -> there's a few options for geomeetry here
        # Then have cross token inteconnection in second layer
        # And maybe just and arrow saying that the Keys/Queries network controls the values weights?


        # Configuration
        INPUT_NEURONS = 5
        HIDDEN_NEURONS = 5
        OUTPUT_NEURONS = 5
        NEURON_RADIUS = 0.2
        LAYER_SPACING = 1.5
        VERTICAL_SPACING = 0.5

        # Create layers
        input_layer = gx.VGroup()
        hidden_layer = gx.VGroup()
        output_layer = gx.VGroup()
        dots = gx.VGroup()


        # Input layer
        for i in range(INPUT_NEURONS):
            if i == 2:  # Middle position for ellipsis
                dot = math_tex("...").rotate(gx.PI/2, gx.OUT).move_to(gx.LEFT * LAYER_SPACING + gx.UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
                dots.add(dot)
            else:
                neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=gx.WHITE)
                neuron.move_to(gx.LEFT * LAYER_SPACING + gx.UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
                input_layer.add(neuron)

        # Hidden layer
        for i in range(HIDDEN_NEURONS):
            if i == 2:  # Middle position for ellipsis
                dot = math_tex("...").rotate(gx.PI/2, gx.OUT).move_to(gx.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
                dots.add(dot)
            else:
                neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=gx.WHITE)
                neuron.move_to(gx.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
                hidden_layer.add(neuron)

        # Queries 
        query_shift=3
        for i in range(HIDDEN_NEURONS):
            if i == 2:  # Middle position for ellipsis
                dot = math_tex("...").rotate(gx.PI/2, gx.OUT).move_to(gx.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
                dot.shift(query_shift*gx.UP)
                dots.add(dot)
            else:
                neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=YELLOW)
                neuron.move_to(gx.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
                neuron.shift(query_shift*gx.UP)
                hidden_layer.add(neuron)

        # Keys 
        query_shift=-3
        for i in range(HIDDEN_NEURONS):
            if i == 2:  # Middle position for ellipsis
                dot = math_tex("...").rotate(gx.PI/2, gx.OUT).move_to(gx.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
                dot.shift(query_shift*gx.UP)
                dots.add(dot)
            else:
                neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=BLUE)
                neuron.move_to(gx.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
                neuron.shift(query_shift*gx.UP)
                hidden_layer.add(neuron)

        connections = gx.VGroup()
        # Connect input to hidden layer
        for in_neuron in input_layer:
            for hidden_neuron in hidden_layer:
                start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
                line = gx.Line(start_point, end_point, stroke_opacity=0.3, stroke_width=1.5)
                connections.add(line)



        #Would like to make sure my hidden to output colored neurons do not violate causality!

        self.add(input_layer, hidden_layer, dots, connections)

        # self.set_view(0, 0, 0, (1.78, 2.79, 0.0), 10.05)
        self.set_view(0, 0, 0, (2.31, -0.03, 0.0), 9.07)
        self.wait()








class MLPAloneHacking(WelchScene):
    def construct(self):
        # Configuration
        INPUT_NEURONS = 5
        HIDDEN_NEURONS = 8
        OUTPUT_NEURONS = 5
        NEURON_RADIUS = 0.2
        LAYER_SPACING = 1.5
        VERTICAL_SPACING = 0.5
        
        # Create layers
        input_layer = gx.VGroup()
        hidden_layer = gx.VGroup()
        output_layer = gx.VGroup()
        dots = gx.VGroup()
        
        # Input layer
        for i in range(INPUT_NEURONS):
            if i == 2:  # Middle position for ellipsis
                dot = math_tex("...").rotate(gx.PI/2, gx.OUT).move_to(gx.LEFT * LAYER_SPACING + gx.UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
                dots.add(dot)
            else:
                neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=gx.WHITE)
                neuron.move_to(gx.LEFT * LAYER_SPACING + gx.UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
                input_layer.add(neuron)
                
        # Hidden layer
        for i in range(HIDDEN_NEURONS):
            if i == 4:  # Middle position for ellipsis
                dot = math_tex("...").rotate(gx.PI/2, gx.OUT).move_to(gx.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
                dots.add(dot)
            else:
                neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=gx.WHITE)
                neuron.move_to(gx.UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
                hidden_layer.add(neuron)
                
        # Output layer
        for i in range(OUTPUT_NEURONS):
            if i == 2:  # Middle position for ellipsis
                dot = math_tex("...").rotate(gx.PI/2, gx.OUT).move_to(gx.RIGHT * LAYER_SPACING + gx.UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
                dots.add(dot)
            else:
                neuron = gx.Circle(radius=NEURON_RADIUS, stroke_color=gx.WHITE)
                neuron.move_to(gx.RIGHT * LAYER_SPACING + gx.UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
                output_layer.add(neuron)
                
        # Create connections with edge points
        connections = gx.VGroup()
        
        # Helper function to get edge points between two circles
        def get_edge_points(circle1, circle2):
            # Get direction vector from circle1 to circle2
            direction = circle2.get_center() - circle1.get_center()
            unit_vector = direction / np.linalg.norm(direction)
            
            # Calculate start and end points
            start_point = circle1.get_center() + unit_vector * NEURON_RADIUS
            end_point = circle2.get_center() - unit_vector * NEURON_RADIUS
            
            return start_point, end_point
        
        # Connect input to hidden layer
        for in_neuron in input_layer:
            for hidden_neuron in hidden_layer:
                start_point, end_point = get_edge_points(in_neuron, hidden_neuron)
                line = gx.Line(start_point, end_point, stroke_opacity=0.7)
                connections.add(line)
                
        # Connect hidden to output layer
        for hidden_neuron in hidden_layer:
            for out_neuron in output_layer:
                start_point, end_point = get_edge_points(hidden_neuron, out_neuron)
                line = gx.Line(start_point, end_point, stroke_opacity=0.7)
                connections.add(line)
                
        self.add(connections, input_layer, hidden_layer, output_layer, dots)
        # self.add(connections)
        # self.add(input_layer, hidden_layer, output_layer, dots)
        self.wait()





# class MLPAloneHacking(InteractiveScene):
#     def construct(self):

#         # Configuration
#         INPUT_NEURONS = 5
#         HIDDEN_NEURONS = 8
#         OUTPUT_NEURONS = 5
#         NEURON_RADIUS = 0.2
#         LAYER_SPACING = 4
#         VERTICAL_SPACING = 0.6

#         # Create layers
#         input_layer = VGroup()
#         hidden_layer = VGroup()
#         output_layer = VGroup()
#         dots=VGroup()

#         # Input layer
#         for i in range(INPUT_NEURONS):
#             if i == 2:  # Middle position for ellipsis
#                 dot = Tex("...").rotate(PI/2, OUT).move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#                 dots.add(dot)
#             else:
#                 neuron = Circle(radius=NEURON_RADIUS, stroke_color=WHITE)
#                 neuron.move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#                 input_layer.add(neuron)

#         # Hidden layer
#         for i in range(HIDDEN_NEURONS):
#             if i == 4:  # Middle position for ellipsis
#                 dot = Tex("...").rotate(PI/2, OUT).move_to(UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
#                 dots.add(dot)
#             else:
#                 neuron = Circle(radius=NEURON_RADIUS, color=WHITE)
#                 neuron.move_to(UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
#                 hidden_layer.add(neuron)

#         # Output layer
#         for i in range(OUTPUT_NEURONS):
#             if i == 2:  # Middle position for ellipsis
#                 dot = Tex("...").rotate(PI/2, OUT).move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#                 dots.add(dot)
#             else:
#                 neuron = Circle(radius=NEURON_RADIUS, color=WHITE)
#                 neuron.move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#                 output_layer.add(neuron)

#         # Create connections
#         connections = VGroup()
        
#         # Connect input to hidden layer
#         for in_neuron in input_layer:
#             for hidden_neuron in hidden_layer:
#                 line = Line(in_neuron.get_center(), hidden_neuron.get_center(), stroke_opacity=0.3)
#                 connections.add(line)

#         # Connect hidden to output layer
#         for hidden_neuron in hidden_layer:
#             for out_neuron in output_layer:
#                 line = Line(hidden_neuron.get_center(), out_neuron.get_center(), stroke_opacity=0.3)
#                 connections.add(line)

#         self.add(connections)
#         self.add(input_layer, hidden_layer, output_layer, dots)
#         self.wait()


# class MLPAloneHacking(InteractiveScene):
#     def construct(self):

#         # Configuration
#         INPUT_NEURONS = 5
#         HIDDEN_NEURONS = 8
#         OUTPUT_NEURONS = 5
#         NEURON_RADIUS = 0.2
#         LAYER_SPACING = 4
#         VERTICAL_SPACING = 0.6

#         # Create layers
#         input_layer = VGroup()
#         hidden_layer = VGroup()
#         output_layer = VGroup()
#         dots=VGroup()

#         # Input layer
#         for i in range(INPUT_NEURONS):
#             if i == 2:  # Middle position for ellipsis
#                 dot = Tex("...").rotate(PI/2, OUT).move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#                 dots.add(dot)
#             else:
#                 disk = Dot(ORIGIN, radius=NEURON_RADIUS, fill_color=BLUE, fill_opacity=1.0, opacity=1.0)
#                 disk.set_opacity(1.0)
#                 disk.move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING)+0.1*OUT)
#                 input_layer.add(disk)

#                 # neuron = Circle(radius=NEURON_RADIUS, stroke_color=WHITE)
#                 # neuron.move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#                 # input_layer.add(neuron)

#         # Hidden layer
#         for i in range(HIDDEN_NEURONS):
#             if i == 4:  # Middle position for ellipsis
#                 dot = Tex("...").rotate(PI/2, OUT).move_to(UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
#                 dots.add(dot)
#             else:
#                 neuron = Circle(radius=NEURON_RADIUS, color=WHITE)
#                 neuron.move_to(UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
#                 hidden_layer.add(neuron)

#         # Output layer
#         for i in range(OUTPUT_NEURONS):
#             if i == 2:  # Middle position for ellipsis
#                 dot = Tex("...").rotate(PI/2, OUT).move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#                 dots.add(dot)
#             else:
#                 neuron = Circle(radius=NEURON_RADIUS, color=WHITE)
#                 neuron.move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#                 output_layer.add(neuron)

#         # Create connections
#         connections = VGroup()
        
#         # Connect input to hidden layer
#         for in_neuron in input_layer:
#             for hidden_neuron in hidden_layer:
#                 line = Line(in_neuron.get_center()-0.1*OUT, hidden_neuron.get_center(), stroke_opacity=0.3)
#                 connections.add(line)

#         # Connect hidden to output layer
#         for hidden_neuron in hidden_layer:
#             for out_neuron in output_layer:
#                 line = Line(hidden_neuron.get_center(), out_neuron.get_center(), stroke_opacity=0.3)
#                 connections.add(line)

#         self.add(connections)
#         self.add(input_layer, hidden_layer, output_layer, dots)
#         self.wait()




















