from welch_paths import welch_path
import manimgx as m
from welch_geometry import diagram_axes_3d
from welch_surface import SampledSurface
import sys
from welch_animation import TransformByGlyphMap
from welch_scene import WelchScene, image_at_height
from welch_text import math_tex
import numpy as np
import glob
import torch

sys.path.append('_2025/backprop_3')
from geometric_dl_utils import BaarleNet

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00aeef' ##00FFFF'
HOT_PINK='#FF00FF'

svg_path=welch_path('backprop_3/graphics/to_manim')
# graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/' #Point to folder where map images are

graphics_dir=welch_path('ai_book/4_deep_learning/graphics/')#Point to folder where map images are
map_filename='baarle_hertog_maps-13.png'


class LinearPlane(SampledSurface):
    """A plane defined by z = m1*x1 + m2*x2 + b"""
    def __init__(self, axes, m1=0.5, m2=0.3, b=1.0, vertical_viz_scale=0.5, **kwargs):
        self.axes = axes
        self.m1 = m1
        self.m2 = m2 
        self.b = b
        self.vertical_viz_scale=vertical_viz_scale
        super().__init__(
            # u_range=(-12, 12),
            # v_range=(-12, 12),
            u_range=(-0.7, 0.7),
            v_range=(-0.7, 0.7),
            resolution=(64, 64), #Looks nice at 256, but is slow, maybe crank for final
            color='#00FFFF',
            **kwargs
        )
    
    def uv_func(self, u, v):
        # u maps to x1, v maps to x2, compute z = m1*x1 + m2*x2 + b
        x1 = u
        x2 = v
        z = self.vertical_viz_scale*(self.m1 * x1 + self.m2 * x2 + self.b)
        # Transform to axes coordinate system
        return self.axes.c2p(x1, x2, z)

class LinearPlaneWithGrid(m.Group):
    """A plane with explicit grid lines"""
    def __init__(self, axes, m1=0.5, m2=0.3, b=1.0, vertical_viz_scale=0.5, 
                 grid_lines=20, **kwargs):
        super().__init__()
        
        # Create the main surface
        plane = LinearPlane(axes, m1, m2, b, vertical_viz_scale, **kwargs)
        self.add(plane)
        
        # Create grid lines
        u_range = (-0.7, 0.7)
        v_range = (-0.7, 0.7)
        
        # Vertical grid lines (constant u)
        for i in range(grid_lines + 1):
            u = u_range[0] + i * (u_range[1] - u_range[0]) / grid_lines
            line_points = []
            for j in range(21):  # 21 points along the line
                v = v_range[0] + j * (v_range[1] - v_range[0]) / 20
                x1, x2 = u, v
                z = vertical_viz_scale * (m1 * x1 + m2 * x2 + b)
                line_points.append(axes.c2p(x1, x2, z))
            
            grid_line = m.VMobject()
            grid_line.set_points_as_corners(line_points)
            grid_line.set_stroke(m.WHITE, width=1.5, opacity=0.9)
            self.add(grid_line)
        
        # Horizontal grid lines (constant v)
        for i in range(grid_lines + 1):
            v = v_range[0] + i * (v_range[1] - v_range[0]) / grid_lines
            line_points = []
            for j in range(21):  # 21 points along the line
                u = u_range[0] + j * (u_range[1] - u_range[0]) / 20
                x1, x2 = u, v
                z = vertical_viz_scale * (m1 * x1 + m2 * x2 + b)
                line_points.append(axes.c2p(x1, x2, z))
            
            grid_line = m.VMobject()
            grid_line.set_points_as_corners(line_points)
            grid_line.set_stroke(m.WHITE, width=1.5, opacity=0.9)
            self.add(grid_line)


#Ok pretty sure that network and plane need to be separte and brought together in editing. 
class p18a(WelchScene):
    def construct(self):

        model_path='_2025/backprop_3/models/2_1.pth'
        model = BaarleNet([2])
        model.load_state_dict(torch.load(model_path))


        map_img=image_at_height(graphics_dir+'/baarle_hertog_maps/'+map_filename)
        map_img.move_to(m.ORIGIN)
        

        net=m.VGroup()
        for p in sorted(glob.glob(svg_path+'/p18_21_to_manim*.svg')):
            net.add(m.SVGMobject(p)[1:])

        map_frame=net[1]
        layer_labels=net[4]
        input_circles=net[6]
        neuron_11_shape=net[7]
        neuron_12_shape=net[8]
        neuron_21_shape=net[9]
        neuron_22_shape=net[10]

        #Alright let me get the variables in place here, then I can start thinking through animations. 
        x1=math_tex('x_1', font_size=10).set_color(CHILL_BROWN)
        x1.move_to([0,0.115,0])
        x2=math_tex('x_2', font_size=10).set_color(CHILL_BROWN)
        x2.move_to([0,-0.21,0])

        #First layer
        # m11_1=Tex(r'\boldsymbol{m^{(1)}_{11}}', font_size=6).set_color(CYAN)
        m11_1=math_tex(r'm^{(1)}_{11}', font_size=6).set_color(CYAN)
        m11_1.move_to([0.27, 0.13, 0])
        # m12_1=Tex(r'\boldsymbol{m^{(1)}_{11}}', font_size=6).set_color(CYAN)
        m12_1=math_tex(r'm^{(1)}_{12}', font_size=6).set_color(CYAN)
        m12_1.move_to([0.31, 0.031, 0])

        m21_1=math_tex(r'm^{(1)}_{21}', font_size=6).set_color(YELLOW)
        m21_1.move_to([0.33, -0.08, 0])

        m22_1=math_tex(r'm^{(1)}_{22}', font_size=6).set_color(YELLOW)
        m22_1.move_to([0.27, -0.20, 0])

        b1_1=math_tex(r'+b^{(1)}_{1}', font_size=7).set_color(CYAN)
        b1_1.move_to([0.55, 0.13, 0])

        b2_1=math_tex(r'+b^{(1)}_{2}', font_size=7).set_color(YELLOW)
        b2_1.move_to([0.55, -0.20, 0])

        #Second layer
        horizontal_offset=0.55
        m11_2=math_tex(r'm^{(2)}_{11}', font_size=6).set_color(CYAN)
        m11_2.move_to([0.27+horizontal_offset, 0.13, 0])

        m12_2=math_tex(r'm^{(2)}_{12}', font_size=6).set_color(CYAN)
        m12_2.move_to([0.315+horizontal_offset, 0.031, 0])

        m21_2=math_tex(r'm^{(2)}_{21}', font_size=6).set_color(YELLOW)
        m21_2.move_to([0.33+horizontal_offset, -0.08, 0])

        m22_2=math_tex(r'm^{(2)}_{22}', font_size=6).set_color(YELLOW)
        m22_2.move_to([0.275+horizontal_offset, -0.20, 0])

        b1_2=math_tex(r'+b^{(2)}_{1}', font_size=7).set_color(CYAN)
        b1_2.move_to([0.55+horizontal_offset, 0.13, 0])

        b2_2=math_tex(r'+b^{(2)}_{2}', font_size=7).set_color(YELLOW)
        b2_2.move_to([0.55+horizontal_offset, -0.20, 0])

        # self.add(input_circles, neuron_11_shape, neuron_12_shape, neuron_21_shape, neuron_22_shape, layer_labels)
        inputs_group=m.Group(input_circles, x1, x2)
        neuron_11_group=m.Group(neuron_11_shape, m11_1, m12_1, b1_1)
        neuron_12_group=m.Group(neuron_12_shape, m21_1, m22_1, b2_1)
        neuron_21_group=m.Group(neuron_21_shape, m11_2, m12_2, b1_2)
        neuron_22_group=m.Group(neuron_22_shape, m21_2, m22_2, b2_2)

        # self.set_view(0, 0, 0, (0.48, -0.05, 0.0), 1.99)
        # Ok so I'll so smooth handoff from illustrator to this - with the network towards the bottom of the screen
        # I'll add in the 3d plane from 18b above in editing
        # self.add(x1, x2, m11_1, m12_1, m21_1, m22_1, b1_1, b2_1)
        # self.add(m11_2, m12_2, m21_2, m22_2, b1_2, b2_2)
        self.set_view(0, 0, 0, (0.52, 0.38, 0.0), 2.00)
        self.add(layer_labels)
        self.add(inputs_group, neuron_11_group, neuron_12_group, neuron_21_group, neuron_22_group)
        self.wait()

        #Ok now lower opacity on everything except the first neuron!
        self.play(
                  # inputs_group.animate.set_opacity(0.3), 
                  neuron_12_group.animate.set_opacity(0.3),
                  neuron_21_group.animate.set_opacity(0.3),
                  neuron_22_group.animate.set_opacity(0.3),
                  layer_labels.animate.set_opacity(0.3), 
                    run_time=2)
        self.wait()

        #Now build out equation!
        eq1=math_tex(r'h_{1}^{(1)}=m_{11}^{(1)}x_{1}+m_{12}^{(1)}x_{2}+b_1^{(1)}', font_size=8).set_color(FRESH_TAN)
        eq1.move_to([1.2, 0.8, 0])
        eq1[6:12].set_color(CYAN) #m11_1
        eq1[15:21].set_color(CYAN) #m12_1
        eq1[24:].set_color(CYAN) #b_1

        self.wait()
        self.play(m.ReplacementTransform(x1.copy(), eq1[12:14]),
                  m.ReplacementTransform(x2.copy(), eq1[21:23]),
                  run_time=3 
                 )

        self.wait()
        self.play(m.ReplacementTransform(m11_1.copy(), eq1[6:12]),
                  m.ReplacementTransform(m12_1.copy(), eq1[15:21]),
                  run_time=3 
                 )
        self.wait()
        self.add(eq1[14])

        self.play(m.ReplacementTransform(b1_1.copy(), eq1[23:]),run_time=3)
        self.wait()

        # Ok sick, so we've got a basic equation, and the plane in frame, 
        # now it's time to add the map, right? Ok I think let's just scale it down and put it in lower left corner. 
        #Let's try it
        map_img.scale(0.22)
        map_img.move_to([-0.65, -0.04, 0])
        map_frame.scale(0.764)
    
        map_frame.move_to(map_img)
        map_frame.shift([0, 0.015, 0])
        self.play(m.FadeIn(map_img), m.FadeIn(map_frame))
        self.wait()

        #ok now we need our point on that map!
        map_coords=math_tex(r'(0.6, 0.4)', font_size=11).set_color('#FF00FF')
        map_pt=m.Dot(m.ORIGIN, radius=0.015).set_color('#FF00FF')
        map_pt.move_to([-0.39, 0.12, 0])
        map_coords.move_to([-0.4, 0.2, 0])
        self.add(map_pt, map_coords)

        #Ok pieces together next equation, then move things over
        #2.51*0.6+(-1.02)*0.4-1.24
        eq2=math_tex(r'h_{1}^{(1)}=(2.51)(0.6)+(-1.02)(0.4)+(-1.22)', font_size=6).set_color(FRESH_TAN)
        eq2.move_to([1.25, 0.65, 0])
        eq2[6:12].set_color(CYAN) #m11_1
        eq2[12:17].set_color('#FF00FF')
        eq2[18:25].set_color(CYAN) #m12_1
        eq2[25:30].set_color('#FF00FF')
        eq2[31:].set_color(CYAN) #b_1

        eq3=math_tex(r'=-0.14', font_size=8).set_color(CYAN)
        eq3.move_to([0.95, 0.53, 0])
        self.wait()

        eq1_copy=eq1.copy()
        self.play(m.ReplacementTransform(map_coords[1:4].copy(), eq2[13:16]), run_time=3)
        self.play(m.Transform(eq1_copy[6:12], eq2[6:12]), run_time=1.5)
        self.add(eq2[5], eq2[12:17])
        self.wait()

        self.play(m.ReplacementTransform(map_coords[5:8].copy(), eq2[26:29]), run_time=3)
        self.play(m.ReplacementTransform(eq1_copy[15:21], eq2[18:25]), run_time=1.5)
        self.add(eq2[17], eq2[25:30])
        self.wait()

        self.play(m.ReplacementTransform(eq1_copy[24:], eq2[31:]), run_time=1.5)
        self.add(eq2[30])     
        self.wait()   

        self.play(m.Write(eq3))
        self.wait()

        # Ok so at this point, I need to add the hieght of this to my plane p18b
        # So I think on p19 I bring back up the opacity of the last two neurons
        # And then do the collapsing trick, I think right?

        self.play(
                  neuron_21_group.animate.set_opacity(1.0),
                  neuron_22_group.animate.set_opacity(1.0),
                  run_time=2)
        self.wait()

        # Ok ok ok ok ok ok ok now how do I animate this collapsing thing?
        # Fade out middle layer, and then bring inputs over? That might kinda work?
        # Ok taking a break then will try that.

        self.remove(inputs_group); self.add(inputs_group)
        self.wait(0)
        self.play(m.FadeOut(neuron_12_group),
                  m.FadeOut(neuron_11_group),
                  m.FadeOut(layer_labels),
                  inputs_group.animate.move_to([0.55,-0.03,0]),
                  run_time=3)
        self.wait()

        #Ok now we'll fade things out here, I think uncollapse networks so I can show all symbols. 

        self.wait()
        self.remove(eq1, eq1_copy, eq2, eq3) #Can't seem to get a nice fade out on these - just remove I think!
        self.play(
                  # FadeIn(neuron_12_group), 
                  m.FadeIn(neuron_11_group),
                  neuron_12_group.animate.set_opacity(1.0),
                  m.FadeOut(map_img),
                  m.FadeOut(map_frame),
                  # eq3.animate.set_opacity(0.0),
                  # eq2.animate.set_opacity(0.0),
                  # eq1_copy.animate.set_opacity(0.0),
                  # eq1.animate.set_opacity(0.0),
                  m.FadeOut(map_pt),
                  m.FadeOut(map_coords),
                  inputs_group.animate.move_to([-7.40739175e-03, -3.67778116e-02, 0]),
                  run_time=3)
        self.wait()
        
        #Hmm that might actually be a nice clean breakpoint a next scene?

        self.wait(20)
        pass

class p20_21(WelchScene):
    def construct(self):

        model_path='_2025/backprop_3/models/2_1.pth'

        model = BaarleNet([2])
        model.load_state_dict(torch.load(model_path))


        map_img=image_at_height(graphics_dir+'/baarle_hertog_maps/'+map_filename)
        map_img.move_to(m.ORIGIN)
        

        net=m.VGroup()
        for p in sorted(glob.glob(svg_path+'/p18_21_to_manim*.svg')):
            net.add(m.SVGMobject(p)[1:])

        map_frame=net[1]
        layer_labels=net[4]
        input_circles=net[6]
        neuron_11_shape=net[7]
        neuron_12_shape=net[8]
        neuron_21_shape=net[9]
        neuron_22_shape=net[10]

        #Alright let me get the variables in place here, then I can start thinking through animations. 
        x1=math_tex('x_1', font_size=10).set_color(CHILL_BROWN)
        x1.move_to([0,0.115,0])
        x2=math_tex('x_2', font_size=10).set_color(CHILL_BROWN)
        x2.move_to([0,-0.21,0])

        #First layer
        # m11_1=Tex(r'\boldsymbol{m^{(1)}_{11}}', font_size=6).set_color(CYAN)
        m11_1=math_tex(r'm^{(1)}_{11}', font_size=6).set_color(CYAN)
        m11_1.move_to([0.27, 0.13, 0])
        # m12_1=Tex(r'\boldsymbol{m^{(1)}_{11}}', font_size=6).set_color(CYAN)
        m12_1=math_tex(r'm^{(1)}_{12}', font_size=6).set_color(CYAN)
        m12_1.move_to([0.31, 0.031, 0])

        m21_1=math_tex(r'm^{(1)}_{21}', font_size=6).set_color(YELLOW)
        m21_1.move_to([0.33, -0.08, 0])

        m22_1=math_tex(r'm^{(1)}_{22}', font_size=6).set_color(YELLOW)
        m22_1.move_to([0.27, -0.20, 0])

        b1_1=math_tex(r'+b^{(1)}_{1}', font_size=7).set_color(CYAN)
        b1_1.move_to([0.55, 0.13, 0])

        b2_1=math_tex(r'+b^{(1)}_{2}', font_size=7).set_color(YELLOW)
        b2_1.move_to([0.55, -0.20, 0])

        #Second layer
        horizontal_offset=0.55
        m11_2=math_tex(r'm^{(2)}_{11}', font_size=6).set_color(CYAN)
        m11_2.move_to([0.27+horizontal_offset, 0.13, 0])

        m12_2=math_tex(r'm^{(2)}_{12}', font_size=6).set_color(CYAN)
        m12_2.move_to([0.315+horizontal_offset, 0.031, 0])

        m21_2=math_tex(r'm^{(2)}_{21}', font_size=6).set_color(YELLOW)
        m21_2.move_to([0.33+horizontal_offset, -0.08, 0])

        m22_2=math_tex(r'm^{(2)}_{22}', font_size=6).set_color(YELLOW)
        m22_2.move_to([0.275+horizontal_offset, -0.20, 0])

        b1_2=math_tex(r'+b^{(2)}_{1}', font_size=7).set_color(CYAN)
        b1_2.move_to([0.55+horizontal_offset, 0.13, 0])

        b2_2=math_tex(r'+b^{(2)}_{2}', font_size=7).set_color(YELLOW)
        b2_2.move_to([0.55+horizontal_offset, -0.20, 0])
        
        

        # self.add(input_circles, neuron_11_shape, neuron_12_shape, neuron_21_shape, neuron_22_shape, layer_labels)
        inputs_group=m.Group(input_circles, x1, x2)
        neuron_11_group=m.Group(neuron_11_shape, m11_1, m12_1, b1_1)
        neuron_12_group=m.Group(neuron_12_shape, m21_1, m22_1, b2_1)
        neuron_21_group=m.Group(neuron_21_shape, m11_2, m12_2, b1_2)
        neuron_22_group=m.Group(neuron_22_shape, m21_2, m22_2, b2_2)

        # self.set_view(0, 0, 0, (0.48, -0.05, 0.0), 1.99)
        # Ok so I'll so smooth handoff from illustrator to this - with the network towards the bottom of the screen
        # I'll add in the 3d plane from 18b above in editing
        # self.add(x1, x2, m11_1, m12_1, m21_1, m22_1, b1_1, b2_1)
        # self.add(m11_2, m12_2, m21_2, m22_2, b1_2, b2_2)
        self.set_view(0, 0, 0, (0.52, 0.38, 0.0), 2.00)
        # self.add(layer_labels)
        self.add(inputs_group, neuron_11_group, neuron_12_group, neuron_21_group, neuron_22_group)
        self.wait()

        h1_2=math_tex(r'h^{(2)}_{1}', font_size=8).set_color(CHILL_BROWN)
        h1_2.move_to([1.43, 0.13, 0])
        h2_2=math_tex(r'h^{(2)}_{2}', font_size=8).set_color(CHILL_BROWN)
        h2_2.move_to([1.43, -0.2, 0])
        # self.add(h1_2, h2_2)

        self.wait()
        self.play(m.FadeIn(h1_2), m.FadeIn(h2_2),
                  m11_2.animate.set_color('#FF00FF'), m12_2.animate.set_color('#FF00FF'), b1_2.animate.set_color('#FF00FF'),
                  m21_2.animate.set_color(CHILL_GREEN), m22_2.animate.set_color(CHILL_GREEN), b2_2.animate.set_color(CHILL_GREEN))
        self.wait()



        # Ok now let's build these equqations out here!
        # Hmm do I want to consider a bit of a color change here? 
        # like make all the neurons match?? 

        eq1 = math_tex(r"h_{1}^{(1)} = m_{11}^{(1)} x_{1} + m_{12}^{(1)} x_{2} + b_{1}^{(1)}", font_size=7).set_color(FRESH_TAN)
        eq2 = math_tex(r"h_{2}^{(1)} = m_{21}^{(1)} x_{1} + m_{22}^{(1)} x_{2} + b_{2}^{(1)}", font_size=7).set_color(FRESH_TAN)
        eq3 = math_tex(r"h_{1}^{(2)} = m_{11}^{(2)} h_{1}^{(1)} + m_{12}^{(2)} h_{2}^{(1)} + b_{1}^{(2)}", font_size=7).set_color(FRESH_TAN)
        
        
        
        # eq3[12:17].set_color(CYAN)
        # eq3[24:29].set_color(YELLOW)

        # eq1.move_to([0.55, 1.1, 0])
        # eq2.move_to([0.55, 0.9, 0])
        # eq3.move_to([0.55, 0.7, 0])

        eq1.move_to([0.27, 0.35, 0])
        eq2.move_to([0.27, -0.42, 0])
        eq3.move_to([1.3, 0.35, 0])
        
        eq3[12:17].set_color(CYAN)
        eq3[24:29].set_color(YELLOW)

        eq1[:5].set_color(CYAN)
        eq1[6:12].set_color(CYAN) #m11_1
        eq1[15:21].set_color(CYAN) #m12_1
        eq1[24:].set_color(CYAN) #b_1

        eq2[:5].set_color(YELLOW)
        eq2[6:12].set_color(YELLOW) #m11_1
        eq2[15:21].set_color(YELLOW) #m12_1
        eq2[24:].set_color(YELLOW) #b_1

        # eq3[:5].set_color('#FF00FF')
        eq3[6:12].set_color('#FF00FF') #m11_1
        eq3[18:24].set_color('#FF00FF') #m12_1
        eq3[30:].set_color('#FF00FF') #b_1

        # self.add(eq1, eq2, eq3)
        self.wait()

        #Ok now build equations from copies of values from neuron drawing!
        self.play(m.ReplacementTransform(m11_1.copy(), eq1[6:12]),
                  m.ReplacementTransform(m12_1.copy(), eq1[15:21]),
                  m.ReplacementTransform(b1_1.copy(), eq1[23:]),
                  run_time=3)
        self.add(eq1)
        self.play(m.ReplacementTransform(m21_1.copy(), eq2[6:12]),
                  m.ReplacementTransform(m22_1.copy(), eq2[15:21]),
                  m.ReplacementTransform(b2_1.copy(), eq2[23:]),
                  run_time=3)
        self.add(eq2)
        self.wait()
        self.play(m.ReplacementTransform(m11_2.copy(), eq3[6:12]),
                  m.ReplacementTransform(m12_2.copy(), eq3[18:24]),
                  m.ReplacementTransform(b1_2.copy(), eq3[29:]),
                  run_time=3)
        self.add(eq3)
        self.wait()

        #Move top and center so we can start working through substitutions! Handoff to Pranav
        self.play(eq3.animate.move_to([0.55, 1.2, 0]).scale(1.1), run_time=2)

        '''eq4 = Tex(
            r"h_{1}^{(2)} = m_{11}^{(2)}\bigg( m_{11}^{(1)}x_{1}+m_{12}^{(1)}x_{2}+b_1^{(1)} \bigg) "
            r"+ m_{12}^{(2)}\bigg( m_{21}^{(1)}x_{1}+m_{22}^{(1)}x_{2}+b_2^{(1)} \bigg) + b_1^{(2)}", font_size=10
        ).move_to(eq3.get_center())'''
        
        eq4 = math_tex(
            r"h_{1}^{(2)} = m_{11}^{(2)}\bigg( m_{11}^{(1)}x_{1}+m_{12}^{(1)}x_{2}+b_1^{(1)} \bigg) "
            r"+ m_{12}^{(2)} h_{2}^{(1)} + b_{1}^{(2)}", font_size =7
        ).move_to(eq3).set_color(FRESH_TAN).shift(m.DOWN*0.2)
        
        eq4[6:12].set_color(HOT_PINK)
        eq4[13:36].set_color(CYAN)
        eq4[44:49].set_color(YELLOW)
        eq4[38:44].set_color(HOT_PINK)
        eq4[50:55].set_color(HOT_PINK)
        
        eq5 = math_tex(
            r"h_{1}^{(2)} = m_{11}^{(2)}\bigg( m_{11}^{(1)}x_{1}+m_{12}^{(1)}x_{2}+b_1^{(1)} \bigg) "
            r"+ m_{12}^{(2)}\bigg( m_{21}^{(1)}x_{1}+m_{22}^{(1)}x_{2}+b_2^{(1)} \bigg) + b_1^{(2)}", font_size=7
        ).move_to(eq4).set_color(FRESH_TAN)
        
        eq5[6:12].set_color(HOT_PINK)
        eq5[13:36].set_color(CYAN)
        eq5[38:44].set_color(HOT_PINK)
        eq5[45:68].set_color(YELLOW)
        eq5[70:75].set_color(HOT_PINK)
        
        
        
        eq6 = eq5.copy()
        
        

        self.wait()
        
        
        
        
        self.play(
            m.ReplacementTransform(eq3[0:12].copy(), eq4[0:12]),
            m.ReplacementTransform(eq3[17:35].copy(), eq4[37:55]),
            m.ReplacementTransform(eq1[6:29].copy(), eq4[13:36]),
            run_time=3
            
        )
        
        self.add(m.VGroup(eq4[12], eq4[36]))
        
        self.wait()
        
        self.remove(eq4[44:49])
        
        
        self.play(
            m.ReplacementTransform(eq4[0:44], eq5[0:44]),
            m.ReplacementTransform(eq4[49:55], eq5[69:75]),
            m.ReplacementTransform(eq2[6:29].copy(), eq5[45:68]),
            run_time=3
        )
        
        self.add(m.VGroup(eq5[44], eq5[68]))
        
        self.play(m.FadeIn(eq6),
                  eq6.animate.shift(m.DOWN*0.2), run_time=2)
        
        eq7 = math_tex(
            r"h_{1}^{(2)} = m_{11}^{(2)} m_{11}^{(1)} x_{1} + m_{11}^{(2)} m_{12}^{(1)} x_{2} + m_{11}^{(2)} b_1^{(1)} "
            r"+ m_{12}^{(2)}\bigg( m_{21}^{(1)}x_{1}+m_{22}^{(1)}x_{2}+b_2^{(1)} \bigg) + b_1^{(2)}", font_size=7
        ).move_to(eq6).set_color(FRESH_TAN)
        
        eq7[6:12].set_color(HOT_PINK)
        eq7[12:20].set_color(CYAN)
        
        eq7[21:27].set_color(HOT_PINK)
        eq7[27:35].set_color(CYAN)
        
        eq7[36:42].set_color(HOT_PINK)
        eq7[42:47].set_color(CYAN)
        
        eq7[48:54].set_color(HOT_PINK)
        
        eq7[55:78].set_color(YELLOW)
        
        eq7[80:85].set_color(HOT_PINK)
        
        eq8 = math_tex(
            r"h_{1}^{(2)} = m_{11}^{(2)} m_{11}^{(1)} x_{1} + m_{11}^{(2)} m_{12}^{(1)} x_{2} + m_{11}^{(2)} b_1^{(1)} "
            r"+ m_{12}^{(2)} m_{21}^{(1)} x_{1} + m_{12}^{(2)} m_{22}^{(1)} x_{2} + m_{12}^{(2)} b_2^{(1)} + b_1^{(2)}", font_size=7
        ).move_to(eq7).set_color(FRESH_TAN)
        
        
        eq8[6:12].set_color(HOT_PINK)
        eq8[12:20].set_color(CYAN)
    
        eq8[21:27].set_color(HOT_PINK)
        eq8[27:35].set_color(CYAN)
    
        eq8[36:42].set_color(HOT_PINK)
        eq8[42:47].set_color(CYAN)
    
        eq8[48:54].set_color(HOT_PINK)
        eq8[54:62].set_color(YELLOW)
        
        eq8[63:69].set_color(HOT_PINK)
        eq8[69:77].set_color(YELLOW)
        
        eq8[78:84].set_color(HOT_PINK)
        eq8[84:89].set_color(YELLOW)
        
        eq8[90:95].set_color(HOT_PINK)
        
        eq9 = math_tex(
            r"h_{1}^{(2)} = m_{11}^{(2)} m_{11}^{(1)} x_{1} + m_{12}^{(2)} m_{21}^{(1)} x_{1} + m_{11}^{(2)} m_{12}^{(1)} x_{2} + m_{12}^{(2)} m_{22}^{(1)} x_{2} + m_{11}^{(2)} b_1^{(1)} + m_{12}^{(2)} b_2^{(1)} + b_1^{(2)}", font_size=7
        ).move_to(eq8).set_color(FRESH_TAN)
        
        eq9[6:12].set_color(HOT_PINK)
        eq9[12:20].set_color(CYAN)
        
        eq9[21:27].set_color(HOT_PINK)
        eq9[27:35].set_color(YELLOW)
        
        eq9[36:42].set_color(HOT_PINK)
        eq9[42:50].set_color(CYAN)
        
        eq9[51:57].set_color(HOT_PINK)
        eq9[57:65].set_color(YELLOW)
        
        eq9[78:84].set_color(HOT_PINK)
        eq9[84:89].set_color(YELLOW)
        
        eq9[66:72].set_color(HOT_PINK)
        eq9[72:77].set_color(CYAN)
        
        eq9[90:95].set_color(HOT_PINK)
        
        eq10 = math_tex(
            r"h_{1}^{(2)} = \bigg(m_{11}^{(2)} m_{11}^{(1)} + m_{12}^{(2)} m_{21}^{(1)}\bigg) x_{1} "
            r"+ \bigg(m_{11}^{(2)} m_{12}^{(1)} + m_{12}^{(2)} m_{22}^{(1)}\bigg) x_{2} "
            r"+ \bigg(m_{11}^{(2)} b_1^{(1)} + m_{12}^{(2)} b_2^{(1)} + b_1^{(2)}\bigg)", font_size=7
        ).move_to(eq9).set_color(FRESH_TAN)
        
        eq10[7:13].set_color(HOT_PINK)
        eq10[13:19].set_color(CYAN)
        
        eq10[20:26].set_color(HOT_PINK)
        eq10[26:32].set_color(YELLOW)
        
        eq10[37:43].set_color(HOT_PINK)
        eq10[43:49].set_color(CYAN)
        
        eq10[50:56].set_color(HOT_PINK)
        eq10[56:62].set_color(YELLOW)
        
        eq10[67:73].set_color(HOT_PINK)
        eq10[73:78].set_color(CYAN)
        
        eq10[79:85].set_color(HOT_PINK)
        eq10[85:90].set_color(YELLOW)
        
        eq10[91:96].set_color(HOT_PINK)
        
        eq11 = math_tex(
            r"h_{1}^{(2)} =  m_{1}  x_{1} "
            r"+  m_{2}  x_{2} "
            r"+  b_{1} ", font_size=10
        ).move_to(eq10).set_color(FRESH_TAN).shift(m.DOWN*0.25)
        
        self.play(
            TransformByGlyphMap(eq6, eq7,
                (list(range(37, 75)), list(range(47, 85))),
                (list(range(0, 6)), list(range(0, 6))),
                ([12, 36],  [], {"run_time":0.0000001}),
                ([21], [20]),
                ([30], [35]),
                (list(range(13, 21)), list(range(12, 20))),
                (list(range(22, 30)), list(range(27, 35))),
                (list(range(31, 36)), list(range(42, 47))),
                (list(range(6, 12)), list(range(6, 12))), # basically no point to put an arc on this because it is in the same spot
                (list(range(6, 12)), list(range(21, 27)), {"path_arc":-2/3*m.PI}),
                (list(range(6, 12)), list(range(36, 42)), {"path_arc":-1/3*m.PI}),
            ), run_time=3
        )
        
        self.wait()
        
        self.play(
            TransformByGlyphMap(eq7, eq8,
                (list(range(0, 48)), list(range(0, 48))),
                ([54, 78], [], {"run_time":0.0000001}),
                ([63], [62]),
                ([72], [77]),
                (list(range(55, 63)), list(range(54, 62))),
                (list(range(64, 72)), list(range(69, 77))),
                (list(range(73, 78)), list(range(84, 89))),
                (list(range(48, 54)), list(range(48, 54))), # basically no point to put an arc on this because it is in the same spot
                (list(range(48, 54)), list(range(63, 69)), {"path_arc":-2/3*m.PI}),
                (list(range(48, 54)), list(range(78, 84)), {"path_arc":-1/3*m.PI}),
            ),
            run_time=3
        )
        
        self.wait()
        

        self.play(
            TransformByGlyphMap(eq8, eq9,
                (list(range(0, 6)), list(range(0, 6))),
                ([20], [20]),
                ([35], [35]),
                ([47], [50]),
                ([62], [65]),
                ([77], [77]),
                ([89], [89]),
                (list(range(78, 89)), list(range(78, 89))),
                (list(range(90, 95)), list(range(90, 95))),
                (list(range(6, 20)), list(range(6, 20))),
                (list(range(36, 47)), list(range(66, 77)), {"path_arc":2/3*m.PI}),
                (list(range(21, 35)), list(range(36, 50)), {"path_arc":1/3*m.PI}),
                (list(range(48, 62)), list(range(21, 35)), {"path_arc":-2/3*m.PI}),
                (list(range(63, 77)), list(range(51, 65)), {"path_arc":-1/3*m.PI}),
            ),
            run_time=3
        )
        
        self.wait()

        eq10[6].set_color(m.BLACK)
        eq10[32].set_color(m.BLACK)
        eq10[36].set_color(m.BLACK)
        eq10[62].set_color(m.BLACK)
        eq10[66].set_color(m.BLACK)
        eq10[96].set_color(m.BLACK)
        
        self.play(
            TransformByGlyphMap(eq9, eq10,
                (list(range(0, 6)), list(range(0, 6))),
                (list(range(6, 18)), list(range(7, 19))),
                (list(range(20, 33)), list(range(19, 32))),
                (list(range(36, 48)), list(range(37, 49))),
                (list(range(66, 95)), list(range(67, 96))),
                ([35], [35]),
                ([65], [65]),
                (m.FadeIn, [6, 32, 36, 62, 66, 96]),
                ([18, 19], [33, 34], {"path_arc":2/3*m.PI}),
                ([33, 34], [33, 34]),
                ([48, 49], [63, 64], {"path_arc":2/3*m.PI}),
                ([63, 64], [63, 64]),

                
                
            ), run_time=3
        )
        
        eq10[6].set_color(FRESH_TAN)
        eq10[32].set_color(FRESH_TAN)
        eq10[36].set_color(FRESH_TAN)
        eq10[62].set_color(FRESH_TAN)
        eq10[66].set_color(FRESH_TAN)
        eq10[96].set_color(FRESH_TAN)
        
        self.wait()
        
                
        self.play(
            m.ReplacementTransform(eq10[0:6].copy(), eq11[0:6]),
            run_time=1.5
        )
        self.play(
            m.ReplacementTransform(eq10[7:32].copy(), eq11[6:8]),
            m.ReplacementTransform(eq10[33:35].copy(), eq11[8:10]),
            run_time=2
        )
        self.play(
            m.ReplacementTransform(eq10[35].copy(), eq11[10]),
            m.ReplacementTransform(eq10[37:62].copy(), eq11[11:13]),
            m.ReplacementTransform(eq10[63:65].copy(), eq11[13:15]),
            run_time=2
        )
        self.play(
            m.ReplacementTransform(eq10[65].copy(), eq11[15]),
            m.ReplacementTransform(eq10[67:96].copy(), eq11[16:18]),
            run_time=2
        )


        #On last little shift to network in the enter and final equation smaller off to the side
        #Then bring in 3d planes in premiere. Bonus points for making h1 magenta. 
        self.wait()
        self.remove(eq10, eq9, eq8, eq7, eq6, eq5, eq4, eq3, eq2, eq1)
        self.wait()

        # self.remove(eq11)
        # self.add(eq11)

        # self.wait()
        # self.play(eq10.animate.set_opacity(0.0), 
        #           eq9.animate.set_opacity(0.0), 
        #           eq8.animate.set_opacity(0.0), 
        #           eq7.animate.set_opacity(0.0), 
        #           eq6.animate.set_opacity(0.0), 
        #           eq5.animate.set_opacity(0.0), 
        #           eq4.animate.set_opacity(0.0), 
        #           eq3.animate.set_opacity(0.0),
        #           eq2.animate.set_opacity(0.0),
        #           eq1.animate.set_opacity(0.0), run_time=2)
        # self.wait()

        # eq11.scale(0.7)
        # eq11.move_to([1.95, 0.7, 0])

        self.wait()
        self.play(self.animate_view(0, 0, 0, (0.61, 0.0, 0.0), 2.27),
                  eq11.animate.scale(0.7).move_to([1.95, 0.7, 0]),
                  # eq11[:5].animate.set_color(HOT_PINK),
                  run_time=3)
        eq11[:5].set_color(HOT_PINK)
        self.wait()
        
        self.wait(20)
        pass





class p18b(WelchScene):
    def construct(self):

        model_path='_2025/backprop_3/models/2_1.pth'
        model = BaarleNet([2])
        model.load_state_dict(torch.load(model_path))


        axes_1 = diagram_axes_3d(
            # x_range=[-15, 15, 1],
            # y_range=[-15, 15, 1],
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            z_range=[-1, 1, 1],
            width=4,
            height=4,
            depth=3,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":4,
                "tip_config": {"width":0.08, "length":0.08}
                }
        )

        w=model.model[0].weight.detach().numpy()
        b=model.model[0].bias.detach().numpy()
        vertical_viz_scale=0.3
        plane_1=LinearPlaneWithGrid(axes_1, w[0,1], w[0,0], b[0], vertical_viz_scale=vertical_viz_scale, grid_lines=12)
        plane_1.set_opacity(0.5)
        plane_1.set_color(CYAN)

        axis_and_plane_11=m.Group(axes_1, plane_1)

        x1=math_tex('x_1', font_size=52).set_color(FRESH_TAN)
        x1.rotate(90*m.DEGREES, [1,0,0])
        x1.next_to(axes_1[0].get_end(), buff=0.15)
        x2=math_tex('x_2', font_size=52).set_color(FRESH_TAN)
        x2.rotate(90*m.DEGREES, [1,0,0])
        x2.rotate(90*m.DEGREES, [0,0,1])
        x2.next_to(axes_1[1].get_end(), buff=0.15, direction=np.array([0,1,0]))

        h1=math_tex('h_1^{(1)}', font_size=42).set_color(CYAN)
        h1.rotate(90*m.DEGREES, [1,0,0])
        h1.rotate(45*m.DEGREES, [0,0,1])
        h1.next_to(axes_1[2].get_end(), buff=0.15, direction=np.array([0,0,1]))


        self.set_view(44, 73, 0, (0.14, 0.04, -0.1), 6.00)
        self.add(axis_and_plane_11)
        # self.add(x1, x2, h1)

        #Book render
        self.set_view(45, 64, 0, (0.15, -0.07, -0.48), 5.21)
        self.wait()









        #Add point, connecting lines and maybe a nice little twist zoom in animation
        dot_point = axes_1.c2p(0.6, 0.4, -0.25)
        magenta_dot = m.Sphere(radius=0.1, color='#FF00FF')
        magenta_dot.move_to(dot_point)

        line1_start = axes_1.c2p(0.6, 0, 0)
        line1_end = axes_1.c2p(0.6, 0.4, 0)
        dashed_line1 = m.DashedLine(line1_start, line1_end, color='#FF00FF', stroke_width=8)
        dashed_line1.rotate(90*m.DEGREES, [0,1,0])
        
        line2_start = axes_1.c2p(0.0, 0.4, 0)
        line2_end = axes_1.c2p(0.6, 0.4, 0)
        dashed_line2 = m.DashedLine(line2_start, line2_end, color='#FF00FF', stroke_width=8)
        dashed_line2.rotate(90*m.DEGREES, [1,0,0])

        line3_start = axes_1.c2p(0.6, 0.4, 0)
        line3_end = axes_1.c2p(0.6, 0.4, -0.25)
        dashed_line3 = m.DashedLine(line3_start, line3_end, color='#FF00FF', stroke_width=8)
        # dashed_line2.rotate(90*DEGREES, [1,0,0])

        # self.add(magenta_dot, dashed_line1, dashed_line2, dashed_line3)
        self.wait()
        self.play(self.animate_view(69, 68, 0, (0.14, 0.09, -0.06), 5.11),
                  m.FadeIn(dashed_line1), m.FadeIn(dashed_line2), m.FadeIn(dashed_line3), m.FadeIn(magenta_dot),
                 run_time=4.0)
        self.wait()


        self.wait()

        self.wait(20)
        pass


class p21_simple_planes(WelchScene):
    def construct(self):

        model_path='_2025/backprop_3/models/2_1.pth'
        model = BaarleNet([2])
        model.load_state_dict(torch.load(model_path))


        axes_1 = diagram_axes_3d(
            # x_range=[-15, 15, 1],
            # y_range=[-15, 15, 1],
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            z_range=[-1, 1, 1],
            width=4,
            height=4,
            depth=3,
            axis_config={
                "color": FRESH_TAN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":4,
                "tip_config": {"width":0.08, "length":0.08}
                }
        )

        w=model.model[0].weight.detach().numpy()
        b=model.model[0].bias.detach().numpy()
        vertical_viz_scale=0.3
        plane_1=LinearPlaneWithGrid(axes_1, w[0,1], w[0,0], b[0], vertical_viz_scale=vertical_viz_scale, grid_lines=12)
        plane_1.set_opacity(0.5)
        plane_1.set_color(CYAN)

        axis_and_plane_11=m.Group(axes_1, plane_1)

        x1=math_tex('x_1', font_size=52).set_color(FRESH_TAN)
        x1.rotate(90*m.DEGREES, [1,0,0])
        x1.next_to(axes_1[0].get_end(), buff=0.15)
        x2=math_tex('x_2', font_size=52).set_color(FRESH_TAN)
        x2.rotate(90*m.DEGREES, [1,0,0])
        x2.rotate(90*m.DEGREES, [0,0,1])
        x2.next_to(axes_1[1].get_end(), buff=0.15, direction=np.array([0,1,0]))

        h1=math_tex('h_1^{(1)}', font_size=42).set_color(CYAN)
        h1.rotate(90*m.DEGREES, [1,0,0])
        h1.rotate(45*m.DEGREES, [0,0,1])
        h1.next_to(axes_1[2].get_end(), buff=0.15, direction=np.array([0,0,1]))


        self.set_view(44, 73, 0, (0.14, 0.04, -0.1), 6.00)
        self.add(axis_and_plane_11)
        self.add(x1, x2, h1)
        self.wait()
        self.remove(plane_1, h1)

        h2=math_tex('h_2^{(1)}', font_size=42).set_color(YELLOW)
        h2.rotate(90*m.DEGREES, [1,0,0])
        h2.rotate(45*m.DEGREES, [0,0,1])
        h2.next_to(axes_1[2].get_end(), buff=0.15, direction=np.array([0,0,1]))

        w=model.model[0].weight.detach().numpy()
        b=model.model[0].bias.detach().numpy()
        vertical_viz_scale=0.2
        plane_2=LinearPlaneWithGrid(axes_1, w[1,1], w[1,0], b[1], vertical_viz_scale=vertical_viz_scale, grid_lines=12)
        plane_2.set_opacity(0.5)
        plane_2.set_color(YELLOW)

        self.set_view(45, 60, 0, (-0.0, 0.08, 0.12), 6.00)

        self.add(h2, plane_2)
        self.wait()
        self.remove(h2, plane_2)

        h3=math_tex('h_2^{(2)}', font_size=42).set_color('#FF00FF')
        h3.rotate(90*m.DEGREES, [1,0,0])
        h3.rotate(45*m.DEGREES, [0,0,1])
        h3.next_to(axes_1[2].get_end(), buff=0.15, direction=np.array([0,0,1]))

        w=model.model[2].weight.detach().numpy()
        b=model.model[2].bias.detach().numpy()
        vertical_viz_scale=0.2
        plane_3=LinearPlaneWithGrid(axes_1, w[0,1], w[0,0], b[0], vertical_viz_scale=vertical_viz_scale, grid_lines=12)
        plane_3.set_opacity(0.5)
        plane_3.set_color('#FF00FF')

        self.set_view(45, 60, 0, (-0.0, 0.08, 0.12), 6.00)

        self.add(h3, plane_3)
        self.wait()
        self.remove(h3, plane_3)


        h4=math_tex('h_2^{(2)}', font_size=42).set_color(CHILL_GREEN)
        h4.rotate(90*m.DEGREES, [1,0,0])
        h4.rotate(45*m.DEGREES, [0,0,1])
        h4.next_to(axes_1[2].get_end(), buff=0.15, direction=np.array([0,0,1]))

        w=model.model[2].weight.detach().numpy()
        b=model.model[2].bias.detach().numpy()
        vertical_viz_scale=0.2
        plane_4=LinearPlaneWithGrid(axes_1, w[1,1], w[1,0], b[1], vertical_viz_scale=vertical_viz_scale, grid_lines=12)
        plane_4.set_opacity(0.5)
        plane_4.set_color(CHILL_GREEN)

        self.set_view(45, 60, 0, (-0.0, 0.08, 0.12), 6.00)

        self.add(h4, plane_4)
        self.wait()


        self.wait()

        self.wait(20)
        pass


