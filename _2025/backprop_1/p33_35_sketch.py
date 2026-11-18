from welch_paths import welch_path
import manimgx as m
from welch_geometry import diagram_arrow
from landscape_geometry import get_pivot_and_scale
import numpy as np
from welch_axes import WelchXAxis, WelchYAxis
from welch_scene import WelchScene
from welch_surface import parametric_surface, set_surface_shading, textured_surface
from welch_text import math_tex

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

surf=np.load(welch_path('backpropagation/animation/p_24_28_losses_4.npy'))
xy=np.load(welch_path('backpropagation/animation/p_24_28_losses_4xy.npy'))
grads_1=np.load(welch_path('backpropagation/animation/p_33_35_grads_1_2.npy'))
grads_2=np.load(welch_path('backpropagation/animation/p_33_35_grads_2_2.npy'))
xy_grads=np.load(welch_path('backpropagation/animation/p_33_35_xy_2.npy'))


def param_surface(u, v):
    u_idx = np.abs(xy[0] - u).argmin()
    v_idx = np.abs(xy[1] - v).argmin()
    try:
        z = surf[u_idx, v_idx]
    except IndexError:
        z = 0
    return np.array([u, v, z])

def param_surface_scaled(u, v):
    u_idx = np.abs(xy[0] - u).argmin()
    v_idx = np.abs(xy[1] - v).argmin()
    try:
        z = 1.5*surf[u_idx, v_idx] #Scale for slightly more dramatic 3d viz. 
    except IndexError:
        z = 0
    return np.array([u, v, z])


def get_grads(u,v):
    u_idx = np.abs(xy_grads[0] - u).argmin()
    v_idx = np.abs(xy_grads[1] - v).argmin()
    try:
        z1 = grads_1[u_idx, v_idx]
    except IndexError:
        z1 = 0
    try:
        z2 = grads_2[u_idx, v_idx]
    except IndexError:
        z2 = 0
    return np.array([u, v, z1, z2])


def map_to_canvas(value, axis_min, axis_max, axis_end, axis_start=0):
    value_scaled=(value-axis_min)/(axis_max-axis_min)
    return (value_scaled+axis_start)*axis_end


class Dot3D(m.Sphere):
    def __init__(self, center=m.ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(center)

class P33v1(WelchScene):
    def construct(self):

        # Ok ok ok ok I need ot decide how automated vs manual I'm going to make this scene
        # Thinking about this scene more, I do think the "right triangle overhead view" of the gradient is going to be nice/important.
        #  
        # Ok ok ok ok ok I do think i want some more Meaat on p33-p35 → specifically it’s interesting and non-obvious that we can put 
        # each slope together in that overhead view and it will point us downhill. Worth thinking about that animation a bit I think → 
        # maybe an actually good use case for WelchAxes lol. And i can draw a connection to part 2 → “as we’ll see in part 2 it turns 
        # out that we can very efficiently estimate the slope of the curves without actually computing an points on them → and then 
        # maybe the little arrows on the curves move to the overhead view? or copies of them? Them more i think about this the more 
        # I think it should be manimy? I can draw the projections/cuts as nice blue/yellow lines in the overhead view too. 
        #
        # Alright kinda low on brain power here, but let me at least try to get the pieces together tonight
        # I do think it probably makes sense to at least try to get all 3 graphs on one canvas, so I can like smoothlly move arrows arround and stuff
        # I might need to go compute a bunch of gradients eh?

        # Original value = y_max=1.7 axis_length_on_canvas=3)
        x_axis_1=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
                            x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_1=WelchYAxis(y_min=0.3, y_max=2.2, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        axes_1=m.VGroup(x_axis_1, y_axis_1)

        points_1 = [param_surface(u, 0) for u in np.linspace(-1, 4, 128)]
        points_mapped=np.array(points_1)[:, (0,2,1)]
        points_mapped[:,0]=map_to_canvas(points_mapped[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        points_mapped[:,1]=map_to_canvas(points_mapped[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
        curve_1 = m.VMobject()
        curve_1.set_points_smoothly(points_mapped)
        curve_1.set_stroke(width=4, color=YELLOW, opacity=0.8)

        x_label_1 = math_tex(r'\theta_{1}', font_size=30).set_color(CHILL_BROWN)
        y_label_1 = math_tex('Loss', font_size=25).set_color(CHILL_BROWN)
        x_label_1.next_to(x_axis_1, m.RIGHT, buff=0.05)
        y_label_1.next_to(y_axis_1, m.UP, buff=0.05)

        # self.add(axes_1, curve_1)
        # self.add(x_label_1, y_label_1)

        # Ok, adding second curve should be pretty straightfoward, maybe I think about gradient a bit now?
        # Intuitively I think I need to go actually compuate all the gradients at each point and cache them? 
        # Let me go take a crack at that. 
        # Man I guess you could show the whole gradient field with this approach - that's kinda interesting
        # Not sure if that has a role in the video -> could be fun in the book!

        # I've created a crappy version of c2p I think! Should be able to roll in or absorb. 
        # Keep roling for now though, I can make it pretty after I decide if I like the pattern.

        #Hmm well this is complicated and not working lol - will pick back up in the morning. 
        p1_values=param_surface(0, 0)
        p1_values[0]=map_to_canvas(p1_values[0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        p1_values[1]=map_to_canvas(p1_values[2], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
        p1_values[2]=0

        p1=m.Dot(p1_values, radius=0.06, fill_color=YELLOW)
        # self.add(p1)

        g=get_grads(0,0)
        grad_viz_scale=1.25*abs(g[2]) #Maybe make arrow length proportional to gradient?
        p1_values_2=param_surface(0, 0)
        g_values=np.array([[p1_values_2[0], p1_values_2[2], 0],
                           [p1_values_2[0]+grad_viz_scale, p1_values_2[2]+grad_viz_scale*g[2]*0.6, 0]]) #Maybe I make arrow length proportional to slope or something?
        g_values[:,0]=map_to_canvas(g_values[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        g_values[:,1]=map_to_canvas(g_values[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        a1=diagram_arrow(start=g_values[0], end=g_values[1], fill_color=YELLOW, thickness=3.0, tip_width_ratio=5, buff=0)
        # self.add(a1)

        # panel_1_shift=[-5, 0.4, 0]
        panel_1=m.VGroup(axes_1, curve_1, x_label_1, y_label_1, p1, a1)
        # panel_1.shift(panel_1_shift)

        # curve_1.set_stroke(opacity=0.25) #Yeah so I a fade out/fade in will bascially be the optning animation for this scene?
        # self.add(panel_1)

        # Ok I might clean up/rewrite this scene, we'll see. 
        # I think it makes sense to to ahead and add panels 2 and 3, and then I can start thinking about how the
        # pieces fit together, and exactly what I want to show from there
        # This workflow feels a bit hacky/uncomfortable, but I do think I have a pretty good feeling of what I want
        # to show, so I'm just going to trust this new/different process. 

        x_axis_2=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
                            x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_2=WelchYAxis(y_min=0.3, y_max=2.2, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        axes_2=m.VGroup(x_axis_2, y_axis_2)


        points_2 = [param_surface(0, v) for v in np.linspace(-1, 4, 128)]
        points_mapped_2=np.array(points_2)[:, (1,2,0)]
        points_mapped_2[:,0]=map_to_canvas(points_mapped_2[:,0], axis_min=x_axis_2.x_min, 
                                         axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
        points_mapped_2[:,1]=map_to_canvas(points_mapped_2[:,1], axis_min=y_axis_2.y_min, 
                                         axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)
        curve_2 = m.VMobject()
        curve_2.set_points_smoothly(points_mapped_2)
        curve_2.set_stroke(width=4, color=BLUE, opacity=0.8)


        x_label_2 = math_tex(r'\theta_{2}', font_size=30).set_color(CHILL_BROWN)
        y_label_2 = math_tex('Loss', font_size=25).set_color(CHILL_BROWN)
        x_label_2.next_to(x_axis_2, m.RIGHT, buff=0.05)
        y_label_2.next_to(y_axis_2, m.UP, buff=0.05) #not sure I need this.

        p2_values=param_surface(0, 0)
        p2_values[0]=map_to_canvas(p2_values[0], axis_min=x_axis_2.x_min, 
                                         axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
        p2_values[1]=map_to_canvas(p2_values[2], axis_min=y_axis_2.y_min, 
                                         axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)
        p2_values[2]=0
        p2=m.Dot(p2_values, radius=0.06, fill_color=BLUE)

        g=get_grads(0,0)
        grad_viz_scale=1.25*abs(g[3]) #Maybe make arrow length proportional to gradient?
        p2_values_2=param_surface(0, 0)
        g_values_2=np.array([[p2_values_2[0], p2_values_2[2], 0],
                           [p2_values_2[0]+grad_viz_scale, p2_values_2[2]+grad_viz_scale*g[3]*1.0, 0]]) #Maybe I make arrow length proportional to slope or something?
        g_values_2[:,0]=map_to_canvas(g_values_2[:,0], axis_min=x_axis_2.x_min, 
                                         axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
        g_values_2[:,1]=map_to_canvas(g_values_2[:,1], axis_min=y_axis_2.y_min, 
                                         axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)

        a2=diagram_arrow(start=g_values_2[0], end=g_values_2[1], fill_color=BLUE, thickness=3.0, tip_width_ratio=5, buff=0)

        # panel_2_shift=[-5, -3.2, 0]
        panel_2=m.VGroup(axes_2, curve_2, x_label_2, p2, a2) #y_label_2
        # panel_2.shift(panel_2_shift)

        # curve_2.set_stroke(opacity=0.25) 
        # self.add(panel_2)


        ## Hmm man you know what I'm pretty srue I can't just have a 3d panel mixed in these
        ## 2d ones right? It will just have to the overhead view, unless I can find some way of like 
        ## rotating it
        ## But really, the move is probably just having separate scenes that 
        ## Oh man man is there some way to actually bring these 2d plots togetehr INTO the 3d one? 
        ## Obviously dont want to add a bunch of random scope ehre, but I do think that woudl show what we're 
        ## about here really nicely. Becuase we bring together the two curves into the 3d surface and it will 
        ## be nice an dclear that the gradient arrows are guiding us downhill. 
        ## I'm also a little confused by the uphill vs downhill thing. 
        ## 
        ## Drafting a Claude prompt, I think I'm starting to figure this out - not sure what it means for 
        ## visualization and this section yet...
        ## I'm confused about why the gradient points uphill in loss landsacpes - the partial derivates 
        ## that make up the gradient are equivalent to the slope at each point - if increasing some 
        ## parameter makes the loss go down, then the slope of the loss with respect to that parameter 
        ## should be negative.
        ##
        ## Ok so it's like the slope/gradient is negative from our starting point, which means if we 
        ## add the gradient to theta, since the gradient is negative we will go uphill. 
        ## That's the deal here -> how do we visualize what's going on?
        ## I guess like adding the gradient to the parameter is not actually that natural of a thing visually?
        ## I feel like it should be though -> it should be a little vector that guides us through parameter space...
        ## Hmmmm....
        ## 
        ## Ok I think i have a better sense of what's going on here - and in the script I'll say something 
        ## like that technically the gradient goes the other way, and we'll get into this in part 2. 
        ## 
        ## Ok there's some details etc that I need to change here
        ## But I have a pretty strong vision for what I think I want the core animation to be
        ## I think doing it like this is actually going to be more clear
        ## I want the two curves to "come together" into the surfce, and then really clearly show that the 
        ## overall gradient is the combination of the cyan and yellow arrows, it will be like nicely in the
        ## center fo thes two....
        ##
        ## So.. let me build out a workign sketch that core animation, and then we can take if from there.  
        ## 


        # TODO -> so I think for the opening of this animation, we opacify the curves, and move the gradient 
        # arrow around a little bit got give the viewer a sense for how it works...
        # Also I think we do add a little math delta notation here. 
        # And then at the end I see gradient descent happening, maybe all at once on all 3 panels...

        curve_1.set_stroke(opacity=0.5)
        curve_2.set_stroke(opacity=0.5)

        panel_2.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN) #Get into 3d space so I can do the bring together animation...
        panel_1.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        self.add(panel_1, panel_2)

        self.set_view(0, 89, 0, (-0.46, 0.0, 1.36), 8.97)

        panel_1_shift=[-5, 0, 2.0]
        panel_2_shift=[-5, 0, -2.0]
        panel_1.shift(panel_1_shift)
        panel_2.shift(panel_2_shift)

        self.wait()

        
        # self.remove(y_axis_2) #Probably fade this out during the animation. 
        
        r=panel_2.get_corner(m.LEFT+(m.config.frame_height / 2 * m.DOWN))
        r[0]=-4.15    
        self.wait()

        #Todo -> fade out tick numbers here?
        # self.play(FadeOut(y_axis_2),
        #           FadeOut(x_axis_1[-1]), #Tick labels
        #           FadeOut(x_axis_2[-1]),
        #           FadeOut(y_axis_1[-1]),
        #           FadeOut(x_axis_1[-2]), #Ticks
        #           FadeOut(x_axis_2[-2]),
        #           FadeOut(y_axis_1[-2]),
        #           run_time=1.0)

        self.play(y_axis_2.animate.set_opacity(0.0),
                  x_axis_1[-1].animate.set_opacity(0.0), #Tick labels
                  x_axis_2[-1].animate.set_opacity(0.0),
                  y_axis_1[-1].animate.set_opacity(0.0),
                  x_axis_1[-2].animate.set_opacity(0.0), #Ticks
                  x_axis_2[-2].animate.set_opacity(0.0),
                  y_axis_1[-2].animate.set_opacity(0.0),
                  run_time=1.0)


        self.play(panel_1.animate.shift([0, 0, -2.0]),
                  panel_2.animate.rotate(90*m.DEGREES, [0,0,1], about_point=r).shift([0, 0, 2.0]),
                  self.animate_view(13, 85, 0, (-1.88, -0.77, 1.56), 5.01),
                  run_time=4)

        

        self.wait()

        # So what's the deal from here, I think pan around while doing some level of axis tick fade out and
        # adding in the surface! Then in comes the joint arrow direction - right? 
        # Also I think wiggling/moving the gradient arrow around at the beginning/2d part would be good. 
        # I wonder if i can bring together copies both arrows together at the same time
        # to create the main magenta gradient vector?
        # Ok big importnat thing to show next here is bring in the surface of course. 
        # Let me do that next for the sketch, and then will come back and fill in details
        # Maybe after a pass at writing. 

        ## ---

        # Create main surface
        surface = parametric_surface(
            param_surface,  
            u_range=[-1, 4],
            v_range=[-1, 4],
            resolution=(256, 256),
        )

        ts = textured_surface(surface, welch_path('backpropagation/animation/p_24_28_losses_4.png'))
        set_surface_shading(ts, 0.0, 0.1, 0)
        

        pivot_x,scale_x=get_pivot_and_scale(axis_min=x_axis_1.x_min, axis_max=x_axis_1.x_max, 
                                        axis_end=x_axis_1.axis_length_on_canvas)
        pivot_y,scale_y=get_pivot_and_scale(axis_min=y_axis_1.y_min, axis_max=y_axis_1.y_max, 
                                        axis_end=y_axis_1.axis_length_on_canvas)
        ts.scale([scale_x, scale_x, scale_y], about_point=[pivot_x, pivot_x, pivot_y])
        surf_shift=[-3.8, 0.34, -0.3] #Gross iterative swagginess, I think i at least have the scale right
        ts.shift(surf_shift)

        ##Ok let's figure out if we want gridlines
        num_lines = 21  # Number of gridlines in each direction
        num_points = 256  # Number of points per line
        u_gridlines = m.VGroup()
        v_gridlines = m.VGroup()
        u_values = np.linspace(-1, 4, num_lines)
        v_points = np.linspace(-1, 4, num_points)
        for u in u_values:
            points = [param_surface(u, v) for v in v_points]
            line = m.VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.0)
            u_gridlines.add(line)
        u_points = np.linspace(-1, 4, num_points)
        for v in u_values:  # Using same number of lines for both directions
            points = [param_surface(u, v) for u in u_points]
            line = m.VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.0)
            v_gridlines.add(line)

        u_gridlines.scale([scale_x, scale_x, scale_y], about_point=[pivot_x, pivot_x, pivot_y])
        u_gridlines.shift(surf_shift)
        v_gridlines.scale([scale_x, scale_x, scale_y], about_point=[pivot_x, pivot_x, pivot_y])
        v_gridlines.shift(surf_shift)
        ts.set_opacity(0.0)

        self.add(ts, u_gridlines, v_gridlines) #Add with zero opacity
        self.remove(a1); self.add(a1) #Occlusions
        self.remove(a2); self.add(a2)
        self.remove(p1); self.add(p1)
        self.remove(p2); self.add(p2)

        self.play(ts.animate.set_opacity(0.5), 
                  p1.animate.set_opacity(0.0),
                  p2.animate.set_opacity(0.0),
                  a1.animate.rotate(-m.DEGREES*135, axis=a1.get_end()-a1.get_start()),
                  a2.animate.rotate(-m.DEGREES*80, axis=a2.get_end()-a2.get_start()),
                  u_gridlines.animate.set_stroke(opacity=0.14), 
                  v_gridlines.animate.set_stroke(opacity=0.14),
                  # self.animate_view(125, 57, 0, (-2.45, 1.36, 2.08), 1.21),
                  # self.animate_view(106, 41, 0, (-2.43, 0.92, 2.55), 3.11),
                  self.animate_view(124, 40, 0, (-2.57, 0.86, 2.7), 1.81),
                  run_time=4.0)

        self.wait()

        # a1.rotate(90*DEGREES, [1, 0, 0])

        ## Ooh could my 2 arrows "twist in and come together to make the gradient" -> that would be dope I think. 

        # Ok let's start with the ending magenta arrow, and then figure out how to animate 
        # each of my existing arrows into this one. 


        a3 =diagram_arrow(start=[a1.get_corner(m.LEFT)[0]+0.03, a1.get_corner(m.LEFT)[1]+0.01, a1.get_corner(m.OUT)[2]],
                  end=[a1.get_corner(m.RIGHT)[0], a2.get_corner(m.UP)[1], a2.get_corner(m.IN)[2]],
                  fill_color='#FF00FF', thickness=3.0, tip_width_ratio=5, buff=0)
        # a3 =Arrow(start=a1.start, end=a2.end, fill_color='#FF00FF', thickness=3.0, tip_width_ratio=5, buff=0)
        #going to need to do some interestign rotation stuff I think...
        #Hmm doesn't look like I can borrow a1.start etc, these musth not update
        # Ok what bout like getting corners etc from a1 and a2?
        # a1.rotate(-DEGREES*135, axis=a1.get_end()-a1.get_start())
        # a2.rotate(-DEGREES*80, axis=a2.get_end()-a2.get_start())
        self.wait()

        self.play(m.TransformFromCopy(a1, a3), #Ah that's fucking dope - how many more cool tricks does Grant have up his sleeve that I know nothing about lol.
                  m.TransformFromCopy(a2, a3),
                  run_time=3.0)
        self.wait()

        # self.play(FadeIn(a3))
        # self.add(a3)
        # self.remove(a3)

        ## Ok it's time for the final boss of this mother fucking scene. 
        ## Actually running grandient descent. 
        ## IN my head it would be dope to show it on all 3 panels at once
        ## Or I guess even have the option to show it maybe on a single pane the first time and 
        ## multiple panes the second time
        ## Either way, I'll render out the multipane version on 2 or 3 different separate panes
        ## And I'll start with big kahuna 3d version here
        ## I'm thinking I'm going to sort of fake it sort of not
        ## The camera view doesn't matter too much here - I think the main thing is 
        ## to get the descent animation working first, then come back and add in my camera moves
        ## Or kidna add them as I go. 

        # self.play(self.animate_view(169, 43, 0, (-3.56, 1.21, 1.79), 3.95), run_time=3.0)
        


        # Hmm Do I want to show the yellow and blue arrows at each step? I kinda feel like yeah 
        # If I'm feeling boojie maybe I even do the cool "animate them together" thing at each step
        # Finally, I think spheres are probably better than dots if I can swing em. 
        # 

        s1=Dot3D(center=a3.get_start(), radius=0.06, color='$FF00FF')
        s2=Dot3D(center=a3.get_end(), radius=0.06, color='$FF00FF')
        self.wait()

        self.play(a1.animate.set_opacity(0.0),
                  a2.animate.set_opacity(0.0),
                  curve_1.animate.set_opacity(0.0),
                  curve_2.animate.set_opacity(0.0),
                  m.FadeIn(s1),
                  m.FadeIn(s2),
                  self.animate_view(175, 47, 0, (-3.89, 1.49, 1.6), 3.75),
                  run_time=2.0)


        # s1=Dot3D(center=a3.get_start(), radius=0.06, color='$FF00FF')
        # self.add(s1)

        self.wait()

        # ok ok ok so I could do this kinda 3d naitively, or I could do it in 2d and rotate shit like I did 
        # the first time. Although the later is more clunky, I think it's actually better becuase when I 
        # go to make the 2d version I can reuse code and make sure that I'm just doing the exact same thing. 
        # Oof man thinking through this - it might get pretty gnarly. 
        # Ok here's an idea -> I know i have fiarly deterministic mapping from 2d panels to my 3d view
        # What about doing the full 2d version first, grouping all the assets with the panels, and then
        # moving them just like I did the first time?
        # Or maybe after I figure out the 2d version I just like come back and add the full gradient 
        # descent above, so I have all the compnent arrows ready to go? 
        # Ok let's try that. 






        pass
        self.wait(20)



class P34_2d_deprecated(WelchScene):
    def construct(self):
        '''
        Figure out 2d full gradient descent panels here, then combine with full method above to get
        3d grad descent - let's go! Maybe make a new class when I do the combining. 
        '''

        x_axis_1=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
                            x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_1=WelchYAxis(y_min=0.3, y_max=2.2, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        axes_1=m.VGroup(x_axis_1, y_axis_1)

        points_1 = [param_surface(u, 0) for u in np.linspace(-1, 4, 128)]
        points_mapped=np.array(points_1)[:, (0,2,1)]
        points_mapped[:,0]=map_to_canvas(points_mapped[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        points_mapped[:,1]=map_to_canvas(points_mapped[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
        curve_1 = m.VMobject()
        curve_1.set_points_smoothly(points_mapped)
        curve_1.set_stroke(width=4, color=YELLOW, opacity=0.8)

        x_label_1 = math_tex(r'\theta_{1}', font_size=30).set_color(CHILL_BROWN)
        y_label_1 = math_tex('Loss', font_size=25).set_color(CHILL_BROWN)
        x_label_1.next_to(x_axis_1, m.RIGHT, buff=0.05)
        y_label_1.next_to(y_axis_1, m.UP, buff=0.05)

        p1_values=param_surface(0, 0)
        p1_values[0]=map_to_canvas(p1_values[0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        p1_values[1]=map_to_canvas(p1_values[2], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
        p1_values[2]=0
        p1=m.Dot(p1_values, radius=0.06, fill_color=YELLOW)

        g=get_grads(0,0)
        grad_viz_scale=1.25*abs(g[2]) #Maybe make arrow length proportional to gradient?
        p1_values_2=param_surface(0, 0)
        g_values=np.array([[p1_values_2[0], p1_values_2[2], 0],
                           [p1_values_2[0]+grad_viz_scale, p1_values_2[2]+grad_viz_scale*g[2]*0.6, 0]]) #Maybe I make arrow length proportional to slope or something?
        g_values[:,0]=map_to_canvas(g_values[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        g_values[:,1]=map_to_canvas(g_values[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        a1=diagram_arrow(start=g_values[0], end=g_values[1], fill_color=YELLOW, thickness=3.0, tip_width_ratio=5, buff=0)
        panel_1=m.VGroup(axes_1, curve_1, x_label_1, y_label_1, p1, a1)


        x_axis_2=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
                            x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_2=WelchYAxis(y_min=0.3, y_max=2.2, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        axes_2=m.VGroup(x_axis_2, y_axis_2)


        points_2 = [param_surface(0, v) for v in np.linspace(-1, 4, 128)]
        points_mapped_2=np.array(points_2)[:, (1,2,0)]
        points_mapped_2[:,0]=map_to_canvas(points_mapped_2[:,0], axis_min=x_axis_2.x_min, 
                                         axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
        points_mapped_2[:,1]=map_to_canvas(points_mapped_2[:,1], axis_min=y_axis_2.y_min, 
                                         axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)
        curve_2 = m.VMobject()
        curve_2.set_points_smoothly(points_mapped_2)
        curve_2.set_stroke(width=4, color=BLUE, opacity=0.8)


        x_label_2 = math_tex(r'\theta_{2}', font_size=30).set_color(CHILL_BROWN)
        y_label_2 = math_tex('Loss', font_size=25).set_color(CHILL_BROWN)
        x_label_2.next_to(x_axis_2, m.RIGHT, buff=0.05)
        y_label_2.next_to(y_axis_2, m.UP, buff=0.05) #not sure I need this.

        p2_values=param_surface(0, 0)
        p2_values[0]=map_to_canvas(p2_values[0], axis_min=x_axis_2.x_min, 
                                         axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
        p2_values[1]=map_to_canvas(p2_values[2], axis_min=y_axis_2.y_min, 
                                         axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)
        p2_values[2]=0
        p2=m.Dot(p2_values, radius=0.06, fill_color=BLUE)

        g=get_grads(0,0)
        grad_viz_scale=1.25*abs(g[3]) #Maybe make arrow length proportional to gradient?
        p2_values_2=param_surface(0, 0)
        g_values_2=np.array([[p2_values_2[0], p2_values_2[2], 0],
                           [p2_values_2[0]+grad_viz_scale, p2_values_2[2]+grad_viz_scale*g[3]*1.0, 0]]) #Maybe I make arrow length proportional to slope or something?
        g_values_2[:,0]=map_to_canvas(g_values_2[:,0], axis_min=x_axis_2.x_min, 
                                         axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
        g_values_2[:,1]=map_to_canvas(g_values_2[:,1], axis_min=y_axis_2.y_min, 
                                         axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)

        a2=diagram_arrow(start=g_values_2[0], end=g_values_2[1], fill_color=BLUE, thickness=3.0, tip_width_ratio=5, buff=0)

        panel_2=m.VGroup(axes_2, curve_2, x_label_2, p2, a2)

        curve_1.set_stroke(opacity=0.5)
        curve_2.set_stroke(opacity=0.5)

        panel_2.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN) #Get into 3d space so I can do the bring together animation...
        panel_1.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        self.add(panel_1, panel_2)

        self.set_view(0, 89, 0, (-0.46, 0.0, 1.36), 8.97)

        panel_1_shift=[-5, 0, 2.0]
        panel_2_shift=[-5, 0, -2.0]
        panel_1.shift(panel_1_shift)
        panel_2.shift(panel_2_shift)

        self.wait()

        #Ok so where do my nexts points land in my original uv space??
        num_steps=7
        grad_adjustment_factors=[0.6,0.6,0.6,0.6,0.6,0.6,0.6] #Not sure why I need these. 
        descent_points_1=[] #Let's try computing all non-mapped points at first, then mapping them alls. 
        arrow_end_points_1=[]
        #Hmm this is kinda subtle, we always get 'snapped back to the curve, right?'
        #Could add my points 2 at a time, the "where the arrow is pointing one, and then the snapped back one?"

        p1_values_3=param_surface(0, 0)
        g=get_grads(0,0)
        grad_viz_scale=1.25*abs(g[2])
        # arrow_end_points_1.append([p1_values_2[0], p1_values_2[2], 0]) #First point
        descent_points_1.append([p1_values_3[0], p1_values_3[2], 0]) #First point

        for i in range(1, num_steps):
            g=get_grads(descent_points_1[i-1][0], 0) #Hmm I think I'm actually running two indpendend grad descents - might not matter? We'll see. 
            # print(g)
            grad_viz_scale=1.25*abs(g[2])
            new_x=descent_points_1[i-1][0]+grad_viz_scale
            arrow_end_points_1.append([new_x, descent_points_1[i-1][1]+grad_viz_scale*g[2]*grad_adjustment_factors[i], 0]) #End of tangent arrow
            descent_points_1.append([new_x, param_surface(new_x,0)[2],0]) #Next point on curve
        descent_points_1=np.array(descent_points_1) #Ok gut check on this array seems fine. 
        arrow_end_points_1=np.array(arrow_end_points_1)

        descent_points_1_mapped=np.zeros_like(descent_points_1)
        descent_points_1_mapped[:,0]=map_to_canvas(descent_points_1[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        descent_points_1_mapped[:,1]=map_to_canvas(descent_points_1[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        arrow_end_points_1_mapped=np.zeros_like(arrow_end_points_1)
        arrow_end_points_1_mapped[:,0]=map_to_canvas(arrow_end_points_1[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        arrow_end_points_1_mapped[:,1]=map_to_canvas(arrow_end_points_1[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        # self.wait()
        arrows_1=m.VGroup()
        points_1=m.VGroup()
        for i in range(num_steps-1):
            arrows_1.add(diagram_arrow(start=descent_points_1_mapped[i], end=arrow_end_points_1_mapped[i], fill_color=YELLOW,
                                 thickness=3.0, 
                                 tip_width_ratio=5, buff=0))
            points_1.add(m.Dot(descent_points_1_mapped[i], radius=0.06, fill_color=YELLOW))

        arrows_1.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        arrows_1.shift(panel_1_shift)
        points_1.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        points_1.shift(panel_1_shift)
        # self.add(arrows_1, points_1) #Ok this gets noisy visually, but I think if I add one arrow at a time, and keep all the 
        #                              #points basically it will be fine. 


        ## --- Ok now second panel --- ##
        grad_adjustment_factors=[1.0,1.0,1.0,1.0,1.0,1.0,1.0] #Not sure why I need these. 
        descent_points_2=[] #Let's try computing all non-mapped points at first, then mapping them alls. 
        arrow_end_points_2=[]
        #Hmm this is kinda subtle, we always get 'snapped back to the curve, right?'
        #Could add my points 2 at a time, the "where the arrow is pointing one, and then the snapped back one?"

        p1_values_3=param_surface(0, 0)
        g=get_grads(0,0)
        grad_viz_scale=1.25*abs(g[3])
        # arrow_end_points_2.append([p1_values_2[0], p1_values_2[2], 0]) #First point
        descent_points_2.append([p1_values_3[1], p1_values_3[2], 0]) #First point

        for i in range(1, num_steps):
            g=get_grads(0, descent_points_2[i-1][0])
            # print(g)
            grad_viz_scale=1.25*abs(g[3])
            new_x=descent_points_2[i-1][0]+grad_viz_scale
            arrow_end_points_2.append([new_x, descent_points_2[i-1][1]+grad_viz_scale*g[3]*grad_adjustment_factors[i], 0]) #End of tangent arrow
            descent_points_2.append([new_x, param_surface(0, new_x)[2],0]) #Next point on curve
        descent_points_2=np.array(descent_points_2) #Ok gut check on this array seems fine. 
        arrow_end_points_2=np.array(arrow_end_points_2)

        descent_points_2_mapped=np.zeros_like(descent_points_2)
        descent_points_2_mapped[:,0]=map_to_canvas(descent_points_2[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        descent_points_2_mapped[:,1]=map_to_canvas(descent_points_2[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        arrow_end_points_2_mapped=np.zeros_like(arrow_end_points_2)
        arrow_end_points_2_mapped[:,0]=map_to_canvas(arrow_end_points_2[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        arrow_end_points_2_mapped[:,1]=map_to_canvas(arrow_end_points_2[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        # self.wait()
        arrows_2=m.VGroup()
        points_2=m.VGroup()
        for i in range(num_steps-1):
            arrows_2.add(diagram_arrow(start=descent_points_2_mapped[i], end=arrow_end_points_2_mapped[i], fill_color=BLUE,
                                 thickness=3.0, 
                                 tip_width_ratio=5, buff=0))
            points_2.add(m.Dot(descent_points_2_mapped[i], radius=0.06, fill_color=BLUE))

        arrows_2.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        arrows_2.shift(panel_2_shift)
        points_2.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        points_2.shift(panel_2_shift)

        self.add(arrows_2, points_2)

        # Hmm man ok there is a problem here with this viz -> my curves should actually be changing at each iteration!!
        # Maybe we don't show the 3 panel thing? 
        # Might be something good to think about when rewriting this section. 
        # My intuition is that this really really won't matter for the big 3d version -> doing these independtly 
        # won't be a big deal. 
        # I am giving this some thought though, becuase a big point of the video is that these dimensions are not the same!
        # And that is a really important point for later. 
        # Hmm hmm hmm. 
        # So, on other thought here -> I don't think that actually doing 2d gradient (basically numerical) gradient descent
        # Should be that bad. 
        # yeah I would have to recomput the curve each time - that might not actually be that bad though. 
        # 
        # HEY MAYBE AFTER I ADD A POINT WITH GRADIENT DESECNT I SWAP OUT THE ARROW FOR A THINNER CONNECTING LINE?? THAT 
        # MIGHT WORK A BIG BETTER VISUALLY. 
        # 
        # Ok this is a tougher scene than exepcted - but it is an important part of the video - 
        # it really informs the whole thing and has some nice teaser components of what comes next
        # With all that in mind, I do think it makes sense to take a crack at full 2d numerical gradient descent and 
        # redrawing the curves. I dont' think it will actually be that bad. 



        ## ----- 





        self.wait()
        pass




        # panel_1=VGroup(axes_1, curve_1, x_label_1, y_label_1, p1, a1)

        # panel_2.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN) #Get into 3d space so I can do the bring together animation...
        # panel_1.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN)
        # self.add(panel_1, panel_2)

        # self.set_view(0, 89, 0, (-0.46, 0.0, 1.36), 8.97)

        # panel_1_shift=[-5, 0, 2.0]
        # panel_2_shift=[-5, 0, -2.0]
        # panel_1.shift(panel_1_shift)
        # panel_2.shift(panel_2_shift)

        ## --


        # p1_values=param_surface(0, 0)
        # p1_values[0]=map_to_canvas(p1_values[0], axis_min=x_axis_1.x_min, 
        #                                  axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        # p1_values[1]=map_to_canvas(p1_values[2], axis_min=y_axis_1.y_min, 
        #                                  axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
        # p1_values[2]=0

        # p1=Dot(p1_values, radius=0.06, fill_color=YELLOW)
        # # self.add(p1)

        # g=get_grads(0,0)
        # grad_viz_scale=1.25*abs(g[2]) #Maybe make arrow length proportional to gradient?
        # p1_values_2=param_surface(0, 0)
        # g_values=np.array([[p1_values_2[0], p1_values_2[2], 0],
        #                    [p1_values_2[0]+grad_viz_scale, p1_values_2[2]+grad_viz_scale*g[2]*0.6, 0]]) #Maybe I make arrow length proportional to slope or something?
        # g_values[:,0]=map_to_canvas(g_values[:,0], axis_min=x_axis_1.x_min, 
        #                                  axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        # g_values[:,1]=map_to_canvas(g_values[:,1], axis_min=y_axis_1.y_min, 
        #                                  axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        # a1=Arrow(start=g_values[0], end=g_values[1], fill_color=YELLOW, thickness=3.0, tip_width_ratio=5, buff=0)
        # # self.add(a1)

        # # panel_1_shift=[-5, 0.4, 0]
        # panel_1=VGroup(axes_1, curve_1, x_label_1, y_label_1, p1, a1)
        # # panel_1.shift(panel_1_shift)

        # # curve_1.set_stroke(opacity=0.25) #Yeah so I a fade out/fade in will bascially be the optning animation for this scene?
        # # self.add(panel_1)

        # # Ok I might clean up/rewrite this scene, we'll see. 
        # # I think it makes sense to to ahead and add panels 2 and 3, and then I can start thinking about how the
        # # pieces fit together, and exactly what I want to show from there
        # # This workflow feels a bit hacky/uncomfortable, but I do think I have a pretty good feeling of what I want
        # # to show, so I'm just going to trust this new/different process. 

        # x_axis_2=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
        #                     x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        # y_axis_2=WelchYAxis(y_min=0.3, y_max=2.2, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], y_tick_width=0.15,        
        #                   y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        # axes_2=VGroup(x_axis_2, y_axis_2)


        # points_2 = [param_surface(0, v) for v in np.linspace(-1, 4, 128)]
        # points_mapped_2=np.array(points_2)[:, (1,2,0)]
        # points_mapped_2[:,0]=map_to_canvas(points_mapped_2[:,0], axis_min=x_axis_2.x_min, 
        #                                  axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
        # points_mapped_2[:,1]=map_to_canvas(points_mapped_2[:,1], axis_min=y_axis_2.y_min, 
        #                                  axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)
        # curve_2 = VMobject()
        # curve_2.set_points_smoothly(points_mapped_2)
        # curve_2.set_stroke(width=4, color=BLUE, opacity=0.8)


        # x_label_2 = Tex(r'\theta_{2}', font_size=30).set_color(CHILL_BROWN)
        # y_label_2 = Tex('Loss', font_size=25).set_color(CHILL_BROWN)
        # x_label_2.next_to(x_axis_2, RIGHT, buff=0.05)
        # y_label_2.next_to(y_axis_2, UP, buff=0.05) #not sure I need this. 

        # p2_values=param_surface(0, 0)
        # p2_values[0]=map_to_canvas(p2_values[0], axis_min=x_axis_2.x_min, 
        #                                  axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
        # p2_values[1]=map_to_canvas(p2_values[2], axis_min=y_axis_2.y_min, 
        #                                  axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)
        # p2_values[2]=0
        # p2=Dot(p2_values, radius=0.06, fill_color=BLUE)

        # g=get_grads(0,0)
        # grad_viz_scale=1.25*abs(g[3]) #Maybe make arrow length proportional to gradient?
        # p2_values_2=param_surface(0, 0)
        # g_values_2=np.array([[p2_values_2[0], p2_values_2[2], 0],
        #                    [p2_values_2[0]+grad_viz_scale, p2_values_2[2]+grad_viz_scale*g[3]*1.0, 0]]) #Maybe I make arrow length proportional to slope or something?
        # g_values_2[:,0]=map_to_canvas(g_values_2[:,0], axis_min=x_axis_2.x_min, 
        #                                  axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
        # g_values_2[:,1]=map_to_canvas(g_values_2[:,1], axis_min=y_axis_2.y_min, 
        #                                  axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)

        # a2=Arrow(start=g_values_2[0], end=g_values_2[1], fill_color=BLUE, thickness=3.0, tip_width_ratio=5, buff=0)

        # panel_2.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN) #Get into 3d space so I can do the bring together animation...
        # panel_1.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN)

        ## ----

        # I'm thinking that after we show what the gradient is, we come back to 3 panels to show Gradient Descent 
        # happening!
        #

        ## ----


        #Ok isn't the answer here, ot at least part of the answer just using the same math: 
        #         points_1 = [param_surface(u, 0) for u in np.linspace(-1, 4, 128)]
        # points_mapped=np.array(points_1)[:, (0,2,1)]
        # points_mapped[:,0]=map_to_canvas(points_mapped[:,0], axis_min=x_axis_1.x_min, 
        #                                  axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        # points_mapped[:,1]=map_to_canvas(points_mapped[:,1], axis_min=y_axis_1.y_min, 
        #                                  axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
        # Not sure what ot make of the rotation just yet
        # Or exaclty how to apply these transformations to the ParametricSurface class hmm, 


        #There's a bunch of ways I coudl solve the alignment problem, let me hack for a minute
        # ts.scale([0.7, 0.7, 1.0])
        # # offset=axes_1.get_corner(BOTTOM+LEFT)-ts.get_corner(BOTTOM+LEFT) #Hmm meh
        # ts.shift([-4.5, -0.5, 0])

        # self.wait()

        #axes_1


        # x_axis_1=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
        #                     x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        # y_axis_1=WelchYAxis(y_min=0.3, y_max=1.7, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], y_tick_width=0.15,        
        #                   y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)

        # Create the surface
        # axes = ThreeDAxes(
        #     x_range=[-1.2, 4.5, 1],
        #     y_range=[-1.2, 4.5, 1],
        #     z_range=[0.0, 2.0, 0.2],
        #     height=4,
        #     width=4,
        #     depth=3,
        #     axis_config={
        #         "include_ticks": True,
        #         "color": CHILL_BROWN,
        #         "stroke_width": 2,
        #         "include_tip": True,
        #         "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
        #     }
        # )
        
        # # Add labels
        # x_label = Tex(r'\theta_{1}', font_size=40).set_color(CHILL_BROWN)
        # y_label = Tex(r'\theta_{2}', font_size=40).set_color(CHILL_BROWN)
        # z_label = Tex('Loss', font_size=30).set_color(CHILL_BROWN)
        # x_label.next_to(axes.x_axis, RIGHT)
        # y_label.next_to(axes.y_axis, UP)
        # z_label.next_to(axes.z_axis, OUT)
        # z_label.rotate(90*DEGREES, [1,0,0])

        # # Create gridlines using polylines instead of parametric curves
        # num_lines = 20  # Number of gridlines in each direction
        # num_points = 256  # Number of points per line
        # u_gridlines = VGroup()
        # v_gridlines = VGroup()
        
        # # Create u-direction gridlines
        # u_values = np.linspace(-1, 4, num_lines)
        # v_points = np.linspace(-1, 4, num_points)
        
        # for u in u_values:
        #     points = [param_surface(u, v) for v in v_points]
        #     line = VMobject()
        #     # line.set_points_as_corners(points)
        #     line.set_points_smoothly(points)
        #     line.set_stroke(width=1, color=WHITE, opacity=0.3)
        #     u_gridlines.add(line)
        
        # # Create v-direction gridlines
        # u_points = np.linspace(-1, 4, num_points)
        # for v in u_values:  # Using same number of lines for both directions
        #     points = [param_surface(u, v) for u in u_points]
        #     line = VMobject()
        #     # line.set_points_as_corners(points)
        #     line.set_points_smoothly(points)
        #     line.set_stroke(width=1, color=WHITE, opacity=0.3)
        #     v_gridlines.add(line)

        # #i think there's a better way to do this
        # offset=surface.get_corner(BOTTOM+LEFT)-axes.get_corner(BOTTOM+LEFT)
        # axes.shift(offset); x_label.shift(offset); y_label.shift(offset); z_label.shift(offset);
        # self.wait()            

        # # Add everything to the scene
        # self.add(axes)
        # axes.shift([-4.0, 0, 0])
        # self.wait()




        # self.add(axes, x_label, y_label, z_label)
        # self.add(u_gridlines)
        # self.add(v_gridlines)
        # self.add(ts)


        
        




# class P33v1(InteractiveScene):
#     def construct(self):

#         # Ok ok ok ok I need ot decide how automated vs manual I'm going to make this scene
#         # Thinking about this scene more, I do think the "right triangle overhead view" of the gradient is going to be nice/important.
#         #  
#         # Ok ok ok ok ok I do think i want some more Meaat on p33-p35 → specifically it’s interesting and non-obvious that we can put 
#         # each slope together in that overhead view and it will point us downhill. Worth thinking about that animation a bit I think → 
#         # maybe an actually good use case for WelchAxes lol. And i can draw a connection to part 2 → “as we’ll see in part 2 it turns 
#         # out that we can very efficiently estimate the slope of the curves without actually computing an points on them → and then 
#         # maybe the little arrows on the curves move to the overhead view? or copies of them? Them more i think about this the more 
#         # I think it should be manimy? I can draw the projections/cuts as nice blue/yellow lines in the overhead view too. 
#         #
#         # Alright kinda low on brain power here, but let me at least try to get the pieces together tonight
#         # I do think it probably makes sense to at least try to get all 3 graphs on one canvas, so I can like smoothlly move arrows arround and stuff
#         # I might need to go compute a bunch of gradients eh?


#         x_axis_1=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
#                             x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
#         y_axis_1=WelchYAxis(y_min=0.3, y_max=1.7, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], y_tick_width=0.15,        
#                           y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
#         axes_1=VGroup(x_axis_1, y_axis_1)

#         points_1 = [param_surface(u, 0) for u in np.linspace(-1, 4, 128)]
#         points_mapped=np.array(points_1)[:, (0,2,1)]
#         points_mapped[:,0]=map_to_canvas(points_mapped[:,0], axis_min=x_axis_1.x_min, 
#                                          axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
#         points_mapped[:,1]=map_to_canvas(points_mapped[:,1], axis_min=y_axis_1.y_min, 
#                                          axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
#         curve_1 = VMobject()
#         curve_1.set_points_smoothly(points_mapped)
#         curve_1.set_stroke(width=4, color=YELLOW, opacity=0.8)

#         x_label_1 = Tex(r'\theta_{1}', font_size=30).set_color(CHILL_BROWN)
#         y_label_1 = Tex('Loss', font_size=25).set_color(CHILL_BROWN)
#         x_label_1.next_to(x_axis_1, RIGHT, buff=0.05)
#         y_label_1.next_to(y_axis_1, UP, buff=0.05)

#         # self.add(axes_1, curve_1)
#         # self.add(x_label_1, y_label_1)

#         # Ok, adding second curve should be pretty straightfoward, maybe I think about gradient a bit now?
#         # Intuitively I think I need to go actually compuate all the gradients at each point and cache them? 
#         # Let me go take a crack at that. 
#         # Man I guess you could show the whole gradient field with this approach - that's kinda interesting
#         # Not sure if that has a role in the video -> could be fun in the book!

#         # I've created a crappy version of c2p I think! Should be able to roll in or absorb. 
#         # Keep roling for now though, I can make it pretty after I decide if I like the pattern.

#         #Hmm well this is complicated and not working lol - will pick back up in the morning. 
#         p1_values=param_surface(0, 0)
#         p1_values[0]=map_to_canvas(p1_values[0], axis_min=x_axis_1.x_min, 
#                                          axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
#         p1_values[1]=map_to_canvas(p1_values[2], axis_min=y_axis_1.y_min, 
#                                          axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
#         p1_values[2]=0

#         p1=Dot(p1_values, radius=0.06, fill_color=YELLOW)
#         # self.add(p1)

#         g=get_grads(0,0)
#         grad_viz_scale=abs(g[2]) #Maybe make arrow length proportional to gradient?
#         p1_values_2=param_surface(0, 0)
#         g_values=np.array([[p1_values_2[0], p1_values_2[2], 0],
#                            [p1_values_2[0]+grad_viz_scale, p1_values_2[2]+grad_viz_scale*g[2]*0.6, 0]]) #Maybe I make arrow length proportional to slope or something?
#         g_values[:,0]=map_to_canvas(g_values[:,0], axis_min=x_axis_1.x_min, 
#                                          axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
#         g_values[:,1]=map_to_canvas(g_values[:,1], axis_min=y_axis_1.y_min, 
#                                          axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

#         a1=Arrow(start=g_values[0], end=g_values[1], fill_color=YELLOW, thickness=3.0, tip_width_ratio=5, buff=0)
#         # self.add(a1)

#         panel_1_shift=[-5, 0.4, 0]
#         panel_1=VGroup(axes_1, curve_1, x_label_1, y_label_1, p1, a1)
#         panel_1.shift(panel_1_shift)

#         curve_1.set_stroke(opacity=0.25) #Yeah so I a fade out/fade in will bascially be the optning animation for this scene?
#         self.add(panel_1)

#         # Ok I might clean up/rewrite this scene, we'll see. 
#         # I think it makes sense to to ahead and add panels 2 and 3, and then I can start thinking about how the
#         # pieces fit together, and exactly what I want to show from there
#         # This workflow feels a bit hacky/uncomfortable, but I do think I have a pretty good feeling of what I want
#         # to show, so I'm just going to trust this new/different process. 

#         x_axis_2=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
#                             x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
#         y_axis_2=WelchYAxis(y_min=0.3, y_max=1.7, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], y_tick_width=0.15,        
#                           y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
#         axes_2=VGroup(x_axis_2, y_axis_2)


#         points_2 = [param_surface(0, v) for v in np.linspace(-1, 4, 128)]
#         points_mapped_2=np.array(points_2)[:, (1,2,0)]
#         points_mapped_2[:,0]=map_to_canvas(points_mapped_2[:,0], axis_min=x_axis_2.x_min, 
#                                          axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
#         points_mapped_2[:,1]=map_to_canvas(points_mapped_2[:,1], axis_min=y_axis_2.y_min, 
#                                          axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)
#         curve_2 = VMobject()
#         curve_2.set_points_smoothly(points_mapped_2)
#         curve_2.set_stroke(width=4, color=BLUE, opacity=0.8)


#         x_label_2 = Tex(r'\theta_{2}', font_size=30).set_color(CHILL_BROWN)
#         y_label_2 = Tex('Loss', font_size=25).set_color(CHILL_BROWN)
#         x_label_2.next_to(x_axis_2, RIGHT, buff=0.05)
#         y_label_2.next_to(y_axis_2, UP, buff=0.05) #not sure I need this. 

#         p2_values=param_surface(0, 0)
#         p2_values[0]=map_to_canvas(p2_values[0], axis_min=x_axis_2.x_min, 
#                                          axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
#         p2_values[1]=map_to_canvas(p2_values[2], axis_min=y_axis_2.y_min, 
#                                          axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)
#         p2_values[2]=0
#         p2=Dot(p2_values, radius=0.06, fill_color=BLUE)

#         g=get_grads(0,0)
#         grad_viz_scale=abs(g[3]) #Maybe make arrow length proportional to gradient?
#         p2_values_2=param_surface(0, 0)
#         g_values_2=np.array([[p2_values_2[0], p2_values_2[2], 0],
#                            [p2_values_2[0]+grad_viz_scale, p2_values_2[2]+grad_viz_scale*g[3]*1.0, 0]]) #Maybe I make arrow length proportional to slope or something?
#         g_values_2[:,0]=map_to_canvas(g_values_2[:,0], axis_min=x_axis_2.x_min, 
#                                          axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
#         g_values_2[:,1]=map_to_canvas(g_values_2[:,1], axis_min=y_axis_2.y_min, 
#                                          axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)

#         a2=Arrow(start=g_values_2[0], end=g_values_2[1], fill_color=BLUE, thickness=3.0, tip_width_ratio=5, buff=0)

#         panel_2_shift=[-5, -3.2, 0]
#         panel_2=VGroup(axes_2, curve_2, x_label_2, p2, a2) #y_label_2
#         panel_2.shift(panel_2_shift)

#         curve_2.set_stroke(opacity=0.25) 
#         self.add(panel_2)


#         ## Hmm man you know what I'm pretty srue I can't just have a 3d panel mixed in these
#         ## 2d ones right? It will just have to the overhead view, unless I can find some way of like 
#         ## rotating it
#         ## But really, the move is probably just having separate scenes that 
#         ## Oh man man is there some way to actually bring these 2d plots togetehr INTO the 3d one? 
#         ## Obviously dont want to add a bunch of random scope ehre, but I do think that woudl show what we're 
#         ## about here really nicely. Becuase we bring together the two curves into the 3d surface and it will 
#         ## be nice an dclear that the gradient arrows are guiding us downhill. 
#         ## I'm also a little confused by the uphill vs downhill thing. 
#         ## 
#         ## Drafting a Claude prompt, I think I'm starting to figure this out - not sure what it means for 
#         ## visualization and this section yet...
#         ## I'm confused about why the gradient points uphill in loss landsacpes - the partial derivates 
#         ## that make up the gradient are equivalent to the slope at each point - if increasing some 
#         ## parameter makes the loss go down, then the slope of the loss with respect to that parameter 
#         ## should be negative.
#         ##
#         ## Ok so it's like the slope/gradient is negative from our starting point, which means if we 
#         ## add the gradient to theta, since the gradient is negative we will go uphill. 
#         ## That's the deal here -> how do we visualize what's going on?
#         ## I guess like adding the gradient to the parameter is not actually that natural of a thing visually?
#         ## I feel like it should be though -> it should be a little vector that guides us through parameter space...
#         ## Hmmmm....
#         ## 
#         ## Ok I think i have a better sense of what's going on here - and in the script I'll say something 
#         ## like that technically the gradient goes the other way, and we'll get into this in part 2. 
#         ## 
#         ## Ok there's some details etc that I need to change here
#         ## But I have a pretty strong vision for what I think I want the core animation to be
#         ## I think doing it like this is actually going to be more clear
#         ## I want the two curves to "come together" into the surfce, and then really clearly show that the 
#         ## overall gradient is the combination of the cyan and yellow arrows, it will be like nicely in the
#         ## center fo thes two....
#         ##
#         ## So.. let me build out a workign sketch that core animation, and then we can take if from there.  
#         ## 


#         # Create main surface
#         surface = ParametricSurface(
#             1.5*param_surface, #Scaling here to make gradients a bit more interesting. 
#             u_range=[-1, 4],
#             v_range=[-1, 4],
#             resolution=(256, 256),
#         )


#         # Create the surface
#         axes = ThreeDAxes(
#             x_range=[-1, 4, 1],
#             y_range=[-1, 4, 1],
#             z_range=[0.0, 3.5, 1.0],
#             height=5,
#             width=5,
#             depth=3.5,
#             axis_config={
#                 "include_ticks": True,
#                 "color": CHILL_BROWN,
#                 "stroke_width": 2,
#                 "include_tip": True,
#                 "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
#             }
#         )
        
#         # Add labels
#         x_label = Tex(r'\theta_{1}', font_size=40).set_color(CHILL_BROWN)
#         y_label = Tex(r'\theta_{2}', font_size=40).set_color(CHILL_BROWN)
#         z_label = Tex('Loss', font_size=30).set_color(CHILL_BROWN)
#         x_label.next_to(axes.x_axis, RIGHT)
#         y_label.next_to(axes.y_axis, UP)
#         z_label.next_to(axes.z_axis, OUT)
#         z_label.rotate(90*DEGREES, [1,0,0])
        
#         ts = TexturedSurface(surface, '/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_24_28_losses_4.png')
#         ts.set_shading(0.0, 0.1, 0)
#         ts.set_opacity(0.7)

#         # Create gridlines using polylines instead of parametric curves
#         num_lines = 20  # Number of gridlines in each direction
#         num_points = 256  # Number of points per line
#         u_gridlines = VGroup()
#         v_gridlines = VGroup()
        
#         # Create u-direction gridlines
#         u_values = np.linspace(-1, 4, num_lines)
#         v_points = np.linspace(-1, 4, num_points)
        
#         for u in u_values:
#             points = [param_surface(u, v) for v in v_points]
#             line = VMobject()
#             # line.set_points_as_corners(points)
#             line.set_points_smoothly(points)
#             line.set_stroke(width=1, color=WHITE, opacity=0.3)
#             u_gridlines.add(line)
        
#         # Create v-direction gridlines
#         u_points = np.linspace(-1, 4, num_points)
#         for v in u_values:  # Using same number of lines for both directions
#             points = [param_surface(u, v) for u in u_points]
#             line = VMobject()
#             # line.set_points_as_corners(points)
#             line.set_points_smoothly(points)
#             line.set_stroke(width=1, color=WHITE, opacity=0.3)
#             v_gridlines.add(line)

#         #i think there's a better way to do this
#         offset=surface.get_corner(BOTTOM+LEFT)-axes.get_corner(BOTTOM+LEFT)
#         axes.shift(offset); x_label.shift(offset); y_label.shift(offset); z_label.shift(offset);
#         # axes.move_to([1,1,1])
        


        
#         # Add everything to the scene
#         self.add(axes, x_label, y_label, z_label)
#         self.add(u_gridlines)
#         self.add(v_gridlines)
#         self.add(ts)


#         self.set_view(32, 59, 0, (1.88, 1.0, 1.52), 7.78)

        
#         self.embed()
#         self.wait(20)





