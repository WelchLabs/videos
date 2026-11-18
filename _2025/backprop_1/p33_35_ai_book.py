from welch_paths import welch_path
import manimgx as m
from welch_geometry import diagram_arrow
from landscape_geometry import get_pivot_and_scale
import numpy as np
from welch_scene import WelchScene
from welch_surface import parametric_surface, set_surface_shading, textured_surface
from welch_text import math_tex

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
RED='#EC2027'
BLUE='#65c8d0'

# surf=np.load('/home/zedaes/Documents/Welch Labs/ai_book/backprop/data/p_24_28_losses_4.npy')
# xy=np.load('/home/zedaes/Documents/Welch Labs/ai_book/backprop/data/p_24_28_losses_4xy.npy')
# grads_1=np.load('/home/zedaes/Documents/Welch Labs/ai_book/backprop/data/p_33_35_grads_1_2.npy') 
# grads_2=np.load('/home/zedaes/Documents/Welch Labs/ai_book/backprop/data/p_33_35_grads_2_2.npy') 
# xy_grads=np.load('/home/zedaes/Documents/Welch Labs/ai_book/backprop/data/p_33_35_xy_2.npy') 

surf=np.load(welch_path('backpropagation/hackin/p_24_28_losses_4.npy'))
xy=np.load(welch_path('backpropagation/hackin/p_24_28_losses_4xy.npy'))
grads_1=np.load(welch_path('backpropagation/hackin/p_33_35_grads_1_2.npy'))
grads_2=np.load(welch_path('backpropagation/hackin/p_33_35_grads_2_2.npy'))
xy_grads=np.load(welch_path('backpropagation/hackin/p_33_35_xy_2.npy'))



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

#Do 2d numerical-ish gradient descent once and use results in a couple places. 
num_steps=10
learning_rate=1.05 #1.25
grad_adjustment_factors_1=[0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6] # Not sure why I need these - only applying to viz - 
grad_adjustment_factors_2=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0] # maybe I should use for descent too?
descent_points=[] 
arrow_end_points_1=[]
arrow_end_points_2=[]

starting_values=param_surface(0, 0)
descent_points.append(list(starting_values)) #First point

for i in range(1, num_steps):
    g=get_grads(descent_points[i-1][0], descent_points[i-1][1]) 
    step_x_1=learning_rate*abs(g[2])
    step_x_2=learning_rate*abs(g[3])
    new_x_1=descent_points[i-1][0]+step_x_1
    new_x_2=descent_points[i-1][1]+step_x_2
    arrow_end_points_1.append([new_x_1, descent_points[i-1][2]+step_x_1*g[2]*grad_adjustment_factors_1[i], 0])
    arrow_end_points_2.append([new_x_2, descent_points[i-1][2]+step_x_2*g[3]*grad_adjustment_factors_2[i], 0])
    descent_points.append([new_x_1, new_x_2, param_surface(new_x_1, new_x_2)[2]])
    # print(g)
# print(descent_points)
arrow_end_points_1=np.array(arrow_end_points_1)
arrow_end_points_2=np.array(arrow_end_points_2)
descent_points=np.array(descent_points)



class P33_35(WelchScene):
    def construct(self):
        '''
        Ok like 100 lines of duplicated code here -> I'm tempted to roll into a shared function, but that seems premature and
        i would lose some variable access - I think i can live with the duplication. 
        '''
        
        x_axis_1=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
                            x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_1=WelchYAxis(y_min=0.3, y_max=2.2, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        axes_1=m.VGroup(x_axis_1, y_axis_1)

        x_label_1 = math_tex(r'\theta_{1}', font_size=30).set_color(CHILL_BROWN)
        y_label_1 = math_tex('Loss', font_size=25).set_color(CHILL_BROWN)
        x_label_1.next_to(x_axis_1, m.RIGHT, buff=0.05)
        y_label_1.next_to(y_axis_1, m.UP, buff=0.05)

        x_axis_2=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
                            x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_2=WelchYAxis(y_min=0.3, y_max=2.2, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        axes_2=m.VGroup(x_axis_2, y_axis_2)

        x_label_2 = math_tex(r'\theta_{2}', font_size=30).set_color(CHILL_BROWN)
        y_label_2 = math_tex('Loss', font_size=25).set_color(CHILL_BROWN)
        x_label_2.next_to(x_axis_2, m.RIGHT, buff=0.05)
        y_label_2.next_to(y_axis_2, m.UP, buff=0.05) #not sure I need this.


        ## Get all Curves, Points, Arrows, and Lines, then Group into panels
        curves_1=m.VGroup(); curves_2=m.VGroup()
        points_1=m.VGroup(); points_2=m.VGroup()
        arrows_1=m.VGroup(); arrows_2=m.VGroup()
        lines_1=m.VGroup(); lines_2=m.VGroup()

        #Map all points
        descent_points_mapped_1=np.zeros_like(descent_points)
        descent_points_mapped_1[:,0]=map_to_canvas(descent_points[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        descent_points_mapped_1[:,1]=map_to_canvas(descent_points[:,2], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        descent_points_mapped_2=np.zeros_like(descent_points)
        descent_points_mapped_2[:,0]=map_to_canvas(descent_points[:,1], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        descent_points_mapped_2[:,1]=map_to_canvas(descent_points[:,2], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        arrow_end_points_1_mapped=np.zeros_like(arrow_end_points_1)
        arrow_end_points_1_mapped[:,0]=map_to_canvas(arrow_end_points_1[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        arrow_end_points_1_mapped[:,1]=map_to_canvas(arrow_end_points_1[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        arrow_end_points_2_mapped=np.zeros_like(arrow_end_points_2)
        arrow_end_points_2_mapped[:,0]=map_to_canvas(arrow_end_points_2[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        arrow_end_points_2_mapped[:,1]=map_to_canvas(arrow_end_points_2[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        for i in range(len(descent_points)):
            #Curves 1
            p1 = np.array([param_surface(u, descent_points[i][1]) for u in np.linspace(-1, 4, 128)])
            points_mapped=np.zeros_like(p1)
            points_mapped[:,0]=map_to_canvas(p1[:,0], axis_min=x_axis_1.x_min, 
                                             axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
            points_mapped[:,1]=map_to_canvas(p1[:,2], axis_min=y_axis_1.y_min, 
                                             axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
            c = m.VMobject()
            c.set_points_smoothly(points_mapped)
            c.set_stroke(width=4, color=RED, opacity=0.8)
            curves_1.add(c)

            #Points 1
            p=m.Dot(descent_points_mapped_1[i], radius=0.06, fill_color=RED)
            points_1.add(p)

            # #Curves 2
            p1 = np.array([param_surface(descent_points[i][0], v) for v in np.linspace(-1, 4, 128)])
            points_mapped=np.zeros_like(p1)
            points_mapped[:,0]=map_to_canvas(p1[:,1], axis_min=x_axis_1.x_min, 
                                             axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
            points_mapped[:,1]=map_to_canvas(p1[:,2], axis_min=y_axis_1.y_min, 
                                             axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
            c = m.VMobject()
            c.set_points_smoothly(points_mapped)
            c.set_stroke(width=4, color=BLUE, opacity=0.8)
            curves_2.add(c)

            #Points 2
            p=m.Dot(descent_points_mapped_2[i], radius=0.06, fill_color=BLUE)
            points_2.add(p)
          
            if i>0:
                lines_1.add(m.Line(start=[descent_points_mapped_1[i-1][0], descent_points_mapped_1[i-1][1], 0],
                                   end=[descent_points_mapped_1[i][0], descent_points_mapped_1[i][1], 0], 
                                   color=RED, buff=0, stroke_width=1.5))   
                arrows_1.add(diagram_arrow(start=[descent_points_mapped_1[i-1][0], descent_points_mapped_1[i-1][1], 0],
                                     end=arrow_end_points_1_mapped[i-1], fill_color=RED, 
                                     thickness=3.0, tip_width_ratio=5, buff=0, max_width_to_length_ratio=0.2))
                lines_2.add(m.Line(start=[descent_points_mapped_2[i-1][0], descent_points_mapped_2[i-1][1], 0],
                                   end=[descent_points_mapped_2[i][0], descent_points_mapped_2[i][1], 0], 
                                   color=BLUE, buff=0, stroke_width=1.5))   
                arrows_2.add(diagram_arrow(start=[descent_points_mapped_2[i-1][0], descent_points_mapped_2[i-1][1], 0],
                                     end=arrow_end_points_2_mapped[i-1], fill_color=BLUE, 
                                     thickness=3.0, tip_width_ratio=5, buff=0, max_width_to_length_ratio=0.2))

        ## Test viz as I go here            
        panel_1=m.VGroup(axes_1, x_label_1, y_label_1, curves_1, points_1, arrows_1, lines_1)
        panel_1.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        panel_2=m.VGroup(axes_2, x_label_2, y_label_2, curves_2, points_2, arrows_2, lines_2)
        panel_2.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)

        panel_1_shift=[-5, 0, 2.0]
        panel_2_shift=[-5, 0, -2.0]
        panel_1.shift(panel_1_shift)
        panel_2.shift(panel_2_shift)

        # self.add(panel_1, panel_2)

        self.set_view(0, 89, 0, (-0.46, 0.0, 1.36), 8.97)

        curves_1[0].set_stroke(opacity=0.5)
        curves_2[0].set_stroke(opacity=0.5)
        panel_1_start=m.VGroup(axes_1, x_label_1, y_label_1, curves_1[0], points_1[0], arrows_1[0])
        panel_2_start=m.VGroup(axes_2, x_label_2, y_label_2, curves_2[0], points_2[0], arrows_2[0])

        self.add(panel_1_start)
        self.add(panel_2_start)
        self.wait()

        ## --- Ok so here I want to add a little "move the arrows back and forth along the parabolas deal..."


        def get_arrow_1(u):
            start=param_surface(u, 0)
            g=get_grads(u+0.2, 0) #I don't know what's up with my grads bu this helps
            # step_x_1=learning_rate*abs(g[2])
            step_x_1=0.6
            new_x=start[0]+step_x_1
            new_y=start[2]+step_x_1*g[2]*0.7  #Swaggy

            #Ok now I need to map to axes and apply rotations
            mapped_values=np.zeros((2,2))
            mapped_values[:,0]=map_to_canvas(np.array([start[0], new_x]), axis_min=x_axis_1.x_min, 
                                             axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
            mapped_values[:,1]=map_to_canvas(np.array([start[2], new_y]), axis_min=y_axis_1.y_min, 
                                             axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)  

            a=diagram_arrow(start=[mapped_values[0,0], mapped_values[0,1], 0],
                     end=[mapped_values[1,0], mapped_values[1,1], 0], fill_color=RED, 
                     thickness=3.0, tip_width_ratio=5, buff=0, max_width_to_length_ratio=0.2)
            a.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
            a.shift(panel_1_shift)    
            return a

        def get_arrow_2(v):
            start=param_surface(0, v)
            g=get_grads(0, v*0.75) #I don't know what's up with my grads bu this helps
            # step_x_1=learning_rate*abs(g[2])
            step_x_1=0.6
            new_x=start[1]+step_x_1
            if v>0:
                new_y=start[2]+step_x_1*g[3] #*0.7  #Swaggy
            else:
                new_y=start[2]+step_x_1*g[3]-(0.2*abs(v)) #Oh my god this is so fucking hacky. 

            #Ok now I need to map to axes and apply rotations
            mapped_values=np.zeros((2,2))
            mapped_values[:,0]=map_to_canvas(np.array([start[1], new_x]), axis_min=x_axis_1.x_min, 
                                             axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
            mapped_values[:,1]=map_to_canvas(np.array([start[2], new_y]), axis_min=y_axis_1.y_min, 
                                             axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)  

            a=diagram_arrow(start=[mapped_values[0,0], mapped_values[0,1], 0],
                     end=[mapped_values[1,0], mapped_values[1,1], 0], fill_color=BLUE, 
                     thickness=3.0, tip_width_ratio=5, buff=0, max_width_to_length_ratio=0.2)
            a.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
            a.shift(panel_2_shift)    
            return a

        # a=get_arrow_1(1)
        # a=get_arrow_2(2)
        # self.add(a)

        initial_time = 0
        t_tracker = m.ValueTracker(initial_time)

        moving_arrow_1=m.always_redraw(lambda: get_arrow_1(t_tracker.get_value()))
        moving_arrow_2=m.always_redraw(lambda: get_arrow_2(t_tracker.get_value()))

        self.remove(arrows_1[0], arrows_2[0])
        self.add(moving_arrow_1, moving_arrow_2)
        self.play(t_tracker.animate.set_value(1.6), run_time=3)
        self.play(t_tracker.animate.set_value(-0.5), run_time=3)
        self.play(t_tracker.animate.set_value(0), run_time=3)
        self.remove(moving_arrow_1, moving_arrow_2)
        self.add(arrows_1[0], arrows_2[0])
        self.wait()


        ## --- End moving arrows back and forth. 


        ## Start move to 3d
        r=panel_2.get_corner(m.LEFT+(m.config.frame_height / 2 * m.DOWN))
        r[0]=-4.15    
        self.wait()

        self.play(y_axis_2.animate.set_opacity(0.0),
                  x_axis_1[-1].animate.set_opacity(0.0), #Tick labels
                  x_axis_2[-1].animate.set_opacity(0.0),
                  y_axis_1[-1].animate.set_opacity(0.0),
                  x_axis_1[-2].animate.set_opacity(0.0), #Ticks
                  x_axis_2[-2].animate.set_opacity(0.0),
                  y_axis_1[-2].animate.set_opacity(0.0),
                  y_label_2.animate.set_opacity(0.0),
                  run_time=1.0)

        self.play(panel_1_start.animate.shift([0, 0, -2.0]),
                  panel_2_start.animate.rotate(90*m.DEGREES, [0,0,1], about_point=r).shift([0, 0, 2.0]),
                  self.animate_view(13, 85, 0, (-1.88, -0.77, 1.56), 5.01),
                  run_time=4)

        self.wait()

        # Create main surface
        surface = parametric_surface(
            param_surface,  
            u_range=[-1, 4],
            v_range=[-1, 4],
            resolution=(256, 256),
        )

        # ts = TexturedSurface(surface, '/home/zedaes/Documents/Welch Labs/ai_book/backprop/data/p_24_28_losses_4.png')
        ts = textured_surface(surface, welch_path('backpropagation/hackin/p_24_28_losses_4.png'))

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
        self.remove(arrows_1[0]); self.add(arrows_1[0]) #Occlusions
        self.remove(arrows_2[0]); self.add(arrows_2[0])
        self.remove(points_1[0]); self.add(points_1[0])
        self.remove(points_2[0]); self.add(points_2[0])

        self.play(ts.animate.set_opacity(0.5), 
                  points_1[0].animate.set_opacity(0.0),
                  points_2[0].animate.set_opacity(0.0),
                  arrows_1[0].animate.rotate(-m.DEGREES*135, axis=arrows_1[0].get_end()-arrows_1[0].get_start()),
                  arrows_2[0].animate.rotate(-m.DEGREES*80, axis=arrows_2[0].get_end()-arrows_2[0].get_start()),
                  u_gridlines.animate.set_stroke(opacity=0.5), #0.14 Trying higher for book
                  v_gridlines.animate.set_stroke(opacity=0.5), #0.14
                  # self.animate_view(125, 57, 0, (-2.45, 1.36, 2.08), 1.21),
                  # self.animate_view(106, 41, 0, (-2.43, 0.92, 2.55), 3.11),
                  self.animate_view(124, 40, 0, (-2.57, 0.86, 2.7), 1.81),
                  run_time=4.0)

        self.wait()

        #Might be a good spot for a book shot here

        self.remove(x_label_1)
        self.remove(x_label_2)
        self.remove(y_label_1)
        self.remove(arrows_1)
        self.remove(arrows_2)
        self.wait()


        # self.set_view(-154, 65, 0, (-3.64, 1.96, 1.53), 4.59)
        # self.set_view(-166, 76, 0, (-3.71, 2.02, 1.28), 4.12)
        # self.set_view(-151, 66, 0, (-3.66, 2.01, 1.37), 4.56)
        self.set_view(-150, 60, 0, (-3.52, 1.99, 1.55), 5.15)
        self.wait(5)

        pass

        #Ok, I'll do a decent amount of overlay, it's not perfect, but I think it can work??

        # self.remove(arrows_1[0]); self.add(arrows_1[0])









        # a3 =Arrow(start=[arrows_1[0].get_corner(LEFT)[0]+0.03, arrows_1[0].get_corner(LEFT)[1]+0.01, arrows_1[0].get_corner(OUT)[2]],
        #           end=[arrows_1[0].get_corner(RIGHT)[0], arrows_2[0].get_corner(UP)[1], arrows_2[0].get_corner(IN)[2]-0.1], 
        #           fill_color='#FF00FF', thickness=3.0, tip_width_ratio=5, buff=0)
        # self.wait()

        # self.play(TransformFromCopy(arrows_1[0], a3), #Ah that's fucking dope - how many more cool tricks does Grant have up his sleeve that I know nothing about lol. 
        #           TransformFromCopy(arrows_2[0], a3),
        #           run_time=3.0)
        # self.wait()


        # s1=Dot3D(center=a3.get_start(), radius=0.06, color='$FF00FF')
        # s2=Dot3D(center=a3.get_end(), radius=0.06, color='$FF00FF')
        # self.wait()

        # self.play(arrows_1[0].animate.set_opacity(0.0),
        #           arrows_2[0].animate.set_opacity(0.0),
        #           curves_1[0].animate.set_opacity(0.0),
        #           curves_2[0].animate.set_opacity(0.0),
        #           FadeIn(s1),
        #           FadeIn(s2),
        #           self.animate_view(175, 47, 0, (-3.89, 1.49, 1.6), 3.75),
        #           run_time=2.0)
        # self.remove(s1); self.add(s1) #Occlusions
        # self.wait()

        # # Now add in next set of RED and blue lines (maybe curves)? And have them merge together again to steer us 
        # # donwhill bro!
        # # Ok so I need to move the rest of the panel stuff...
        # # panel_1=VGroup(axes_1, x_label_1, y_label_1, curves_1, points_1, arrows_1, lines_1)
        # # panel_1_start=VGroup(axes_1, x_label_1, y_label_1, curves_1[0], points_1[0], arrows_1[0])
        # panel_1_leftovers=VGroup(curves_1[1:], points_1[1:], arrows_1[1:])
        # panel_2_leftovers=VGroup(curves_2[1:], points_2[1:], arrows_2[1:])
        # panel_1_leftovers.shift([0, 0, -2.0])
        # panel_2_leftovers.rotate(90*DEGREES, [0,0,1], about_point=r).shift([0, 0, 2.0])

        # # --- Step 2
        # arrows_1[1].move_to(s2.get_center(), aligned_edge=LEFT)
        # arrows_1[1].rotate(-DEGREES*130, axis=arrows_1[1].get_end()-arrows_1[1].get_start())
        # arrows_1[1].shift([0,0,-0.06])
        
        # arrows_2[1].move_to(s2.get_center(), aligned_edge=LEFT)
        # arrows_2[1].rotate(-DEGREES*75, axis=arrows_2[1].get_end()-arrows_2[1].get_start())
        # arrows_2[1].shift([0,0.12,-0.06])

        # self.add(arrows_1[1], arrows_2[1])

        # # self.add(curves_1[1])
        # # curves_1[1].set_stroke(opacity=0.15)
        # # curves_1[1].shift([0, (arrows_1[1].get_center()-curves_1[1].get_center())[1],0])
        # #Ok I don't think adding the curves here is really helpful/interesting. 
        # a4 =Arrow(start=[arrows_1[1].get_corner(LEFT)[0]+0.00, arrows_1[1].get_corner(LEFT)[1]+0.00, arrows_1[1].get_corner(OUT)[2]],
        #           end=[arrows_1[1].get_corner(RIGHT)[0], arrows_2[1].get_corner(UP)[1], arrows_2[1].get_corner(IN)[2]-0.07], 
        #           fill_color='#FF00FF', thickness=3.0, tip_width_ratio=5, buff=0, max_width_to_length_ratio=0.2)
        # # self.add(a4)
        # self.play(TransformFromCopy(arrows_1[1], a4), 
        #           TransformFromCopy(arrows_2[1], a4),
        #           # self.animate_view(175, 47, 0, (-3.89, 1.49, 1.6), 3.75),
        #           self.animate_view(179, 48, 0, (-3.66, 1.68, 1.37), 2.39),
        #           run_time=3.0)

        # # self.set_view(-141, 63, 0, (-3.87, 0.05, -0.09), 2.29) #Debug view
        # # self.wait()

        # s3=Dot3D(center=a4.get_end(), radius=0.05, color='$FF00FF')
        # self.play(arrows_1[1].animate.set_opacity(0.0),
        #           arrows_2[1].animate.set_opacity(0.0),
        #           FadeIn(s3),
        #           # self.animate_view(175, 47, 0, (-3.89, 1.49, 1.6), 3.75),
        #           run_time=2.0)
        # self.remove(s2); self.add(s2) #Occlusions
        # self.wait()

        # # --- Step 3
        # arrows_1[2].move_to(s3.get_center(), aligned_edge=LEFT)
        # arrows_1[2].rotate(-DEGREES*120, axis=arrows_1[2].get_end()-arrows_1[2].get_start())
        # # arrows_1[2].shift([0,0,-0.05])
        
        # arrows_2[2].move_to(s3.get_center(), aligned_edge=LEFT)
        # arrows_2[2].rotate(-DEGREES*70, axis=arrows_2[2].get_end()-arrows_2[2].get_start())
        # arrows_2[2].shift([0,0.09,0.03])

        # self.add(arrows_1[2], arrows_2[2])

        # a5 =Arrow(start=[arrows_1[2].get_corner(LEFT)[0]+0.00, arrows_1[2].get_corner(LEFT)[1]+0.00, arrows_1[2].get_corner(OUT)[2]],
        #           end=[arrows_1[2].get_corner(RIGHT)[0], arrows_2[2].get_corner(UP)[1], arrows_2[2].get_corner(IN)[2]-0.05], 
        #           fill_color='#FF00FF', thickness=3.0, tip_width_ratio=3, buff=0, max_width_to_length_ratio=0.2)
        # # self.add(a5)
        # self.wait()
        
        # self.play(TransformFromCopy(arrows_1[2], a5), 
        #           TransformFromCopy(arrows_2[2], a5),
        #           self.animate_view(179, 54, 0, (-3.44, 1.77, 1.25), 1.62),
        #           run_time=3.0)
        # # self.wait()
        # # self.set_view(-141, 63, 0, (-3.87, 0.05, -0.09), 2.29) #Debug view

        # s4=Dot3D(center=a5.get_end(), radius=0.035, color='$FF00FF')
        # self.play(arrows_1[2].animate.set_opacity(0.0),
        #           arrows_2[2].animate.set_opacity(0.0),
        #           FadeIn(s4),
        #           # self.animate_view(175, 47, 0, (-3.89, 1.49, 1.6), 3.75),
        #           run_time=2.0)
        # self.remove(s3); self.add(s3) #Occlusions
        # self.wait()

        # # --- Step 4
        # arrows_1[3].move_to(s4.get_center(), aligned_edge=LEFT)
        # arrows_1[3].rotate(-DEGREES*120, axis=arrows_1[3].get_end()-arrows_1[3].get_start())
        # # arrows_1[3].shift([0,0,-0.05])
        
        # arrows_2[3].move_to(s4.get_center(), aligned_edge=LEFT)
        # arrows_2[3].rotate(-DEGREES*70, axis=arrows_2[3].get_end()-arrows_2[3].get_start())
        # arrows_2[3].shift([0,0.04,0.01])

        # self.add(arrows_1[3], arrows_2[3])

        # a6 =Arrow(start=[arrows_1[3].get_corner(LEFT)[0]+0.00, arrows_1[3].get_corner(LEFT)[1]+0.00, arrows_1[3].get_corner(OUT)[2]],
        #           end=[arrows_1[3].get_corner(RIGHT)[0], arrows_2[3].get_corner(UP)[1], arrows_2[3].get_corner(IN)[2]], 
        #           fill_color='#FF00FF', thickness=3.0, tip_width_ratio=3, buff=0, max_width_to_length_ratio=0.2)
        # # self.add(a6)
        # self.wait()
        
        # self.play(TransformFromCopy(arrows_1[3], a6), 
        #           TransformFromCopy(arrows_2[3], a6),
        #           self.animate_view(179, 54, 0, (-3.43, 1.82, 1.18), 1.18),
        #           run_time=3.0)
        # self.wait()

        # # self.set_view(-141, 63, 0, (-3.87, 0.05, -0.09), 2.29) #Debug view

        # s5=Dot3D(center=a6.get_end(), radius=0.02, color='$FF00FF')
        # self.play(arrows_1[3].animate.set_opacity(0.0),
        #           arrows_2[3].animate.set_opacity(0.0),
        #           FadeIn(s5),
        #           # self.animate_view(175, 47, 0, (-3.89, 1.49, 1.6), 3.75),
        #           run_time=2.0)
        # self.remove(s4); self.add(s4) #Occlusions
        # self.wait()

        # # --- Step 5 - probably our last step
        # # --- Ok, I don't think step 5 actually adds much value. 
        # # arrows_1[4].move_to(s5.get_center(), aligned_edge=LEFT)
        # # arrows_1[4].rotate(-DEGREES*90, axis=arrows_1[4].get_end()-arrows_1[4].get_start())
        # # # arrows_1[4].shift([0,0,-0.05])
        
        # # arrows_2[4].move_to(s5.get_center(), aligned_edge=LEFT)
        # # arrows_2[4].rotate(-DEGREES*90, axis=arrows_2[4].get_end()-arrows_2[4].get_start())
        # # arrows_2[4].shift([0,0.02,0.005])

        # # self.add(arrows_1[4], arrows_2[4])

        # # a7 =Arrow(start=[arrows_1[4].get_corner(LEFT)[0]+0.00, arrows_1[4].get_corner(LEFT)[1]+0.00, arrows_1[4].get_corner(OUT)[2]],
        # #           end=[arrows_1[4].get_corner(RIGHT)[0], arrows_2[4].get_corner(UP)[1], arrows_2[4].get_corner(IN)[2]], 
        # #           fill_color='#FF00FF', thickness=3.0, tip_width_ratio=3, buff=0, max_width_to_length_ratio=0.2)
        # # # self.add(a7)
        # # self.wait()
        
        # # self.play(TransformFromCopy(arrows_1[4], a7), 
        # #           TransformFromCopy(arrows_2[4], a7),
        # #           self.animate_view(179, 54, 0, (-3.43, 1.82, 1.18), 1.0),
        # #           run_time=3.0)
        # # self.wait()

        # # s6=Dot3D(center=a7.get_end(), radius=0.01, color='$FF00FF')
        # # self.play(arrows_1[4].animate.set_opacity(0.0),
        # #           arrows_2[4].animate.set_opacity(0.0),
        # #           FadeIn(s6),
        # #           # self.animate_view(179, 54, 0, (-3.39, 1.87, 1.12), 0.72),
        # #           run_time=2.0)
        # # self.remove(s5); self.add(s5) #Occlusions
        # # self.wait()

        # #Ok this scene could be tuned more for sure, but it's not terrible - it's time to go do the rewrite! 


        # self.play(self.animate_view(360-135, 69, 0, (-3.49, 1.61, 1.42), 2.91), run_time=6)

        # self.embed()
        # self.wait(20)








class P33_35_2D(WelchScene):
    def construct(self):
        '''
        Let's begin by running 2d numerical-ish gradient descent and visiaulizing it in 2 1d panes
        Curves should update as we move in the 2d space. 
        Ideas where initially developed in p33_35_sketch.py - lots of notes there too. 
        '''

        #Now that grads are computed, start viz. 
        x_axis_1=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
                            x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_1=WelchYAxis(y_min=0.3, y_max=2.2, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        axes_1=m.VGroup(x_axis_1, y_axis_1)

        x_label_1 = math_tex(r'\theta_{1}', font_size=30).set_color(CHILL_BROWN)
        y_label_1 = math_tex('Loss', font_size=25).set_color(CHILL_BROWN)
        x_label_1.next_to(x_axis_1, m.RIGHT, buff=0.05)
        y_label_1.next_to(y_axis_1, m.UP, buff=0.05)

        x_axis_2=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
                            x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_2=WelchYAxis(y_min=0.3, y_max=2.2, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        axes_2=m.VGroup(x_axis_2, y_axis_2)

        x_label_2 = math_tex(r'\theta_{2}', font_size=30).set_color(CHILL_BROWN)
        y_label_2 = math_tex('Loss', font_size=25).set_color(CHILL_BROWN)
        x_label_2.next_to(x_axis_2, m.RIGHT, buff=0.05)
        y_label_2.next_to(y_axis_2, m.UP, buff=0.05) #not sure I need this.



        ## Get all Curves, Points, Arrows, and Lines, then Group into panels
        curves_1=m.VGroup(); curves_2=m.VGroup()
        points_1=m.VGroup(); points_2=m.VGroup()
        arrows_1=m.VGroup(); arrows_2=m.VGroup()
        lines_1=m.VGroup(); lines_2=m.VGroup()

        #Map all points
        descent_points_mapped_1=np.zeros_like(descent_points)
        descent_points_mapped_1[:,0]=map_to_canvas(descent_points[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        descent_points_mapped_1[:,1]=map_to_canvas(descent_points[:,2], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        descent_points_mapped_2=np.zeros_like(descent_points)
        descent_points_mapped_2[:,0]=map_to_canvas(descent_points[:,1], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        descent_points_mapped_2[:,1]=map_to_canvas(descent_points[:,2], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        arrow_end_points_1_mapped=np.zeros_like(arrow_end_points_1)
        arrow_end_points_1_mapped[:,0]=map_to_canvas(arrow_end_points_1[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        arrow_end_points_1_mapped[:,1]=map_to_canvas(arrow_end_points_1[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        arrow_end_points_2_mapped=np.zeros_like(arrow_end_points_2)
        arrow_end_points_2_mapped[:,0]=map_to_canvas(arrow_end_points_2[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        arrow_end_points_2_mapped[:,1]=map_to_canvas(arrow_end_points_2[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        for i in range(len(descent_points)):
            #Curves 1
            p1 = np.array([param_surface(u, descent_points[i][1]) for u in np.linspace(-1, 4, 128)])
            points_mapped=np.zeros_like(p1)
            points_mapped[:,0]=map_to_canvas(p1[:,0], axis_min=x_axis_1.x_min, 
                                             axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
            points_mapped[:,1]=map_to_canvas(p1[:,2], axis_min=y_axis_1.y_min, 
                                             axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
            c = m.VMobject()
            c.set_points_smoothly(points_mapped)
            c.set_stroke(width=4, color=RED, opacity=0.8)
            curves_1.add(c)

            #Points 1
            p=m.Dot(descent_points_mapped_1[i], radius=0.06, fill_color=RED)
            points_1.add(p)

            # #Curves 2
            p1 = np.array([param_surface(descent_points[i][0], v) for v in np.linspace(-1, 4, 128)])
            points_mapped=np.zeros_like(p1)
            points_mapped[:,0]=map_to_canvas(p1[:,1], axis_min=x_axis_1.x_min, 
                                             axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
            points_mapped[:,1]=map_to_canvas(p1[:,2], axis_min=y_axis_1.y_min, 
                                             axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
            c = m.VMobject()
            c.set_points_smoothly(points_mapped)
            c.set_stroke(width=4, color=BLUE, opacity=0.8)
            curves_2.add(c)

            #Points 2
            p=m.Dot(descent_points_mapped_2[i], radius=0.06, fill_color=BLUE)
            points_2.add(p)
          
            if i>0:
                lines_1.add(m.Line(start=[descent_points_mapped_1[i-1][0], descent_points_mapped_1[i-1][1], 0],
                                   end=[descent_points_mapped_1[i][0], descent_points_mapped_1[i][1], 0], 
                                   color=RED, buff=0, stroke_width=1.5))   
                arrows_1.add(diagram_arrow(start=[descent_points_mapped_1[i-1][0], descent_points_mapped_1[i-1][1], 0],
                                     end=arrow_end_points_1_mapped[i-1], fill_color=RED, 
                                     thickness=3.0, tip_width_ratio=5, buff=0))
                lines_2.add(m.Line(start=[descent_points_mapped_2[i-1][0], descent_points_mapped_2[i-1][1], 0],
                                   end=[descent_points_mapped_2[i][0], descent_points_mapped_2[i][1], 0], 
                                   color=BLUE, buff=0, stroke_width=1.5))   
                arrows_2.add(diagram_arrow(start=[descent_points_mapped_2[i-1][0], descent_points_mapped_2[i-1][1], 0],
                                     end=arrow_end_points_2_mapped[i-1], fill_color=BLUE, 
                                     thickness=3.0, tip_width_ratio=5, buff=0))

        ## Test viz as I go here            
        panel_1=m.VGroup(axes_1, x_label_1, y_label_2, curves_1, points_1, arrows_1, lines_1)
        panel_1.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        panel_2=m.VGroup(axes_2, x_label_2, y_label_2, curves_2, points_2, arrows_2, lines_2)
        panel_2.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)

        panel_1_shift=[-5, 0, 2.0]
        panel_2_shift=[-5, 0, -2.0]
        panel_1.shift(panel_1_shift)
        panel_2.shift(panel_2_shift)

        self.add(panel_1, panel_2)

        self.set_view(0, 89, 0, (-0.46, 0.0, 1.36), 8.97)
        self.wait()


        #WONDER IF I CAN DO A COOL TRANSFOMR INTO ANIMATION DEAL AS THE CUVES CHANGE
        #
        # Ok, I think all 2d pieces are in place here, after i rewrite these paragraphs I'll come figure out the animatino. 
        # Ok so I'm kinda 50/50 on the "3 panel animation" - but I think it's worth a try, and shouldn't be to hard 
        # to render out to of the panels here. 



        pass
        self.wait(20)

from functools import partial

CHILL_BROWN='#948979'
BLUE='#65c8d0'
# WELCH_ASSET_PATH='/home/zedaes/Documents/Welch Labs/manim_videos/welch_assets'
WELCH_ASSET_PATH='/Users/stephen/manim/videos/welch_assets'

    # def copy_frame_positioning(self):
    #     frame = self.camera.frame
    #     center = frame.get_center()
    #     height = frame.height
    #     angles = frame.get_euler_angles()

    #     call = f"reorient("
    #     theta, phi, gamma = (angles / DEG).astype(int)
    #     call += f"{theta}, {phi}, {gamma}"
    #     if any(center != 0):
    #         call += f", {tuple(np.round(center, 2))}"
    #     if height != FRAME_HEIGHT:
    #         call += ", {:.2f}".format(height)
    #     call += ")"
    #     pyperclip.copy(call)


def generate_nice_ticks(min_val, max_val, min_ticks=3, max_ticks=16, ignore=[0]):
    """
    Generate a list of nice-looking tick values between min_val and max_val,
    and return extended range values for the full axis.
    
    Args:
        min_val (float): Minimum value for the data range
        max_val (float): Maximum value for the data range
        min_ticks (int): Minimum number of ticks desired
        max_ticks (int): Maximum number of ticks desired
        ignore (list): List of values to exclude from the ticks
        
    Returns:
        tuple: (tick_values, axis_min, axis_max)
            - tick_values (list): A list of tick values
            - axis_min (float): Suggested minimum value for the axis (one tick before min_val)
            - axis_max (float): Suggested maximum value for the axis (one tick after max_val)
    """
    # Ensure min_val < max_val
    if min_val > max_val:
        min_val, max_val = max_val, min_val
        
    # Handle case where min_val and max_val are equal or very close
    if abs(max_val - min_val) < 1e-10:
        # Create a small range around the value
        min_val = min_val - 1
        max_val = max_val + 1
    
    # Find the appropriate order of magnitude for the tick spacing
    range_val = max_val - min_val
    power = np.floor(np.log10(range_val))
    
    # Try different multiples of the base power of 10
    possible_step_sizes = [10**power, 5 * 10**(power-1), 2 * 10**(power-1), 10**(power-1)]
    
    # Find the first step size that gives us fewer than max_ticks
    chosen_step = possible_step_sizes[0]  # Default to the largest step
    
    for step in possible_step_sizes:
        # Calculate how many ticks we'd get with this step size
        first_tick = np.ceil(min_val / step) * step
        last_tick = np.floor(max_val / step) * step
        
        # Count ticks, excluding ignored values
        num_ticks = 0
        current = first_tick
        while current <= last_tick * (1 + 1e-10):
            if not any(abs(current - ignored_val) < 1e-10 for ignored_val in ignore):
                num_ticks += 1
            current += step
        
        if min_ticks <= num_ticks <= max_ticks:
            chosen_step = step
            break
        elif num_ticks > max_ticks:
            # If we have too many ticks, stop and use the previous step size
            break
    
    # Calculate the first tick at or below min_val
    first_tick = np.floor(min_val / chosen_step) * chosen_step
    
    # Calculate the last tick at or above max_val
    last_tick = np.ceil(max_val / chosen_step) * chosen_step
    
    # Calculate one tick before first_tick for axis_min
    axis_min = first_tick - chosen_step
    
    # Calculate one tick after last_tick for axis_max
    axis_max = last_tick + chosen_step
    
    # Generate the tick values that fall within the data range, excluding ignored values
    ticks = []
    current = np.ceil(min_val / chosen_step) * chosen_step
    
    while current <= max_val * (1 + 1e-10):  # Add a small epsilon to handle floating point errors
        # Only add the tick if it's not in the ignore list
        if not any(abs(current - ignored_val) < 1e-10 for ignored_val in ignore):
            ticks.append(float(current))  # Convert to float to avoid numpy types
        current += chosen_step
    
    # If we still have too few ticks, try the next smaller step size
    if len(ticks) < min_ticks and possible_step_sizes.index(chosen_step) < len(possible_step_sizes) - 1:
        return generate_nice_ticks(min_val, max_val, min_ticks, max_ticks, ignore)
    
    return ticks, float(axis_min), float(axis_max)


class WelchXAxis(m.VGroup):
    def __init__(
        self,
        x_min=0,
        x_max=6, 
        x_ticks=[1, 2, 3, 4, 5],  # Default tick values
        x_tick_height=0.15,        # Default tick height
        x_label_font_size=24,     # Default font size
        stroke_width=3,           # Default stroke width
        color=CHILL_BROWN,        # Default color (using predefined BROWN)
        arrow_tip_scale=0.1, 
        axis_length_on_canvas=5,
        include_tip=True,
        **kwargs
    ):
        
        m.VGroup.__init__(self, **kwargs)

        # Store parameters
        self.x_ticks = x_ticks
        self.x_tick_height = x_tick_height
        self.x_label_font_size = x_label_font_size
        self.stroke_width = stroke_width
        self.axis_color = color
        self.arrow_tip_scale=arrow_tip_scale
        self.x_min = x_min
        self.x_max = x_max
        self.axis_length_on_canvas=axis_length_on_canvas
        self.include_tip=include_tip

        self.axis_to_canvas_scale=(self.x_max-self.x_min)/axis_length_on_canvas
        self.x_ticks_scaled=(np.array(x_ticks)-self.x_min)/self.axis_to_canvas_scale

        # Create the basic components
        self._create_axis_line()
        self._create_ticks()
        self._create_labels()
        
    def _create_axis_line(self):
        
        # Create a line for the x-axis
        axis_line = m.Line(
            start=np.array([0, 0, 0]),
            end=np.array([self.axis_length_on_canvas, 0, 0]),
            color=self.axis_color,
            stroke_width=self.stroke_width
        )
        if self.include_tip:
            arrow_tip=m.SVGMobject(WELCH_ASSET_PATH+'/welch_arrow_tip_1.svg')
            arrow_tip.scale(self.arrow_tip_scale)
            arrow_tip.move_to([self.axis_length_on_canvas, 0, 0])
            axis_line = m.VGroup(axis_line, arrow_tip)

        self.add(axis_line)
        
    def _create_ticks(self):
        self.ticks = m.VGroup()
        
        for x_val in self.x_ticks_scaled:
            tick = m.Line(
                start=np.array([x_val, 0, 0]),
                end=np.array([x_val, -self.x_tick_height, 0]),  # Ticks extend downward
                color=self.axis_color,
                stroke_width=self.stroke_width
            )
            self.ticks.add(tick)
            
        self.add(self.ticks)
        
    def _create_labels(self):
        self.labels = m.VGroup()
        
        for x_val, x_val_label in zip(self.x_ticks_scaled, self.x_ticks):
            # In 3B1B's manim, use TexMobject instead of MathTex
            label = math_tex(str(round(x_val_label, 4)))
            label.scale(self.x_label_font_size / 48)  # Approximate scaling
            label.set_color(self.axis_color)
            label.next_to(
                np.array([x_val, -self.x_tick_height, 0]),
                m.DOWN,
                buff=0.1
            )
            self.labels.add(label)
            
        self.add(self.labels)
    
    # Helper method to get the axis line
    def get_axis_line(self):
        return self.axis_line
    
    # Helper method to get ticks
    def get_ticks(self):
        return self.ticks
    
    # Helper method to get labels
    def get_labels(self):
        return self.labels

    def map_to_canvas(self, value, axis_start=0):
        value_scaled=(value-self.x_min)/(self.x_max-self.x_min)
        return (value_scaled+axis_start)*self.axis_length_on_canvas


class WelchYAxis(m.VGroup):
    def __init__(
        self,
        y_min=0,
        y_max=6, 
        y_ticks=[1, 2, 3, 4, 5],  # Default tick values
        y_tick_width=0.15,        # Default tick width
        y_label_font_size=24,     # Default font size
        stroke_width=3,           # Default stroke width
        color=CHILL_BROWN,        # Default color
        arrow_tip_scale=0.1,
        axis_length_on_canvas=5,
        include_tip=True,
        **kwargs
    ):
        m.VGroup.__init__(self, **kwargs)
        
        # Store parameters
        self.y_ticks = y_ticks
        self.y_tick_width = y_tick_width
        self.y_label_font_size = y_label_font_size
        self.stroke_width = stroke_width
        self.axis_color = color
        self.arrow_tip_scale = arrow_tip_scale
        self.y_min = y_min
        self.y_max = y_max
        self.axis_length_on_canvas = axis_length_on_canvas
        self.include_tip=include_tip
        
        self.axis_to_canvas_scale = (self.y_max - self.y_min) / axis_length_on_canvas
        self.y_ticks_scaled = (np.array(y_ticks)-self.y_min)/ self.axis_to_canvas_scale
        
        # Create the basic components
        self._create_axis_line()
        self._create_ticks()
        self._create_labels()
        
    def _create_axis_line(self):
        # Create a line for the y-axis
        axis_line = m.Line(
            start=np.array([0, 0, 0]),
            end=np.array([0, self.axis_length_on_canvas, 0]),
            color=self.axis_color,
            stroke_width=self.stroke_width
        )
        
        # Add SVG arrow tip at the end
        if self.include_tip:
            arrow_tip = m.SVGMobject(WELCH_ASSET_PATH+'/welch_arrow_tip_1.svg')
            arrow_tip.scale(self.arrow_tip_scale)
            arrow_tip.move_to([0, self.axis_length_on_canvas, 0])
            # Rotate the arrow tip to point upward
            arrow_tip.rotate(m.PI/2)  # Rotate 90 degrees to point up
            axis_line = m.VGroup(axis_line, arrow_tip)


        self.add(axis_line)
        
    def _create_ticks(self):
        self.ticks = m.VGroup()
        
        for y_val in self.y_ticks_scaled:
            tick = m.Line(
                start=np.array([0, y_val, 0]),
                end=np.array([-self.y_tick_width, y_val, 0]),  # Ticks extend to the left
                color=self.axis_color,
                stroke_width=self.stroke_width
            )
            self.ticks.add(tick)
            
        self.add(self.ticks)
        
    def _create_labels(self):
        self.labels = m.VGroup()
        
        for y_val, y_val_label in zip(self.y_ticks_scaled, self.y_ticks):
            # Use Tex for labels
            label = math_tex(str(round(y_val_label,5)))
            label.scale(self.y_label_font_size / 48)  # Approximate scaling
            label.set_color(self.axis_color)
            label.next_to(
                np.array([-self.y_tick_width, y_val, 0]),
                m.LEFT,
                buff=0.1
            )
            self.labels.add(label)
            
        self.add(self.labels)
    
    # Helper method to get the axis line
    def get_axis_line(self):
        return self.axis_line
    
    # Helper method to get ticks
    def get_ticks(self):
        return self.ticks
    
    # Helper method to get labels
    def get_labels(self):

        return self.labels
    def map_to_canvas(self, value, axis_start=0):
        value_scaled=(value-self.y_min)/(self.y_max-self.y_min)
        return (value_scaled+axis_start)*self.axis_length_on_canvas
























