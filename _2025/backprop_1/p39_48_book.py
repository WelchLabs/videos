from welch_paths import welch_path
import manimgx as m
from welch_scene import manual_camera_interpolation
from landscape_geometry import get_pivot_and_scale
from landscape_geometry import get_numerical_gradient
import numpy as np
from welch_axes import WelchXAxis, WelchYAxis
from welch_scene import WelchScene
from welch_surface import parametric_surface, set_surface_shading, textured_surface
from welch_text import math_tex
from functools import partial
from tqdm import tqdm

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

loss_curve_1=np.load(welch_path('backpropagation/hackin/apr_24_5/all_execpt_embedding_random_64.npy'))
loss_curve_2=np.load(welch_path('backpropagation/hackin/apr_24_5/all_execpt_embedding_random_51.npy'))
loss_curve_3=np.load(welch_path('backpropagation/hackin/apr_24_7/all_execpt_embedding_pretrained_19.npy'))
loss_curve_4=np.load(welch_path('backpropagation/hackin/apr_24_7/all_execpt_embedding_pretrained_27.npy'))

wormhole_dir=welch_path('backpropagation/hackin/wormhole_merged/')
alphas_1=np.linspace(-2.5, 2.5, 512)
loss_2d_1=np.load(wormhole_dir+'000.npy')

class Dot3D(m.Sphere):
    def __init__(self, center=m.ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(center)


##Only need to run this when the underlying npy file changes
# import matplotlib.pyplot as plt
# plt.figure(frameon=False)
# ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
# ax.set_axis_off()
# plt.gcf().add_axes(ax)
# plt.imshow(np.rot90(loss_2d_1.T)) #have to transpose if transposing u and v and param_surface_1
# plt.savefig(wormhole_dir+'loss_2d_1.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()

# plt.clf()
# plt.figure(frameon=False)
# ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
# ax.set_axis_off()
# plt.gcf().add_axes(ax)
# plt.imshow(np.rot90(loss_2d_1.T)[128:-128, 128:-128])
# plt.savefig('loss_2d_1_inner.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()

# Comman for rendering all scenes
# manimgx render /Users/stephen/manim/videos/_2025/backprop_1/p39_48.py P39_47 P48_moving_view_1 P48_moving_view_2 P48_moving_view_3 P48_fixed_view -w


def param_surface_1(u, v):
    u_idx = np.abs(alphas_1 - u).argmin()
    v_idx = np.abs(alphas_1 - v).argmin()
    try:
        # z = loss_2d_1[u_idx, v_idx]
        z = 0.07*loss_2d_1[v_idx, u_idx] #Add vertical scaling here?
    except IndexError:
        z = 0
    return np.array([u, v, z])

def param_surface_2(u, v, surf_array):
    u_idx = np.abs(alphas_1 - u).argmin()
    v_idx = np.abs(alphas_1 - v).argmin()
    try:
        # z = loss_2d_1[u_idx, v_idx]
        z = 0.07*surf_array[v_idx, u_idx] #Add vertical scaling here?
    except IndexError:
        z = 0
    return np.array([u, v, z])





class P39_47_book(WelchScene):
    def construct(self):
        '''
        Ok this time I'm goint to start with setting up the surface and gridlines, at the origin. 
        Then I'll go back to the 2d curves and make sure I can land on the 2d surface nicely. 
        '''

        ## ----- 1. Initial Surface Setup ----- ##
        surface = parametric_surface(
            param_surface_1,  
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=(512, 512),
        )

        ts = textured_surface(surface, wormhole_dir+'loss_2d_1.png')
        set_surface_shading(ts, 0.0, 0.1, 0)

        num_lines = 64  # Number of gridlines in each direction
        num_points = 512  # Number of points per line
        u_gridlines = m.VGroup()
        v_gridlines = m.VGroup()
        u_values = np.linspace(-2.5, 2.5, num_lines)
        v_points = np.linspace(-2.5, 2.5, num_points)
        for u in u_values:
            points = [param_surface_1(u, v) for v in v_points]
            line = m.VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
            u_gridlines.add(line)

        u_points = np.linspace(-2.5, 2.5, num_points)
        for v in u_values:  # Using same number of lines for both directions
            points = [param_surface_1(u, v) for u in u_points]
            line = m.VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
            v_gridlines.add(line)
 
        # self.add(ts)
        # self.add(u_gridlines, v_gridlines)

        ## ----- 2. 2D Curves ----- ##
        x_axis_1=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=5)
        y_axis_1=WelchYAxis(y_min=8, y_max=16, y_ticks=[8, 9, 10, 11, 12, 13, 14, 15 ], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)

        x_label_1 = math_tex(r'\alpha', font_size=28).set_color(CHILL_BROWN)
        y_label_1 = math_tex('Loss', font_size=22).set_color(CHILL_BROWN)
        x_label_1.next_to(x_axis_1, m.RIGHT, buff=0.05)
        y_label_1.next_to(y_axis_1, m.UP, buff=0.08)

        mapped_x_1=x_axis_1.map_to_canvas(loss_curve_1[0,:]) 
        mapped_y_1=y_axis_1.map_to_canvas(loss_curve_1[1,:])

        curve_1=m.VMobject()
        curve_1.set_points_smoothly(np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T)
        curve_1.set_stroke(width=4, color=YELLOW, opacity=1.0)

        axes_1=m.VGroup(x_axis_1, y_axis_1, x_label_1, y_label_1, curve_1)
        axes_1.move_to([-12.5, 0, 0]) #Start way over here, so the final 3d column of plots land close to origin
        axes_1.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        self.set_view(0, 89, 0, (-12.43, -0.01, 0.0), 4.99)
        self.add(x_axis_1, y_axis_1, x_label_1, y_label_1)
        self.wait(0)
        self.play(m.Create(curve_1), run_time=5)

        self.wait() 

        x_axis_2=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=5)
        y_axis_2=WelchYAxis(y_min=8, y_max=16, y_ticks=[8, 9, 10, 11, 12, 13, 14, 15 ], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)

        x_label_2 = math_tex(r'\alpha', font_size=28).set_color(CHILL_BROWN)
        y_label_2 = math_tex('Loss', font_size=22).set_color(CHILL_BROWN)
        x_label_2.next_to(x_axis_2, m.RIGHT, buff=0.05)
        y_label_2.next_to(y_axis_2, m.UP, buff=0.08)

        mapped_x_2=x_axis_2.map_to_canvas(loss_curve_2[0,:]) 
        mapped_y_2=y_axis_2.map_to_canvas(loss_curve_2[1,:])

        curve_2=m.VMobject()
        curve_2.set_points_smoothly(np.vstack((mapped_x_2, mapped_y_2, np.zeros_like(mapped_x_2))).T)
        curve_2.set_stroke(width=4, color=YELLOW, opacity=1.0)

        axes_2=m.VGroup(x_axis_2, y_axis_2, x_label_2, y_label_2, curve_2)
        axes_2.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        axes_2.move_to([-6.26, 0, 0])

        self.play(self.animate_view(0, 89, 0, (-9.28, 0.05, -0.11), 7.80),
                  m.FadeIn(m.VGroup(x_axis_2, y_axis_2, x_label_2, y_label_2)),
                  m.Create(curve_2),
                  run_time=3.0)
        self.wait()

        x_axis_3=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=5)
        y_axis_3=WelchYAxis(y_min=0, y_max=40, y_ticks=[0, 5, 10, 15, 20, 25, 30, 35], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)

        x_label_3 = math_tex(r'\alpha', font_size=28).set_color(CHILL_BROWN)
        y_label_3 = math_tex('Loss', font_size=22).set_color(CHILL_BROWN)
        x_label_3.next_to(x_axis_3, m.RIGHT, buff=0.05)
        y_label_3.next_to(y_axis_3, m.UP, buff=0.08)

        mapped_x_3=x_axis_3.map_to_canvas(loss_curve_3[0,:]) 
        mapped_y_3=y_axis_3.map_to_canvas(loss_curve_3[1,:])

        curve_3=m.VMobject()
        curve_3.set_points_smoothly(np.vstack((mapped_x_3, mapped_y_3, np.zeros_like(mapped_x_3))).T)
        curve_3.set_stroke(width=4, color=BLUE, opacity=1.0)

        axes_3=m.VGroup(x_axis_3, y_axis_3, x_label_3, y_label_3, curve_3)
        axes_3.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        axes_3.move_to([-6.25, 0, 0])


        x_axis_4=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=5)
        y_axis_4=WelchYAxis(y_min=0, y_max=40, y_ticks=[0, 5, 10, 15, 20, 25, 30, 35], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)

        x_label_4 = math_tex(r'\alpha', font_size=28).set_color(CHILL_BROWN)
        y_label_4 = math_tex('Loss', font_size=22).set_color(CHILL_BROWN)
        x_label_4.next_to(x_axis_4, m.RIGHT, buff=0.05)
        y_label_4.next_to(y_axis_4, m.UP, buff=0.08)

        mapped_x_4=x_axis_4.map_to_canvas(loss_curve_4[0,:]) 
        mapped_y_4=y_axis_4.map_to_canvas(loss_curve_4[1,:])

        curve_4=m.VMobject()
        curve_4.set_points_smoothly(np.vstack((mapped_x_4, mapped_y_4, np.zeros_like(mapped_x_3))).T)
        curve_4.set_stroke(width=4, color=BLUE, opacity=1.0)

        axes_4=m.VGroup(x_axis_4, y_axis_4, x_label_4, y_label_4, curve_4)
        axes_4.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        axes_4.move_to([-6.25, 0, -3.75])

        self.wait()

        self.play(axes_2.animate.move_to([-12.5, 0, -3.75]),
                 self.animate_view(0, 89, 0, (-9.24, -3.5, -2.04), 5.85),
                 run_time=2.0)

        self.play(m.FadeIn(m.VGroup(x_axis_3, y_axis_3, x_label_3, y_label_3)),
                  m.FadeIn(m.VGroup(x_axis_4, y_axis_4, x_label_4, y_label_4)),
                  m.Create(curve_3),
                  m.Create(curve_4),
                  run_time=4)

        self.wait()

        ## Ok now axes 5 & 6. 
        slice_1=loss_2d_1[255, :]
        slice_2=loss_2d_1[:, 255]

        x_axis_5=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=5)
        y_axis_5=WelchYAxis(y_min=0, y_max=25, y_ticks=[0, 5, 10, 15, 20], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)

        x_label_5 = math_tex(r'\alpha', font_size=28).set_color(CHILL_BROWN)
        y_label_5 = math_tex('Loss', font_size=22).set_color(CHILL_BROWN)
        x_label_5.next_to(x_axis_5, m.RIGHT, buff=0.05)
        y_label_5.next_to(y_axis_5, m.UP, buff=0.08)

        mapped_x_5=x_axis_5.map_to_canvas(alphas_1) 
        mapped_y_5=y_axis_5.map_to_canvas(slice_1)

        curve_5=m.VMobject()
        curve_5.set_points_smoothly(np.vstack((mapped_x_5, mapped_y_5, np.zeros_like(mapped_x_5))).T)
        curve_5.set_stroke(width=4, color="#FF00FF", opacity=1.0)

        axes_5=m.VGroup(x_axis_5, y_axis_5, x_label_5, y_label_5, curve_5)
        axes_5.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        axes_5.move_to([0, 0, 0])


        x_axis_6=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=5)
        y_axis_6=WelchYAxis(y_min=0, y_max=25, y_ticks=[0, 5, 10, 15, 20], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)

        x_label_6 = math_tex(r'\alpha', font_size=28).set_color(CHILL_BROWN)
        y_label_6 = math_tex('Loss', font_size=22).set_color(CHILL_BROWN)
        x_label_6.next_to(x_axis_6, m.RIGHT, buff=0.05)
        y_label_6.next_to(y_axis_6, m.UP, buff=0.08)

        mapped_x_6=x_axis_6.map_to_canvas(alphas_1) 
        mapped_y_6=y_axis_6.map_to_canvas(slice_2)

        curve_6=m.VMobject()
        curve_6.set_points_smoothly(np.vstack((mapped_x_6, mapped_y_6, np.zeros_like(mapped_x_6))).T)
        curve_6.set_stroke(width=4, color="#FF00FF", opacity=1.0)

        axes_6=m.VGroup(x_axis_6, y_axis_6, x_label_6, y_label_6, curve_6)
        axes_6.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        axes_6.move_to([0, 0, -3.75])

        self.wait()
        # self.add(axes_5, axes_6)
        self.play(m.FadeIn(m.VGroup(x_axis_5, y_axis_5, x_label_5, y_label_5)),
          m.FadeIn(m.VGroup(x_axis_6, y_axis_6, x_label_6, y_label_6)),
          m.Create(curve_5),
          m.Create(curve_6),
          self.animate_view(0, 89, 0, (-6.26, -3.5, -2.13), 8.15),
          run_time=6)
        
        self.wait()


        ## ----- 3. 2D -> 3D Transition ----- ##
        pivot_x,scale_x=get_pivot_and_scale(axis_min=x_axis_5.x_min, axis_max=x_axis_5.x_max, 
                                        axis_end=x_axis_5.axis_length_on_canvas)
        pivot_y,scale_y=get_pivot_and_scale(axis_min=y_axis_5.y_min, axis_max=y_axis_5.y_max, 
                                        axis_end=y_axis_5.axis_length_on_canvas)

        self.wait()
        #Anmation 1
        self.play(axes_1.animate.set_opacity(0),
                axes_2.animate.set_opacity(0),
                axes_3.animate.set_opacity(0),
                axes_4.animate.set_opacity(0),
                y_axis_6.animate.set_opacity(0),
                y_label_6.animate.set_opacity(0),
                y_axis_5.animate.set_opacity(0), #might be nice to keep these, but difficult with scaling
                y_label_5.animate.set_opacity(0),
                x_label_5.animate.set_opacity(0),
                x_label_6.animate.set_opacity(0),
                axes_6[0][-1][2].animate.set_opacity(0), #Zero tick label on axis 6
                curve_5.animate.scale([1/scale_x, 1/scale_x, 0.07/scale_y]),
                curve_6.animate.scale([1/scale_x, 1/scale_x, 0.07/scale_y]),
                axes_6[0].animate.shift([0,0,0.4]),
                axes_5[0].animate.shift([0,0,0.4]),
                self.animate_view(0, 90, 0, (0.0, -3.5, -2.24), 5.28),
                run_time=3.0)

        self.wait()

        #animation 2
        self.play(curve_5.animate.move_to([0,0,0.72]),
                 curve_6.animate.move_to([0,0,0.65]).rotate(90*m.DEGREES, axis=[0,0,1]),
                 axes_6[0].animate.move_to([0,0,-0.2]).rotate(90*m.DEGREES, axis=[0,0,1]),
                 axes_5[0].animate.move_to([0,0,-0.2]),
                 # self.animate_view(9, 74, 0, (0.6, -3.05, 0.58), 4.05),
                 self.animate_view(36, 64, 0, (-0.07, 0.22, 0.23), 6.77),
                 run_time=5.0)

        self.wait()
        self.play(m.Create(u_gridlines),
                  m.Create(v_gridlines),
                  self.animate_view(42, 58, 0, (-0.03, 0.06, 0.02), 6.19),
                  run_time=4.0)

        ts.set_opacity(0.0)
        self.add(ts)
        self.add(u_gridlines, v_gridlines) #Occlusions
        self.add(curve_5, curve_6) #Occlusions
        self.play(ts.animate.set_opacity(1.0),
                  self.animate_view(51, 47, 0, (0.16, 0.06, -0.14), 6.10),
                  run_time=3.0)

        self.wait(0)


        self.play(axes_6[0].animate.set_opacity(0.0),
                  axes_5[0].animate.set_opacity(0.0),
                  curve_5.animate.set_opacity(0.0),
                  curve_6.animate.set_opacity(0.0),
                  self.animate_view(143, 26, 0, (0.08, 0.19, -0.04), 7.38),
                  run_time=6.0)
        self.wait()


        ## ---- 4. "Drop point in and, add gradient arrow, show global min. " ---- #


        self.play(self.animate_view(122, 58, 0, (-1.15, -0.02, 0.11), 5.42), run_time=4.0)
        self.wait()

        starting_coords=[-0.7,0.95]  #[-0.7,1.0]  #
        starting_point=param_surface_1(*starting_coords)
        s1=Dot3D(center=starting_point, radius=0.06, color='$FF00FF')
        s1.shift([0,0,1]) #move up so we can "drop it in to skate bro"
        
        self.add(s1)

        self.play(s1.animate.shift([0,0,-1]),
                  # self.animate_view(120, 39, 0, (-0.89, 0.27, -0.07), 4.05),
                  run_time=2.0)

        self.wait()

        self.play(self.animate_view(124, 37, 0, (-0.96, 0.01, 0.23), 3.41), run_time=4.0)
        self.wait()

        # Ok now I gotta add a little downhill arrow -> Hmm illustrator maybe?
        # yeah let's assume I'm going to add a gradient arrow in illustrator
        # From here then, I want to try to run some gradient descent from this location. 
        # Then in a bit I switch to the other location? We'll see how that feels.
        # Ok yeah so think we drive this one point to a single local minima, then zoom out to show a bunch more
        # Then start from another point to show tunneling I think


        num_steps=128 # I think it gest stuck around 30-40 at lr -0.01 - play with this for smoother/longer animation
        learning_rate=3e-3
        momentum = 0.95 #lil momentum to get to a decent minima lol
        trajectory=[[starting_point[0], starting_point[1], param_surface_1(starting_point[0], starting_point[1])[2]]]
        velocity = np.zeros(2)
        for i in range(num_steps):
            g=get_numerical_gradient(param_surface_1, trajectory[-1][0], trajectory[-1][1], epsilon=0.01)
            # delta=learning_rate*np.array(g)
            # new_x=trajectory[-1][0]-delta[0]
            # new_y=trajectory[-1][1]-delta[1]
            # Update velocity with momentum
            velocity = momentum * velocity - learning_rate * np.array(g)
            
            # Update position using velocity
            new_x = trajectory[-1][0] + velocity[0]
            new_y = trajectory[-1][1] + velocity[1]

            trajectory.append([new_x, new_y, param_surface_1(new_x, new_y)[2]])

        # num_total_steps=256
        start_orientation=[124, 37, 0, (-0.96, 0.01, 0.23), 3.41]
        end_orientation=[141, 38, 0, (-0.72, 0.13, 0.06), 2.85]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=num_steps)

        self.wait()

        t = m.VMobject()
        t.set_stroke(width=6, color="#FF00FF", opacity=1.0)
        self.add(t)
        for i in range(num_steps):
            s1.move_to(trajectory[i])
            t.set_points_smoothly(trajectory[:i])
            self.set_view(*interp_orientations[i])
            self.wait(0.1)
        self.wait()


        self.play(self.animate_view(90, 0, 0, (0.04, -0.02, 0.0), 6.80), run_time=6.0)
        self.wait()

        self.play(t.animate.set_opacity(0.0),
                  s1.animate.set_opacity(0.0))
        self.wait()
        self.remove(s1)

        ## Now label some local minima from this overhead view in illustrator bro!

        ## ----- Second Descent, very slow ----- ##

        starting_coords=[0.05,-0.9] #[0.1,-0.8] is pretty good, [0.05,-0.9] is a bit better
        starting_point=param_surface_1(*starting_coords)
     
        s1=Dot3D(center=starting_point, radius=0.06, color='$FF00FF')
        # self.add(s1)

        self.wait()
        self.play(self.animate_view(110, 38, 0, (-0.74, -0.43, 0.24), 3.00),
                  run_time=4.0)
        self.add(s1)
        self.wait()

        ##Book scene 1, before optimizing
        self.set_view(89, 29, 0, (-0.45, -0.21, 0.03), 2.67)
        self.wait(5)


        num_steps=128 # I think it gest stuck around 30-40 at lr -0.01
        learning_rate=1e-3
        momentum=0.8
        trajectory=[[starting_point[0], starting_point[1], param_surface_1(starting_point[0], starting_point[1])[2]]]
        velocity = np.zeros(2)
        for i in range(num_steps):
            g=get_numerical_gradient(param_surface_1, trajectory[-1][0], trajectory[-1][1], epsilon=0.01)
            velocity = momentum * velocity - learning_rate * np.array(g)
            new_x = trajectory[-1][0] + velocity[0]
            new_y = trajectory[-1][1] + velocity[1]
            trajectory.append([new_x, new_y, param_surface_1(new_x, new_y)[2]])

        #Ok let me go ahead and hack for a minute here on the fake/magic tunneling scence
        ending_coords=[0,0]
        ending_point=param_surface_1(*ending_coords)

        # s2=Dot3D(center=ending_point, radius=0.06, color='$FF00FF')
        # self.add(s2)

        num_steps2=90 #Plotting 1k points is kinda slow - slower than I thought 
        learning_rate_2=5e-3
        for i in range(num_steps2):
            g=-np.array([ending_coords[0]-trajectory[-1][0], ending_coords[1]-trajectory[-1][1]])
            delta=learning_rate_2*np.array(g)
            new_x=trajectory[-1][0]-delta[0]
            new_y=trajectory[-1][1]-delta[1]
            trajectory.append([new_x, new_y, param_surface_1(new_x, new_y)[2]])

        #Break into 3 parts, slower learning rate as we descent into the valley
        #Let me try a different approach at the end here, my gradient proxy shrinks as we get close
        num_steps3=128 #512 
        trajectory_waypoint=trajectory[-1]
        g=np.array([ending_coords[0]-trajectory[-1][0], ending_coords[1]-trajectory[-1][1]])
        for i in range(num_steps3):
            new_x=trajectory_waypoint[0]+(i/num_steps3)*g[0]
            new_y=trajectory_waypoint[1]+(i/num_steps3)*g[1]
            trajectory.append([new_x, new_y, param_surface_1(new_x, new_y)[2]])

        trajectory=np.array(trajectory)

        start_orientation=[110, 38, 0, (-0.74, -0.43, 0.24), 3.00]
        end_orientation=[88, 31, 0, (-0.52, -0.34, 0.09), 2.60]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=num_steps)

        self.wait()

        dot_path=m.Group()
        self.add(dot_path)
        for i in range(num_steps): #First leg -> getting stuck
            s1.move_to(trajectory[i])
            # t.set_points_smoothly(trajectory[:i])
            dot_path.add(Dot3D(center=trajectory[i], radius=0.017, color='$FF00FF'))
            # self.set_view(*interp_orientations[i]) #Don't change orientation. Book
            self.wait(0.1)
        self.wait()


        end_orientation_2=[91, 28, 0, (-0.49, -0.34, 0.08), 2.60]
        interp_orientations=manual_camera_interpolation(end_orientation, end_orientation_2, num_steps=len(trajectory)-num_steps)

        self.wait()
        for i in range(num_steps, len(trajectory)): #First leg -> getting stuck
            s1.move_to(trajectory[i])
            # t.set_points_smoothly(trajectory[:i])
            dot_path.add(Dot3D(center=trajectory[i], radius=0.017, color='$FF00FF'))
            # self.set_view(*interp_orientations[i-num_steps]) #Book
            # self.wait(0.1)
        self.wait()

        # self.play(self.animate_view(104, 11, 0, (-0.09, -0.35, -0.06), 2.60), run_time=4)


        ## ------------ Wormhole Time ---------------- ##

        ## Ok now we want to basically rewind what i just did - remove path, point back on top of hill
        ## And it's mothee fucking fuckety fucking wormhole time. 

        self.wait()
        self.play(s1.animate.set_opacity(0.0), 
                  dot_path.animate.set_opacity(0.0),
                  # self.animate_view(142, 34, 0, (-0.09, -0.77, 0.15), 3.55),
                  run_time=4.0)
        self.remove(s1)
        self.remove(dot_path)
        self.wait()

        starting_coords=[0.05,-0.9]
        starting_point=param_surface_1(*starting_coords)
        s2=Dot3D(center=starting_point, radius=0.06, color='$FF00FF')
        self.add(s2)
        self.wait()

        #Load up other surfaces to visualize
        loss_arrays=[]
        num_time_steps=66
        for i in range(num_time_steps):
            loss_arrays.append(np.load(wormhole_dir+str(i).zfill(3)+'.npy'))

        # import matplotlib.pyplot as plt
        # for i in range(num_time_steps):
        #     plt.clf()
        #     plt.figure(frameon=False)
        #     ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
        #     ax.set_axis_off()
        #     plt.gcf().add_axes(ax)
        #     plt.imshow(np.rot90(loss_arrays[i].T)) #have to transpose if transposing u and v and param_surface_1
        #     plt.savefig(wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png', bbox_inches='tight', pad_inches=0, dpi=300)
        #     plt.close()


        surfaces=m.Group()
        surf_functions=[] #Need this later to move dot around.
        grids=m.Group()
        for i in range(num_time_steps):
            surf_func=partial(param_surface_2, surf_array=loss_arrays[i])
            surf_functions.append(surf_func)
            surface = parametric_surface(
                surf_func,  
                u_range=[-2.5, 2.5],
                v_range=[-2.5, 2.5],
                resolution=(512, 512),
            )

            ts2 = textured_surface(surface, wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png')
            set_surface_shading(ts2, 0.0, 0.1, 0)
            surfaces.add(ts2)

            num_lines = 64  # Number of gridlines in each direction
            num_points = 512  # Number of points per line
            u_gridlines = m.VGroup()
            v_gridlines = m.VGroup()
            u_values = np.linspace(-2.5, 2.5, num_lines)
            v_points = np.linspace(-2.5, 2.5, num_points)
            for u in u_values:
                points = [surf_func(u, v) for v in v_points]
                line = m.VMobject()
                line.set_points_smoothly(points)
                line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
                u_gridlines.add(line)

            u_points = np.linspace(-2.5, 2.5, num_points)
            for v in u_values:  # Using same number of lines for both directions
                points = [surf_func(u, v) for u in u_points]
                line = m.VMobject()
                line.set_points_smoothly(points)
                line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
                v_gridlines.add(line)
            grids.add(m.VGroup(u_gridlines, v_gridlines))

        self.wait()


        # self.remove(ts)
        # self.remove(u_gridlines, v_gridlines) 
        # ts.animate.set_opacity(0.0)


        num_total_steps=num_time_steps*2 #Crank this for final viz
        start_orientation=[142, 34, 0, (-0.09, -0.77, 0.15), 3.55]
        end_orientation=[121, 20, 0, (0.01, -0.46, 0.57), 1.95]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=num_total_steps)

        surface_update_counter=1
        frames_per_surface_upddate=np.floor(num_total_steps/num_time_steps)
        for i in range(1, num_total_steps):
            # print(i, len(interp_orientations))
            if i%frames_per_surface_upddate==0 and surface_update_counter<len(surfaces):
                if surface_update_counter==1:
                    self.remove(ts)
                    self.remove(u_gridlines, v_gridlines) 
                else:
                    self.remove(surfaces[surface_update_counter-1])
                    self.remove(grids[surface_update_counter-1])

                self.add(surfaces[surface_update_counter])
                self.add(grids[surface_update_counter])

                new_point_coords=surf_functions[surface_update_counter](*starting_coords)
                s2.move_to(new_point_coords) #This should make point move down smoothly. 
                surface_update_counter+=1

            # print(i, len(interp_orientations))
            # self.set_view(*interp_orientations[i])
            # self.wait(0.1)

        self.wait()


        # Ok, based on the script - > I think what I really want to do here is a little camera move and then play the
        # animation again
        # This should be totally do-able in a separate viz. 

        # self.play(self.animate_view(159, 35, 0, (-0.03, -0.62, 0.65), 1.95), run_time=8.0)

        #ok ok ok ok ok let's actually go really with on the "WHat is going on here" question,
        #And then pan to the left in premiere, bring in the text, and replay the animation from this wide shot. 
        #Then we stay wide for a bit as we look at wikitext I think. 
        # self.play(self.animate_view(180, 23, 0, (-0.06, 0.09, 0.43), 5.81), run_time=10.0)

        self.wait() #Pick up in fixed animation from here
        self.wait(20)



        # self.play(self.animate_view(360-103, 12, 0, (0.01, -0.46, 0.57), 1.95), run_time=20.0) #Pan Around
        # self.wait()
        # self.play(self.animate_view(360-89, 0, 0, (0.05, -0.09, 0.59), 5.82), run_time=8.0) #Zoom out
        # self.wait()
        # self.play(self.animate_view(360-85, 99, 0, (0.05, -0.09, 0.59), 5.82), run_time=8.0) #Pan below
        # self.wait()
        # self.play(self.animate_view(84, 102, 0, (0.05, -0.09, 0.59), 5.82), run_time=10.0) #Pan around below
        # self.wait()
        # self.play(self.animate_view(89, 0, 0, (-0.03, -0.14, 0.51), 6.38), run_time=8.0) #Back to overheaad
        # self.wait()

        # Ok I think from here it might make sense to do a "cut" (finally lol) for P48. 
        # I'll play opening the wormhole again as the probabilities update
        # I think maybe no camera move this time!
        # Really closer here -> need to include "dot falling" with my animation - and get rid of otehr black dot. 


        ## ----------------------------------------------------------------------- ##
        # self.set_view(88, 41, 0, (-0.71, -0.37, 0.21), 2.60)
        # reorient(91, 28, 0, (-0.49, -0.34, 0.08), 2.60)

        

        # t = VMobject()
        # t.set_points_smoothly(trajectory)
        # t.set_stroke(width=6, color="#FF00FF", opacity=1.0)
        # self.add(t)


        # dot_path=Group()
        # for t in trajectory:
        #     dot_path.add(Dot3D(center=t, radius=0.017, color='$FF00FF'))
        # self.add(dot_path)


        
        # ts.set_opacity(1.0)
        # self.play(FadeIn(ts))

        #Aminmation 3 - running into weird perspecgive issues -> maybe it's two steps -> squish, and then move?
        # self.play(curve_5.animate.move_to([0,0,0.72]).scale([1/scale_x, 1/scale_x, 0.07/scale_y]),
        #          curve_6.animate.move_to([0,0,0.65]).scale([1/scale_x, 1/scale_x, 0.07/scale_y]).rotate(90*DEGREES, axis=[0,0,1]),
        #          axes_6[0].animate.move_to([0,0,-0.2]).rotate(90*DEGREES, axis=[0,0,1]),
        #          axes_5[0].animate.move_to([0,0,-0.2]),
        #          self.animate_view(9, 74, 0, (0.6, -3.05, 0.58), 4.05),
        #          run_time=5.0)

        # axes_5[0].move_to([0,0,-0.2])
        # axes_6[0].move_to([0,0,-0.8])
        # axes_6[0].rotate(90*DEGREES, axis=[0,0,1])


        # self.add(ts)
        # self.add(u_gridlines, v_gridlines)

        # self.add(curve_5, curve_6) #Occlusions

        # curve_5.scale([1/scale_x, 1/scale_x, 0.07/scale_y])
        # curve_6.scale([1/scale_x, 1/scale_x, 0.07/scale_y])

        # axes_6.rotate(90*DEGREES, axis=[0,0,1]) #This seems to have kidna worked out. 
        # curve_5.move_to([0,0,0.72])
        # curve_6.move_to([0,0,0.65])


        # self.set_view(35, 64, 0, (1.46, -2.03, 1.62), 4.55)


        # self.set_view(43, 51, 0, (0.07, 0.08, 0.01), 6.29)

        #Scale and move axes in a similiar way, but move them up more. 



        # axes_6[0].move_to([0,0,-1])
        # axes_5[0].move_to([0,0,-1])
        # axes_6.move_to([0, 0, 0])
        

        # self.set_view(40, 65, 0, (2.14, -2.09, -0.04), 7.91)

        # self.wait()

        # self.add(ts)
        # self.add(u_gridlines, v_gridlines)


        # self.embed()
        # self.wait(20)


class P48_moving_view_1(WelchScene):
    def construct(self):
        
        starting_coords=[0.05,-0.9]
        starting_point=param_surface_1(*starting_coords)
        s2=Dot3D(center=starting_point, radius=0.06, color='$FF00FF')
    
        #Load up other surfaces to visualize
        loss_arrays=[]
        num_time_steps=66
        print('Loading Surface Arrays')
        for i in tqdm(range(num_time_steps)):
            loss_arrays.append(np.load(wormhole_dir+str(i).zfill(3)+'.npy'))

        # import matplotlib.pyplot as plt
        # for i in range(num_time_steps):
        #     plt.clf()
        #     plt.figure(frameon=False)
        #     ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
        #     ax.set_axis_off()
        #     plt.gcf().add_axes(ax)
        #     plt.imshow(np.rot90(loss_arrays[i].T)) #have to transpose if transposing u and v and param_surface_1
        #     plt.savefig(wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png', bbox_inches='tight', pad_inches=0, dpi=300)
        #     plt.close()


        surfaces=m.Group()
        surf_functions=[] #Need this later to move dot around.
        grids=m.Group()
        print("Loading Surfaces and Gridlines...")
        for i in tqdm(range(num_time_steps)):
            surf_func=partial(param_surface_2, surf_array=loss_arrays[i])
            surf_functions.append(surf_func)
            surface = parametric_surface(
                surf_func,  
                u_range=[-2.5, 2.5],
                v_range=[-2.5, 2.5],
                resolution=(512, 512),
            )

            ts2 = textured_surface(surface, wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png')
            set_surface_shading(ts2, 0.0, 0.1, 0)
            surfaces.add(ts2)

            num_lines = 64  # Number of gridlines in each direction
            num_points = 512  # Number of points per line
            u_gridlines = m.VGroup()
            v_gridlines = m.VGroup()
            u_values = np.linspace(-2.5, 2.5, num_lines)
            v_points = np.linspace(-2.5, 2.5, num_points)
            for u in u_values:
                points = [surf_func(u, v) for v in v_points]
                line = m.VMobject()
                line.set_points_smoothly(points)
                line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
                u_gridlines.add(line)

            u_points = np.linspace(-2.5, 2.5, num_points)
            for v in u_values:  # Using same number of lines for both directions
                points = [surf_func(u, v) for u in u_points]
                line = m.VMobject()
                line.set_points_smoothly(points)
                line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
                v_gridlines.add(line)
            grids.add(m.VGroup(u_gridlines, v_gridlines))

        self.wait()


        self.add(surfaces[0])
        self.add(grids[0])
        self.add(s2)


        num_total_steps=num_time_steps*2 #Crank this for final viz
        start_orientation=[142, 34, 0, (-0.09, -0.77, 0.15), 3.55]
        # end_orientation=[131, 31, 0, (-0.12, -0.88, 0.22), 2.90]
        # end_orientation=[122, 11, 0, (-0.01, -0.28, 0.55), 2.05] #Move overhead
        end_orientation=[121, 20, 0, (0.01, -0.46, 0.57), 1.95]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=num_total_steps)

        surface_update_counter=1
        frames_per_surface_upddate=np.floor(num_total_steps/num_time_steps)
        print('frames_per_surface_upddate', frames_per_surface_upddate)
        self.wait()
        for i in range(1, num_total_steps):
            # print(i, len(interp_orientations))
            if i%frames_per_surface_upddate==0 and surface_update_counter<len(surfaces):

                self.remove(surfaces[surface_update_counter-1])
                self.remove(grids[surface_update_counter-1])
                self.add(surfaces[surface_update_counter])
                self.add(grids[surface_update_counter])
                # print('surface_update_counter', surface_update_counter)

                new_point_coords=surf_functions[surface_update_counter](*starting_coords)
                s2.move_to(new_point_coords) #This should make point move down smoothly. 
                surface_update_counter+=1

            # print(i, len(interp_orientations))
            self.set_view(*interp_orientations[i])
            self.wait(0.1)



        self.wait()
        self.play(self.animate_view(360-103, 12, 0, (0.01, -0.46, 0.57), 1.95), run_time=20.0) #Pan Around
        self.wait()
        self.play(self.animate_view(360-89, 0, 0, (0.05, -0.09, 0.59), 5.82), run_time=8.0) #Zoom out
        self.wait()
        self.play(self.animate_view(360-85, 99, 0, (0.05, -0.09, 0.59), 5.82), run_time=8.0) #Pan below
        self.wait()
        self.play(self.animate_view(84, 102, 0, (0.05, -0.09, 0.59), 5.82), run_time=10.0) #Pan around below
        self.wait()
        self.play(self.animate_view(89, 0, 0, (-0.03, -0.14, 0.51), 6.38), run_time=8.0) #Back to overheaad
        self.wait()


        self.wait(20)
        pass

class P48_experimental(WelchScene):
    '''
    Hack on single frames. 
    '''
    def construct(self):
        starting_coords=[0.05,-0.9]
        starting_point=param_surface_1(*starting_coords)
        s2=Dot3D(center=starting_point, radius=0.06, color='$FF00FF')
    
        #Load up other surfaces to visualize
        loss_arrays=[]
        num_time_steps=66
        print('Loading Surface Arrays')
        for i in tqdm(range(num_time_steps)):
            loss_arrays.append(np.load(wormhole_dir+str(i).zfill(3)+'.npy'))

        surfaces=m.Group()
        surf_functions=[] #Need this later to move dot around.
        grids=m.Group()
        print("Loading Surfaces and Gridlines...")
        
        i=65

        surf_func=partial(param_surface_2, surf_array=loss_arrays[i])
        surf_functions.append(surf_func)
        surface = parametric_surface(
            surf_func,  
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=(512, 512),
        )

        ts2 = textured_surface(surface, wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png')
        set_surface_shading(ts2, 0.0, 0.1, 0)
        surfaces.add(ts2)

        num_lines = 64  # Number of gridlines in each direction
        num_points = 512  # Number of points per line
        u_gridlines = m.VGroup()
        v_gridlines = m.VGroup()
        u_values = np.linspace(-2.5, 2.5, num_lines)
        v_points = np.linspace(-2.5, 2.5, num_points)
        for u in u_values:
            points = [surf_func(u, v) for v in v_points]
            line = m.VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
            u_gridlines.add(line)

        u_points = np.linspace(-2.5, 2.5, num_points)
        for v in u_values:  # Using same number of lines for both directions
            points = [surf_func(u, v) for u in u_points]
            line = m.VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
            v_gridlines.add(line)
        grids.add(m.VGroup(u_gridlines, v_gridlines))


        new_point_coords=surf_func(*starting_coords)
        s2.move_to(new_point_coords)

        self.add(surfaces[0])
        self.add(grids[0])
        self.add(s2)

        # self.set_view(142, 34, 0, (-0.09, -0.77, 0.15), 3.55)
        # end_orientation=[131, 31, 0, (-0.12, -0.88, 0.22), 2.90]
        # self.set_view(122, 13, 0, (0.03, -0.5, 0.57), 1.95) #Move overhead
        # self.set_view(121, 20, 0, (0.01, -0.46, 0.57), 1.95)
        # self.set_view(121, 20, 0, (0.01, -0.46, 0.57), 1.95)
        # self.set_view(159, 35, 0, (-0.03, -0.62, 0.65), 1.95)
        # self.set_view(121, 20, 0, (0.01, -0.46, 0.57), 1.95)
        # self.set_view(-180, 23, 0, (-0.06, 0.09, 0.43), 5.81)
        self.set_view(180, 23, 0, (-0.06, 0.09, 0.43), 5.81)
        pass


class P48_moving_view_2(WelchScene):
    '''
    Zoom "straight on"
    '''
    def construct(self):
        starting_coords=[0.05,-0.9]
        starting_point=param_surface_1(*starting_coords)
        s2=Dot3D(center=starting_point, radius=0.06, color='$FF00FF')
    
        #Load up other surfaces to visualize
        loss_arrays=[]
        num_time_steps=66
        print('Loading Surface Arrays')
        for i in tqdm(range(num_time_steps)):
            loss_arrays.append(np.load(wormhole_dir+str(i).zfill(3)+'.npy'))

        # import matplotlib.pyplot as plt
        # for i in range(num_time_steps):
        #     plt.clf()
        #     plt.figure(frameon=False)
        #     ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
        #     ax.set_axis_off()
        #     plt.gcf().add_axes(ax)
        #     plt.imshow(np.rot90(loss_arrays[i].T)) #have to transpose if transposing u and v and param_surface_1
        #     plt.savefig(wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png', bbox_inches='tight', pad_inches=0, dpi=300)
        #     plt.close()


        surfaces=m.Group()
        surf_functions=[] #Need this later to move dot around.
        grids=m.Group()
        print("Loading Surfaces and Gridlines...")
        for i in tqdm(range(num_time_steps)):
            surf_func=partial(param_surface_2, surf_array=loss_arrays[i])
            surf_functions.append(surf_func)
            surface = parametric_surface(
                surf_func,  
                u_range=[-2.5, 2.5],
                v_range=[-2.5, 2.5],
                resolution=(512, 512),
            )

            ts2 = textured_surface(surface, wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png')
            set_surface_shading(ts2, 0.0, 0.1, 0)
            surfaces.add(ts2)

            num_lines = 64  # Number of gridlines in each direction
            num_points = 512  # Number of points per line
            u_gridlines = m.VGroup()
            v_gridlines = m.VGroup()
            u_values = np.linspace(-2.5, 2.5, num_lines)
            v_points = np.linspace(-2.5, 2.5, num_points)
            for u in u_values:
                points = [surf_func(u, v) for v in v_points]
                line = m.VMobject()
                line.set_points_smoothly(points)
                line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
                u_gridlines.add(line)

            u_points = np.linspace(-2.5, 2.5, num_points)
            for v in u_values:  # Using same number of lines for both directions
                points = [surf_func(u, v) for u in u_points]
                line = m.VMobject()
                line.set_points_smoothly(points)
                line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
                v_gridlines.add(line)
            grids.add(m.VGroup(u_gridlines, v_gridlines))

        self.wait()

        self.add(surfaces[0])
        self.add(grids[0])
        self.add(s2)

        num_total_steps=num_time_steps*8 #Crank this for final viz
        start_orientation=[-178, 45, 0, (-0.0, -0.12, 0.19), 5.61]
        end_orientation=[-179, 16, 0, (0.05, -0.45, 0.48), 2.83]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=num_total_steps)

        surface_update_counter=1
        frames_per_surface_upddate=np.floor(num_total_steps/num_time_steps)
        print('frames_per_surface_upddate', frames_per_surface_upddate)
        self.wait()
        for i in range(1, num_total_steps):
            # print(i, len(interp_orientations))
            if i%frames_per_surface_upddate==0 and surface_update_counter<len(surfaces):

                self.remove(surfaces[surface_update_counter-1])
                self.remove(grids[surface_update_counter-1])
                self.add(surfaces[surface_update_counter])
                self.add(grids[surface_update_counter])
                self.remove(s2)
                # print('surface_update_counter', surface_update_counter)

                new_point_coords=surf_functions[surface_update_counter](*starting_coords)
                s2.move_to(new_point_coords) #This should make point move down smoothly. 
                surface_update_counter+=1

                self.add(s2) #Occlusions

            # print(i, len(interp_orientations))
            self.set_view(*interp_orientations[i])
            self.wait(0.1)

        self.wait()

        # self.wait()
        # self.play(self.animate_view(-103, 12, 0, (0.01, -0.46, 0.57), 1.95), run_time=10.0) #Pan Around
        # self.wait()
        # self.play(self.animate_view(-89, 0, 0, (0.05, -0.09, 0.59), 5.82), run_time=8.0) #Zoom out
        # self.wait()
        # self.play(self.animate_view(-85, 99, 0, (0.05, -0.09, 0.59), 5.82), run_time=8.0) #Pan below
        # self.wait()
        # self.play(self.animate_view(84, 102, 0, (0.05, -0.09, 0.59), 5.82), run_time=10.0) #Pan around below
        # self.wait()
        # self.play(self.animate_view(89, 0, 0, (-0.03, -0.14, 0.51), 6.38), run_time=8.0) #Back to overheaad
        # self.wait()

        self.wait(20)
        pass

class P48_moving_view_3(WelchScene):
    '''
    Pan/zoom
    '''
    def construct(self):
        starting_coords=[0.05,-0.9]
        starting_point=param_surface_1(*starting_coords)
        s2=Dot3D(center=starting_point, radius=0.06, color='$FF00FF')
    
        #Load up other surfaces to visualize
        loss_arrays=[]
        num_time_steps=66
        print('Loading Surface Arrays')
        for i in tqdm(range(num_time_steps)):
            loss_arrays.append(np.load(wormhole_dir+str(i).zfill(3)+'.npy'))

        # import matplotlib.pyplot as plt
        # for i in range(num_time_steps):
        #     plt.clf()
        #     plt.figure(frameon=False)
        #     ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
        #     ax.set_axis_off()
        #     plt.gcf().add_axes(ax)
        #     plt.imshow(np.rot90(loss_arrays[i].T)) #have to transpose if transposing u and v and param_surface_1
        #     plt.savefig(wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png', bbox_inches='tight', pad_inches=0, dpi=300)
        #     plt.close()


        surfaces=m.Group()
        surf_functions=[] #Need this later to move dot around.
        grids=m.Group()
        print("Loading Surfaces and Gridlines...")
        for i in tqdm(range(num_time_steps)):
            surf_func=partial(param_surface_2, surf_array=loss_arrays[i])
            surf_functions.append(surf_func)
            surface = parametric_surface(
                surf_func,  
                u_range=[-2.5, 2.5],
                v_range=[-2.5, 2.5],
                resolution=(512, 512),
            )

            ts2 = textured_surface(surface, wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png')
            set_surface_shading(ts2, 0.0, 0.1, 0)
            surfaces.add(ts2)

            num_lines = 64  # Number of gridlines in each direction
            num_points = 512  # Number of points per line
            u_gridlines = m.VGroup()
            v_gridlines = m.VGroup()
            u_values = np.linspace(-2.5, 2.5, num_lines)
            v_points = np.linspace(-2.5, 2.5, num_points)
            for u in u_values:
                points = [surf_func(u, v) for v in v_points]
                line = m.VMobject()
                line.set_points_smoothly(points)
                line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
                u_gridlines.add(line)

            u_points = np.linspace(-2.5, 2.5, num_points)
            for v in u_values:  # Using same number of lines for both directions
                points = [surf_func(u, v) for u in u_points]
                line = m.VMobject()
                line.set_points_smoothly(points)
                line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
                v_gridlines.add(line)
            grids.add(m.VGroup(u_gridlines, v_gridlines))

        self.wait()

        self.add(surfaces[0])
        self.add(grids[0])
        self.add(s2)

        num_total_steps=num_time_steps*8 #Crank this for final viz
        start_orientation=[137, 41, 0, (-0.05, -0.51, 0.75), 3.24]
        end_orientation=[360-139, 17, 0, (-0.08, -0.51, 0.75), 2.26]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=num_total_steps)

        surface_update_counter=1
        frames_per_surface_upddate=np.floor(num_total_steps/num_time_steps)
        print('frames_per_surface_upddate', frames_per_surface_upddate)
        self.wait()
        for i in range(1, num_total_steps):
            # print(i, len(interp_orientations))
            if i%frames_per_surface_upddate==0 and surface_update_counter<len(surfaces):

                self.remove(surfaces[surface_update_counter-1])
                self.remove(grids[surface_update_counter-1])
                self.add(surfaces[surface_update_counter])
                self.add(grids[surface_update_counter])
                # print('surface_update_counter', surface_update_counter)

                new_point_coords=surf_functions[surface_update_counter](*starting_coords)
                s2.move_to(new_point_coords) #This should make point move down smoothly. 
                surface_update_counter+=1

            # print(i, len(interp_orientations))
            self.set_view(*interp_orientations[i])
            self.wait(0.1)

        self.wait()

        # self.wait()
        # self.play(self.animate_view(-103, 12, 0, (0.01, -0.46, 0.57), 1.95), run_time=10.0) #Pan Around
        # self.wait()
        # self.play(self.animate_view(-89, 0, 0, (0.05, -0.09, 0.59), 5.82), run_time=8.0) #Zoom out
        # self.wait()
        # self.play(self.animate_view(-85, 99, 0, (0.05, -0.09, 0.59), 5.82), run_time=8.0) #Pan below
        # self.wait()
        # self.play(self.animate_view(84, 102, 0, (0.05, -0.09, 0.59), 5.82), run_time=10.0) #Pan around below
        # self.wait()
        # self.play(self.animate_view(89, 0, 0, (-0.03, -0.14, 0.51), 6.38), run_time=8.0) #Back to overheaad
        # self.wait()

        self.wait(20)
        pass


class P48_fixed_view_2(WelchScene):
    def construct(self):
        starting_coords=[0.05,-0.9]
        starting_point=param_surface_1(*starting_coords)
        s2=Dot3D(center=starting_point, radius=0.06, color='$FF00FF')
    
        #Load up other surfaces to visualize
        loss_arrays=[]
        num_time_steps=66
        print('Loading Surface Arrays')
        for i in tqdm(range(num_time_steps)):
            loss_arrays.append(np.load(wormhole_dir+str(i).zfill(3)+'.npy'))

        # import matplotlib.pyplot as plt
        # for i in range(num_time_steps):
        #     plt.clf()
        #     plt.figure(frameon=False)
        #     ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
        #     ax.set_axis_off()
        #     plt.gcf().add_axes(ax)
        #     plt.imshow(np.rot90(loss_arrays[i].T)) #have to transpose if transposing u and v and param_surface_1
        #     plt.savefig(wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png', bbox_inches='tight', pad_inches=0, dpi=300)
        #     plt.close()


        surfaces=m.Group()
        surf_functions=[] #Need this later to move dot around.
        grids=m.Group()
        print("Loading Surfaces and Gridlines...")
        for i in tqdm(range(num_time_steps)):
            surf_func=partial(param_surface_2, surf_array=loss_arrays[i])
            surf_functions.append(surf_func)
            surface = parametric_surface(
                surf_func,  
                u_range=[-2.5, 2.5],
                v_range=[-2.5, 2.5],
                resolution=(512, 512),
            )

            ts2 = textured_surface(surface, wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png')
            set_surface_shading(ts2, 0.0, 0.1, 0)
            surfaces.add(ts2)

            num_lines = 64  # Number of gridlines in each direction
            num_points = 512  # Number of points per line
            u_gridlines = m.VGroup()
            v_gridlines = m.VGroup()
            u_values = np.linspace(-2.5, 2.5, num_lines)
            v_points = np.linspace(-2.5, 2.5, num_points)
            for u in u_values:
                points = [surf_func(u, v) for v in v_points]
                line = m.VMobject()
                line.set_points_smoothly(points)
                line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
                u_gridlines.add(line)

            u_points = np.linspace(-2.5, 2.5, num_points)
            for v in u_values:  # Using same number of lines for both directions
                points = [surf_func(u, v) for u in u_points]
                line = m.VMobject()
                line.set_points_smoothly(points)
                line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
                v_gridlines.add(line)
            grids.add(m.VGroup(u_gridlines, v_gridlines))

        self.wait()

        self.add(surfaces[0])
        self.add(grids[0])
        self.add(s2)

        #Fixed orentation
        # self.set_view(132, 28, 0, (-0.12, -0.56, 0.33), 4.50) #Kinda wide, but nice I think, could do a closer one too
        # self.set_view(159, 35, 0, (-0.03, -0.62, 0.65), 1.95) #Match with long render
        # self.set_view(180, 23, 0, (-0.06, 0.09, 0.43), 5.81) #Fixed wide veiew I can use at the beginning of 48
        self.set_view(135, 47, 0, (0.15, 0.28, -0.04), 5.61) #V2 View for side-by-side in p49

        surface_update_counter=1
        # frames_per_surface_upddate=np.floor(num_total_steps/num_time_steps)
        self.wait()
        for i in range(1, len(surfaces)):
            # print(i, len(interp_orientations))

            self.remove(surfaces[surface_update_counter-1])
            self.remove(grids[surface_update_counter-1])
            self.add(surfaces[surface_update_counter])
            self.add(grids[surface_update_counter])
            # print('surface_update_counter', surface_update_counter)

            new_point_coords=surf_functions[surface_update_counter](*starting_coords)
            s2.move_to(new_point_coords) #This should make point move down smoothly. 
            surface_update_counter+=1
            self.wait(0.1)

        self.wait()

        # self.wait()
        # self.play(self.animate_view(-103, 12, 0, (0.01, -0.46, 0.57), 1.95), run_time=10.0) #Pan Around
        # self.wait()
        # self.play(self.animate_view(-89, 0, 0, (0.05, -0.09, 0.59), 5.82), run_time=8.0) #Zoom out
        # self.wait()
        # self.play(self.animate_view(-85, 99, 0, (0.05, -0.09, 0.59), 5.82), run_time=8.0) #Pan below
        # self.wait()
        # self.play(self.animate_view(84, 102, 0, (0.05, -0.09, 0.59), 5.82), run_time=10.0) #Pan around below
        # self.wait()
        # self.play(self.animate_view(89, 0, 0, (-0.03, -0.14, 0.51), 6.38), run_time=8.0) #Back to overheaad
        # self.wait()

        self.wait(20)
        pass

class P49_paris(WelchScene):
    def construct(self):
        starting_coords=[0.05,-0.9]
        starting_point=param_surface_1(*starting_coords)
        s2=Dot3D(center=starting_point, radius=0.06, color='$FF00FF')
    
        #Load up other surfaces to visualize
        loss_arrays=[]
        num_time_steps=66
        print('Loading Surface Arrays')
        for i in tqdm(range(num_time_steps)):
            loss_arrays.append(np.load(wormhole_dir+str(i).zfill(3)+'.npy'))

        # import matplotlib.pyplot as plt
        # for i in range(num_time_steps):
        #     plt.clf()
        #     plt.figure(frameon=False)
        #     ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
        #     ax.set_axis_off()
        #     plt.gcf().add_axes(ax)
        #     plt.imshow(np.rot90(loss_arrays[i].T)) #have to transpose if transposing u and v and param_surface_1
        #     plt.savefig(wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png', bbox_inches='tight', pad_inches=0, dpi=300)
        #     plt.close()


        surfaces=m.Group()
        surf_functions=[] #Need this later to move dot around.
        grids=m.Group()
        print("Loading Surfaces and Gridlines...")
        for i in tqdm(range(num_time_steps)):
            surf_func=partial(param_surface_2, surf_array=loss_arrays[i])
            surf_functions.append(surf_func)
            surface = parametric_surface(
                surf_func,  
                u_range=[-2.5, 2.5],
                v_range=[-2.5, 2.5],
                resolution=(512, 512),
            )

            ts2 = textured_surface(surface, wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png')
            set_surface_shading(ts2, 0.0, 0.1, 0)
            surfaces.add(ts2)

            num_lines = 64  # Number of gridlines in each direction
            num_points = 512  # Number of points per line
            u_gridlines = m.VGroup()
            v_gridlines = m.VGroup()
            u_values = np.linspace(-2.5, 2.5, num_lines)
            v_points = np.linspace(-2.5, 2.5, num_points)
            for u in u_values:
                points = [surf_func(u, v) for v in v_points]
                line = m.VMobject()
                line.set_points_smoothly(points)
                line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
                u_gridlines.add(line)

            u_points = np.linspace(-2.5, 2.5, num_points)
            for v in u_values:  # Using same number of lines for both directions
                points = [surf_func(u, v) for u in u_points]
                line = m.VMobject()
                line.set_points_smoothly(points)
                line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
                v_gridlines.add(line)
            grids.add(m.VGroup(u_gridlines, v_gridlines))

        self.wait()

        self.add(surfaces[0])
        self.add(grids[0])
        self.add(s2)

        #Fixed orentation
        # self.set_view(132, 28, 0, (-0.12, -0.56, 0.33), 4.50) #Kinda wide, but nice I think, could do a closer one too
        # self.set_view(159, 35, 0, (-0.03, -0.62, 0.65), 1.95) #Match with long render
        # self.set_view(180, 23, 0, (-0.06, 0.09, 0.43), 5.81) #Fixed wide veiew I can use at the beginning of 48
        self.set_view(135, 47, 0, (0.15, 0.28, -0.04), 5.61) #Side by side for p349

        surface_update_counter=1
        # frames_per_surface_upddate=np.floor(num_total_steps/num_time_steps)
        self.wait()
        for i in range(1, len(surfaces)):
            # print(i, len(interp_orientations))

            self.remove(surfaces[surface_update_counter-1])
            self.remove(grids[surface_update_counter-1])
            self.add(surfaces[surface_update_counter])
            self.add(grids[surface_update_counter])
            # print('surface_update_counter', surface_update_counter)

            new_point_coords=surf_functions[surface_update_counter](*starting_coords)
            s2.move_to(new_point_coords) #This should make point move down smoothly. 
            surface_update_counter+=1
            self.wait(0.1)

        self.wait()

        # self.wait()
        # self.play(self.animate_view(-103, 12, 0, (0.01, -0.46, 0.57), 1.95), run_time=10.0) #Pan Around
        # self.wait()
        # self.play(self.animate_view(-89, 0, 0, (0.05, -0.09, 0.59), 5.82), run_time=8.0) #Zoom out
        # self.wait()
        # self.play(self.animate_view(-85, 99, 0, (0.05, -0.09, 0.59), 5.82), run_time=8.0) #Pan below
        # self.wait()
        # self.play(self.animate_view(84, 102, 0, (0.05, -0.09, 0.59), 5.82), run_time=10.0) #Pan around below
        # self.wait()
        # self.play(self.animate_view(89, 0, 0, (-0.03, -0.14, 0.51), 6.38), run_time=8.0) #Back to overheaad
        # self.wait()

        self.wait(20)
        pass



