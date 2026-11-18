from welch_paths import welch_path
import numpy as np
import manimgx as m
from welch_scene import manual_camera_interpolation
from landscape_geometry import get_pivot_and_scale
from landscape_geometry import get_numerical_gradient
from welch_axes import WelchXAxis, WelchYAxis
from welch_scene import WelchScene
from welch_surface import parametric_surface, set_surface_shading, textured_surface
from welch_text import math_tex
from functools import partial

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

# surf_1=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_24_28_losses_4.npy')
# xy_1=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_24_28_losses_4xy.npy')

loss_curve_1=np.load(welch_path('backpropagation/hackin/apr_24_5/all_execpt_embedding_random_64.npy'))
loss_curve_2=np.load(welch_path('backpropagation/hackin/apr_24_5/all_execpt_embedding_random_51.npy'))
loss_curve_3=np.load(welch_path('backpropagation/hackin/apr_24_7/all_execpt_embedding_pretrained_19.npy'))
loss_curve_4=np.load(welch_path('backpropagation/hackin/apr_24_7/all_execpt_embedding_pretrained_27.npy'))

alphas_1=np.linspace(-2.5, 2.5, 512)
# loss_2d_1=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/apr_24_3/pretrained_11_111_first_8.npy')
# loss_2d_1=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/apr_24_11/pre_training_landscape.npy')
# loss_2d_1=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/apr_24_12/000.npy')
loss_2d_1=np.load(welch_path('backpropagation/hackin/apr_25_1/000.npy'))

class Dot3D(m.Sphere):
    def __init__(self, center=m.ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(center)

# import matplotlib.pyplot as plt
# plt.figure(frameon=False)
# ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
# ax.set_axis_off()
# plt.gcf().add_axes(ax)
# plt.imshow(np.rot90(loss_2d_1.T)) #have to transpose if transposing u and v and param_surface_1
# plt.savefig('loss_2d_1.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()

# plt.clf()
# plt.figure(frameon=False)
# ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
# ax.set_axis_off()
# plt.gcf().add_axes(ax)
# plt.imshow(np.rot90(loss_2d_1.T)[128:-128, 128:-128])
# plt.savefig('loss_2d_1_inner.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()

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



import numpy as np

def find_local_minima(surface):
    """
    Finds all local minima in a 2D numpy array.
    
    A local minimum is defined as a point where all neighboring points
    have a greater or equal value.
    
    Args:
        surface (numpy.ndarray): 2D array representing surface heights
        
    Returns:
        list: List of tuples (row, col) representing the coordinates of local minima
    """
    # Get the shape of the array
    rows, cols = surface.shape
    
    # List to store coordinates of local minima
    minima = []
    
    # Check each point in the array
    for i in range(rows):
        for j in range(cols):
            # Get the current value
            current_val = surface[i, j]
            
            # Define the neighborhood (8-connected)
            # Generate all possible neighbor coordinates
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    # Skip the center point
                    if di == 0 and dj == 0:
                        continue
                    
                    # Calculate neighbor coordinates
                    ni, nj = i + di, j + dj
                    
                    # Check if neighbor is within bounds
                    if 0 <= ni < rows and 0 <= nj < cols:
                        neighbors.append((ni, nj))
            
            # Check if current point is a local minimum
            is_minimum = True
            for ni, nj in neighbors:
                if surface[ni, nj] < current_val:
                    is_minimum = False
                    break
            
            # If it's a local minimum, add to list
            if is_minimum:
                minima.append((i, j))
    
    return minima


class P39_48(WelchScene):
    def construct(self):
        '''
        Not going to lie I'm kinda terrified of this scene - buuuuut if I can pull of 
        some level of what's in my head I do think it will be DOPE. 
        Ok one step at a time here. Starting with one and then 6 2d panels, then being the last two 
        together into 3d very much like I did in the last big scene. 

        Ok making progress here - I'm pretty certain at this point that all these need to be 
        in "3d upright view" for the transition to 3d to work -> let me go ahead and make that change early-ish. 
        '''

        x_axis_1=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=8)
        y_axis_1=WelchYAxis(y_min=8, y_max=16, y_ticks=[8, 9, 10, 11, 12, 13, 14, 15 ], y_tick_width=0.15,        
                          y_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=6)

        x_label_1 = math_tex(r'\alpha', font_size=36).set_color(CHILL_BROWN)
        y_label_1 = math_tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label_1.next_to(x_axis_1, m.RIGHT, buff=0.05)
        y_label_1.next_to(y_axis_1, m.UP, buff=0.08)

        mapped_x_1=x_axis_1.map_to_canvas(loss_curve_1[0,:]) 
        mapped_y_1=y_axis_1.map_to_canvas(loss_curve_1[1,:])

        # np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T.shape

        curve_1=m.VMobject()
        curve_1.set_points_smoothly(np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T)
        curve_1.set_stroke(width=4, color=YELLOW, opacity=1.0)

        axes_1=m.VGroup(x_axis_1, y_axis_1, x_label_1, y_label_1, curve_1)
        axes_1.move_to([0, 0, 0])
        axes_1.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        self.set_view(0, 90, 0, (-0.21, -0.01, -0.02), 8.00)
        self.add(x_axis_1, y_axis_1, x_label_1, y_label_1)
        self.wait(0)
        self.play(m.Create(curve_1), run_time=5)

        self.wait() #Ok let's try the "expand in the negative direction deal"


        x_axis_2=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=8)
        y_axis_2=WelchYAxis(y_min=8, y_max=16, y_ticks=[8, 9, 10, 11, 12, 13, 14, 15 ], y_tick_width=0.15,        
                          y_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=6)

        x_label_2 = math_tex(r'\alpha', font_size=36).set_color(CHILL_BROWN)
        y_label_2 = math_tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label_2.next_to(x_axis_2, m.RIGHT, buff=0.05)
        y_label_2.next_to(y_axis_2, m.UP, buff=0.08)

        mapped_x_2=x_axis_2.map_to_canvas(loss_curve_2[0,:]) 
        mapped_y_2=y_axis_2.map_to_canvas(loss_curve_2[1,:])

        curve_2=m.VMobject()
        curve_2.set_points_smoothly(np.vstack((mapped_x_2, mapped_y_2, np.zeros_like(mapped_x_2))).T)
        curve_2.set_stroke(width=4, color=YELLOW, opacity=1.0)

        axes_2=m.VGroup(x_axis_2, y_axis_2, x_label_2, y_label_2, curve_2)
        axes_2.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        axes_2.move_to([10, 0, 0])

        self.play(self.animate_view(0, 90, 0, (5.03, 0.05, 0.0), 12.15),
                  m.FadeIn(m.VGroup(x_axis_2, y_axis_2, x_label_2, y_label_2)),
                  m.Create(curve_2),
                  run_time=3.0)
        self.wait()


        x_axis_3=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=8)
        y_axis_3=WelchYAxis(y_min=0, y_max=40, y_ticks=[0, 5, 10, 15, 20, 25, 30, 35], y_tick_width=0.15,        
                          y_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=6)

        x_label_3 = math_tex(r'\alpha', font_size=36).set_color(CHILL_BROWN)
        y_label_3 = math_tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label_3.next_to(x_axis_3, m.RIGHT, buff=0.05)
        y_label_3.next_to(y_axis_3, m.UP, buff=0.08)

        mapped_x_3=x_axis_3.map_to_canvas(loss_curve_3[0,:]) 
        mapped_y_3=y_axis_3.map_to_canvas(loss_curve_3[1,:])

        curve_3=m.VMobject()
        curve_3.set_points_smoothly(np.vstack((mapped_x_3, mapped_y_3, np.zeros_like(mapped_x_3))).T)
        curve_3.set_stroke(width=4, color=BLUE, opacity=1.0)

        axes_3=m.VGroup(x_axis_3, y_axis_3, x_label_3, y_label_3, curve_3)
        axes_3.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        axes_3.move_to([10, 0, 0])


        x_axis_4=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=8)
        y_axis_4=WelchYAxis(y_min=0, y_max=40, y_ticks=[0, 5, 10, 15, 20, 25, 30, 35], y_tick_width=0.15,        
                          y_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=6)

        x_label_4 = math_tex(r'\alpha', font_size=36).set_color(CHILL_BROWN)
        y_label_4 = math_tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label_4.next_to(x_axis_4, m.RIGHT, buff=0.05)
        y_label_4.next_to(y_axis_4, m.UP, buff=0.08)

        mapped_x_4=x_axis_4.map_to_canvas(loss_curve_4[0,:]) 
        mapped_y_4=y_axis_4.map_to_canvas(loss_curve_4[1,:])

        curve_4=m.VMobject()
        curve_4.set_points_smoothly(np.vstack((mapped_x_4, mapped_y_4, np.zeros_like(mapped_x_3))).T)
        curve_4.set_stroke(width=4, color=BLUE, opacity=1.0)

        axes_4=m.VGroup(x_axis_4, y_axis_4, x_label_4, y_label_4, curve_4)
        axes_4.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        axes_4.move_to([10, 0, -7])

        self.wait()

        self.play(axes_2.animate.move_to([0, 0, -7]),
                 self.animate_view(0, 90, 0, (4.6, -3.5, -3.6), 12.54),
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
                            x_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=8)
        y_axis_5=WelchYAxis(y_min=0, y_max=25, y_ticks=[0, 5, 10, 15, 20], y_tick_width=0.15,        
                          y_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=6)

        x_label_5 = math_tex(r'\alpha', font_size=36).set_color(CHILL_BROWN)
        y_label_5 = math_tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label_5.next_to(x_axis_5, m.RIGHT, buff=0.05)
        y_label_5.next_to(y_axis_5, m.UP, buff=0.08)

        mapped_x_5=x_axis_5.map_to_canvas(alphas_1) 
        mapped_y_5=y_axis_5.map_to_canvas(slice_1)

        curve_5=m.VMobject()
        curve_5.set_points_smoothly(np.vstack((mapped_x_5, mapped_y_5, np.zeros_like(mapped_x_5))).T)
        curve_5.set_stroke(width=4, color="#FF00FF", opacity=1.0)

        axes_5=m.VGroup(x_axis_5, y_axis_5, x_label_5, y_label_5, curve_5)
        axes_5.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        axes_5.move_to([20, 0, 0])


        x_axis_6=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=8)
        y_axis_6=WelchYAxis(y_min=0, y_max=25, y_ticks=[0, 5, 10, 15, 20], y_tick_width=0.15,        
                          y_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=6)

        x_label_6 = math_tex(r'\alpha', font_size=36).set_color(CHILL_BROWN)
        y_label_6 = math_tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label_6.next_to(x_axis_6, m.RIGHT, buff=0.05)
        y_label_6.next_to(y_axis_6, m.UP, buff=0.08)

        mapped_x_6=x_axis_6.map_to_canvas(alphas_1) 
        mapped_y_6=y_axis_6.map_to_canvas(slice_2)

        curve_6=m.VMobject()
        curve_6.set_points_smoothly(np.vstack((mapped_x_6, mapped_y_6, np.zeros_like(mapped_x_6))).T)
        curve_6.set_stroke(width=4, color="#FF00FF", opacity=1.0)

        axes_6=m.VGroup(x_axis_6, y_axis_6, x_label_6, y_label_6, curve_6)
        axes_6.rotate(90*m.DEGREES, [1,0,0], about_point=m.ORIGIN)
        axes_6.move_to([20, 0, -7])

        self.wait()
        # self.add(axes_5, axes_6)
        self.play(m.FadeIn(m.VGroup(x_axis_5, y_axis_5, x_label_5, y_label_5)),
          m.FadeIn(m.VGroup(x_axis_6, y_axis_6, x_label_6, y_label_6)),
          m.Create(curve_5),
          m.Create(curve_6),
          self.animate_view(0, 89, 0, (9.97, -3.5, -3.39), 14.41),
          run_time=6)
        
        self.wait()

        # Alright time for the big move. 
        # I'm really going to need to move these axes to the around the origin I think to support nice camera motion?
        # That might get clunky, we'll see here. Maybe I don't? Will have to experiment. 
        # Ok seems ok to have it at 20? I'll go back and move everything over if I need to
        # Create main surface
        surface = parametric_surface(
            param_surface_1,  
            u_range=[-2.5, 2.5],   #[-2.5, 2.5]
            v_range=[-2.5, 2.5],  #[-2.5, 2.5],
            resolution=(128, 128), #(512, 512), #TODO -> CRANK THIS BACK UP TO LIKE 512

        )
        ts = textured_surface(surface, welch_path('backpropagation/hackin/loss_2d_1.png'))
        set_surface_shading(ts, 0.0, 0.1, 0)
        # ts.rotate(90*DEGREES, axis=[0,0,1])

        pivot_x,scale_x=get_pivot_and_scale(axis_min=x_axis_5.x_min, axis_max=x_axis_5.x_max, 
                                        axis_end=x_axis_5.axis_length_on_canvas)
        pivot_y,scale_y=get_pivot_and_scale(axis_min=y_axis_5.y_min, axis_max=y_axis_5.y_max, 
                                        axis_end=y_axis_5.axis_length_on_canvas)
        # ts.scale([scale_x, scale_x, scale_y*0.6], about_point=[pivot_x, pivot_x, pivot_y]) #EXTRA VERTICAL SCALING FACTOR HERE
        # ts.scale([[1.7, 1.7, 0.24]], about_point=[pivot_x, pivot_x, pivot_y])
        # surf_shift=[19, 0, 0] #Gross iterative swagginess, I think i at least have the scale right
        # ts.shift([19, 0, 0])
        # ts.move_to([20, 0, 0])
        # ts.shift([0,0,-0.45])
        # ts.shift([0.1, 0, 0])
        # ts.shift([0, 0.075, 0])
        # ts.shift([0.1, 0.075, -0.45])

        num_lines = 32  # Number of gridlines in each direction
        num_points = 512  # Number of points per line
        u_gridlines = m.VGroup()
        v_gridlines = m.VGroup()
        u_values = np.linspace(-2.5, 2.5, num_lines)
        v_points = np.linspace(-2.5, 2.5, num_points)
        for u in u_values:
            points = [param_surface_1(u, v) for v in v_points]
            line = m.VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.3)
            u_gridlines.add(line)

        u_points = np.linspace(-2.5, 2.5, num_points)
        for v in u_values:  # Using same number of lines for both directions
            points = [param_surface_1(u, v) for u in u_points]
            line = m.VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.3)
            v_gridlines.add(line)

        # u_gridlines.scale([scale_x, scale_x, scale_y*0.6], about_point=[pivot_x, pivot_x, pivot_y])
        # u_gridlines.move_to([20, 0, 0])
        # v_gridlines.scale([scale_x, scale_x, scale_y*0.6], about_point=[pivot_x, pivot_x, pivot_y])
        # v_gridlines.move_to([20, 0, 0])    

        surf_and_mesh=m.Group(ts, u_gridlines, v_gridlines)
        surf_and_mesh.scale([scale_x, scale_x, scale_y*0.6], about_point=[pivot_x, pivot_x, pivot_y]) #EXTRA VERTICAL SCALING FACTOR HERE
        surf_and_mesh.move_to([20, 0, 0])
        surf_and_mesh.shift([0.1, 0.075, -0.45])


        # self.add(u_gridlines)
        # self.add(v_gridlines)


        ## Ok what if I just squish the curves vertically when the camera moves
        ## And take off the tick marks, take off the labels, and mabye even lose the vertical axes?

        self.wait()
        axes_1.set_opacity(0)
        axes_2.set_opacity(0)
        axes_3.set_opacity(0)
        axes_4.set_opacity(0)
        y_axis_6.set_opacity(0)
        y_label_6.set_opacity(0)
        y_axis_5.set_opacity(0) #might be nice to keep these, but difficult with scaling
        y_label_5.set_opacity(0)
        axes_6[0][-1][2].set_opacity(0) #Zero tick label on axis 6

        curve_5.scale([1, 1, 0.6])
        curve_6.scale([1, 1, 0.6])
        axes_6.move_to([0, 0, 0])
        axes_6.rotate(90*m.DEGREES, axis=[0,0,1]) #This seems to have kidna worked out.


        self.wait(0)

        self.set_view(27, 37, 0, (18.74, 3.76, -6.28), 17.77)

        self.add(surf_and_mesh)


        # self.add(u_gridlines, v_gridlines)
        # self.add(ts)
        # self.add(curve_5, curve_6)


        # axes_5.scale([1.0, 1.0, 0.5])
        # axes_6.scale([1.0, 1.0, 0.5])
        # ts.scale([1.0, 1.0, 0.5])

        # axes_5.move_to([20, 0, 0]) 


        
       


        # self.add(ts)
        # self.add(curve_5) #, curve_6)
        # self.add(curve_6)
        # self.wait()

        # # ts.shift([2,2,0])
        # # ts.shift([0.125, 0.125, 0])
        # ts.shift([0,0,-0.45])
        # ts.shift([0.1, 0, 0])
        # ts.shift([0, 0.075, 0])

        self.wait()


        self.set_view(27, 37, 0, (18.74, 3.76, -6.28), 17.77)

        # self.set_view(29, 41, 0, (18.76, 3.99, -3.97), 8.30)

        # ts.shift([0.1,0,0])

        

        # ts.shift([0.1, 0.1, 0])

        # ts.scale([1.05, 1.05, 1.0], about_point=[-2.5, -2.5, 0])


        # OH shit maybe I draw the grid first then fill in the colors?!?! That would fit with the 
        # script pretty well and look pretty dope I think .





        pass
        self.wait(20)



class sketch_3d(WelchScene):
    def construct(self):
        '''
        Hack on 3d surface a bit before I get there, make sure it can do what I want. 
        '''

        # Create main surface
        surface = parametric_surface(
            param_surface_1,  
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=(512, 512),
        )

    
        ts = textured_surface(surface, welch_path('backpropagation/hackin/loss_2d_1.png'))
        set_surface_shading(ts, 0.0, 0.1, 0)

        ts.scale([1,1,0.1])
        ts.move_to([0,0,0])

        # self.add(ts)
        self.add(ts)
        
        #Dare we try gridlines lol?

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
            line.set_stroke(width=1, color=m.WHITE, opacity=0.3)
            u_gridlines.add(line)

        u_points = np.linspace(-2.5, 2.5, num_points)
        for v in u_values:  # Using same number of lines for both directions
            points = [param_surface_1(u, v) for u in u_points]
            line = m.VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.3)
            v_gridlines.add(line)

        u_gridlines.scale([1,1,0.1]) #[scale_x, scale_x, scale_y], about_point=[pivot_x, pivot_x, pivot_y])
        u_gridlines.move_to([0,0,0])
        v_gridlines.scale([1,1,0.1]) #[scale_x, scale_x, scale_y], about_point=[pivot_x, pivot_x, pivot_y])
        v_gridlines.move_to([0,0,0])     

        self.wait()

        self.set_view(32, 38, 0, (0.12, -0.18, -0.23), 7.45)
        self.play(m.Create(u_gridlines),
                  m.Create(v_gridlines),
                  self.animate_view(161, 33, 0, (0.11, -0.12, -0.23), 4.84),
                  run_time=10)
        self.wait()
        self.add(ts)

        # self.add(u_gridlines, v_gridlines)  

        # self.remove(ts)

        pass
        self.wait(20)



        # self.set_view(70, 29, 0, (-4.94, 1.73, 1.26), 15.55)
        
        # pivot_x,scale_x=get_pivot_and_scale(axis_min=x_axis_1.x_min, axis_max=x_axis_1.x_max, 
        #                                 axis_end=x_axis_1.axis_length_on_canvas)
        # pivot_y,scale_y=get_pivot_and_scale(axis_min=y_axis_1.y_min, axis_max=y_axis_1.y_max, 
        #                                 axis_end=y_axis_1.axis_length_on_canvas)
        # ts.scale([scale_x, scale_x, scale_y], about_point=[pivot_x, pivot_x, pivot_y])



class sketch_getting_stuck(WelchScene):
    def construct(self):
        '''
        Hack on 3d surface a bit before I get there, make sure it can do what I want. 
        '''

        # Create main surface
        surface = parametric_surface(
            param_surface_1,  
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=(512, 512),
        )

        surface_inner = parametric_surface( #Ok I do think that shifting to an inner surface like this can be helpful as we zoom in.
            param_surface_1,  
            u_range=[-1.25, 1.25],
            v_range=[-1.25, 1.25],
            resolution=(128, 128),
        )
    
        # ts = TexturedSurface(surface, '/Users/stephen/Stephsencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/loss_2d_1.png')
        ts = textured_surface(surface, '/Users/stephen/manim/videos/loss_2d_1.png')
        set_surface_shading(ts, 0.0, 0.1, 0)

        tsi = textured_surface(surface_inner, '/Users/stephen/manim/videos/loss_2d_1_inner.png')
        set_surface_shading(tsi, 0.0, 0.1, 0)

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
 
        self.wait()

        self.set_view(32, 38, 0, (0.12, -0.18, -0.23), 7.45)
        self.play(m.Create(u_gridlines),
                  m.Create(v_gridlines),
                  self.animate_view(161, 33, 0, (0.11, -0.12, -0.23), 4.84),
                  run_time=10)
        self.wait()


        # self.add(tsi)
        # self.remove(tsi)
        self.add(ts)

        starting_coords=[0.05,-0.9] #[0.1,-0.8] is pretty good, [0.05,-0.9] is a bit better
        starting_point=param_surface_1(*starting_coords)
     
        s1=Dot3D(center=starting_point, radius=0.06, color='$FF00FF')
        self.add(s1)

        # Ok I think big question from here is can I some not-totally terrible numerical grad descent
        # and then after then take it warping into the global minimum.  
        # Might not be a terrible job for claude...
        num_steps=200 # I think it gest stuck around 30-40 at lr -0.01
        learning_rate=2e-3
        trajectory=[[starting_point[0], starting_point[1], param_surface_1(starting_point[0], starting_point[1])[2]]]
        for i in range(num_steps):
            g=get_numerical_gradient(param_surface_1, trajectory[-1][0], trajectory[-1][1], epsilon=0.01)
            delta=learning_rate*np.array(g)
            new_x=trajectory[-1][0]-delta[0]
            new_y=trajectory[-1][1]-delta[1]
            trajectory.append([new_x, new_y, param_surface_1(new_x, new_y)[2]])
        # trajectory=np.array(trajectory)

        #Ok let me go ahead and hack for a minute here on the fake/magic tunneling scence
        ending_coords=[0,0]
        ending_point=param_surface_1(*ending_coords)

        s2=Dot3D(center=ending_point, radius=0.06, color='$FF00FF')
        self.add(s2)

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
        num_steps3=512 #Plotting 1k points is kinda slow - slower than I thought 
        trajectory_waypoint=trajectory[-1]
        g=np.array([ending_coords[0]-trajectory[-1][0], ending_coords[1]-trajectory[-1][1]])
        for i in range(num_steps3):
            new_x=trajectory_waypoint[0]+(i/num_steps3)*g[0]
            new_y=trajectory_waypoint[1]+(i/num_steps3)*g[1]
            trajectory.append([new_x, new_y, param_surface_1(new_x, new_y)[2]])

        #GRR I still have some gaps in my dots -> might have to live with this, we'll see -> need to keep moving. 

        # learning_rate_3=1e-3
        # for i in range(num_steps3):
        #     g=-np.array([ending_coords[0]-trajectory[-1][0], ending_coords[1]-trajectory[-1][1]])
        #     delta=learning_rate_3*np.array(g)
        #     new_x=trajectory[-1][0]-delta[0]
        #     new_y=trajectory[-1][1]-delta[1]
        #     trajectory.append([new_x, new_y, param_surface_1(new_x, new_y)[2]])

        #Tracjectory is looking kinda jagged - could try smoothing the path or maybe just little spheres? And smaller learning rate if needed?
        trajectory=np.array(trajectory)
        # t = VMobject()
        # # t.set_points_smoothly(trajectory)
        # # t.set_points_as_corners(trajectory)
        # t.set_points(trajectory)
        # t.set_stroke(width=6, color="#FF00FF", opacity=1.0)
        # self.add(t)

        dot_path=m.Group()
        for t in trajectory:
            dot_path.add(Dot3D(center=t, radius=0.017, color='$FF00FF'))

        self.add(dot_path)


        # ts.set_opacity(0.2)
        # self.add(tsi)

        # # self.add(ts)
        self.add(u_gridlines, v_gridlines)  



        # self.wait()

        # self.set_view(-145, 29, 0, (0.39, 0.18, -0.31), 3.44)

        # u_gridlines.set_opacity(0.0)
        # v_gridlines.set_opacity(0.0)
        # # ts.set_opacity(0.7)

        pass
        self.wait(20)

class sketch_local_minima(WelchScene):
    '''
    Hmm ok I'm actually leanding towards just arrows in illustrator for this -> there's just way too many actual minima!!!
    '''
    def construct(self):

        surface = parametric_surface(
            param_surface_1,  
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=(512, 512),
        )

        ts = textured_surface(surface, '/Users/stephen/manim/videos/loss_2d_1.png')
        set_surface_shading(ts, 0.0, 0.1, 0)

        self.add(ts)

        minima=find_local_minima(loss_2d_1)
        minima=np.array(minima)
        minima=(minima/512.0)*5.0-2.5

        minima_dots=m.Group()
        minima_to_show=8
        for i in np.random.choice(range(len(minima)), minima_to_show):
            minima_dots.add(Dot3D(center=[minima[i][1], minima[i][0], param_surface_1(minima[i][1], minima[i][0])[2]], radius=0.04, color='$FF00FF'))
        self.add(minima_dots)

        

class sketch_wormole(WelchScene):
    def construct(self):
        '''
        Ok I need to sanity check the wormhole animatations, and then I think we're FINALLY ready for systhesis. 
        '''
        loss_arrays=[]
        num_time_steps=3
        for i in range(num_time_steps):
            loss_arrays.append(np.load(welch_path('backpropagation/hackin/apr_25_1/')+str(i).zfill(3)+'.npy'))

        # import matplotlib.pyplot as plt
        # for i in range(num_time_steps):
        #     plt.clf()
        #     plt.figure(frameon=False)
        #     ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
        #     ax.set_axis_off()
        #     plt.gcf().add_axes(ax)
        #     plt.imshow(np.rot90(loss_arrays[i].T)) #have to transpose if transposing u and v and param_surface_1
        #     plt.savefig('loss_2d_1_'+str(i).zfill(3)+'.png', bbox_inches='tight', pad_inches=0, dpi=300)
        #     plt.close()

        surfaces=m.Group()
        for i in range(num_time_steps):
            surf_func=partial(param_surface_2, surf_array=loss_arrays[i])

            surface = parametric_surface(
                surf_func,  
                u_range=[-2.5, 2.5],
                v_range=[-2.5, 2.5],
                resolution=(512, 512),
            )

            ts = textured_surface(surface, '/Users/stephen/manim/videos/'+'loss_2d_1_'+str(i).zfill(3)+'.png')
            set_surface_shading(ts, 0.0, 0.1, 0)
            surfaces.add(ts)

        #Would be nice to moch up the little camera move I want to do here....
        #Let me start with just swapping in the surfaces though
        self.add(surfaces[0])
        self.set_view(155, 35, 0, (-0.06, -0.74, 0.16), 2.66)

        # for i in range(1, num_time_steps):
        #     self.remove(surfaces[i-1])
        #     self.add(surfaces[i])
        #     self.wait(0.1)
            
        num_total_steps=16
        start_orientation=[142, 34, 0, (-0.09, -0.77, 0.15), 3.55]
        end_orientation=[131, 31, 0, (-0.12, -0.88, 0.22), 2.90]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=num_total_steps)

        self.set_view(*start_orientation)

        surface_update_counter=1
        frames_per_surface_upddate=np.floor(num_total_steps/num_time_steps)
        for i in range(1, num_total_steps):
            # print(i, len(interp_orientations))
            if i%frames_per_surface_upddate==0 and surface_update_counter<len(surfaces):
                self.remove(surfaces[surface_update_counter-1])
                self.add(surfaces[surface_update_counter])
                surface_update_counter+=1
            # print(i, len(interp_orientations))
            self.set_view(*interp_orientations[i])
            self.wait(0.1)

        self.wait()
        pass







