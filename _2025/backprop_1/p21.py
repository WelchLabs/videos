import manimgx as m
import numpy as np
from welch_axes import WelchXAxis, WelchYAxis, generate_nice_ticks
from welch_scene import WelchScene
from functools import partial
import sys

sys.path.append('/Users/stephen/manim/videos/_2025/backprop_1')
from backprop_data import xs1, losses1, all_probs_1

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

def get_x_axis(t, intial_bounds, final_bounds, position=None):
    lower_bound, upper_bound = time_to_bounds(t, intial_bounds, final_bounds)
    x_ticks, x_axis_min, x_axis_max=generate_nice_ticks(lower_bound, 0.95*upper_bound, min_ticks=3, max_ticks=16, ignore=[])
    x_axis=WelchXAxis(
        x_min=lower_bound,
        x_max=upper_bound,      
        x_ticks=x_ticks,  
        x_tick_height=0.15,        
        x_label_font_size=24,           
        stroke_width=3, 
        arrow_tip_scale=0.1,
        axis_length_on_canvas=7
    )
    if position is not None:
        x_axis.move_to(position)
    return x_axis

# def get_y_axis(t, intial_bounds, final_bounds, position=None):
#     lower_bound, upper_bound = time_to_bounds(t, intial_bounds, final_bounds) 

#     y_ticks, x_axis_min, x_axis_max=generate_nice_ticks(lower_bound, upper_bound, min_ticks=3, max_ticks=16, ignore=[])
#     y_axis=WelchYAxis(
#         y_min=lower_bound,
#         y_max=upper_bound,      
#         y_ticks=y_ticks,  
#         y_tick_width=0.15,        
#         y_label_font_size=24,           
#         stroke_width=3, 
#         arrow_tip_scale=0.1,
#         axis_length_on_canvas=5
#     )
#     if position is not None:
#         y_axis.move_to(position)
#     return y_axis


def get_y_axis(t, intial_bounds, final_bounds, position=None):
    lower_bound_x, upper_bound_x = time_to_bounds(t, intial_bounds, final_bounds)
    #Kinda hacky, these are x_bounds - this should really be done somewhere else. 
    indices_in_range=np.logical_and(xs1>lower_bound_x, xs1<upper_bound_x)
    y_to_viz=all_probs_1[indices_in_range]
    # Add padding to match scatter plot calculation
    upper_bound = 1.01 * np.max(y_to_viz)
    lower_bound = 0.99 * np.min(y_to_viz)

    y_ticks, x_axis_min, x_axis_max=generate_nice_ticks(lower_bound, upper_bound, min_ticks=3, max_ticks=16, ignore=[])
    y_axis=WelchYAxis(
        y_min=lower_bound,
        y_max=upper_bound,      
        y_ticks=y_ticks,  
        y_tick_width=0.15,        
        y_label_font_size=24,           
        stroke_width=3, 
        arrow_tip_scale=0.1,
        axis_length_on_canvas=5
    )
    if position is not None:
        y_axis.move_to(position)
    return y_axis

def get_fixed_y_axis(t, intial_bounds, final_bounds, position=None, y_zoom_t=0.0):
    """
    Creates a y-axis with bounds that interpolate between the current data-based bounds
    and fixed bounds (0.15 to 1.6) based on the y_zoom_t parameter.
    """
    # We still need to calculate x bounds for consistency
    lower_bound_x, upper_bound_x = time_to_bounds(t, intial_bounds, final_bounds)
    
    # Calculate current y bounds based on data
    indices_in_range = np.logical_and(xs1 > lower_bound_x, xs1 < upper_bound_x)
    y_to_viz = all_probs_1[indices_in_range]
    current_upper_bound = 1.01 * np.max(y_to_viz)
    current_lower_bound = 0.99 * np.min(y_to_viz)
    
    # Fixed y bounds for the second zoom out
    fixed_lower_bound = 0.15
    fixed_upper_bound = 1.6
    
    # Interpolate between current bounds and fixed bounds based on y_zoom_t
    lower_bound = y_zoom_t * fixed_lower_bound + (1 - y_zoom_t) * current_lower_bound
    upper_bound = y_zoom_t * fixed_upper_bound + (1 - y_zoom_t) * current_upper_bound

    y_ticks, x_axis_min, x_axis_max = generate_nice_ticks(lower_bound, upper_bound, min_ticks=3, max_ticks=16, ignore=[])
    y_axis = WelchYAxis(
        y_min=lower_bound,
        y_max=upper_bound,      
        y_ticks=y_ticks,  
        y_tick_width=0.15,        
        y_label_font_size=24,           
        stroke_width=3, 
        arrow_tip_scale=0.1,
        axis_length_on_canvas=5
    )
    if position is not None:
        y_axis.move_to(position)
    return y_axis

def get_scatter_points(t, initial_bounds, final_bounds, x_axis_position, y_axis_position):
    """
    Generate scatter points using the current time parameter directly rather than
    depending on x_axis and y_axis objects that might have stale values.
    """
    # Get the current x range based on time
    lower_bound_x, upper_bound_x = time_to_bounds(t, initial_bounds, final_bounds)
    
    # Filter data points within the current x range
    indices_in_range = np.logical_and(xs1 > lower_bound_x, xs1 < upper_bound_x)
    x_values = xs1[indices_in_range]
    y_values = all_probs_1[indices_in_range]
    # print(y_values)
    
    # Get y-axis bounds based on filtered data (same logic as in get_y_axis)
    if len(y_values) > 0:
        y_min = 0.99 * np.min(y_values)
        y_max = 1.01 * np.max(y_values)
    else:
        y_min = 0
        y_max = 1
    
    # Calculate axis scaling factors directly
    x_axis_length = 7  # Same as in get_x_axis
    y_axis_length = 5  # Same as in get_y_axis
    
    x_scale = (upper_bound_x - lower_bound_x) / x_axis_length
    y_scale = (y_max - y_min) / y_axis_length
    
    # Extract origin positions from the provided positions
    # Adjust for the fact that the position is the center of the axis
    origin_x = x_axis_position[0] - x_axis_length / 2
    origin_y = y_axis_position[1] - y_axis_length / 2
    
    # Create scatter plot points
    dots = m.VGroup()
    
    for x_val, y_val in zip(x_values, y_values):
        # Calculate normalized position within the data range (0 to 1)
        x_norm = (x_val - lower_bound_x) / (upper_bound_x - lower_bound_x)
        y_norm = (y_val - y_min) / (y_max - y_min)
        
        # Convert to canvas position
        x_pos = origin_x + x_norm * x_axis_length
        y_pos = origin_y + y_norm * y_axis_length
        
        # Create dot
        dot = m.Dot(
            point=[x_pos, y_pos, 0],
            radius=0.05,
            stroke_width=0,
            fill_opacity=0.8
        )
        dot.set_color(YELLOW)
        dots.add(dot)
    
    return dots

def get_scatter_points_with_interpolated_y(t, initial_bounds, final_bounds, x_axis_position, y_axis_position, y_zoom_t=0.0):
    """
    Generate scatter points using interpolated y-axis bounds based on y_zoom_t.
    This allows the scatter plot to scale and move with the y-axis during the second animation.
    """
    # Get the current x range based on time
    lower_bound_x, upper_bound_x = time_to_bounds(t, initial_bounds, final_bounds)
    
    # Filter data points within the current x range
    indices_in_range = np.logical_and(xs1 > lower_bound_x, xs1 < upper_bound_x)
    x_values = xs1[indices_in_range]
    y_values = all_probs_1[indices_in_range]
    
    # Calculate current y bounds based on data
    if len(y_values) > 0:
        current_y_min = 0.99 * np.min(y_values)
        current_y_max = 1.01 * np.max(y_values)
    else:
        current_y_min = 0
        current_y_max = 1
    
    # Fixed y bounds for the second zoom out
    fixed_y_min = 0.15
    fixed_y_max = 1.6
    
    # Interpolate between current bounds and fixed bounds based on y_zoom_t
    y_min = y_zoom_t * fixed_y_min + (1 - y_zoom_t) * current_y_min
    y_max = y_zoom_t * fixed_y_max + (1 - y_zoom_t) * current_y_max
    
    # Calculate axis scaling factors directly
    x_axis_length = 7  # Same as in get_x_axis
    y_axis_length = 5  # Same as in get_y_axis
    
    x_scale = (upper_bound_x - lower_bound_x) / x_axis_length
    y_scale = (y_max - y_min) / y_axis_length
    
    # Extract origin positions from the provided positions
    # Adjust for the fact that the position is the center of the axis
    origin_x = x_axis_position[0] - x_axis_length / 2
    origin_y = y_axis_position[1] - y_axis_length / 2
    
    # Create scatter plot points
    dots = m.VGroup()
    
    for x_val, y_val in zip(x_values, y_values):
        # Calculate normalized position within the data range (0 to 1)
        x_norm = (x_val - lower_bound_x) / (upper_bound_x - lower_bound_x)
        y_norm = (y_val - y_min) / (y_max - y_min)
        
        # Convert to canvas position
        x_pos = origin_x + x_norm * x_axis_length
        y_pos = origin_y + y_norm * y_axis_length
        
        # Create dot
        dot = m.Dot(
            point=[x_pos, y_pos, 0],
            radius=0.05,
            stroke_width=0,
            fill_opacity=0.8
        )
        dot.set_color(YELLOW)
        dots.add(dot)
    
    return dots

def time_to_bounds(t, intial_bounds, final_bounds):
    lower_bound=t*(final_bounds[0]-intial_bounds[0])+intial_bounds[0]
    upper_bound=t*(final_bounds[1]-intial_bounds[1])+intial_bounds[1]
    return lower_bound, upper_bound

class P21(WelchScene):
    '''
    Code is messy and animation is a little stuttery - but I think I coudl ship this if I need to
    Probably makes sense to keep moving, this is a small scene.

    Hmm stuttering fix could be really simple - I should try a quick linear interpolation for y axis scaling. 
    '''
    def construct(self):
        initial_x_range = [-0.027, 0.013]
        final_x_range = [-1.1, 4.1]

        initial_y_range = [0.3887, 0.3940]
        final_y_range = [0.15, 0.6]

        initial_time = 0.0
        t_tracker = m.ValueTracker(initial_time)
        
        # Add a second tracker for the y-axis zoom
        y_zoom_tracker = m.ValueTracker(0.0)

        x_axis_position = [0, -2, 0]
        y_axis_position = [-3.84, 0.73, 0]
        
        # Create axes
        x_axis = m.always_redraw(lambda: get_x_axis(
            t_tracker.get_value(), 
            initial_x_range, 
            final_x_range, 
            x_axis_position
        ))
        
        # First y-axis that changes with the scatter plot
        y_axis = m.always_redraw(lambda: get_y_axis(
            t_tracker.get_value(), 
            initial_x_range, 
            final_x_range, 
            y_axis_position
        ))
        
        # Second y-axis with fixed bounds that will be used for the second zoom
        fixed_y_axis = m.always_redraw(lambda: get_fixed_y_axis(
            t_tracker.get_value(), 
            initial_x_range, 
            final_x_range, 
            y_axis_position,
            y_zoom_tracker.get_value()
        ))
        
        # Create scatter plot with direct time parameter
        scatter = m.always_redraw(lambda: get_scatter_points(
            t_tracker.get_value(),
            initial_x_range,
            final_x_range,
            x_axis_position,
            y_axis_position
        ))
        
        # Create scatter plot that scales with the interpolated y-axis
        interpolated_scatter = m.always_redraw(lambda: get_scatter_points_with_interpolated_y(
            t_tracker.get_value(),
            initial_x_range,
            final_x_range,
            x_axis_position,
            y_axis_position,
            y_zoom_tracker.get_value()
        ))
        
        self.add(x_axis, y_axis, scatter)
        self.wait()
        
        # First animation: zoom out with scatter plot
        self.play(t_tracker.animate.set_value(1.0), run_time=4)
        self.wait()
        
        # Second animation: replace y-axis with fixed bounds version and zoom out
        self.remove(y_axis, scatter)
        self.add(fixed_y_axis, interpolated_scatter)
        self.play(y_zoom_tracker.animate.set_value(1.0), run_time=3)
        self.wait()




# class P21(InteractiveScene):
#     def construct(self):

#         initial_x_range=[-0.027, 0.013]
#         final_x_range=[-1.1, 4.1]

#         # indices_in_range=np.logical_and(xs1>initial_x_range[0], xs1<initial_x_range[1])

#         initial_time=0.0
#         t_tracker = ValueTracker(initial_time)

#         x_axis_position = [0, -2, 0]  # Adjusted position
#         y_axis_position = [-3.84, 0.73, 0]  # Adjusted position
#         x_axis = always_redraw(lambda: get_x_axis(t_tracker.get_value(), initial_x_range, final_x_range, x_axis_position))
#         y_axis = always_redraw(lambda: get_y_axis(t_tracker.get_value(), initial_x_range, final_x_range, y_axis_position))

#         # Create scatter plot that updates with the axes
#         scatter = always_redraw(lambda: get_scatter_points(t_tracker.get_value(), initial_x_range, final_x_range, x_axis, y_axis))
        

#         self.add(x_axis, y_axis, scatter)
#         self.wait()


#         self.play(t_tracker.animate.set_value(0.1), run_time=2)
#         self.wait()









# class P21Hacking2(InteractiveScene):
#     '''
#     Ok I'm thinking that the version of this scene where we zoom out on an alrady constructed big Graph might be easier
#     than growing the graph in place. The weird thing will be x-axis spacing etc, that will take some noodling. 
#     Hmm figuring out how to handle the axes while I do the big zoom out is tricker than expected
#     I think it's probably worth figuring out, the laternative ot to do maptlotlib, but I really want the 
#     buttery manim animation start/ends, and I think these are components I will reuse. 

#     '''
#     def construct(self):

#         initial_time=3
#         t_tracker = ValueTracker(initial_time)

#         x_axis = always_redraw(lambda: get_x_axis(t_tracker.get_value()))

#         self.add(x_axis)
#         self.wait()

#         self.play(t_tracker.animate.set_value(50), run_time=2)
#         self.wait()



        # Ok this patter is pretty clunky, but I think it can work. 
        # I was thinking I could zoom out as I grew the axis, but 
        # That makes my ticks and numbers shrink!
        # I need to modify my axis to have like an in place mode, or just act that way be default
        # where it gets longer as far as teh tick marks are concerned, but stays the length 
        # on the canvas. basically I need to change the start and stop values/scale on the
        # a static line - that should be do-able. man this is more complicated than I thought. 



        # x_axis=WelchXAxis(        
        #     x_ticks=[-0.003, 0.007, 0.017],  
        #     x_tick_height=0.15,        
        #     x_label_font_size=24,           
        #     stroke_width=3, 
        #     arrow_tip_scale=0.1
        # )

        # y_axis=WelchYAxis(  
        #     y_ticks=[0.3901, 0.3916, 0.3930],  
        #     y_tick_width=0.15,        
        #     y_label_font_size=24,            
        #     stroke_width=3,          
        #     arrow_tip_scale=0.1,      
        # )



        # x_ticks, x_axis_min, x_axis_max=generate_nice_ticks(0, 4, min_ticks=3, max_ticks=16)
        # y_ticks, y_axis_min, y_axis_max=generate_nice_ticks(0, 2.0, min_ticks=3, max_ticks=16)

        # x_axis=WelchXAxis(        
        #     x_ticks=x_ticks,  
        #     x_tick_height=0.15,        
        #     x_label_font_size=24,           
        #     stroke_width=3, 
        #     arrow_tip_scale=0.1,
        #     x_min=x_axis_min, 
        #     x_max=x_axis_max
        # )

        # y_axis=WelchYAxis(        
        #     y_ticks=y_ticks,  
        #     y_tick_width=0.15,        
        #     y_label_font_size=24,           
        #     stroke_width=3, 
        #     arrow_tip_scale=0.1,
        #     y_min=0, 
        #     y_max=y_axis_max
        # )

        # Is it like i try to figure out how to smoothly animate the length of the axis and I have the tick marks automatically "fill in"?
        # Ok i tried that with claude at least, looks terrible lol. 
        # Hmm...
        # Ok I think it's worth looking at stephen_playground_og.py -> looks like grant showed me how to solve an analagous problem there. 
        

        # # x_axis.move_to([-2, -2, 0])
        # initial_length=5
        # l_tracker = ValueTracker(initial_length)

        # self.add(x_axis) #, y_axis)

        # self.play.animate(x_axis.animate.set_max_val(5), run_time=5)

        # x_axis.set_max_val(5)
        # # self.wait()

        # x_axis.set_max_val(6)

        # x_axis.set_max_val(7)


        # x_axis.update_from_range(-1, 7)
        # self.play.animate()
        # # self.embed()



class AxisHacking(WelchScene):
    def construct(self):

        x_axis=WelchXAxis(        
            x_ticks=[1, 2, 3, 4, 5],  
            x_tick_height=0.15,        
            x_label_font_size=24,           
            stroke_width=3, 
            arrow_tip_scale=0.1
        )

        y_axis=WelchYAxis(  
            y_ticks=[1, 2, 3, 4, 5],  
            y_tick_width=0.15,        
            y_label_font_size=24,            
            stroke_width=3,          
            arrow_tip_scale=0.1,      
        )

        # # x_axis.move_to([-2, -2, 0])


        self.add(x_axis, y_axis)
        # self.wait()

        # self.embed()
