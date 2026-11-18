import manimgx as m
from welch_geometry import diagram_axes
import sys
from welch_scene import WelchScene
sys.path.append('/Users/stephen/manim/videos/_2025/backprop_1')
from backprop_data import xs1, losses1, all_probs_1

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'


def get_prob_graph(axes):
    curve = m.VMobject()
    curve.set_points_smoothly(axes.c2p(xs1, all_probs_1)) #To Do: replace with scatter plot
    return curve

def get_loss_graph(axes):
    curve = m.VMobject()
    curve.set_points_smoothly(axes.c2p(xs1, losses1)) #To Do: replace with scatter plot
    return curve


def create_axes_with_ranges(x_range, y_range):
    """
    Create a new Axes object with the specified ranges.
    
    Parameters:
    -----------
    x_range : [float, float, float]
        The x-range [min, max, step]
    y_range : [float, float, float]
        The y-range [min, max, step]
        
    Returns:
    --------
    Axes
        A new Axes object with the specified ranges
    """
    return diagram_axes(
        x_range=x_range,
        y_range=y_range,
        axis_config={
            "include_ticks": True,
            "color": CHILL_BROWN,
            "stroke_width": 2,
            "include_tip": True,
            "tip_config": {
                "fill_opacity": 1,
                "width": 0.1,
                "length": 0.1
            }
        },
    )


class P21(WelchScene):
    def construct(self):
        # Initial axes setup
        initial_x_range = [-1, 4, 1.0]
        initial_y_range = [0, 1.6, 0.2]
        
        # Target ranges for zoom
        target_x_range = [1, 2, 0.1]
        target_y_range = [0.5, 1.0, 0.1]
        
        # Create a tracker for the animation
        zoom_tracker = m.ValueTracker(0)
        
        # Create the axes with always_redraw to update during animation
        axes = m.always_redraw(lambda: create_axes_with_ranges(
            [
                m.interpolate(initial_x_range[0], target_x_range[0], zoom_tracker.get_value()),
                m.interpolate(initial_x_range[1], target_x_range[1], zoom_tracker.get_value()),
                m.interpolate(initial_x_range[2], target_x_range[2], zoom_tracker.get_value())
            ],
            [
                m.interpolate(initial_y_range[0], target_y_range[0], zoom_tracker.get_value()),
                m.interpolate(initial_y_range[1], target_y_range[1], zoom_tracker.get_value()),
                m.interpolate(initial_y_range[2], target_y_range[2], zoom_tracker.get_value())
            ]
        ))
        
        # Create the graph that will update with the axes
        graph = m.always_redraw(lambda: get_prob_graph(axes))
        
        # Add everything to the scene
        self.add(axes, graph)
        self.wait()
        
        # Animate the zoom
        self.play(
            zoom_tracker.animate.set_value(1),
            run_time=2,
            rate_func=m.smooth
        )
        
        self.wait()

