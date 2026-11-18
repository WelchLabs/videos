import manimgx as m
from welch_geometry import diagram_axes
import sys
from welch_scene import WelchScene
from functools import partial
sys.path.append('/Users/stephen/manim_videos/_2025/backprop_1')
from backprop_data import all_probs_1, losses1, xs1


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

class P21(WelchScene):
    def construct(self):

        # Create custom axes for first quadrant only
        # y-axis at x=-1, x-axis at y=0.15
        y_axis = m.Line(
            start=(-1, -1.2),
            end=(-1, 1.1),  # Shortened to make room for arrow
            color=CHILL_BROWN,
            stroke_width=2,
        )
        
        # Create custom arrow tip for y-axis
        y_arrow_tip = m.Triangle(
            fill_opacity=1,
            stroke_width=0,
            color=CHILL_BROWN,
        )
        y_arrow_tip.scale(0.1)  # Adjust size as needed
        y_arrow_tip.rotate(m.PI/2)  # Rotate to point upward
        # Position the arrow tip at the top of y-axis
        y_arrow_tip.shift([-1, 1.2, 0])
        
        # x-axis at y=0.15
        x_axis = m.Line(
            start=(-1.2, 0.15),
            end=(1.1, 0.15),  # Shortened to make room for arrow
            color=CHILL_BROWN,
            stroke_width=2,
        )
        
        # Create custom arrow tip for x-axis
        x_arrow_tip = m.Triangle(
            fill_opacity=1,
            stroke_width=0,
            color=CHILL_BROWN,
        )
        x_arrow_tip.scale(0.1)  # Adjust size as needed
        # Position the arrow tip at the right of x-axis
        x_arrow_tip.shift([1.2, 0.15, 0])
        
        # Create a custom axes object for plotting
        axes = diagram_axes(
            x_range=[-1.2, 1.2, 1.0],
            y_range=[-1.2, 1.2, 1.0],
            axis_config={
                "include_ticks": False,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip": False,  # We'll add custom tips
            },
        )
        
        # Hide the default axes
        axes.x_axis.set_opacity(0)
        axes.y_axis.set_opacity(0)
        
        # Create the graph for all_probs_1
        prob_graph = axes.plot(
            lambda x: all_probs_1[int((x + 1.2) * len(all_probs_1) / 2.4)],
            x_range=[-1.2, 1.2],
            color=YELLOW,
        )
        
        # Create the graph for losses1
        loss_graph = axes.plot(
            lambda x: losses1[int((x + 1.2) * len(losses1) / 2.4)],
            x_range=[-1.2, 1.2],
            color=BLUE,
        )
        
        # Add everything to the scene
        self.add(axes, y_axis, y_arrow_tip, x_axis, x_arrow_tip, prob_graph, loss_graph)