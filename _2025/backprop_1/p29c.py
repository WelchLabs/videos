import manimgx as m
from welch_geometry import diagram_axes_3d
import numpy as np
from welch_scene import WelchScene

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'


class Line3D(m.VMobject):
    def __init__(self, start, end, **kwargs):
        super().__init__(**kwargs)
        self.set_points_as_corners([start, end])

class Dot3D(m.Sphere):
    def __init__(self, point=m.ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(point)


class p29cV3(WelchScene):
    def construct(self):

        
        x_range = [-1, 4]
        y_range = [-1, 4]
        z_range = [-1, 4]
        
        # Create axes
        axes = diagram_axes_3d(
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            height=5,
            width=5,
            depth=5,
            axis_config={
                "include_ticks": True,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip": True,
                "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
            }
        )
        
        # Create the grid values
        num_lines = 15  # Adjust as needed - 20 looks kinda nuts - I don't want to move to far off reality here though!
        u_values = np.linspace(x_range[0], x_range[1], num_lines)
        v_values = np.linspace(y_range[0], y_range[1], num_lines)
        w_values = np.linspace(z_range[0], z_range[1], num_lines)
        
        # Create 3D grid lines
        grid_lines = m.VGroup()
        
        # X-direction lines
        for y in v_values:  # Skip every other line for cleaner look
            for z in w_values:
                start = np.array([x_range[0], y, z])
                end = np.array([x_range[1], y, z])
                line = Line3D(start, end, color=m.WHITE) #, width=1)
                line.set_opacity(0.25)
                line.set_stroke(width=1.5)
                grid_lines.add(line)
        
        # Y-direction lines
        for x in u_values:
            for z in w_values:
                start = np.array([x, y_range[0], z])
                end = np.array([x, y_range[1], z])
                line = Line3D(start, end, color=m.WHITE) #, width=1)
                line.set_stroke(width=1.5)
                line.set_opacity(0.25)
                grid_lines.add(line)
        
        # Z-direction lines
        for x in u_values:
            for y in v_values:
                start = np.array([x, y, z_range[0]])
                end = np.array([x, y, z_range[1]])
                line = Line3D(start, end, color=m.WHITE) #, width=1)
                line.set_stroke(width=1.5)
                line.set_opacity(0.25)
                grid_lines.add(line)
        
        self.wait()
    
        self.set_view(-67, 64, 0, (0.75, 1.02, 1.24), 10.52)
        self.play(m.Create(grid_lines), run_time=2)

        stride = 1  # Only use every second value
        intersection_dots = m.Group()

        # for x in np.concatenate((u_values[:3], u_values[-3:])): #u_values: #[::stride]:
        #     for y in np.concatenate((v_values[:3], v_values[-3:])): #v_values: #[::stride]:
        #         for z in np.concatenate((w_values[:3], w_values[-3:])): #w_values: #[::stride]:
        #             dot = Dot3D(
        #                 point=[x, y, z],
        #                 radius=0.035,
        #                 color=WHITE
        #             )
        #             dot.set_opacity(0.8)
        #             intersection_dots.add(dot)

        # outer_indices = [0, 1, num_lines-2, num_lines-1]
        outer_indices = [0, num_lines-1]
        
        intersection_dots = m.Group()
        for x_idx, x in enumerate(u_values):
            for y_idx, y in enumerate(v_values):
                for z_idx, z in enumerate(w_values):
                    # Only add dots if the point is in any of the 3 outer layers on any dimension
                    if (x_idx in outer_indices or 
                        y_idx in outer_indices or 
                        z_idx in outer_indices):
                        dot = Dot3D(
                            point=[x, y, z],
                            radius=0.035,
                            color=m.WHITE
                        )
                        dot.set_opacity(0.8)
                        intersection_dots.add(dot)

        # Add animation to show the dots
        self.play(m.Create(intersection_dots), run_time=2)
        self.wait()

        self.play(self.animate_view(-38, 63, 0, (0.75, 1.02, 1.24), 10.52), run_time=4)




        self.wait(20)
        pass



