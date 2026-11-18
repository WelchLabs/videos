from welch_paths import welch_path
import manimgx as m
from welch_geometry import diagram_axes_3d
from welch_surface import SampledSurface
from welch_scene import WelchScene, image_at_height
from functools import partial
import numpy as np
import torch

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'
GREEN='#00a14b'

svg_path=welch_path('backprop2/graphics/to_manim')

def manual_camera_interpolation(start_orientation, end_orientation, num_steps):
    """
    Linearly interpolate between two camera orientations.
    
    Parameters:
    - start_orientation: List containing camera parameters with a tuple at index 3
    - end_orientation: List containing camera parameters with a tuple at index 3
    - num_steps: Number of interpolation steps (including start and end)
    
    Returns:
    - List of interpolated orientations
    """
    result = []
    
    for step in range(num_steps):
        # Calculate interpolation factor (0 to 1)
        t = step / (num_steps - 1) if num_steps > 1 else 0
        
        # Create a new orientation for this step
        interpolated = []
        
        for i in range(len(start_orientation)):
            if i == 3:  # Handle the tuple at position 3
                start_tuple = start_orientation[i]
                end_tuple = end_orientation[i]
                
                # Interpolate each element of the tuple
                interpolated_tuple = tuple(
                    start_tuple[j] + t * (end_tuple[j] - start_tuple[j])
                    for j in range(len(start_tuple))
                )
                
                interpolated.append(interpolated_tuple)
            else:  # Handle regular numeric values
                start_val = start_orientation[i]
                end_val = end_orientation[i]
                interpolated_val = start_val + t * (end_val - start_val)
                interpolated.append(interpolated_val)
        
        result.append(interpolated)
    
    return result

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
            u_range=(-5, 5),
            v_range=(-5, 5),
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



class p50(WelchScene):
    def construct(self):

        baarle_map=image_at_height(svg_path +'/map_exports.00_01_07_25.Still005.png')
        # baarle_map.rotate(90*DEGREES, [1,0,0])

        baarle_map.scale(0.25)
        # baarle_map.move_to([0.96,0,0])


        axes_1 = diagram_axes_3d(
            # x_range=[-15, 15, 1],
            # y_range=[-15, 15, 1],
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            z_range=[-3.5, 3.5, 1],
            width=1,
            height=1,
            depth=1,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )

        plane_1=LinearPlane(axes_1, 0.5, 1.2, 4, vertical_viz_scale=0.2)
        plane_1.set_opacity(0.4)
        plane_1.set_color('#00FFFF')

        plane_2=LinearPlane(axes_1, -1, 0.2, 4, vertical_viz_scale=0.2)
        plane_2.set_opacity(0.4)
        plane_2.set_color(YELLOW)


        # self.set_view(0, 0, 0, (0.02, -0.01, 0.0), 1.23)
        # self.set_view(0, 0, 0, (-0.08, -0.05, -0.15), 1.37)
        self.set_view(0, 0, 0, (-0.06, -0.0, -0.15), 1.23)
        self.add(baarle_map)
        self.wait()

        self.play(self.animate_view(-34, 52, 0, (-0.04, -0.01, -0.05), 1.58),
                 m.FadeIn(axes_1), m.FadeIn(plane_1), m.FadeIn(plane_2), run_time=5)

        # self.add(axes_1, plane_1, plane_2)
        # self.set_view(-29, 53, 0, (0.04, 0.06, 0.09), 2.05)
        self.wait()

        #ok static looks good, now I want to pan around while smootly changing the LinearPlane's parameters. 
        num_total_steps=320
        # start_orientation = [-33, 58, 0, (-0.1, 0.01, 0.05), 1.83]
        start_orientation = [-34, 52, 0, (-0.04, -0.01, -0.05), 1.58]
        end_orientation = [27, 48, 0, (-0.04, -0.01, -0.05), 1.58]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=num_total_steps)

        m11s=np.linspace(0.5, 0.8, num_total_steps)
        m12s=np.linspace(1.2, 1.5, num_total_steps)
        m21s=np.linspace(-1, -1.2, num_total_steps)
        m22s=np.linspace(0.2, 0.5, num_total_steps)

        for i in range(num_total_steps):

            self.remove(plane_1, plane_2)

            plane_1=LinearPlane(axes_1, m11s[i], m12s[i], 4, vertical_viz_scale=0.2)
            plane_1.set_opacity(0.4)
            plane_1.set_color('#00FFFF')

            plane_2=LinearPlane(axes_1, m21s[i], m22s[i], 4, vertical_viz_scale=0.2)
            plane_2.set_opacity(0.4)
            plane_2.set_color(YELLOW)

            self.add(plane_1, plane_2)
            self.set_view(*interp_orientations[i])

            self.wait(0.1)


        self.wait()


        # num_total_steps=num_time_steps*2 #Crank this for final viz
        # start_orientation=[142, 34, 0, (-0.09, -0.77, 0.15), 3.55]
        # end_orientation=[121, 20, 0, (0.01, -0.46, 0.57), 1.95]
        # interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=num_total_steps)

        # surface_update_counter=1
        # frames_per_surface_upddate=np.floor(num_total_steps/num_time_steps)
        # for i in range(1, num_total_steps):
        #     # print(i, len(interp_orientations))
        #     if i%frames_per_surface_upddate==0 and surface_update_counter<len(surfaces):
        #         if surface_update_counter==1:
        #             self.remove(ts)
        #             self.remove(u_gridlines, v_gridlines) 
        #         else:
        #             self.remove(surfaces[surface_update_counter-1])
        #             self.remove(grids[surface_update_counter-1])

        #         self.add(surfaces[surface_update_counter])
        #         self.add(grids[surface_update_counter])

        #         new_point_coords=surf_functions[surface_update_counter](*starting_coords)
        #         s2.move_to(new_point_coords) #This should make point move down smoothly. 
        #         surface_update_counter+=1

        #     # print(i, len(interp_orientations))
        #     self.set_view(*interp_orientations[i])
        #     self.wait(0.1)

        # self.wait()



        self.wait()
        pass



























