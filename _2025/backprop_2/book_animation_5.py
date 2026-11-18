from welch_paths import welch_path
import manimgx as m
from welch_geometry import diagram_axes_3d
from welch_surface import SampledSurface
from welch_scene import WelchScene, image_at_height
import numpy as np

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
graphics_dir=welch_path('backprop_3/graphics/') #Point to folder where map images are

BLUE='#65c8d0'
GREEN='#00a14b'
RED='#ed1c24'
PURPLE='#7f3f98'

class LinearPlane(SampledSurface):
    """A plane defined by z = m1*x1 + m2*x2 + b"""
    def __init__(self, axes, m1=0.5, m2=0.3, b=1.0, vertical_viz_scale=0.5, **kwargs):
        self.axes = axes
        self.m1 = m1
        self.m2 = m2 
        self.b = b
        self.vertical_viz_scale = vertical_viz_scale
        super().__init__(
            u_range=(-5, 5),
            v_range=(-5, 5),
            resolution=(64, 64),
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

def find_plane_intersection(plane1, plane2):
    """
    Find the intersection line of two planes.
    Each plane is defined as z = m1*x1 + m2*x2 + b
    
    For plane1: z = m1_1*x1 + m2_1*x2 + b1
    For plane2: z = m1_2*x1 + m2_2*x2 + b2
    
    At intersection: m1_1*x1 + m2_1*x2 + b1 = m1_2*x1 + m2_2*x2 + b2
    Rearranging: (m1_1 - m1_2)*x1 + (m2_1 - m2_2)*x2 = b2 - b1
    
    This gives us a line in the x1-x2 plane. We can parameterize it.
    """
    # Coefficients of the intersection equation
    a = plane1.m1 - plane2.m1  # coefficient of x1
    b = plane1.m2 - plane2.m2  # coefficient of x2  
    c = plane2.b - plane1.b    # constant term
    
    # We need to solve: a*x1 + b*x2 = c
    # We'll parameterize this line using parameter t
    
    if abs(a) > abs(b):  # Use x2 as parameter
        # x1 = (c - b*x2) / a, let x2 = t
        def get_point(t):
            x2 = t
            x1 = (c - b * x2) / a if abs(a) > 1e-10 else 0
            # Calculate z using either plane (they should be equal at intersection)
            z = plane1.vertical_viz_scale * (plane1.m1 * x1 + plane1.m2 * x2 + plane1.b)
            return np.array([x1, x2, z])
    else:  # Use x1 as parameter
        # x2 = (c - a*x1) / b, let x1 = t
        def get_point(t):
            x1 = t
            x2 = (c - a * x1) / b if abs(b) > 1e-10 else 0
            # Calculate z using either plane (they should be equal at intersection)
            z = plane1.vertical_viz_scale * (plane1.m1 * x1 + plane1.m2 * x2 + plane1.b)
            return np.array([x1, x2, z])
    
    return get_point

class IntersectionLine(m.ParametricFunction):
    """A parametric line representing the intersection of two planes"""
    def __init__(self, axes, plane1, plane2, t_range=(-5, 5, 0.1), **kwargs):
        self.axes = axes
        self.plane1 = plane1
        self.plane2 = plane2
        
        # Get the intersection function
        self.intersection_func = find_plane_intersection(plane1, plane2)
        
        super().__init__(
            lambda t: self.axes.c2p(*self.intersection_func(t)),
            t_range=t_range,
            **kwargs
        )

class HalfPlane(SampledSurface):
    def __init__(self, axes, base_plane, other_plane, above=True, **kwargs):
        self.axes = axes
        self.base_plane = base_plane
        self.other_plane = other_plane
        self.above = above
        super().__init__(
            u_range=(-5, 5),
            v_range=(-5, 5),
            resolution=(32, 32),
            **kwargs
        )
    
    def uv_func(self, u, v):
        x1, x2 = u, v
        z1 = self.base_plane.vertical_viz_scale * (self.base_plane.m1 * x1 + self.base_plane.m2 * x2 + self.base_plane.b)
        z2 = self.other_plane.vertical_viz_scale * (self.other_plane.m1 * x1 + self.other_plane.m2 * x2 + self.other_plane.b)
        
        # Only show points where this plane is above/below the other
        if self.above and z1 <= z2:
            return self.axes.c2p(x1, x2, -10)  # Hide below visible range
        elif not self.above and z1 >= z2:
            return self.axes.c2p(x1, x2, -10)  # Hide below visible range
        else:
            return self.axes.c2p(x1, x2, z1)

class FlatHalfPlane(SampledSurface):
    def __init__(self, axes, base_plane, other_plane, above=True, **kwargs):
        self.axes = axes
        self.base_plane = base_plane
        self.other_plane = other_plane
        self.above = above
        super().__init__(
            u_range=(-5, 5),
            v_range=(-5, 5),
            resolution=(32, 32),
            **kwargs
        )
    
    def uv_func(self, u, v):
        x1, x2 = u, v
        z1 = self.base_plane.vertical_viz_scale * (self.base_plane.m1 * x1 + self.base_plane.m2 * x2 + self.base_plane.b)
        z2 = self.other_plane.vertical_viz_scale * (self.other_plane.m1 * x1 + self.other_plane.m2 * x2 + self.other_plane.b)
        
        # Only show points where this plane is above/below the other, but flatten to z=0
        if self.above and z1 <= z2:
            return self.axes.c2p(x1, x2, -10)  # Hide below visible range
        elif not self.above and z1 >= z2:
            return self.axes.c2p(x1, x2, -10)  # Hide below visible range
        else:
            return self.axes.c2p(x1, x2, 0)  # Flatten to z=0

class p17(WelchScene):
    def construct(self):
        # map_img=ImageMobject(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        map_img=image_at_height(welch_path('ai_book/3_backprop_2/graphics/barrle_hertog_map_32.png'))
        map_img.move_to(m.ORIGIN)
        map_img.scale(0.25)
        
        axes_1 = diagram_axes_3d(
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
        plane_1.set_color(BLUE)
        
        plane_2=LinearPlane(axes_1, -1, 0.2, 4, vertical_viz_scale=0.2)
        plane_2.set_opacity(0.4)
        plane_2.set_color(GREEN)
        
        # Create the intersection line
        intersection_line = IntersectionLine(
            axes_1, 
            plane_1, 
            plane_2, 
            t_range=(-5, 5, 0.1),
            color='#000000',
            stroke_width=4
        )
            
        self.set_view(0, 0, 0, (0.01, -0.01, 0.0), 1.29)
        self.add(map_img)

        self.wait()
        self.play(m.Create(plane_1), m.Create(plane_2), self.animate_view(0, 46, 0, (-0.03, 0.02, 0.01), 1.62), run_time=5)
        self.wait()

        self.play(self.animate_view(-16, 47, 0, (-0.04, 0.02, 0.0), 1.62))
        # self.play(ShowCreation(intersection_line))
        self.wait()
        map_img.set_opacity(0.4)

        self.set_view(0, 48, 0, (-0.03, -0.14, 0.11), 1.17)



        self.wait()








        ## Move to overhead view, bring down line flat onto z=0, and each color half plane onto z=0
        
        # class FlatIntersectionLine(m.ParametricFunction):
        #     def __init__(self, axes, plane1, plane2, t_range=(-5, 5, 0.1), **kwargs):
        #         self.axes = axes
        #         self.plane1 = plane1
        #         self.plane2 = plane2
        #         self.intersection_func = find_plane_intersection(plane1, plane2)
                
        #         super().__init__(
        #             lambda t: self.axes.c2p(*self.intersection_func(t)[:2], 0),  # Flatten to z=0
        #             t_range=t_range,
        #             **kwargs
        #         )
        
        # flat_intersection_line = FlatIntersectionLine(
        #     axes_1, 
        #     plane_1, 
        #     plane_2, 
        #     t_range=(-5, 5, 0.1),
        #     color='#FF00FF',
        #     stroke_width=4
        # )

        # self.wait()

        # # self.play(ReplacementTransform(intersection_line, flat_intersection_line), 
        # #           plane_1.animate.set_opacity(0.0),
        # #           plane_2.animate.set_opacity(0.0),
        # #           self.animate_view(0, 2, 0, (-0.04, 0.02, 0.0), 1.62),
        # #           run_time=4)

        # self.wait()

        #Ok not perfect, but good for an overhead pass -> gut to my head I just added the final shaded flat regions in illustrator!




        # First, move to overhead view
        # self.play(self.animate_view(-90, 0, 0, (0, 0, 0), 1.62), run_time=3)
        # self.wait()
        
        # # Create flattened intersection line (same line but at z=0)

        
        # # Create the half planes
        # half_plane_1 = HalfPlane(axes_1, plane_1, plane_2, above=True, color='#00FFFF', opacity=0.3)
        # half_plane_2 = HalfPlane(axes_1, plane_2, plane_1, above=True, color=YELLOW, opacity=0.3)
        
        # # Create flattened versions at z=0
        # flat_half_plane_1 = FlatHalfPlane(axes_1, plane_1, plane_2, above=True, color='#00FFFF', opacity=0.6)
        # flat_half_plane_2 = FlatHalfPlane(axes_1, plane_2, plane_1, above=True, color=YELLOW, opacity=0.6)
        
        # # Animate the transformation
        # self.play(
        #     Transform(intersection_line, flat_intersection_line),
        #     Transform(plane_1, flat_half_plane_1),
        #     Transform(plane_2, flat_half_plane_2),
        #     run_time=4
        # )
        
        # self.wait(3)

        self.wait(20)
        pass

class p21_final_planes(WelchScene):
    def construct(self):
        map_img=image_at_height(welch_path('ai_book/3_backprop_2/graphics/barrle_hertog_map_32.png'))
        map_img.move_to(m.ORIGIN)
        map_img.scale(0.25)
        
        axes_1 = diagram_axes_3d(
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
        plane_1.set_opacity(0.2)
        plane_1.set_color('#FF00FF')
        
        plane_2=LinearPlane(axes_1, -1, 0.2, 4, vertical_viz_scale=0.2)
        plane_2.set_opacity(0.5)
        plane_2.set_color(CHILL_GREEN)
        
        # Create the intersection line
        intersection_line = IntersectionLine(
            axes_1, 
            plane_1, 
            plane_2, 
            t_range=(-5, 5, 0.1),
            color=m.WHITE,
            stroke_width=4
        )
            
        self.set_view(0, 0, 0, (0.01, -0.01, 0.0), 1.29)
        self.add(map_img)

        self.wait()
        self.play(m.Create(plane_1), m.Create(plane_2), self.animate_view(0, 46, 0, (-0.03, 0.02, 0.01), 1.62), run_time=5)
        self.wait()

        self.play(self.animate_view(-16, 47, 0, (-0.04, 0.02, 0.0), 1.62))
        self.play(m.Create(intersection_line))
        self.wait()
        
        ## Move to overhead view, bring down line flat onto z=0, and each color half plane onto z=0
        
        class FlatIntersectionLine(m.ParametricFunction):
            def __init__(self, axes, plane1, plane2, t_range=(-5, 5, 0.1), **kwargs):
                self.axes = axes
                self.plane1 = plane1
                self.plane2 = plane2
                self.intersection_func = find_plane_intersection(plane1, plane2)
                
                super().__init__(
                    lambda t: self.axes.c2p(*self.intersection_func(t)[:2], 0),  # Flatten to z=0
                    t_range=t_range,
                    **kwargs
                )
        
        flat_intersection_line = FlatIntersectionLine(
            axes_1, 
            plane_1, 
            plane_2, 
            t_range=(-5, 5, 0.1),
            color=m.WHITE,
            stroke_width=4
        )

        self.wait()

        # self.play(ReplacementTransform(intersection_line, flat_intersection_line), 
        #           plane_1.animate.set_opacity(0.0),
        #           plane_2.animate.set_opacity(0.0),
        #           self.animate_view(0, 2, 0, (-0.04, 0.02, 0.0), 1.62),
        #           run_time=4)

        self.wait()

        #Ok not perfect, but good for an overhead pass -> gut to my head I just added the final shaded flat regions in illustrator!




        # First, move to overhead view
        # self.play(self.animate_view(-90, 0, 0, (0, 0, 0), 1.62), run_time=3)
        # self.wait()
        
        # # Create flattened intersection line (same line but at z=0)

        
        # # Create the half planes
        # half_plane_1 = HalfPlane(axes_1, plane_1, plane_2, above=True, color='#00FFFF', opacity=0.3)
        # half_plane_2 = HalfPlane(axes_1, plane_2, plane_1, above=True, color=YELLOW, opacity=0.3)
        
        # # Create flattened versions at z=0
        # flat_half_plane_1 = FlatHalfPlane(axes_1, plane_1, plane_2, above=True, color='#00FFFF', opacity=0.6)
        # flat_half_plane_2 = FlatHalfPlane(axes_1, plane_2, plane_1, above=True, color=YELLOW, opacity=0.6)
        
        # # Animate the transformation
        # self.play(
        #     Transform(intersection_line, flat_intersection_line),
        #     Transform(plane_1, flat_half_plane_1),
        #     Transform(plane_2, flat_half_plane_2),
        #     run_time=4
        # )
        
        # self.wait(3)

        self.wait(20)
        pass















