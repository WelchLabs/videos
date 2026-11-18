from welch_paths import welch_path
from welch_surface import set_surface_shading
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
import manimgx as gx
import numpy as np
from decision_boundary_utils import create_decision_boundary_lines
from plane_folding_utils import (
    create_3d_polygon_regions_multi,
    get_polygon_corners_multi,
    get_relu_joint,
    line_from_joint_points_1,
    surface_func_general,
    surface_func_second_layer_no_relu_multi,
)
from welch_scene import WelchScene, image_at_height
from welch_surface import parametric_surface, textured_surface


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'


graphics_dir=welch_path('backprop_3/graphics/') #Point to folder where map images are

class plane_folding_sketch_single_layer_4(WelchScene):
    def construct(self):

        map_img=image_at_height(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        map_img.scale_to_fit_height(2)  # This sets the height to 2 units (-1 to 1)
        map_img.scale_to_fit_width(2)   # This sets the width to 2 units (-1 to 1)
        map_img.move_to(gx.ORIGIN)

        viz_scale_1=0.25
        viz_scale_2=0.1

        #New 3 layer weights!
        w1 = np.array([[-1.8741, 2.12215],
         [-2.39381, -1.24014],
         [-0.940185, 1.40271],
         [2.04548, 0.489156]], dtype=np.float32)
        b1 = np.array([-0.00892048, -1.32954, 1.71349, 0.940607], dtype=np.float32)
        w2 = np.array([[2.56674, 2.26244, -1.40175, 0.737865],
         [-2.58904, -3.0681, 1.08007, -1.32219]], dtype=np.float32)
        b2 = np.array([-0.852075, 0.492386], dtype=np.float32)

        # First hidden neuron
        surface_func_11=partial(surface_func_general, w1=w1[0,0], w2=w1[0,1], b=b1[0], viz_scale=viz_scale_1)
        bent_surface_11 = parametric_surface(surface_func_11, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts11=textured_surface(bent_surface_11, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts11, 0,0,0)
        ts11.set_opacity(0.75)
        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.5)
        group_11=gx.Group(ts11, joint_line_11)

        # Second hidden neuron
        surface_func_12=partial(surface_func_general, w1=w1[1,0], w2=w1[1,1], b=b1[1], viz_scale=viz_scale_1)
        bent_surface_12 = parametric_surface(surface_func_12, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts12=textured_surface(bent_surface_12, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts12, 0,0,0)
        ts12.set_opacity(0.75)
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.5)
        group_12=gx.Group(ts12, joint_line_12)

        # Third hidden neuron (NEW)
        surface_func_13=partial(surface_func_general, w1=w1[2,0], w2=w1[2,1], b=b1[2], viz_scale=viz_scale_1)
        bent_surface_13 = parametric_surface(surface_func_13, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts13=textured_surface(bent_surface_13, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts13, 0,0,0)
        ts13.set_opacity(0.75)
        joint_points_13 = get_relu_joint(w1[2,0], w1[2,1], b1[2], extent=1)
        joint_line_13=line_from_joint_points_1(joint_points_13).set_opacity(0.5)
        group_13=gx.Group(ts13, joint_line_13)

        surface_func_14=partial(surface_func_general, w1=w1[3,0], w2=w1[3,1], b=b1[3], viz_scale=viz_scale_1)
        bent_surface_14 = parametric_surface(surface_func_14, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts14=textured_surface(bent_surface_14, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts14, 0,0,0)
        ts14.set_opacity(0.75)
        joint_points_14 = get_relu_joint(w1[3,0], w1[3,1], b1[3], extent=1)
        joint_line_14=line_from_joint_points_1(joint_points_14).set_opacity(0.5)
        group_14=gx.Group(ts14, joint_line_14)

        self.set_view(0, 0, 0, (0.03, -0.02, 0.0), 3.27)
        self.add(group_11, group_12, group_13, group_14)
        
        # Position the three hidden layer visualizations
        group_14.shift([-3, 0, 3])    # Top
        group_13.shift([-3, 0, 2])    # Top
        group_12.shift([-3, 0, 1])  # Middle
        group_11.shift([-3, 0, 0])    # Bottom
        self.wait()

        # First output neuron
        neuron_idx=0
        surface_func_21 = partial(
            surface_func_second_layer_no_relu_multi, 
            w1=w1, b1=b1, w2=w2, b2=b2, 
            neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )

        bent_surface_21 = parametric_surface(surface_func_21, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts21 = textured_surface(bent_surface_21, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts21, 0,0,0)
        ts21.set_opacity(0.75)

        bs21_copy=bent_surface_21.copy()
        ts21_copy=ts21.copy()

        # Get all joint points for the 3 neurons
        joint_points_list = [joint_points_11, joint_points_12, joint_points_13, joint_points_14]
        polygons = get_polygon_corners_multi(joint_points_list, extent=1)

        ts21.shift([0, 0, 1.5])
        self.add(ts21)
        
        # Create 3D polygon regions for first output neuron
        polygon_3d_objects = create_3d_polygon_regions_multi(
            polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )
        polygon_3d_objects_copy = create_3d_polygon_regions_multi(
            polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )
        
        for poly in polygon_3d_objects:
            poly.set_opacity(0.3)
            poly.shift([0, 0, 1.5])
            self.add(poly)
        self.wait()

        # Second output neuron
        neuron_idx=1
        surface_func_22 = partial(
            surface_func_second_layer_no_relu_multi, 
            w1=w1, b1=b1, w2=w2, b2=b2, 
            neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )

        bent_surface_22 = parametric_surface(surface_func_22, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts22 = textured_surface(bent_surface_22, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts22, 0,0,0)
        ts22.set_opacity(0.75)

        bs22_copy=bent_surface_22.copy()
        ts22_copy=ts22.copy()

        self.add(ts22)

        # colors = [RED, BLUE, GREEN, YELLOW, PURPLE, RED_A, BLUE_B, GREEN_B, PURPLE_B]
        # polygon_mobjects = []

        # for i, polygon_points in enumerate(polygons):
        #     if len(polygon_points) >= 3:
        #         # Convert 2D points to 3D for Manim
        #         points_3d = [[p[0], p[1], 0] for p in polygon_points]
                
        #         # Create the polygon
        #         poly = Polygon(*points_3d, 
        #                       fill_color=colors[i % len(colors)], 
        #                       fill_opacity=0.4,
        #                       stroke_color=colors[i % len(colors)],
        #                       stroke_width=2)
                
        #         polygon_mobjects.append(poly)
        #         self.add(poly)

        # self.wait()


        # Create 3D polygon regions for second output neuron
        polygon_3d_objects_2 = create_3d_polygon_regions_multi(
            polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )
        polygon_3d_objects_2_copy = create_3d_polygon_regions_multi(
            polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )
        
        for poly in polygon_3d_objects_2:
            poly.set_opacity(0.3)
            self.add(poly)
        self.wait()




        # Final visualization positioning
        bs21_copy.shift([3, 0, 0.75])
        ts21_copy.shift([3, 0, 0.75])
        bs22_copy.shift([3, 0, 0.75])
        ts22_copy.shift([3, 0, 0.75])

        map_img.move_to([3, 0, 0.5])
        self.add(map_img)

        # Add the copy polygons for visualization
        for poly in polygon_3d_objects_copy:
            poly.set_opacity(0.3).set_color(BLUE)
            poly.shift([3, 0, 0.75])
            self.add(poly)

        for poly in polygon_3d_objects_2_copy:
            poly.set_opacity(0.3).set_color(YELLOW)
            poly.shift([3, 0, 0.75])
            self.add(poly)

        self.set_view(11, 62, 0, (3.2, 0.86, 0.02), 4.39)

        self.wait()

        ## Ok this is pretty freaking dope. 
        ## What I want now (and would probably be nice for the 2 neuron case)
        ## is to draw in the borders/intserctions between the two shells
        ## 

        decision_boundaries=create_decision_boundary_lines(w1, b1, w2, b2, polygons, extent=1, z_offset=0, color=gx.WHITE, stroke_width=4)
        decision_boundaries[0].shift([3, 0, 0.75])
        self.add(decision_boundaries[0])
        self.wait()

        # ok decision boundaries seem slightly too high in the 4 neuron case - not a huge problem right now - but will need to figure 
        # out


        self.wait(20)
        pass






        # colors = [RED, BLUE, GREEN, YELLOW, PURPLE]
        # polygon_mobjects = []

        # for i, polygon_points in enumerate(polygons):
        #     if len(polygon_points) >= 3:
        #         # Convert 2D points to 3D for Manim
        #         points_3d = [[p[0], p[1], 0] for p in polygon_points]
                
        #         # Create the polygon
        #         poly = Polygon(*points_3d, 
        #                       fill_color=colors[i % len(colors)], 
        #                       fill_opacity=0.4,
        #                       stroke_color=colors[i % len(colors)],
        #                       stroke_width=2)
                
        #         polygon_mobjects.append(poly)
        #         self.add(poly)

        # self.wait()

        # self.add(polygon_mobjects[0])
        # self.add(polygon_mobjects[1])
        # self.add(polygon_mobjects[2])
        # self.add(polygon_mobjects[3])
        # self.add(polygon_mobjects[4])


