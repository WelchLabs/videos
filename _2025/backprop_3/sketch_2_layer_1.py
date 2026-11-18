from welch_paths import welch_path
from welch_surface import set_surface_shading
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
import manimgx as gx
import numpy as np
from plane_folding_utils import (
    create_3d_polygon_regions_multi,
    create_3d_polygon_regions_with_relu,
    get_polygon_corners_multi,
    get_relu_joint,
    line_from_joint_points_1,
    surface_func_general,
    surface_func_second_layer_multi,
    surface_func_second_layer_no_relu_multi,
    surface_func_third_layer_no_relu_multi,
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

class sketch_2_layer_1(WelchScene):
    def construct(self):

        map_img=image_at_height(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        map_img.scale_to_fit_height(2)  # This sets the height to 2 units (-1 to 1)
        map_img.scale_to_fit_width(2)   # This sets the width to 2 units (-1 to 1)
        map_img.move_to(gx.ORIGIN)

        viz_scale_1=0.25
        viz_scale_2=0.25

        w1 = np.array([[0.517837, 1.75789],
         [-1.27265, -0.087746]], dtype=np.float32)
        b1 = np.array([-0.127576, 1.05139], dtype=np.float32)
        w2 = np.array([[1.20559, 0.960618],
         [-0.819331, -1.2194]], dtype=np.float32)
        b2 = np.array([-1.40305, 1.17917], dtype=np.float32)
        w3 = np.array([[3.48773, 3.53277],
         [-3.32003, -2.6953]], dtype=np.float32)
        b3 = np.array([-1.18046, 0.768749], dtype=np.float32)

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


        self.set_view(0, 0, 0, (0.03, -0.02, 0.0), 3.27)
        self.add(group_11, group_12)
        
        # Position the three hidden layer visualizations
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
        joint_points_list = [joint_points_11, joint_points_12]
        polygons = get_polygon_corners_multi(joint_points_list, extent=1)

        ts21.shift([0, 0, 1.5])
        self.add(ts21)
        
        # Create 3D polygon regions for first output neuron
        polygon_3d_objects = create_3d_polygon_regions_multi(
            polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )
        # polygon_3d_objects_copy = create_3d_polygon_regions_multi(
        #     polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
        # )
        
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
        polygon_3d_objects_2 = create_3d_polygon_regions_multi(
            polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )
        # polygon_3d_objects_2_copy = create_3d_polygon_regions_multi(
        #     polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
        # )
        
        for poly in polygon_3d_objects_2:
            poly.set_opacity(0.3)
            self.add(poly)
        self.wait()

        # Ok right so this is going to be the importand first deviation i think - we 
        # now need to do the Relu version here! And if it helps, maybe add the z=0 plane!

        ## --- Second layer, Post Relu --- ##
        neuron_idx=0
        surface_func_21r = partial(
            surface_func_second_layer_multi, 
            w1=w1, b1=b1, w2=w2, b2=b2, 
            neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )

        bent_surface_21r = parametric_surface(surface_func_21r, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts21r = textured_surface(bent_surface_21r, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts21r, 0,0,0)
        ts21r.set_opacity(0.75)

        bs21r_copy=bent_surface_21r.copy()
        ts21r_copy=ts21r.copy()

        # Get all joint points for the 3 neurons
        joint_points_list = [joint_points_11, joint_points_12]
        polygons = get_polygon_corners_multi(joint_points_list, extent=1)

        ts21r.shift([3, 0, 1.5])
        self.add(ts21r)
        
        # Create 3D polygon regions for first output neuron
        polygon_3d_objects = create_3d_polygon_regions_with_relu(
            polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )
        # polygon_3d_objects_copy = create_3d_polygon_regions_with_relu(
        #     polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
        # )
        
        for poly in polygon_3d_objects:
            poly.set_opacity(0.3)
            poly.shift([3.0, 0, 1.5])
            self.add(poly)
       
        plane = gx.Rectangle(
            width=2,  # -1 to +1 = 2 units wide
            height=2, # -1 to +1 = 2 units tall
            fill_color=gx.GREY,
            fill_opacity=0.3,
            stroke_color=gx.WHITE,
            stroke_width=1
        )
        plane.move_to([0, 0, 1.5])  # Position at z=0
        self.add(plane)
        plane_2=plane.copy()
        plane_2.move_to([0, 0, 0])  # Position at z=0
        self.add(plane_2)

        # Ok first non-trivial technical problem here - definitely the right thing to work on -
        # I need to recompute 2d polygons post relu intersection
        # Ok cool pretty sure I already solved this in plane_folding_sketch_3.py 
        # Let me see if solution is clean enough to pull in here 
        # Ok yeah i actually already had the solution - dope. 
        # I suspect it just doesn't work for the >2 neuron case -> on problem at a time though. 

        neuron_idx=1
        surface_func_22r = partial(
            surface_func_second_layer_multi, 
            w1=w1, b1=b1, w2=w2, b2=b2, 
            neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )

        bent_surface_22r = parametric_surface(surface_func_22r, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts22r = textured_surface(bent_surface_22r, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts22r, 0,0,0)
        ts22r.set_opacity(0.75)

        bs22r_copy=bent_surface_22r.copy()
        ts22r_copy=ts22r.copy()

        ts22r.shift([3, 0, 0])
        self.add(ts22r)

        polygon_3d_objects_2 = create_3d_polygon_regions_with_relu(
            polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )
        # polygon_3d_objects_2_copy = create_3d_polygon_regions_with_relu(
        #     polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
        # )
        
        for poly in polygon_3d_objects_2:
            poly.set_opacity(0.3)
            poly.shift([3.0, 0, 0])
            self.add(poly)
        self.wait()

        # Ok dope -> now how do we get to final decision boundary here?
        # Hmm we need to another scaled/adding thing right? 
        # And then the output stuff?
        # First output neuron
        neuron_idx=0
        surface_func_31 = partial(
            surface_func_third_layer_no_relu_multi, 
            w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3,
            neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )

        bent_surface_31 = parametric_surface(surface_func_31, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts31 = textured_surface(bent_surface_31, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts31, 0,0,0)
        ts31.set_opacity(0.75)

        bs31_copy=bent_surface_31.copy()
        ts31_copy=ts31.copy()


        ts31.shift([6.0, 0, 1.5])
        self.add(ts31)


        #Ok, how do we get the cool polygon borders now?




        self.set_view(5, 63, 0, (1.48, -0.01, 0.53), 8.28)


        self.waiffdt()



        self.wait()
        pass











