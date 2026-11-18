from welch_paths import welch_path
from welch_surface import set_surface_shading
from functools import partial
import sys
import pickle

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

class goemetry_while_learning_seed_90(WelchScene):
    def construct(self):

        map_img=image_at_height(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        map_img.scale_to_fit_height(2)  # This sets the height to 2 units (-1 to 1)
        map_img.scale_to_fit_width(2)   # This sets the width to 2 units (-1 to 1)
        map_img.move_to(gx.ORIGIN)

        viz_scale_1=0.25
        viz_scale_2=0.1

        # pickle_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/hackin/3_training_weights_1/training_data_seed_13_acc_0.6098.pkl'
        # pickle_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/hackin/3_starting_configs/training_data_seed_09_acc_0.8091.pkl'
        # pickle_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/hackin/3_starting_configs/training_data_seed_02_acc_0.6045.pkl'
        # pickle_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/hackin/3_starting_configs/training_data_seed_20_acc_0.6083.pkl'
        # pickle_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/hackin/3_starting_configs/training_data_seed_21_acc_0.8108.pkl'
        # pickle_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/hackin/3_starting_configs/training_data_seed_27_acc_0.6364.pkl'
        # pickle_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/hackin/3_starting_configs/training_data_seed_14_acc_0.6400.pkl'

        #Worst
        # pickle_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/hackin/3_starting_configs_3/training_data_seed_10_acc_0.4999.pkl'
        pickle_path=welch_path('backprop_3/hackin/3_starting_configs_4/training_data_seed_90_acc_0.6159.pkl')

        

        #Best
        # pickle_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/hackin/3_starting_configs_3/training_data_seed_76_acc_0.8510.pkl'
        # pickle_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/hackin/3_starting_configs_4/training_data_seed_01_acc_0.8561.pkl'


        

        with open(pickle_path, 'rb') as f:
            p=pickle.load(f)

        self.set_view(1, 58, 0, (-0.17, 2.27, -0.1), 8.46)

        step_size=10
        num_steps=int(np.floor(len(p['weights_history'])/step_size))
        print('\n', 'num_steps=', num_steps)
        for i in range(num_steps):

            train_step=step_size*i
            w1=p['weights_history'][train_step]['model.0.weight'].numpy()
            b1=p['weights_history'][train_step]['model.0.bias'].numpy()
            w2=p['weights_history'][train_step]['model.2.weight'].numpy()
            b2=p['weights_history'][train_step]['model.2.bias'].numpy()

            # Helper function to safely create joint lines
            def create_joint_line_safely(joint_points):
                """Create a joint line if joint_points exist, otherwise return None"""
                if joint_points and len(joint_points) >= 2:
                    return line_from_joint_points_1(joint_points).set_opacity(0.5)
                else:
                    return None

            # First hidden neuron
            surface_func_11=partial(surface_func_general, w1=w1[0,0], w2=w1[0,1], b=b1[0], viz_scale=viz_scale_1)
            bent_surface_11 = parametric_surface(surface_func_11, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
            ts11=textured_surface(bent_surface_11, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
            set_surface_shading(ts11, 0,0,0)
            ts11.set_opacity(0.75)
            joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
            joint_line_11 = create_joint_line_safely(joint_points_11)
            
            # Create group, adding joint line only if it exists
            group_11_objects = [ts11]
            if joint_line_11 is not None:
                group_11_objects.append(joint_line_11)
            group_11 = gx.Group(*group_11_objects)

            # Second hidden neuron
            surface_func_12=partial(surface_func_general, w1=w1[1,0], w2=w1[1,1], b=b1[1], viz_scale=viz_scale_1)
            bent_surface_12 = parametric_surface(surface_func_12, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
            ts12=textured_surface(bent_surface_12, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
            set_surface_shading(ts12, 0,0,0)
            ts12.set_opacity(0.75)
            joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
            joint_line_12 = create_joint_line_safely(joint_points_12)
            
            # Create group, adding joint line only if it exists
            group_12_objects = [ts12]
            if joint_line_12 is not None:
                group_12_objects.append(joint_line_12)
            group_12 = gx.Group(*group_12_objects)

            # Third hidden neuron (NEW)
            surface_func_13=partial(surface_func_general, w1=w1[2,0], w2=w1[2,1], b=b1[2], viz_scale=viz_scale_1)
            bent_surface_13 = parametric_surface(surface_func_13, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
            ts13=textured_surface(bent_surface_13, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
            set_surface_shading(ts13, 0,0,0)
            ts13.set_opacity(0.75)
            joint_points_13 = get_relu_joint(w1[2,0], w1[2,1], b1[2], extent=1)
            joint_line_13 = create_joint_line_safely(joint_points_13)
            
            # Create group, adding joint line only if it exists
            group_13_objects = [ts13]
            if joint_line_13 is not None:
                group_13_objects.append(joint_line_13)
            group_13 = gx.Group(*group_13_objects)

            # Position the three hidden layer visualizations
            group_13.shift([-3, 0, 3])    # Top
            group_12.shift([-3, 0, 1.5])  # Middle
            group_11.shift([-3, 0, 0])    # Bottom

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

            # Get all joint points for the neurons that have them
            joint_points_list = []
            if joint_points_11:
                joint_points_list.append(joint_points_11)
            if joint_points_12:
                joint_points_list.append(joint_points_12)  
            if joint_points_13:
                joint_points_list.append(joint_points_13)

            # Only create polygons if we have at least one joint line
            if len(joint_points_list) > 0:
                polygons = get_polygon_corners_multi(joint_points_list, extent=1)
                
                # Create 3D polygon regions for first output neuron
                polygon_3d_objects = create_3d_polygon_regions_multi(
                    polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
                )
                polygon_3d_objects_copy = create_3d_polygon_regions_multi(
                    polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
                )
            else:
                # No joint lines, so no polygon regions to create
                polygon_3d_objects = []
                polygon_3d_objects_copy = []

            ts21.shift([0, 0, 1.5])

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

            # Create 3D polygon regions for second output neuron (only if we have polygons)
            if len(joint_points_list) > 0:
                polygon_3d_objects_2 = create_3d_polygon_regions_multi(
                    polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
                )
                polygon_3d_objects_2_copy = create_3d_polygon_regions_multi(
                    polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
                )
            else:
                polygon_3d_objects_2 = []
                polygon_3d_objects_2_copy = []

            # Final visualization positioning
            bs21_copy.move_to([3, 0, 0.75])
            ts21_copy.move_to([3, 0, 0.75])
            bs22_copy.move_to([3, 0, 0.75])
            ts22_copy.move_to([3, 0, 0.75])

            map_img.move_to([3, 0, 0.5])
            
            # Create decision boundaries only if we have polygons
            if len(joint_points_list) > 0:
                decision_boundaries = create_decision_boundary_lines(w1, b1, w2, b2, polygons, extent=1, z_offset=0, color=gx.WHITE, stroke_width=4)
            else:
                decision_boundaries = []

            # Add objects to scene
            self.add(group_11, group_12, group_13)
            self.add(ts21)

            # Add polygon objects only if they exist
            for poly in polygon_3d_objects:
                poly.set_opacity(0.3)
                poly.shift([0, 0, 1.5])
                self.add(poly)

            self.add(ts22)

            for poly in polygon_3d_objects_2:
                poly.set_opacity(0.3)
                self.add(poly)

            self.add(map_img)

            for poly in polygon_3d_objects_copy:
                poly.set_opacity(0.3).set_color(BLUE)
                poly.shift([3, 0, 0.75])
                self.add(poly)

            for poly in polygon_3d_objects_2_copy:
                poly.set_opacity(0.3).set_color(YELLOW)
                poly.shift([3, 0, 0.75])
                self.add(poly)
            
            # Add decision boundaries only if they exist
            if len(decision_boundaries) > 0:
                decision_boundaries[0].shift([3, 0, 0.75])
                self.add(decision_boundaries[0])

            self.wait(0.1)
            
            # Remove objects from scene
            self.remove(group_11, group_12, group_13, ts21, ts22, map_img)
            
            if len(decision_boundaries) > 0:
                self.remove(decision_boundaries[0])
                
            for poly in polygon_3d_objects+polygon_3d_objects_2+polygon_3d_objects_copy+polygon_3d_objects_2_copy: 
                self.remove(poly)

        pass