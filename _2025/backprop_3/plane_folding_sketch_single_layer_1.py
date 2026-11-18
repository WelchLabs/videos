from welch_paths import welch_path
from welch_surface import set_surface_shading
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
import manimgx as gx
import numpy as np
from plane_folding_utils import (
    create_3d_polygon_regions,
    get_polygon_corners,
    get_relu_joint,
    line_from_joint_points_1,
    surface_func_general,
    surface_func_second_layer_no_relu,
)
from welch_scene import WelchScene, image_at_height
from welch_surface import parametric_surface, textured_surface

graphics_dir=welch_path('backprop_3/graphics/') #Point to folder where map images are

class plane_folding_sketch_single_layer_1(WelchScene):
    def construct(self):

        map_img=image_at_height(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        map_img.scale_to_fit_height(2)  # This sets the height to 2 units (-1 to 1)
        map_img.scale_to_fit_width(2)   # This sets the width to 2 units (-1 to 1)
        map_img.move_to(gx.ORIGIN)

        #Fold up 2 edges
        # w1=np.array([[-0.70856154,  1.809896  ],
        #              [-1.7940422 , -0.4643133 ]], dtype=np.float32)
        # b1=np.array([-0.47732198, -1.0138882 ], dtype=np.float32)
        # w2=np.array([[ 1.5246898,  2.049856 ],
        #             [-1.6014509, -1.3020881]], dtype=np.float32)
        # b2=np.array([-0.40461758,  0.05192775], dtype=np.float32)

        #The cone
        w1 = np.array([[2.5135, -1.02481],
         [-1.4043, 2.41291]], dtype=np.float32)
        b1 = np.array([-1.23981, -0.450078], dtype=np.float32)
        w2 = np.array([[3.17024, 1.32567],
         [-3.40372, -1.53878]], dtype=np.float32)
        b2 = np.array([-0.884835, 0.0332228], dtype=np.float32)

        surface_func_11=partial(surface_func_general, w1=w1[0,0], w2=w1[0,1], b=b1[0], viz_scale=0.42) #Larger for intput layer
        bent_surface_11 = parametric_surface(surface_func_11, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts11=textured_surface(bent_surface_11, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts11, 0,0,0)
        ts11.set_opacity(0.75)
        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.5)
        group_11=gx.Group(ts11, joint_line_11)

        surface_func_12=partial(surface_func_general, w1=w1[1,0], w2=w1[1,1], b=b1[1], viz_scale=0.42) #Larger for intput layer
        bent_surface_12 = parametric_surface(surface_func_12, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts12=textured_surface(bent_surface_12, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts12, 0,0,0)
        ts12.set_opacity(0.75)
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.5)
        group_12=gx.Group(ts12, joint_line_12)

        self.set_view(0, 0, 0, (0.03, -0.02, 0.0), 3.27)
        self.add(group_11, group_12)
        group_12.shift([-3, 0, 1.5])
        group_11.shift([-3, 0, 0])
        self.wait()

        neuron_idx=0
        surface_func_21 = partial(
            surface_func_second_layer_no_relu, 
            w1=w1, b1=b1, w2=w2, b2=b2, 
            neuron_idx=neuron_idx, viz_scale=0.25
        )

        bent_surface_21 = parametric_surface(surface_func_21, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts21 = textured_surface(bent_surface_21, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts21, 0,0,0)
        ts21.set_opacity(0.75)

        bs21_copy=bent_surface_21.copy()
        ts21_copy=ts21.copy()

        polygons = get_polygon_corners(joint_points_11, joint_points_12, extent=1)

        ts21.shift([0, 0, 1.5])
        self.add(ts21)
        polygon_3d_objects = create_3d_polygon_regions( polygons, w1, b1, w2, b2,  neuron_idx=neuron_idx, viz_scale=0.25)
        polygon_3d_objects_copy=create_3d_polygon_regions( polygons, w1, b1, w2, b2,  neuron_idx=neuron_idx, viz_scale=0.25)
        for poly in polygon_3d_objects:
            poly.set_opacity(0.3)
            poly.shift([0, 0, 1.5])
            self.add(poly)
        self.wait()


        neuron_idx=1
        surface_func_22 = partial(
            surface_func_second_layer_no_relu, 
            w1=w1, b1=b1, w2=w2, b2=b2, 
            neuron_idx=neuron_idx, viz_scale=0.25
        )

        bent_surface_22 = parametric_surface(surface_func_22, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts22 = textured_surface(bent_surface_22, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts22, 0,0,0)
        ts22.set_opacity(0.75)

        bs22_copy=bent_surface_22.copy()
        ts22_copy=ts22.copy()

        self.add(ts22)
        polygon_3d_objects_2 = create_3d_polygon_regions( polygons, w1, b1, w2, b2,  neuron_idx=neuron_idx, viz_scale=0.25)
        polygon_3d_objects_2_copy=create_3d_polygon_regions( polygons, w1, b1, w2, b2,  neuron_idx=neuron_idx, viz_scale=0.25)
        for poly in polygon_3d_objects_2:
            poly.set_opacity(0.3)
            self.add(poly)
        self.wait()

        # Ok this all makes sense and I thikk can be be visualized nicely - now how do we get from here to final heatmap?
        # it's kinda like Relu again I think right? Hmm -> Well I mean I guess where are the logits -> I'm looking 
        # at the logits...
        # Ok ok ok ok I guess I way to see what's going on would be to just color these whole planes in different colors 
        # Bring them to the same z level, and then whoever is on top wins? 
        # And then for extra credit we do softmax? Ok yeah I think that's potentially pretty dope -> let's see here...
        # I could recolor individual polygons -> but I think it's going to maybe animate better if I have
        # the full contiguous bent plane too? 
        # Let me poke at that. 

        bs21_copy.move_to([3, 0, 0.75])
        ts21_copy.move_to([3, 0, 0.75])
        bs22_copy.move_to([3, 0, 0.75])
        ts22_copy.move_to([3, 0, 0.75])

        map_img.move_to([3, 0, 0.5]) #Don't quite put the map at 0. 
        self.add(map_img)

        for poly in polygon_3d_objects_copy:
            poly.set_opacity(0.3).set_color(gx.BLUE)
            poly.shift([3, 0, 0.75])
            self.add(poly)

        for poly in polygon_3d_objects_2_copy:
            poly.set_opacity(0.3).set_color(gx.YELLOW)
            poly.shift([3, 0, 0.75])
            self.add(poly)

        self.set_view(11, 62, 0, (3.2, 0.86, 0.02), 4.39)

        self.wait()

        # self.add(ts21_copy)
        # bs21_copy.set_color(BLUE).set_opacity(0.5)
        # self.add(bs21_copy)
        
        # polygon_3d_objects

        # self.wait()

        # Ok that's pretty dope!
        # Alright I'm tempted to work on softmax, but I think better to go back to writing
        # Softmax is technical and there shouldn't be any surprises there!




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




        self.wait(20)
        pass
