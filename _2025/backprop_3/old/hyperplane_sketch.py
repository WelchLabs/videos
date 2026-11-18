from welch_paths import welch_path
import manimgx as m
from welch_surface import set_surface_shading
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
import copy
import manimgx as gx
import numpy as np
import torch
from geometric_dl_utils import (
    BaarleNet,
    apply_viz_scale_to_3d_polygons,
    carve_plane_with_relu_joints,
    compute_adaptive_viz_scales,
    find_polygon_intersections,
    get_3d_polygons,
    get_3d_polygons_layer_1,
    get_polygon_corners_layer_1,
    get_relu_intersection_planes,
    split_polygons_with_relu,
    surface_func_from_model,
    viz_3d_polygons,
    viz_carved_regions_flat,
)
from welch_geometry import diagram_axes_3d
from welch_scene import WelchScene
from welch_surface import parametric_surface, textured_surface
from polytope_intersection_utils import intersect_polytopes

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'


graphics_dir=welch_path('backprop_3/graphics/') #Point to folder where map images are
heatmaps_dir=welch_path('backprop_3/hackin/heatmaps')

class refactor_sketch_1(WelchScene):
    def construct(self):
        
        #2x2
        # model_path='_2025/backprop_3/models/2_2_1.pth'
        # model = BaarleNet([2,2])
        # model.load_state_dict(torch.load(model_path))
        # viz_scales=[0.25, 0.25, 0.3, 0.3, 0.15]
        # num_neurons=[2, 2, 2, 2, 2]

        #3x3
        model_path='_2025/backprop_3/models/3_3_1.pth'
        model = BaarleNet([3,3])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.1, 0.1, 0.05, 0.05, 0.15]
        num_neurons=[3, 3, 3, 3, 2]

        #4x4


        vertical_spacing=1.5
        horizontal_spacing=3
        colors = [BLUE, gx.RED, GREEN, YELLOW, gx.PURPLE, gx.ORANGE, gx.PINK, gx.TEAL]

        adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=1.0, extent=1)
        final_layer_viz=scale=2*min(adaptive_viz_scales[-1]) #little manual ramp here
        adaptive_viz_scales[-1]=[final_layer_viz, final_layer_viz]


        surfaces=[]
        surface_funcs=[]
        surface_funcs_no_viz_scale=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
            surface_funcs_no_viz_scale.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx])
                surface_func_no_scaling=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=1.0) #adaptive_viz_scales[layer_idx][neuron_idx])
                bent_surface = parametric_surface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
                ts=textured_surface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
                set_surface_shading(ts, 0,0,0).set_opacity(0.75)
                s.add(ts)
                surface_funcs[-1].append(surface_func)
                surface_funcs_no_viz_scale[-1].append(surface_func_no_scaling)
            surfaces.append(s)

        for layer_idx, sl in enumerate(surfaces):
            for neuron_idx, s in enumerate(sl):
                s.shift([horizontal_spacing*layer_idx-6, 0, vertical_spacing*neuron_idx])
                self.add(s)


        # self.set_view(0, 54, 0, (1.41, 1.82, 4.15), 15.71)
        self.set_view(0, 56, 0, (-0.06, -0.01, 1.23), 9.57)


        #Optional but kinda nice RuLu intersection planes
        layer_idx=0
        relu_intersections_planes_1=get_relu_intersection_planes(num_neurons[layer_idx], layer_idx, neuron_idx, horizontal_spacing, vertical_spacing)
        self.add(relu_intersections_planes_1)


        #Layer 1 polygons - use surface function to find heights
        layer_idx=1
        layer_1_polygons=get_polygon_corners_layer_1(model)
        layer_1_polygons_3d=get_3d_polygons_layer_1(layer_1_polygons, surface_funcs_no_viz_scale, num_neurons=num_neurons[layer_idx], layer_idx=1)
        

        scaled_layer_1_polygons_3d=apply_viz_scale_to_3d_polygons(layer_1_polygons_3d, adaptive_viz_scales[layer_idx])
        polygons_vgroup=viz_3d_polygons(scaled_layer_1_polygons_3d, layer_idx, colors=None, color_gray_index=1)
        self.add(polygons_vgroup)


        #2d shadow of these regions
        layer_2_polygons=carve_plane_with_relu_joints([o['relu_line'] for o in layer_1_polygons])
        output_poygons_2d=viz_carved_regions_flat(layer_2_polygons, horizontal_spacing, layer_idx, colors=None)
        self.add(output_poygons_2d)



        #Layer 2 linear
        layer_idx=2
        layer_2_polygons_3d=get_3d_polygons(layer_2_polygons, num_neurons[layer_idx], surface_funcs_no_viz_scale, layer_idx)
        scaled_layer_2_polygons_3d=apply_viz_scale_to_3d_polygons(layer_2_polygons_3d, adaptive_viz_scales[layer_idx])
        polygons_vgroup_2=viz_3d_polygons(scaled_layer_2_polygons_3d, layer_idx, colors=None, color_gray_index=None)
        self.add(polygons_vgroup_2)
        relu_intersections_planes_2=get_relu_intersection_planes(num_neurons[layer_idx], layer_idx, neuron_idx, horizontal_spacing, vertical_spacing)
        self.add(relu_intersections_planes_2)


        #Layer 2 Relu
        layer_idx=3
        all_polygons, merged_zero_polygons, unmerged_polygons = split_polygons_with_relu(layer_2_polygons_3d)

        all_polygons_after_merging=copy.deepcopy(merged_zero_polygons)
        for i, o in enumerate(unmerged_polygons):
            all_polygons_after_merging[i].extend(o)


        all_polygons_after_merging_scaled=apply_viz_scale_to_3d_polygons(all_polygons_after_merging, adaptive_viz_scales[layer_idx])
        layer_2_polygons_split_vgroup=viz_3d_polygons(all_polygons_after_merging_scaled, layer_idx, colors=None)
        self.add(layer_2_polygons_split_vgroup)


        #2D Projection of Layer 2 After Relu Cuts
        all_polygons_after_merging_2d=[]
        for p in all_polygons_after_merging:
            pd2=[o[:,:2] for o in p]
            all_polygons_after_merging_2d.append(pd2)

        layer3_regions_2d = find_polygon_intersections(all_polygons_after_merging_2d)
        output_poygons_2d_2=viz_carved_regions_flat(layer3_regions_2d, horizontal_spacing, layer_idx, colors=None)
        self.add(output_poygons_2d_2)


        # Ok cool cool. So there's a bunch of places I could jump in and try to understand the hyperplane angle 
        # But I think going from output_poygons_2d to output_poygons_2d_2 is potentially a really nice place to do 
        # this - can it help me understand which polygons get cut and why? 
        # I guess in the 2x2 case they call kinda get cut -> well not all of them actually. 
        # Ok let's get a 3d axis going and move a copy of all the output_poygons_2d to this activation space. 


        # Ok so here's my list of list of numpy arrays of the 4 polygons in 3d space, so the third coordinates
        # from the corresponding enteries in each first list will give me the polygons coordinates in activation
        # space 
        # polygons_to_viz=np.array(layer_2_polygons_3d) #Eh this won't work when polygon num vertices are different
        # np.max(polygons_to_viz[:,:,:,2])
        # np.min(polygons_to_viz[:,:,:,2])


        # axes_1 = ThreeDAxes(
        #     x_range=[-4, 4, 1],
        #     y_range=[-4, 4, 1],
        #     z_range=[-4, 4, 1],
        #     height=3,
        #     width=3,
        #     depth=3,
        #     axis_config={
        #         "include_ticks": True,
        #         "color": CHILL_BROWN,
        #         "stroke_width": 2,
        #         "include_tip": True,
        #         "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
        #     }
        # )

        # axes_1.move_to([0, 0, 4])
        # self.add(axes_1)

        # polygons_activation_space_1=VGroup()
        # for polygon_index in range(polygons_to_viz.shape[1]):
        #     # polygon_index=2
        #     color = colors[polygon_index%len(colors)]
        #     #Ok kinda tricky part, extracting the right coordinates and then using c2p
        #     pts=[]
        #     for pt_idx in range(polygons_to_viz.shape[2]):
        #         #For the 2d case:
        #         pts.append(axes_1.c2p(polygons_to_viz[0,polygon_index, pt_idx, 2], polygons_to_viz[1, polygon_index, pt_idx, 2], 0))
        #     # print(pts)
        #     poly_3d=Polygon(*pts, fill_color=color, fill_opacity=0.7, stroke_color=color, stroke_width=2)
        #     poly_3d.set_opacity(0.9)
        #     polygons_activation_space_1.add(poly_3d)

        # self.add(polygons_activation_space_1)

        hyperspace_polygons=[]
        for poly_2d in layer_2_polygons: # Eh dis?
            hyperspace_polygon=[]
            for pt in poly_2d:
                pt_copy=copy.deepcopy(list(pt))
                for j, surf_func in enumerate(surface_funcs_no_viz_scale[2]):
                    pt_copy.append(surf_func(*pt)[-1])
                hyperspace_polygon.append(pt_copy)
            hyperspace_polygons.append(np.array(hyperspace_polygon))


        axes_1 = diagram_axes_3d(
            x_range=[-8, 8, 1],
            y_range=[-8, 8, 1],
            z_range=[-8, 8, 1],
            height=3,
            width=3,
            depth=3,
            axis_config={
                "include_ticks": True,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip": True,
                "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
            }
        )

        axes_1.move_to([0*horizontal_spacing, 0, 6])
        self.add(axes_1)

        polygons_activation_space_1=gx.VGroup()
        for j, polygon in enumerate(hyperspace_polygons):
            color = colors[j%len(colors)]
            pts=[]
            for pt in polygon:
                #For the 3d case:
                pts.append(axes_1.c2p(pt[2], pt[3], pt[4]))
            poly_3d=gx.Polygon(*pts, fill_color=color, fill_opacity=0.7, stroke_color=color, stroke_width=2)
            poly_3d.set_opacity(0.3)
            polygons_activation_space_1.add(poly_3d)

        self.add(polygons_activation_space_1)

        # self.embed()


        # Ok hmm not sure what to make of this just yet
        # So in the 2x2 case, the 4th gree region (where both nuerons are off in layer 1) 
        # Collapses to a single point! And this make sense becuase the whole plane 
        # has a single value/height in each neuron! Ah maybe that's why it doesn't get split - interesting
        # Soo...seems like the off of region just becomes a point  
        #Ok and the region where both planes areon becomes a plane
        # Let me think for a minute sabout hwye the regions where only one plane are on become lines...
        # Hmm it's not immediately obvoius to me why it's a line, but given that the green region collapeses to just a point
        # This seems ok - I can come back to it if it feels important. 
        # Now, what's the Relu version??
        # all_polygons
        # Ok interesting yeah this doesn't quite work b/c after these get split, our old polygons
        # Have different numbers of vertices in each dimesion
        # There must be some connection to the picture still though, right? 
        # Hmmmm. 
        # Ok this was not as clean as I thought, buuuuuut I think there's actually a non-insane solution
        # that may be generally kinda useful. 
        # I think what I can do here is take every new 2d polygon, and 
        # just send int N dimensional space using that layer's surface functions, ya know? 
        # If this perspective pans out at all, I think that could be pretty useful. 
        # It might be kinda interesting too that it's like the same tiling from 
        # 3 different persepctives now I think? The activation space it was cut in, 
        # the map space, and then projected to the new space (?) 
        # Let me take a little hack at it here. 

        # def move_to_hyperspace lol - shouldn't i already have some version of this? Or am I coming at it differently?

        hyperspace_polygons=[]
        for poly_2d in layer3_regions_2d:
            hyperspace_polygon=[]
            for pt in poly_2d:
                pt_copy=copy.deepcopy(list(pt))
                for j, surf_func in enumerate(surface_funcs_no_viz_scale[3]):
                    pt_copy.append(surf_func(*pt)[-1])
                hyperspace_polygon.append(pt_copy)
            hyperspace_polygons.append(np.array(hyperspace_polygon))


        axes_2 = diagram_axes_3d(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-4, 4, 1],
            height=3,
            width=3,
            depth=3,
            axis_config={
                "include_ticks": True,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip": True,
                "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
            }
        )

        axes_2.move_to([horizontal_spacing, 0, 6])
        self.add(axes_2)

        polygons_activation_space_2=gx.VGroup()
        for j, polygon in enumerate(hyperspace_polygons):
            color = colors[j%len(colors)]
            pts=[]
            for pt in polygon:
                #For the 2d case:
                pts.append(axes_2.c2p(pt[2], pt[3], pt[4]))
            poly_3d=gx.Polygon(*pts, fill_color=color, fill_opacity=0.7, stroke_color=color, stroke_width=2)
            poly_3d.set_opacity(0.9)
            polygons_activation_space_2.add(poly_3d)

        self.add(polygons_activation_space_2)


        layer_idx=4
        layer_3_polygons_3d=get_3d_polygons(layer3_regions_2d, num_neurons[layer_idx], surface_funcs_no_viz_scale, layer_idx)
        scaled_layer_3_polygons_3d=apply_viz_scale_to_3d_polygons(layer_3_polygons_3d, adaptive_viz_scales[layer_idx])
        polygons_vgroup_3=viz_3d_polygons(scaled_layer_3_polygons_3d, layer_idx, colors=None)
        self.add(polygons_vgroup_3)





        #Hyperspace agains!
        hyperspace_polygons=[]
        for poly_2d in layer3_regions_2d:
            hyperspace_polygon=[]
            for pt in poly_2d:
                pt_copy=copy.deepcopy(list(pt))
                for j, surf_func in enumerate(surface_funcs_no_viz_scale[4]):
                    pt_copy.append(surf_func(*pt)[-1])
                hyperspace_polygon.append(pt_copy)
            hyperspace_polygons.append(np.array(hyperspace_polygon))


        axes_3 = diagram_axes_3d(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-4, 4, 1],
            height=3,
            width=3,
            depth=3,
            axis_config={
                "include_ticks": True,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip": True,
                "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
            }
        )

        axes_3.move_to([2*horizontal_spacing, 0, 5])
        self.add(axes_3)

        polygons_activation_space_3=gx.VGroup()
        for j, polygon in enumerate(hyperspace_polygons):
            color = colors[j%len(colors)]
            pts=[]
            for pt in polygon:
                #For the 2d case:
                pts.append(axes_3.c2p(pt[2], pt[3], 0))
            poly_3d=gx.Polygon(*pts, fill_color=color, fill_opacity=0.7, stroke_color=color, stroke_width=2)
            poly_3d.set_opacity(0.9)
            polygons_activation_space_3.add(poly_3d)

        self.add(polygons_activation_space_3)






        self.wait()
        pass



