from welch_paths import welch_path
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
import copy
import manimgx as gx
import numpy as np
import pickle
import torch
from geometric_dl_utils import BaarleNet
from geometric_dl_utils_simplified import (
    clip_polygons,
    compute_top_polytope,
    process_with_layers,
    recompute_tiling,
    split_polygons_with_relu_simple,
)
from welch_scene import WelchScene
from welch_surface import parametric_surface, textured_surface

from polytope_intersection_utils import intersect_polytopes
# from decision_boundary_utils import *
from welch_surface import set_surface_shading
from gap_filler import fill_gaps

graphics_dir=welch_path('ai_book/4_deep_learning/graphics/')
colors = [gx.RED, gx.BLUE, gx.GREEN, gx.YELLOW, gx.PURPLE, gx.ORANGE, gx.PINK, gx.TEAL]
map_filename='baarle_hertog_maps-13.png'

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'

def order_closed_loops_with_closure(segments, tol=1e-6):
    """
    Like before, but returns each loop as an (N,3) array of vertices
    where the first point is repeated at the end to explicitly close the loop.
    """
    used = [False] * len(segments)
    loops_pts = []

    for i, seg in enumerate(segments):
        if used[i]:
            continue

        # build ordered, oriented segment list for this loop
        loop_segs = [seg.copy()]
        used[i] = True
        start_pt = seg[0].copy()
        curr_pt = seg[1].copy()

        while True:
            found = False
            for j, seg2 in enumerate(segments):
                if used[j]:
                    continue
                p0, p1 = seg2[0], seg2[1]

                if np.linalg.norm(p0 - curr_pt) < tol:
                    loop_segs.append(seg2.copy())
                    curr_pt = p1.copy()
                    used[j] = True
                    found = True
                    break

                if np.linalg.norm(p1 - curr_pt) < tol:
                    rev = seg2[::-1].copy()
                    loop_segs.append(rev)
                    curr_pt = rev[1].copy()
                    used[j] = True
                    found = True
                    break

            # stop if no continuation or we’re back at start
            if not found or np.linalg.norm(curr_pt - start_pt) < tol:
                break

        # collect vertices and explicitly close the loop
        pts = [loop_segs[0][0]]
        for s in loop_segs:
            pts.append(s[1])

        # if it didn’t naturally close, append the start_pt
        if np.linalg.norm(pts[-1] - start_pt) > tol:
            pts.append(start_pt)

        loops_pts.append(np.vstack(pts))

    return loops_pts

class p12b_flat_2(WelchScene):
    def construct(self):
    	#Start with 512, move to as big as I can for final
        model_path='_2025/backprop_3/models/32_32_32_32_1.pth'
        model = BaarleNet([32, 32, 32, 32])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.07, 0.07, 0.04]
        num_neurons=[32, 32, 32, 32, 2]

        #Move polygons through network
        polygons={} #dict of all polygones as we go. 
        polygons['-1.new_tiling_unraveled']=[np.array([[-1., -1, 0], #First polygon is just input plane
                                            [-1, 1, 0], 
                                            [1, 1, 0], 
                                            [1, -1, 0]])]

        for layer_id in range(len(model.model)//2): #Move polygont through layers     
            polygons[str(layer_id)+'.linear_out']=process_with_layers(model.model[:2*layer_id+1], polygons[str(layer_id-1)+'.new_tiling_unraveled']) 

            polygons[str(layer_id)+'.split_polygons_nested']=split_polygons_with_relu_simple(polygons[str(layer_id)+'.linear_out']) #Triple nested list so we can simplify merging process layer. 
            polygons[str(layer_id)+'.split_polygons_nested_clipped'] = clip_polygons(polygons[str(layer_id)+'.split_polygons_nested'])


            polygons[str(layer_id)+'.new_tiling']=recompute_tiling(polygons[str(layer_id)+'.split_polygons_nested'], min_area=0)
            polygons[str(layer_id)+'.new_tiling_unraveled']=[item for sublist in polygons[str(layer_id)+'.new_tiling'] for item in sublist]

            print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling_unraveled'])), ' polygons.')

            #Optional filtering step
            # polygons[str(layer_id)+'.new_tiling_unraveled'] = filter_small_polygons(polygons[str(layer_id)+'.new_tiling_unraveled'], min_area=1e-5)
            # print(str(len(polygons[str(layer_id)+'.new_tiling_unraveled'])), ' polygons remaining after filtering out small polygons')

        #Last linear layer & output
        polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling_unraveled'])
        intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
        my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)

        print('finished computing polygons and surfaces')
        
        #Not quite sure why i have a few small gaps, quick fix
        print(len(my_top_polygons), len(my_indicator))
        my_top_polygons, my_indicator=fill_gaps(my_top_polygons, my_indicator)
        print(len(my_top_polygons), len(my_indicator))

        with open('_2025/backprop_3/models/32_32_32_32_1_borders.p', 'rb') as file:
            borders_interp = pickle.load(file)

        #I guess I don't have to do this every time -> coudl just draw all the lines? Might be better when we get really big here
        # loops=order_closed_loops_with_closure(intersection_lines)

        # loops=order_closed_loops_with_closure(intersection_lines)
        # lines=VGroup()
        # for loop in loops: 
        #     # loop=loop*np.array([1, 1, viz_scales[2]])
        #     loop[:,2]=0 #Flatten
        #     line = VMobject()
        #     line.set_points_as_corners(loop)
        #     line.set_stroke(color='#FF00FF', width=5)
        #     lines.add(line)
        # lines.shift([3, 0, 0])

        lines=gx.VGroup()
        for loop in borders_interp: 
            loop=np.hstack((loop, np.zeros((len(loop), 1))))
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=5)
            lines.add(line)
        lines.shift([3, 0, 0])

        top_polygons_vgroup_flat=gx.VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=0 #p_scaled[:,2]*viz_scales[2] #Flatten that shit!
            # p_scaled[:, -1] = np.clip(p_scaled[:, -1], -polygon_max_height, polygon_max_height)
            poly_3d = gx.Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=2)
            poly_3d.set_opacity(0.3)
            poly_3d.shift([3, 0, 0])
            top_polygons_vgroup_flat.add(poly_3d)

        top_polygons_vgroup_flat.set_opacity(0.5) #Eh?

        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/'+map_filename)
        set_surface_shading(flat_map, 0,0,0).set_opacity(0.8)
        flat_map.shift([5.7, 0, 0])

        lines_flat_cleaner=gx.VGroup()
        for loop in borders_interp: 
            loop=np.hstack((loop, np.zeros((len(loop), 1))))
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=4)
            lines_flat_cleaner.add(line)
        lines_flat_cleaner.shift([5.7, 0, 0])      

        self.set_view(0, 0, 0, (4.34, 0.02, -0.19), 3.47)
        self.add(top_polygons_vgroup_flat)
        self.add(lines)

        self.add(flat_map, lines_flat_cleaner)
        self.wait()

        #Optional fade and center on just map -> although I think i actually want to do it on 1024!
        # self.play(top_polygons_vgroup_flat.animate.set_opacity(0.0),
        #           lines.animate.set_opacity(0.0),
        #           self.animate_view(0, 0, 0, (5.67, -0.01, -0.19), 3.47),
        #           run_time=5.0)


        self.wait(20)
        pass







## Ok ok ok so here i need 4 different panels. I think i render the 2d overhead map first, then take it away, then
## do some cool pan around the 3d surface. 



# class p12a(InteractiveScene):
#     def construct(self):
#     	#Start with 512, move to as big as I can for final
#         # model_path='_2025/backprop_3/models/512_1_longer.pth'
#         # model = BaarleNet([512])
#         # model.load_state_dict(torch.load(model_path))
#         # viz_scales=[0.07, 0.07, 0.04]
#         # num_neurons=[512, 512, 2]

#         model_path='_2025/backprop_3/models/one_layer_100k_neurons_long.pth'
#         model = BaarleNet([100000])
#         model.load_state_dict(torch.load(model_path))
#         viz_scales=[0.07, 0.07, 0.04]
#         num_neurons=[100000, 100000, 2]


#         #Precompute my surfaces, and polygons moving through network
#         surfaces=[]
#         surface_funcs=[]
#         for layer_idx in [2]: #Skip first layers, dont need em
#             s=Group()
#             surface_funcs.append([])
#             for neuron_idx in range(num_neurons[layer_idx]):
#                 surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=viz_scales[layer_idx])
#                 bent_surface = ParametricSurface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
#                 ts=TexturedSurface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
#                 ts.set_shading(0,0,0).set_opacity(0.8)
#                 s.add(ts)
#                 surface_funcs[-1].append(surface_func)
#             surfaces.append(s)

#         print('Finished computing surfaces...')

#         #Move polygons through network
#         polygons={} #dict of all polygones as we go. 
#         polygons['-1.new_tiling']=[np.array([[-1., -1, 0], #First polygon is just input plane
#                                             [-1, 1, 0], 
#                                             [1, 1, 0], 
#                                             [1, -1, 0]])]

#         for layer_id in range(len(model.model)//2): #Move polygont through layers     
#             polygons[str(layer_id)+'.linear_out']=process_with_layers(model.model[:2*layer_id+1], polygons[str(layer_id-1)+'.new_tiling']) 

#             #Split polygons w/ Relu and clip negative values to z=0
#             polygons[str(layer_id)+'.split_polygons_nested']=split_polygons_with_relu_simple(polygons[str(layer_id)+'.linear_out']) #Triple nested list so we can simplify merging process layer. 
#             polygons[str(layer_id)+'.split_polygons_nested_clipped'] = clip_polygons(polygons[str(layer_id)+'.split_polygons_nested'])
#             #Merge zero regions
#             polygons[str(layer_id)+'.split_polygons_merged'] = merge_zero_regions(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
#             #Compute new tiling
#             #This min_area thing is helpful for compute time!!! 1e-5 has lots of detail
#             polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'], min_area=1e-5)
#             print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')

#             #Optional filtering step
#             #polygons[str(layer_id)+'.new_tiling'] = filter_small_polygons(polygons[str(layer_id)+'.new_tiling'], min_area=1e-5)
#             #print(str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons remaining after filtering out small polygons')

#         #Last linear layer & output
#         polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
#         intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
#         my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)

#         print('finished computing polygons and surfaces')
        


#         #I guess I don't have to do this every time -> coudl just draw all the lines? Might be better when we get really big here
#         loops=order_closed_loops_with_closure(intersection_lines)
#         lines=VGroup()
#         print('Computing loops...')
#         for loop in tqdm(loops): 
#             loop=loop*np.array([1, 1, viz_scales[2]])
#             line = VMobject()
#             line.set_points_as_corners(loop)
#             line.set_stroke(color='#FF00FF', width=5)
#             lines.add(line)
#         lines.shift([3, 0, 0])

#         polygon_max_height=0.8

#         print('Creating top polygons...')
#         top_polygons_vgroup=VGroup()
#         for j, p in tqdm(enumerate(my_top_polygons)):
#             if len(p)<3: continue
#             if my_indicator[j]: color=YELLOW
#             else: color=BLUE
            
#             p_scaled=copy.deepcopy(p) #Scaling for viz
#             p_scaled[:,2]=p_scaled[:,2]*viz_scales[2]
#             p_scaled[:, -1] = np.clip(p_scaled[:, -1], -polygon_max_height, polygon_max_height)
#             poly_3d = Polygon(*p_scaled,
#                              fill_color=color,
#                              fill_opacity=0.4,
#                              stroke_color=color,
#                              stroke_width=0.5)
#             poly_3d.set_opacity(0.3)
#             poly_3d.shift([3, 0, 0])
#             top_polygons_vgroup.add(poly_3d)

#         surfaces[0][0].shift([3,0,0])
#         polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], 
#                                                 polygon_max_height=polygon_max_height, stroke_width=0.5)
#         polygons_21_copy=polygons_21.copy()
#         polygons_21.shift([3, 0, 0.001]) #Move slightly above map

#         surfaces[0][1].shift([3,0,0])
#         polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], 
#                                                 polygon_max_height=polygon_max_height, stroke_width=0.5)
#         polygons_22_copy=polygons_22.copy()
#         polygons_22.shift([3, 0, 0.002]) #Move slightly above map

#         polygons_22.set_color(YELLOW)
#         polygons_21.set_color(BLUE)

#         polygons_21.set_opacity(0.1)
#         polygons_22.set_opacity(0.1)
#         surfaces[0][0].set_opacity(0.4)
#         surfaces[0][1].set_opacity(0.4)
#         top_polygons_vgroup.set_opacity(0.5)



#         def flat_surf_func(u, v): return [u, v, 0]
#         flat_map_surf = ParametricSurface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
#         flat_map=TexturedSurface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
#         flat_map.set_shading(0,0,0).set_opacity(0.8)
#         # flat_map.shift([5.7, 0, 0])

#         lines_flat=VGroup()
#         for loop in loops: 
#             loop=loop*np.array([1, 1, 0])
#             line = VMobject()
#             line.set_points_as_corners(loop)
#             line.set_stroke(color='#FF00FF', width=4)
#             lines_flat.add(line)
#         # lines_flat.shift([5.7, 0, 0])

#         #First just render out overhead
#         self.add(flat_map, lines_flat)	
#         self.set_view(0, 0, 0, (-0.18, 0.01, 0.0), 2.39)
#         self.wait()
#         self.remove(flat_map, lines_flat)	
#         self.wait()

#         self.add(polygons_21, polygons_22)
#         self.add(top_polygons_vgroup)
#         self.add(lines)

#         # self.add(surfaces[0][0], surfaces[2][1]) #ok i want to include these, but the max value clipping is tricky -> ignore for now. 
#         self.add(polygons_21, polygons_22)
#         self.add(top_polygons_vgroup)
#         self.add(lines)

#         self.set_view(0, 0, 0, (2.95, -0.01, 0.15), 3.18)
#         self.wait()

#         self.play(self.animate_view(0, 38, 0, (2.97, -0.11, 0.06), 3.18), run_time=8)
#         self.wait()

#         self.play(self.animate_view(-40, 58, 0, (2.89, -0.22, -0.2), 3.20), run_time=8)
#         self.wait()

#         self.play(self.animate_view(42, 0, 0, (3.0, -0.04, -0.08), 3.86), run_time=8)
#         self.wait()

        

#         self.wait(20)
#         self.embed()


# class p12b(InteractiveScene):
#     def construct(self):
#     	#Start with 512, move to as big as I can for final
#         model_path='_2025/backprop_3/models/32_32_32_32_1.pth'
#         model = BaarleNet([32, 32, 32, 32])
#         model.load_state_dict(torch.load(model_path))
#         viz_scales=[0.07, 0.07, 0.002] #0.04]
#         num_neurons=[32, 32, 32, 32, 2]


#         #Precompute my surfaces, and polygons moving through network
#         #Skiping the zero region merging due to combinatorial explosion - not sure how much this matters right now
#         #
#         surfaces=[]
#         surface_funcs=[]
#         for layer_idx in [2]: #Skip first layers, dont need em
#             s=Group()
#             surface_funcs.append([])
#             for neuron_idx in range(num_neurons[layer_idx]):
#                 surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=viz_scales[layer_idx])
#                 bent_surface = ParametricSurface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
#                 ts=TexturedSurface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
#                 ts.set_shading(0,0,0).set_opacity(0.8)
#                 s.add(ts)
#                 surface_funcs[-1].append(surface_func)
#             surfaces.append(s)

#         print('Finished computing surfaces...')

#         #Move polygons through network
#         polygons={} #dict of all polygones as we go. 
#         polygons['-1.new_tiling_unraveled']=[np.array([[-1., -1, 0], #First polygon is just input plane
#                                             [-1, 1, 0], 
#                                             [1, 1, 0], 
#                                             [1, -1, 0]])]

#         for layer_id in range(len(model.model)//2): #Move polygont through layers     
#             polygons[str(layer_id)+'.linear_out']=process_with_layers(model.model[:2*layer_id+1], polygons[str(layer_id-1)+'.new_tiling_unraveled']) 

#             #Split polygons w/ Relu and clip negative values to z=0
#             polygons[str(layer_id)+'.split_polygons_nested']=split_polygons_with_relu_simple(polygons[str(layer_id)+'.linear_out']) #Triple nested list so we can simplify merging process layer. 
#             polygons[str(layer_id)+'.split_polygons_nested_clipped'] = clip_polygons(polygons[str(layer_id)+'.split_polygons_nested'])
#             #Merge zero regions
#             # polygons[str(layer_id)+'.split_polygons_merged'] = merge_zero_regions(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
#             #Compute new tiling
#             #This min_area thing is helpful for compute time!!! 1e-5 has lots of detail
#             # polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'], min_area=1e-5)

#             polygons[str(layer_id)+'.new_tiling']=recompute_tiling(polygons[str(layer_id)+'.split_polygons_nested'])
#             polygons[str(layer_id)+'.new_tiling_unraveled']=[item for sublist in polygons[str(layer_id)+'.new_tiling'] for item in sublist]

#             print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling_unraveled'])), ' polygons.')

#             #Optional filtering step
#             # polygons[str(layer_id)+'.new_tiling_unraveled'] = filter_small_polygons(polygons[str(layer_id)+'.new_tiling_unraveled'], min_area=1e-5)
#             # print(str(len(polygons[str(layer_id)+'.new_tiling_unraveled'])), ' polygons remaining after filtering out small polygons')

#         #Last linear layer & output
#         polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling_unraveled'])
#         intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
#         my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)

#         print('finished computing polygons and surfaces')
        


#         #I guess I don't have to do this every time -> coudl just draw all the lines? Might be better when we get really big here
#         loops=order_closed_loops_with_closure(intersection_lines)
#         lines=VGroup()
#         print('Computing loops...')
#         for loop in tqdm(loops): 
#             loop=loop*np.array([1, 1, viz_scales[2]])
#             line = VMobject()
#             line.set_points_as_corners(loop)
#             line.set_stroke(color='#FF00FF', width=5)
#             lines.add(line)
#         lines.shift([3, 0, 0])

#         polygon_max_height=0.8

#         print('Creating top polygons...')
#         top_polygons_vgroup=VGroup()
#         for j, p in tqdm(enumerate(my_top_polygons)):
#             if len(p)<3: continue
#             if my_indicator[j]: color=YELLOW
#             else: color=BLUE
            
#             p_scaled=copy.deepcopy(p) #Scaling for viz
#             p_scaled[:,2]=p_scaled[:,2]*viz_scales[2]
#             p_scaled[:, -1] = np.clip(p_scaled[:, -1], -polygon_max_height, polygon_max_height)
#             poly_3d = Polygon(*p_scaled,
#                              fill_color=color,
#                              fill_opacity=0.4,
#                              stroke_color=color,
#                              stroke_width=0.5)
#             poly_3d.set_opacity(0.3)
#             poly_3d.shift([3, 0, 0])
#             top_polygons_vgroup.add(poly_3d)

#         surfaces[0][0].shift([3,0,0])
#         polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], 
#                                                 polygon_max_height=polygon_max_height, stroke_width=0.5)
#         polygons_21_copy=polygons_21.copy()
#         polygons_21.shift([3, 0, 0.001]) #Move slightly above map

#         surfaces[0][1].shift([3,0,0])
#         polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], 
#                                                 polygon_max_height=polygon_max_height, stroke_width=0.5)
#         polygons_22_copy=polygons_22.copy()
#         polygons_22.shift([3, 0, 0.002]) #Move slightly above map

#         polygons_22.set_color(YELLOW)
#         polygons_21.set_color(BLUE)

#         polygons_21.set_opacity(0.1)
#         polygons_22.set_opacity(0.1)
#         surfaces[0][0].set_opacity(0.4)
#         surfaces[0][1].set_opacity(0.4)
#         top_polygons_vgroup.set_opacity(0.5)



#         def flat_surf_func(u, v): return [u, v, 0]
#         flat_map_surf = ParametricSurface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
#         flat_map=TexturedSurface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
#         flat_map.set_shading(0,0,0).set_opacity(0.8)
#         # flat_map.shift([5.7, 0, 0])

#         lines_flat=VGroup()
#         for loop in loops: 
#             loop=loop*np.array([1, 1, 0])
#             line = VMobject()
#             line.set_points_as_corners(loop)
#             line.set_stroke(color='#FF00FF', width=4)
#             lines_flat.add(line)
#         # lines_flat.shift([5.7, 0, 0])

#         #First just render out overhead
#         self.add(flat_map, lines_flat)	
#         self.set_view(0, 0, 0, (-0.18, 0.01, 0.0), 2.39)
#         self.wait()
#         self.remove(flat_map, lines_flat)	
#         self.wait()

#         self.add(polygons_21, polygons_22)
#         self.add(top_polygons_vgroup)
#         self.add(lines)

#         # self.add(surfaces[0][0], surfaces[2][1]) #ok i want to include these, but the max value clipping is tricky -> ignore for now. 
#         self.add(polygons_21, polygons_22)
#         self.add(top_polygons_vgroup)
#         self.add(lines)

#         self.set_view(0, 0, 0, (2.95, -0.01, 0.15), 3.18)
#         self.wait()

#         self.play(self.animate_view(0, 38, 0, (2.97, -0.11, 0.06), 3.18), run_time=8)
#         self.wait()

#         self.play(self.animate_view(-40, 58, 0, (2.89, -0.22, -0.2), 3.20), run_time=8)
#         self.wait()

#         self.play(self.animate_view(42, 0, 0, (3.0, -0.04, -0.08), 3.86), run_time=8)
#         self.wait()

        

#         self.wait(20)
#         self.embed()










