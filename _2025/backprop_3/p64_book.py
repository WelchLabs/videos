from welch_paths import welch_path
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
import copy
import manimgx as gx
import numpy as np
import pickle
import torch
from geometric_dl_utils import (
    BaarleNet,
    compute_adaptive_viz_scales,
    manim_polygons_from_np_list,
    surface_func_from_model,
)
from geometric_dl_utils_simplified import (
    clip_polygons,
    compute_top_polytope,
    merge_zero_regions,
    process_with_layers,
    recompute_tiling_polygonize,
    split_polygons_with_relu_simple,
)
from welch_scene import WelchScene
from welch_surface import parametric_surface, textured_surface
from welch_text import markup_text
from polytope_intersection_utils import intersect_polytopes
from welch_surface import set_surface_shading
from tqdm import tqdm
from order_matching_tools import reorder_polygons_optimal, reorder_polygons_greedy
from gap_filler import fill_gaps

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'

# graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/' #Point to folder where map images are
graphics_dir=welch_path('ai_book/4_deep_learning/graphics/')
map_filename='baarle_hertog_maps-13.png'
colors = [BLUE, gx.GREY, GREEN, gx.TEAL, gx.PURPLE, gx.PINK, gx.TEAL, YELLOW, FRESH_TAN, CHILL_BLUE, CHILL_GREEN, YELLOW_FADE]
# colors = [GREY, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]


class p64(WelchScene):
    '''Ok so this is the transition of my 3d shape for the 2d view -> try blending in premiere!'''
    def construct(self):

        model = BaarleNet([32, 32, 32, 32])
        viz_scales=[0.06, 0.06, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.15]
        num_neurons=[32, 32, 32, 32, 32, 32, 32, 32, 2]
        vertical_spacing=1.0


        data_path=welch_path('backprop_3/hackin/training_caches/32_32_32_32_1.pkl')
        with open(data_path, 'rb') as file:
            training_cache = pickle.load(file) #Training cache

        prev_layer_1_polygons=None
        prev_layer_2_polygons=None
        prev_layer_3_polygons=None
        prev_layer_4_polygons=None

        self.set_view(0, 0, 0, (3.94, 0.38, 0.0), 2.04)
        # for train_step in tqdm(range(len(training_cache['weights_history']))): #All steps
        # for train_step in tqdm(np.arange(0, len(training_cache['weights_history']), 100)):
        # for train_step in tqdm(np.arange(0,  1000, 1)):
        # for train_step in tqdm(np.arange(0, 600, 100)):
        # train_step=2697 #OK FUCKING DOPE -> If I can make it to 2697, we get all the regions!
        train_step=2697 #tqdm(list(range(0,  2690, 10))+[2697]):




        w1=training_cache['weights_history'][train_step]['model.0.weight'].numpy()
        b1=training_cache['weights_history'][train_step]['model.0.bias'].numpy()
        w2=training_cache['weights_history'][train_step]['model.2.weight'].numpy()
        b2=training_cache['weights_history'][train_step]['model.2.bias'].numpy()
        w3=training_cache['weights_history'][train_step]['model.4.weight'].numpy()
        b3=training_cache['weights_history'][train_step]['model.4.bias'].numpy()
        w4=training_cache['weights_history'][train_step]['model.6.weight'].numpy()
        b4=training_cache['weights_history'][train_step]['model.6.bias'].numpy()
        w5=training_cache['weights_history'][train_step]['model.8.weight'].numpy()
        b5=training_cache['weights_history'][train_step]['model.8.bias'].numpy()

        with torch.no_grad():
            model.model[0].weight.copy_(torch.from_numpy(w1))
            model.model[0].bias.copy_(torch.from_numpy(b1))
            model.model[2].weight.copy_(torch.from_numpy(w2))
            model.model[2].bias.copy_(torch.from_numpy(b2))
            model.model[4].weight.copy_(torch.from_numpy(w3))
            model.model[4].bias.copy_(torch.from_numpy(b3))
            model.model[6].weight.copy_(torch.from_numpy(w4))
            model.model[6].bias.copy_(torch.from_numpy(b4))
            model.model[8].weight.copy_(torch.from_numpy(w5))
            model.model[8].bias.copy_(torch.from_numpy(b5))


        adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=0.6, extent=1)
        #For the interesection to make sense, these scales need to match - either need to manual overide or chnage method above
        final_layer_viz=scale=1.4*min(adaptive_viz_scales[-1]) #little manual ramp here
        adaptive_viz_scales[-1]=[final_layer_viz, final_layer_viz]

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
            if layer_idx>7: #Skip early surfaces to save time. 
                for neuron_idx in range(num_neurons[layer_idx]):
                    surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx]) #viz_scales[layer_idx])
                    bent_surface = parametric_surface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
                    ts=textured_surface(bent_surface, graphics_dir+'/baarle_hertog_maps/'+map_filename)
                    set_surface_shading(ts, 0,0,0).set_opacity(0.8)
                    s.add(ts)
                    surface_funcs[-1].append(surface_func)
            surfaces.append(s)

        #Move polygons through network
        polygons={} #dict of all polygones as we go. 
        polygons['-1.new_tiling']=[np.array([[-1., -1, 0], #First polygon is just input plane
                                            [-1, 1, 0], 
                                            [1, 1, 0], 
                                            [1, -1, 0]])]

        for layer_id in range(len(model.model)//2): #Move polygont through layers     
            polygons[str(layer_id)+'.linear_out']=process_with_layers(model.model[:2*layer_id+1], polygons[str(layer_id-1)+'.new_tiling']) 

            #Split polygons w/ Relu and clip negative values to z=0
            polygons[str(layer_id)+'.split_polygons_nested']=split_polygons_with_relu_simple(polygons[str(layer_id)+'.linear_out']) #Triple nested list so we can simplify merging process layer. 
            polygons[str(layer_id)+'.split_polygons_nested_clipped'] = clip_polygons(polygons[str(layer_id)+'.split_polygons_nested'])
            #Merge zero regions
            polygons[str(layer_id)+'.split_polygons_merged'] = merge_zero_regions(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
            
            #Compute new tiling - general method with merging - should be more accurate but slow - buggy?
            # polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
            
            #Less general method - less accurate, faster, maybe less buggy
            # polygons[str(layer_id)+'.new_tiling_nested']=recompute_tiling(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
            polygons[str(layer_id)+'.new_tiling_nested']=recompute_tiling_polygonize(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
            polygons[str(layer_id)+'.new_tiling']=[item for sublist in polygons[str(layer_id)+'.new_tiling_nested'] for item in sublist]


            print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')
            #Optional filtering step
            #polygons[str(layer_id)+'.new_tiling'] = filter_small_polygons(polygons[str(layer_id)+'.new_tiling'], min_area=1e-5)
            #print(str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons remaining after filtering out small polygons')


        #Last linear layer & output
        polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
        intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
        my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)

        print(len(my_top_polygons), len(my_indicator))
        my_top_polygons, my_indicator=fill_gaps(my_top_polygons, my_indicator) #Will this help with my random gaps?
        print(len(my_top_polygons), len(my_indicator))


        if prev_layer_1_polygons is not None: 
            prev_layer_1_polygons=reorder_polygons_optimal(prev_layer_1_polygons, polygons['0.new_tiling'])
        else:
            prev_layer_1_polygons=polygons['0.new_tiling']
        layer_1_polygons_flat=manim_polygons_from_np_list(prev_layer_1_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6, stroke_width=0.6)

        if prev_layer_2_polygons is not None: 
            prev_layer_2_polygons=reorder_polygons_optimal(prev_layer_2_polygons, polygons['1.new_tiling'])
        else:
            prev_layer_2_polygons=polygons['1.new_tiling']         
        layer_2_polygons_flat=manim_polygons_from_np_list(prev_layer_2_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6, stroke_width=0.6)

        if prev_layer_3_polygons is not None: 
            prev_layer_3_polygons=reorder_polygons_optimal(prev_layer_3_polygons, polygons['2.new_tiling'])
        else:
            prev_layer_3_polygons=polygons['2.new_tiling']
        layer_3_polygons_flat=manim_polygons_from_np_list(prev_layer_3_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6, stroke_width=0.6)

        if prev_layer_4_polygons is not None: 
            prev_layer_4_polygons=reorder_polygons_optimal(prev_layer_4_polygons, polygons['3.new_tiling'])
        else:
            prev_layer_4_polygons=polygons['3.new_tiling']
        layer_4_polygons_flat=manim_polygons_from_np_list(prev_layer_4_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6, stroke_width=0.6)



        #Outputs surfaces
        groups_output=gx.Group()
        layer_idx=len(model.model)-1
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['3.linear_out'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=gx.Group(s, pgs)
            groups_output.add(g)


        #Output surfaces together
        group_combined_output=groups_output.copy()
        group_combined_output[0].set_color(BLUE)
        group_combined_output[1].set_color(YELLOW)

        top_polygons_vgroup=gx.VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=p_scaled[:,2]*adaptive_viz_scales[-1][0] #Flatten that shit!
            poly_3d = gx.Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.6,
                             stroke_color=color,
                             stroke_width=0.6)
            poly_3d.set_opacity(0.5)
            top_polygons_vgroup.add(poly_3d)

        lines=gx.VGroup()
        for loop in intersection_lines: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=3)
            lines.add(line)

        top_polygons_vgroup_flat=gx.VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=0 #p_scaled[:,2]*viz_scales[2] #Flatten that shit!
            poly_3d = gx.Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.6,
                             stroke_color=color,
                             stroke_width=0.6)
            poly_3d.set_opacity(0.5)
            top_polygons_vgroup_flat.add(poly_3d)

        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map_2=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/'+map_filename)
        set_surface_shading(flat_map_2, 0,0,0).set_opacity(0.8)

        lines_flat=gx.VGroup()
        for loop in intersection_lines: 
            # loop=np.hstack((loop, np.zeros((len(loop), 1))))
            loop[:,2]=0
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=3)
            lines_flat.add(line)    

        lines_flat_copy=lines_flat.copy()

        group_combined_output.set_opacity(0.3)
        final_map_group=gx.Group(flat_map_2, top_polygons_vgroup_flat, lines_flat)
        border_map_only=gx.Group(flat_map_2.copy(), lines_flat.copy())

        # I can change locations in premiere pretty easily if I need to!
        layer_1_polygons_flat.shift([ 0.,  2., 0.])                                                                                                                                                       
        layer_2_polygons_flat.shift([ 2.35 , 2.,   0.  ])                                                                                                                                                
        layer_3_polygons_flat.shift([ 0.,   -0.35, 0.  ])

        layer_4_group=gx.Group(layer_4_polygons_flat, lines_flat_copy)
        layer_4_group.shift([ 2.35, -0.35, 0.  ])  
        # layer_4_polygons_flat.shift([ 2.35, -0.35, 0.  ])                                                                                                                                                
        
        final_map_group.shift([ 2*2.35, 2, 0.  ])
        border_map_only.shift([ 2*2.35, -0.35, 0.  ])

        # self.wait()
        # self.set_view(0, 0, 0, (3.94, 0.38, 0.0), 2.04) #Could do a nice zoom in animation from not sure if it's worth it or not
        # self.set_view(0, 0, 0, (3.07, 0.82, 0.0), 0.77)
        self.set_view(0, 0, 0, (2.44, 0.83, 0.0), 4.55)
        layer_1_polygons_flat.set_opacity(1.0)
        layer_2_polygons_flat.set_opacity(1.0)
        layer_3_polygons_flat.set_opacity(1.0)
        layer_4_polygons_flat.set_opacity(1.0)
        self.add(layer_1_polygons_flat)
        self.add(layer_2_polygons_flat)
        self.add(layer_3_polygons_flat)
        self.add(layer_4_polygons_flat)
        # self.add(layer_4_group)
        self.add(final_map_group)
        self.add(border_map_only)
        self.wait()

        lines_flat_copy.set_stroke(color=gx.WHITE)
        self.add(lines_flat_copy)

        self.set_view(0, 0, 0, (2.49, -0.35, 0.0), 2.00)
        self.wait()
        # self.set_view(0, 0, 0, (2.35, -0.78, 0.0), 1.15)
        # self.wait()

        self.remove(lines_flat_copy)
        self.wait()
        # self.set_view(0, 0, 0, (2.36, 0.08, 0.0), 1.15)
        # self.wait()
        # self.set_view(0, 0, 0, (2.35, -0.78, 0.0), 1.15)
        # self.wait()

        self.set_view(0, 0, 0, (4.84, -0.37, 0.0), 2.16)
        self.wait()

        self.set_view(0, 0, 0, (4.86, 2.0, 0.0), 2.04)
        self.wait()

        self.wait(0.1)


        # self.remove(final_map_group)
        # self.add(flat_map_2)
        # self.add(lines_flat)


        self.wait(20)
        pass


class p64_v2(WelchScene):
    '''Ok so this is the transition of my 3d shape for the 2d view -> try blending in premiere!'''
    def construct(self):

        model = BaarleNet([32, 32, 32, 32])
        viz_scales=[0.06, 0.06, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.15]
        num_neurons=[32, 32, 32, 32, 32, 32, 32, 32, 2]
        vertical_spacing=1.0


        data_path=welch_path('backprop_3/hackin/training_caches/32_32_32_32_1.pkl')
        with open(data_path, 'rb') as file:
            training_cache = pickle.load(file) #Training cache

        prev_layer_1_polygons=None
        prev_layer_2_polygons=None
        prev_layer_3_polygons=None
        prev_layer_4_polygons=None

        self.set_view(0, 0, 0, (3.94, 0.38, 0.0), 2.04)
        # for train_step in tqdm(range(len(training_cache['weights_history']))): #All steps
        # for train_step in tqdm(np.arange(0, len(training_cache['weights_history']), 100)):
        # for train_step in tqdm(np.arange(0,  1000, 1)):
        # for train_step in tqdm(np.arange(0, 600, 100)):
        # train_step=2697 #OK FUCKING DOPE -> If I can make it to 2697, we get all the regions!
        train_step=240 #tqdm(list(range(0,  2690, 10))+[2697]):




        w1=training_cache['weights_history'][train_step]['model.0.weight'].numpy()
        b1=training_cache['weights_history'][train_step]['model.0.bias'].numpy()
        w2=training_cache['weights_history'][train_step]['model.2.weight'].numpy()
        b2=training_cache['weights_history'][train_step]['model.2.bias'].numpy()
        w3=training_cache['weights_history'][train_step]['model.4.weight'].numpy()
        b3=training_cache['weights_history'][train_step]['model.4.bias'].numpy()
        w4=training_cache['weights_history'][train_step]['model.6.weight'].numpy()
        b4=training_cache['weights_history'][train_step]['model.6.bias'].numpy()
        w5=training_cache['weights_history'][train_step]['model.8.weight'].numpy()
        b5=training_cache['weights_history'][train_step]['model.8.bias'].numpy()

        with torch.no_grad():
            model.model[0].weight.copy_(torch.from_numpy(w1))
            model.model[0].bias.copy_(torch.from_numpy(b1))
            model.model[2].weight.copy_(torch.from_numpy(w2))
            model.model[2].bias.copy_(torch.from_numpy(b2))
            model.model[4].weight.copy_(torch.from_numpy(w3))
            model.model[4].bias.copy_(torch.from_numpy(b3))
            model.model[6].weight.copy_(torch.from_numpy(w4))
            model.model[6].bias.copy_(torch.from_numpy(b4))
            model.model[8].weight.copy_(torch.from_numpy(w5))
            model.model[8].bias.copy_(torch.from_numpy(b5))


        adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=0.6, extent=1)
        #For the interesection to make sense, these scales need to match - either need to manual overide or chnage method above
        final_layer_viz=scale=1.4*min(adaptive_viz_scales[-1]) #little manual ramp here
        adaptive_viz_scales[-1]=[final_layer_viz, final_layer_viz]

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
            if layer_idx>7: #Skip early surfaces to save time. 
                for neuron_idx in range(num_neurons[layer_idx]):
                    surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx]) #viz_scales[layer_idx])
                    bent_surface = parametric_surface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
                    ts=textured_surface(bent_surface, graphics_dir+'/baarle_hertog_maps/'+map_filename)
                    set_surface_shading(ts, 0,0,0).set_opacity(0.8)
                    s.add(ts)
                    surface_funcs[-1].append(surface_func)
            surfaces.append(s)

        #Move polygons through network
        polygons={} #dict of all polygones as we go. 
        polygons['-1.new_tiling']=[np.array([[-1., -1, 0], #First polygon is just input plane
                                            [-1, 1, 0], 
                                            [1, 1, 0], 
                                            [1, -1, 0]])]

        for layer_id in range(len(model.model)//2): #Move polygont through layers     
            polygons[str(layer_id)+'.linear_out']=process_with_layers(model.model[:2*layer_id+1], polygons[str(layer_id-1)+'.new_tiling']) 

            #Split polygons w/ Relu and clip negative values to z=0
            polygons[str(layer_id)+'.split_polygons_nested']=split_polygons_with_relu_simple(polygons[str(layer_id)+'.linear_out']) #Triple nested list so we can simplify merging process layer. 
            polygons[str(layer_id)+'.split_polygons_nested_clipped'] = clip_polygons(polygons[str(layer_id)+'.split_polygons_nested'])
            #Merge zero regions
            polygons[str(layer_id)+'.split_polygons_merged'] = merge_zero_regions(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
            
            #Compute new tiling - general method with merging - should be more accurate but slow - buggy?
            # polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
            
            #Less general method - less accurate, faster, maybe less buggy
            # polygons[str(layer_id)+'.new_tiling_nested']=recompute_tiling(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
            polygons[str(layer_id)+'.new_tiling_nested']=recompute_tiling_polygonize(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
            polygons[str(layer_id)+'.new_tiling']=[item for sublist in polygons[str(layer_id)+'.new_tiling_nested'] for item in sublist]


            print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')
            #Optional filtering step
            #polygons[str(layer_id)+'.new_tiling'] = filter_small_polygons(polygons[str(layer_id)+'.new_tiling'], min_area=1e-5)
            #print(str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons remaining after filtering out small polygons')


        #Last linear layer & output
        polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
        intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
        my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)

        print(len(my_top_polygons), len(my_indicator))
        my_top_polygons, my_indicator=fill_gaps(my_top_polygons, my_indicator) #Will this help with my random gaps?
        print(len(my_top_polygons), len(my_indicator))


        if prev_layer_1_polygons is not None: 
            prev_layer_1_polygons=reorder_polygons_optimal(prev_layer_1_polygons, polygons['0.new_tiling'])
        else:
            prev_layer_1_polygons=polygons['0.new_tiling']
        layer_1_polygons_flat=manim_polygons_from_np_list(prev_layer_1_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6, stroke_width=0.6)

        if prev_layer_2_polygons is not None: 
            prev_layer_2_polygons=reorder_polygons_optimal(prev_layer_2_polygons, polygons['1.new_tiling'])
        else:
            prev_layer_2_polygons=polygons['1.new_tiling']         
        layer_2_polygons_flat=manim_polygons_from_np_list(prev_layer_2_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6, stroke_width=0.6)

        if prev_layer_3_polygons is not None: 
            prev_layer_3_polygons=reorder_polygons_optimal(prev_layer_3_polygons, polygons['2.new_tiling'])
        else:
            prev_layer_3_polygons=polygons['2.new_tiling']
        layer_3_polygons_flat=manim_polygons_from_np_list(prev_layer_3_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6, stroke_width=0.6)

        if prev_layer_4_polygons is not None: 
            prev_layer_4_polygons=reorder_polygons_optimal(prev_layer_4_polygons, polygons['3.new_tiling'])
        else:
            prev_layer_4_polygons=polygons['3.new_tiling']
        layer_4_polygons_flat=manim_polygons_from_np_list(prev_layer_4_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6, stroke_width=0.6)



        #Outputs surfaces
        groups_output=gx.Group()
        layer_idx=len(model.model)-1
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['3.linear_out'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=gx.Group(s, pgs)
            groups_output.add(g)


        #Output surfaces together
        group_combined_output=groups_output.copy()
        group_combined_output[0].set_color(BLUE)
        group_combined_output[1].set_color(YELLOW)

        top_polygons_vgroup=gx.VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=p_scaled[:,2]*adaptive_viz_scales[-1][0] #Flatten that shit!
            poly_3d = gx.Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.6,
                             stroke_color=color,
                             stroke_width=0.6)
            poly_3d.set_opacity(0.5)
            top_polygons_vgroup.add(poly_3d)

        lines=gx.VGroup()
        for loop in intersection_lines: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=3)
            lines.add(line)

        top_polygons_vgroup_flat=gx.VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=0 #p_scaled[:,2]*viz_scales[2] #Flatten that shit!
            poly_3d = gx.Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.6,
                             stroke_color=color,
                             stroke_width=0.6)
            poly_3d.set_opacity(0.5)
            top_polygons_vgroup_flat.add(poly_3d)

        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map_2=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/'+map_filename)
        set_surface_shading(flat_map_2, 0,0,0).set_opacity(0.8)

        lines_flat=gx.VGroup()
        for loop in intersection_lines: 
            # loop=np.hstack((loop, np.zeros((len(loop), 1))))
            loop[:,2]=0
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=3)
            lines_flat.add(line)    

        lines_flat_copy=lines_flat.copy()

        group_combined_output.set_opacity(0.3)
        final_map_group=gx.Group(flat_map_2, top_polygons_vgroup_flat, lines_flat)
        border_map_only=gx.Group(flat_map_2.copy(), lines_flat.copy())

        # I can change locations in premiere pretty easily if I need to!
        layer_1_polygons_flat.shift([ 0.,  2., 0.])                                                                                                                                                       
        layer_2_polygons_flat.shift([ 2.35 , 2.,   0.  ])                                                                                                                                                
        layer_3_polygons_flat.shift([ 0.,   -0.35, 0.  ])

        layer_4_group=gx.Group(layer_4_polygons_flat, lines_flat_copy)
        layer_4_group.shift([ 2.35, -0.35, 0.  ])  
        # layer_4_polygons_flat.shift([ 2.35, -0.35, 0.  ])                                                                                                                                                
        
        final_map_group.shift([ 2*2.35, 2, 0.  ])
        border_map_only.shift([ 2*2.35, -0.35, 0.  ])

        # self.wait()
        # self.set_view(0, 0, 0, (3.94, 0.38, 0.0), 2.04) #Could do a nice zoom in animation from not sure if it's worth it or not
        # self.set_view(0, 0, 0, (3.07, 0.82, 0.0), 0.77)
        self.set_view(0, 0, 0, (2.44, 0.83, 0.0), 4.55)
        layer_1_polygons_flat.set_opacity(1.0)
        layer_2_polygons_flat.set_opacity(1.0)
        layer_3_polygons_flat.set_opacity(1.0)
        layer_4_polygons_flat.set_opacity(1.0)
        self.add(layer_1_polygons_flat)
        self.add(layer_2_polygons_flat)
        self.add(layer_3_polygons_flat)
        self.add(layer_4_polygons_flat)
        # self.add(layer_4_group)
        self.add(final_map_group)
        self.add(border_map_only)
        self.wait()

        lines_flat_copy.set_stroke(color=gx.WHITE)
        self.add(lines_flat_copy)

        self.set_view(0, 0, 0, (2.49, -0.35, 0.0), 2.00)
        self.wait()
        # self.set_view(0, 0, 0, (2.35, -0.78, 0.0), 1.15)
        # self.wait()

        self.remove(lines_flat_copy)
        self.wait()
        # self.set_view(0, 0, 0, (2.36, 0.08, 0.0), 1.15)
        # self.wait()
        # self.set_view(0, 0, 0, (2.35, -0.78, 0.0), 1.15)
        # self.wait()

        self.set_view(0, 0, 0, (4.84, -0.37, 0.0), 2.16)
        self.wait()

        self.set_view(0, 0, 0, (4.86, 2.0, 0.0), 2.04)
        self.wait()

        self.wait(0.1)


        # self.remove(final_map_group)
        # self.add(flat_map_2)
        # self.add(lines_flat)


        self.wait(20)
        pass




class p64b(WelchScene):
    '''Ok so this is the transition of my 3d shape for the 2d view -> try blending in premiere!'''
    def construct(self):

        model = BaarleNet([32, 32, 32, 32])
        viz_scales=[0.06, 0.06, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.15]
        num_neurons=[32, 32, 32, 32, 32, 32, 32, 32, 2]
        vertical_spacing=1.0


        data_path=welch_path('backprop_3/hackin/training_caches/32_32_32_32_1.pkl')
        with open(data_path, 'rb') as file:
            training_cache = pickle.load(file) #Training cache

        self.set_view(27, 54, 0, (-0.02, 0.05, -0.55), 3.99)
        # for train_step in tqdm(range(len(training_cache['weights_history']))): #All steps
        # for train_step in tqdm(np.arange(0, len(training_cache['weights_history']), 100)):
        # for train_step in tqdm(np.arange(0,  1000, 1)):
        # for train_step in tqdm(np.arange(0, 600, 100)):
        # train_step=2697 #OK FUCKING DOPE -> If I can make it to 2697, we get all the regions!
        layer_1_polygons_flat = layer_2_polygons_flat = layer_3_polygons_flat = layer_4_polygons_flat = final_map_group = border_map_only = None
        for train_step in [2697]: #tqdm(list(range(0,  2690, 10))+[2697]):

            if layer_1_polygons_flat is not None:
                self.remove(layer_1_polygons_flat, layer_2_polygons_flat, layer_3_polygons_flat, layer_4_polygons_flat, final_map_group, border_map_only)


            w1=training_cache['weights_history'][train_step]['model.0.weight'].numpy()
            b1=training_cache['weights_history'][train_step]['model.0.bias'].numpy()
            w2=training_cache['weights_history'][train_step]['model.2.weight'].numpy()
            b2=training_cache['weights_history'][train_step]['model.2.bias'].numpy()
            w3=training_cache['weights_history'][train_step]['model.4.weight'].numpy()
            b3=training_cache['weights_history'][train_step]['model.4.bias'].numpy()
            w4=training_cache['weights_history'][train_step]['model.6.weight'].numpy()
            b4=training_cache['weights_history'][train_step]['model.6.bias'].numpy()
            w5=training_cache['weights_history'][train_step]['model.8.weight'].numpy()
            b5=training_cache['weights_history'][train_step]['model.8.bias'].numpy()

            with torch.no_grad():
                model.model[0].weight.copy_(torch.from_numpy(w1))
                model.model[0].bias.copy_(torch.from_numpy(b1))
                model.model[2].weight.copy_(torch.from_numpy(w2))
                model.model[2].bias.copy_(torch.from_numpy(b2))
                model.model[4].weight.copy_(torch.from_numpy(w3))
                model.model[4].bias.copy_(torch.from_numpy(b3))
                model.model[6].weight.copy_(torch.from_numpy(w4))
                model.model[6].bias.copy_(torch.from_numpy(b4))
                model.model[8].weight.copy_(torch.from_numpy(w5))
                model.model[8].bias.copy_(torch.from_numpy(b5))


            adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=0.6, extent=1)
            #For the interesection to make sense, these scales need to match - either need to manual overide or chnage method above
            # final_layer_viz=scale=1.4*min(adaptive_viz_scales[-1]) #little manual ramp here
            # adaptive_viz_scales[-1]=[final_layer_viz, final_layer_viz]
            adaptive_viz_scales[-1]=[0.014, 0.014] #Hard code bro

            #Precompute my surfaces, and polygons moving through network
            surfaces=[]
            surface_funcs=[]
            for layer_idx in range(len(model.model)):
                s=gx.Group()
                surface_funcs.append([])
                if layer_idx>7: #Skip early surfaces to save time. 
                    for neuron_idx in range(num_neurons[layer_idx]):
                        surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx]) #viz_scales[layer_idx])
                        bent_surface = parametric_surface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
                        ts=textured_surface(bent_surface, graphics_dir+'/baarle_hertog_maps/'+map_filename)
                        set_surface_shading(ts, 0,0,0).set_opacity(0.8)
                        s.add(ts)
                        surface_funcs[-1].append(surface_func)
                surfaces.append(s)

            #Move polygons through network
            polygons={} #dict of all polygones as we go. 
            polygons['-1.new_tiling']=[np.array([[-1., -1, 0], #First polygon is just input plane
                                                [-1, 1, 0], 
                                                [1, 1, 0], 
                                                [1, -1, 0]])]

            for layer_id in range(len(model.model)//2): #Move polygont through layers     
                polygons[str(layer_id)+'.linear_out']=process_with_layers(model.model[:2*layer_id+1], polygons[str(layer_id-1)+'.new_tiling']) 

                #Split polygons w/ Relu and clip negative values to z=0
                polygons[str(layer_id)+'.split_polygons_nested']=split_polygons_with_relu_simple(polygons[str(layer_id)+'.linear_out']) #Triple nested list so we can simplify merging process layer. 
                polygons[str(layer_id)+'.split_polygons_nested_clipped'] = clip_polygons(polygons[str(layer_id)+'.split_polygons_nested'])
                #Merge zero regions
                polygons[str(layer_id)+'.split_polygons_merged'] = merge_zero_regions(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
                
                #Compute new tiling - general method with merging - should be more accurate but slow - buggy?
                # polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
                
                #Less general method - less accurate, faster, maybe less buggy
                # polygons[str(layer_id)+'.new_tiling_nested']=recompute_tiling(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
                polygons[str(layer_id)+'.new_tiling_nested']=recompute_tiling_polygonize(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
                polygons[str(layer_id)+'.new_tiling']=[item for sublist in polygons[str(layer_id)+'.new_tiling_nested'] for item in sublist]


                print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')
                #Optional filtering step
                #polygons[str(layer_id)+'.new_tiling'] = filter_small_polygons(polygons[str(layer_id)+'.new_tiling'], min_area=1e-5)
                #print(str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons remaining after filtering out small polygons')


            #Last linear layer & output
            polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
            intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
            my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)

            print(len(my_top_polygons), len(my_indicator))
            my_top_polygons, my_indicator=fill_gaps(my_top_polygons, my_indicator) #Will this help with my random gaps?
            print(len(my_top_polygons), len(my_indicator))

            #Outputs surfaces
            groups_output=gx.Group()
            layer_idx=len(model.model)-1
            total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
            start_z = total_height / 2  # Start from top
            for neuron_idx in range(num_neurons[layer_idx]):
                # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
                pgs=manim_polygons_from_np_list(polygons['3.linear_out'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
                s=surfaces[layer_idx][neuron_idx]
                g=gx.Group(s, pgs)
                groups_output.add(g)


            #Output surfaces together
            group_combined_output=groups_output.copy()
            group_combined_output[0].set_color(BLUE)
            group_combined_output[1].set_color(YELLOW)

            top_polygons_vgroup=gx.VGroup()
            for j, p in enumerate(my_top_polygons):
                if len(p)<3: continue
                if my_indicator[j]: color=YELLOW
                else: color=BLUE
                p_scaled=copy.deepcopy(p) #Scaling for viz
                p_scaled[:,2]=p_scaled[:,2]*adaptive_viz_scales[-1][0] #Flatten that shit!
                poly_3d = gx.Polygon(*p_scaled,
                                 fill_color=color,
                                 fill_opacity=0.4,
                                 stroke_color=color,
                                 stroke_width=0.6)
                poly_3d.set_opacity(0.5)
                top_polygons_vgroup.add(poly_3d)

            lines=gx.VGroup()
            for loop in intersection_lines: 
                loop = loop * np.array([1, 1, adaptive_viz_scales[-1][0]])
                line = gx.VMobject()
                line.set_points_as_corners(loop)
                line.set_stroke(color='#ec008c', width=4)
                lines.add(line)

            top_polygons_vgroup_flat=gx.VGroup()
            for j, p in enumerate(my_top_polygons):
                if len(p)<3: continue
                if my_indicator[j]: color=YELLOW
                else: color=BLUE
                p_scaled=copy.deepcopy(p) #Scaling for viz
                p_scaled[:,2]=0 #p_scaled[:,2]*viz_scales[2] #Flatten that shit!
                poly_3d = gx.Polygon(*p_scaled,
                                 fill_color=color,
                                 fill_opacity=0.4,
                                 stroke_color=color,
                                 stroke_width=0.6)
                poly_3d.set_opacity(0.5)
                top_polygons_vgroup_flat.add(poly_3d)


            group_combined_output.set_opacity(0.3)
            top_polygons_vgroup.set_opacity(0.6)
            
            # self.add(layer_1_polygons_flat, layer_2_polygons_flat, layer_3_polygons_flat, final_map_group)

            combined_3d_group=gx.Group(group_combined_output, top_polygons_vgroup, lines)
            # combined_3d_group.move_to(ORIGIN)
            self.add(combined_3d_group)
            self.wait(0.1)


        # self.remove(final_map_group)
        # self.add(flat_map_2)
        # self.add(lines_flat)


        self.wait(20)
        pass



class p64_playaround_on_last_frame_d(WelchScene):
    '''Ok so this is the transition of my 3d shape for the 2d view -> try blending in premiere!'''
    def construct(self):

        model = BaarleNet([32, 32, 32, 32])
        viz_scales=[0.06, 0.06, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.15]
        num_neurons=[32, 32, 32, 32, 32, 32, 32, 32, 2]
        vertical_spacing=1.0


        data_path=welch_path('backprop_3/hackin/training_caches/32_32_32_32_1.pkl')
        with open(data_path, 'rb') as file:
            training_cache = pickle.load(file) #Training cache

        prev_layer_1_polygons=None
        prev_layer_2_polygons=None
        prev_layer_3_polygons=None
        prev_layer_4_polygons=None

        self.set_view(0, 0, 0, (3.94, 0.38, 0.0), 2.04)
        # for train_step in tqdm(range(len(training_cache['weights_history']))): #All steps
        # for train_step in tqdm(np.arange(0, len(training_cache['weights_history']), 100)):
        # for train_step in tqdm(np.arange(0,  1000, 1)):
        # for train_step in tqdm(np.arange(0, 600, 100)):
        # train_step=2697 #OK FUCKING DOPE -> If I can make it to 2697, we get all the regions!
        train_step=2697



        w1=training_cache['weights_history'][train_step]['model.0.weight'].numpy()
        b1=training_cache['weights_history'][train_step]['model.0.bias'].numpy()
        w2=training_cache['weights_history'][train_step]['model.2.weight'].numpy()
        b2=training_cache['weights_history'][train_step]['model.2.bias'].numpy()
        w3=training_cache['weights_history'][train_step]['model.4.weight'].numpy()
        b3=training_cache['weights_history'][train_step]['model.4.bias'].numpy()
        w4=training_cache['weights_history'][train_step]['model.6.weight'].numpy()
        b4=training_cache['weights_history'][train_step]['model.6.bias'].numpy()
        w5=training_cache['weights_history'][train_step]['model.8.weight'].numpy()
        b5=training_cache['weights_history'][train_step]['model.8.bias'].numpy()

        with torch.no_grad():
            model.model[0].weight.copy_(torch.from_numpy(w1))
            model.model[0].bias.copy_(torch.from_numpy(b1))
            model.model[2].weight.copy_(torch.from_numpy(w2))
            model.model[2].bias.copy_(torch.from_numpy(b2))
            model.model[4].weight.copy_(torch.from_numpy(w3))
            model.model[4].bias.copy_(torch.from_numpy(b3))
            model.model[6].weight.copy_(torch.from_numpy(w4))
            model.model[6].bias.copy_(torch.from_numpy(b4))
            model.model[8].weight.copy_(torch.from_numpy(w5))
            model.model[8].bias.copy_(torch.from_numpy(b5))


        adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=0.6, extent=1)
        #For the interesection to make sense, these scales need to match - either need to manual overide or chnage method above
        final_layer_viz=scale=1.4*min(adaptive_viz_scales[-1]) #little manual ramp here
        adaptive_viz_scales[-1]=[final_layer_viz, final_layer_viz]

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
            if layer_idx>7: #Skip early surfaces to save time. 
                for neuron_idx in range(num_neurons[layer_idx]):
                    surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx]) #viz_scales[layer_idx])
                    bent_surface = parametric_surface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
                    ts=textured_surface(bent_surface, graphics_dir+'/baarle_hertog_maps/'+map_filename)
                    set_surface_shading(ts, 0,0,0).set_opacity(0.8)
                    s.add(ts)
                    surface_funcs[-1].append(surface_func)
            surfaces.append(s)

        #Move polygons through network
        polygons={} #dict of all polygones as we go. 
        polygons['-1.new_tiling']=[np.array([[-1., -1, 0], #First polygon is just input plane
                                            [-1, 1, 0], 
                                            [1, 1, 0], 
                                            [1, -1, 0]])]

        for layer_id in range(len(model.model)//2): #Move polygont through layers     
            polygons[str(layer_id)+'.linear_out']=process_with_layers(model.model[:2*layer_id+1], polygons[str(layer_id-1)+'.new_tiling']) 

            #Split polygons w/ Relu and clip negative values to z=0
            polygons[str(layer_id)+'.split_polygons_nested']=split_polygons_with_relu_simple(polygons[str(layer_id)+'.linear_out']) #Triple nested list so we can simplify merging process layer. 
            polygons[str(layer_id)+'.split_polygons_nested_clipped'] = clip_polygons(polygons[str(layer_id)+'.split_polygons_nested'])
            #Merge zero regions
            polygons[str(layer_id)+'.split_polygons_merged'] = merge_zero_regions(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
            
            #Compute new tiling - general method with merging - should be more accurate but slow - buggy?
            # polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
            
            #Less general method - less accurate, faster, maybe less buggy
            # polygons[str(layer_id)+'.new_tiling_nested']=recompute_tiling(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
            polygons[str(layer_id)+'.new_tiling_nested']=recompute_tiling_polygonize(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
            polygons[str(layer_id)+'.new_tiling']=[item for sublist in polygons[str(layer_id)+'.new_tiling_nested'] for item in sublist]


            print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')
            #Optional filtering step
            #polygons[str(layer_id)+'.new_tiling'] = filter_small_polygons(polygons[str(layer_id)+'.new_tiling'], min_area=1e-5)
            #print(str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons remaining after filtering out small polygons')


        #Last linear layer & output
        polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
        intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
        my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)

        print(len(my_top_polygons), len(my_indicator))
        my_top_polygons, my_indicator=fill_gaps(my_top_polygons, my_indicator) #Will this help with my random gaps?
        print(len(my_top_polygons), len(my_indicator))


        if prev_layer_1_polygons is not None: 
            prev_layer_1_polygons=reorder_polygons_optimal(prev_layer_1_polygons, polygons['0.new_tiling'])
        else:
            prev_layer_1_polygons=polygons['0.new_tiling']
        layer_1_polygons_flat=manim_polygons_from_np_list(prev_layer_1_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6, stroke_width=0.6)

        if prev_layer_2_polygons is not None: 
            prev_layer_2_polygons=reorder_polygons_optimal(prev_layer_2_polygons, polygons['1.new_tiling'])
        else:
            prev_layer_2_polygons=polygons['1.new_tiling']         
        layer_2_polygons_flat=manim_polygons_from_np_list(prev_layer_2_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6, stroke_width=0.6)

        if prev_layer_3_polygons is not None: 
            prev_layer_3_polygons=reorder_polygons_optimal(prev_layer_3_polygons, polygons['2.new_tiling'])
        else:
            prev_layer_3_polygons=polygons['2.new_tiling']
        layer_3_polygons_flat=manim_polygons_from_np_list(prev_layer_3_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6, stroke_width=0.6)

        if prev_layer_4_polygons is not None: 
            prev_layer_4_polygons=reorder_polygons_optimal(prev_layer_4_polygons, polygons['3.new_tiling'])
        else:
            prev_layer_4_polygons=polygons['3.new_tiling']
        layer_4_polygons_flat=manim_polygons_from_np_list(prev_layer_4_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6, stroke_width=0.6)



        #Outputs surfaces
        groups_output=gx.Group()
        layer_idx=len(model.model)-1
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['3.linear_out'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=gx.Group(s, pgs)
            groups_output.add(g)


        #Output surfaces together
        group_combined_output=groups_output.copy()
        group_combined_output[0].set_color(BLUE)
        group_combined_output[1].set_color(YELLOW)

        top_polygons_vgroup=gx.VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=p_scaled[:,2]*adaptive_viz_scales[-1][0] #Flatten that shit!
            poly_3d = gx.Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=0.6)
            poly_3d.set_opacity(0.5)
            top_polygons_vgroup.add(poly_3d)

        lines=gx.VGroup()
        for loop in intersection_lines: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=4)
            lines.add(line)

        top_polygons_vgroup_flat=gx.VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=0 #p_scaled[:,2]*viz_scales[2] #Flatten that shit!
            poly_3d = gx.Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=0.6)
            poly_3d.set_opacity(0.5)
            top_polygons_vgroup_flat.add(poly_3d)

        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map_2=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/'+map_filename)
        set_surface_shading(flat_map_2, 0,0,0).set_opacity(0.8)

        lines_flat=gx.VGroup()
        for loop in intersection_lines: 
            # loop=np.hstack((loop, np.zeros((len(loop), 1))))
            loop[:,2]=0
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=4)
            lines_flat.add(line)    

        group_combined_output.set_opacity(0.3)
        final_map_group=gx.Group(flat_map_2, top_polygons_vgroup_flat, lines_flat)
        border_map_only=gx.Group(flat_map_2.copy(), lines_flat.copy())

        # I can change locations in premiere pretty easily if I need to!
        layer_1_polygons_flat.shift([ 0.,  2., 0.])                                                                                                                                                       
        layer_2_polygons_flat.shift([ 2.35 , 2.,   0.  ])                                                                                                                                                
        layer_3_polygons_flat.shift([ 0.,   -0.35, 0.  ])
        layer_4_polygons_flat.shift([ 2.35, -0.35, 0.  ])                                                                                                                                                
        final_map_group.shift([ 2*2.35, 2, 0.  ])
        border_map_only.shift([ 2*2.35, -0.35, 0.  ])

        # self.wait()
        # self.set_view(0, 0, 0, (3.94, 0.38, 0.0), 2.04) #Could do a nice zoom in animation from not sure if it's worth it or not
        # self.set_view(0, 0, 0, (3.07, 0.82, 0.0), 0.77)
        self.set_view(0, 0, 0, (2.97, 0.81, 0.0), 4.80)
        self.add(layer_1_polygons_flat)
        self.add(layer_2_polygons_flat)
        self.add(layer_3_polygons_flat)
        self.add(layer_4_polygons_flat)
        self.add(final_map_group)
        self.add(border_map_only)

        self.wait()

        #Some nice moves at the end here, and region count - at least on screen - maybe say it?

        self.play(self.animate_view(0, 0, 0, (4.77, -0.36, 0.0), 2.26), run_time=12, rate_func=gx.linear)
        self.wait()


        #man i keep moving like not quite on center!
        #Do i fade in the number of regions below here maybe?


        
        regions_label=markup_text('29,797 REGIONS', font_size=11, font='myriad-pro')
        regions_label.set_color(CHILL_BROWN)
        regions_label.move_to([2.35, -1.45, 0])

        self.add(regions_label)
        # self.remove(regions_label)

        self.play(self.animate_view(0, 0, 0, (2.36, -0.4, 0.0), 2.31), run_time=5)
        self.wait()


        # self.play(self.animate_view(0, 0, 0, (2.61, -0.35, 0.0), 2.26), run_time=5)
        # self.wait()
        self.remove(regions_label)
        self.play(self.animate_view(0, 0, 0, (2.37, 0.81, 0.0), 4.77), run_time=12)
        self.wait()


        # self.play(self.animate_view(0, 0, 0, (2.53, -0.45, 0.0), 0.52), run_time=12)
        # self.wait()

        # self.play(self.animate_view(0, 0, 0, (2.97, 0.81, 0.0), 4.80), run_time=12)
        # self.wait()

        # self.play(self.animate_view(0, 0, 0, (4.82, -0.38, 0.0), 0.80), run_time=12)
        # self.wait()

        # self.play(self.animate_view(0, 0, 0, (2.97, 0.81, 0.0), 4.80), run_time=12)
        # self.wait()

        # self.play(self.animate_view(0, 45, 0, (2.33, -0.1, -0.63), 2.02), run_time=12)
        # self.wait()

        # self.play(self.animate_view(0, 0, 0, (2.97, 0.81, 0.0), 4.80), run_time=12)
        # self.wait()

        # self.play(self.animate_view(35, 52, 0, (3.27, 0.42, -0.34), 4.85), run_time=12)
        # self.wait()

        # self.play(self.animate_view(0, 0, 0, (2.97, 0.81, 0.0), 4.80), run_time=12)
        # self.wait()


        # self.play(self.camera.frame.animate.)
        # self.wait()

#Maybe polygon count below each panel?
# Retiled plane into  265  polygons.
# Retiled plane into  1920  polygons.                                                                     
# /opt/anaconda3/lib/python3.11/site-packages/shapely/constructive.py:180: RuntimeWarning: divide by zero encountered in buffer
#   return lib.buffer(                                                                                    
# /opt/anaconda3/lib/python3.11/site-packages/shapely/constructive.py:180: RuntimeWarning: invalid value encountered in buffer                                                            
#   return lib.buffer(                                                                                    
# Retiled plane into  7687  polygons.                                             
# Retiled plane into  28397  polygons.                                                                    
# 29795 29795                                                                                             
# 29797 29797




        # self.remove(final_map_group)
        # self.add(flat_map_2)
        # self.add(lines_flat)


        self.wait(20)
        pass

