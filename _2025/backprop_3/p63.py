from welch_paths import welch_path
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
import copy
import manimgx as gx
import numpy as np
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
from plane_folding_utils import get_relu_joint, line_from_joint_points_1
from welch_scene import WelchScene
from welch_surface import parametric_surface, textured_surface
from polytope_intersection_utils import intersect_polytopes
from welch_surface import set_surface_shading
from tqdm import tqdm
from order_matching_tools import reorder_polygons_optimal, reorder_polygons_greedy

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'

graphics_dir=welch_path('backprop_3/graphics/') #Point to folder where map images are
colors = [BLUE, gx.GREY, GREEN, gx.TEAL, gx.PURPLE, gx.PINK, gx.TEAL, YELLOW, FRESH_TAN, CHILL_BLUE, CHILL_GREEN, YELLOW_FADE]
# colors = [GREY, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]

def create_first_layer_relu_groups(model, surfaces, num_neurons_first_layer=8, extent=1, vertical_spacing=0.9):
    """
    Creates ReLU joint groups for all neurons in the first layer.
    
    Args:
        model: PyTorch model containing the neural network
        surfaces: List of surface groups (surfaces[0] contains first layer surfaces)
        num_neurons_first_layer: Number of neurons in first layer (default 8)
        extent: Extent parameter for ReLU joint calculation (default 1)
        vertical_spacing: Vertical spacing between groups (default 0.3)
    
    Returns:
        Group: A manim Group containing num_neurons_first_layer groups, 
               each with a surface and its corresponding joint line
    """
    
    # Extract weights and biases from first layer (layer 0 in model.model)
    with torch.no_grad():
        w1 = model.model[0].weight.cpu().numpy()  # Shape: [num_neurons, 2]
        b1 = model.model[0].bias.cpu().numpy()    # Shape: [num_neurons]
    
    # Create the main group to hold all neuron groups
    all_relu_groups = gx.Group()
    
    # Calculate total height needed for centering
    total_height = (num_neurons_first_layer - 1) * vertical_spacing
    start_z = total_height / 2  # Start from top
    
    for neuron_idx in range(num_neurons_first_layer):
        # Get ReLU joint points for this neuron
        joint_points = get_relu_joint(w1[neuron_idx, 0], w1[neuron_idx, 1], b1[neuron_idx], extent=extent)
        
        # Create joint line from points
        joint_line = line_from_joint_points_1(joint_points)
        if joint_line is not None: 
            joint_line.set_opacity(0.9)
            neuron_group = gx.Group(surfaces[1][neuron_idx], joint_line)
        else:
            neuron_group = gx.Group(surfaces[1][neuron_idx])
        
        # Position the group vertically
        z_position = start_z - neuron_idx * vertical_spacing
        neuron_group.shift([0, 0, z_position])
        
        # Add to main group
        all_relu_groups.add(neuron_group)
    
    return all_relu_groups


class p63(WelchScene):
    '''
    Ok no need for animating this one, right? And just the flat panels and one 3d view right??
    '''
    def construct(self):

        model_path='_2025/backprop_3/models/16_16_16_1.pth'
        model = BaarleNet([16,16, 16])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.06, 0.06, 0.042, 0.042, 0.15]
        num_neurons=[16, 16, 16, 16, 16, 16, 2]
        vertical_spacing=1.0

        #Do i actually need
        w1=model.model[0].weight.detach().numpy()
        b1=model.model[0].bias.detach().numpy()
        w2=model.model[2].weight.detach().numpy()
        b2=model.model[2].bias.detach().numpy()
        w3=model.model[4].weight.detach().numpy()
        b3=model.model[4].bias.detach().numpy()
        w4=model.model[6].weight.detach().numpy()
        b4=model.model[6].bias.detach().numpy()

        adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=0.6, extent=1)
        #For the interesection to make sense, these scales need to match - either need to manual overide or chnage method above
        final_layer_viz=scale=1.4*min(adaptive_viz_scales[-1]) #little manual ramp here
        adaptive_viz_scales[-1]=[final_layer_viz, final_layer_viz]


        prev_layer_1_polygons=None
        prev_layer_2_polygons=None
        prev_layer_3_polygons=None

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx]) #viz_scales[layer_idx])
                bent_surface = parametric_surface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
                ts=textured_surface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
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


        if prev_layer_1_polygons is not None: 
            prev_layer_1_polygons=reorder_polygons_optimal(prev_layer_1_polygons, polygons['0.new_tiling'])
        else:
            prev_layer_1_polygons=polygons['0.new_tiling']
        layer_1_polygons_flat=manim_polygons_from_np_list(prev_layer_1_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6)

        if prev_layer_2_polygons is not None: 
            prev_layer_2_polygons=reorder_polygons_optimal(prev_layer_2_polygons, polygons['1.new_tiling'])
        else:
            prev_layer_2_polygons=polygons['1.new_tiling']         
        layer_2_polygons_flat=manim_polygons_from_np_list(prev_layer_2_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6)

        if prev_layer_3_polygons is not None: 
            prev_layer_3_polygons=reorder_polygons_optimal(prev_layer_3_polygons, polygons['2.new_tiling'])
        else:
            prev_layer_3_polygons=polygons['2.new_tiling']
        layer_3_polygons_flat=manim_polygons_from_np_list(prev_layer_3_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6)


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
                             stroke_width=2)
            poly_3d.set_opacity(0.5)
            top_polygons_vgroup.add(poly_3d)

        lines=gx.VGroup()
        for loop in intersection_lines: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=4)
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
                             stroke_width=2)
            poly_3d.set_opacity(0.5)
            top_polygons_vgroup_flat.add(poly_3d)

        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map_2=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(flat_map_2, 0,0,0).set_opacity(0.8)

        lines_flat=gx.VGroup()
        for loop in intersection_lines: 
            # loop=np.hstack((loop, np.zeros((len(loop), 1))))
            loop[:,2]=0
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=10)
            lines_flat.add(line)    

        group_combined_output.set_opacity(0.3)
        final_map_group=gx.Group(flat_map_2, top_polygons_vgroup_flat, lines_flat)

        layer_1_polygons_flat.shift([ 0.,  2., -5.])                                                                                                                                                       
        layer_2_polygons_flat.shift([ 2.35 , 2.,   -5.  ])                                                                                                                                                
        layer_3_polygons_flat.shift([ 0.,   -0.35, -5.  ])                                                                                                                                                
        final_map_group.shift([ 2.35, -0.35, -5.  ])


        self.wait()
        # self.set_view(0, 0, 0, (3.94, 0.38, 0.0), 2.04) #Could do a nice zoom in animation from not sure if it's worth it or not
        self.set_view(0, 0, 0, (3.07, 0.82, 0.0), 0.77)
        self.add(layer_1_polygons_flat)
        self.add(layer_2_polygons_flat)
        self.add(layer_3_polygons_flat)
        self.add(final_map_group)
        self.wait()


        #final_map_group
        #layer_1_polygons_flat
        #layer_2_polygons_flat
        # layer_3_polygons_flat



        self.wait(20)
        pass

class p63b(WelchScene):
    '''
    3d version, maybe add a little "rotation around" action that I can use
    '''
    def construct(self):

        model_path='_2025/backprop_3/models/16_16_16_1.pth'
        model = BaarleNet([16,16, 16])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.06, 0.06, 0.042, 0.042, 0.15]
        num_neurons=[16, 16, 16, 16, 16, 16, 2]
        vertical_spacing=1.0

        #Do i actually need
        w1=model.model[0].weight.detach().numpy()
        b1=model.model[0].bias.detach().numpy()
        w2=model.model[2].weight.detach().numpy()
        b2=model.model[2].bias.detach().numpy()
        w3=model.model[4].weight.detach().numpy()
        b3=model.model[4].bias.detach().numpy()
        w4=model.model[6].weight.detach().numpy()
        b4=model.model[6].bias.detach().numpy()

        adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=0.6, extent=1)
        #For the interesection to make sense, these scales need to match - either need to manual overide or chnage method above
        final_layer_viz=scale=1.8*min(adaptive_viz_scales[-1]) #little manual ramp here
        adaptive_viz_scales[-1]=[final_layer_viz, final_layer_viz]


        prev_layer_1_polygons=None
        prev_layer_2_polygons=None
        prev_layer_3_polygons=None

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=adaptive_viz_scales[-1][0]) #viz_scales[layer_idx])
                bent_surface = parametric_surface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
                ts=textured_surface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
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


        if prev_layer_1_polygons is not None: 
            prev_layer_1_polygons=reorder_polygons_optimal(prev_layer_1_polygons, polygons['0.new_tiling'])
        else:
            prev_layer_1_polygons=polygons['0.new_tiling']
        layer_1_polygons_flat=manim_polygons_from_np_list(prev_layer_1_polygons, colors=colors, viz_scale=adaptive_viz_scales[-1][0], opacity=0.6)

        if prev_layer_2_polygons is not None: 
            prev_layer_2_polygons=reorder_polygons_optimal(prev_layer_2_polygons, polygons['1.new_tiling'])
        else:
            prev_layer_2_polygons=polygons['1.new_tiling']         
        layer_2_polygons_flat=manim_polygons_from_np_list(prev_layer_2_polygons, colors=colors, viz_scale=adaptive_viz_scales[-1][0], opacity=0.6)

        if prev_layer_3_polygons is not None: 
            prev_layer_3_polygons=reorder_polygons_optimal(prev_layer_3_polygons, polygons['2.new_tiling'])
        else:
            prev_layer_3_polygons=polygons['2.new_tiling']
        layer_3_polygons_flat=manim_polygons_from_np_list(prev_layer_3_polygons, colors=colors, viz_scale=adaptive_viz_scales[-1][0], opacity=0.6)


        #Outputs surfaces
        groups_output=gx.Group()
        layer_idx=len(model.model)-1
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['3.linear_out'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[-1][0], opacity=0.6)
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
            p_scaled[:,2]=p_scaled[:,2]*adaptive_viz_scales[-1][0] 
            poly_3d = gx.Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=2)
            poly_3d.set_opacity(0.5)
            top_polygons_vgroup.add(poly_3d)

        lines=gx.VGroup()
        for loop in intersection_lines: 
            loop=loop*np.array([1, 1, adaptive_viz_scales[-1][0]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=4)
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
                             stroke_width=2)
            poly_3d.set_opacity(0.5)
            top_polygons_vgroup_flat.add(poly_3d)

        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map_2=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(flat_map_2, 0,0,0).set_opacity(0.8)

        lines_flat=gx.VGroup()
        for loop in intersection_lines: 
            # loop=np.hstack((loop, np.zeros((len(loop), 1))))
            loop[:,2]=0
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=8)
            lines_flat.add(line)    

        group_combined_output.set_opacity(0.3)
        final_map_group=gx.Group(flat_map_2, top_polygons_vgroup_flat, lines_flat)

        # layer_1_polygons_flat.shift([ 0.,  2., -5.])                                                                                                                                                       
        # layer_2_polygons_flat.shift([ 2.35 , 2.,   -5.  ])                                                                                                                                                
        # layer_3_polygons_flat.shift([ 0.,   -0.35, -5.  ])                                                                                                                                                
        # final_map_group.shift([ 2.35, -0.35, -5.  ])


        # self.wait()

        # group_combined_output.set_opacity(0.3)
        # top_polygons_vgroup.set_opacity(0.6)
        
        # self.remove(group_combined_output)

        combined_3d_group=gx.Group(group_combined_output, top_polygons_vgroup, lines)


        self.wait()
        self.add(combined_3d_group)

        self.set_view(-46, 60, 0, (-0.05, -0.03, -0.3), 3.23)
        self.wait()
        self.play(self.animate_view(20, 44, 0, (-0.01, -0.1, -0.38), 3.23), run_time=10) #optional slow pan around that I can usssse
        self.wait()







        # self.set_view(0, 0, 0, (3.94, 0.38, 0.0), 2.04) #Could do a nice zoom in animation from not sure if it's worth it or not
        # self.set_view(0, 0, 0, (3.07, 0.82, 0.0), 0.77)
        # self.add(layer_1_polygons_flat)
        # self.add(layer_2_polygons_flat)
        # self.add(layer_3_polygons_flat)
        # self.add(final_map_group)
        # self.wait()



        #final_map_group
        #layer_1_polygons_flate
        #layer_2_polygons_flat
        # layer_3_polygons_flat



        self.wait(20)
        pass




















    