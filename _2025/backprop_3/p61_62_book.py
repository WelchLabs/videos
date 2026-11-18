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
    recompute_tiling_general,
    recompute_tiling_polygonize,
    split_polygons_with_relu_simple,
)
from plane_folding_utils import get_relu_joint, line_from_joint_points_1
from welch_scene import WelchScene
from welch_surface import parametric_surface, textured_surface
from polytope_intersection_utils import intersect_polytopes
from welch_surface import set_surface_shading
from gap_filler import fill_gaps
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

# graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/' #Point to folder where map images are
graphics_dir=welch_path('ai_book/4_deep_learning/graphics/')
map_filename='baarle_hertog_maps-13.png'

colors = [BLUE, gx.GREY, GREEN, gx.TEAL, gx.PURPLE, gx.PINK, gx.TEAL, YELLOW, FRESH_TAN, CHILL_BLUE, CHILL_GREEN, YELLOW_FADE]
# colors_old = [BLUE, GREY, GREEN, TEAL, PURPLE, ORANGE, PINK, TEAL, RED, YELLOW ]
colors_old = [gx.GREY, BLUE, GREEN, YELLOW, gx.PURPLE, gx.ORANGE, gx.PINK, gx.TEAL]

def create_first_layer_relu_groups(model, surfaces, num_neurons_first_layer=8, extent=1, vertical_spacing=0.9, line_color='#FFFFFF'):
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
        joint_line = line_from_joint_points_1(joint_points, color=colors[neuron_idx%len(colors)])
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


class p61e(WelchScene):
    def construct(self):

        model_path='_2025/backprop_3/models/8_8_1.pth'
        model = BaarleNet([8,8])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.06, 0.06, 0.042, 0.042, 0.15]
        num_neurons=[8, 8, 8, 8, 2]
        vertical_spacing=1.0

        w1=model.model[0].weight.detach().numpy()
        b1=model.model[0].bias.detach().numpy()
        w2=model.model[2].weight.detach().numpy()
        b2=model.model[2].bias.detach().numpy()

        adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=1.0, extent=1)
        #For the interesection to make sense, these scales need to match - either need to manual overide or chnage method above
        final_layer_viz=scale=2*min(adaptive_viz_scales[-1]) #little manual ramp here
        adaptive_viz_scales[-1]=[final_layer_viz, final_layer_viz]

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
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
            #Compute new tiling
            polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
            print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')

            #Optional filtering step
            #polygons[str(layer_id)+'.new_tiling'] = filter_small_polygons(polygons[str(layer_id)+'.new_tiling'], min_area=1e-5)
            #print(str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons remaining after filtering out small polygons')

        #Last linear layer & output
        polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
        intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
        my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)


        groups_1=create_first_layer_relu_groups(model, surfaces, num_neurons_first_layer=8, extent=1, vertical_spacing=vertical_spacing)


        #Ok let's get everything (including shadows) up first, and then decide if I want to animate the creation of anything?

        layer_1_polygons_flat=manim_polygons_from_np_list(polygons['0.new_tiling'], colors=colors_old, viz_scale=viz_scales[2], opacity=0.6)
        layer_1_polygons_flat.shift([0, 0, -5.0])

        #Create layer 2 polygons -> I guess post cut right?
        groups_2=gx.Group()
        layer_idx=3
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['1.split_polygons_merged'][neuron_idx], colors=colors_old, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=gx.Group(s, pgs[1:]) #Crazy to leave off first merged/flat group here?
            g.shift([3*(layer_idx-1)/2, 0, start_z - neuron_idx * vertical_spacing])
            groups_2.add(g)


        layer_2_polygons_flat=manim_polygons_from_np_list(polygons['1.new_tiling'], colors=colors_old, viz_scale=viz_scales[2], opacity=0.6)
        layer_2_polygons_flat.shift([3, 0, -5.0])

        #Outputs surfaces
        groups_output=gx.Group()
        layer_idx=len(model.model)-1
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['2.linear_out'][neuron_idx], colors=colors_old, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=gx.Group(s, pgs) #[1:]) #Crazy to leave off first merged/flat group here?
            g.shift([6, 0, start_z - neuron_idx * vertical_spacing])
            groups_output.add(g)

        #Output surfaces together
        group_combined_output=groups_output.copy()
        group_combined_output[0].set_color(BLUE)
        group_combined_output[1].set_color(YELLOW)
        group_combined_output[0].shift([3, 0, -vertical_spacing/2])
        group_combined_output[1].shift([3, 0, vertical_spacing/2])

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
            poly_3d.shift([9, 0, 0])
            top_polygons_vgroup.add(poly_3d)

        # loops=order_closed_loops_with_closure(intersection_lines)
        lines=gx.VGroup()
        for loop in intersection_lines: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=3)
            lines.add(line)
        lines.shift([9, 0, 0.])

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
            poly_3d.shift([9, 0, -2])
            top_polygons_vgroup_flat.add(poly_3d)

        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map_2=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/'+map_filename)
        set_surface_shading(flat_map_2, 0,0,0).set_opacity(0.8)
        flat_map_2.shift([9, 0, -2])

        lines_flat=gx.VGroup()
        for loop in intersection_lines: 
            # loop=np.hstack((loop, np.zeros((len(loop), 1))))
            loop[:,2]=0
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=3)
            lines_flat.add(line)
        lines_flat.shift([9, 0, -2])    

        # surface_51.set_opacity(0.2)
        # surface_52.set_opacity(0.9)
        # polygons_51.set_opacity(0.4)
        # polygons_52.set_opacity(0.5)



        # polygons_41=manim_polygons_from_np_list(polygons['2.linear_out'][0], colors=colors_old_5, viz_scale=viz_scales[4], opacity=0.6)
        # polygons_41.shift([6, 0, 1.501]) #Move slightly above map

        ##Feeling like a fly around of this static scene would be non-terrible?

        group_combined_output.set_opacity(0.3)
        # top_polygons_vgroup.set_opacity(0.6)

        self.wait()
        self.set_view(0, 64, 0, (4.62, 2.85, -1.51), 12.99)

        self.add(groups_1)
        self.add(layer_1_polygons_flat)
        self.add(groups_2)
        self.add(layer_2_polygons_flat)
        self.add(groups_output)
        self.add(group_combined_output)
        self.add(top_polygons_vgroup)
        self.add(lines)
        self.add(flat_map_2)
        self.add(top_polygons_vgroup_flat)
        self.add(lines_flat)

        self.wait()

        #Book here-ish? 
        self.set_view(0, 57, 0, (4.53, 2.72, -1.76), 12.04)
        self.wait()

        self.remove(groups_1)
        self.remove(groups_2)
        self.remove(groups_output)
        self.remove(group_combined_output)
        self.remove(top_polygons_vgroup)
        self.remove(lines)

        self.remove(layer_2_polygons_flat)


        layer_1_polygons_flat.move_to(gx.ORIGIN)
        self.set_view(0, 0, 0, (0.05, -0.04, 0.0), 2.37)
        self.wait()
        self.remove(layer_1_polygons_flat)

        self.add(layer_2_polygons_flat)
        layer_2_polygons_flat.move_to(gx.ORIGIN)
        self.wait()

        self.remove(layer_2_polygons_flat)

        lines_flat.set_stroke(width=8)
        final_group=gx.Group(flat_map_2, top_polygons_vgroup_flat, lines_flat)
        final_group.move_to(gx.ORIGIN)
        self.wait()
        self.remove(final_group)

        self.add(groups_2[1])
        self.set_view(0, 49, 0, (3.01, 3.36, -0.22), 6.18)
        self.wait()



        pass



        #19, 102

        #Focus on tiling
        # self.play(self.animate_view(0, 58, 0, (1.45, 1.08, -5.07), 5.78), run_time=6)
        # self.wait()

        # self.play(self.animate_view(18, 58, 0, (2.47, 1.79, 0.9), 5.81),
        #           groups_1.animate.set_opacity(0.1), 
        #           groups_2[0].animate.set_opacity(0.1),
        #           groups_2[2:].animate.set_opacity(0.1),
        #           groups_output.animate.set_opacity(0.1), 
        #           run_time=6
        #           )
        # self.wait()

        # self.play(self.animate_view(-4, 58, 0, (7.86, 1.54, -1.02), 6.99),
        #           groups_1.animate.set_opacity(0.6), 
        #           groups_2[0].animate.set_opacity(0.6),
        #           groups_2[2:].animate.set_opacity(0.6),
        #           groups_output.animate.set_opacity(0.6), 
        #           run_time=6
        #           )
        # self.wait()

        # self.play(self.animate_view(0, 38, 0, (9.09, 0.29, -2.69), 3.98),
        #          group_combined_output.animate.set_opacity(0.05),
        #          top_polygons_vgroup.animate.set_opacity(0.05),
        #          lines.animate.set_opacity(0.05),
        #          run_time=4)
        # self.wait()

        # #Overall view
        # self.play(self.animate_view(0, 74, 0, (5.17, 1.54, -0.87), 11.81),
        #          group_combined_output.animate.set_opacity(0.2),
        #          top_polygons_vgroup.animate.set_opacity(0.5),
        #          lines.animate.set_opacity(0.8),
        #          run_time=6)
        # self.wait()

        # #One more little zoom in at the end on the dead neurons
        # self.play(self.animate_view(-36, 72, 0, (3.44, 1.56, 0.35), 6.39), run_time=6)
        # self.wait()

        # self.play(self.animate_view(36, 70, 0, (3.21, 1.43, 0.42), 6.39), run_time=12, rate_func=linear)
        # self.wait()

        # # Then back to wide shot in case I need it. 
        # self.play(self.animate_view(0, 74, 0, (5.17, 1.54, -0.87), 11.81), run_time=6)



        self.wait(20)
        pass




class p62(WelchScene):
    def construct(self):

        model_path='_2025/backprop_3/models/8_8_8_1.pth'
        model = BaarleNet([8,8, 8])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.06, 0.06, 0.042, 0.042, 0.042, 0.042, 0.15]
        num_neurons=[8, 8, 8, 8, 8, 8, 2]
        vertical_spacing=1.0

        w1=model.model[0].weight.detach().numpy()
        b1=model.model[0].bias.detach().numpy()
        w2=model.model[2].weight.detach().numpy()
        b2=model.model[2].bias.detach().numpy()

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
            #Compute new tiling
            polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
            print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')

            #Optional filtering step
            #polygons[str(layer_id)+'.new_tiling'] = filter_small_polygons(polygons[str(layer_id)+'.new_tiling'], min_area=1e-5)
            #print(str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons remaining after filtering out small polygons')

        #Last linear layer & output
        polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
        intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
        my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)


        groups_1=create_first_layer_relu_groups(model, surfaces, num_neurons_first_layer=8, extent=1, vertical_spacing=vertical_spacing)


        #Ok let's get everything (including shadows) up first, and then decide if I want to animate the creation of anything?

        layer_1_polygons_flat=manim_polygons_from_np_list(polygons['0.new_tiling'], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        layer_1_polygons_flat.shift([0, 0, -5.0])

        #Create layer 2 polygons -> I guess post cut right?
        groups_2=gx.Group()
        layer_idx=3
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['1.split_polygons_merged'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=gx.Group(s, pgs[1:]) #Crazy to leave off first merged/flat group here?
            g.shift([3*(layer_idx-1)/2, 0, start_z - neuron_idx * vertical_spacing])
            groups_2.add(g)


        layer_2_polygons_flat=manim_polygons_from_np_list(polygons['1.new_tiling'], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        layer_2_polygons_flat.shift([3, 0, -5.0])



        groups_3=gx.Group()
        layer_idx=5
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['2.split_polygons_merged'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=gx.Group(s, pgs[1:]) #Crazy to leave off first merged/flat group here?
            g.shift([3*(layer_idx-1)/2, 0, start_z - neuron_idx * vertical_spacing])
            groups_3.add(g)

        layer_3_polygons_flat=manim_polygons_from_np_list(polygons['2.new_tiling'], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        layer_3_polygons_flat.shift([6, 0, -5.0])




        #Outputs surfaces
        output_horizontal_offset=9
        groups_output=gx.Group()
        layer_idx=len(model.model)-1
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['3.linear_out'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=gx.Group(s, pgs) #[1:]) #Crazy to leave off first merged/flat group here?
            g.shift([output_horizontal_offset, 0, start_z - neuron_idx * vertical_spacing])
            groups_output.add(g)

        #Output surfaces together
        group_combined_output=groups_output.copy()
        group_combined_output[0].set_color(BLUE)
        group_combined_output[1].set_color(YELLOW)
        group_combined_output[0].shift([3, 0, -vertical_spacing/2])
        group_combined_output[1].shift([3, 0, vertical_spacing/2])

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
            poly_3d.shift([output_horizontal_offset+3, 0, 0])
            top_polygons_vgroup.add(poly_3d)

        # loops=order_closed_loops_with_closure(intersection_lines)
        lines=gx.VGroup()
        for loop in intersection_lines: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=4)
            lines.add(line)
        lines.shift([output_horizontal_offset+3, 0, 0.])

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
            poly_3d.shift([output_horizontal_offset+3, 0, -2])
            top_polygons_vgroup_flat.add(poly_3d)

        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map_2=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/'+map_filename)
        set_surface_shading(flat_map_2, 0,0,0).set_opacity(0.8)
        flat_map_2.shift([output_horizontal_offset+3, 0, -2])

        lines_flat=gx.VGroup()
        for loop in intersection_lines: 
            # loop=np.hstack((loop, np.zeros((len(loop), 1))))
            loop[:,2]=0
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=5)
            lines_flat.add(line)
        lines_flat.shift([output_horizontal_offset+3, 0, -2])    


        group_combined_output.set_opacity(0.3)
        # top_polygons_vgroup.set_opacity(0.6)

        self.wait()
        self.set_view(0, 65, 0, (5.72, 2.88, -1.46), 12.99)

        self.add(groups_1)
        self.add(layer_1_polygons_flat)
        self.add(groups_2)
        self.add(layer_2_polygons_flat)
        self.add(groups_3)
        self.add(layer_3_polygons_flat)
        self.add(groups_output)
        self.add(group_combined_output)
        self.add(top_polygons_vgroup)
        self.add(lines)
        self.add(flat_map_2)
        self.add(top_polygons_vgroup_flat)
        self.add(lines_flat)

        self.wait()


        #Book here-ish? 
        self.set_view(0, 57, 0, (4.53, 2.72, -1.76), 12.04)
        self.wait()

        self.remove(groups_1)
        self.remove(groups_2)
        self.remove(groups_output)
        self.remove(group_combined_output)
        self.remove(top_polygons_vgroup)
        self.remove(lines)

        self.remove(layer_2_polygons_flat)


        layer_1_polygons_flat.move_to(gx.ORIGIN)
        self.set_view(0, 0, 0, (0.05, -0.04, 0.0), 2.37)
        self.wait()
        self.remove(layer_1_polygons_flat)

        self.add(layer_2_polygons_flat)
        layer_2_polygons_flat.move_to(gx.ORIGIN)
        self.wait()

        self.remove(layer_2_polygons_flat)

        lines_flat.set_stroke(width=8)
        final_group=gx.Group(flat_map_2, top_polygons_vgroup_flat, lines_flat)
        final_group.move_to(gx.ORIGIN)
        self.wait()
        self.remove(final_group)

        self.add(groups_2[1])
        self.set_view(0, 49, 0, (3.01, 3.36, -0.22), 6.18)
        self.wait()



        pass


        #20, 119, 430

  
        # self.play(self.animate_view(0, 58, 0, (1.45, 1.08, -5.07), 5.78), run_time=6)
        # self.wait()


        self.play(self.animate_view(-42, 56, 0, (6.58, 0.65, 0.58), 3.23),
                  groups_3[:2].animate.set_opacity(0.05), 
                  groups_3[3:].animate.set_opacity(0.05), 
                  groups_2.animate.set_opacity(0.05), 
                  groups_output.animate.set_opacity(0.05), 
                  group_combined_output.animate.set_opacity(0.05), 
                  top_polygons_vgroup.animate.set_opacity(0.05),
                  lines.animate.set_opacity(0.05), 
                  flat_map_2.animate.set_opacity(0.05), 
                  top_polygons_vgroup_flat.animate.set_opacity(0.05), 
                  lines_flat.animate.set_opacity(0.05), 
                  run_time=6)
        self.wait()
        

        self.play(self.animate_view(-69, 56, 0, (6.83, 0.24, 0.61), 3.23), run_time=5) #little peak around the side
        self.wait()

        self.play(self.animate_view(0, 65, 0, (5.72, 2.88, -1.46), 12.99), #Back to wide view!
                  groups_3[:2].animate.set_opacity(0.6), 
                  groups_3[3:].animate.set_opacity(0.6), 
                  groups_2.animate.set_opacity(0.6), 
                  groups_output.animate.set_opacity(0.6), 
                  group_combined_output.animate.set_opacity(0.2), 
                  top_polygons_vgroup.animate.set_opacity(0.6),
                  lines.animate.set_opacity(0.8), 
                  flat_map_2.animate.set_opacity(0.5), 
                  top_polygons_vgroup_flat.animate.set_opacity(0.5), 
                  lines_flat.animate.set_opacity(0.8), 
                  run_time=6)
        self.wait()

        #Ok I'm losing the map -> I think I'll do a separate scene (kidna clunky but fine I think) for this zoom out. 

        # Ok so now I want to move to a nice 2d panel view, going to get into a little pickle with wanting
        # a 3d view and d flat view
        # One thing I could try is actually doing 2 separate transitions, and then blending them in premiere? 
        # That might now be terrible - let's try it. 
        #First a fade out transition here, and then maybe two separate classes, one of each flavof of transition. 

        self.play(
                  groups_1.animate.set_opacity(0), 
                  groups_2.animate.set_opacity(0), 
                  groups_3.animate.set_opacity(0), 
                  groups_output.animate.set_opacity(0), 
                  # group_combined_output.animate.set_opacity(0), 
                  # top_polygons_vgroup.animate.set_opacity(0),
                  # lines.animate.set_opacity(0), 
                  # flat_map_2.animate.set_opacity(0), 
                  # top_polygons_vgroup_flat.animate.set_opacity(0), 
                  # lines_flat.animate.set_opacity(0), 
                  run_time=3)
        self.wait()

        #Ok, maybe let's actually try the 2d/flat transition here?
        self.play(group_combined_output.animate.set_opacity(0), 
                    top_polygons_vgroup.animate.set_opacity(0),
                    lines.animate.set_opacity(0), 
                    run_time=3)
        self.wait()

        final_map_group=gx.Group(flat_map_2, top_polygons_vgroup_flat, lines_flat)


        # layer_1_polygons_flat.shift([0, 2, 0])
        # layer_2_polygons_flat.shift([-0.65, 2, 0])
        # layer_3_polygons_flat.shift([-6, 0.5-0.85, 0])
        # final_map_group.shift([-12+2.5-0.15, -0.35, -3])
        # self.set_view(0, 0, 0, (3.94, 0.38, 0.0), 2.04)

        self.wait()

        # layer_1_polygons_flat.set_opacity(0.8)
        # layer_2_polygons_flat.set_opacity(0.8)
        # layer_3_polygons_flat.set_opacity(0.8)
        # final_map_group.set_opacity(0.7)

        self.play(
                layer_1_polygons_flat.animate.shift([0, 2, 0]).set_opacity(0.65), #May be an opacity mismatch to the training scene?
                layer_2_polygons_flat.animate.shift([-0.65, 2, 0]).set_opacity(0.65),
                layer_3_polygons_flat.animate.shift([-6, 0.5-0.85, 0]).set_opacity(0.65),
                final_map_group.animate.shift([-12+2.5-0.15, -0.35, -3]).set_opacity(0.6),
                self.animate_view(0, 0, 0, (3.94, 0.38, 0.0), 2.04),
                run_time=6)
        self.wait()



        self.wait(20)
        pass



class p62b(WelchScene):
    def construct(self):

        model_path='_2025/backprop_3/models/8_8_8_1.pth'
        model = BaarleNet([8,8, 8])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.06, 0.06, 0.042, 0.042, 0.042, 0.042, 0.15]
        num_neurons=[8, 8, 8, 8, 8, 8, 2]
        vertical_spacing=1.0

        w1=model.model[0].weight.detach().numpy()
        b1=model.model[0].bias.detach().numpy()
        w2=model.model[2].weight.detach().numpy()
        b2=model.model[2].bias.detach().numpy()

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
            #Compute new tiling
            polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
            print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')

            #Optional filtering step
            #polygons[str(layer_id)+'.new_tiling'] = filter_small_polygons(polygons[str(layer_id)+'.new_tiling'], min_area=1e-5)
            #print(str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons remaining after filtering out small polygons')

        #Last linear layer & output
        polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
        intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
        my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)


        groups_1=create_first_layer_relu_groups(model, surfaces, num_neurons_first_layer=8, extent=1, vertical_spacing=vertical_spacing)


        #Ok let's get everything (including shadows) up first, and then decide if I want to animate the creation of anything?

        layer_1_polygons_flat=manim_polygons_from_np_list(polygons['0.new_tiling'], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        layer_1_polygons_flat.shift([0, 0, -5.0])

        #Create layer 2 polygons -> I guess post cut right?
        groups_2=gx.Group()
        layer_idx=3
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['1.split_polygons_merged'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=gx.Group(s, pgs[1:]) #Crazy to leave off first merged/flat group here?
            g.shift([3*(layer_idx-1)/2, 0, start_z - neuron_idx * vertical_spacing])
            groups_2.add(g)


        layer_2_polygons_flat=manim_polygons_from_np_list(polygons['1.new_tiling'], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        layer_2_polygons_flat.shift([3, 0, -5.0])



        groups_3=gx.Group()
        layer_idx=5
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['2.split_polygons_merged'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=gx.Group(s, pgs[1:]) #Crazy to leave off first merged/flat group here?
            g.shift([3*(layer_idx-1)/2, 0, start_z - neuron_idx * vertical_spacing])
            groups_3.add(g)

        layer_3_polygons_flat=manim_polygons_from_np_list(polygons['2.new_tiling'], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        layer_3_polygons_flat.shift([6, 0, -5.0])




        #Outputs surfaces
        output_horizontal_offset=9
        groups_output=gx.Group()
        layer_idx=len(model.model)-1
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['3.linear_out'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=gx.Group(s, pgs) #[1:]) #Crazy to leave off first merged/flat group here?
            g.shift([output_horizontal_offset, 0, start_z - neuron_idx * vertical_spacing])
            groups_output.add(g)

        #Output surfaces together
        group_combined_output=groups_output.copy()
        group_combined_output[0].set_color(BLUE)
        group_combined_output[1].set_color(YELLOW)
        group_combined_output[0].shift([3, 0, -vertical_spacing/2])
        group_combined_output[1].shift([3, 0, vertical_spacing/2])

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
            poly_3d.shift([output_horizontal_offset+3, 0, 0])
            top_polygons_vgroup.add(poly_3d)

        # loops=order_closed_loops_with_closure(intersection_lines)
        lines=gx.VGroup()
        for loop in intersection_lines: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=2)
            lines.add(line)
        lines.shift([output_horizontal_offset+3, 0, 0.])

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
            poly_3d.shift([output_horizontal_offset+3, 0, -2])
            top_polygons_vgroup_flat.add(poly_3d)

        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map_2=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/'+map_filename)
        set_surface_shading(flat_map_2, 0,0,0).set_opacity(0.8)
        flat_map_2.shift([output_horizontal_offset+3, 0, -2])

        lines_flat=gx.VGroup()
        for loop in intersection_lines: 
            # loop=np.hstack((loop, np.zeros((len(loop), 1))))
            loop[:,2]=0
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=2)
            lines_flat.add(line)
        lines_flat.shift([output_horizontal_offset+3, 0, -2])    


        group_combined_output.set_opacity(0.3)
        # top_polygons_vgroup.set_opacity(0.6)

        self.wait()
        self.set_view(0, 65, 0, (5.72, 2.88, -1.46), 12.99)


        self.add(groups_1)
        self.add(layer_1_polygons_flat)
        self.add(groups_2)
        self.add(layer_2_polygons_flat)
        self.add(groups_3)
        self.add(layer_3_polygons_flat)
        self.add(groups_output)
        self.add(group_combined_output)
        self.add(top_polygons_vgroup)
        self.add(lines)
        self.add(flat_map_2)
        self.add(top_polygons_vgroup_flat)
        self.add(lines_flat)



        #Book here-ish? 
        self.set_view(0, 57, 0, (5.41, 3.3, -2.33), 13.12)
        self.wait()

        self.remove(groups_1)
        self.remove(groups_2)
        self.remove(groups_3)
        self.remove(groups_output)
        self.remove(group_combined_output)
        self.remove(top_polygons_vgroup)
        self.remove(lines)

        self.remove(layer_2_polygons_flat)
        self.remove(layer_3_polygons_flat)


        layer_1_polygons_flat.move_to(gx.ORIGIN)
        self.set_view(0, 0, 0, (0.05, -0.04, 0.0), 2.37)
        self.wait()
        self.remove(layer_1_polygons_flat)

        self.add(layer_2_polygons_flat)
        layer_2_polygons_flat.move_to(gx.ORIGIN)
        self.wait()
        self.remove(layer_2_polygons_flat)

        self.add(layer_3_polygons_flat)
        layer_3_polygons_flat.move_to(gx.ORIGIN)
        self.wait()
        self.remove(layer_3_polygons_flat)

        lines_flat.set_stroke(width=8)
        final_group=gx.Group(flat_map_2, top_polygons_vgroup_flat, lines_flat)
        final_group.move_to(gx.ORIGIN)
        self.wait()
        self.remove(final_group)

        self.add(groups_2[1])
        self.set_view(0, 49, 0, (3.01, 3.36, -0.22), 6.18)
        self.wait()





        self.wait()

        self.play(
                  groups_1.animate.set_opacity(0), 
                  groups_2.animate.set_opacity(0), 
                  groups_3.animate.set_opacity(0), 
                  groups_output.animate.set_opacity(0), 
                  # group_combined_output.animate.set_opacity(0), 
                  # top_polygons_vgroup.animate.set_opacity(0),
                  # lines.animate.set_opacity(0), 
                  # flat_map_2.animate.set_opacity(0), 
                  # top_polygons_vgroup_flat.animate.set_opacity(0), 
                  # lines_flat.animate.set_opacity(0), 
                  run_time=3)
        self.wait()

        self.play(group_combined_output.animate.set_opacity(0), 
                    top_polygons_vgroup.animate.set_opacity(0),
                    lines.animate.set_opacity(0), 
                    run_time=3)
        self.wait()

        final_map_group=gx.Group(flat_map_2, top_polygons_vgroup_flat, lines_flat)

        self.remove(groups_1, groups_2, groups_3, groups_output)
        flat_map_2.set_opacity(1.0)
        top_polygons_vgroup_flat.set_opacity(0.45)
        # lines.set_stroke(width=20)


        self.wait()
        self.play(
                layer_1_polygons_flat.animate.shift([0, 2, 0]), 
                layer_2_polygons_flat.animate.shift([-0.65, 2, 0]),
                layer_3_polygons_flat.animate.shift([-6, 0.5-0.85, 0]),
                final_map_group.animate.shift([-12+2.5-0.15, -0.35, -3]),
                self.animate_view(0, 0, 0, (3.94, 0.38, 0.0), 2.04),
                run_time=6)
        self.wait()

        # self.add(flat_map_2)
        # self.remove(flat_map_2)
        # self.remove(top_polygons_vgroup_flat)
        # self.remove(lines_flat)

        self.wait(20)
        pass






class p62c2(WelchScene):
    '''Ok so this is the transition of my 3d shape for the 2d view -> try blending in premiere!'''
    def construct(self):

        model_path='_2025/backprop_3/models/8_8_8_1.pth'
        model = BaarleNet([8,8, 8])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.06, 0.06, 0.042, 0.042, 0.042, 0.042, 0.15]
        num_neurons=[8, 8, 8, 8, 8, 8, 2]
        vertical_spacing=1.0

        w1=model.model[0].weight.detach().numpy()
        b1=model.model[0].bias.detach().numpy()
        w2=model.model[2].weight.detach().numpy()
        b2=model.model[2].bias.detach().numpy()

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
            #Compute new tiling
            polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
            print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')

            #Optional filtering step
            #polygons[str(layer_id)+'.new_tiling'] = filter_small_polygons(polygons[str(layer_id)+'.new_tiling'], min_area=1e-5)
            #print(str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons remaining after filtering out small polygons')

        #Last linear layer & output
        polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
        intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
        my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)


        groups_1=create_first_layer_relu_groups(model, surfaces, num_neurons_first_layer=8, extent=1, vertical_spacing=vertical_spacing)


        #Ok let's get everything (including shadows) up first, and then decide if I want to animate the creation of anything?

        layer_1_polygons_flat=manim_polygons_from_np_list(polygons['0.new_tiling'], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        layer_1_polygons_flat.shift([0, 0, -5.0])

        #Create layer 2 polygons -> I guess post cut right?
        groups_2=gx.Group()
        layer_idx=3
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['1.split_polygons_merged'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=gx.Group(s, pgs[1:]) #Crazy to leave off first merged/flat group here?
            g.shift([3*(layer_idx-1)/2, 0, start_z - neuron_idx * vertical_spacing])
            groups_2.add(g)


        layer_2_polygons_flat=manim_polygons_from_np_list(polygons['1.new_tiling'], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        layer_2_polygons_flat.shift([3, 0, -5.0])



        groups_3=gx.Group()
        layer_idx=5
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['2.split_polygons_merged'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=gx.Group(s, pgs[1:]) #Crazy to leave off first merged/flat group here?
            g.shift([3*(layer_idx-1)/2, 0, start_z - neuron_idx * vertical_spacing])
            groups_3.add(g)

        layer_3_polygons_flat=manim_polygons_from_np_list(polygons['2.new_tiling'], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        layer_3_polygons_flat.shift([6, 0, -5.0])




        #Outputs surfaces
        output_horizontal_offset=9
        groups_output=gx.Group()
        layer_idx=len(model.model)-1
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['3.linear_out'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=gx.Group(s, pgs) #[1:]) #Crazy to leave off first merged/flat group here?
            g.shift([output_horizontal_offset, 0, start_z - neuron_idx * vertical_spacing])
            groups_output.add(g)

        #Output surfaces together
        group_combined_output=groups_output.copy()
        group_combined_output[0].set_color(BLUE)
        group_combined_output[1].set_color(YELLOW)
        group_combined_output[0].shift([3, 0, -vertical_spacing/2])
        group_combined_output[1].shift([3, 0, vertical_spacing/2])

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
            poly_3d.shift([output_horizontal_offset+3, 0, 0])
            top_polygons_vgroup.add(poly_3d)

        # loops=order_closed_loops_with_closure(intersection_lines)
        lines=gx.VGroup()
        for loop in intersection_lines: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=4)
            lines.add(line)
        lines.shift([output_horizontal_offset+3, 0, 0.])

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
            poly_3d.shift([output_horizontal_offset+3, 0, -2])
            top_polygons_vgroup_flat.add(poly_3d)

        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map_2=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/'+map_filename)
        set_surface_shading(flat_map_2, 0,0,0).set_opacity(0.8)
        flat_map_2.shift([output_horizontal_offset+3, 0, -2])

        lines_flat=gx.VGroup()
        for loop in intersection_lines: 
            # loop=np.hstack((loop, np.zeros((len(loop), 1))))
            loop[:,2]=0
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=5)
            lines_flat.add(line)
        lines_flat.shift([output_horizontal_offset+3, 0, -2])    


        group_combined_output.set_opacity(0.3)
        # top_polygons_vgroup.set_opacity(0.6)

        self.wait()
        self.set_view(0, 65, 0, (5.72, 2.88, -1.46), 12.99)

        combined_3d_group=gx.Group(group_combined_output, top_polygons_vgroup, lines)
        self.add(combined_3d_group)

        # self.add(groups_1)
        # self.add(layer_1_polygons_flat)
        # self.add(groups_2)
        # self.add(layer_2_polygons_flat)
        # self.add(groups_3)
        # self.add(layer_3_polygons_flat)
        # self.add(groups_output)
        # self.add(group_combined_output)
        # self.add(top_polygons_vgroup)
        # self.add(lines)
        # self.add(flat_map_2)
        # self.add(top_polygons_vgroup_flat)
        # self.add(lines_flat)

        self.wait()
        self.play(self.animate_view(48, 50, 0, (-0.04, 0.13, -0.47), 3.99), #Kinda chill isometricish
                 combined_3d_group.animate.move_to(gx.ORIGIN),
                 run_time=6, 
                 rate_func=gx.linear
              )
        self.wait()
        self.wait(20)
        pass


# class p62d_debug(InteractiveScene):
#     '''Ok so this is the transition of my 3d shape for the 2d view -> try blending in premiere!'''
#     def construct(self):

#         model = BaarleNet([8, 8, 8])
#         viz_scales=[0.06, 0.06, 0.042, 0.042, 0.042, 0.042, 0.15]
#         num_neurons=[8, 8, 8, 8, 8, 8, 2]
#         vertical_spacing=1.0


#         data_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/hackin/training_caches/8_8_8_2.pkl'
#         with open(data_path, 'rb') as file:
#             training_cache = pickle.load(file) #Training cache


#         self.set_view(48, 50, 0, (-0.04, 0.13, -0.47), 3.99)
#         train_step=2000

#         w1=training_cache['weights_history'][train_step]['model.0.weight'].numpy()
#         b1=training_cache['weights_history'][train_step]['model.0.bias'].numpy()
#         w2=training_cache['weights_history'][train_step]['model.2.weight'].numpy()
#         b2=training_cache['weights_history'][train_step]['model.2.bias'].numpy()
#         w3=training_cache['weights_history'][train_step]['model.4.weight'].numpy()
#         b3=training_cache['weights_history'][train_step]['model.4.bias'].numpy()
#         w4=training_cache['weights_history'][train_step]['model.6.weight'].numpy()
#         b4=training_cache['weights_history'][train_step]['model.6.bias'].numpy()

#         with torch.no_grad():
#             model.model[0].weight.copy_(torch.from_numpy(w1))
#             model.model[0].bias.copy_(torch.from_numpy(b1))
#             model.model[2].weight.copy_(torch.from_numpy(w2))
#             model.model[2].bias.copy_(torch.from_numpy(b2))
#             model.model[4].weight.copy_(torch.from_numpy(w3))
#             model.model[4].bias.copy_(torch.from_numpy(b3))
#             model.model[6].weight.copy_(torch.from_numpy(w4))
#             model.model[6].bias.copy_(torch.from_numpy(b4))

#         adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=0.6, extent=1)
#         #For the interesection to make sense, these scales need to match - either need to manual overide or chnage method above
#         final_layer_viz=scale=1.4*min(adaptive_viz_scales[-1]) #little manual ramp here
#         adaptive_viz_scales[-1]=[final_layer_viz, final_layer_viz]

#         #Precompute my surfaces, and polygons moving through network
#         surfaces=[]
#         surface_funcs=[]
#         for layer_idx in range(len(model.model)):
#             s=Group()
#             surface_funcs.append([])
#             for neuron_idx in range(num_neurons[layer_idx]):
#                 surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx]) #viz_scales[layer_idx])
#                 bent_surface = ParametricSurface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
#                 ts=TexturedSurface(bent_surface, graphics_dir+'/baarle_hertog_maps/'+map_filename)
#                 ts.set_shading(0,0,0).set_opacity(0.8)
#                 s.add(ts)
#                 surface_funcs[-1].append(surface_func)
#             surfaces.append(s)


#         self.add(surfaces[-1])

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
            
#             #Compute new tiling - general method with merging - should be more accurate but slow - buggy?
#             polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
            
#             #Less general method - less accurate, faster, maybe less buggy
#             # polygons[str(layer_id)+'.new_tiling_nested']=recompute_tiling(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
#             # polygons[str(layer_id)+'.new_tiling_nested']=recompute_tiling_polygonize(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
#             # polygons[str(layer_id)+'.new_tiling']=[item for sublist in polygons[str(layer_id)+'.new_tiling_nested'] for item in sublist]


#             print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')
#             #Optional filtering step
#             #polygons[str(layer_id)+'.new_tiling'] = filter_small_polygons(polygons[str(layer_id)+'.new_tiling'], min_area=1e-5)
#             #print(str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons remaining after filtering out small polygons')


#         #Last linear layer & output
#         polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
#         intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
#         my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)

#         output_horizontal_offset=9
#         groups_output=Group()
#         layer_idx=len(model.model)-1
#         total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
#         start_z = total_height / 2  # Start from top
#         for neuron_idx in range(num_neurons[layer_idx]):
#             # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
#             pgs=manim_polygons_from_np_list(polygons['3.linear_out'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
#             s=surfaces[layer_idx][neuron_idx]
#             g=Group(s, pgs) 
#             # g.shift([output_horizontal_offset, 0, start_z - neuron_idx * vertical_spacing])
#             groups_output.add(g)

#         # self.add(groups_output)

#         group_combined_output=groups_output.copy()
#         group_combined_output[0].set_color(BLUE)
#         group_combined_output[1].set_color(YELLOW)

#         self.add(group_combined_output)

#         self.wait(0)


class p62d(WelchScene):
    '''Ok so this is the transition of my 3d shape for the 2d view -> try blending in premiere!'''
    def construct(self):

        model = BaarleNet([8, 8, 8])
        viz_scales=[0.06, 0.06, 0.042, 0.042, 0.042, 0.042, 0.15]
        num_neurons=[8, 8, 8, 8, 8, 8, 2]
        vertical_spacing=1.0


        data_path=welch_path('backprop_3/hackin/training_caches/8_8_8_2.pkl')
        with open(data_path, 'rb') as file:
            training_cache = pickle.load(file) #Training cache

        #figure out a decent constant viz scale
        # for train_step in tqdm(np.arange(0, 1000, 10)):
        #     w1=training_cache['weights_history'][train_step]['model.0.weight'].numpy()
        #     b1=training_cache['weights_history'][train_step]['model.0.bias'].numpy()
        #     w2=training_cache['weights_history'][train_step]['model.2.weight'].numpy()
        #     b2=training_cache['weights_history'][train_step]['model.2.bias'].numpy()
        #     w3=training_cache['weights_history'][train_step]['model.4.weight'].numpy()
        #     b3=training_cache['weights_history'][train_step]['model.4.bias'].numpy()
        #     w4=training_cache['weights_history'][train_step]['model.6.weight'].numpy()
        #     b4=training_cache['weights_history'][train_step]['model.6.bias'].numpy()

        #     with torch.no_grad():
        #         model.model[0].weight.copy_(torch.from_numpy(w1))
        #         model.model[0].bias.copy_(torch.from_numpy(b1))
        #         model.model[2].weight.copy_(torch.from_numpy(w2))
        #         model.model[2].bias.copy_(torch.from_numpy(b2))
        #         model.model[4].weight.copy_(torch.from_numpy(w3))
        #         model.model[4].bias.copy_(torch.from_numpy(b3))
        #         model.model[6].weight.copy_(torch.from_numpy(w4))
        #         model.model[6].bias.copy_(torch.from_numpy(b4))


        #     adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=0.6, extent=1)
        #     #For the interesection to make sense, these scales need to match - either need to manual overide or chnage method above
        #     final_layer_viz=scale=1.4*min(adaptive_viz_scales[-1]) #little manual ramp here
        #     adaptive_viz_scales[-1]=[final_layer_viz, final_layer_viz]
        #     print(adaptive_viz_scales[-1])



        # self.set_view(48, 50, 0, (-0.04, 0.13, -0.47), 3.99)
        self.set_view(27, 54, 0, (-0.02, 0.05, -0.55), 3.99)
        # for train_step in tqdm(range(len(training_cache['weights_history']))): #All steps
        # for train_step in tqdm(np.arange(0, len(training_cache['weights_history']), 100)):
        # for train_step in tqdm(np.arange(0, 1000, 100)):
        combined_3d_group = None
        for train_step in tqdm(np.arange(0, 1000, 1)):
        # train_step=0

            if combined_3d_group is not None:
                self.remove(combined_3d_group)


            w1=training_cache['weights_history'][train_step]['model.0.weight'].numpy()
            b1=training_cache['weights_history'][train_step]['model.0.bias'].numpy()
            w2=training_cache['weights_history'][train_step]['model.2.weight'].numpy()
            b2=training_cache['weights_history'][train_step]['model.2.bias'].numpy()
            w3=training_cache['weights_history'][train_step]['model.4.weight'].numpy()
            b3=training_cache['weights_history'][train_step]['model.4.bias'].numpy()
            w4=training_cache['weights_history'][train_step]['model.6.weight'].numpy()
            b4=training_cache['weights_history'][train_step]['model.6.bias'].numpy()

            with torch.no_grad():
                model.model[0].weight.copy_(torch.from_numpy(w1))
                model.model[0].bias.copy_(torch.from_numpy(b1))
                model.model[2].weight.copy_(torch.from_numpy(w2))
                model.model[2].bias.copy_(torch.from_numpy(b2))
                model.model[4].weight.copy_(torch.from_numpy(w3))
                model.model[4].bias.copy_(torch.from_numpy(b3))
                model.model[6].weight.copy_(torch.from_numpy(w4))
                model.model[6].bias.copy_(torch.from_numpy(b4))


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
            output_horizontal_offset=9
            groups_output=gx.Group()
            layer_idx=len(model.model)-1
            total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
            start_z = total_height / 2  # Start from top
            for neuron_idx in range(num_neurons[layer_idx]):
                # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
                pgs=manim_polygons_from_np_list(polygons['3.linear_out'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[-1][0], opacity=0.6)
                s=surfaces[layer_idx][neuron_idx]
                g=gx.Group(s, pgs) #[1:]) #Crazy to leave off first merged/flat group here?
                # g.shift([output_horizontal_offset, 0, start_z - neuron_idx * vertical_spacing])
                groups_output.add(g)

            #Output surfaces together
            group_combined_output=groups_output.copy()
            group_combined_output[0].set_color(BLUE)
            group_combined_output[1].set_color(YELLOW)
            # group_combined_output[0].shift([3, 0, -vertical_spacing/2])
            # group_combined_output[1].shift([3, 0, vertical_spacing/2])

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
                # poly_3d.shift([output_horizontal_offset+3, 0, 0])
                top_polygons_vgroup.add(poly_3d)

            # loops=order_closed_loops_with_closure(intersection_lines)
            lines=gx.VGroup()
            for loop in intersection_lines: 
                loop=loop*np.array([1, 1, adaptive_viz_scales[-1][0]])
                line = gx.VMobject()
                line.set_points_as_corners(loop)
                line.set_stroke(color='#ec008c', width=4)
                lines.add(line)
            # lines.shift([output_horizontal_offset+3, 0, 0.])
            group_combined_output.set_opacity(0.3)
            top_polygons_vgroup.set_opacity(0.6)
            
            # self.add(layer_1_polygons_flat, layer_2_polygons_flat, layer_3_polygons_flat, final_map_group)

            combined_3d_group=gx.Group(group_combined_output, top_polygons_vgroup, lines)
            # combined_3d_group.move_to(ORIGIN)
            self.add(combined_3d_group)
            self.wait(0.1)

        # self.wait()


        # self.remove(group_combined_output)

        # self.add(top_polygons_vgroup)

        self.wait(20)
        pass




#Ok now I need a training view of the surface - not sure how well this is going to work lol!

class p62e(WelchScene):
    '''Ok so this is the transition of my 3d shape for the 2d view -> try blending in premiere!'''
    def construct(self):

        model = BaarleNet([8, 8, 8])
        viz_scales=[0.06, 0.06, 0.042, 0.042, 0.042, 0.042, 0.15]
        num_neurons=[8, 8, 8, 8, 8, 8, 2]
        vertical_spacing=1.0


        data_path=welch_path('backprop_3/hackin/training_caches/8_8_8_2.pkl')
        with open(data_path, 'rb') as file:
            training_cache = pickle.load(file) #Training cache

        prev_layer_1_polygons=None
        prev_layer_2_polygons=None
        prev_layer_3_polygons=None

        self.set_view(0, 0, 0, (3.94, 0.38, 0.0), 2.04)
        # for train_step in tqdm(range(len(training_cache['weights_history']))): #All steps
        # for train_step in tqdm(np.arange(0, len(training_cache['weights_history']), 100)):
        layer_1_polygons_flat = layer_2_polygons_flat = layer_3_polygons_flat = final_map_group = None
        for train_step in tqdm(np.arange(0,  1000, 1)):
        # for train_step in tqdm(np.arange(0, 600, 100)):

            if layer_1_polygons_flat is not None:
                self.remove(layer_1_polygons_flat, layer_2_polygons_flat, layer_3_polygons_flat, final_map_group)


            w1=training_cache['weights_history'][train_step]['model.0.weight'].numpy()
            b1=training_cache['weights_history'][train_step]['model.0.bias'].numpy()
            w2=training_cache['weights_history'][train_step]['model.2.weight'].numpy()
            b2=training_cache['weights_history'][train_step]['model.2.bias'].numpy()
            w3=training_cache['weights_history'][train_step]['model.4.weight'].numpy()
            b3=training_cache['weights_history'][train_step]['model.4.bias'].numpy()
            w4=training_cache['weights_history'][train_step]['model.6.weight'].numpy()
            b4=training_cache['weights_history'][train_step]['model.6.bias'].numpy()

            with torch.no_grad():
                model.model[0].weight.copy_(torch.from_numpy(w1))
                model.model[0].bias.copy_(torch.from_numpy(b1))
                model.model[2].weight.copy_(torch.from_numpy(w2))
                model.model[2].bias.copy_(torch.from_numpy(b2))
                model.model[4].weight.copy_(torch.from_numpy(w3))
                model.model[4].bias.copy_(torch.from_numpy(b3))
                model.model[6].weight.copy_(torch.from_numpy(w4))
                model.model[6].bias.copy_(torch.from_numpy(b4))



            adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=0.6, extent=1)
            #For the interesection to make sense, these scales need to match - either need to manual overide or chnage method above
            # final_layer_viz=scale=1.4*min(adaptive_viz_scales[-1]) #little manual ramp here
            adaptive_viz_scales[-1]=[0.014, 0.014] #Hard code bro

            #Precompute my surfaces, and polygons moving through network
            surfaces=[]
            surface_funcs=[]
            for layer_idx in range(len(model.model)):
                s=gx.Group()
                surface_funcs.append([])
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
            layer_1_polygons_flat=manim_polygons_from_np_list(prev_layer_1_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6)
            layer_1_polygons_flat.shift([0, 0, -5.0])

            if prev_layer_2_polygons is not None: 
                prev_layer_2_polygons=reorder_polygons_optimal(prev_layer_2_polygons, polygons['1.new_tiling'])
            else:
                prev_layer_2_polygons=polygons['1.new_tiling']         
            layer_2_polygons_flat=manim_polygons_from_np_list(prev_layer_2_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6)
            layer_2_polygons_flat.shift([3, 0, -5.0])

            if prev_layer_3_polygons is not None: 
                prev_layer_3_polygons=reorder_polygons_optimal(prev_layer_3_polygons, polygons['2.new_tiling'])
            else:
                prev_layer_3_polygons=polygons['2.new_tiling']
            layer_3_polygons_flat=manim_polygons_from_np_list(prev_layer_3_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6)
            layer_3_polygons_flat.shift([6, 0, -5.0])

            #Outputs surfaces
            output_horizontal_offset=9
            groups_output=gx.Group()
            layer_idx=len(model.model)-1
            total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
            start_z = total_height / 2  # Start from top
            for neuron_idx in range(num_neurons[layer_idx]):
                # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
                pgs=manim_polygons_from_np_list(polygons['3.linear_out'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
                s=surfaces[layer_idx][neuron_idx]
                g=gx.Group(s, pgs) #[1:]) #Crazy to leave off first merged/flat group here?
                g.shift([output_horizontal_offset, 0, start_z - neuron_idx * vertical_spacing])
                groups_output.add(g)

            #Output surfaces together
            group_combined_output=groups_output.copy()
            group_combined_output[0].set_color(BLUE)
            group_combined_output[1].set_color(YELLOW)
            group_combined_output[0].shift([3, 0, -vertical_spacing/2])
            group_combined_output[1].shift([3, 0, vertical_spacing/2])

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
                poly_3d.shift([output_horizontal_offset+3, 0, 0])
                top_polygons_vgroup.add(poly_3d)

            # loops=order_closed_loops_with_closure(intersection_lines)
            lines=gx.VGroup()
            for loop in intersection_lines: 
                loop=loop*np.array([1, 1, viz_scales[2]])
                line = gx.VMobject()
                line.set_points_as_corners(loop)
                line.set_stroke(color='#ec008c', width=4)
                lines.add(line)
            lines.shift([output_horizontal_offset+3, 0, 0.])

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
                poly_3d.shift([output_horizontal_offset+3, 0, -2])
                top_polygons_vgroup_flat.add(poly_3d)

            def flat_surf_func(u, v): return [u, v, 0]
            flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
            flat_map_2=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/'+map_filename)
            set_surface_shading(flat_map_2, 0,0,0).set_opacity(0.8)
            flat_map_2.shift([output_horizontal_offset+3, 0, -2])

            lines_flat=gx.VGroup()
            for loop in intersection_lines: 
                # loop=np.hstack((loop, np.zeros((len(loop), 1))))
                loop[:,2]=0
                line = gx.VMobject()
                line.set_points_as_corners(loop)
                line.set_stroke(color='#ec008c', width=5)
                lines_flat.add(line)
            lines_flat.shift([output_horizontal_offset+3, 0, -2])    

            group_combined_output.set_opacity(0.3)
            final_map_group=gx.Group(flat_map_2, top_polygons_vgroup_flat, lines_flat)

            layer_1_polygons_flat.shift([0, 2, 0])
            layer_2_polygons_flat.shift([-0.65, 2, 0])
            layer_3_polygons_flat.shift([-6, 0.5-0.85, 0])
            final_map_group.shift([-12+2.5-0.15, -0.35, -3])

            
            self.add(layer_1_polygons_flat, layer_2_polygons_flat, layer_3_polygons_flat, final_map_group)
            self.wait(0.1)

            # print(layer_1_polygons_flat.get_center())
            # print(layer_2_polygons_flat.get_center())
            # print(layer_3_polygons_flat.get_center())
            # print(final_map_group.get_center())



        self.wait(20)
        pass

















