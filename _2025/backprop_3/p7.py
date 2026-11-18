from welch_paths import welch_path
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
import copy
import manimgx as gx
import numpy as np
import pickle
import torch
from geometric_dl_utils import BaarleNet, manim_polygons_from_np_list, surface_func_from_model
from geometric_dl_utils_simplified import (
    clip_polygons,
    compute_top_polytope,
    merge_zero_regions,
    process_with_layers,
    recompute_tiling_general,
    split_polygons_with_relu_simple,
)
from plane_folding_utils import get_relu_joint, line_from_joint_points_1
from welch_scene import WelchScene
from welch_surface import parametric_surface, textured_surface

from polytope_intersection_utils import intersect_polytopes
# from decision_boundary_utils import *
from welch_surface import set_surface_shading

graphics_dir=welch_path('backprop_3/graphics/') #Point to folder where map images are
colors = [gx.RED, gx.BLUE, gx.GREEN, gx.YELLOW, gx.PURPLE, gx.ORANGE, gx.PINK, gx.TEAL]

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'


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




class p7a(WelchScene):
    def construct(self):
        model_path='_2025/backprop_3/models/8_2.pth'
        model = BaarleNet([8])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.07, 0.07, 0.05]
        num_neurons=[8, 8, 2]

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=viz_scales[layer_idx])
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


        #Ok now I need to get my 8 surfaces and ReLu joints for my 8 first layer neurons. 
        first_layer_groups=create_first_layer_relu_groups(model, surfaces, num_neurons_first_layer=8, extent=1, vertical_spacing=0.8)


        self.wait() 
        self.set_view(-4, 60, 0, (2.47, 0.04, -0.01), 8.00)
        self.add(first_layer_groups)

        surfaces[2][0].shift([3,0,0.6])
        polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2])
        polygons_21_copy=polygons_21.copy()
        polygons_21.shift([3, 0, 0.601]) #Move slightly above map

        surfaces[2][1].shift([3,0,-0.6])
        polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2])
        polygons_22_copy=polygons_22.copy()
        polygons_22.shift([3, 0, -0.599]) #Move slightly above map

        self.wait()
        # self.play(*[ReplacementTransform(surfaces[1][i].copy(), surfaces[2][0]) for i in range(len(surfaces[1]))],
        #     run_time=3.0)


        surfaces_1_copy=surfaces[1].copy()
        surfaces_1_copy_2=surfaces[1].copy()
        self.add(surfaces_1_copy)
        self.add(surfaces_1_copy_2)

        first_layer_groups_flat=create_first_layer_relu_groups(model, surfaces, num_neurons_first_layer=8, extent=1, vertical_spacing=0.0)
        shifted_line_copies=gx.Group()
        for i in range(len(first_layer_groups_flat)):
            if len(first_layer_groups_flat[i])>1:
                shifted_line_copies.add(first_layer_groups_flat[i][1].copy())
        shifted_line_copies_2 = shifted_line_copies.copy()
        shifted_line_copies.shift([3,0,0.9])

        og_lines=gx.Group()
        for i in range(len(first_layer_groups)):
            # print(len(first_layer_groups[i]))
            if len(first_layer_groups[i])>1:
                og_lines.add(first_layer_groups[i][1])

        og_line_copies=og_lines.copy()
        og_line_copies_2=og_lines.copy()
        self.add(og_line_copies)

        #Start going forward looking
        self.set_view(0, 67, 0, (3.11, 0.28, 0.16), 7.05)
        self.wait()

        #Move overhead to make my animation look less crappy
        self.play(self.animate_view(-2, 38, 0, (3.02, 0.27, 0.15), 7.05), run_time=3)


        #ok this is definitely not perfect - but I think I gotta ship it -> it's at minimum shippabale level I would say. 
        self.wait()
        self.play(*[gx.ReplacementTransform(og_line_copies[j], shifted_line_copies[j]) for j in range(len(shifted_line_copies))]+
                    [gx.ReplacementTransform(surfaces_1_copy[i], surfaces[2][0]) for i in range(len(surfaces[1]))],
                    run_time=2) #Play fast becuase it kidna sucks lol
        self.remove(shifted_line_copies); self.add(polygons_21)
        self.wait()


        #I think slight camera move while I do the second animation?
        shifted_line_copies_2.shift([3,0,-0.7])
        self.add(og_line_copies_2)

        self.wait()
        self.play(*[gx.ReplacementTransform(og_line_copies_2[j], shifted_line_copies_2[j]) for j in range(len(shifted_line_copies_2))]+
                    [gx.ReplacementTransform(surfaces_1_copy_2[i], surfaces[2][1]) for i in range(len(surfaces[1]))],
                    self.animate_view(-3, 61, 0, (2.87, 0.24, 0.08), 7.01),
                    run_time=2) #Play fast becuase it kidna sucks lol
        self.remove(shifted_line_copies_2); self.add(polygons_22)
        self.wait()


        #Now bring together on same axis - I think with copies this time, so I can just swap stuff out as we get bigger
        #Drop opacity on input maps and lines too!

        # Hmm hmm hmm maybe I dont' need a third panel -> maybe it's enough to bring them together like I did last time??
        # Then I'm just showing final surfaces and border as we expand -> that might be kinda a nice vibe actually. 


        # So it's definitely not that easy to see the border here -> i should briefly look at my top polytope option I think!
        # We could have them merge into this, or cross fade to it or something -> let me mess around here. 


        self.wait()


        top_polygons_vgroup=gx.VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=p_scaled[:,2]*viz_scales[2]
            poly_3d = gx.Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=2)
            poly_3d.set_opacity(0.3)
            poly_3d.shift([3, 0, 0])
            top_polygons_vgroup.add(poly_3d)

        self.wait()

        self.play(surfaces[2][0].animate.shift([0, 0, -0.6]), 
                  surfaces[2][1].animate.shift([0, 0, 0.6]), 
                  polygons_21.animate.shift([0, 0, -0.6]).set_color(BLUE),
                  polygons_22.animate.shift([0, 0, 0.6]).set_color(YELLOW),
                  surfaces[1].animate.set_opacity(0.1), 
                  og_lines.animate.set_opacity(0.2),
                  self.animate_view(1, 42, 0, (3.0, 0.05, -0.17), 3.54),
                  run_time=3)
        self.add(top_polygons_vgroup) #This actually worked pretty well!


        loops=order_closed_loops_with_closure(intersection_lines)
        lines=gx.VGroup()
        for loop in loops: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines.add(line)
        lines.shift([3, 0, 0])

        self.wait()
        self.play(gx.Create(lines), run_time=3)
        self.wait()


        ## Ok ok ok getting close ot finishing the 8 neuron model -> it's messy but I do think 
        ## that this will be a cool opening. 
        ## Last move I think is: 
        ## Move cameara to the right as I make the copy
        ## Toally fade out layer 1, make a copy of the map and final borders, flatten both and move a copy over
        ## Then as I step through bigger models I'll just render these 2 things!
        ## Then on P8 I can move to nice overhead view or something fun on the 512 map. 


        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(flat_map, 0,0,0).set_opacity(0.8)
        flat_map.shift([5.7, 0, 0])

        lines_flat=gx.VGroup()
        for loop in loops: 
            loop=loop*np.array([1, 1, 0])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines_flat.add(line)
        lines_flat.shift([5.7, 0, 0])

        self.wait()
        self.play(gx.ReplacementTransform(lines.copy(), lines_flat),
                 gx.ReplacementTransform(surfaces[2][0], flat_map),
                  surfaces[1].animate.set_opacity(0.0), 
                  og_lines.animate.set_opacity(0.0),
                 self.animate_view(0, 45, 0, (4.28, 0.08, -0.19), 3.97),
                 run_time=3)
        self.add(lines_flat) #Occlusions
        self.wait()

        self.wait(20)
        pass

        #Ok sweet -> now i just need these same static images for the bigger models. 
        # A few tweaks I might want to make, but I think this is good for now. 




class p7b_16(WelchScene):
    def construct(self):
        model_path='_2025/backprop_3/models/16_1.pth'
        model = BaarleNet([16])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.07, 0.07, 0.04]
        num_neurons=[16, 16, 2]

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=viz_scales[layer_idx])
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

        


        #I guess I don't have to do this every time -> coudl just draw all the lines? Might be better when we get really big here
        loops=order_closed_loops_with_closure(intersection_lines)
        lines=gx.VGroup()
        for loop in loops: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines.add(line)
        lines.shift([3, 0, 0])

        polygon_max_height=0.8

        top_polygons_vgroup=gx.VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=p_scaled[:,2]*viz_scales[2]
            p_scaled[:, -1] = np.clip(p_scaled[:, -1], -polygon_max_height, polygon_max_height)
            poly_3d = gx.Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=2)
            poly_3d.set_opacity(0.3)
            poly_3d.shift([3, 0, 0])
            top_polygons_vgroup.add(poly_3d)

        surfaces[2][0].shift([3,0,0])
        polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], polygon_max_height=polygon_max_height)
        polygons_21_copy=polygons_21.copy()
        polygons_21.shift([3, 0, 0.001]) #Move slightly above map

        surfaces[2][1].shift([3,0,0])
        polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], polygon_max_height=polygon_max_height)
        polygons_22_copy=polygons_22.copy()
        polygons_22.shift([3, 0, 0.002]) #Move slightly above map

        polygons_22.set_color(YELLOW)
        polygons_21.set_color(BLUE)

        polygons_21.set_opacity(0.1)
        polygons_22.set_opacity(0.1)
        surfaces[2][0].set_opacity(0.4)
        surfaces[2][1].set_opacity(0.4)
        top_polygons_vgroup.set_opacity(0.5)

        self.set_view(0, 45, 0, (4.28, 0.08, -0.19), 3.97)
        # self.add(surfaces[2][0], surfaces[2][1]) #ok i want to include these, but the max value clipping is tricky -> ignore for now. 
        self.add(polygons_21, polygons_22)
        self.add(top_polygons_vgroup)
        self.add(lines)


        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(flat_map, 0,0,0).set_opacity(0.8)
        flat_map.shift([5.7, 0, 0])

        lines_flat=gx.VGroup()
        for loop in loops: 
            loop=loop*np.array([1, 1, 0])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines_flat.add(line)
        lines_flat.shift([5.7, 0, 0])


        self.add(flat_map, lines_flat)
        self.wait()

        self.play(self.animate_view(0, 20, 0, (4.28, 0.08, -0.19), 3.97), run_time=4)
        self.wait()


        self.wait(20)
        pass




class p7c_32(WelchScene):
    def construct(self):
        model_path='_2025/backprop_3/models/32_1.pth'
        model = BaarleNet([32])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.07, 0.07, 0.04]
        num_neurons=[32, 32, 2]

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=viz_scales[layer_idx])
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

        


        #I guess I don't have to do this every time -> coudl just draw all the lines? Might be better when we get really big here
        loops=order_closed_loops_with_closure(intersection_lines)
        lines=gx.VGroup()
        for loop in loops: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines.add(line)
        lines.shift([3, 0, 0])

        polygon_max_height=0.8

        top_polygons_vgroup=gx.VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=p_scaled[:,2]*viz_scales[2]
            p_scaled[:, -1] = np.clip(p_scaled[:, -1], -polygon_max_height, polygon_max_height)
            poly_3d = gx.Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=2)
            poly_3d.set_opacity(0.3)
            poly_3d.shift([3, 0, 0])
            top_polygons_vgroup.add(poly_3d)

        surfaces[2][0].shift([3,0,0])
        polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], polygon_max_height=polygon_max_height)
        polygons_21_copy=polygons_21.copy()
        polygons_21.shift([3, 0, 0.001]) #Move slightly above map

        surfaces[2][1].shift([3,0,0])
        polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], polygon_max_height=polygon_max_height)
        polygons_22_copy=polygons_22.copy()
        polygons_22.shift([3, 0, 0.002]) #Move slightly above map

        polygons_22.set_color(YELLOW)
        polygons_21.set_color(BLUE)

        polygons_21.set_opacity(0.1)
        polygons_22.set_opacity(0.1)
        surfaces[2][0].set_opacity(0.4)
        surfaces[2][1].set_opacity(0.4)
        top_polygons_vgroup.set_opacity(0.5)

        self.set_view(0, 20, 0, (4.28, 0.08, -0.19), 3.97)
        # self.add(surfaces[2][0], surfaces[2][1]) #ok i want to include these, but the max value clipping is tricky -> ignore for now. 
        self.add(polygons_21, polygons_22)
        self.add(top_polygons_vgroup)
        self.add(lines)


        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(flat_map, 0,0,0).set_opacity(0.8)
        flat_map.shift([5.7, 0, 0])

        lines_flat=gx.VGroup()
        for loop in loops: 
            loop=loop*np.array([1, 1, 0])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines_flat.add(line)
        lines_flat.shift([5.7, 0, 0])


        self.add(flat_map, lines_flat)
        self.wait()

        # self.play(self.animate_view(0, 20, 0, (4.28, 0.08, -0.19), 3.97), run_time=4)
        self.wait()


        self.wait(20)
        pass



class p7d_64(WelchScene):
    def construct(self):
        model_path='_2025/backprop_3/models/64_1.pth'
        model = BaarleNet([64])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.07, 0.07, 0.04]
        num_neurons=[64, 64, 2]

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=viz_scales[layer_idx])
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

        


        #I guess I don't have to do this every time -> coudl just draw all the lines? Might be better when we get really big here
        loops=order_closed_loops_with_closure(intersection_lines)
        lines=gx.VGroup()
        for loop in loops: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines.add(line)
        lines.shift([3, 0, 0])

        polygon_max_height=0.8

        top_polygons_vgroup=gx.VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=p_scaled[:,2]*viz_scales[2]
            p_scaled[:, -1] = np.clip(p_scaled[:, -1], -polygon_max_height, polygon_max_height)
            poly_3d = gx.Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=2)
            poly_3d.set_opacity(0.3)
            poly_3d.shift([3, 0, 0])
            top_polygons_vgroup.add(poly_3d)

        surfaces[2][0].shift([3,0,0])
        polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], polygon_max_height=polygon_max_height)
        polygons_21_copy=polygons_21.copy()
        polygons_21.shift([3, 0, 0.001]) #Move slightly above map

        surfaces[2][1].shift([3,0,0])
        polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], polygon_max_height=polygon_max_height)
        polygons_22_copy=polygons_22.copy()
        polygons_22.shift([3, 0, 0.002]) #Move slightly above map

        polygons_22.set_color(YELLOW)
        polygons_21.set_color(BLUE)

        polygons_21.set_opacity(0.1)
        polygons_22.set_opacity(0.1)
        surfaces[2][0].set_opacity(0.4)
        surfaces[2][1].set_opacity(0.4)
        top_polygons_vgroup.set_opacity(0.5)

        self.set_view(0, 20, 0, (4.28, 0.08, -0.19), 3.97)
        # self.add(surfaces[2][0], surfaces[2][1]) #ok i want to include these, but the max value clipping is tricky -> ignore for now. 
        self.add(polygons_21, polygons_22)
        self.add(top_polygons_vgroup)
        self.add(lines)


        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(flat_map, 0,0,0).set_opacity(0.8)
        flat_map.shift([5.7, 0, 0])

        lines_flat=gx.VGroup()
        for loop in loops: 
            loop=loop*np.array([1, 1, 0])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines_flat.add(line)
        lines_flat.shift([5.7, 0, 0])


        self.add(flat_map, lines_flat)
        self.wait()

        # self.play(self.animate_view(0, 20, 0, (4.28, 0.08, -0.19), 3.97), run_time=4)
        self.wait()


        self.wait(20)
        pass


class p7e_128(WelchScene):
    def construct(self):
        model_path='_2025/backprop_3/models/128_1.pth'
        model = BaarleNet([128])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.07, 0.07, 0.04]
        num_neurons=[128, 128, 2]

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=viz_scales[layer_idx])
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

        
        with open('_2025/backprop_3/models/128_1_borders.p', 'rb') as file:
            borders_interp = pickle.load(file)

        #I guess I don't have to do this every time -> coudl just draw all the lines? Might be better when we get really big here
        # loops=order_closed_loops_with_closure(intersection_lines)
        lines=gx.VGroup()
        for loop in borders_interp: 
            loop=np.hstack((loop, np.zeros((len(loop), 1))))
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines.add(line)
        lines.shift([3, 0, 0])

        polygon_max_height=0.8

        top_polygons_vgroup=gx.VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=p_scaled[:,2]*viz_scales[2]
            p_scaled[:, -1] = np.clip(p_scaled[:, -1], -polygon_max_height, polygon_max_height)
            poly_3d = gx.Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=2)
            poly_3d.set_opacity(0.3)
            poly_3d.shift([3, 0, 0])
            top_polygons_vgroup.add(poly_3d)

        surfaces[2][0].shift([3,0,0])
        polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], polygon_max_height=polygon_max_height)
        polygons_21_copy=polygons_21.copy()
        polygons_21.shift([3, 0, 0.001]) #Move slightly above map

        surfaces[2][1].shift([3,0,0])
        polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], polygon_max_height=polygon_max_height)
        polygons_22_copy=polygons_22.copy()
        polygons_22.shift([3, 0, 0.002]) #Move slightly above map

        polygons_22.set_color(YELLOW)
        polygons_21.set_color(BLUE)

        polygons_21.set_opacity(0.1)
        polygons_22.set_opacity(0.1)
        surfaces[2][0].set_opacity(0.4)
        surfaces[2][1].set_opacity(0.4)
        top_polygons_vgroup.set_opacity(0.5)

        self.set_view(0, 20, 0, (4.28, 0.08, -0.19), 3.97)
        # self.add(surfaces[2][0], surfaces[2][1]) #ok i want to include these, but the max value clipping is tricky -> ignore for now. 
        self.add(polygons_21, polygons_22)
        self.add(top_polygons_vgroup)
        self.add(lines)

        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(flat_map, 0,0,0).set_opacity(0.8)
        flat_map.shift([5.7, 0, 0])

        lines_flat_cleaner=gx.VGroup()
        for loop in borders_interp: 
            loop=np.hstack((loop, np.zeros((len(loop), 1))))
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=4)
            lines_flat_cleaner.add(line)
        lines_flat_cleaner.shift([5.7, 0, 0])      

        self.add(flat_map, lines_flat_cleaner)
        self.wait()

        # self.play(self.animate_view(0, 20, 0, (4.28, 0.08, -0.19), 3.97), run_time=4)
        # self.wait()

        # Ok ok ok ok let's try switching to an overhead/flattened view, and then will us overhead going forward. 

        top_polygons_vgroup_flat=gx.VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=0 #p_scaled[:,2]*viz_scales[2] #Flatten that shit!
            p_scaled[:, -1] = np.clip(p_scaled[:, -1], -polygon_max_height, polygon_max_height)
            poly_3d = gx.Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=2)
            poly_3d.set_opacity(0.3)
            poly_3d.shift([3, 0, 0])
            top_polygons_vgroup_flat.add(poly_3d)


        polygon_arrays_1_flat=copy.deepcopy(polygons['1.linear_out'][0])
        for p in polygon_arrays_1_flat: p[:,2]=0

        polygon_arrays_2_flat=copy.deepcopy(polygons['1.linear_out'][1])
        for p in polygon_arrays_2_flat: p[:,2]=0

        polygons_21_flat=manim_polygons_from_np_list(polygon_arrays_1_flat, colors=colors, viz_scale=viz_scales[2], polygon_max_height=polygon_max_height)
        polygons_21_flat.shift([3, 0, 0.001]) #Move slightly above map
        polygons_22_flat=manim_polygons_from_np_list(polygon_arrays_2_flat, colors=colors, viz_scale=viz_scales[2], polygon_max_height=polygon_max_height)
        polygons_22_flat.shift([3, 0, 0.002]) #Move slightly above map
        polygons_22_flat.set_color(YELLOW)
        polygons_21_flat.set_color(BLUE)
        polygons_21_flat.set_opacity(0.1)
        polygons_22_flat.set_opacity(0.1)


        top_polygons_vgroup_flat.set_opacity(0.5)

        self.wait()
        self.remove(polygons_21, polygons_22)
        self.play(gx.ReplacementTransform(top_polygons_vgroup, top_polygons_vgroup_flat),
                  # ReplacementTransform(polygons_21, polygons_21_flat),
                  # ReplacementTransform(polygons_22, polygons_22_flat),
                  # polygons_21_flat.animate.set_opacity(0.0),
                  # polygons_22_flat.animate.set_opacity(0.0),
                  self.animate_view(0, 0, 0, (4.34, 0.02, -0.19), 3.47),
                  run_time=6)
        self.remove(lines); self.add(lines)
        self.wait()
        # self.remove(top_polygons_vgroup_flat); self.add(top_polygons_vgroup_flat)

        # polygons_21_flat.set_opacity(0.0)
        # polygons_22_flat.set_opacity(0.0)

        self.wait(20)
        pass



class p7f_256(WelchScene):
    def construct(self):
        model_path='_2025/backprop_3/models/256_1.pth'
        model = BaarleNet([256])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.07, 0.07, 0.04]
        num_neurons=[256, 256, 2]

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

        
        with open('_2025/backprop_3/models/256_1_borders.p', 'rb') as file:
            borders_interp = pickle.load(file)

        #I guess I don't have to do this every time -> coudl just draw all the lines? Might be better when we get really big here
        # loops=order_closed_loops_with_closure(intersection_lines)
        lines=gx.VGroup()
        for loop in borders_interp: 
            loop=np.hstack((loop, np.zeros((len(loop), 1))))
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
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
        flat_map=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(flat_map, 0,0,0).set_opacity(0.8)
        flat_map.shift([5.7, 0, 0])

        lines_flat_cleaner=gx.VGroup()
        for loop in borders_interp: 
            loop=np.hstack((loop, np.zeros((len(loop), 1))))
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=4)
            lines_flat_cleaner.add(line)
        lines_flat_cleaner.shift([5.7, 0, 0])      

        self.set_view(0, 0, 0, (4.34, 0.02, -0.19), 3.47)
        self.add(top_polygons_vgroup_flat)
        self.add(lines)

        self.add(flat_map, lines_flat_cleaner)
        self.wait()

        self.wait(20)
        pass




class p7g_512(WelchScene):
    def construct(self):
        model_path='_2025/backprop_3/models/512_1_longer.pth'
        model = BaarleNet([512])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.07, 0.07, 0.04]
        num_neurons=[512, 512, 2]

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

        
        with open('_2025/backprop_3/models/512_1_borders.p', 'rb') as file:
            borders_interp = pickle.load(file)

        #I guess I don't have to do this every time -> coudl just draw all the lines? Might be better when we get really big here
        # loops=order_closed_loops_with_closure(intersection_lines)
        lines=gx.VGroup()
        for loop in borders_interp: 
            loop=np.hstack((loop, np.zeros((len(loop), 1))))
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
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
        flat_map=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(flat_map, 0,0,0).set_opacity(0.8)
        flat_map.shift([5.7, 0, 0])

        lines_flat_cleaner=gx.VGroup()
        for loop in borders_interp: 
            loop=np.hstack((loop, np.zeros((len(loop), 1))))
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=4)
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



class p7h_1024(WelchScene):
    def construct(self):
        model_path='_2025/backprop_3/models/one_layer_1024_nuerons_long.pth'
        model = BaarleNet([1024])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.07, 0.07, 0.04]
        num_neurons=[1024, 1024, 2]

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

        
        with open('_2025/backprop_3/models/1024_borders.p', 'rb') as file:
            borders_interp = pickle.load(file)

        #I guess I don't have to do this every time -> coudl just draw all the lines? Might be better when we get really big here
        # loops=order_closed_loops_with_closure(intersection_lines)
        lines=gx.VGroup()
        for loop in borders_interp: 
            loop=np.hstack((loop, np.zeros((len(loop), 1))))
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
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
        flat_map=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(flat_map, 0,0,0).set_opacity(0.8)
        flat_map.shift([5.7, 0, 0])

        lines_flat_cleaner=gx.VGroup()
        for loop in borders_interp: 
            loop=np.hstack((loop, np.zeros((len(loop), 1))))
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=4)
            lines_flat_cleaner.add(line)
        lines_flat_cleaner.shift([5.7, 0, 0])      

        self.set_view(0, 0, 0, (4.34, 0.02, -0.19), 3.47)
        self.add(top_polygons_vgroup_flat)
        self.add(lines)

        self.add(flat_map, lines_flat_cleaner)
        self.wait()

        #maybe a fade out and center on map/border? That's all we'll see for these last few rigth?
        self.play(top_polygons_vgroup_flat.animate.set_opacity(0.0),
                  lines.animate.set_opacity(0.0),
                  self.animate_view(0, 0, 0, (5.67, -0.01, -0.19), 3.47),
                  run_time=5.0)


        self.wait(20)
        pass


#Ok for these we just want the map and refined border!

class p7i_10k(WelchScene):
    def construct(self):
        # model_path='_2025/backprop_3/models/one_layer_10k_neurons_long.pth'
        # model = BaarleNet([10000])
        # model.load_state_dict(torch.load(model_path))
        # viz_scales=[0.07, 0.07, 0.04]
        # num_neurons=[10000, 10000, 2]

        
        with open('_2025/backprop_3/models/10k_borders.p', 'rb') as file:
            borders_interp = pickle.load(file)

        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(flat_map, 0,0,0).set_opacity(0.8)
        flat_map.shift([5.7, 0, 0])

        lines_flat_cleaner=gx.VGroup()
        for loop in borders_interp: 
            loop=np.hstack((loop, np.zeros((len(loop), 1))))
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=4)
            lines_flat_cleaner.add(line)
        lines_flat_cleaner.shift([5.7, 0, 0])      

        self.set_view(0, 0, 0, (5.67, -0.01, -0.19), 3.47)


        self.add(flat_map, lines_flat_cleaner)

        self.wait(20)
        pass




class p7j_100k(WelchScene):
    def construct(self):
        # model_path='_2025/backprop_3/models/one_layer_100k_neurons_long.pth'
        # model = BaarleNet([100000])
        # model.load_state_dict(torch.load(model_path))
        # viz_scales=[0.07, 0.07, 0.04]
        # num_neurons=[100000, 100000, 2]


        with open('_2025/backprop_3/models/100k_borders.p', 'rb') as file:
            borders_interp = pickle.load(file)

        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(flat_map, 0,0,0).set_opacity(0.8)
        flat_map.shift([5.7, 0, 0])

        lines_flat_cleaner=gx.VGroup()
        for loop in borders_interp: 
            loop=np.hstack((loop, np.zeros((len(loop), 1))))
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=4)
            lines_flat_cleaner.add(line)
        lines_flat_cleaner.shift([5.7, 0, 0])      

        self.set_view(0, 0, 0, (5.67, -0.01, -0.19), 3.47)


        self.add(flat_map, lines_flat_cleaner)

        self.wait(20)
        pass






