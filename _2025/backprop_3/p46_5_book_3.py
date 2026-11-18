from welch_paths import welch_path
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
import copy
import manimgx as gx
import numpy as np
import torch
from geometric_dl_utils import BaarleNet, manim_polygons_from_np_list, surface_func_from_model
from geometric_dl_utils_simplified import (
    clip_polygons,
    compute_top_polytope,
    merge_zero_regions,
    order_closed_loops_with_closure,
    process_with_layers,
    recompute_tiling_general,
    split_polygons_with_relu_simple,
)
from plane_folding_utils import get_relu_joint, line_from_joint_points_1
from welch_scene import WelchScene, image_at_height
from welch_surface import parametric_surface, textured_surface
from polytope_intersection_utils import intersect_polytopes
from welch_surface import set_surface_shading

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
# colors = [BLUE, GREY, GREEN, TEAL, PURPLE, ORANGE, PINK, TEAL, RED, YELLOW ]
colors = [gx.GREY, BLUE, GREEN, YELLOW, gx.PURPLE, gx.ORANGE, gx.PINK, gx.TEAL]

class p46(WelchScene):
    def construct(self):

        map_img=image_at_height(graphics_dir+'/baarle_hertog_maps/'+map_filename)
        map_img.scale_to_fit_height(2)  # This sets the height to 2 units (-1 to 1)
        map_img.scale_to_fit_width(2)   # This sets the width to 2 units (-1 to 1)
        map_img.move_to(gx.ORIGIN)

        model = BaarleNet([2])

        w1 = np.array([[2.5135, -1.02481],
         [-1.4043, 2.41291]], dtype=np.float32)
        b1 = np.array([-1.23981, -0.450078], dtype=np.float32)
        w2 = np.array([[3.17024, 1.32567],
         [-3.40372, -1.53878]], dtype=np.float32)
        b2 = np.array([-0.884835, 0.0332228], dtype=np.float32)

        with torch.no_grad():
            model.model[0].weight.copy_(torch.from_numpy(w1))
            model.model[0].bias.copy_(torch.from_numpy(b1))
            model.model[2].weight.copy_(torch.from_numpy(w2))
            model.model[2].bias.copy_(torch.from_numpy(b2))

        viz_scales=[0.2, 0.2, 0.13]
        num_neurons=[2, 2, 2]

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=viz_scales[layer_idx])
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



        #Get first layer Relu Joints - I havea method that automates this if I need ot scale it. 

        # polygons_11=manim_polygons_from_np_list([item for sublist in polygons['0.split_polygons_nested_clipped'][0] for item in sublist], 
        #                                         colors=[GREY, BLACK], viz_scale=viz_scales[1], opacity=0.3)
        # polygons_11.shift([0, 0, 0.001]) #Move slightly above map

        # polygons_12=manim_polygons_from_np_list([item for sublist in polygons['0.split_polygons_nested_clipped'][1] for item in sublist], 
        #                                         colors=[GREY, BLACK], viz_scale=viz_scales[1], opacity=0.3)
        # polygons_12.shift([0, 0, 0.001]) #Move slightly above map


        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        group_11=gx.Group(surfaces[1][0]) #, polygons_11)
        if len(joint_points_11)>0:
            joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.9)
            group_11.add(joint_line_11)

        group_12=gx.Group(surfaces[1][1]) #, polygons_12)
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        if len(joint_points_12)>0:
            joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.9)
            group_12.add(joint_line_12)

        group_11.shift([0, 0, 1.5])

        polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        polygons_21.shift([0, 0, 0.001]) #Move slightly above map

        polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        polygons_22.shift([0, 0, 0.001]) #Move slightly above map

        group_21=gx.Group(surfaces[2][0], polygons_21)
        group_22=gx.Group(surfaces[2][1], polygons_22)
        group_21.shift([3.0, 0, 1.5])
        group_22.shift([3.0, 0, 0])
        group_21.set_opacity(0.5)
        group_22.set_opacity(0.5)


        #Ok now some plane intersction action in a third "panel"
        group_31=group_21.copy()
        group_31[1].set_color(BLUE)
        group_31.shift([3, 0, -0.75])

        group_32=group_22.copy()
        group_32[1].set_color(YELLOW)
        group_32.shift([3, 0, 0.75])

        loops=order_closed_loops_with_closure(intersection_lines)
        lines=gx.VGroup()
        for loop in loops: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=5)
            lines.add(line)
        lines.shift([6, 0, 0.75])

        #Make Baarle hertog maps a little mroe pronounced. 
        group_31[0].set_opacity(0.9)
        group_32[0].set_opacity(0.9)
        group_31[1].set_opacity(0.4)
        group_32[1].set_opacity(0.4)

        # group_21.set_opacity(0.9)
        # group_22.set_opacity(0.9)

        # self.set_view(-1, 42, 0, (3.1, 0.59, -0.39), 6.92)
        # self.set_view(-2, 46, 0, (3.11, 0.22, -0.73), 6.89)

        # self.wait()
        # self.play(FadeIn(group_11), FadeIn(group_12))
        # self.wait()

        # self.play(FadeIn(group_21), FadeIn(group_22)) #Would be nice to animate layer 1 neurons coming together instead - we'll see. 
        # self.wait()

        bent_plane_joint_lines=gx.VGroup()
        pre_move_lines=gx.VGroup()

        line_start=polygons['1.linear_out'][0][1][1]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][1][2]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][1][1]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][1][2]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 1.5])
        pre_move_lines.add(joint_line)


        line_start=polygons['1.linear_out'][0][0][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][0][1]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][0][0]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][0][1]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 0])
        pre_move_lines.add(joint_line)


        line_start=polygons['1.linear_out'][0][2][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][2][1]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][2][0]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][2][1]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        # joint_line.shift([0, 0, 1.5])
        pre_move_lines.add(joint_line)


        line_start=polygons['1.linear_out'][0][2][3]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][2][0]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][2][3]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][2][0]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 1.5])
        pre_move_lines.add(joint_line)


        self.set_view(-2, 46, 0, (3.11, 0.22, -0.73), 6.89)
        self.wait()
        self.play(gx.FadeIn(group_11[0]), gx.FadeIn(group_12[0]), gx.FadeIn(pre_move_lines))
        self.wait()


        # self.add(joint_line)  
        self.wait()
        self.play(gx.ReplacementTransform(surfaces[1][0].copy(),surfaces[2][0]),
                  gx.ReplacementTransform(surfaces[1][1].copy(),surfaces[2][0]),
                  gx.ReplacementTransform(pre_move_lines.copy(), bent_plane_joint_lines),
                    run_time=3)
        self.add(polygons_21)
        self.remove(bent_plane_joint_lines); self.add(bent_plane_joint_lines)
        self.wait()

        bent_plane_joint_lines_2=gx.VGroup()
        pre_move_lines_2=gx.VGroup()

        line_start=polygons['1.linear_out'][1][1][1]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][1][1][2]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0.0])
        bent_plane_joint_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][1][1]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][1][1][2]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 1.5])
        pre_move_lines_2.add(joint_line)


        line_start=polygons['1.linear_out'][1][0][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][1][0][1]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0.0])
        bent_plane_joint_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][0][0]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][1][0][1]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 0])
        pre_move_lines_2.add(joint_line)


        line_start=polygons['1.linear_out'][1][2][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][1][2][1]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0.0])
        bent_plane_joint_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][2][0]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][1][2][1]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        # joint_line.shift([0, 0, 1.5])
        pre_move_lines_2.add(joint_line)


        line_start=polygons['1.linear_out'][1][2][3]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][1][2][0]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0.0])
        bent_plane_joint_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][2][3]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][1][2][0]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 1.5])
        pre_move_lines_2.add(joint_line)

        self.wait()
        self.play(gx.ReplacementTransform(surfaces[1][0].copy(),surfaces[2][1]),
                  gx.ReplacementTransform(surfaces[1][1].copy(),surfaces[2][1]),
                  gx.ReplacementTransform(pre_move_lines_2.copy(), bent_plane_joint_lines_2),
                  run_time=3)
        self.add(polygons_22)
        self.remove(bent_plane_joint_lines_2); self.add(bent_plane_joint_lines_2)
        self.wait()


        #Ok now a little animation bringin the two bent surfaces together and changing their colors? 
        self.play(gx.ReplacementTransform(group_21.copy(), group_31),
                  gx.ReplacementTransform(group_22.copy(), group_32),
                 run_time=3.0)
        self.play(gx.Create(lines))
        self.wait()

        #Tempted to zoom in on decision boundary, but mabye we've seen it enough?
        # Ok now need ot hink through how this fits with illustrator network and then exapnding it! Good start!
        #Ok will do a new scene for the 2 layer model!


        # self.add(group_31, group_32, lines)
        # self.wait()

        # self.play(ReplacementTransform(polygons_11[0].copy(), polygons_21[2:]))


        self.wait(20)
        pass


class p47m(WelchScene):
    def construct(self):

        map_img=image_at_height(graphics_dir+'/baarle_hertog_maps/'+map_filename)
        map_img.scale_to_fit_height(2)  # This sets the height to 2 units (-1 to 1)
        map_img.scale_to_fit_width(2)   # This sets the width to 2 units (-1 to 1)
        map_img.move_to(gx.ORIGIN)


        model_path='_2025/backprop_3/models/2_2_1.pth'
        model = BaarleNet([2,2])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.25, 0.25, 0.3, 0.3, 0.15]
        num_neurons=[2, 2, 2, 2, 2]

        w1=model.model[0].weight.detach().numpy()
        b1=model.model[0].bias.detach().numpy()
        w2=model.model[2].weight.detach().numpy()
        b2=model.model[2].bias.detach().numpy()

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=viz_scales[layer_idx])
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


        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        group_11=gx.Group(surfaces[1][0]) #, polygons_11)
        if len(joint_points_11)>0:
            joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.9)
            group_11.add(joint_line_11)

        group_12=gx.Group(surfaces[1][1]) #, polygons_12)
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        if len(joint_points_12)>0:
            joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.9)
            group_12.add(joint_line_12)

        group_11.shift([0, 0, 1.5])

        polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        polygons_21.shift([0, 0, 0.001]) #Move slightly above map

        polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        polygons_22.shift([0, 0, 0.001]) #Move slightly above map

        group_21=gx.Group(surfaces[2][0], polygons_21)
        group_22=gx.Group(surfaces[2][1], polygons_22)
        group_21.shift([3.0, 0, 1.5])
        group_22.shift([3.0, 0, 0])
        group_21.set_opacity(0.5)
        group_22.set_opacity(0.5)


        #Ok now some plane intersction action in a third "panel"
        group_31_fakeout=group_21.copy()
        group_31_fakeout[1].set_color(BLUE)
        group_31_fakeout.shift([3, 0, -0.75])

        group_32_fakeout=group_22.copy()
        group_32_fakeout[1].set_color(YELLOW)
        group_32_fakeout.shift([3, 0, 0.75])


        #Make Baarle hertog maps a little mroe pronounced. 
        group_31_fakeout[0].set_opacity(0.9)
        group_32_fakeout[0].set_opacity(0.9)
        group_31_fakeout[1].set_opacity(0.4)
        group_32_fakeout[1].set_opacity(0.4)

        # group_21.set_opacity(0.9)
        # group_22.set_opacity(0.9)

        # self.set_view(-1, 42, 0, (3.1, 0.59, -0.39), 6.92)
        bent_plane_joint_lines=gx.VGroup()
        pre_move_lines=gx.VGroup()

        line_start=polygons['1.linear_out'][0][1][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][1][1]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][1][0]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][1][1]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 1.5])
        pre_move_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][0][3]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][0][0]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][0][3]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][0][0]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 1.5])
        pre_move_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][2][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][2][1]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][2][0]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][2][1]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        # joint_line.shift([0, 0, 1.5])
        pre_move_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][0][2]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][0][3]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][0][2]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][0][3]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        # joint_line.shift([0, 0, 1.5])
        pre_move_lines.add(joint_line)


        self.set_view(-2, 46, 0, (3.11, 0.22, -0.73), 6.89)

        self.wait()
        self.play(gx.FadeIn(group_11[0]), gx.FadeIn(group_12[0]), gx.FadeIn(pre_move_lines))
        self.wait()

        # Ok I think it's imporant here (and maybe in p46 above) to actually do the brining the 
        # lines together animation, as annoying as it is
        # It's only 2 lines - I can do it -> and I've found a way to make it work -> it's just annoying!

        # self.remove(group_11[1]); self.remove(group_12[1]) #Remove existing Relu lines to replace with "premove lines"
        # self.add(pre_move_lines)


        # self.add(joint_line)  
        self.wait()
        self.play(gx.ReplacementTransform(surfaces[1][0].copy(),surfaces[2][0]),
                  gx.ReplacementTransform(surfaces[1][1].copy(),surfaces[2][0]),
                  gx.ReplacementTransform(pre_move_lines.copy(), bent_plane_joint_lines),
                    run_time=3)
        self.add(polygons_21)
        self.remove(bent_plane_joint_lines); self.add(bent_plane_joint_lines)
        self.wait()

        bent_plane_joint_lines_2=gx.VGroup()
        pre_move_lines_2=gx.VGroup()

        line_start=polygons['1.linear_out'][1][1][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][1][1][1]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0])
        bent_plane_joint_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][1][0]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][1][1][1]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 1.5])
        pre_move_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][0][3]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][1][0][0]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0])
        bent_plane_joint_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][0][3]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][1][0][0]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 1.5])
        pre_move_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][2][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][1][2][1]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0])
        bent_plane_joint_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][2][0]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][1][2][1]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        # joint_line.shift([0, 0, 1.5])
        pre_move_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][0][2]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][1][0][3]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0])
        bent_plane_joint_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][0][2]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][1][0][3]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        # joint_line.shift([0, 0, 1.5])
        pre_move_lines_2.add(joint_line)


        self.wait()
        self.play(gx.ReplacementTransform(surfaces[1][0].copy(),surfaces[2][1]),
                  gx.ReplacementTransform(surfaces[1][1].copy(),surfaces[2][1]),
                  gx.ReplacementTransform(pre_move_lines_2.copy(), bent_plane_joint_lines_2),
                  run_time=3)
        self.add(polygons_22)
        self.remove(bent_plane_joint_lines_2); self.add(bent_plane_joint_lines_2)
        self.wait()

        #Ok now quick "fake out 3 layer", then remove everything from that. 
        self.play(gx.ReplacementTransform(group_21.copy(), group_31_fakeout),
                  gx.ReplacementTransform(group_22.copy(), group_32_fakeout),
                 run_time=3.0)
        # self.play(ShowCreation(lines_fakeout))
        self.wait()
        self.play(gx.FadeOut(group_31_fakeout), gx.FadeOut(group_32_fakeout))
        self.wait()

        # I'm a bit torn on working off a copy to the right vs doing this in place -> 
        # I guess I'm leaning towards doing it in place?
        # Ok I think we add z=0 planes, actually let's focus/zoom on top neuron, and add the plane there
        relu_intersections_planes_1=gx.VGroup()
        for neuron_idx in range(2):
            # plane = Rectangle( width=2,  height=2, fill_color=GREY, fill_opacity=0.3, stroke_color=WHITE, stroke_width=2)
            plane = gx.NumberPlane(
                x_range=(-1, 1, 0.2),  # x from -1 to 1, grid every 0.2 units
                y_range=(-1, 1, 0.2),  # y from -1 to 1, grid every 0.2 units
                background_line_style={
                    "stroke_color": gx.WHITE,
                    "stroke_width": 0.5,
                    "stroke_opacity": 0.9
                },
                faded_line_style={
                    "stroke_color": gx.WHITE,
                    "stroke_width": 0.0,
                    "stroke_opacity": 0.0
                },
                axis_config={
                    "stroke_color": gx.WHITE,
                    "stroke_width": 0.5
                }
            ).scale_to_fit_width(2).scale_to_fit_height(2)
            plane.shift([3, 0, 1.5*neuron_idx])
            relu_intersections_planes_1.add(plane)
        
        self.wait()
        self.play(self.animate_view(30, 70, 0, (2.55, 1.16, 0.93), 4.27),
                  group_11.animate.set_opacity(0.2), 
                  group_12.animate.set_opacity(0.2),
                  pre_move_lines.animate.set_opacity(0.0),
                  run_time=3)
        self.play(gx.Create(relu_intersections_planes_1[1]))
        self.wait()

        self.play(self.animate_view(32, 83, 0, (2.55, 1.16, 0.93), 4.27), run_time=3)
        self.wait()

        #Outline to call out planes, I can cut this if I don't like it. 
        outline = polygons_21[1].copy()
        outline.set_fill(opacity=0)
        outline.set_stroke(width=4, opacity=0.9)
        self.play(gx.Create(outline, run_time=2))
        # self.play(FadeOut(outline))

        outline_2 = polygons_21[3].copy()
        outline_2.set_fill(opacity=0)
        outline_2.set_stroke(width=4, opacity=0.9)
        self.play(gx.Create(outline_2), gx.FadeOut(outline), run_time=2)

        outline_3 = polygons_21[2].copy()
        outline_3.set_fill(opacity=0)
        outline_3.set_stroke(width=4, opacity=0.9)
        self.play(gx.Create(outline_3), gx.FadeOut(outline_2), run_time=2)
        self.play(gx.FadeOut(outline_3))
        self.remove(bent_plane_joint_lines); self.add(bent_plane_joint_lines)
        self.wait()

        # Ok ok ok now how do I animate folidng this surface up, and what's a good camera angle for it?
        # In the middle of p49. 
        self.play(self.animate_view(33, 61, 0, (2.54, 0.96, 0.38), 4.24), run_time=2)

        surfaces[3][0].shift([3, 0, 1.5])

        # split_polygons_merged
        # split_polygons_nested_clipped
        #Maybe i transform to split_polygons_nested_clipped, and then once we're flat then merge to split_polygons_merged
        
        split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_nested'][0] for item in sublist]
        split_polygons_unraveled_clipped=[item for sublist in polygons['1.split_polygons_nested_clipped'][0] for item in sublist] #Need to unravel this i think? And then try to animate to it? 

        colors_2 = [gx.GREY, BLUE, BLUE, GREEN, GREEN, YELLOW, YELLOW]
        colors_3 = [gx.GREY, BLUE, GREEN, YELLOW]
        polygons_31=manim_polygons_from_np_list(split_polygons_unraveled, colors=colors_2, viz_scale=viz_scales[2], opacity=0.6)
        polygons_31.shift([3, 0, 1.501]) #Move slightly above map

        polygons_31_clipped=manim_polygons_from_np_list(split_polygons_unraveled_clipped, colors=colors_2, viz_scale=viz_scales[2], opacity=0.6)
        polygons_31_clipped.shift([3, 0, 1.501]) #Move slightly above map

        polygons_31_merged=manim_polygons_from_np_list(polygons['1.split_polygons_merged'][0], colors=colors_3, viz_scale=viz_scales[2], opacity=0.6)
        polygons_31_merged.shift([3, 0, 1.501]) #Move slightly above map

        

        # Man this polygon transform is secretly kinda tricky lol. 
        # What if I grabbed the pre-clipped polygons, and colored them to  match, then moved them? 
        # Ok yeah I think that's the move! Yeah we just swap out the split ones that aren't clipped rigth befor the move I think
        #Ok yeah that's the motion I want -> I'll dropped the dashed lines right before too I think 

        self.wait()
        self.remove(polygons_21); self.add(polygons_31); self.remove(bent_plane_joint_lines)
        self.play(gx.ReplacementTransform(polygons_31, polygons_31_clipped),
                  gx.ReplacementTransform(surfaces[2][0], surfaces[3][0]),
                  gx.FadeOut(relu_intersections_planes_1[1]),
                 run_time=3)
        self.remove(polygons_31_clipped); self.add(polygons_31_clipped)
        self.wait()
  
        # self.play(ReplacementTransform(VGroup(*[polygons_31_clipped[i] for i in [0,2,4,6]]).copy(), polygons_31_merged[0]))

        self.remove(polygons_31_clipped[1], polygons_31_clipped[3], polygons_31_clipped[5])
        self.add(polygons_31_merged[1:]) #, polygons_31_merged[3], polygons_31_merged[5])
        self.play(gx.FadeOut(gx.VGroup(*[polygons_31_clipped[i] for i in [0,2,4,6]])), gx.FadeIn(polygons_31_merged[0])) #Nice!
        self.wait()

        self.play(self.animate_view(-1, 38, 0, (3.12, 1.06, 0.18), 4.24), run_time=6)
        self.wait()

        #Ok now second layer. 
        self.play(gx.FadeIn(relu_intersections_planes_1[0]),
                  self.animate_view(-132, 58, 0, (2.41, 0.82, 0.12), 2.31),
                  run_time=4)
        self.wait()

        surfaces[3][1].shift([3, 0, 0])
        split_polygons_unraveled_2=[item for sublist in polygons['1.split_polygons_nested'][1] for item in sublist]
        split_polygons_unraveled_clipped_2=[item for sublist in polygons['1.split_polygons_nested_clipped'][1] for item in sublist] #Need to unravel this i think? And then try to animate to it? 

        polygons_32=manim_polygons_from_np_list(split_polygons_unraveled_2, colors=colors_2, viz_scale=viz_scales[2], opacity=0.6)
        polygons_32.shift([3, 0, 0.001]) #Move slightly above map

        polygons_32_clipped=manim_polygons_from_np_list(split_polygons_unraveled_clipped_2, colors=colors_2, viz_scale=viz_scales[2], opacity=0.6)
        polygons_32_clipped.shift([3, 0, 0.001]) #Move slightly above map

        colors_4 = [gx.GREY, gx.GREY, BLUE, GREEN, YELLOW]
        polygons_32_merged=manim_polygons_from_np_list(polygons['1.split_polygons_merged'][1], colors=colors_4, viz_scale=viz_scales[2], opacity=0.6)
        polygons_32_merged.shift([3, 0, 0.001]) #Move slightly above map

        self.wait()
        self.remove(polygons_22); self.add(polygons_32); self.remove(bent_plane_joint_lines_2)
        self.play(gx.ReplacementTransform(polygons_32, polygons_32_clipped),
                  gx.ReplacementTransform(surfaces[2][1], surfaces[3][1]),
                  gx.FadeOut(relu_intersections_planes_1[0]),
                 run_time=3)
        self.remove(polygons_32_clipped); self.add(polygons_32_clipped)
        self.wait()

        self.remove(polygons_32_clipped[1], polygons_32_clipped[3], polygons_32_clipped[5], polygons_32_clipped[0])
        self.add(polygons_32_merged[1:]) #, polygons_31_merged[3], polygons_31_merged[5])
        self.play(gx.FadeOut(gx.VGroup(*[polygons_32_clipped[i] for i in [2,4,6]])), gx.FadeIn(polygons_32_merged[0])) #Nice!
        self.wait()


        self.play(self.animate_view(0, 57, 0, (2.72, 0.7, -0.61), 6.70),
                  group_11.animate.set_opacity(0.8), 
                  group_12.animate.set_opacity(0.8), run_time=6)
        self.wait()

        #Ok now cool 2d collapsing down to tiling for first layer and then second, let's go!

        #Go ahead and establish enpoints and then figure out how we want to get there!
        layer_1_polygons_flat=manim_polygons_from_np_list(polygons['0.new_tiling'], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        layer_1_polygons_flat.shift([0, 0, -1.5])

        colors_5=[gx.GREY, BLUE, GREEN, YELLOW, gx.PURPLE, gx.ORANGE, gx.PINK, gx.TEAL, gx.MAROON, gx.GREEN_B]
        layer_2_polygons_flat=manim_polygons_from_np_list(polygons['1.new_tiling'], colors=colors_5, viz_scale=viz_scales[2], opacity=0.6)
        layer_2_polygons_flat.shift([3, 0, -1.5])

        #Ok bring down copies of the Relu folds? maps too or nah
        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/'+map_filename)
        set_surface_shading(flat_map, 0,0,0).set_opacity(0.8)
        flat_map.shift([0, 0, -1.5])

        fold_copy_1=group_11[1].copy()
        fold_copy_2=group_12[1].copy()
        self.wait()
        self.play(fold_copy_1.animate.shift([0,0,-3]), 
                  fold_copy_2.animate.shift([0,0,-1.5]), 
                  gx.ReplacementTransform(group_11[0].copy(), flat_map),
                  gx.ReplacementTransform(group_12[0].copy(), flat_map),
                  run_time=3) 
        self.add(layer_1_polygons_flat)
        self.remove(fold_copy_1); self.add(fold_copy_1)
        self.remove(fold_copy_2); self.add(fold_copy_2)
        self.remove(flat_map)
        self.wait()

        # Ok so for projecting down the second layer stuff, how about just bringing down the outlines? 
        # I think that could be cool!

        outline_4 = polygons_31_merged.copy()
        outline_4.set_fill(opacity=0)
        # outline_4.set_stroke(width=4, opacity=0.9)
        outline_5 = polygons_32_merged.copy()
        outline_5.set_fill(opacity=0)
        # outline_5.set_stroke(width=4, opacity=0.9)
        
        ##Hmm I need a flat version of these down on the bottom to move to. 
        ## Might need to create another little set of polygons?
        polygons_31_merged_flat_arrays=polygons['1.split_polygons_merged'][0]
        for o in polygons_31_merged_flat_arrays: o[:,2]=0 #Flatten that shit
        polygons_31_merged_flat=manim_polygons_from_np_list(polygons_31_merged_flat_arrays, colors=colors_3, viz_scale=viz_scales[2], opacity=0.6)
        polygons_31_merged_flat.set_fill(opacity=0)
        polygons_31_merged_flat.shift([3, 0, -1.5])

        polygons_32_merged_flat_arrays=polygons['1.split_polygons_merged'][1]
        for o in polygons_32_merged_flat_arrays: o[:,2]=0 #Flatten that shit
        polygons_32_merged_flat=manim_polygons_from_np_list(polygons_32_merged_flat_arrays, colors=colors_4, viz_scale=viz_scales[2], opacity=0.6)
        polygons_32_merged_flat.set_fill(opacity=0)
        polygons_32_merged_flat.shift([3, 0, -1.5])

        self.wait()
        self.play(gx.ReplacementTransform(outline_4, polygons_31_merged_flat), run_time=4)
        self.play(gx.ReplacementTransform(outline_5, polygons_32_merged_flat), run_time=3)
        self.wait()

        self.play(
                surfaces[3][1].animate.set_opacity(0.2),
                polygons_32_merged.animate.set_opacity(0.2),
                layer_1_polygons_flat.animate.set_opacity(0.2),
                self.animate_view(0, 44, 0, (3.15, -0.12, -1.73), 3.42),
                run_time=4)
        self.wait()
        self.play(gx.Create(layer_2_polygons_flat), run_time=3)
        self.wait()

        self.play(self.animate_view(-1, 68, 0, (3.09, 0.76, -0.31), 6.95),
                surfaces[3][1].animate.set_opacity(0.6),
                polygons_32_merged.animate.set_opacity(0.8),
                layer_1_polygons_flat.animate.set_opacity(0.25), #Keep projections but don't make them a main focus. 
                layer_2_polygons_flat.animate.set_opacity(0.25),
                polygons_31_merged_flat.animate.set_opacity(0.0),
                polygons_32_merged_flat.animate.set_opacity(0.0),
                run_time=4)
        self.wait()

        # self.remove(polygons_31_merged_flat)

        # Ok now the classic merging things deal, and batch colors to the 2d projections!
        # Ok a little tricky, but should be ok -> kinda dtemps to try to move the wireframe again - that 
        # could make for some nice consistency!
        outline_6 = polygons_31_merged.copy()
        outline_6.set_fill(opacity=0)
        # outline_6.set_stroke(width=4, opacity=0.9)
        outline_7 = polygons_32_merged.copy()
        outline_7.set_fill(opacity=0)
        # outline_7.set_stroke(width=4, opacity=0.9)


        final_layer_middle_tiling_arrays_1=process_with_layers(model.model, polygons_31_merged_flat_arrays)
        final_layer_middle_tiling_1=manim_polygons_from_np_list(final_layer_middle_tiling_arrays_1[0], colors=colors_3, viz_scale=viz_scales[4], opacity=0.6)
        final_layer_middle_tiling_1.set_fill(opacity=0)
        final_layer_middle_tiling_1.shift([6, 0, 1.5])

        final_layer_middle_tiling_arrays_2=process_with_layers(model.model, polygons_32_merged_flat_arrays) 
        final_layer_middle_tiling_2=manim_polygons_from_np_list(final_layer_middle_tiling_arrays_2[0], colors=colors_4, viz_scale=viz_scales[4], opacity=0.6)
        final_layer_middle_tiling_2.set_fill(opacity=0)
        final_layer_middle_tiling_2.shift([6, 0, 1.5])
    
        polygons_41=manim_polygons_from_np_list(polygons['2.linear_out'][0], colors=colors_5, viz_scale=viz_scales[4], opacity=0.6)
        polygons_41.shift([6, 0, 1.501]) #Move slightly above map
        polygons_41_outline=polygons_41.copy()
        polygons_41_outline.set_fill(opacity=0)

        surfaces[4][0].shift([6, 0, 1.5])

        self.wait()
        self.play(gx.ReplacementTransform(outline_6.copy(), final_layer_middle_tiling_1),
                  gx.ReplacementTransform(outline_7.copy(), final_layer_middle_tiling_2),
                  # ReplacementTransform(surfaces[3][0].copy(), surfaces[4][0]),
                run_time=3)
        self.remove(final_layer_middle_tiling_1, final_layer_middle_tiling_2)
        self.add(polygons_41_outline)
        self.wait()


        self.play(gx.ReplacementTransform(layer_2_polygons_flat.copy(), polygons_41), run_time=3)
        self.add(surfaces[4][0]);
        self.remove(polygons_41_outline)
        self.remove(polygons_41); self.add(polygons_41)
        self.wait()


        #Same thing again for second neuron, then zoom in. 
        final_layer_middle_tiling_arrays_1b=process_with_layers(model.model, polygons_31_merged_flat_arrays)
        final_layer_middle_tiling_1b=manim_polygons_from_np_list(final_layer_middle_tiling_arrays_1b[1], colors=colors_3, viz_scale=viz_scales[4], opacity=0.6)
        final_layer_middle_tiling_1b.set_fill(opacity=0)
        final_layer_middle_tiling_1b.shift([6, 0, 0.2]) #move up a smidge

        final_layer_middle_tiling_arrays_2b=process_with_layers(model.model, polygons_32_merged_flat_arrays) 
        final_layer_middle_tiling_2b=manim_polygons_from_np_list(final_layer_middle_tiling_arrays_2b[1], colors=colors_4, viz_scale=viz_scales[4], opacity=0.6)
        final_layer_middle_tiling_2b.set_fill(opacity=0)
        final_layer_middle_tiling_2b.shift([6, 0, 0.2])
    
        polygons_41b=manim_polygons_from_np_list(polygons['2.linear_out'][1], colors=colors_5, viz_scale=viz_scales[4], opacity=0.6)
        polygons_41b.shift([6, 0, 0.201]) #Move slightly above map
        polygons_41_outline_b=polygons_41b.copy()
        polygons_41_outline_b.set_fill(opacity=0)

        surfaces[4][1].shift([6, 0, 0.2])

        self.wait()
        self.play(gx.ReplacementTransform(outline_6, final_layer_middle_tiling_1b),
                  gx.ReplacementTransform(outline_7, final_layer_middle_tiling_2b),
                  # ReplacementTransform(surfaces[3][0].copy(), surfaces[4][0]),
                run_time=3)
        self.remove(final_layer_middle_tiling_1b, final_layer_middle_tiling_2b)
        self.add(polygons_41_outline_b)
        self.wait()


        self.play(gx.ReplacementTransform(layer_2_polygons_flat.copy(), polygons_41b), run_time=3)
        self.add(surfaces[4][1]);
        self.remove(polygons_41_outline_b)
        self.remove(polygons_41b); self.add(polygons_41b)
        self.wait()


        #Maybe a semi-overhead view for a moment is nice? Can change later if I hate it. 
        # self.set_view(-1, 68, 0, (3.09, 0.76, -0.31), 6.95)
        self.play(self.animate_view(-29, 48, 0, (6.05, 0.45, -0.12), 6.53), run_time=4)
        self.wait()

        #Now bring surfaces together and change colors for final layer -> and finally decision boundary lol. 
        polygons_51=polygons_41.copy()
        polygons_52=polygons_41b.copy()
        surface_51=surfaces[4][0].copy()
        surface_52=surfaces[4][1].copy()

        polygons_51.shift([3, 0, -0.7])
        polygons_52.shift([3, 0, 0.6])
        surface_51.shift([3, 0, -0.7])
        surface_52.shift([3, 0, 0.6])
        polygons_51.set_color(BLUE)
        polygons_52.set_color(YELLOW)


        # loops=order_closed_loops_with_closure(intersection_lines)
        lines=gx.VGroup()
        for loop in intersection_lines: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=5)
            lines.add(line)
        lines.shift([9, 0, 0.8])

        surface_51.set_opacity(0.2)
        surface_52.set_opacity(0.9)
        polygons_51.set_opacity(0.4)
        polygons_52.set_opacity(0.5)

        self.wait()
        self.play(self.animate_view(-11, 31, 0, (7.76, 0.49, 0.19), 5.53),
                  gx.ReplacementTransform(surfaces[4][0].copy(), surface_51),
                  gx.ReplacementTransform(surfaces[4][1].copy(), surface_52),
                  gx.ReplacementTransform(polygons_41.copy(), polygons_51),
                  gx.ReplacementTransform(polygons_41b.copy(), polygons_52),
                  run_time=4)
        self.remove(polygons_51); self.add(polygons_51)
        self.remove(polygons_52); self.add(polygons_52)
        self.wait()

        self.play(gx.Create(lines), run_time=2)
        self.wait()

        #Final step here, I think 3d flat projection showing decision boundary as clearly as I can. 
        #Definitely top polytopes flat is the vibe!


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
            poly_3d.shift([9, 0, -1.499])
            top_polygons_vgroup_flat.add(poly_3d)

        polygon_arrays_1_flat=copy.deepcopy(polygons['2.linear_out'][0])
        for p in polygon_arrays_1_flat: p[:,2]=0

        polygon_arrays_2_flat=copy.deepcopy(polygons['2.linear_out'][1])
        for p in polygon_arrays_2_flat: p[:,2]=0

        polygons_51_flat=manim_polygons_from_np_list(polygon_arrays_1_flat, colors=colors, viz_scale=viz_scales[2])
        polygons_51_flat.shift([9, 0, -1.499]) #Move slightly above map
        polygons_52_flat=manim_polygons_from_np_list(polygon_arrays_2_flat, colors=colors, viz_scale=viz_scales[2])
        polygons_52_flat.shift([9, 0, -1.499]) #Move slightly above map
        polygons_51_flat.set_color(YELLOW)
        polygons_51_flat.set_color(BLUE)
        polygons_51_flat.set_opacity(0.5)
        polygons_52_flat.set_opacity(0.5)

        # def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map_2=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/'+map_filename)
        set_surface_shading(flat_map_2, 0,0,0).set_opacity(0.8)
        flat_map_2.shift([9, 0, -1.5])

        lines_flat=gx.VGroup()
        for loop in intersection_lines: 
            # loop=np.hstack((loop, np.zeros((len(loop), 1))))
            loop[:,2]=0
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=4)
            lines_flat.add(line)
        lines_flat.shift([9, 0, -1.5])      

        polygons_51_copy=polygons_51.copy()
        polygons_52_copy=polygons_52.copy()
        surface_51_copy=surface_51.copy()
        polygons_51_copy.set_opacity(0.3)
        polygons_52_copy.set_opacity(0.3)
        surface_51_copy.set_opacity(0.3)

        self.wait()
        self.play(gx.ReplacementTransform(polygons_51_copy, polygons_51_flat),
                    gx.ReplacementTransform(polygons_52_copy, polygons_52_flat),
                    # ReplacementTransform(surface_51_copy, flat_map_2),
                    gx.ReplacementTransform(lines.copy(), lines_flat),
                    self.animate_view(0, 39, 0, (9.07, -0.55, -0.82), 3.80),
                    run_time=3)
        self.add(flat_map_2)
        self.remove(polygons_51_flat, polygons_52_flat)
        self.add(top_polygons_vgroup_flat)
        self.remove(lines_flat); self.add(lines_flat)
        self.wait()

        #Book pause. 
        self.set_view(0, 59, 0, (4.47, 0.45, -0.17), 8.13)
        self.wait(5)



        # #Quick Overview/summary 
        # self.play(self.animate_view(0, 57, 0, (4.6, -0.08, -0.8), 8.34),
        #         layer_1_polygons_flat.animate.set_opacity(0.55), 
        #         layer_2_polygons_flat.animate.set_opacity(0.55),
        #         run_time=6)
        # self.wait()

        ##Clear everything and render simple overhead for use in side by side in p53
        layer_2_polygons_flat.shift([1.7, 0, 0])
        flat_map_2.shift([-1.7, 0, 0])
        top_polygons_vgroup_flat.shift([-1.7, 0, 0])
        lines_flat.shift([-1.7, 0, 0])

        self.clear()
        self.set_view(0, 50, 0, (4.32, -0.02, -0.75), 8.18)
        self.add(flat_map_2)
        self.add(top_polygons_vgroup_flat)
        self.add(lines_flat)
        self.add(layer_2_polygons_flat)
        self.wait()




        self.wait(20)
        pass


class p52b(WelchScene):
    def construct(self):

        w1 = np.array([[-1.8741, 2.12215],
         [-2.39381, -1.24014],
         [-0.940185, 1.40271],
         [2.04548, 0.489156]], dtype=np.float32)
        b1 = np.array([-0.00892048, -1.32954, 1.71349, 0.940607], dtype=np.float32)
        w2 = np.array([[2.56674, 2.26244, -1.40175, 0.737865],
         [-2.58904, -3.0681, 1.08007, -1.32219]], dtype=np.float32)
        b2 = np.array([-0.852075, 0.492386], dtype=np.float32)


        model = BaarleNet([4])

        with torch.no_grad():
            model.model[0].weight.copy_(torch.from_numpy(w1))
            model.model[0].bias.copy_(torch.from_numpy(b1))
            model.model[2].weight.copy_(torch.from_numpy(w2))
            model.model[2].bias.copy_(torch.from_numpy(b2))

        viz_scales=[0.15, 0.15, 0.13]
        num_neurons=[4, 4, 2]


        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=viz_scales[layer_idx])
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



        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        group_11=gx.Group(surfaces[1][0]) #, polygons_11)
        if len(joint_points_11)>0:
            joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.9)
            group_11.add(joint_line_11)

        group_12=gx.Group(surfaces[1][1]) #, polygons_12)
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        if len(joint_points_12)>0:
            joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.9)
            group_12.add(joint_line_12)

        group_13=gx.Group(surfaces[1][2]) #, polygons_11)
        joint_points_13 = get_relu_joint(w1[2,0], w1[2,1], b1[2], extent=1)
        if len(joint_points_13)>0:
            joint_line_13=line_from_joint_points_1(joint_points_13).set_opacity(0.9)
            group_13.add(joint_line_13)

        group_14=gx.Group(surfaces[1][3]) #, polygons_12)
        joint_points_14 = get_relu_joint(w1[3,0], w1[3,1], b1[3], extent=1)
        if len(joint_points_14)>0:
            joint_line_14=line_from_joint_points_1(joint_points_14).set_opacity(0.9)
            group_14.add(joint_line_14)


        group_11.shift([0, 0, 3.0])
        group_12.shift([0, 0, 2.0])
        group_13.shift([0, 0, 1.0])


        polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        polygons_21.shift([0, 0, 0.001]) #Move slightly above map

        polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        polygons_22.shift([0, 0, 0.001]) #Move slightly above map

        group_21=gx.Group(surfaces[2][0], polygons_21)
        group_22=gx.Group(surfaces[2][1], polygons_22)
        group_21.shift([3.0, 0, 2.4])
        group_22.shift([3.0, 0, 1.1])
        group_21.set_opacity(0.5)
        group_22.set_opacity(0.5)


        #Ok now some plane intersction action in a third "panel"
        group_31=group_21.copy()
        group_31[1].set_color(BLUE)
        group_31.shift([3, 0, -0.65])

        group_32=group_22.copy()
        group_32[1].set_color(YELLOW)
        group_32.shift([3, 0, 0.65])

        loops=order_closed_loops_with_closure(intersection_lines)
        lines=gx.VGroup()
        for loop in loops: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=4)
            lines.add(line)
        lines.shift([6, 0, 1.1+0.65])

        #Make Baarle hertog maps a little mroe pronounced. 
        group_31[0].set_opacity(0.9)
        group_32[0].set_opacity(0.9)
        group_31[1].set_opacity(0.4)
        group_32[1].set_opacity(0.4)



        #Don't forget 2d shadows! That's an important juxtoposition. 
        layer_1_polygons_flat=manim_polygons_from_np_list(polygons['0.new_tiling'], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        layer_1_polygons_flat.shift([0, 0, -1.0])


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
            poly_3d.shift([6, 0, -0.99])
            top_polygons_vgroup_flat.add(poly_3d)

        # polygon_arrays_1_flat=copy.deepcopy(polygons['1.linear_out'][0])
        # for p in polygon_arrays_1_flat: p[:,2]=0

        # polygon_arrays_2_flat=copy.deepcopy(polygons['1.linear_out'][1])
        # for p in polygon_arrays_2_flat: p[:,2]=0

        # polygons_51_flat=manim_polygons_from_np_list(polygon_arrays_1_flat, colors=colors, viz_scale=viz_scales[2])
        # polygons_51_flat.shift([9, 0, -1.499]) #Move slightly above map
        # polygons_52_flat=manim_polygons_from_np_list(polygon_arrays_2_flat, colors=colors, viz_scale=viz_scales[2])
        # polygons_52_flat.shift([9, 0, -1.499]) #Move slightly above map
        # polygons_51_flat.set_color(YELLOW)
        # polygons_51_flat.set_color(BLUE)
        # polygons_51_flat.set_opacity(0.5)
        # polygons_52_flat.set_opacity(0.5)

        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map_2=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/'+map_filename)
        set_surface_shading(flat_map_2, 0,0,0).set_opacity(0.8)
        flat_map_2.shift([6, 0, -1.0])

        lines_flat=gx.VGroup()
        for loop in intersection_lines: 
            # loop=np.hstack((loop, np.zeros((len(loop), 1))))
            loop[:,2]=0
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#ec008c', width=4)
            lines_flat.add(line)
        lines_flat.shift([6, 0, -1.0])     



        #Was thinking thsi would be static, but one little collapsing animation migth be good?
        #Eh maybe just like a top to front pan is good? Would be faster obv.

        self.set_view(0, 32, 0, (2.97, -0.05, 0.59), 7.12)
        self.add(group_11, group_12, group_13, group_14)
        self.add(group_21, group_22)
        self.add(group_31, group_32, lines)
        self.add(layer_1_polygons_flat)
        self.add(flat_map_2)
        self.add(top_polygons_vgroup_flat)
        self.add(lines_flat)
        self.wait()


        #Book
        #Let's draw joint lines in illusrtraots
        self.remove(group_11[1], group_12[0], group_12[0], group_13[0])
        self.set_view(0, 56, 0, (2.96, 1.01, 0.78), 7.02)
        self.wait(5)


        #Ok I think simple pan down is all the motion we need, this is quick sentence. 
        #Add netowrk in premeire in the gap below
        # self.play(self.animate_view(-1, 56, 0, (2.97, -0.05, 0.59), 7.12), run_time=8)
        self.wait()

        self.play(gx.FadeOut(group_11),
                  gx.FadeOut(group_12),
                  gx.FadeOut(group_13),
                  gx.FadeOut(group_14),
                  gx.FadeOut(group_21),
                  gx.FadeOut(group_22),
                  gx.FadeOut(group_31),
                  gx.FadeOut(group_32),
                  gx.FadeOut(lines),
                  layer_1_polygons_flat.animate.shift([1.7, 0, 0]),
                  top_polygons_vgroup_flat.animate.shift([-1.7, 0, 0]),
                  lines_flat.animate.shift([-1.7, 0, 0]),
                  flat_map_2.animate.shift([-1.7, 0, 0]),
                  self.animate_view(0, 0, 0, (3.06, -1.13, 0.0), 4.36),
                  run_time=6, 
                  rate_func=gx.linear
                  )

        self.set_view(0, 0, 0, (2.97, 0.01, 0.0), 2.06)
        self.wait() 




        self.wait(20)
        pass






        # polygon_arrays_1_flat=copy.deepcopy(polygons['2.linear_out'][0])
        # for p in polygon_arrays_1_flat: p[:,2]=0

        # polygon_arrays_2_flat=copy.deepcopy(polygons['2.linear_out'][1])
        # for p in polygon_arrays_2_flat: p[:,2]=0

        # polygons_21_flat=manim_polygons_from_np_list(polygon_arrays_1_flat, colors=colors, viz_scale=viz_scales[2], polygon_max_height=polygon_max_height)
        # polygons_21_flat.shift([3, 0, 0.001]) #Move slightly above map
        # polygons_22_flat=manim_polygons_from_np_list(polygon_arrays_2_flat, colors=colors, viz_scale=viz_scales[2], polygon_max_height=polygon_max_height)
        # polygons_22_flat.shift([3, 0, 0.002]) #Move slightly above map
        # polygons_22_flat.set_color(YELLOW)
        # polygons_21_flat.set_color(BLUE)
        # polygons_21_flat.set_opacity(0.1)
        # polygons_22_flat.set_opacity(0.1)


        # self.add(surface_51, surface_52, polygons_51, polygons_52)

    


        # self.add(final_layer_middle_tiling_1[1:], final_layer_middle_tiling_2[1:])

        # self.add(surfaces[4][0])
        
        # self.add(polygons_41)


        #Hmm I guess once i bring wireframes together I could actually bring up the 2d projections to color in the shape, that
        # could be cool. 

        # self.add(polygons_31_merged_flat)
        # self.add(polygons_32_merged_flat)


        # self.add(layer_1_polygons_flat)
        # self.add(layer_2_polygons_flat)
        # self.remove(layer_2_polygons_flat)

        # self.add(polygons_31)
        # self.remove(polygons_31)

        # self.wait()
        
        # self.wait()
        # self.play(ReplacementTransform(surfaces[2][0], surfaces[3][0]), run_time=3)

        # self.add(bent_plane_joint_lines)
        # self.add(pre_move_lines)













