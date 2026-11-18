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



class p47d(WelchScene):
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



        # self.wait()
        # self.play(FadeIn(group_11[0]), FadeIn(group_12[0]), FadeIn(pre_move_lines))
        # self.wait()

        # # Ok I think it's imporant here (and maybe in p46 above) to actually do the brining the 
        # # lines together animation, as annoying as it is
        # # It's only 2 lines - I can do it -> and I've found a way to make it work -> it's just annoying!

        # # self.remove(group_11[1]); self.remove(group_12[1]) #Remove existing Relu lines to replace with "premove lines"
        # # self.add(pre_move_lines)


        # # self.add(joint_line)  
        # self.wait()
        # self.play(ReplacementTransform(surfaces[1][0].copy(),surfaces[2][0]),
        #           ReplacementTransform(surfaces[1][1].copy(),surfaces[2][0]),
        #           ReplacementTransform(pre_move_lines.copy(), bent_plane_joint_lines), 
        #             run_time=3)
        # self.add(polygons_21)
        # self.remove(bent_plane_joint_lines); self.add(bent_plane_joint_lines)
        # self.wait()

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


        # self.wait()
        # self.play(ReplacementTransform(surfaces[1][0].copy(),surfaces[2][1]),
        #           ReplacementTransform(surfaces[1][1].copy(),surfaces[2][1]),
        #           ReplacementTransform(pre_move_lines_2.copy(), bent_plane_joint_lines_2), 
        #           run_time=3)
        # self.add(polygons_22)
        # self.remove(bent_plane_joint_lines_2); self.add(bent_plane_joint_lines_2)
        # self.wait()

        # #Ok now quick "fake out 3 layer", then remove everything from that. 
        # self.play(ReplacementTransform(group_21.copy(), group_31_fakeout), 
        #           ReplacementTransform(group_22.copy(), group_32_fakeout), 
        #          run_time=3.0)
        # # self.play(ShowCreation(lines_fakeout))
        # self.wait()
        # self.play(FadeOut(group_31_fakeout), FadeOut(group_32_fakeout))
        # self.wait()



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
                    "stroke_color": gx.BLACK,
                    "stroke_width": 1.2,
                    "stroke_opacity": 0.9
                },
                faded_line_style={
                    "stroke_color": gx.BLACK,
                    "stroke_width": 0.0,
                    "stroke_opacity": 0.0
                },
                axis_config={
                    "stroke_color": gx.BLACK,
                    "stroke_width": 1.2
                }
            ).scale_to_fit_width(2).scale_to_fit_height(2)
            plane.shift([3, 0, 1.5*neuron_idx])
            relu_intersections_planes_1.add(plane)
        
        # self.wait()
        # self.play(self.animate_view(30, 70, 0, (2.55, 1.16, 0.93), 4.27),
        #           group_11.animate.set_opacity(0.2), 
        #           group_12.animate.set_opacity(0.2),
        #           pre_move_lines.animate.set_opacity(0.0),
        #           run_time=3)
        # self.play(ShowCreation(relu_intersections_planes_1[1]))
        # self.wait()

        # self.play(self.animate_view(32, 83, 0, (2.55, 1.16, 0.93), 4.27), run_time=3)
        # self.wait()




        #Outline to call out planes, I can cut this if I don't like it. 
        outline = polygons_21[1].copy()
        outline.set_fill(opacity=0)
        outline.set_stroke(width=4, opacity=0.9)
        # self.play(ShowCreation(outline, run_time=2))
        # self.play(FadeOut(outline))

        outline_2 = polygons_21[3].copy()
        outline_2.set_fill(opacity=0)
        outline_2.set_stroke(width=4, opacity=0.9)
        # self.play(ShowCreation(outline_2), FadeOut(outline), run_time=2)

        outline_3 = polygons_21[2].copy()
        outline_3.set_fill(opacity=0)
        outline_3.set_stroke(width=4, opacity=0.9)
        # self.play(ShowCreation(outline_3), FadeOut(outline_2), run_time=2)
        # self.play(FadeOut(outline_3))
        # self.remove(bent_plane_joint_lines); self.add(bent_plane_joint_lines)
        # self.wait()

        # Ok ok ok now how do I animate folidng this surface up, and what's a good camera angle for it?
        # In the middle of p49. 
        # self.play(self.animate_view(33, 61, 0, (2.54, 0.96, 0.38), 4.24), run_time=2)

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



        # ---- Thinking through book ads/figures. 
        # Hmm a straight on side/call-out view might be nice!
        # Ok let's modularize the problem first and really focus in on the first 2 layer fold deal
        self.wait()

        group_11[0].move_to(gx.ORIGIN)
        # self.set_view(0, 44, 0, (-0.0, -0.13, -0.16), 2.74)
        self.set_view(0, 58, 0, (-0.0, -0.13, -0.16), 2.74)
        self.add(group_11[0])
        self.wait()
        self.remove(group_11[0])


        group_12[0].move_to(gx.ORIGIN)
        self.add(group_12[0])
        self.wait()
        self.remove(group_12[0])

        new_group_1=gx.Group(surfaces[2][0], polygons_21, relu_intersections_planes_1[1])
        new_group_1.move_to(gx.ORIGIN)
        # surfaces[2][0].move_to(ORIGIN)
        # polygons_21.move_to(ORIGIN)
        self.add(new_group_1[:2])
        self.wait()
        self.add(new_group_1[2])
        self.wait()

        self.set_view(44, 88, 0, (-0.0, -0.13, -0.16), 2.74)
        self.wait()
        self.remove(new_group_1[2])
        self.set_view(0, 58, 0, (-0.0, -0.13, -0.16), 2.74)
        self.wait()
        self.remove(new_group_1)

        new_group_2=gx.Group(surfaces[2][1], polygons_22, relu_intersections_planes_1[0])
        new_group_2.move_to(gx.ORIGIN)
        self.set_view(1, 45, 0, (-0.0, -0.13, -0.16), 2.74)
        # surfaces[2][0].move_to(ORIGIN)
        # polygons_21.move_to(ORIGIN)
        self.add(new_group_2[:2])
        self.wait()
        self.add(new_group_2[2])
        self.wait()

        self.set_view(-123, 65, 0, (0.02, -0.05, -0.24), 2.74)
        self.wait()
        self.remove(new_group_2[2])
        self.set_view(1, 45, 0, (-0.0, -0.13, -0.16), 2.74)
        self.wait()
        self.remove(new_group_2)

        self.set_view(0, 58, 0, (-0.0, -0.13, -0.16), 2.74)


        # relu_intersections_planes_1.move_to(ORIGIN)
        # self.add(relu_intersections_planes_1[1])


        # self.remove(surfaces[2][0], polygons_21)
        # self.add(surfaces[2][1])
        # self.add(polygons_22)
  
        self.wait()

        new_group_3=gx.Group(surfaces[3][0], polygons_31_clipped, polygons_31_merged)
        new_group_3.move_to(gx.ORIGIN)


        self.add(surfaces[3][0])
        self.add(polygons_31_clipped)
        self.wait()

        self.remove(polygons_31_clipped)
        self.add(polygons_31_merged)
        self.wait()

        self.remove(new_group_3)
        self.wait()


        new_group_4=gx.Group(surfaces[3][1], polygons_32_clipped, polygons_32_merged)
        new_group_4.move_to(gx.ORIGIN)

        self.set_view(0, 50, 0, (-0.02, -0.15, -0.19), 2.74)
        self.add(surfaces[3][1])
        self.add(polygons_32_clipped)
        self.wait()

        self.remove(polygons_32_clipped)
        self.add(polygons_32_merged)
        self.wait()

        self.remove(new_group_4)
        self.wait()


        # self.remove(surfaces[3][0])

        
        # self.remove(polygons_31_clipped)

        # self.add(polygons_31_merged)
        # self.remove(polygons_31_merged)

        # self.wait()
        # self.add(surfaces[3][1])
        # self.remove(surfaces[3][1])

        # self.add(polygons_32_clipped)
        # self.remove(polygons_32_clipped)

        # self.add(polygons_32_merged)
        # self.remove(polygons_32_merged)

        # self.add(layer_1_polygons_flat)
        # self.add(layer_2_polygons_flat)
        # self.add(flat_map)

        # self.add(fold_copy_1)
        # self.add(fold_copy_2)

        ## --- End Book stuff







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
            line.set_stroke(color='#FF00FF', width=4)
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
            line.set_stroke(color='#FF00FF', width=4)
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

        #Ok I think simple pan down is all the motion we need, this is quick sentence. 
        #Add netowrk in premeire in the gap below
        self.play(self.animate_view(-1, 56, 0, (2.97, -0.05, 0.59), 7.12), run_time=8)
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













