from welch_paths import welch_path
import manimgx as m
from welch_surface import set_surface_shading
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
import manimgx as gx
import numpy as np
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
from plane_folding_utils import CHILL_BROWN, get_relu_joint, line_from_joint_points_1
from welch_geometry import diagram_axes_3d
from welch_scene import WelchScene, image_at_height
from welch_surface import parametric_surface, textured_surface
from welch_text import math_tex
from polytope_intersection_utils import intersect_polytopes
# from decision_boundary_utils import *

graphics_dir=welch_path('backprop_3/graphics/') #Point to folder where map images are
colors = [gx.RED, gx.BLUE, gx.GREEN, gx.YELLOW, gx.PURPLE, gx.ORANGE, gx.PINK, gx.TEAL]


class p3_6(WelchScene):
    def construct(self):

        map_img=image_at_height(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        map_img.scale_to_fit_height(2)  # This sets the height to 2 units (-1 to 1)
        map_img.scale_to_fit_width(2)   # This sets the width to 2 units (-1 to 1)
        map_img.move_to(gx.ORIGIN)

        model = BaarleNet([3])

        w1 = np.array([[-2.00458, 2.24611],
         [-2.56046, -1.21349],
         [-1.94774, 0.716835]], dtype=np.float32)
        b1 = np.array([0.00728259, -1.38003, 1.77056], dtype=np.float32)
        w2 = np.array([[2.46867, 3.78735, -1.90977],
         [-2.55351, -2.95687, 1.74294]], dtype=np.float32)
        b2 = np.array([1.41342, -1.23457], dtype=np.float32)

        with torch.no_grad():
            model.model[0].weight.copy_(torch.from_numpy(w1))
            model.model[0].bias.copy_(torch.from_numpy(b1))
            model.model[2].weight.copy_(torch.from_numpy(w2))
            model.model[2].bias.copy_(torch.from_numpy(b2))

        viz_scales=[0.2, 0.2, 0.13]
        num_neurons=[3, 3, 2]


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



        #Get first layer Relu Joints
        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.9)
        group_11=gx.Group(surfaces[1][0], joint_line_11)
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.9)
        group_12=gx.Group(surfaces[1][1], joint_line_12)
        joint_points_13 = get_relu_joint(w1[2,0], w1[2,1], b1[2], extent=1)
        joint_line_13=line_from_joint_points_1(joint_points_13).set_opacity(0.9)
        group_13=gx.Group(surfaces[1][2], joint_line_13)

        group_11.shift([0, 0, 1.5])
        group_13.shift([0, 0, -1.5])


        #Ok i think a move over while scaling, and then vertical stack for adding is going to make more sense 
        # How i do this lol. 
        

        #Get surfaces after scaling - this is hacky but probably fine
        #If this animation kinda works, then we definitely want to bring along the fold lines!

        surface_func=partial(surface_func_from_model, model=model, layer_idx=1, neuron_idx=0, viz_scale=w2[0, 0]*viz_scales[2])
        bent_surface = parametric_surface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        scaled_surface_1=textured_surface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(scaled_surface_1, 0,0,0).set_opacity(0.9)
        scaled_surface_1.shift([3, 0, 1.5])

        surface_func=partial(surface_func_from_model, model=model, layer_idx=1, neuron_idx=1, viz_scale=w2[0, 1]*viz_scales[2])
        bent_surface = parametric_surface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        scaled_surface_2=textured_surface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(scaled_surface_2, 0,0,0).set_opacity(0.9)
        scaled_surface_2.shift([3, 0, 0])

        surface_func=partial(surface_func_from_model, model=model, layer_idx=1, neuron_idx=2, viz_scale=w2[0, 2]*viz_scales[2])
        bent_surface = parametric_surface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        scaled_surface_3=textured_surface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(scaled_surface_3, 0,0,0).set_opacity(0.9)
        scaled_surface_3.shift([3, 0, -1.5])


        # self.add(scaled_surface_1)
        group_11_scaled=gx.Group(scaled_surface_1, joint_line_11.copy().shift([3,0,0]))
        group_12_scaled=gx.Group(scaled_surface_2, joint_line_12.copy().shift([3,0,0]))
        group_13_scaled=gx.Group(scaled_surface_3, joint_line_13.copy().shift([3,0,0]))



        bent_plane_joint_lines=gx.VGroup()
        pre_move_lines=gx.VGroup()
        #Kinda hacky, but it might actually be easiest just to pick out edges from my polygons?
        # Line Section 1
        line_start=polygons['1.linear_out'][0][0][2]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][0][1]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0.0])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][0][2]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][0][1]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, -1.5])
        pre_move_lines.add(joint_line)

        # Line Section 2
        line_start=polygons['1.linear_out'][0][1][3]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][1][4]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0.0])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][1][3]*np.array([1,1,0])
        line_end=polygons['1.linear_out'][0][1][4]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        pre_move_lines.add(joint_line)

        # Line Section 3
        line_start=polygons['1.linear_out'][0][2][3]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][2][0]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0.0])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][2][3]*np.array([1,1,0])
        line_end=polygons['1.linear_out'][0][2][0]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        pre_move_lines.add(joint_line)

        #Line Section 4
        line_start=polygons['1.linear_out'][0][2][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][2][1]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0.0])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][2][0]*np.array([1,1,0])
        line_end=polygons['1.linear_out'][0][2][1]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0.0])
        pre_move_lines.add(joint_line)

        #Line Section 5
        line_start=polygons['1.linear_out'][0][4][2]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][4][0]*np.array([1,1,viz_scales[2]])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0.0])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][4][2]*np.array([1,1,0])
        line_end=polygons['1.linear_out'][0][4][0]*np.array([1,1,0])
        joint_line = gx.DashedLine(start=line_start, end=line_end, color=gx.WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0.0])
        pre_move_lines.add(joint_line)

        #Ok grr bu tthis kinda makes sense -> pretty sure i need to predivide 2 of my lines before moving

        surfaces[2][0].shift([3,0,0])
        self.wait()


        self.set_view(-8, 57, 0, (3.16, 0.5, -0.45), 7.91)
        self.add(group_11, group_12, group_13)

        #Scale and move over planes - I think one at a time is actually better. 
        self.wait()
        self.play(gx.ReplacementTransform(group_11.copy(), group_11_scaled),
                  # ReplacementTransform(group_12.copy(), group_12_scaled),
                  # ReplacementTransform(group_13.copy(), group_13_scaled), 
                  self.animate_view(8, 63, 0, (3.25, 0.84, 0.06), 7.46),
                  run_time=3)
        self.wait()

        self.play(gx.ReplacementTransform(group_12.copy(), group_12_scaled), run_time=3)
        self.wait()
        self.play(gx.ReplacementTransform(group_13.copy(), group_13_scaled), run_time=3)
        self.wait()


        self.add(pre_move_lines)
        self.remove(group_11_scaled[1], group_12_scaled[1], group_13_scaled[1])
        self.play(gx.ReplacementTransform(scaled_surface_1, surfaces[2][0]),
                  gx.ReplacementTransform(scaled_surface_2, surfaces[2][0]),
                  gx.ReplacementTransform(scaled_surface_3, surfaces[2][0]),
                  gx.ReplacementTransform(pre_move_lines, bent_plane_joint_lines), run_time=3)
        self.wait()

        #Ok dope 
        polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2])
        polygons_21.shift([3, 0, 0.001]) #Move slightly above map

        self.play(self.animate_view(18, 52, 0, (3.23, 0.7, -0.1), 6.18), run_time=2.5)
        self.wait()
        for p in polygons_21:
            self.add(p)
            self.wait(0.2)
        self.wait()

        # Reframe and clear room for second neuron
        # Cant quite decide if I need a little "guide/reference network along the way - ya know?"

        self.play(self.animate_view(-8, 65, 0, (3.04, 0.61, -0.23), 7.05),
                  bent_plane_joint_lines.animate.set_opacity(0.0), 
                  bent_plane_joint_lines.animate.shift([0,0,0.8]).set_opacity(0.0),
                  surfaces[2][0].animate.shift([0,0,0.8]),
                  polygons_21.animate.shift([0,0,0.8]), run_time=3)
        self.wait()


        surfaces[2][1].shift([3,0,-0.8])
        polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2])
        polygons_22.shift([3, 0, -0.799])

        self.play(gx.ReplacementTransform(surfaces[1][0].copy(), surfaces[2][1]),
                  gx.ReplacementTransform(surfaces[1][1].copy(), surfaces[2][1]),
                  gx.ReplacementTransform(surfaces[1][2].copy(), surfaces[2][1]),
                  run_time=3)
        self.wait()
        self.play(gx.FadeIn(polygons_22))
        self.wait()

        self.play(self.animate_view(17, 53, 0, (3.06, 0.86, -0.02), 7.05), run_time=3.0)
        self.wait()

        # self.add(polygons_22)
        # self.add(surfaces[2][1], polygons_22)

        # Ok, now we're brinign our planes together "onto the same axis"
        # Probalby should draw actually axis?
        # Wondering if I make a monochromatic copy first - modify script a little to 
        # Ok ok ok here's what I think I want to do 
        # Let's actually create two basic 3d axes, and I can label the z axis with something like
        # ~P(Netherlands)
        # I can then bring these two axes together. 

        axes_1 = diagram_axes_3d(
            x_range=[-1.2, 1.2, 1],
            y_range=[-1.2, 1.2, 1],
            z_range=[-0.7, 0.7, 1],
            height=2.4,
            width=2.4,
            depth=1.4,
            axis_config={
                "include_ticks": False,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip": True,
                "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
            }
        )

        axes_1.shift([3, 0, 0.8])


        # self.add(axes_1)
        self.wait()
        self.play(self.animate_view(0, 71, 0, (2.65, 1.0, -0.06), 6.31),
                  gx.FadeIn(axes_1),
                  group_11[0].animate.set_opacity(0.7),
                  group_11[1].animate.set_opacity(0.4),
                  group_12[0].animate.set_opacity(0.7),
                  group_12[1].animate.set_opacity(0.4),
                  group_13[0].animate.set_opacity(0.7),
                  group_13[1].animate.set_opacity(0.4),
                  run_time=3)

        netherlands_label=math_tex(r'\sim P(Netherlands)', font_size=20) #, alignment='left')
        netherlands_label.set_color(gx.BLUE)
        netherlands_label.rotate(90*gx.DEGREES, [1, 0, 0])
        # netherlands_label.next_to(axes_1, direction=LEFT, buff=0.05)
        netherlands_label.move_to([3.85, 0, 1.65])
        # self.add(netherlands_label)
        self.play(gx.Write(netherlands_label))
        self.wait()


        axes_2 = axes_1.copy()
        axes_2.shift([0, 0, -2*0.8])
        self.play(gx.FadeIn(axes_2))


        belgium_label=math_tex(r'\sim P(Belgium)', font_size=20) #, alignment='left')
        belgium_label.set_color(gx.YELLOW)
        belgium_label.rotate(90*gx.DEGREES, [1, 0, 0])
        belgium_label.move_to([3.65, 0, 0])
        self.play(gx.Write(belgium_label))
        self.wait()


        self.play(polygons_21.animate.set_color(gx.BLUE), run_time=1.5)
        self.play(polygons_22.animate.set_color(gx.YELLOW), run_time=1.5)

        self.remove(netherlands_label, belgium_label)
        self.play(polygons_21.animate.shift([0, 0, -0.8]), 
                  surfaces[2][0].animate.shift([0, 0, -0.8]), 
                  axes_1.animate.shift([0, 0, -0.8]), 
                  polygons_22.animate.shift([0, 0, 0.8]),
                  surfaces[2][1].animate.shift([0, 0, 0.8]), 
                  axes_2.animate.shift([0, 0, 0.8]),
                  # self.animate_view(-7, 40, 0, (3.21, 0.51, -0.66), 3.95),
                  group_11[0].animate.set_opacity(0.0),
                  group_11[1].animate.set_opacity(0.0),
                  group_12[0].animate.set_opacity(0.0),
                  group_12[1].animate.set_opacity(0.0),
                  group_13[0].animate.set_opacity(0.0),
                  group_13[1].animate.set_opacity(0.0),      
                  run_time=3)
        self.wait()

        #Trace outline, maybe while zooming in?

        intersection_points_raveled=np.array(intersection_lines).reshape(8, 3)
        intersection_points_raveled=intersection_points_raveled*np.array([1, 1, viz_scales[2]])
        # intersection_points_raveled=intersection_points_raveled[(0, 1, 5, 6, 2),:] #Change ordering for smooth animation in
        intersection_points_raveled=intersection_points_raveled[(2, 6, 5, 1, 0),:] #Change ordering for smooth animation in

        line = gx.VMobject()
        line.set_points_as_corners(intersection_points_raveled)
        line.set_stroke(color='#FF00FF', width=5)
        line.shift([3, 0, 0])
        # self.add(line)


        # self.remove(line)
        self.wait()
        self.play(gx.Create(line),
                 self.animate_view(-16, 45, 0, (3.02, 0.49, -0.61), 4.31),
                 run_time=5)
        self.wait()

        # Ok phew getting closer here. 
        # No I need to flatten everything and show final border, and move to an overhead view!
        # And then I think simple heatmaps yellow/blue regions fade in when I say Belgium/Netherlands at the end
        # Wonder if it makes sense to add back in the plot legend here?
        # Let me try to flatten than map first. 

        # map_img.move_to([3, 0, 0])
        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(flat_map, 0,0,0).set_opacity(0.8)
        flat_map.shift([3, 0, 0])


        self.wait()

        self.remove(polygons_21, polygons_22, axes_1, axes_2)
        self.play(
                  # polygons_21.animate.set_opacity(0.0),
                  # polygons_22.animate.set_opacity(0.0), 
                  # FadeOut(axes_1), FadeOut(axes_2),
                  gx.ReplacementTransform(surfaces[2][0], flat_map),
                  gx.ReplacementTransform(surfaces[2][1], flat_map),
                  # line.animate.shift([0,0,0.05]),
                  self.animate_view(0, 0, 0, (3.0, -0.01, -0.67), 3.82),
                  run_time=4)
        self.remove(line); self.add(line) #Occlusions
        self.wait()

        # Ok ok ok ok ok for this last little region highlighting thing -> i kinda feel like I want to actually higlight the 
        # regions of the map, which i can do crossfadding to different textures - I'll do that next
        # We'll then be in pretty good shape, buuut I think it's worth thinking about how the ball and stick overlay 
        # Will work exactly - that's going to matter alot in p6! I assum it needs to be in manim at some point,
        # but it's probably ok to be in illustrator at the very beginning? 
        # This opening scene is taking some time, but hopefully there's some good reusable approaches. 
        flat_map_belgium=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-18.png')
        set_surface_shading(flat_map_belgium, 0,0,0).set_opacity(0.0)
        flat_map_belgium.shift([3, 0, 0.01])

        flat_map_netherlands=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-19.png')
        set_surface_shading(flat_map_netherlands, 0,0,0).set_opacity(0.0)
        flat_map_netherlands.shift([3, 0, 0.01])
        self.add(flat_map_belgium, flat_map_netherlands)

        self.wait()
        self.play(flat_map_belgium.animate.set_opacity(0.6))
        # self.remove(line); self.add(line) #Occlusions
        self.wait()
        self.play(flat_map_netherlands.animate.set_opacity(0.6))


        self.add()




        # self.play(ReplacementTransform(surfaces[2][0], flat_map), 
        #           ReplacementTransform(surfaces[2][1], flat_map))


        # i=1
        # intersection_line=Line(intersection_lines[i][0], intersection_lines[i][0])
        # intersection_line.shift([3, 0, 0])
        # self.add(intersection_line)
        # intersection_line.set_stroke(width=5).set_color('#FF00FF')

        # self.remove(axes_1)

        # Eh I'm kinda starting to think that keeping these surfaces in the sample place might make more sense?
        # At least for this first walk through?
        # yeah let's try leaving this in place, and adding axes/labels to existing structure, then change color, and maybe 
        # with a little camera move. 


        # surface_21_copy=surfaces[2][0].copy()
        # surface_21_copy.shift([3, 0, 0])
        # polygons_21_copy=polygons_21.copy()
        # polygons_21_copy.shift([3, 0, 0]) #.set_color(BLUE).set_opacity(0.3)
        # self.add(surface_21_copy, polygons_21_copy)
        # self.wait()
        # self.play(self.animate_view(1, 72, 0, (3.32, 0.84, 0.01), 7.05), run_time=3)
        # polygons_21_copy=polygons_21.copy()
        # polygons_21_copy.shift([0,0,0.01])

        # #This move is still a little annoying/noisy - not sure how to fix it exactly. 
        # self.wait()
        # self.play(
        #           surfaces[2][0].copy().animate.shift([3, 0, 0]), 
        #           polygons_21_copy.animate.shift([3, 0, 0]), 
        #           # polygons_21.copy().animate.shift([3, 0, 0.00]).set_color(BLUE).set_opacity(0.3),
        #           run_time=3)
        # self.wait()

        # self.add(axes_1)



        self.wait(20)
        pass







      # ReplacementTransform(group_11_scaled[1].copy(), bent_plane_joint_lines[1]),
      # ReplacementTransform(group_11_scaled[1], bent_plane_joint_lines[2]), 
      # ReplacementTransform(group_12_scaled[1].copy(), bent_plane_joint_lines[3]),
      # ReplacementTransform(group_12_scaled[1], bent_plane_joint_lines[4]),
      # ReplacementTransform(group_13_scaled[1], bent_plane_joint_lines[0]),
      # run_time=3.0)

        # self.wait()



        # self.add(bent_plane_joint_lines)
        # self.remove(bent_plane_joint_lines)


        # self.play(ReplacementTransform(group_11_scaled[1].copy(), bent_plane_joint_lines[1]),
        #           ReplacementTransform(group_11_scaled[1], bent_plane_joint_lines[2]), 
        #           ReplacementTransform(group_12_scaled[1].copy(), bent_plane_joint_lines[3]),
        #           ReplacementTransform(group_12_scaled[1], bent_plane_joint_lines[4]),
        #           ReplacementTransform(group_13_scaled[1], bent_plane_joint_lines[0]),
        #           run_time=3.0)


        # joint_line_11_copy=joint_line_11.copy()
        # self.add(joint_line_11_copy)
        # joint_line_11_copy.shift([3, 0, 0])


        # self.wait()
        # self.play(ReplacementTransform(surfaces[1][0].copy(), scaled_surface_1), 
        #           ReplacementTransform(surfaces[1][1].copy(), scaled_surface_2), 
        #           ReplacementTransform(surfaces[1][2].copy(), scaled_surface_3),
        #           joint_line_11_copy.animate.shift([3,0,0]),
        #            run_time=3)
        # self.wait()

        # Ok so, moving these maps over first feels good!
        # Big thing I'm missing is my 3 dotted lines on my final merged surface
        # let me go ahead and make the merged surface, then I'll figure this out. 

        # surfaces[2][0].shift([3,0,0])
        # self.play(ReplacementTransform(scaled_surface_1, surfaces[2][0]),
        #           ReplacementTransform(scaled_surface_2, surfaces[2][0]),
        #           ReplacementTransform(scaled_surface_3, surfaces[2][0]), run_time=3.0)
        # self.wait()






        #Figure out end point, then replacement transform. 
        #Trying to decide if we introduce polygons visually on the 
        # bent planes - maybe! Could modify the script a smidge



        # group_21=Group(surfaces[2][0], polygons_21) #Add polygons
        # group_21.shift([3, 0, 0.0])
        # self.add(group_21)



        # self.play(ReplacementTransform(surfaces[1][0].copy(), surfaces[2][0]), 
        #           ReplacementTransform(surfaces[1][1].copy(), surfaces[2][0]),
        #           ReplacementTransform(surfaces[1][2].copy(), surfaces[2][0]),
        #           run_time=3)




        #Hmm so we definitely want to bring along the lines in some capacity - visually the shaded regions are nice



        # for layer_idx, sl in enumerate(surfaces):
        #     for neuron_idx, s in enumerate(sl):
        #         s.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
        #         self.add(s)

        # self.wait()






