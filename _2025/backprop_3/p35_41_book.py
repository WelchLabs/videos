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
    order_closed_loops_with_closure,
    process_with_layers,
    recompute_tiling_general,
    split_polygons_with_relu_simple,
)
from plane_folding_utils import get_relu_joint, line_from_joint_points_1
from welch_geometry import diagram_axes_3d
from welch_scene import WelchScene
from welch_surface import SampledSurface, parametric_surface, textured_surface
from welch_text import math_tex
from polytope_intersection_utils import intersect_polytopes
from welch_surface import set_surface_shading

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
BLUE2='#00aeef'
GREEN='#00a14b' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
RED='#ed1c24'


graphics_dir=welch_path('ai_book/4_deep_learning/graphics/')
map_filename='baarle_hertog_maps-13.png'
colors = [gx.GREY, BLUE, GREEN, YELLOW, gx.PURPLE, gx.ORANGE, gx.PINK, gx.TEAL]

class LinearPlane(SampledSurface):
    """A plane defined by z = m1*x1 + m2*x2 + b"""
    def __init__(self, axes, m1=0.5, m2=0.3, b=1.0, vertical_viz_scale=0.5, **kwargs):
        self.axes = axes
        self.m1 = m1
        self.m2 = m2 
        self.b = b
        self.vertical_viz_scale=vertical_viz_scale
        super().__init__(
            # u_range=(-12, 12),
            # v_range=(-12, 12),
            u_range=(-1.0, 1.0),
            v_range=(-1.0, 1.0),
            resolution=(64, 64), #Looks nice at 256, but is slow, maybe crank for final
            color='#00FFFF',
            **kwargs
        )
    
    def uv_func(self, u, v):
        # u maps to x1, v maps to x2, compute z = m1*x1 + m2*x2 + b
        x1 = u
        x2 = v
        z = self.vertical_viz_scale*(self.m1 * x1 + self.m2 * x2 + self.b)
        # Transform to axes coordinate system
        return self.axes.c2p(x1, x2, z)

class LinearPlaneWithGrid(gx.Group):
    """A plane with explicit grid lines"""
    def __init__(self, axes, m1=0.5, m2=0.3, b=1.0, vertical_viz_scale=0.5, 
                 grid_lines=20, **kwargs):
        super().__init__()
        
        # Create the main surface
        plane = LinearPlane(axes, m1, m2, b, vertical_viz_scale, **kwargs)
        self.add(plane)
        
        # Create grid lines
        u_range = (-1.0, 1.0)
        v_range = (-1.0, 1.0)
        
        # Vertical grid lines (constant u)
        for i in range(grid_lines + 1):
            u = u_range[0] + i * (u_range[1] - u_range[0]) / grid_lines
            line_points = []
            for j in range(21):  # 21 points along the line
                v = v_range[0] + j * (v_range[1] - v_range[0]) / 20
                x1, x2 = u, v
                z = vertical_viz_scale * (m1 * x1 + m2 * x2 + b)
                line_points.append(axes.c2p(x1, x2, z))
            
            grid_line = gx.VMobject()
            grid_line.set_points_as_corners(line_points)
            grid_line.set_stroke(gx.WHITE, width=0.5, opacity=0.3)
            self.add(grid_line)
        
        # Horizontal grid lines (constant v)
        for i in range(grid_lines + 1):
            v = v_range[0] + i * (v_range[1] - v_range[0]) / grid_lines
            line_points = []
            for j in range(21):  # 21 points along the line
                u = u_range[0] + j * (u_range[1] - u_range[0]) / 20
                x1, x2 = u, v
                z = vertical_viz_scale * (m1 * x1 + m2 * x2 + b)
                line_points.append(axes.c2p(x1, x2, z))
            
            grid_line = gx.VMobject()
            grid_line.set_points_as_corners(line_points)
            grid_line.set_stroke(gx.WHITE, width=0.5, opacity=0.3)
            self.add(grid_line)


class p35_41b(WelchScene):
    def construct(self):

        model = BaarleNet([3])
        pickle_path=welch_path('backprop_3/hackin/3_starting_configs_4/training_data_seed_01_acc_0.8561.pkl')
        with open(pickle_path, 'rb') as f:
            p=pickle.load(f)

        print(len(p['weights_history']), ' training steps loaded.')

        viz_scales=[0.2, 0.2, 0.1]
        num_neurons=[3, 3, 2]

        # train_step=2400
        train_step=7 #Start on step 7, this has the example I want
        w1=p['weights_history'][train_step]['model.0.weight'].numpy()
        b1=p['weights_history'][train_step]['model.0.bias'].numpy()
        w2=p['weights_history'][train_step]['model.2.weight'].numpy()
        b2=p['weights_history'][train_step]['model.2.bias'].numpy()

        with torch.no_grad():
            model.model[0].weight.copy_(torch.from_numpy(w1))
            model.model[0].bias.copy_(torch.from_numpy(b1))
            model.model[2].weight.copy_(torch.from_numpy(w2))
            model.model[2].bias.copy_(torch.from_numpy(b2))


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
            polygons[str(layer_id)+'.split_polygons_nested']=split_polygons_with_relu_simple(polygons[str(layer_id)+'.linear_out']) #Triple nested list so we can simplify merging process layer. 
            polygons[str(layer_id)+'.split_polygons_nested_clipped'] = clip_polygons(polygons[str(layer_id)+'.split_polygons_nested'])
            polygons[str(layer_id)+'.split_polygons_merged'] = merge_zero_regions(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
            polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
            print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')

        #Last linear layer & output
        polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
        intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
        my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)


        #Get first layer Relu Joints - 
        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        group_11=gx.Group(surfaces[1][0])
        if len(joint_points_11)>0:
            joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.9)
            group_11.add(joint_line_11)

        group_12=gx.Group(surfaces[1][1])
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        if len(joint_points_12)>0:
            joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.9)
            group_12.add(joint_line_12)

        group_13=gx.Group(surfaces[1][2])
        joint_points_13 = get_relu_joint(w1[2,0], w1[2,1], b1[2], extent=1)
        if len(joint_points_13)>0:
            joint_line_13=line_from_joint_points_1(joint_points_13).set_opacity(0.9)
            group_13.add(joint_line_13)

        group_11.shift([0, 0, 1.5])
        group_13.shift([0, 0, -1.5])

        polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        polygons_21.shift([0, 0, 0.001]) #Move slightly above map

        polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        polygons_22.shift([0, 0, 0.001]) #Move slightly above map

        group_21=gx.Group(surfaces[2][0], polygons_21)
        group_22=gx.Group(surfaces[2][1], polygons_22)
        group_21.shift([3.0, 0, 0.6])
        group_22.shift([3.0, 0, -0.6])
        group_21.set_opacity(0.5)
        group_22.set_opacity(0.5)


        #Ok now some plane intersction action in a third "panel"
        group_31=group_21.copy()
        group_31[1].set_color(BLUE)
        group_31.shift([3, 0, -0.6])

        group_32=group_22.copy()
        group_32[1].set_color(YELLOW)
        group_32.shift([3, 0, 0.6])

        lines=gx.VGroup()
        for loop in intersection_lines: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines.add(line)
        lines.shift([6, 0, 0])

        #Make Baarle hertog maps a little mroe pronounced. 
        group_31[0].set_opacity(0.9)
        group_32[0].set_opacity(0.9)

        # group_21.set_opacity(0.9)
        # group_22.set_opacity(0.9)

        self.set_view(-1, 46, 0, (3.09, 0.56, -0.42), 7.25)
        self.add(group_12, group_13)
        self.add(group_21, group_22)
        self.add(group_31, group_32, lines)
        self.add(group_11)


        #Hmm i feel like some kinda legend or labelling might be helpful here! Minimal i think is ok, but something
        # I can think on that next. Ok done did it in illustrator. 
        # Now is it worth adding a small final decision bounday ochear kidna deal? 
        # Eh that's going to be a mess in 3d. Let my try making maps a little more proncounced in final surface here. 
        # Alright time to start moving the example point through the model!
        # Can already tell that I probalby want a more detailed play by play in the narration. 

        map_coords_1=math_tex(r'(0.6, 0.4)', font_size=11).set_color('#FF00FF')
        map_pt_1=gx.Dot(gx.ORIGIN, radius=0.015).set_color('#FF00FF')
        map_pt_1.move_to([-0.39, 0.12, 0])
        map_coords_1.move_to([-0.4, 0.04, 0])
        coords_group_1=gx.VGroup(map_pt_1, map_coords_1)
        coords_group_1.scale(2.0)
        coords_group_1.move_to([0.56,0.22,1.5])

        coords_group_2=coords_group_1.copy()
        coords_group_3=coords_group_1.copy()
        coords_group_2.move_to([0.56,0.22,0.0])
        coords_group_3.move_to([0.56,0.22,-1.5])

        coords_group_4=coords_group_2.copy()
        coords_group_5=coords_group_2.copy()
        coords_group_4.shift([3,0.0,0.77])
        coords_group_5.shift([3,0.0,-0.83])

        #Zoom in to overhead of first map and draw on point!
        self.wait()


        # self.add(group_11) #Do i need to fix occlusions again or no? 
        self.play(self.animate_view(0, 3, 0, (-0.01, 0.19, -0.42), 4.81),
                  group_11.animate.set_opacity(1.0),
                 run_time=4)
        self.wait()

        #Now write on points!
        self.play(gx.Write(coords_group_1[1]), gx.FadeIn(coords_group_1[0]), run_time=2)
        self.wait()

        self.play(self.animate_view(-1, 47, 0, (3.25, 0.26, -0.22), 4.15),
                  gx.FadeIn(coords_group_2[0]), #Try just the dot
                  gx.FadeIn(coords_group_3[0]),
                  gx.FadeIn(coords_group_5[0]),
                  gx.FadeIn(coords_group_4[0]),
                  gx.Write(coords_group_4[1]), #But include coords on top layer!
                 run_time=5.0)
        self.wait()

        # Oh man could I do a cool highlight of the linear region I'm talking about?!
        # self.play(Indicate(polygons_21[0], color=GREY))
        # Oh man could I do a cool highlight of the linear region I'm talking about?!
        outline = polygons_21[0].copy()
        outline.set_fill(opacity=0)
        outline.set_stroke('#FF00FF', width=4, opacity=0.9)
        self.play(gx.Create(outline, run_time=2))
        self.wait()

        #Ok ok ok I think i want to go overhead, may switch to top polygons, and for bonus points actually REplaceTransformt eh called
        # out polygon and label! Than I can add netherlands belgium labels in illustrator. 

        # Ok let me think ahead for a minute first though -> I want to show a single gradient descent step
        # And I do think that showing all the actually gradient values on the network would be pretty cool!
        # Anyway though, I need some example that actually makes sense lol -> Let me render out a loop of the first N steps 
        # And see if I can make sense of what's going on -> I may need to combine steps too -> I think that would be ok!

        #Ok so I think that top polygons is probably a nice/decent way to get to an overhead view
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
            poly_3d.set_opacity(0.5)
            poly_3d.shift([6, 0, 0])
            top_polygons_vgroup.add(poly_3d)

        self.wait()

        # self.play(ReplacementTransform(outline.copy(), group_31[1][0]))
        # self.play(coords_group_4.animate.shift([3, 0, -0.6]))

        outline_2=top_polygons_vgroup[0].copy()
        outline_2.set_fill(opacity=0)
        outline_2.set_stroke(BLUE, width=2) #, opacity=0.9)


        coords_group_6=coords_group_4.copy()
        self.wait()
        self.play(self.animate_view(0, 0, 0, (6.15, 0.01, -0.17), 3.55),
                  gx.FadeIn(top_polygons_vgroup),
                  group_31[1].animate.set_opacity(0),
                  group_32[1].animate.set_opacity(0),
                  gx.ReplacementTransform(outline.copy(), outline_2), #group_31[1][0]),
                  coords_group_6.animate.shift([3, 0, -0.6]), 
                  run_time=3
                  )
        self.remove(lines); self.add(lines)
        self.remove(coords_group_6); self.add(coords_group_6)
        self.wait()

        # self.remove(top_polygons_vgroup[0])
        self.play(self.animate_view(0, 42, 0, (2.99, -0.49, -0.81), 7.09), run_time=4) #Eh or schmeH?
        self.wait()

        #Bring in illustrator overlay of full newtwork!
        #Ok now zoom in on final play and add magent non bent plane!


        self.play(self.animate_view(-1, 44, 0, (0.02, -0.84, -1.64), 2.68), run_time=4)
        self.wait()



        axes_1 = diagram_axes_3d(
            # x_range=[-15, 15, 1],
            # y_range=[-15, 15, 1],
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            z_range=[-1, 1, 1],
            width=2,
            height=2.05,
            depth=2,
            axis_config={
                "color": FRESH_TAN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":4,
                "tip_config": {"width":0.08, "length":0.08}
                }
        )

        w=model.model[0].weight.detach().numpy()
        b=model.model[0].bias.detach().numpy()
        # vertical_viz_scale=0.3
        # plane_1=LinearPlaneWithGrid(axes_1, w1[2,0], w[2,1], b1[2], vertical_viz_scale=vertical_viz_scale, grid_lines=12)
        plane_1=LinearPlane(axes_1, w1[2,0], w[2,1], b1[2], vertical_viz_scale=viz_scales[0])
        plane_1.set_opacity(0.5)
        plane_1.set_color('#FF00FF')

        axis_and_plane_31=gx.Group(axes_1, plane_1)
        axis_and_plane_31.shift([0, 0.02, -1.475-0.007])
        # axis_and_plane_31.shift([0, 0.0, -0.007])

        # self.add(plane_1)
        self.wait()
        self.play(gx.Create(plane_1))
        self.wait()

        #ok now add illustrator overlay. 
        #Then update (just layer 1 I think!)


        with open(pickle_path, 'rb') as f:
            p=pickle.load(f)

        train_step=7+10 #Actually move more steps for more clear viz
        w1=p['weights_history'][train_step]['model.0.weight'].numpy()
        b1=p['weights_history'][train_step]['model.0.bias'].numpy()
        w2=p['weights_history'][train_step]['model.2.weight'].numpy()
        b2=p['weights_history'][train_step]['model.2.bias'].numpy()

        with torch.no_grad():
            model.model[0].weight.copy_(torch.from_numpy(w1))
            model.model[0].bias.copy_(torch.from_numpy(b1))
            model.model[2].weight.copy_(torch.from_numpy(w2))
            model.model[2].bias.copy_(torch.from_numpy(b2))


        #Precompute my surfaces, and polygons moving through network
        surfaces_2=[]
        surface_funcs_2=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs_2.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=viz_scales[layer_idx])
                bent_surface = parametric_surface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
                ts=textured_surface(bent_surface, graphics_dir+'/baarle_hertog_maps/'+map_filename)
                set_surface_shading(ts, 0,0,0).set_opacity(0.8)
                s.add(ts)
                surface_funcs_2[-1].append(surface_func)
            surfaces_2.append(s)

        #Move polygons through network
        polygons_2={} #dict of all polygones as we go. 
        polygons_2['-1.new_tiling']=[np.array([[-1., -1, 0], #First polygon is just input plane
                                            [-1, 1, 0], 
                                            [1, 1, 0], 
                                            [1, -1, 0]])]

        for layer_id in range(len(model.model)//2): #Move polygont through layers     
            polygons_2[str(layer_id)+'.linear_out']=process_with_layers(model.model[:2*layer_id+1], polygons_2[str(layer_id-1)+'.new_tiling']) 
            polygons_2[str(layer_id)+'.split_polygons_nested']=split_polygons_with_relu_simple(polygons_2[str(layer_id)+'.linear_out']) #Triple nested list so we can simplify merging process layer. 
            polygons_2[str(layer_id)+'.split_polygons_nested_clipped'] = clip_polygons(polygons_2[str(layer_id)+'.split_polygons_nested'])
            polygons_2[str(layer_id)+'.split_polygons_merged'] = merge_zero_regions(polygons_2[str(layer_id)+'.split_polygons_nested_clipped'])
            polygons_2[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons_2[str(layer_id)+'.split_polygons_merged'])
            print('Retiled plane into ', str(len(polygons_2[str(layer_id)+'.new_tiling'])), ' polygons_2.')

        #Last linear layer & output
        polygons_2[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons_2[str(layer_id)+'.new_tiling'])
        intersection_lines_2, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons_2[str(layer_id+1)+'.linear_out'])
        my_indicator, my_top_polygons_2 = compute_top_polytope(model, new_2d_tiling)


        #Get first layer Relu Joints - 
        joint_points_11_2 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        group_11_2=gx.Group(surfaces_2[1][0])
        if len(joint_points_11_2)>0:
            joint_line_11=line_from_joint_points_1(joint_points_11_2).set_opacity(0.9)
            group_11_2.add(joint_line_11)

        group_12_2=gx.Group(surfaces_2[1][1])
        joint_points_12_2 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        if len(joint_points_12_2)>0:
            joint_line_12=line_from_joint_points_1(joint_points_12_2).set_opacity(0.9)
            group_12_2.add(joint_line_12)

        group_13_2=gx.Group(surfaces_2[1][2])
        joint_points_13_2 = get_relu_joint(w1[2,0], w1[2,1], b1[2], extent=1)
        if len(joint_points_13_2)>0:
            joint_line_13=line_from_joint_points_1(joint_points_13_2).set_opacity(0.9)
            group_13_2.add(joint_line_13)

        group_11_2.shift([0, 0, 1.5])
        group_13_2.shift([0, 0, -1.5])



        polygons_21_2=manim_polygons_from_np_list(polygons_2['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        polygons_21_2.shift([0, 0, 0.001]) #Move slightly above map

        polygons_22_2=manim_polygons_from_np_list(polygons_2['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        polygons_22_2.shift([0, 0, 0.001]) #Move slightly above map

        group_21_2=gx.Group(surfaces_2[2][0], polygons_21_2)
        group_22_2=gx.Group(surfaces_2[2][1], polygons_22_2)
        group_21_2.shift([3.0, 0, 0.6])
        group_22_2.shift([3.0, 0, -0.6])
        group_21_2.set_opacity(0.5)
        group_22_2.set_opacity(0.5)


        #Ok now some plane intersction action in a third "panel"
        group_31_2=group_21_2.copy()
        group_31_2[1].set_color(BLUE)
        group_31_2.shift([3, 0, -0.6])

        group_32_2=group_22_2.copy()
        group_32_2[1].set_color(YELLOW)
        group_32_2.shift([3, 0, 0.6])

        lines_2=gx.VGroup()
        for loop in intersection_lines_2: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines_2.add(line)
        lines_2.shift([6, 0, 0])

        #Make Baarle hertog maps a little mroe pronounced. 
        group_31_2[0].set_opacity(0.9)
        group_32_2[0].set_opacity(0.9)

        # group_21_2.set_opacity(0.9)
        # group_22_2.set_opacity(0.9)

        plane_2=LinearPlane(axes_1, w1[2,0], w[2,1], b1[2], vertical_viz_scale=viz_scales[0])
        plane_2.set_opacity(0.5)
        plane_2.set_color('#FF00FF')

        axis_and_plane_31_2=gx.Group(axes_1, plane_2)
        # axis_and_plane_31_2.shift([0, 0.02, -1.475-0.007])

        plane_2.scale([1, 1.01, 1])
        plane_2.shift([0,0,0.002])
        self.wait()
        self.play(gx.ReplacementTransform(plane_1, plane_2),
                  # ReplacementTransform(surfaces[0][2], surfaces_2[0][2]),
                  gx.ReplacementTransform(group_13, group_13_2),
                  run_time=1)
        self.remove(plane_2); self.add(plane_2)
        self.remove(coords_group_3[0]); self.add(coords_group_3[0])

        self.wait()

        #Ok lets try losing added coords here, maybe everwhere excecpt final output?
        self.remove(coords_group_1, coords_group_2, coords_group_4, coords_group_5, outline, outline_2)
        #Remove top polygons and switch back to surfaces too
        self.remove(top_polygons_vgroup)
        group_31[0].set_opacity(0.9); group_31[1].set_opacity(0.6); #Bring back up these opacities
        group_32[0].set_opacity(0.9); group_32[1].set_opacity(0.6); #Bring back up these opacities

        # plane_2.shift([0,0,-0.001])
        self.wait()
        self.play(self.animate_view(-1, 55, 0, (2.78, 0.37, -0.29), 6.21), run_time=4)
        self.wait()


        self.remove(group_11, group_12, group_21, group_22)
        self.add(group_11_2, group_12_2, group_21_2, group_22_2)
        self.wait()
        # self.add(group_21_2, group_22_2)

        coords_group_7=coords_group_6.copy()
        coords_group_7.shift([0, 0, -0.03])
        self.remove(group_31, group_32, lines)
        self.add(group_31_2, group_32_2, lines_2)
        self.remove(coords_group_6); self.add(coords_group_7)
        self.wait()



        # self.add(group_31_2, group_32_2, lines)
        # Ok lets try running the loop in a different class -> I think that will be easier!


        # self.remove(surfaces_2[1][2])
        # self.add(surfaces_2[1][2])

        # self.remove(surfaces[1][2])


        # self.add(group_12_2, group_13_2)
        # self.add(group_21_2, group_22_2)
        # self.add(group_31_2, group_32_2, lines)
        # self.add(group_11



        # self.add(top_polygons_vgroup)

        # group_31_2[1].set_opacity(0)
        # group_32[1].set_opacity(0)
    

        # self.add(coords_group_5)
        # self.add(coords_group_4)
        # self.add(coords_group_2)
        # self.add(coords_group_3)

        # Ok I think a starting step of 7, then a step size of 5 can make for a nice viz, and I'll focus on the bottom 
        # first plane getting less negative steep - I think that makes sense - just gotta work through the visuals, work on the 
        # Script a bit, then play the whole thing - oh yeah and I need to finish the last part of the move above. 
        # Complicated but cool/important scene I think!
        



        # self.add(coords_group_1)
        # self.add(coords_group_2)
        # self.add(coords_group_3)

        self.wait()


        

class p36_loop_v2b(WelchScene):
    def construct(self):

        model = BaarleNet([3])
        pickle_path=welch_path('backprop_3/hackin/3_starting_configs_4/training_data_seed_01_acc_0.8561.pkl')

        #Maybe I can just switch to this sucker for the non-coverging one? 
        # pickle_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/hackin/3_starting_configs_4/training_data_seed_32_acc_0.6159.pkl'
        
        with open(pickle_path, 'rb') as f:
            p=pickle.load(f)

        print(len(p['weights_history']), ' training steps loaded.')

        viz_scales=[0.2, 0.2, 0.1]
        num_neurons=[3, 3, 2]

        self.set_view(-1, 55, 0, (2.78, 0.37, -0.29), 6.21)

        #Ok having trouble reproducing this training config, so we're going to grab a straring point that 
        # has the same label and is in the same neighborhood!

        # train_step=2400
        start_step=0 #Start here to get the gradietns I want to show
        step_size=1
        # print('starting weights', p['weights_history'][start_step])

        # print('starting grads', p['gradients_history'][start_step])
        # print('empirical grads w1: ', -100*(p['weights_history'][8]['model.0.weight'].numpy()-
        #                                    p['weights_history'][7]['model.0.weight'].numpy())) #-1/lr (0.01)
        # print('empirical grads w2: ', -100*(p['weights_history'][8]['model.2.weight'].numpy()-
        #                                    p['weights_history'][7]['model.2.weight'].numpy())) #-1/lr (0.01)
        # print('empirical grads b1: ', -100*(p['weights_history'][8]['model.0.bias'].numpy()-
        #                                    p['weights_history'][7]['model.0.bias'].numpy())) #-1/lr (0.01)
        # print('empirical grads b2: ', -100*(p['weights_history'][8]['model.2.bias'].numpy()-
        #                                    p['weights_history'][7]['model.2.bias'].numpy())) #-1/lr (0.01)

        # self.wait()
        # In [5]: print('empirical grads: ', -100*(p['weights_history'][8]['model.0.weight'].numpy()-p['weights_history'][7]['model.0.weight'].numpy()))                                      
        # empirical grads:  [[ 0.38974938 -0.45838356]                                                                                                                                        
        #  [ 0.16942024 -0.63158274]                                                                                                                                                          
        #  [-1.0046124   0.9809494 ]]                                                                                                                                                         
        # In [6]: print('starting grads', p['gradients_history'][start_step])                                                                                                                 
        # starting grads {'model.0.weight': tensor([[ 0.0021, -0.0006],                                                                                                                       
        #         [-0.0005,  0.0004],                                                                                                                                                         
        #         [-0.3566,  0.2963]]), 'model.0.bias': tensor([-0.0006, -0.0005, -0.4256]), 'model.2.weight': tensor([[-0.0074, -0.0003, -0.0616],                                           
        #         [ 0.0074,  0.0003,  0.0616]]), 'model.2.bias': tensor([ 0.2037, -0.2037])} 

        #Ok I think there's some issue with my analytical gradients -> updates don't seem to match -> 
        # For numbers on screen I'll show empirical grads then


        group_12 = group_13 = group_21 = group_22 = group_31 = group_32 = lines = group_11 = None
        for train_step in np.arange(start_step, 200): #len(p['weights_history']), step_size):
            if group_12 is not None:
                self.remove(group_12, group_13)
                self.remove(group_21, group_22)
                self.remove(group_31, group_32, lines)
                self.remove(group_11)

            w1=p['weights_history'][train_step]['model.0.weight'].numpy()
            b1=p['weights_history'][train_step]['model.0.bias'].numpy()
            w2=p['weights_history'][train_step]['model.2.weight'].numpy()
            b2=p['weights_history'][train_step]['model.2.bias'].numpy()

            with torch.no_grad():
                model.model[0].weight.copy_(torch.from_numpy(w1))
                model.model[0].bias.copy_(torch.from_numpy(b1))
                model.model[2].weight.copy_(torch.from_numpy(w2))
                model.model[2].bias.copy_(torch.from_numpy(b2))


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
                polygons[str(layer_id)+'.split_polygons_nested']=split_polygons_with_relu_simple(polygons[str(layer_id)+'.linear_out']) #Triple nested list so we can simplify merging process layer. 
                polygons[str(layer_id)+'.split_polygons_nested_clipped'] = clip_polygons(polygons[str(layer_id)+'.split_polygons_nested'])
                polygons[str(layer_id)+'.split_polygons_merged'] = merge_zero_regions(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
                polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
                print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')

            #Last linear layer & output
            polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
            intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
            my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)


            #Get first layer Relu Joints - 
            joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
            group_11=gx.Group(surfaces[1][0])
            if len(joint_points_11)>0:
                joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.95)
                joint_line_11.set_stroke(color=RED, width=11)
                group_11.add(joint_line_11)

            group_12=gx.Group(surfaces[1][1])
            joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
            if len(joint_points_12)>0:
                joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.95)
                joint_line_12.set_stroke(color=BLUE2, width=11)
                group_12.add(joint_line_12)

            group_13=gx.Group(surfaces[1][2])
            joint_points_13 = get_relu_joint(w1[2,0], w1[2,1], b1[2], extent=1)
            if len(joint_points_13)>0:
                joint_line_13=line_from_joint_points_1(joint_points_13).set_opacity(0.95)
                joint_line_13.set_stroke(color=GREEN, width=11)
                group_13.add(joint_line_13)

            group_11.shift([0, 0, 1.5])
            group_13.shift([0, 0, -1.5])

            polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
            polygons_21.shift([0, 0, 0.001]) #Move slightly above map

            polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
            polygons_22.shift([0, 0, 0.001]) #Move slightly above map

            group_21=gx.Group(surfaces[2][0], polygons_21)
            group_22=gx.Group(surfaces[2][1], polygons_22)
            group_21.shift([3.0, 0, 0.6])
            group_22.shift([3.0, 0, -0.6])
            group_21.set_opacity(0.5)
            group_22.set_opacity(0.5)


            #Ok now some plane intersction action in a third "panel"
            group_31=group_21.copy()
            group_31[1].set_color(BLUE)
            group_31.shift([3, 0, -0.6])

            group_32=group_22.copy()
            group_32[1].set_color(YELLOW)
            group_32.shift([3, 0, 0.6])

            lines=gx.VGroup()
            for loop in intersection_lines: 
                loop=loop*np.array([1, 1, viz_scales[2]])
                line = gx.VMobject()
                line.set_points_as_corners(loop)
                line.set_stroke(color='#ec008c', width=11)
                lines.add(line)
            lines.shift([6, 0, 0])

            group_32.add(lines)

            #Make Baarle hertog maps a little mroe pronounced. 
            group_31[0].set_opacity(0.9)
            group_32[0].set_opacity(0.9)

            group_31[1].set_opacity(0.4) #BLue
            group_32[1].set_opacity(0.4) #Yellow

            # group_21.set_opacity(0.9)
            # group_22.set_opacity(0.9)
            #Book lets go

            group_11.move_to(gx.ORIGIN)
            group_12.move_to(gx.ORIGIN)
            group_13.move_to(gx.ORIGIN)
            group_21.move_to(gx.ORIGIN)
            group_22.move_to(gx.ORIGIN)
            group_31.shift([-6, 0, 0])
            group_32.shift([-6, 0, 0])


            self.set_view(0, 50, 0, (0, 0.0, 0), 3.08)
            self.add(group_11)
            self.wait()
            self.remove(group_11)

            self.add(group_12)
            self.wait()
            self.remove(group_12)

            self.add(group_13)
            self.wait()
            self.remove(group_13)

            self.add(group_21)
            self.wait()
            self.remove(group_21)

            self.add(group_22)
            self.wait()
            self.remove(group_22)

            self.add(group_31, group_32, lines)
            self.wait()
            self.remove(group_31, group_32, lines)


            top_polygons_vgroup_flat=gx.VGroup()
            for j, pp in enumerate(my_top_polygons):
                if len(pp)<3: continue
                if my_indicator[j]: color=YELLOW
                else: color=BLUE
                
                p_scaled=copy.deepcopy(pp) #Scaling for viz
                p_scaled[:,2]=0 #p_scaled[:,2]*viz_scales[2] #Flatten that shit!
                # p_scaled[:, -1] = np.clip(p_scaled[:, -1], -polygon_max_height, polygon_max_height)
                poly_3d = gx.Polygon(*p_scaled,
                                 fill_color=color,
                                 fill_opacity=0.4,
                                 stroke_color=color,
                                 stroke_width=2)
                poly_3d.set_opacity(0.3)
                # poly_3d.shift([3, 0, 0])
                top_polygons_vgroup_flat.add(poly_3d)

            top_polygons_vgroup_flat.set_opacity(0.5) #Eh?

            def flat_surf_func(u, v): return [u, v, 0]
            flat_map_surf = parametric_surface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
            flat_map=textured_surface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/'+map_filename)
            set_surface_shading(flat_map, 0,0,0).set_opacity(0.8)
            # flat_map.shift([5.7, 0, 0])

            loops=order_closed_loops_with_closure(intersection_lines)
            lines_flat=gx.VGroup()
            for loop in loops: 
                loop=loop*np.array([1, 1, 0])
                line = gx.VMobject()
                line.set_points_as_corners(loop)
                line.set_stroke(color='#ec008c', width=9)
                lines_flat.add(line)


            #Overhead - eh for this to work I need top polygons -> how hard/annoying is that here?
            self.set_view(0, 0, 0, (-0.0, 0.0, 0.0), 3.08)

            self.add(flat_map)
            self.add(top_polygons_vgroup_flat)
            # self.add(lines_flat)

            self.wait()

            self.remove(flat_map)
            self.remove(top_polygons_vgroup_flat)
            # self.remove(lines_flat)
        

            # self.add(group_11)
            # self.wait()
            # self.remove(group_11)


            # self.set_view(-1, 46, 0, (3.09, 0.56, -0.42), 7.25)
            # self.add(group_12, group_13)
            # self.add(group_21, group_22)
            # self.add(group_31, group_32, lines)
            # self.add(group_11)
            self.wait(0.1)

        #Optional zoom in on final ish
        self.wait()
        # self.play(self.animate_view(0, 0, 0, (6.12, 0.0, -0.3), 3.43))
        self.wait(20)
        pass


class p36_train_end_zoom_border_and_back(WelchScene):
    def construct(self):

        model = BaarleNet([3])
        pickle_path=welch_path('backprop_3/hackin/3_starting_configs_4/training_data_seed_01_acc_0.8561.pkl')
        # pickle_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/hackin/3_starting_configs_4/training_data_seed_32_acc_0.6159.pkl'
        with open(pickle_path, 'rb') as f:
            p=pickle.load(f)

        print(len(p['weights_history']), ' training steps loaded.')

        viz_scales=[0.2, 0.2, 0.1]
        num_neurons=[3, 3, 2]

        self.set_view(-1, 55, 0, (2.78, 0.37, -0.29), 6.21)

        #Ok having trouble reproducing this training config, so we're going to grab a straring point that 
        # has the same label and is in the same neighborhood!

        # train_step=2400
        start_step=17 #Start here to get the gradietns I want to show
        step_size=3
        # print('starting weights', p['weights_history'][start_step])

        # print('starting grads', p['gradients_history'][start_step])
        # print('empirical grads w1: ', -100*(p['weights_history'][8]['model.0.weight'].numpy()-
        #                                    p['weights_history'][7]['model.0.weight'].numpy())) #-1/lr (0.01)
        # print('empirical grads w2: ', -100*(p['weights_history'][8]['model.2.weight'].numpy()-
        #                                    p['weights_history'][7]['model.2.weight'].numpy())) #-1/lr (0.01)
        # print('empirical grads b1: ', -100*(p['weights_history'][8]['model.0.bias'].numpy()-
        #                                    p['weights_history'][7]['model.0.bias'].numpy())) #-1/lr (0.01)
        # print('empirical grads b2: ', -100*(p['weights_history'][8]['model.2.bias'].numpy()-
        #                                    p['weights_history'][7]['model.2.bias'].numpy())) #-1/lr (0.01)

        # self.wait()
        # In [5]: print('empirical grads: ', -100*(p['weights_history'][8]['model.0.weight'].numpy()-p['weights_history'][7]['model.0.weight'].numpy()))                                      
        # empirical grads:  [[ 0.38974938 -0.45838356]                                                                                                                                        
        #  [ 0.16942024 -0.63158274]                                                                                                                                                          
        #  [-1.0046124   0.9809494 ]]                                                                                                                                                         
        # In [6]: print('starting grads', p['gradients_history'][start_step])                                                                                                                 
        # starting grads {'model.0.weight': tensor([[ 0.0021, -0.0006],                                                                                                                       
        #         [-0.0005,  0.0004],                                                                                                                                                         
        #         [-0.3566,  0.2963]]), 'model.0.bias': tensor([-0.0006, -0.0005, -0.4256]), 'model.2.weight': tensor([[-0.0074, -0.0003, -0.0616],                                           
        #         [ 0.0074,  0.0003,  0.0616]]), 'model.2.bias': tensor([ 0.2037, -0.2037])} 

        #Ok I think there's some issue with my analytical gradients -> updates don't seem to match -> 
        # For numbers on screen I'll show empirical grads then


        train_step = len(p['weights_history'])-1

        w1=p['weights_history'][train_step]['model.0.weight'].numpy()
        b1=p['weights_history'][train_step]['model.0.bias'].numpy()
        w2=p['weights_history'][train_step]['model.2.weight'].numpy()
        b2=p['weights_history'][train_step]['model.2.bias'].numpy()

        with torch.no_grad():
            model.model[0].weight.copy_(torch.from_numpy(w1))
            model.model[0].bias.copy_(torch.from_numpy(b1))
            model.model[2].weight.copy_(torch.from_numpy(w2))
            model.model[2].bias.copy_(torch.from_numpy(b2))


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
            polygons[str(layer_id)+'.split_polygons_nested']=split_polygons_with_relu_simple(polygons[str(layer_id)+'.linear_out']) #Triple nested list so we can simplify merging process layer. 
            polygons[str(layer_id)+'.split_polygons_nested_clipped'] = clip_polygons(polygons[str(layer_id)+'.split_polygons_nested'])
            polygons[str(layer_id)+'.split_polygons_merged'] = merge_zero_regions(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
            polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
            print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')

        #Last linear layer & output
        polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
        intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
        my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)


        #Get first layer Relu Joints - 
        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        group_11=gx.Group(surfaces[1][0])
        if len(joint_points_11)>0:
            joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.9)
            group_11.add(joint_line_11)

        group_12=gx.Group(surfaces[1][1])
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        if len(joint_points_12)>0:
            joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.9)
            group_12.add(joint_line_12)

        group_13=gx.Group(surfaces[1][2])
        joint_points_13 = get_relu_joint(w1[2,0], w1[2,1], b1[2], extent=1)
        if len(joint_points_13)>0:
            joint_line_13=line_from_joint_points_1(joint_points_13).set_opacity(0.9)
            group_13.add(joint_line_13)

        group_11.shift([0, 0, 1.5])
        group_13.shift([0, 0, -1.5])

        polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        polygons_21.shift([0, 0, 0.001]) #Move slightly above map

        polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        polygons_22.shift([0, 0, 0.001]) #Move slightly above map

        group_21=gx.Group(surfaces[2][0], polygons_21)
        group_22=gx.Group(surfaces[2][1], polygons_22)
        group_21.shift([3.0, 0, 0.6])
        group_22.shift([3.0, 0, -0.6])
        group_21.set_opacity(0.5)
        group_22.set_opacity(0.5)


        #Ok now some plane intersction action in a third "panel"
        group_31=group_21.copy()
        group_31[1].set_color(BLUE)
        group_31.shift([3, 0, -0.6])

        group_32=group_22.copy()
        group_32[1].set_color(YELLOW)
        group_32.shift([3, 0, 0.6])

        lines=gx.VGroup()
        for loop in intersection_lines: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = gx.VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines.add(line)
        lines.shift([6, 0, 0])

        #Make Baarle hertog maps a little mroe pronounced. 
        group_31[0].set_opacity(0.9)
        group_32[0].set_opacity(0.9)

        #Book

        # group_21.set_opacity(0.9)
        # group_22.set_opacity(0.9)

        # self.set_view(-1, 46, 0, (3.09, 0.56, -0.42), 7.25)
        self.add(group_12, group_13)
        self.add(group_21, group_22)
        self.add(group_31, group_32, lines)
        self.add(group_11)
        self.wait(0.1)

        #Optional zoom in on final ish
        self.wait()
        self.play(self.animate_view(0, 0, 0, (6.12, 0.0, -0.3), 3.43), run_time=5)
        self.wait()
        self.play(self.animate_view(-1, 55, 0, (2.78, 0.37, -0.29), 6.21), run_time=5)

        self.wait(20)
        pass


class p36_loop_2(WelchScene):
    def construct(self):

        model = BaarleNet([3])
        pickle_path=welch_path('backprop_3/hackin/3_starting_configs_4/training_data_seed_32_acc_0.6159.pkl')
        with open(pickle_path, 'rb') as f:
            p=pickle.load(f)

        print(len(p['weights_history']), ' training steps loaded.')

        viz_scales=[0.2, 0.2, 0.1]
        num_neurons=[3, 3, 2]

        self.set_view(-1, 55, 0, (2.78, 0.37, -0.29), 6.21)

        #Ok having trouble reproducing this training config, so we're going to grab a straring point that 
        # has the same label and is in the same neighborhood!

        # train_step=2400
        start_step=0 #Start here to get the gradietns I want to show
        step_size=3
        # print('starting weights', p['weights_history'][start_step])


        group_12 = group_13 = group_21 = group_22 = group_31 = group_32 = lines = group_11 = None
        for train_step in np.arange(start_step, len(p['weights_history']), step_size):
            if group_12 is not None:
                self.remove(group_12, group_13)
                self.remove(group_21, group_22)
                self.remove(group_31, group_32, lines)
                self.remove(group_11)

            w1=p['weights_history'][train_step]['model.0.weight'].numpy()
            b1=p['weights_history'][train_step]['model.0.bias'].numpy()
            w2=p['weights_history'][train_step]['model.2.weight'].numpy()
            b2=p['weights_history'][train_step]['model.2.bias'].numpy()

            with torch.no_grad():
                model.model[0].weight.copy_(torch.from_numpy(w1))
                model.model[0].bias.copy_(torch.from_numpy(b1))
                model.model[2].weight.copy_(torch.from_numpy(w2))
                model.model[2].bias.copy_(torch.from_numpy(b2))


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
                polygons[str(layer_id)+'.split_polygons_nested']=split_polygons_with_relu_simple(polygons[str(layer_id)+'.linear_out']) #Triple nested list so we can simplify merging process layer. 
                polygons[str(layer_id)+'.split_polygons_nested_clipped'] = clip_polygons(polygons[str(layer_id)+'.split_polygons_nested'])
                polygons[str(layer_id)+'.split_polygons_merged'] = merge_zero_regions(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
                polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
                print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')

            #Last linear layer & output
            polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
            intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
            my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)


            #Get first layer Relu Joints - 
            joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
            group_11=gx.Group(surfaces[1][0])
            if len(joint_points_11)>0:
                joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.9)
                group_11.add(joint_line_11)

            group_12=gx.Group(surfaces[1][1])
            joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
            if len(joint_points_12)>0:
                joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.9)
                group_12.add(joint_line_12)

            group_13=gx.Group(surfaces[1][2])
            joint_points_13 = get_relu_joint(w1[2,0], w1[2,1], b1[2], extent=1)
            if len(joint_points_13)>0:
                joint_line_13=line_from_joint_points_1(joint_points_13).set_opacity(0.9)
                group_13.add(joint_line_13)

            group_11.shift([0, 0, 1.5])
            group_13.shift([0, 0, -1.5])

            polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
            polygons_21.shift([0, 0, 0.001]) #Move slightly above map

            polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
            polygons_22.shift([0, 0, 0.001]) #Move slightly above map

            group_21=gx.Group(surfaces[2][0], polygons_21)
            group_22=gx.Group(surfaces[2][1], polygons_22)
            group_21.shift([3.0, 0, 0.6])
            group_22.shift([3.0, 0, -0.6])
            group_21.set_opacity(0.5)
            group_22.set_opacity(0.5)


            #Ok now some plane intersction action in a third "panel"
            group_31=group_21.copy()
            group_31[1].set_color(BLUE)
            group_31.shift([3, 0, -0.6])

            group_32=group_22.copy()
            group_32[1].set_color(YELLOW)
            group_32.shift([3, 0, 0.6])

            lines=gx.VGroup()
            for loop in intersection_lines: 
                loop=loop*np.array([1, 1, viz_scales[2]])
                line = gx.VMobject()
                line.set_points_as_corners(loop)
                line.set_stroke(color='#FF00FF', width=5)
                lines.add(line)
            lines.shift([6, 0, 0])

            #Make Baarle hertog maps a little mroe pronounced. 
            group_31[0].set_opacity(0.9)
            group_32[0].set_opacity(0.9)

            # group_21.set_opacity(0.9)
            # group_22.set_opacity(0.9)

            # self.set_view(-1, 46, 0, (3.09, 0.56, -0.42), 7.25)
            self.add(group_12, group_13)
            self.add(group_21, group_22)
            self.add(group_31, group_32, lines)
            self.add(group_11)
            self.wait(0.1)


        self.wait()
        self.play(self.animate_view(0, 0, 0, (6.12, 0.0, -0.3), 3.43))
            

        self.wait(20)
        pass
