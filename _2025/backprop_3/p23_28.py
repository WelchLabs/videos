from welch_paths import welch_path
from functools import partial
import sys

sys.path.append('_2025/backprop_3')
import manimgx as gx
import numpy as np
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
from welch_surface import parametric_surface, textured_surface
from polytope_intersection_utils import intersect_polytopes


import manimgx as m
from welch_surface import set_surface_shading
# from MF_Tools import *
import glob
import torch


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
MAGENTA='#FF00FF'

svg_path=welch_path('backprop_3/graphics/to_manim')
graphics_dir=welch_path('backprop_3/graphics/') #Point to folder where map images are

colors = [gx.GREY, BLUE, GREEN, YELLOW, gx.PURPLE, gx.ORANGE, gx.PINK, gx.TEAL]


def surface_func_from_model_with_axes(u, v, model, layer_idx, neuron_idx, axes=None, viz_scale=0.5):
    """
    Create a surface function for visualizing activations at any layer.
    
    Args:
        u, v: Input coordinates (in parameter space, typically -1 to 1)
        model: BaarleNet model
        layer_idx: Direct index into model.model (e.g., 0, 1, 2, 3...)
        neuron_idx: Which neuron in that layer
        axes: Manim axes object to use for coordinate transformation
        viz_scale: Scaling factor for visualization
    """
    input_tensor = torch.tensor([[u, v]], dtype=torch.float32)
    
    with torch.no_grad():
        x = input_tensor
        # Forward through layers up to and including target layer
        for i in range(layer_idx + 1):
            x = model.model[i](x)
        
        activation = x[0, neuron_idx].item()
        z = activation * viz_scale
        
        # If axes provided, transform to axes coordinate system
        if axes is not None:
            return axes.c2p(u, v, z)
        else:
            return np.array([u, v, z])

class p23b(WelchScene):
    def construct(self):
        '''Ok so I think 23 is very much an extention of 21, so once we come back from overhead table
           the network will still be in the center (but now with ReLu drawn on it - and I'll animate foling up the h(1)
           planes, then I thhink the network goes down or to the corner, and I being back the map...)
        '''
        model_path='_2025/backprop_3/models/2_1.pth'
        model = BaarleNet([2])
        model.load_state_dict(torch.load(model_path))

        w1=model.model[0].weight.detach().numpy()
        b1=model.model[0].bias.detach().numpy()
        w2=model.model[2].weight.detach().numpy()
        b2=model.model[2].bias.detach().numpy()


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



        # Ok, so we want to start out with room still in the center for the ball and stick diagram
        # With planes not bent yet, no fold lines yet, probably add thos right before folding
        # And coloring/shading to match neuron colors would be cool! Migth be able to just use polygons I've already compute?
        # Hmm also might want axes? Yeah this whole scene kidna feels like we might want axes...
        # Should be ok/fine. Just need to figure out how to map various ish to axes


        axes_1 = diagram_axes_3d(
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            z_range=[-1, 1, 1],
            width=2, height=2, depth=1.5,
            axis_config={"color": FRESH_TAN, "include_ticks": False, "include_numbers": False, "include_tip": True,
                "stroke_width":4, "tip_config": {"width":0.08, "length":0.08}}
                )
        axes_2 = diagram_axes_3d(
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            z_range=[-1, 1, 1],
            width=2, height=2, depth=1.5,
            axis_config={"color": FRESH_TAN, "include_ticks": False, "include_numbers": False, "include_tip": True,
                "stroke_width":4, "tip_config": {"width":0.08, "length":0.08}}
                )
        axes_1.move_to([0, 0, 1.7])
        axes_2.move_to([0, 0, -1.7])


        # Ok this is kinda annoying -> i think for this to work with axes like this if have to 
        # do my viz scale before shifting? I guess that makes sense. 
        polygons_11_pts=[]
        for p in polygons['0.linear_out'][0][0]:
            p[2]=p[2]*viz_scales[0]
            polygons_11_pts.append(axes_1.c2p(*p))
        polygons_11_pts=np.array(polygons_11_pts)
        polygons_11=manim_polygons_from_np_list([polygons_11_pts], colors=[CYAN], viz_scale=1, opacity=0.3)

        polygons_12_pts=[]
        for p in polygons['0.linear_out'][1][0]:
            p[2]=p[2]*viz_scales[0]
            polygons_12_pts.append(axes_2.c2p(*p))
        polygons_12_pts=np.array(polygons_12_pts)
        polygons_12=manim_polygons_from_np_list([polygons_12_pts], colors=[YELLOW], viz_scale=1, opacity=0.3)


        #Ok now how do I get my surface and ReLu join onto the same axis?
        # array([[-1.        , -1.        ,  1.27255271],                                                                                                                     
        #        [-1.        ,  0.91999996,  0.98150721],                                                                                                                     
        #        [ 0.91999996,  0.91999996,  1.69534064],                                                                                                                     
        #        [ 0.91999996, -1.        ,  1.98638611]])   

        # In [2]: axes_1.c2p(-1, -1, 0)                                                                                                                                       
        # Out[2]: array([-1.        , -1.        ,  1.65999994])   
        # In [5]: axes_1.c2p(1, 1, 0)                                                                                                                                         
        # Out[5]: array([0.91999996, 0.91999996, 1.65999994]) 

        surface_func_11=partial(surface_func_from_model_with_axes, model=model, layer_idx=0, neuron_idx=0, axes=axes_1, viz_scale=viz_scales[0])
        bent_surface_11 = parametric_surface(surface_func_11, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        surface_11=textured_surface(bent_surface_11, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(surface_11, 0,0,0).set_opacity(0.8)

        surface_func_12=partial(surface_func_from_model_with_axes, model=model, layer_idx=0, neuron_idx=1, axes=axes_2, viz_scale=viz_scales[0])
        bent_surface_12 = parametric_surface(surface_func_12, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        surface_12=textured_surface(bent_surface_12, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(surface_12, 0,0,0).set_opacity(0.8)

        #Get first layer Relu Joints
        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        joint_points_11a=axes_1.c2p(joint_points_11[0][0], joint_points_11[0][1], 0)
        joint_points_11b=axes_1.c2p(joint_points_11[1][0], joint_points_11[1][1], 0)
        joint_line_11 = gx.DashedLine( start=joint_points_11a, end=joint_points_11b, color=gx.WHITE, stroke_width=5, dash_length=0.08)

        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        joint_points_12a=axes_2.c2p(joint_points_12[0][0], joint_points_12[0][1], 0)
        joint_points_12b=axes_2.c2p(joint_points_12[1][0], joint_points_12[1][1], 0)
        joint_line_12 = gx.DashedLine( start=joint_points_12a, end=joint_points_12b, color=gx.WHITE, stroke_width=5, dash_length=0.08)


        #I think we need polygons both pre and post bend!
        polygons_21a=gx.VGroup()
        for i in range(2):
            polygons_21_pts=[]
            for p in polygons['0.split_polygons_nested'][0][0][i]:
                p[2]=p[2]*viz_scales[0]
                polygons_21_pts.append(axes_1.c2p(*p))
            polygons_21_pts=np.array(polygons_21_pts)
            polygons_21a.add(manim_polygons_from_np_list([polygons_21_pts], colors=[CYAN], viz_scale=1, opacity=0.3))

        polygons_22a=gx.VGroup()
        for i in range(2):
            polygons_22_pts=[]
            for p in polygons['0.split_polygons_nested'][1][0][i]:
                p[2]=p[2]*viz_scales[0]
                polygons_22_pts.append(axes_2.c2p(*p))
            polygons_22_pts=np.array(polygons_22_pts)
            # print(polygons_22_pts)
            polygons_22a.add(manim_polygons_from_np_list([polygons_22_pts], colors=[YELLOW], viz_scale=1, opacity=0.3))

        polygons_21b=gx.VGroup()
        for i in range(2):
            polygons_21_pts=[]
            for p in polygons['0.split_polygons_nested_clipped'][0][0][i]:
                p[2]=p[2]*viz_scales[0]
                polygons_21_pts.append(axes_1.c2p(*p))
            polygons_21_pts=np.array(polygons_21_pts)
            polygons_21b.add(manim_polygons_from_np_list([polygons_21_pts], colors=[CYAN], viz_scale=1, opacity=0.3))

        polygons_22b=gx.VGroup()
        for i in range(2):
            polygons_22_pts=[]
            for p in polygons['0.split_polygons_nested_clipped'][1][0][i]:
                p[2]=p[2]*viz_scales[0]
                polygons_22_pts.append(axes_2.c2p(*p))
            polygons_22_pts=np.array(polygons_22_pts)
            # print(polygons_22_pts)
            polygons_22b.add(manim_polygons_from_np_list([polygons_22_pts], colors=[YELLOW], viz_scale=1, opacity=0.3))


        surface_func_21=partial(surface_func_from_model_with_axes, model=model, layer_idx=1, neuron_idx=0, axes=axes_1, viz_scale=viz_scales[0])
        bent_surface_21 = parametric_surface(surface_func_21, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        surface_21=textured_surface(bent_surface_21, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(surface_21, 0,0,0).set_opacity(0.8)

        surface_func_22=partial(surface_func_from_model_with_axes, model=model, layer_idx=1, neuron_idx=1, axes=axes_2, viz_scale=viz_scales[0])
        bent_surface_22 = parametric_surface(surface_func_22, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        surface_22=textured_surface(bent_surface_22, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        set_surface_shading(surface_22, 0,0,0).set_opacity(0.8)

        #Need these for a move later
        surface_11_copy=surface_11.copy()
        polygons_21a_copy=polygons_21a.copy()
        polygons_11_copy=polygons_11.copy()
        surface_12_copy=surface_12.copy()
        polygons_22a_copy=polygons_22a.copy()
        polygons_12_copy=polygons_12.copy()
        surface_21_copy=surface_21.copy()
        surface_22_copy=surface_22.copy()


        self.set_view(-53, 68, 0, (0.01, -0.1, 0.09), 5.58)

        self.add(surface_11, polygons_11, axes_1) #, joint_line_11)
        self.add(surface_12, polygons_12, axes_2) #, joint_line_12)

        self.wait()
        self.remove(polygons_11); self.add(polygons_21a); self.add(joint_line_11)
        self.play(gx.ReplacementTransform(polygons_21a[1], polygons_21b[1]),
                  gx.ReplacementTransform(surface_11, surface_21),
                  # FadeIn(joint_line_11),
                  run_time=3.2)
        self.remove(polygons_21b); self.remove(polygons_21a); self.add(polygons_21b)
        self.remove(joint_line_11); self.add(joint_line_11)
        self.remove(axes_1); self.add(axes_1)
        self.wait()

        self.remove(polygons_12); self.add(polygons_22a); self.add(joint_line_12)
        self.play(gx.ReplacementTransform(polygons_22a[1], polygons_22b[1]),
                  gx.ReplacementTransform(surface_12, surface_22),
                  # FadeIn(joint_line_12),
                  run_time=3.2)
        self.remove(polygons_22b); self.remove(polygons_22a); self.add(polygons_22b)
        self.remove(joint_line_12); self.add(joint_line_12)
        self.remove(axes_2); self.add(axes_2)
        self.wait()

        #Ok dope dope dope - now we move thhese suckers to the beginning and start the "regular flow"
        # I think maybe use linear motion b/c I'll want to move the network to the bottom right 
        # probably at the same time. 

        # Hmm reading 24 kind feeling like I want to maybe show some equatoins- probably just from illustrator
        # and maybe on the bottom of the screen? Let me hack on some layouts. 
        group_11=gx.Group(surface_21, polygons_21b, axes_1, joint_line_11)
        group_12=gx.Group(surface_22, polygons_22b, axes_2, joint_line_12)



        # Ok this kinda stucks, but I think that the top and bottom panels need ot be their own axes/scenes?
        # Can't seem to get and angle that works for both plots like this...
        # Ok so I think I need two different moves/scenes for this, and that's ok -> 
        # And also on this move, pretty sure I need to reverse the fold so we can play it again as the data moves through the 
        # network!!
        # group_11.move_to([0, 0, 0.8])
        # group_12.move_to([0, 0, -0.8])
        # self.set_view(0, 66, 0, (-0.06, -0.02, 0.06), 3.82)

        surface_11_copy.shift([0, 0, -0.9])
        polygons_21a_copy.shift([0, 0, -0.9])
        polygons_11_copy.shift([0, 0, -0.9])
        surface_12_copy.shift([0, 0, 0.9])
        polygons_22a_copy.shift([0, 0, 0.9])
        polygons_12_copy.shift([0, 0, 0.9])        
        # self.remove(group_12) #Eh i think no: Animate this move in a seperate scene, bring together in editing. 

        self.wait()
        self.play(self.animate_view(0, 58, 0, (0.19, -0.12, -0.1), 3.82),
                   axes_1.animate.shift([0, 0, -0.9]),
                   joint_line_11.animate.shift([0, 0, -0.9]), #.set_opacity(0.0),
                   gx.ReplacementTransform(polygons_21b, polygons_21a_copy),
                   gx.ReplacementTransform(surface_21, surface_11_copy),
                   axes_2.animate.shift([0, 0, 0.9]),
                   joint_line_12.animate.shift([0, 0, 0.9]), #.set_opacity(0.0),
                   gx.ReplacementTransform(polygons_22b, polygons_22a_copy),
                   gx.ReplacementTransform(surface_22, surface_12_copy),
                  run_time=4)
        self.remove(surface_11_copy); self.add(surface_11_copy)
        self.remove(polygons_21a_copy); self.add(polygons_11_copy)
        self.remove(axes_1); self.add(axes_1)
        self.remove(surface_12_copy); self.add(surface_12_copy)
        self.remove(polygons_22a_copy); self.add(polygons_12_copy)
        self.remove(axes_2); self.add(axes_2)
        self.remove(joint_line_11); self.add(joint_line_11)
        self.remove(joint_line_12); self.add(joint_line_12)
        self.wait()

        # Ok i think that will work pretty well. 
        # Fighting with overall composition a little - probably 3 panels at the end, right?
        # I'm going to work on equations next for a bit. 
        # Ok i think i got a good solution for the first 2 equations!


        # Alright now how to i put map points on the surface? 
        # And maybe add 3d guidelines? Let me look at how I did this last time...
        # Ok p35_41.mp4 has that stuuuf
        # And 46 is also helpful -> it has all the panels for this model!
        # Ok so i think first let me just try to get a magenta 3d sphere onto these maps!

        # self.set_view(0, 58, 0, (0.19, -0.12, -0.1), 3.82),

        d1=gx.Dot(gx.ORIGIN, radius=0.04, fill_color=MAGENTA)
        d1.move_to(axes_1.c2p(0.59, 0.4, -0.04))
        d1.rotate(-20*gx.DEGREES, [0, 1, 0])

        d2=gx.Dot(gx.ORIGIN, radius=0.05, fill_color=MAGENTA)
        d2.move_to(axes_2.c2p(0.59, 0.4, -0.05))
        d2.rotate(20*gx.DEGREES, [1, 0, 0])


        self.wait()
        # self.add(joint_line_11, , joint_line_12) #Just keep from before?
        self.play(self.animate_view(-37, 56, 0, (0.59, -0.49, -0.13), 3.75),
                 gx.FadeIn(d1),
                 gx.FadeIn(d2),
                 run_time=4, 
                 rate_func=gx.linear #Linear rate func here to blend with Premiere move!
                 )
        self.wait()


        #Alright time to bend these puppies back up! I think bend them at the same time. 

        polygons_21a.shift([0, 0, -0.9])
        surface_21_copy.shift([0, 0, -0.9])
        polygons_22a.shift([0, 0, 0.9])
        surface_22_copy.shift([0, 0, 0.9])

        d1b=gx.Dot(gx.ORIGIN, radius=0.04, fill_color=MAGENTA)
        d1b.move_to(axes_1.c2p(0.59, 0.4, 0))
        # d1b.rotate(-20*DEGREES, [0, 1, 0])
        d2b=gx.Dot(gx.ORIGIN, radius=0.05, fill_color=MAGENTA)
        d2b.move_to(axes_2.c2p(0.59, 0.4, 0))
        # d2b.rotate(20*DEGREES, [1, 0, 0])


        self.wait()
        self.remove(polygons_11_copy); self.add(polygons_21a_copy)
        self.play(gx.ReplacementTransform(polygons_21a_copy, polygons_21a),
                  gx.ReplacementTransform(surface_11_copy, surface_21_copy),
                  gx.ReplacementTransform(d1, d1b),
                 run_time=3.5)

        self.remove(polygons_21a); self.add(polygons_21a)
        self.remove(axes_1); self.add(axes_1)
        self.remove(joint_line_11); self.add(joint_line_11)
        self.remove(d1b); self.add(d1b)
        self.wait()

        #Eh maybe seqeuntial works best w/ directing focus and the script?
        self.remove(polygons_12_copy); self.add(polygons_22a_copy)
        self.play(gx.ReplacementTransform(polygons_22a_copy, polygons_22a),
                  gx.ReplacementTransform(surface_12_copy, surface_22_copy),
                  gx.ReplacementTransform(d2, d2b),
                 run_time=3.5)

        self.remove(polygons_22a); self.add(polygons_22a)
        self.remove(axes_2); self.add(axes_2)
        self.remove(joint_line_12); self.add(joint_line_12)
        self.remove(d2b); self.add(d2b)
        self.wait()

        # Ok great! Now I can add in next layer of surfaces etc. 
        # Not sure if I want to show second bent planes separately -> 
        # Maybe? Ok if I can get to the right starting points, I basically already have a 
        # alot ot this work done, if I can get to the right starting point! Might even be
        # to do two separate scenes and a clean transition, and just
        # add magenta dot to p46. Let me see if I can get to the start point here!
        # I guuess I should need to fade out axes, but that would probably be fine, right?
        # Ok so i think our axes went up to 1.7, and then got move down by 0.8, and we need
        # to land on 1.5 Hmm is the scaling going to actually match tho since I've mapped stuff
        #to axes? Hmmm. 
        # Looking at this further, it might not be terrible to actually Replacement
        # Transform to the real things?
        # self.play()

        #Grabbing stuff from p46
        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        group_11=gx.Group(surfaces[1][0]) #, polygons_11)
        if len(joint_points_11)>0:
            joint_line_11b=line_from_joint_points_1(joint_points_11).set_opacity(0.9)
            group_11.add(joint_line_11b)

        group_12=gx.Group(surfaces[1][1]) #, polygons_12)
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        if len(joint_points_12)>0:
            joint_line_12b=line_from_joint_points_1(joint_points_12).set_opacity(0.9)
            group_12.add(joint_line_12b)

        group_11.shift([0, 0, 1.5])


        d1c=gx.Dot([0.59, 0.4, 1.5], radius=0.04, fill_color=MAGENTA)
        d2c=gx.Dot([0.59, 0.4, 0.0], radius=0.04, fill_color=MAGENTA)
        self.wait()
        self.remove(axes_1); self.remove(polygons_21a); #self.remove()
        self.remove(axes_2); self.remove(polygons_22a); 
        self.play(gx.ReplacementTransform(surface_21_copy, group_11[0]),
                  gx.ReplacementTransform(joint_line_11, group_11[1]),
                  gx.ReplacementTransform(d1b, d1c),
                  gx.ReplacementTransform(surface_22_copy, group_12[0]),
                  gx.ReplacementTransform(joint_line_12, group_12[1]),
                  gx.ReplacementTransform(d2b, d2c),
                  self.animate_view(-2, 46, 0, (3.11, 0.22, -0.73), 6.89),
                  run_time=3, 
                  rate_func=gx.linear)
        self.wait()

        # Ok that worked pretty well I think!
        # Now let's try a cut to a new scene for a all the panels
        self.wait(20)
        pass



class p25(WelchScene):
    def construct(self):

        # model = BaarleNet([2])
        # w1 = np.array([[2.5135, -1.02481],
        #  [-1.4043, 2.41291]], dtype=np.float32)
        # b1 = np.array([-1.23981, -0.450078], dtype=np.float32)
        # w2 = np.array([[3.17024, 1.32567],
        #  [-3.40372, -1.53878]], dtype=np.float32)
        # b2 = np.array([-0.884835, 0.0332228], dtype=np.float32)

        # with torch.no_grad():
        #     model.model[0].weight.copy_(torch.from_numpy(w1))
        #     model.model[0].bias.copy_(torch.from_numpy(b1))
        #     model.model[2].weight.copy_(torch.from_numpy(w2))
        #     model.model[2].bias.copy_(torch.from_numpy(b2))

        # viz_scales=[0.2, 0.2, 0.13]
        # num_neurons=[2, 2, 2]

        model_path='_2025/backprop_3/models/2_1.pth'
        model = BaarleNet([2])
        model.load_state_dict(torch.load(model_path))

        w1=model.model[0].weight.detach().numpy()
        b1=model.model[0].bias.detach().numpy()
        w2=model.model[2].weight.detach().numpy()
        b2=model.model[2].bias.detach().numpy()

        viz_scales=[0.2, 0.2, 0.2] #0.13]
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
            line.set_stroke(color='#FF00FF', width=3)
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


        #Start adding Magenta dots here I think!
        d1a=gx.Dot([0.59, 0.4, 1.5], radius=0.05, fill_color=MAGENTA)
        d2a=gx.Dot([0.59, 0.4, 0.0], radius=0.05, fill_color=MAGENTA)

        d1b=d1a.copy()
        d1b.shift([3, 0, 0-0.89*viz_scales[1]+0.06]) #Eh, plus some fuuudge

        d2b=d2a.copy()
        d2b.shift([3, 0, 0+0.03*viz_scales[1]]) #Eh?

        d1c=d1b.copy()
        d1c.shift([3, 0, -0.75])

        d2c=d2b.copy()
        d2c.shift([3, 0, 0.75])

        # d1b.shift([0, 0, 0.01])

        self.set_view(-2, 46, 0, (3.11, 0.22, -0.73), 6.89)
        self.wait()
        # self.play(FadeIn(group_11[0]), FadeIn(group_12[0]), FadeIn(pre_move_lines))
        
        self.add(group_11[0], group_12[0], pre_move_lines, d1a, d2a)
        self.wait()


        # self.add(joint_line)  
        self.wait()
        self.play(gx.ReplacementTransform(surfaces[1][0].copy(),surfaces[2][0]),
                  gx.ReplacementTransform(surfaces[1][1].copy(),surfaces[2][0]),
                  gx.ReplacementTransform(pre_move_lines.copy(), bent_plane_joint_lines),
                  gx.ReplacementTransform(d1a.copy(), d1b),
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
                  gx.ReplacementTransform(d2a.copy(), d2b),
                  run_time=3)
        self.add(polygons_22)
        self.remove(bent_plane_joint_lines_2); self.add(bent_plane_joint_lines_2)
        self.wait()


        #Ok now a little animation bringin the two bent surfaces together and changing their colors? 
        d1c.set_opacity(0.5)
        self.play(gx.ReplacementTransform(group_21.copy(), group_31),
                  gx.ReplacementTransform(group_22.copy(), group_32),
                  gx.ReplacementTransform(d1b.copy(), d1c),
                  gx.ReplacementTransform(d2b.copy(), d2c),
                 run_time=3.0)
        self.play(gx.Create(lines))
        self.wait()
        #Ok not bad - now some nice zooming etc?


        self.play(self.animate_view(-2, 36, 0, (6.96, 0.51, -0.16), 3.55),
                  run_time=4)
        self.wait()

        # Ok ok ok getting close -> now zoom out to overally network, 
        # change point to 0.3, 0.7
        # Zoom in on a couple panels, then one last wide zoom with 
        # all the equations - phew this scene is taking some time. 
        self.play( #self.animate_view(-2, 46, 0, (3.11, 0.22, -0.73), 6.89),
                  self.animate_view(-2, 46, 0, (3.16, 0.8, -0.13), 6.89),
                  run_time=4)        
        self.wait()

        #Zoom in and change poitn
        self.wait()
        self.play(self.animate_view(0, 0, 0, (0.13, -0.08, -0.18), 3.99),
                  gx.FadeOut(d2a), gx.FadeOut(d2b), gx.FadeOut(d2c),
                  gx.FadeOut(d1b), gx.FadeOut(d1c),
                  group_12.animate.set_opacity(0.0),
                  pre_move_lines[1:3].animate.set_opacity(0.0),
                  run_time=4)
        self.wait()

        self.play(gx.FadeOut(d1a))
        

        #[0.3 0.7] tensor([-1.2031,  0.8177]) tensor([0.0000, 0.8177]) tensor([ 0.1991, -1.2250])
        d1a=gx.Dot([0.3, 0.7, 1.5], radius=0.05, fill_color=MAGENTA)
        d2a=gx.Dot([0.3, 0.7, 0.0+viz_scales[1]*0.82], radius=0.05, fill_color=MAGENTA)
        d2a.rotate(25*gx.DEGREES, [1, 0, 0])

        d1b=d1a.copy()
        d1b.shift([3, 0, 0+0.199*viz_scales[1]]) #Eh, plus some fuuudge
        d1b.rotate(25*gx.DEGREES, [1, 0, 0])

        d2b=d2a.copy()
        d2b.shift([3, 0, 0-1.23*viz_scales[1]]) #Eh?
        d2b.shift([0, 0, -0.15])
        d2b.rotate(-60*gx.DEGREES, [1, 0, 0])

        d1c=d1b.copy()
        d1c.shift([3, 0, -0.75])

        d2c=d2b.copy()
        d2c.shift([3, 0, 0.75])
        d2c.set_opacity(0.6)

        self.wait()
        self.play(gx.FadeIn(d1a))
        self.wait()

        self.play(group_12.animate.set_opacity(0.8),
                  pre_move_lines[1:3].animate.set_opacity(1.0), 
                  self.animate_view(-1, 55, 0, (0.12, 0.12, 0.06), 3.05),
                  gx.FadeIn(d2a),
                  run_time=4)
        self.wait()

        self.add(d1b, d2b, d1c, d2c)
        self.wait()
        self.play(self.animate_view(0, 31, 0, (6.13, 0.13, 0.44), 3.52), run_time=4)
        self.wait()

        #Final zoom out and leave room for equations on the bottom!
        self.play(self.animate_view(-1, 37, 0, (3.14, -0.4, -0.06), 6.37), run_time=4)
        self.wait()


        self.wait(20)
        pass
























        # self.set_view(-2, 46, 0, (3.11, 0.22, -0.73), 6.89)




        # self.add(surface_21_copy)
        # self.add(surface_22_copy)

        
        # self.remove(surface_21_copy)

        # surface_21.set_opacity(1.0)

        # self.add(surface_21)
        # self.remove(surface_21)

        # self.add(polygons_21b)
        # self.remove(polygons_21b)

        # self.add(polygons_21a)
        # self.remove(polygons_21a)

        # # polygons_21b.shift([0, 0, 0.9])
        # self.add(polygons_21b)
        # self.remove(polygons_21b)




        # self.add(d1)

        # self.add(joint_line_11)
        # joint_line_11.set_opacity(1.0)

        # self.add(d2)

        # self.add(joint_line_12)
        # joint_line_12.set_opacity(1.0)

        # self.set_view(-37, 56, 0, (0.59, -0.49, -0.13), 3.75)
        # self.wait()





        #Maybe I just do the separate moves in the same scene?
        # self.remove(group_11); self.remove(polygons_11_copy); self.remove(surface_11_copy)
        # self.wait()


        # self.set_view(-53, 68, 0, (0.01, -0.1, 0.09), 5.58)
        # self.add(group_12)

        # # Ok cool now similiar move to above, just with different surfaces, and 
        # # end up at a little different camera angle!
        # surface_12_copy.shift([0, 0, 0.9])
        # polygons_22a_copy.shift([0, 0, 0.9])
        # polygons_12_copy.shift([0, 0, 0.9])

        # self.wait()
        # self.play(self.animate_view(0, 83, 0, (-0.06, -0.06, 0.0), 3.82),
        #            axes_2.animate.shift([0, 0, 0.9]),
        #            joint_line_12.animate.shift([0, 0, 0.9]).set_opacity(0.0),
        #            ReplacementTransform(polygons_22b, polygons_22a_copy),
        #            ReplacementTransform(surface_22, surface_12_copy),
        #           run_time=4)
        # self.remove(surface_12_copy); self.add(surface_12_copy)
        # self.remove(polygons_22a_copy); self.add(polygons_12_copy)
        # self.remove(axes_2); self.add(axes_2)
        # self.wait()

        # Eh reading the script again I'm not so sure 
        # I think i should keep them locked to the same FoV if I can
        # Let me push forward on that path!







        # self.set_view(0, 59, 0, (-0.06, -0.06, 0.0), 3.82)
        # self.add(group_11)

        # self.add(surface_12, polygons_22a, axes_2, joint_line_12)

        
        # self.add(surface_21, polygons_21b, axes_1, joint_line_11)
        # self.add(surface_22, polygons_22b, axes_2, joint_line_12)


        # self.wait()

        # self.add(surface_11, polygons_11)
        # # self.add(polygons_11)
        # self.add(surface_12, polygons_12)
        # self.add(axes_1, axes_2)
        # self.add(joint_line_11)
        # self.add(joint_line_12)

        # self.wait()
        
        #Ok great things are aligned - now I'm not going to bring in the map right away I think! Eh actually it helps alot. 
        #Ok so I'm getting close to starting working out the moves etc -> but I think makes sense to go ahead 
        #And get the bent planes in place, and test the folding transition
        #Then I can align everything to the script 







 













        # surfaces[0][0].move_to([0, 0, 1.7])

        # self.add(surfaces[0][0])






        # group_11.move_to([0, 0, 1.5])
        # group_12.move_to([0, 0, -1.5])



        # self.add(group_11)
        # self.add(group_12)


        # self.wait()







        # group_11.shift([0, 0, 1.5])






        # self.wait()









