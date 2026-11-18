from welch_paths import welch_path
from welch_surface import set_surface_shading
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
import copy
import manimgx as gx
import numpy as np
import torch
from geometric_dl_utils import (
    BaarleNet,
    carve_plane_with_relu_joints,
    find_polygon_intersections,
    get_polygon_corners_layer_1,
    split_polygons_with_relu,
    surface_func_from_model,
)
from welch_scene import WelchScene
from welch_surface import parametric_surface, textured_surface


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'


graphics_dir=welch_path('backprop_3/graphics/') #Point to folder where map images are

class refactor_sketch_1(WelchScene):
    def construct(self):
        # Ok I generally don't think refactoring on these projects is a good use of time
        # buuut this stuff is gettring pretty unwieldy, and I think Claude can write me a clean API quickly.
        # I'm realizing that there's really just a few operations that I repeate a bunch, so I want to get them uniform
        # I only have lik 9 days to finishe this fucker, but I think this will be worth it!
        # First change i want to make is just loading full torch models here - this will streamlien things
        
        #2x2x2
        model_path='_2025/backprop_3/models/2_2_2_1.pth'
        model = BaarleNet([2,2, 2])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.25, 0.25, 0.3, 0.3, 0.15, 0.15, 0.15]
        num_neurons=[2, 2, 2, 2, 2, 2, 2]

        #3x3
        # model_path='_2025/backprop_3/models/3_3_1.pth'
        # model = BaarleNet([3,3])
        # model.load_state_dict(torch.load(model_path))
        # viz_scales=[0.1, 0.1, 0.05, 0.05, 0.15]
        # num_neurons=[3, 3, 3, 3, 2]


        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=viz_scales[layer_idx])
                bent_surface = parametric_surface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
                ts=textured_surface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
                set_surface_shading(ts, 0,0,0).set_opacity(0.75)
                s.add(ts)
                surface_funcs[-1].append(surface_func)
            surfaces.append(s)

        for layer_idx, sl in enumerate(surfaces):
            for neuron_idx, s in enumerate(sl):
                s.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
                self.add(s)

        # Ok looking good, tomorrow we tackle polygons!
        # Hmm ok yeah so i think the dashed fold line is nice early on in animtation
        # But here I guess at the very first relu (and you could even do before that),
        # I just want ot start figuing out zero crossing polygons.
        self.set_view(0, 38, 0, (-0.16, 0.17, 0.1), 9.70)

        #Optional but kinda nice RuLu intersection planes
        relu_intersections_planes_1=gx.VGroup()
        layer_idx=0
        for neuron_idx in range(num_neurons[layer_idx]):
            plane = gx.Rectangle( width=2,  height=2, fill_color=gx.GREY, fill_opacity=0.3, stroke_color=gx.WHITE, stroke_width=1)
            plane.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
            relu_intersections_planes_1.add(plane)
        self.add(relu_intersections_planes_1)


        #Layer 1 polygons - use surface function to find heights
        layer_idx=1
        layer_1_polygons=get_polygon_corners_layer_1(model)
        layer_1_polygons_3d=[]
        for neuron_idx in range(num_neurons[layer_idx]):
            layer_1_polygons_3d.append([])
            for region in ['positive_region', 'negative_region']:
                a=[]
                for pt_idx in range(len(layer_1_polygons[neuron_idx][region])):
                    a.append(surface_funcs[layer_idx][neuron_idx](*layer_1_polygons[neuron_idx][region][pt_idx]))
                a=np.array(a)
                layer_1_polygons_3d[-1].append(a)

        #Now move to rigth locations and visualize polygons. 
        layer_1_polygons_vgroup=gx.VGroup()
        layer_1_colors=[gx.TEAL, gx.GREY]
        for neuron_idx, polygons in enumerate(layer_1_polygons_3d):
            for j, p in enumerate(polygons):
                poly_3d = gx.Polygon(*p,
                                 fill_color=layer_1_colors[j],
                                 fill_opacity=0.7,
                                 stroke_color=layer_1_colors[j],
                                 stroke_width=2)
                poly_3d.set_opacity(0.3)
                poly_3d.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
                layer_1_polygons_vgroup.add(poly_3d)

        self.add(layer_1_polygons_vgroup)
        self.wait()

        # Ok, that's pretty dope! Now we need to merge these borders
        # Start in 2d again, then back up to 3d
        # I like the idea of a "collapsing down" animation to show the borders 
        # comign together - I can do that next time though!
        layer_idx=2
        layer_2_polygons=carve_plane_with_relu_joints([o['relu_line'] for o in layer_1_polygons])
        layer_2_polygons_3d=[]
        for neuron_idx in range(num_neurons[layer_idx]):
            layer_2_polygons_3d.append([])
            for region in layer_2_polygons:
                a=[]
                for pt_idx in range(len(region)):
                    a.append(surface_funcs[layer_idx][neuron_idx](*region[pt_idx])) #Might be a batch way to do this
                a=np.array(a)
                layer_2_polygons_3d[-1].append(a)

        layer_2_polygons_vgroup=gx.VGroup()
        layer_2_colors = [gx.RED, BLUE, GREEN, YELLOW, gx.PURPLE, gx.ORANGE, gx.PINK, gx.TEAL]
        for neuron_idx, polygons in enumerate(layer_2_polygons_3d):
            for j, p in enumerate(polygons):
                poly_3d = gx.Polygon(*p,
                                 fill_color=layer_2_colors[j%len(layer_2_colors)],
                                 fill_opacity=0.7,
                                 stroke_color=layer_2_colors[j%len(layer_2_colors)],
                                 stroke_width=2)
                poly_3d.set_opacity(0.3)
                poly_3d.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
                layer_2_polygons_vgroup.add(poly_3d)

        self.add(layer_2_polygons_vgroup)
        self.wait()

        #Optional but kinda nice RuLu intersection planes
        relu_intersections_planes_2=gx.VGroup()
        layer_idx=2
        for neuron_idx in range(num_neurons[layer_idx]):
            plane = gx.Rectangle( width=2,  height=2, fill_color=gx.GREY, fill_opacity=0.3, stroke_color=gx.WHITE, stroke_width=1)
            plane.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
            relu_intersections_planes_2.add(plane)
        self.add(relu_intersections_planes_2)

        #Ok, making progress! Now we need to recompute each set of regions based on ReLu clipping!
        # On to the last layer here!
        # So I think it makes sense to merge the zerod out regions before doing th efinal region computation for 
        # the last layer?
        # Visually could make the zero'd out regions gray and then put the overall polygon on top or something.
        layer_idx=3
        # layer_2_polygons_3d_split=split_polygons_with_relu(layer_2_polygons_3d)
        all_polygons, merged_zero_polygons, unmerged_polygons = split_polygons_with_relu(layer_2_polygons_3d)

        ## Display all polygons (dont' zero out merged ones)
        # layer_2_polygons_split_vgroup=VGroup()
        # layer_2_colors = [RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]
        # for neuron_idx, polygons in enumerate(all_polygons):
        #     for j, p in enumerate(polygons):
        #         poly_3d = Polygon(*p,
        #                          fill_color=layer_2_colors[j%len(layer_2_colors)],
        #                          fill_opacity=0.7,
        #                          stroke_color=layer_2_colors[j%len(layer_2_colors)],
        #                          stroke_width=2)
        #         poly_3d.set_opacity(0.3)
        #         poly_3d.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
        #         layer_2_polygons_split_vgroup.add(poly_3d)
        # self.add(layer_2_polygons_split_vgroup)
        # self.wait()

        # layer_2_polygons_split_vgroup=VGroup()
        # layer_2_colors = [RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]
        # for neuron_idx, polygons in enumerate(unmerged_polygons):
        #     for j, p in enumerate(polygons):
        #         poly_3d = Polygon(*p,
        #                          fill_color=layer_2_colors[j%len(layer_2_colors)],
        #                          fill_opacity=0.7,
        #                          stroke_color=layer_2_colors[j%len(layer_2_colors)],
        #                          stroke_width=2)
        #         poly_3d.set_opacity(0.3)
        #         poly_3d.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
        #         layer_2_polygons_split_vgroup.add(poly_3d)
        # self.add(layer_2_polygons_split_vgroup)

        # #A little clunky, but viz zero regions separately
        # layer_2_polygons_zero_vgroup=VGroup()
        # for neuron_idx, polygons in enumerate(merged_zero_polygons):
        #     for j, p in enumerate(polygons):
        #         poly_3d = Polygon(*p,
        #                          fill_color=GREY,
        #                          fill_opacity=0.7,
        #                          stroke_color=GREY,
        #                          stroke_width=2)
        #         poly_3d.set_opacity(0.3)
        #         poly_3d.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
        #         layer_2_polygons_zero_vgroup.add(poly_3d)
        # self.add(layer_2_polygons_zero_vgroup)
        # self.wait()

        #Ok a little clunky to do a post merge like this, but I think this gives some good flexiblity!
        all_polygons_after_merging=copy.deepcopy(merged_zero_polygons)
        for i, o in enumerate(unmerged_polygons):
            all_polygons_after_merging[i].extend(o)

        layer_2_polygons_split_vgroup=gx.VGroup()
        layer_2_colors = [gx.GREY, gx.RED, BLUE, GREEN, YELLOW, gx.PURPLE, gx.ORANGE, gx.PINK, gx.TEAL]
        for neuron_idx, polygons in enumerate(all_polygons_after_merging):
            for j, p in enumerate(polygons):
                poly_3d = gx.Polygon(*p,
                                 fill_color=layer_2_colors[j%len(layer_2_colors)],
                                 fill_opacity=0.7,
                                 stroke_color=layer_2_colors[j%len(layer_2_colors)],
                                 stroke_width=2)
                poly_3d.set_opacity(0.3)
                poly_3d.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
                layer_2_polygons_split_vgroup.add(poly_3d)
        self.add(layer_2_polygons_split_vgroup)

        # Ok this is looking pretty good. 
        # So I need to do another merge step -> 
        # At layer 1 i was able to get away with carve_plane_with_relu_joints
        # Same principle here, but obviously a bit more complex
        # Now my intuition, and the animation that comes to mind, is dropping down all the lines
        # from all 3 layers and then recarcing based on the number of intersections
        # between all these lines - is that right? Line combination happens in 2d basically?
        # Ok i'm like 90% sure that's right, if it's not I think it will shake out in the viz
        # Let's assume it's right and see what happens here exaclty. 

        #Drop last coords. 
        all_polygons_after_merging_2d=[]
        for p in all_polygons_after_merging:
            pd2=[o[:,:2] for o in p]
            all_polygons_after_merging_2d.append(pd2)

        # layer3_regions_2d = compute_layer3_regions(all_polygons_after_merging)
        layer3_regions_2d = find_polygon_intersections(all_polygons_after_merging_2d)

        #Let's do a quick 2d viz to see how things are looking here
        output_poygons_2d=gx.VGroup()
        layer_idx=4
        neuron_idx=-1
        layer_3_colors = [gx.RED, BLUE, GREEN, YELLOW, gx.PURPLE, gx.ORANGE, gx.PINK, gx.TEAL]
        for j, polygon in enumerate(layer3_regions_2d):
                polygon = gx.Polygon(*np.hstack((polygon, np.zeros((polygon.shape[0],1)))),
                                 fill_color=layer_3_colors[j%len(layer_3_colors)],
                                 fill_opacity=0.7,
                                 stroke_color=layer_3_colors[j%len(layer_3_colors)],
                                 stroke_width=2)
                polygon.set_opacity(0.3)
                polygon.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
                output_poygons_2d.add(polygon)
        self.add(output_poygons_2d)

        # Ok making progress here - now last step and then I want to think about exponential growth here
        # Last step will be sending these to 3d! Which should be maybe not terrible?

        layer_3_polygons_3d=[]
        for neuron_idx in range(num_neurons[layer_idx]):
            layer_3_polygons_3d.append([])
            for polygon_2d in layer3_regions_2d:
                a=[]
                for pt_idx in range(len(polygon_2d)):
                    a.append(surface_funcs[layer_idx][neuron_idx](*polygon_2d[pt_idx])) #Might be a batch way to do this
                a=np.array(a)
                layer_3_polygons_3d[-1].append(a)           


        layer_3_polygons_vgroup=gx.VGroup()
        for neuron_idx, polygons in enumerate(layer_3_polygons_3d):
            for j, p in enumerate(polygons):
                poly_3d = gx.Polygon(*p,
                                 fill_color=layer_3_colors[j%len(layer_3_colors)],
                                 fill_opacity=0.7,
                                 stroke_color=layer_3_colors[j%len(layer_3_colors)],
                                 stroke_width=2)
                poly_3d.set_opacity(0.3)
                poly_3d.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
                layer_3_polygons_vgroup.add(poly_3d)
        self.add(layer_3_polygons_vgroup)


      #Ok once more here I think!
        layer_idx=5
        # layer_2_polygons_3d_split=split_polygons_with_relu(layer_2_polygons_3d)
        all_polygons, merged_zero_polygons, unmerged_polygons = split_polygons_with_relu(layer_3_polygons_3d) #Maybe?


        all_polygons_after_merging=copy.deepcopy(merged_zero_polygons)
        for i, o in enumerate(unmerged_polygons):
            all_polygons_after_merging[i].extend(o)

        layer_3_polygons_split_vgroup=gx.VGroup()
        layer_3_colors = [gx.GREY, gx.RED, BLUE, GREEN, YELLOW, gx.PURPLE, gx.ORANGE, gx.PINK, gx.TEAL]
        for neuron_idx, polygons in enumerate(all_polygons_after_merging):
            for j, p in enumerate(polygons):
                poly_3d = gx.Polygon(*p,
                                 fill_color=layer_3_colors[j%len(layer_3_colors)],
                                 fill_opacity=0.7,
                                 stroke_color=layer_3_colors[j%len(layer_3_colors)],
                                 stroke_width=2)
                poly_3d.set_opacity(0.3)
                poly_3d.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
                layer_3_polygons_split_vgroup.add(poly_3d)
        self.add(layer_3_polygons_split_vgroup)

        all_polygons_after_merging_2d=[]
        for p in all_polygons_after_merging:
            pd2=[o[:,:2] for o in p]
            all_polygons_after_merging_2d.append(pd2)

        # layer3_regions_2d = compute_layer3_regions(all_polygons_after_merging)
        layer3_regions_2d = find_polygon_intersections(all_polygons_after_merging_2d)

        #Let's do a quick 2d viz to see how things are looking here
        output_poygons_2d=gx.VGroup()
        neuron_idx=-1
        layer_3_colors = [gx.RED, BLUE, GREEN, YELLOW, gx.PURPLE, gx.ORANGE, gx.PINK, gx.TEAL]
        for j, polygon in enumerate(layer3_regions_2d):
                polygon = gx.Polygon(*np.hstack((polygon, np.zeros((polygon.shape[0],1)))),
                                 fill_color=layer_3_colors[j%len(layer_3_colors)],
                                 fill_opacity=0.7,
                                 stroke_color=layer_3_colors[j%len(layer_3_colors)],
                                 stroke_width=2)
                polygon.set_opacity(0.3)
                polygon.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
                output_poygons_2d.add(polygon)
        self.add(output_poygons_2d)





        self.wait()

        

        self.wait()





        self.wait()
        pass






