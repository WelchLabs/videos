from welch_paths import welch_path
from welch_surface import set_surface_shading
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
import manimgx as gx
import numpy as np
import torch
from geometric_dl_utils import (
    BaarleNet,
    carve_plane_with_relu_joints,
    get_polygon_corners_layer_1,
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

class refactor_sketch_2(WelchScene):
    def construct(self):

        #2
        model_path='_2025/backprop_3/models/2_1.pth'
        model = BaarleNet([2])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.15, 0.3, 0.15]
        num_neurons=[2, 2, 2]

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

        self.set_view(0, 38, 0, (-0.16, 0.17, 0.1), 9.70)

        # Ok now here's where I'm getting thown off ->  oh wait I think I might have forgotten that the 
        # first layer weights are actually just rotating/shifting our INITIAL planes -> that's right. 
        # Alright let me add some polygons, but I think i solved my current writing issue. 

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




        self.wait()
        pass




        