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
    apply_viz_scale_to_3d_polygons,
    carve_plane_with_relu_joints,
    compute_adaptive_viz_scales,
    find_polygon_intersections,
    get_3d_polygons,
    get_3d_polygons_layer_1,
    get_polygon_corners_layer_1,
    get_relu_intersection_planes,
    split_polygons_with_relu,
    surface_func_from_model,
    viz_3d_polygons,
    viz_carved_regions_flat,
)
from welch_scene import WelchScene, image_at_height
from welch_surface import parametric_surface, textured_surface
from polytope_intersection_utils import intersect_polytopes

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'


graphics_dir=welch_path('backprop_3/graphics/') #Point to folder where map images are
heatmaps_dir=welch_path('backprop_3/hackin/heatmaps')

class refactor_sketch_1(WelchScene):
    def construct(self):
        
        #2x2
        # model_path='_2025/backprop_3/models/2_2_1.pth'
        # model = BaarleNet([2,2])
        # model.load_state_dict(torch.load(model_path))
        # viz_scales=[0.25, 0.25, 0.3, 0.3, 0.15]
        # num_neurons=[2, 2, 2, 2, 2]

        #3x3
        # model_path='_2025/backprop_3/models/3_3_1.pth'
        # model = BaarleNet([3,3])
        # model.load_state_dict(torch.load(model_path))
        # viz_scales=[0.1, 0.1, 0.05, 0.05, 0.15]
        # num_neurons=[3, 3, 3, 3, 2]

        #8x8
        # model_path='_2025/backprop_3/models/8_8_1.pth'
        # model = BaarleNet([8,8])
        # model.load_state_dict(torch.load(model_path))
        # ## viz_scales=[0.1, 0.1, 0.05, 0.05, 0.15]
        # num_neurons=[8, 8, 8, 8, 2]

        #16x16
        # model_path='_2025/backprop_3/models/16_16_1.pth'
        # model = BaarleNet([16, 16])
        # model.load_state_dict(torch.load(model_path))
        # ## viz_scales=[0.1, 0.1, 0.05, 0.05, 0.15]
        # num_neurons=[16, 16, 16, 16, 2]

        #16x16x16
        model_path='_2025/backprop_3/models/16_16_16_1.pth'
        model = BaarleNet([16, 16, 16])
        model.load_state_dict(torch.load(model_path))
        num_neurons=[16, 16, 16, 16, 16, 16, 2]        

        vertical_spacing=1.5
        horizontal_spacing=3
        colors = [BLUE, gx.RED, GREEN, YELLOW, gx.PURPLE, gx.ORANGE, gx.PINK, gx.TEAL]

        # Hmm ok I think I need to compute a different viz scale for each surface probably 
        # I don't want to totally normalize the heights though, ya know?
        # Maybe there's a discrete set of possible viz scales: 
        adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=1.0, extent=1)
        #For the interesection to make sense, these scales need to match - either need to manual overide or chnage method above
        final_layer_viz=scale=2*min(adaptive_viz_scales[-1]) #little manual ramp here
        adaptive_viz_scales[-1]=[final_layer_viz, final_layer_viz]


        surfaces=[]
        surface_funcs=[]
        surface_funcs_no_viz_scale=[]
        for layer_idx in range(len(model.model)):
            s=gx.Group()
            surface_funcs.append([])
            surface_funcs_no_viz_scale.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx])
                surface_func_no_scaling=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=1.0) #adaptive_viz_scales[layer_idx][neuron_idx])
                bent_surface = parametric_surface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
                ts=textured_surface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
                set_surface_shading(ts, 0,0,0).set_opacity(0.75)
                s.add(ts)
                surface_funcs[-1].append(surface_func)
                surface_funcs_no_viz_scale[-1].append(surface_func_no_scaling)
            surfaces.append(s)

        for layer_idx, sl in enumerate(surfaces):
            for neuron_idx, s in enumerate(sl):
                s.shift([horizontal_spacing*layer_idx-6, 0, vertical_spacing*neuron_idx])
                self.add(s)


        # self.set_view(0, 54, 0, (1.41, 1.82, 4.15), 15.71)
        self.set_view(0, 62, 0, (4.08, 5.37, 9.7), 32.21)


        #Optional but kinda nice RuLu intersection planes
        layer_idx=0
        relu_intersections_planes_1=get_relu_intersection_planes(num_neurons[layer_idx], layer_idx, neuron_idx, horizontal_spacing, vertical_spacing)
        self.add(relu_intersections_planes_1)


        #Layer 1 polygons - use surface function to find heights
        layer_idx=1
        layer_1_polygons=get_polygon_corners_layer_1(model)

        #I think i need to use unscaled surface function here so Relus hit at the right spot
        layer_1_polygons_3d=get_3d_polygons_layer_1(layer_1_polygons, surface_funcs_no_viz_scale, num_neurons=num_neurons[layer_idx], layer_idx=1)
        
        #Ok now I need to rescale. 
        scaled_layer_1_polygons_3d=apply_viz_scale_to_3d_polygons(layer_1_polygons_3d, adaptive_viz_scales[layer_idx])

        polygons_vgroup=viz_3d_polygons(scaled_layer_1_polygons_3d, layer_idx, colors=None, color_gray_index=1)
        self.add(polygons_vgroup)
        self.wait()


        #Shadow of layer 1 polygons, basically the regions available to our second layer
        layer_2_polygons=carve_plane_with_relu_joints([o['relu_line'] for o in layer_1_polygons])


        #2d shadow of these regions
        output_poygons_2d=viz_carved_regions_flat(layer_2_polygons, horizontal_spacing, layer_idx, colors=None)
        self.add(output_poygons_2d)


        #Layer 2 linear
        layer_idx=2
        layer_2_polygons_3d=get_3d_polygons(layer_2_polygons, num_neurons[layer_idx], surface_funcs_no_viz_scale, layer_idx)
        scaled_layer_2_polygons_3d=apply_viz_scale_to_3d_polygons(layer_2_polygons_3d, adaptive_viz_scales[layer_idx])


        polygons_vgroup_2=viz_3d_polygons(scaled_layer_2_polygons_3d, layer_idx, colors=None, color_gray_index=None)
        self.add(polygons_vgroup_2)

   
        relu_intersections_planes_2=get_relu_intersection_planes(num_neurons[layer_idx], layer_idx, neuron_idx, horizontal_spacing, vertical_spacing)
        self.add(relu_intersections_planes_2)


        #Layer 2 Relu
        layer_idx=3
        all_polygons, merged_zero_polygons, unmerged_polygons = split_polygons_with_relu(layer_2_polygons_3d)


        #Ok a little clunky to do a post merge like this, but I think this gives some good flexiblity!
        all_polygons_after_merging=copy.deepcopy(merged_zero_polygons)
        for i, o in enumerate(unmerged_polygons):
            all_polygons_after_merging[i].extend(o)


        all_polygons_after_merging_scaled=apply_viz_scale_to_3d_polygons(all_polygons_after_merging, adaptive_viz_scales[layer_idx])

        layer_2_polygons_split_vgroup=viz_3d_polygons(all_polygons_after_merging_scaled, layer_idx, colors=None)
        self.add(layer_2_polygons_split_vgroup)


        #2D Projection of Layer 2 After Relu Cuts
        #Drop last coords - using unnscaled polgons - dont think it matters?
        all_polygons_after_merging_2d=[]
        for p in all_polygons_after_merging:
            pd2=[o[:,:2] for o in p]
            all_polygons_after_merging_2d.append(pd2)

        layer3_regions_2d = find_polygon_intersections(all_polygons_after_merging_2d)
        output_poygons_2d_2=viz_carved_regions_flat(layer3_regions_2d, horizontal_spacing, layer_idx, colors=None)
        self.add(output_poygons_2d_2)






        #Layer 3 linear
        layer_idx=4
        layer_3_polygons_3d=get_3d_polygons(layer3_regions_2d, num_neurons[layer_idx], surface_funcs_no_viz_scale, layer_idx)
        scaled_layer_3_polygons_3d=apply_viz_scale_to_3d_polygons(layer_3_polygons_3d, adaptive_viz_scales[layer_idx])


        polygons_vgroup_3=viz_3d_polygons(scaled_layer_3_polygons_3d, layer_idx, colors=None, color_gray_index=None)
        self.add(polygons_vgroup_3)

   
        relu_intersections_planes_3=get_relu_intersection_planes(num_neurons[layer_idx], layer_idx, neuron_idx, horizontal_spacing, vertical_spacing)
        self.add(relu_intersections_planes_3)


        #Layer 3 Relu
        layer_idx=5
        all_polygons, merged_zero_polygons, unmerged_polygons = split_polygons_with_relu(layer_3_polygons_3d)


        #Ok a little clunky to do a post merge like this, but I think this gives some good flexiblity!
        all_polygons_after_merging=copy.deepcopy(merged_zero_polygons)
        for i, o in enumerate(unmerged_polygons):
            all_polygons_after_merging[i].extend(o)


        all_polygons_after_merging_scaled=apply_viz_scale_to_3d_polygons(all_polygons_after_merging, adaptive_viz_scales[layer_idx])

        layer_3_polygons_split_vgroup=viz_3d_polygons(all_polygons_after_merging_scaled, layer_idx, colors=None)
        self.add(layer_3_polygons_split_vgroup)


        #2D Projection of Layer 2 After Relu Cuts
        #Drop last coords - using unnscaled polgons - dont think it matters?
        all_polygons_after_merging_2d=[]
        for p in all_polygons_after_merging:
            pd2=[o[:,:2] for o in p]
            all_polygons_after_merging_2d.append(pd2)

        layer3_regions_2d = find_polygon_intersections(all_polygons_after_merging_2d)
        output_poygons_2d_2=viz_carved_regions_flat(layer3_regions_2d, horizontal_spacing, layer_idx, colors=None)
        self.add(output_poygons_2d_2)









        ## --- Start output stuff ---- ##
        # Layer 3 Linear
        # Ok output layer and decision boundary
        # I kinda want a version tha thas all the fun colors and then a yellow and blue version
        layer_idx=4


        layer_3_polygons_3d=get_3d_polygons(layer3_regions_2d, num_neurons[layer_idx], surface_funcs_no_viz_scale, layer_idx)
        scaled_layer_3_polygons_3d=apply_viz_scale_to_3d_polygons(layer_3_polygons_3d, adaptive_viz_scales[layer_idx])

        polygons_vgroup_3=viz_3d_polygons(scaled_layer_3_polygons_3d, layer_idx, colors=None)
        self.add(polygons_vgroup_3)


        #Final map regions
        map_img=image_at_height(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png').scale_to_fit_width(2).scale_to_fit_height(2)
        map_img.shift([horizontal_spacing*(layer_idx+1)-6, 0, -1.5])
        self.add(map_img)

        map_region_1=image_at_height(heatmaps_dir+'/8_8_0.png').scale_to_fit_width(2).scale_to_fit_height(2).set_opacity(0.3)
        map_region_1.shift([horizontal_spacing*(layer_idx+1)-6, 0, -1.5])
        self.add(map_region_1)

        map_region_2=image_at_height(heatmaps_dir+'/8_8_1.png').scale_to_fit_width(2).scale_to_fit_height(2).set_opacity(0.5)
        map_region_2.shift([horizontal_spacing*(layer_idx+1)-6, 0, -1.5])
        self.add(map_region_2)


        # Ok this is looking dope! 
        # Now I think it's more more layer (that isn't really a layer)
        # With output planes brought together, solid colors, and decision boundary!
        scaled_final_polygons=copy.deepcopy(scaled_layer_3_polygons_3d)
        polygons_vgroup_4a=viz_3d_polygons([scaled_final_polygons[0]], layer_idx=5, colors=[BLUE])
        polygons_vgroup_4b=viz_3d_polygons([scaled_final_polygons[1]], layer_idx=5, colors=[YELLOW])

        # self.add(polygons_vgroup_4a)

        # self.add(polygons_vgroup_4b)
        # polygons_vgroup_4b.set_opacity(0.9)
        # polygons_vgroup_4b.set_fill


        self.add(polygons_vgroup_4a, polygons_vgroup_4b)


        #Ok final tiling/intersection time -> I think I've found a better/cleaner way to do this
        # See notes in notion!

        final_polygons=copy.deepcopy(layer_3_polygons_3d) #Need to to intersection on non-scaled polytopes
        intersection_line_coords, new_tiling, top_polygons, indicator = intersect_polytopes(final_polygons[0], final_polygons[1])
        # intersection_line_coords=find_polytope_intersection(final_polygons[0], final_polygons[1])


        intersection_line_coords_scaled=copy.deepcopy(intersection_line_coords)
        for line in intersection_line_coords_scaled:
            for l in line:
                l[2]=l[2]*adaptive_viz_scales[layer_idx][0]


        decision_boundary_lines = gx.Group()
        for line_segment in intersection_line_coords_scaled:
            if len(line_segment) == 2:
                start_point, end_point = line_segment
                line = gx.Line3D(
                    start=start_point,
                    end=end_point,
                    color="#FF00FF",
                    width=0.02,
                )
                # Shift to match your layer positioning (layer_idx=5 for final output)
                line.shift([horizontal_spacing*(layer_idx+1)-6, 0, 0])  # horizontal_spacing * layer_idx - 6
                decision_boundary_lines.add(line)

        # Add the decision boundary to the scene
        self.add(decision_boundary_lines)


        decision_boundary_lines_flat = gx.Group()
        for line_segment in intersection_line_coords_scaled:
            if len(line_segment) == 2:
                start_point, end_point = line_segment
                start_point[2]=-1.5 #Flatten that shit!
                end_point[2]=-1.5
                line = gx.Line3D(
                    start=start_point,
                    end=end_point,
                    color="#FF00FF",
                    width=0.02, #0.02 is probably better
                )
                # Shift to match your layer positioning (layer_idx=5 for final output)
                line.shift([horizontal_spacing*(layer_idx+1)-6, 0, 0])  # horizontal_spacing * layer_idx - 6
                decision_boundary_lines_flat.add(line)

        # Add the decision boundary to the scene
        self.add(decision_boundary_lines_flat)


        # Ok I'm interesting in 3 new visualizations here, then I can decide which ones i want to show 
        # and probably do a little more refactoring
        # Ok, so I definitely want to see the new tiling, with the borders on top of it, that will be dope!
        # And I want to make the colored "top polytope" with decision borders - I think this will be alot more clear!
        # Let me just hack stuff together here, than I can worry about making it clean/wrapped up
        # Oh also, I think the toppolytome with individual plane coloring coudl be cool? Maybe? We'll see here. 


        # polygons_vgroup=VGroup()
        # for j, p in enumerate(top_polygons):
        #     if len(p)<3: continue
        #     if indicator[j]: color=YELLOW
        #     else: color=BLUE
            
        #     p_scaled=copy.deepcopy(p) #Scaling for viz
        #     p_scaled[:,2]=p_scaled[:,2]*adaptive_viz_scales[layer_idx][0]
        #     poly_3d = Polygon(*p_scaled,
        #                      fill_color=color,
        #                      fill_opacity=0.4,
        #                      stroke_color=color,
        #                      stroke_width=2)
        #     poly_3d.set_opacity(0.8)
        #     poly_3d.shift([horizontal_spacing*(layer_idx+1)-6, 0, 4])
        #     polygons_vgroup.add(poly_3d)

        # self.add(polygons_vgroup)


        output_poygons_2d_final=viz_carved_regions_flat(new_tiling, horizontal_spacing, layer_idx+2, colors=None)
        self.add(output_poygons_2d_final)

        #Lol gotta wrap up this sucka
        decision_boundary_lines_flat = gx.Group()
        for line_segment in intersection_line_coords_scaled:
            if len(line_segment) == 2:
                start_point, end_point = line_segment
                start_point[2]=-1.5 #Flatten that shit!
                end_point[2]=-1.5
                line = gx.Line3D(
                    start=start_point,
                    end=end_point,
                    color="#FF00FF",
                    width=0.02, #0.02 is probably better
                )
                # Shift to match your layer positioning (layer_idx=5 for final output)
                line.shift([horizontal_spacing*(layer_idx+2)-6, 0, 0])  # horizontal_spacing * layer_idx - 6
                decision_boundary_lines_flat.add(line)

        # Add the decision boundary to the scene
        self.add(decision_boundary_lines_flat)
        decision_boundary_lines_flat.set_opacity(0.3)


        # Ok let my try my own idea on the top polytope thing here -
        # I feel like I can just run it through the surface function myself???
        hyperspace_polygons=[]
        for poly_2d in new_tiling:
            hyperspace_polygon=[]
            for pt in poly_2d:
                pt_copy=copy.deepcopy(list(pt))
                for j, surf_func in enumerate(surface_funcs_no_viz_scale[4]):
                    pt_copy.append(surf_func(*pt)[-1])
                hyperspace_polygon.append(pt_copy)
            hyperspace_polygons.append(np.array(hyperspace_polygon))


        my_top_polygons=[]
        my_indicator=[]
        for p in hyperspace_polygons:
            if np.all(p[:,2]>p[:,3]):
                my_top_polygons.append(p[:, (0,1,2)])
                my_indicator.append(0)
            elif np.all(p[:,3]>p[:,2]):
                my_top_polygons.append(p[:, (0,1,3)])
                my_indicator.append(1)
            elif np.max(p[:,2])>np.max(p[:,3]): #Is this crazy? Seems like it works? I think i only need these last 2 really. 
                my_top_polygons.append(p[:, (0,1,2)])
                my_indicator.append(0)
            elif np.max(p[:,3])>np.max(p[:,2]): #Is this crazy?
                my_top_polygons.append(p[:, (0,1,3)])
                my_indicator.append(1)


        polygons_vgroup=gx.VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=p_scaled[:,2]*adaptive_viz_scales[layer_idx][0]
            poly_3d = gx.Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=2)
            poly_3d.set_opacity(0.3)
            poly_3d.shift([horizontal_spacing*(layer_idx+1)-6, 0, 2])
            polygons_vgroup.add(poly_3d)

        self.add(polygons_vgroup)






        # Jul 27 am
        # Ok ok ok ok ok getting close here on I think some nice viz options
        # I haven't tested intersect_polytopes yet -> seems like it's returning too few results
        # I'll get into that when I get back. 
        # But I think this is going to be a nice approach!


        #Coping over some notes from notions
        # - Ok, so I definitely want to explicitly compute the new 2d tiling
        # - From there, the “upper polytope” i’m looking for should be pretty straigthfoward to get I think!
        # - man this perspective is actually intersesting in a bunch of ways
        # - Many and maybe even a cool intro/hoook
        # - Because what I didn’t realize here, which is actually really cool and makes sense is that:
        #     - THE MODEL LEARNS A TILING THAT MATCHES THE CITY!!!
        #     - Ah that’s so cool. Totally makes sense. c
        # - **Ok yeah yeah yeah and the smooth animation from the 2d multicolored tiling to the 3d “top” polytope, still in mulitcolor or maybe fading to blue/yellow would be SICK!**
        # - Ok yeah yeah yeah → the algorithm is strating to click in my head a bit here.
        # - Oh wow → the connection to the “original” way the 2 neuron model tried to solve the problem is pretty interesting!
        #     - It’s like:
        #         - **Divide the map into these regions.**
        #         - **For each one you have 2 copies of the the polygone to fit togeether at some angle**
        #         - **Their intersection becomes the decision boundary! Just like the 2 neuron case, just all the layers before have chopped up the space for you.**
        #         - **Hmm that’s pretty interesting!**


        # Ok dope - maybe want to mess with final layer scaling, we'll see
        # Now we just need a decision boundary. 

        # def find_polytope_intersection(polygons_1, polygons_2):
        #     '''
        #     Given two lists of Nx3 numpy arrays, (polygons_1, polygons_2), where each 
        #     numpy array gives the vertices of face of a polytope, find all intersection lines between the 
        #     two polytope surfaces, and return as a list of starting an dending points for each line. 
        #     '''

        #     return intersection_line_coords


        #Hmm like 3k results? Seems like a lot lol. Maybe analtyical in not 



        # Ok ChatGPT's solution is getting there, it still misses a segmmetn it looks like, and 
        # I need to scale it back down for viz!
        # Ok, so I'll pick up here when I'm back and see what I can figure out!
        # Might be worth looking at what I did in my first round of sketches -> that seemed to work ok??
        # Hmm after I viz scale, will they still intersect at the right point?
        # And final final question -> intersection line Z does not seem arbitrary -> is it fixed Z somehow? 
        # If so, why, and does it help me????

        #Will want to wrap this stuff up after things are kinda working
        # decision_boundary_lines_flat = Group()
        # for line_segment in intersection_line_coords_scaled:
        #     if len(line_segment) == 2:
        #         start_point, end_point = line_segment
        #         start_point[2]=-1.5 #Flatten that shit!
        #         end_point[2]=-1.5
        #         line = Line3D(
        #             start=start_point,
        #             end=end_point,
        #             color="#FF00FF",
        #             width=0.02, #0.02 is probably better
        #         )
        #         # Shift to match your layer positioning (layer_idx=5 for final output)
        #         line.shift([horizontal_spacing*(layer_idx+1)-6, 0, 0])  # horizontal_spacing * layer_idx - 6
        #         decision_boundary_lines_flat.add(line)

        # # Add the decision boundary to the scene
        # self.add(decision_boundary_lines_flat)

        # Ok so we're getting close 
        # Probably right now is that it's really hard to see how the the yellow and blue polytopes intersect
        # I think what might help here is having a mostly or full opaque "top surface"
        # Then I coudl imagine the yellow plane "comming up from below" (which I think matching flipping everythong over, which 
        # i think want to do to match the matrices better)
        # Then this "top surface gradually forms with magenta boundaries, and solid or very opaque yellow and blue faces"
        # To do this though, I need to compute a "top polytope", and I need to know the color for each face. 
        # Let me start designing my interface here!
        # Once i get that working, i can slowly adjust up the yellow polytope to get a nice "growing up " animation.
        # Ahh it's the SAME TILING for top and bottom, that's important! It will be same tiling after too. 
        # The 2d version of that is actually kinda interesting too right?
        # Let me add that to the final map for a second and see what that vibe is. 

        # output_poygons_2d_2_copy=copy.deepcopy(output_poygons_2d_2)
        # output_poygons_2d_2_copy=output_poygons_2d_2_copy.shift([horizontal_spacing*2,0, 0])
        # self.add(output_poygons_2d_2_copy)
        # self.remove(map_region_1, map_region_2, map_img)


        # def get_upper_polytope(polygons_1, polygons_2):
        #     '''
        #     Given two lists of Nx3 numpy arrays, (polygons_1, polygons_2), where each 
        #     numpy array gives the vertices of face of a polytope that tiles the -1,1 plane, 
        #     find all intersection lines between the two polytope surfaces, and split copies of thethe existing polytopes along 
        #     the intersection lines. 
        #     '''

        #     return intersection_line_coords







        self.wait()
        pass






