from welch_paths import welch_path
import manimgx as gx
import numpy as np
from welch_geometry import diagram_axes
from welch_scene import WelchScene
from welch_surface import parametric_surface, set_surface_shading, textured_surface

from functools import partial

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b' #6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
MAGENTA='#FF00FF'

data_dir=welch_path('grokking/from_linux/grokking/grok_1764101670/')
resolution=113

alphas_1=np.linspace(0, 1, resolution) #Crank up here for better spatial resolution I think
def param_surface(u, v, surf_array, scale=0.15):
    u_idx = np.abs(alphas_1 - u).argmin()
    v_idx = np.abs(alphas_1 - v).argmin()
    try:
        z = scale*surf_array[v_idx, u_idx] #Add vertical scaling here?
    except IndexError:
        z = 0
    return np.array([u, v, z])


class GrokkingHackingOne(WelchScene):
    def construct(self):  

        mlp_hook_pre=np.load(data_dir+'mlp_hook_pre.npy') #Lots of data but seems fast enough?



        neuron_idx=453 #343

        surf_func=partial(param_surface, surf_array=mlp_hook_pre[:,:,2,neuron_idx], scale=0.15)
        surface = parametric_surface(
            surf_func,  
            u_range=[0, 1.0],
            v_range=[0, 1.0],
            resolution=(resolution, resolution),
        )

        ts = textured_surface(surface, data_dir+'activations_'+str(neuron_idx).zfill(3)+'.png')
        set_surface_shading(ts, 0.0, 0.1, 0)

        # self.add(ts)
        # self.add(ts)

        mlp_hook_post=np.load(data_dir+'hook_mlp_out.npy')

        neuron_idx=0 #343

        surf_func=partial(param_surface, surf_array=mlp_hook_post[:,:,2,neuron_idx], scale=0.01)
        surface = parametric_surface(
            surf_func,  
            u_range=[0, 1.0],
            v_range=[0, 1.0],
            resolution=(resolution, resolution),
        )

        ts = textured_surface(surface, data_dir+'activations_post_'+str(neuron_idx).zfill(3)+'.png')
        set_surface_shading(ts, 0.0, 0.1, 0)

        # self.add(ts)
        # self.wait()


        logits=np.load(data_dir+'logits.npy')
        logit_index=2


        surf_func=partial(param_surface, surf_array=logits[:,:,logit_index], scale=0.001)
        surface = parametric_surface(
            surf_func,  
            u_range=[0, 1.0],
            v_range=[0, 1.0],
            resolution=(resolution, resolution),
        )

        ts = textured_surface(surface, data_dir+'logit_'+str(logit_index)+'.png')
        set_surface_shading(ts, 0.0, 0.1, 0)

        # self.add(ts)
        # self.wait()


        probe_1=np.load(data_dir+'probe_1.npy')
        probe_2=np.load(data_dir+'probe_2.npy')


        axis_1 = diagram_axes(
            x_range=[0, 1.0, 1],
            y_range=[-1.0, 1.0, 1],
            width=3,
            height=1,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":3,
                "tip_config": {"width":0.02, "length":0.02}
                }
            )

        probe_1a_pts = [axis_1.c2p(i/len(probe_1), probe_1[i,0]) for i in range(len(probe_1))]
        probe_1a_line = gx.VMobject(stroke_width=3)
        probe_1a_line.set_points_smoothly(probe_1a_pts)
        probe_1a_line.set_color(CYAN)

        probe_1b_pts = [axis_1.c2p(i/len(probe_1), probe_1[i,1]) for i in range(len(probe_1))]
        probe_1b_line = gx.VMobject(stroke_width=3)
        probe_1b_line.set_points_smoothly(probe_1b_pts)
        probe_1b_line.set_color(gx.RED)

        self.add(axis_1)
        self.add(probe_1a_line, probe_1b_line)

        probe_2a_pts = [axis_1.c2p(i/len(probe_2), probe_2[i,0]) for i in range(len(probe_2))]
        probe_2a_line = gx.VMobject(stroke_width=3)
        probe_2a_line.set_points_smoothly(probe_2a_pts)
        probe_2a_line.set_color(YELLOW)

        probe_2b_pts = [axis_1.c2p(i/len(probe_2), probe_2[i,1]) for i in range(len(probe_2))]
        probe_2b_line = gx.VMobject(stroke_width=3)
        probe_2b_line.set_points_smoothly(probe_2b_pts)
        probe_2b_line.set_color(MAGENTA)


        self.add(axis_1)
        self.add(probe_2a_line, probe_2b_line)


        self.wait()
        # Scene authoring stops here in the interactive editor.

        # num_lines = 64  # Number of gridlines in each direction
        # num_points = 512  # Number of points per line
        # u_gridlines = VGroup()
        # v_gridlines = VGroup()
        # u_values = np.linspace(-2.5, 2.5, num_lines)
        # v_points = np.linspace(-2.5, 2.5, num_points)
        # for u in u_values:
        #     points = [surf_func(u, v) for v in v_points]
        #     line = VMobject()
        #     line.set_points_smoothly(points)
        #     line.set_stroke(width=1, color=WHITE, opacity=0.15)
        #     u_gridlines.add(line)

        # u_points = np.linspace(-2.5, 2.5, num_points)
        # for v in u_values:  # Using same number of lines for both directions
        #     points = [surf_func(u, v) for u in u_points]
        #     line = VMobject()
        #     line.set_points_smoothly(points)
        #     line.set_stroke(width=1, color=WHITE, opacity=0.15)
        #     v_gridlines.add(line)
        # grids.add(VGroup(u_gridlines, v_gridlines))
        

        self.wait(20)
        # Scene authoring stops here in the interactive editor.
