from welch_paths import welch_path
import manimgx as gx
from activation_diagrams import black_to_tan_hex, draw_attention_patterns, draw_attention_values, draw_embeddings, draw_embeddings_2, draw_inputs, draw_logits, draw_mlp_1, draw_mlp_2, draw_mlp_3, softmax_with_temperature, viridis_hex
import numpy as np
from welch_geometry import diagram_axes, diagram_axes_3d
from welch_scene import WelchScene
from welch_surface import parametric_surface, set_surface_shading, textured_surface
from welch_text import math_tex

from functools import partial

from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as colors
from tqdm import tqdm

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

resolution=113
svg_dir=Path(welch_path('grokking/graphics/to_manim'))
data_dir=Path(welch_path('grokking/from_linux/grok_1764706121'))

alphas_1=np.linspace(0, 1, resolution) #Crank up here for better spatial resolution I think
def param_surface(u, v, surf_array, scale=0.15):
    u_idx = np.abs(alphas_1 - u).argmin()
    v_idx = np.abs(alphas_1 - v).argmin()
    try:
        z = scale*surf_array[v_idx, u_idx] #Add vertical scaling here?
    except IndexError:
        z = 0
    return np.array([u, v, z])


def surf_func(u, v, axes, surf_array, scale=1.0):
    # Map u,v from [0,1] to your data indices
    i = int(u * (surf_array.shape[0] - 1))
    j = int(v * (surf_array.shape[1] - 1))
    
    # Get data coordinates
    x = u * 113  # or whatever your x range is
    y = v * 113
    z = surf_array[i, j] * scale
    
    # Transform to axes coordinate system
    return axes.c2p(x, y, z)


class Dot3D(gx.Sphere):
    def __init__(self, center=gx.ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(center)


def make_fourier_surf_func(axes, comp_func):
    def func(u, v):
        i = u * 113  # Map u from [0,1] to [0,113]
        j = v * 113  # Map v from [0,1] to [0,113]
        x = i
        y = j
        z = comp_func(i, j)
        return axes.c2p(x, y, z)
    return func


class P59_mlp_surface_1b(WelchScene):
    def construct(self): 
        p=113
        neuron_idx=341

        ckpt_dir=Path(welch_path('grokking/from_linux/grok_1764706121'))
        mlp_pre=np.load(data_dir/('training_mlp_pre'+str(neuron_idx)+'.npy'))

        axes_1 = diagram_axes_3d(
            x_range=[0, p, 10],
            y_range=[0, p, 10],
            z_range=[-1, 1, 1],
            width=4,
            height=4,
            depth=1.4,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.05, "length":0.05}
                }
        )

        x_label = math_tex("x", font_size=32).next_to(axes_1.x_axis.get_end(), gx.RIGHT, buff=0.1).set_color(CHILL_BROWN)
        y_label = math_tex("y", font_size=32).next_to(axes_1.y_axis.get_end(), gx.UP, buff=0.1).set_color(CHILL_BROWN)
        axes_1_group=gx.VGroup(axes_1[:2], x_label, y_label)
        x_label.rotate(gx.DEGREES*90, [1, 0, 0])
        y_label.rotate(gx.DEGREES*90, [1, 0, 0])
        y_label.rotate(gx.DEGREES*90, [0, 0, 1])
        axes_1[0].rotate(gx.DEGREES*90, [1, 0, 0])
        axes_1[1].rotate(gx.DEGREES*90, [0, 1, 0])


        # self.set_view(43, 57, 0, (-0.15, -0.07, -0.33), 6.60)
        # self.set_view(43, 42, 0, (0.07, -0.2, -0.58), 6.21)
        # self.set_view(45, 45, 0, (-0.07, -0.17, -0.49), 6.21)
        # self.set_view(45, 44, 0, (-0.07, -0.17, -0.49), 6.21)
        self.set_view(45, 57, 0, (-0.07, -0.17, -0.49), 6.21)
        self.add(axes_1[:2]) #Leaving out x and y for now. 

        for time_step in range(1000):
            neuron_1_mean=np.mean(mlp_pre[time_step, :, :])
            neuron_1_max=np.max(np.abs(mlp_pre[time_step, :, :]-neuron_1_mean))

            surf_func_with_axes = partial(
                surf_func, 
                axes=axes_1,
                surf_array=(mlp_pre[time_step, :, :]-neuron_1_mean)/neuron_1_max, 
                scale=1.0
            )

            surface = parametric_surface(
                surf_func_with_axes,  
                u_range=[0, 1.0],
                v_range=[0, 1.0],
                resolution=(resolution, resolution),
            )

            ts = textured_surface(surface, str(data_dir/'training_heatmaps'/('mlp_pre_'+str(10*time_step).zfill(4)+'_'+str(neuron_idx).zfill(3)+'.png')))
            set_surface_shading(ts, 0.0, 0.1, 0)
            ts.set_opacity(0.8)

            self.add(ts)
            self.wait(0.1)
            self.remove(ts)


        self.wait(20)
        # Scene authoring stops here in the interactive editor.

class P59_mlp_surface_2b(WelchScene):
    def construct(self): 
        p=113
        neuron_idx=106

        ckpt_dir=Path(welch_path('grokking/from_linux/grok_1764706121'))
        mlp_pre=np.load(data_dir/('training_mlp_pre'+str(neuron_idx)+'.npy'))

        axes_1 = diagram_axes_3d(
            x_range=[0, p, 10],
            y_range=[0, p, 10],
            z_range=[-1, 1, 1],
            width=4,
            height=4,
            depth=1.4,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.05, "length":0.05}
                }
        )

        x_label = math_tex("x", font_size=32).next_to(axes_1.x_axis.get_end(), gx.RIGHT, buff=0.1).set_color(CHILL_BROWN)
        y_label = math_tex("y", font_size=32).next_to(axes_1.y_axis.get_end(), gx.UP, buff=0.1).set_color(CHILL_BROWN)
        axes_1_group=gx.VGroup(axes_1[:2], x_label, y_label)
        x_label.rotate(gx.DEGREES*90, [1, 0, 0])
        y_label.rotate(gx.DEGREES*90, [1, 0, 0])
        y_label.rotate(gx.DEGREES*90, [0, 0, 1])
        axes_1[0].rotate(gx.DEGREES*90, [1, 0, 0])
        axes_1[1].rotate(gx.DEGREES*90, [0, 1, 0])


        # self.set_view(43, 57, 0, (-0.15, -0.07, -0.33), 6.60)
        # self.set_view(43, 42, 0, (0.07, -0.2, -0.58), 6.21)
        # self.set_view(45, 45, 0, (-0.07, -0.17, -0.49), 6.21)
        # self.set_view(45, 44, 0, (-0.07, -0.17, -0.49), 6.21)
        # self.set_view(49, 47, 0, (-0.07, -0.17, -0.49), 6.21)
        self.set_view(45, 57, 0, (-0.07, -0.17, -0.49), 6.21)
        self.add(axes_1[:2]) #Leaving out x and y for now. 

        for time_step in range(1000):
            neuron_1_mean=np.mean(mlp_pre[time_step, :, :])
            neuron_1_max=np.max(np.abs(mlp_pre[time_step, :, :]-neuron_1_mean))

            surf_func_with_axes = partial(
                surf_func, 
                axes=axes_1,
                surf_array=(mlp_pre[time_step, :, :]-neuron_1_mean)/neuron_1_max, 
                scale=1.0
            )

            surface = parametric_surface(
                surf_func_with_axes,  
                u_range=[0, 1.0],
                v_range=[0, 1.0],
                resolution=(resolution, resolution),
            )

            ts = textured_surface(surface, str(data_dir/'training_heatmaps'/('mlp_pre_'+str(10*time_step).zfill(4)+'_'+str(neuron_idx).zfill(3)+'.png')))
            set_surface_shading(ts, 0.0, 0.1, 0)
            ts.set_opacity(0.8)

            self.add(ts)
            self.wait(0.1)
            self.remove(ts)


        self.wait(20)
        # Scene authoring stops here in the interactive editor.

class P59_mlp_surface_3b(WelchScene):
    def construct(self): 
        p=113
        neuron_idx=1

        ckpt_dir=Path(welch_path('grokking/from_linux/grok_1764706121'))
        mlp_pre=np.load(data_dir/('training_mlp_out'+str(neuron_idx).zfill(3)+'.npy'))

        axes_1 = diagram_axes_3d(
            x_range=[0, p, 10],
            y_range=[0, p, 10],
            z_range=[-1, 1, 1],
            width=4,
            height=4,
            depth=1.4,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.05, "length":0.05}
                }
        )

        x_label = math_tex("x", font_size=32).next_to(axes_1.x_axis.get_end(), gx.RIGHT, buff=0.1).set_color(CHILL_BROWN)
        y_label = math_tex("y", font_size=32).next_to(axes_1.y_axis.get_end(), gx.UP, buff=0.1).set_color(CHILL_BROWN)
        axes_1_group=gx.VGroup(axes_1[:2], x_label, y_label)
        x_label.rotate(gx.DEGREES*90, [1, 0, 0])
        y_label.rotate(gx.DEGREES*90, [1, 0, 0])
        y_label.rotate(gx.DEGREES*90, [0, 0, 1])
        axes_1[0].rotate(gx.DEGREES*90, [1, 0, 0])
        axes_1[1].rotate(gx.DEGREES*90, [0, 1, 0])


        # self.set_view(43, 57, 0, (-0.15, -0.07, -0.33), 6.60)
        # self.set_view(43, 42, 0, (0.07, -0.2, -0.58), 6.21)
        # self.set_view(45, 45, 0, (-0.07, -0.17, -0.49), 6.21)
        # self.set_view(45, 44, 0, (-0.07, -0.17, -0.49), 6.21)
        self.set_view(45, 57, 0, (-0.07, -0.17, -0.49), 6.21)
        self.add(axes_1[:2]) #Leaving out x and y for now. 

        for time_step in range(1000):
            neuron_1_mean=np.mean(mlp_pre[time_step, :, :])
            neuron_1_max=np.max(np.abs(mlp_pre[time_step, :, :]-neuron_1_mean))

            surf_func_with_axes = partial(
                surf_func, 
                axes=axes_1,
                surf_array=(mlp_pre[time_step, :, :]-neuron_1_mean)/neuron_1_max, 
                scale=1.0
            )

            surface = parametric_surface(
                surf_func_with_axes,  
                u_range=[0, 1.0],
                v_range=[0, 1.0],
                resolution=(resolution, resolution),
            )

            ts = textured_surface(surface, str(data_dir/'training_heatmaps'/('mlp_out_'+str(10*time_step).zfill(4)+'_'+str(neuron_idx).zfill(3)+'.png')))
            set_surface_shading(ts, 0.0, 0.1, 0)
            ts.set_opacity(0.8)

            self.add(ts)
            self.wait(0.1)
            self.remove(ts)


        self.wait(20)
        # Scene authoring stops here in the interactive editor.

class P59_mlp_surface_4b(WelchScene):
    def construct(self): 
        p=113
        neuron_idx=7

        ckpt_dir=Path(welch_path('grokking/from_linux/grok_1764706121'))
        mlp_pre=np.load(data_dir/('logits'+str(neuron_idx).zfill(3)+'.npy'))

        axes_1 = diagram_axes_3d(
            x_range=[0, p, 10],
            y_range=[0, p, 10],
            z_range=[-1, 1, 1],
            width=4,
            height=4,
            depth=1.4,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.05, "length":0.05}
                }
        )

        x_label = math_tex("x", font_size=32).next_to(axes_1.x_axis.get_end(), gx.RIGHT, buff=0.1).set_color(CHILL_BROWN)
        y_label = math_tex("y", font_size=32).next_to(axes_1.y_axis.get_end(), gx.UP, buff=0.1).set_color(CHILL_BROWN)
        axes_1_group=gx.VGroup(axes_1[:2], x_label, y_label)
        x_label.rotate(gx.DEGREES*90, [1, 0, 0])
        y_label.rotate(gx.DEGREES*90, [1, 0, 0])
        y_label.rotate(gx.DEGREES*90, [0, 0, 1])
        axes_1[0].rotate(gx.DEGREES*90, [1, 0, 0])
        axes_1[1].rotate(gx.DEGREES*90, [0, 1, 0])


        # self.set_view(43, 57, 0, (-0.15, -0.07, -0.33), 6.60)
        # self.set_view(43, 42, 0, (0.07, -0.2, -0.58), 6.21)
        # self.set_view(45, 45, 0, (-0.07, -0.17, -0.49), 6.21)
        # self.set_view(45, 44, 0, (-0.07, -0.17, -0.49), 6.21)
        # self.set_view(49, 47, 0, (-0.07, -0.17, -0.49), 6.21)
        self.set_view(45, 57, 0, (-0.07, -0.17, -0.49), 6.21)
        self.add(axes_1[:2]) #Leaving out x and y for now. 

        for time_step in range(1000):
            neuron_1_mean=np.mean(mlp_pre[time_step, :, :])
            neuron_1_max=np.max(np.abs(mlp_pre[time_step, :, :]-neuron_1_mean))

            surf_func_with_axes = partial(
                surf_func, 
                axes=axes_1,
                surf_array=(mlp_pre[time_step, :, :]-neuron_1_mean)/neuron_1_max, 
                scale=1.0
            )

            surface = parametric_surface(
                surf_func_with_axes,  
                u_range=[0, 1.0],
                v_range=[0, 1.0],
                resolution=(resolution, resolution),
            )

            ts = textured_surface(surface, str(data_dir/'training_heatmaps'/('logits_'+str(10*time_step).zfill(4)+'_'+str(neuron_idx).zfill(3)+'.png')))
            set_surface_shading(ts, 0.0, 0.1, 0)
            ts.set_opacity(0.8)

            self.add(ts)
            self.wait(0.1)
            self.remove(ts)


        self.wait(20)
        # Scene authoring stops here in the interactive editor.

class P59_probe_1(WelchScene):
    def construct(self): 

        ckpt_dir=Path(welch_path('grokking/from_linux/grok_1764706121'))

        p=113

        '''
        Ok not quite sure how I want to architect this last big scene just yet
        I think it's probably (6) different scenes, 4 surfaces, two probe plots
        And I guess we have a bunch of model checkpoints...
        So I'll probably actually load up checkpoints here and do my own little forward pass? 
        I think that's the vibe...
        Ok I think running model on Mac is probably not the move actually, 
        let me jump to linux and export what I need for these 6 animations. 

        '''

        
        probe_cos_0=np.load(data_dir/'training_sparse_probe_cos_0.npy')
        probe_sin_0=np.load(data_dir/'training_sparse_probe_sin_0.npy')
        # probe_cos_1=np.load(data_dir/'training_sparse_probe_cos_1.npy')
        # probe_sin_1=np.load(data_dir/'training_sparse_probe_sin_1.npy')


        axis_1 = diagram_axes(
            x_range=[0, 1.0, 1],
            y_range=[-1.0, 1.0, 1],
            width=2*2.4,
            height=2*0.56,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":1.8,
                "tip_config": {"width":0.02, "length":0.02}
                }
            )

        # axis_1.move_to([-4, 3.05, 0])
        x_label=math_tex('x', font_size=24)
        x_label.set_color(CHILL_BROWN)
        x_label.next_to(axis_1, gx.RIGHT, buff=0.1)
        x_label.shift([0, -0.1, 0])

        self.add(axis_1, x_label)
        for time_step in range(1000):

            pts_curve_1=[]
            for j in range(p):
                x = j / p
                y = probe_cos_0[time_step][j]
                pts_curve_1.append(axis_1.c2p(x, y))

            curve_1 = gx.VMobject(stroke_width=3)
            curve_1.set_points_smoothly(pts_curve_1)
            curve_1.set_color(YELLOW)

            pts_curve_2=[]
            for j in range(p):
                x = j / p
                y = probe_sin_0[time_step][j]
                pts_curve_2.append(axis_1.c2p(x, y))

            curve_2 = gx.VMobject(stroke_width=3)
            curve_2.set_points_smoothly(pts_curve_2)
            curve_2.set_color(MAGENTA)
        

            self.add(curve_1, curve_2)
            self.wait(0.1)
            self.remove(curve_1, curve_2)





        self.wait(20)
        # Scene authoring stops here in the interactive editor.

class P59_probe_2(WelchScene):
    def construct(self): 

        ckpt_dir=Path(welch_path('grokking/from_linux/grok_1764706121'))

        p=113

        probe_cos_1=np.load(data_dir/'training_sparse_probe_cos_1.npy')
        probe_sin_1=np.load(data_dir/'training_sparse_probe_sin_1.npy')


        axis_1 = diagram_axes(
            x_range=[0, 1.0, 1],
            y_range=[-1.0, 1.0, 1],
            width=2*2.4,
            height=2*0.56,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":1.8,
                "tip_config": {"width":0.02, "length":0.02}
                }
            )

        # axis_1.move_to([-4, 3.05, 0])
        x_label=math_tex('x', font_size=24)
        x_label.set_color(CHILL_BROWN)
        x_label.next_to(axis_1, gx.RIGHT, buff=0.1)
        x_label.shift([0, -0.1, 0])

        self.add(axis_1, x_label)
        for time_step in range(1000):

            pts_curve_1=[]
            for j in range(p):
                x = j / p
                y = probe_cos_1[time_step][j]
                pts_curve_1.append(axis_1.c2p(x, y))

            curve_1 = gx.VMobject(stroke_width=3)
            curve_1.set_points_smoothly(pts_curve_1)
            curve_1.set_color(CYAN)

            pts_curve_2=[]
            for j in range(p):
                x = j / p
                y = probe_sin_1[time_step][j]
                pts_curve_2.append(axis_1.c2p(x, y))

            curve_2 = gx.VMobject(stroke_width=3)
            curve_2.set_points_smoothly(pts_curve_2)
            curve_2.set_color(gx.RED)
        

            self.add(curve_1, curve_2)
            self.wait(0.1)
            self.remove(curve_1, curve_2)





        self.wait(20)
        # Scene authoring stops here in the interactive editor.
