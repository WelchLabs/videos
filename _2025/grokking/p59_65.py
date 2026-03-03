from manimlib import *
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
svg_dir=Path('/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/grokking/graphics/to_manim')
data_dir=Path('/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/grokking/from_linux/grok_1764706121')

def viridis_hex(value, vmin, vmax):
    """
    Map a scalar `value` in [vmin, vmax] to a Viridis color (hex string).
    """
    # Normalize into [0,1]
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    # Get RGBA from viridis
    rgba = cm.viridis(norm(value))
    # Convert to hex
    return colors.to_hex(rgba)

def black_to_tan_hex(value, vmin, vmax=1):
    """
    Map a scalar `value` in [vmin, vmax] to a color from black to FRESH_TAN (#dfd0b9).
    """
    cmap = colors.LinearSegmentedColormap.from_list('black_tan', ['#000000', '#dfd0b9'])
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return colors.to_hex(cmap(norm(value)))

def softmax_with_temperature(logits, temperature=1.0, axis=-1):
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled, axis=axis, keepdims=True))
    return exp_scaled / np.sum(exp_scaled, axis=axis, keepdims=True)




def draw_inputs(self, activations, all_svgs, reset=False, example_index=0, wait=0.0):

    #Borders and fills
    input_mapping_1a=[[0, 1], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]]
    input_mapping_1b=[[19, 20], [21, 22]]
    input_mapping_2a=[[23, 24], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37], [38, 39], [40, 41]]
    input_mapping_2b=[[42, 43], [44, 45]]
    input_mapping_3a=[[46, 47], [51, 52], [53, 54], [55, 56], [57, 58], [59, 60], [61, 62], [63, 64]]
    input_mapping_3b=[[65, 66], [67, 68]]


    #Color inputs
    for mapping, activations_index, offset in zip([input_mapping_1a, input_mapping_1b, input_mapping_2a, input_mapping_2b, input_mapping_3a, input_mapping_3b], 
                                          [0, 0, 1, 1, 2, 2], [0, 112, 0, 112, 0, 112]):
        for i, idx in enumerate(mapping):
            if i+offset == activations['x'][example_index][activations_index]:
                all_svgs[2][idx[0]].set_color(FRESH_TAN)
            else:
                all_svgs[2][idx[0]].set_color(BLACK)
            if reset:
                all_svgs[2][idx[0]].set_color(BLACK)
        if wait!=0.0: self.wait(wait)

def draw_embeddings(self, activations, all_svgs, reset=False, example_index=0, wait=0, colormap=black_to_tan_hex):

    embedding_fill_indices_1=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    embedding_fill_indices_2=[28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
    embedding_fill_indices_3=[53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73]
    # vmin=np.min(activations['blocks.0.hook_resid_pre'])*0.9 
    # vmax=np.max(activations['blocks.0.hook_resid_pre'])*0.9
    for i, indices in enumerate([embedding_fill_indices_1, embedding_fill_indices_2, embedding_fill_indices_3]):
        vmin=np.min(activations['blocks.0.hook_resid_pre'][example_index][i])*1.5 #Scaling by column
        vmax=np.max(activations['blocks.0.hook_resid_pre'][example_index][i])*1.5
        for j, idx in enumerate(indices):
            # c=viridis_hex(activations['blocks.0.hook_resid_pre'][example_index, 0, i], vmin, vmax)
            c=colormap(activations['blocks.0.hook_resid_pre'][example_index, i, j], vmin, vmax)
            # print(activations['blocks.0.hook_resid_pre'][example_index, 0, j])
            all_svgs[4][idx].set_color(c)
            if reset: all_svgs[4][idx].set_color(BLACK)
            if wait!=0.0: self.wait(wait)

def draw_embeddings_2(self, activations, all_svgs, reset=False, example_index=0, wait=0, colormap=black_to_tan_hex):

    embedding_fill_indices_1=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    embedding_fill_indices_2=[28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
    embedding_fill_indices_3=[53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73]
    # vmin=np.min(activations['blocks.0.hook_resid_pre'])*0.9 
    # vmax=np.max(activations['blocks.0.hook_resid_pre'])*0.9
    for i, indices in enumerate([embedding_fill_indices_1, embedding_fill_indices_2, embedding_fill_indices_3]):
        vmin=np.min(activations['blocks.0.hook_resid_pre'][example_index][i])*0.8 #Scaling by column
        vmax=np.max(activations['blocks.0.hook_resid_pre'][example_index][i])*0.8
        for j, idx in enumerate(indices):
            # c=viridis_hex(activations['blocks.0.hook_resid_pre'][example_index, 0, i], vmin, vmax)
            c=colormap(activations['blocks.0.hook_resid_pre'][example_index, i, j], vmin, vmax)
            # print(activations['blocks.0.hook_resid_pre'][example_index, 0, j])
            all_svgs[4][idx].set_color(c)
            if reset: all_svgs[4][idx].set_color(BLACK)
            if wait!=0.0: self.wait(wait)


def draw_attention_values(self, activations, all_svgs, reset=False, example_index=0, wait=0, colormap=black_to_tan_hex):

        ## Attention - Values - mapping again is hacky here, technicall first layer should be same for all
        ## Keep moving for now.
        vmin=np.min(activations['blocks.0.attn.hook_v'][example_index])*0.25 #Scaling
        vmax=np.max(activations['blocks.0.attn.hook_v'][example_index])*0.25
        value_fill_indices_1=[0, 2, 4, 6, 8]
        value_fill_indices_2=[10, 12, 14, 16, 18]
        value_fill_indices_3=[20, 22, 24, 26, 28]
        value_fill_indices_4=[30, 32, 34, 36, 38]

        for head_id, indices in enumerate([value_fill_indices_1, value_fill_indices_2, value_fill_indices_3, value_fill_indices_4]):
            for j, idx in enumerate(indices):
                c=colormap(activations['blocks.0.attn.hook_v'][example_index, head_id, 1, j], vmin, vmax)
                all_svgs[12][idx].set_color(c)
                if reset: all_svgs[4][idx].set_color(BLACK)
                if wait!=0.0: self.wait(wait)

def draw_attention_patterns(self, activations, all_svgs, reset=False, example_index=0, wait=0, colormap=black_to_tan_hex):
        #Attention - attention patterns - need to be more precise with these!
        vmin=0
        vmax=0.8
        attn_fill_indices=[[0,0], [1,0], [1,1], [2,0], [2,1], [2,2]] #Indices to sample matrix at
        head_id=0
        for head_id, offset in enumerate([0, 6, 12, 18]):
            for j, idx in enumerate(attn_fill_indices):
                a=activations['blocks.0.attn.hook_attn'][example_index, head_id, idx[0], idx[1]]
                c=black_to_tan_hex(a, vmin, vmax)
                all_svgs[13][offset+j].set_color(c)
                if reset: all_svgs[13][offset+j].set_color(BLACK)
                self.remove(all_svgs[7]); self.add(all_svgs[7]) #Stacking for grid
                if wait!=0.0: self.wait(wait)

def draw_mlp_1(self, activations, all_svgs, reset=False, example_index=0, wait=0, colormap=black_to_tan_hex):
    #MLP layer 1 (565, 3, 128) 
    vmin=np.min(activations['blocks.0.hook_resid_mid'][example_index])*0.85 #Scaling
    vmax=np.max(activations['blocks.0.hook_resid_mid'][example_index])*0.85
    mlp_indices_1=[0, 2, 4, 6, 8, 10, 12]
    for i, idx in enumerate(mlp_indices_1):
        c=black_to_tan_hex(activations['blocks.0.hook_resid_mid'][example_index, 2, i], vmin, vmax)
        all_svgs[9][idx].set_color(c)
        if reset: all_svgs[9][idx].set_color(BLACK)
        if wait!=0.0: self.wait(wait)

def draw_mlp_2(self, activations, all_svgs, reset=False, example_index=0, wait=0, colormap=black_to_tan_hex):
    #MLP Layer 2 (565, 3, 512)
    vmin=np.min(activations['blocks.0.mlp.hook_pre'][example_index])*0.85 #Scaling
    vmax=np.max(activations['blocks.0.mlp.hook_pre'][example_index])*0.85
    mlp_indices_2=[14, 16, 18, 20, 22, 24, 26, 28, 30]
    for i, idx in enumerate(mlp_indices_2):
        c=black_to_tan_hex(activations['blocks.0.mlp.hook_pre'][example_index, 2, i], vmin, vmax)
        all_svgs[9][idx].set_color(c)
        if reset: all_svgs[9][idx].set_color(BLACK)
        if wait!=0.0: self.wait(wait)

def draw_mlp_3(self, activations, all_svgs, reset=False, example_index=0, wait=0, colormap=black_to_tan_hex):
    # MLP Layer 3 (565, 3, 128) 
    vmin=np.min(activations['blocks.0.hook_mlp_out'][example_index])*0.85 #Scaling
    vmax=np.max(activations['blocks.0.hook_mlp_out'][example_index])*0.85
    mlp_indices_3=[32, 34, 36, 38, 40, 42, 44]
    for i, idx in enumerate(mlp_indices_3):
        c=black_to_tan_hex(activations['blocks.0.hook_mlp_out'][example_index, 2, i], vmin, vmax)
        all_svgs[9][idx].set_color(c)
        if reset: all_svgs[9][idx].set_color(BLACK)
        if wait!=0.0: self.wait(wait)



def draw_logits(self, activations, all_svgs, reset=False, example_index=0, wait=0, colormap=black_to_tan_hex, temperature=25.0):

        #Logits or probs (565, 113)
        logit_indices_1=[3, 5, 7, 9, 11, 13, 15, 17]
        logit_indices_2=[19, 21]

        #Don't love probs or logits, how about temperature?
        probs_sortof=softmax_with_temperature(activations['logits'][example_index], temperature=temperature, axis=0)


        vmin=np.min(probs_sortof)*1.0 #Scaling
        vmax=np.max(probs_sortof)*1.0

        for i, idx in enumerate(logit_indices_1):
            c=black_to_tan_hex(probs_sortof[i], vmin, vmax)
            all_svgs[11][idx].set_color(c)
            if reset: all_svgs[11][idx].set_color(BLACK)
            if wait!=0.0: self.wait(wait)

        for i, idx in enumerate(logit_indices_2):
            c=black_to_tan_hex(probs_sortof[i+111], vmin, vmax)
            all_svgs[11][idx].set_color(c)
            if reset: all_svgs[11][idx].set_color(BLACK)
            if wait!=0.0: self.wait(wait)


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


class Dot3D(Sphere):
    def __init__(self, center=ORIGIN, radius=0.05, **kwargs):
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


class P59_mlp_surface_1b(InteractiveScene):
    def construct(self): 
        p=113
        neuron_idx=341

        ckpt_dir=Path('/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/grokking/from_linux/grok_1764706121')
        mlp_pre=np.load(data_dir/('training_mlp_pre'+str(neuron_idx)+'.npy'))

        axes_1 = ThreeDAxes(
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

        x_label = Tex("x", font_size=32).next_to(axes_1.x_axis.get_end(), RIGHT, buff=0.1).set_color(CHILL_BROWN)
        y_label = Tex("y", font_size=32).next_to(axes_1.y_axis.get_end(), UP, buff=0.1).set_color(CHILL_BROWN)
        axes_1_group=VGroup(axes_1[:2], x_label, y_label)
        x_label.rotate(DEGREES*90, [1, 0, 0])
        y_label.rotate(DEGREES*90, [1, 0, 0])
        y_label.rotate(DEGREES*90, [0, 0, 1])
        axes_1[0].rotate(DEGREES*90, [1, 0, 0])
        axes_1[1].rotate(DEGREES*90, [0, 1, 0])


        # self.frame.reorient(43, 57, 0, (-0.15, -0.07, -0.33), 6.60)
        # self.frame.reorient(43, 42, 0, (0.07, -0.2, -0.58), 6.21)
        # self.frame.reorient(45, 45, 0, (-0.07, -0.17, -0.49), 6.21)
        # self.frame.reorient(45, 44, 0, (-0.07, -0.17, -0.49), 6.21)
        self.frame.reorient(45, 57, 0, (-0.07, -0.17, -0.49), 6.21)
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

            surface = ParametricSurface(
                surf_func_with_axes,  
                u_range=[0, 1.0],
                v_range=[0, 1.0],
                resolution=(resolution, resolution),
            )

            ts = TexturedSurface(surface, str(data_dir/'training_heatmaps'/('mlp_pre_'+str(10*time_step).zfill(4)+'_'+str(neuron_idx).zfill(3)+'.png')))
            ts.set_shading(0.0, 0.1, 0)
            ts.set_opacity(0.8)

            self.add(ts)
            self.wait(0.1)
            self.remove(ts)


        self.wait(20)
        self.embed()


class P59_mlp_surface_2b(InteractiveScene):
    def construct(self): 
        p=113
        neuron_idx=106

        ckpt_dir=Path('/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/grokking/from_linux/grok_1764706121')
        mlp_pre=np.load(data_dir/('training_mlp_pre'+str(neuron_idx)+'.npy'))

        axes_1 = ThreeDAxes(
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

        x_label = Tex("x", font_size=32).next_to(axes_1.x_axis.get_end(), RIGHT, buff=0.1).set_color(CHILL_BROWN)
        y_label = Tex("y", font_size=32).next_to(axes_1.y_axis.get_end(), UP, buff=0.1).set_color(CHILL_BROWN)
        axes_1_group=VGroup(axes_1[:2], x_label, y_label)
        x_label.rotate(DEGREES*90, [1, 0, 0])
        y_label.rotate(DEGREES*90, [1, 0, 0])
        y_label.rotate(DEGREES*90, [0, 0, 1])
        axes_1[0].rotate(DEGREES*90, [1, 0, 0])
        axes_1[1].rotate(DEGREES*90, [0, 1, 0])


        # self.frame.reorient(43, 57, 0, (-0.15, -0.07, -0.33), 6.60)
        # self.frame.reorient(43, 42, 0, (0.07, -0.2, -0.58), 6.21)
        # self.frame.reorient(45, 45, 0, (-0.07, -0.17, -0.49), 6.21)
        # self.frame.reorient(45, 44, 0, (-0.07, -0.17, -0.49), 6.21)
        # self.frame.reorient(49, 47, 0, (-0.07, -0.17, -0.49), 6.21)
        self.frame.reorient(45, 57, 0, (-0.07, -0.17, -0.49), 6.21)
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

            surface = ParametricSurface(
                surf_func_with_axes,  
                u_range=[0, 1.0],
                v_range=[0, 1.0],
                resolution=(resolution, resolution),
            )

            ts = TexturedSurface(surface, str(data_dir/'training_heatmaps'/('mlp_pre_'+str(10*time_step).zfill(4)+'_'+str(neuron_idx).zfill(3)+'.png')))
            ts.set_shading(0.0, 0.1, 0)
            ts.set_opacity(0.8)

            self.add(ts)
            self.wait(0.1)
            self.remove(ts)


        self.wait(20)
        self.embed()

class P59_mlp_surface_3b(InteractiveScene):
    def construct(self): 
        p=113
        neuron_idx=1

        ckpt_dir=Path('/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/grokking/from_linux/grok_1764706121')
        mlp_pre=np.load(data_dir/('training_mlp_out'+str(neuron_idx).zfill(3)+'.npy'))

        axes_1 = ThreeDAxes(
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

        x_label = Tex("x", font_size=32).next_to(axes_1.x_axis.get_end(), RIGHT, buff=0.1).set_color(CHILL_BROWN)
        y_label = Tex("y", font_size=32).next_to(axes_1.y_axis.get_end(), UP, buff=0.1).set_color(CHILL_BROWN)
        axes_1_group=VGroup(axes_1[:2], x_label, y_label)
        x_label.rotate(DEGREES*90, [1, 0, 0])
        y_label.rotate(DEGREES*90, [1, 0, 0])
        y_label.rotate(DEGREES*90, [0, 0, 1])
        axes_1[0].rotate(DEGREES*90, [1, 0, 0])
        axes_1[1].rotate(DEGREES*90, [0, 1, 0])


        # self.frame.reorient(43, 57, 0, (-0.15, -0.07, -0.33), 6.60)
        # self.frame.reorient(43, 42, 0, (0.07, -0.2, -0.58), 6.21)
        # self.frame.reorient(45, 45, 0, (-0.07, -0.17, -0.49), 6.21)
        # self.frame.reorient(45, 44, 0, (-0.07, -0.17, -0.49), 6.21)
        self.frame.reorient(45, 57, 0, (-0.07, -0.17, -0.49), 6.21)
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

            surface = ParametricSurface(
                surf_func_with_axes,  
                u_range=[0, 1.0],
                v_range=[0, 1.0],
                resolution=(resolution, resolution),
            )

            ts = TexturedSurface(surface, str(data_dir/'training_heatmaps'/('mlp_out_'+str(10*time_step).zfill(4)+'_'+str(neuron_idx).zfill(3)+'.png')))
            ts.set_shading(0.0, 0.1, 0)
            ts.set_opacity(0.8)

            self.add(ts)
            self.wait(0.1)
            self.remove(ts)


        self.wait(20)
        self.embed()


class P59_mlp_surface_4b(InteractiveScene):
    def construct(self): 
        p=113
        neuron_idx=7

        ckpt_dir=Path('/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/grokking/from_linux/grok_1764706121')
        mlp_pre=np.load(data_dir/('logits'+str(neuron_idx).zfill(3)+'.npy'))

        axes_1 = ThreeDAxes(
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

        x_label = Tex("x", font_size=32).next_to(axes_1.x_axis.get_end(), RIGHT, buff=0.1).set_color(CHILL_BROWN)
        y_label = Tex("y", font_size=32).next_to(axes_1.y_axis.get_end(), UP, buff=0.1).set_color(CHILL_BROWN)
        axes_1_group=VGroup(axes_1[:2], x_label, y_label)
        x_label.rotate(DEGREES*90, [1, 0, 0])
        y_label.rotate(DEGREES*90, [1, 0, 0])
        y_label.rotate(DEGREES*90, [0, 0, 1])
        axes_1[0].rotate(DEGREES*90, [1, 0, 0])
        axes_1[1].rotate(DEGREES*90, [0, 1, 0])


        # self.frame.reorient(43, 57, 0, (-0.15, -0.07, -0.33), 6.60)
        # self.frame.reorient(43, 42, 0, (0.07, -0.2, -0.58), 6.21)
        # self.frame.reorient(45, 45, 0, (-0.07, -0.17, -0.49), 6.21)
        # self.frame.reorient(45, 44, 0, (-0.07, -0.17, -0.49), 6.21)
        # self.frame.reorient(49, 47, 0, (-0.07, -0.17, -0.49), 6.21)
        self.frame.reorient(45, 57, 0, (-0.07, -0.17, -0.49), 6.21)
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

            surface = ParametricSurface(
                surf_func_with_axes,  
                u_range=[0, 1.0],
                v_range=[0, 1.0],
                resolution=(resolution, resolution),
            )

            ts = TexturedSurface(surface, str(data_dir/'training_heatmaps'/('logits_'+str(10*time_step).zfill(4)+'_'+str(neuron_idx).zfill(3)+'.png')))
            ts.set_shading(0.0, 0.1, 0)
            ts.set_opacity(0.8)

            self.add(ts)
            self.wait(0.1)
            self.remove(ts)


        self.wait(20)
        self.embed()


class P59_probe_1(InteractiveScene):
    def construct(self): 

        ckpt_dir=Path('/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/grokking/from_linux/grok_1764706121')

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


        axis_1 = Axes(
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
        x_label=Tex('x', font_size=24)
        x_label.set_color(CHILL_BROWN)
        x_label.next_to(axis_1, RIGHT, buff=0.1)
        x_label.shift([0, -0.1, 0])

        self.add(axis_1, x_label)
        for time_step in range(1000):

            pts_curve_1=[]
            for j in range(p):
                x = j / p
                y = probe_cos_0[time_step][j]
                pts_curve_1.append(axis_1.c2p(x, y))

            curve_1 = VMobject(stroke_width=3)
            curve_1.set_points_smoothly(pts_curve_1)
            curve_1.set_color(YELLOW)

            pts_curve_2=[]
            for j in range(p):
                x = j / p
                y = probe_sin_0[time_step][j]
                pts_curve_2.append(axis_1.c2p(x, y))

            curve_2 = VMobject(stroke_width=3)
            curve_2.set_points_smoothly(pts_curve_2)
            curve_2.set_color(MAGENTA)
        

            self.add(curve_1, curve_2)
            self.wait(0.1)
            self.remove(curve_1, curve_2)





        self.wait(20)
        self.embed() 



class P59_probe_2(InteractiveScene):
    def construct(self): 

        ckpt_dir=Path('/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/grokking/from_linux/grok_1764706121')

        p=113

        probe_cos_1=np.load(data_dir/'training_sparse_probe_cos_1.npy')
        probe_sin_1=np.load(data_dir/'training_sparse_probe_sin_1.npy')


        axis_1 = Axes(
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
        x_label=Tex('x', font_size=24)
        x_label.set_color(CHILL_BROWN)
        x_label.next_to(axis_1, RIGHT, buff=0.1)
        x_label.shift([0, -0.1, 0])

        self.add(axis_1, x_label)
        for time_step in range(1000):

            pts_curve_1=[]
            for j in range(p):
                x = j / p
                y = probe_cos_1[time_step][j]
                pts_curve_1.append(axis_1.c2p(x, y))

            curve_1 = VMobject(stroke_width=3)
            curve_1.set_points_smoothly(pts_curve_1)
            curve_1.set_color(CYAN)

            pts_curve_2=[]
            for j in range(p):
                x = j / p
                y = probe_sin_1[time_step][j]
                pts_curve_2.append(axis_1.c2p(x, y))

            curve_2 = VMobject(stroke_width=3)
            curve_2.set_points_smoothly(pts_curve_2)
            curve_2.set_color(RED)
        

            self.add(curve_1, curve_2)
            self.wait(0.1)
            self.remove(curve_1, curve_2)





        self.wait(20)
        self.embed() 














