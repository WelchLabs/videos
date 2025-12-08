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


svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/grokking/graphics/to_manim')
data_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/grokking/from_linux/grok_1764706121')


class P44_50(InteractiveScene):
    def construct(self): 

        p=113
        svg_files=list(sorted(svg_dir.glob('*network_to_manim*')))

        with open(data_dir/'final_model_activations_sample.p', 'rb') as f:
            activations = pickle.load(f)

        all_svgs=Group()
        for svg_file in svg_files[1:21]: #Expand if I add more artboards
            svg_image=SVGMobject(str(svg_file))
            all_svgs.add(svg_image[1:]) #Thowout background

        all_svgs.scale(6.0) #Eh?

        # p41 - Ok need to pick up with my linear probes, 
        # then turn them into circles

        example_index=0
        self.frame.reorient(0, 0, 0, (0, 0, 0), 8.0)

        draw_inputs(self, activations, all_svgs, reset=False, example_index=example_index, wait=0)
        draw_embeddings(self, activations, all_svgs, reset=False, example_index=example_index, wait=0, colormap=black_to_tan_hex)
        draw_attention_values(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex)
        draw_attention_patterns(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex)
        draw_mlp_1(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex)
        draw_mlp_2(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex)
        draw_mlp_3(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex)
        draw_logits(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex, temperature=25.0)
        self.add(all_svgs[:15], all_svgs[16])
        self.remove(all_svgs[7]); self.add(all_svgs[7]) 


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

        axis_2 = Axes(
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

        axis_1.move_to([-4, 3.05, 0])
        x_label=Tex('x', font_size=24)
        x_label.set_color(CHILL_BROWN)
        x_label.next_to(axis_1, RIGHT, buff=0.1)
        x_label.shift([0, -0.1, 0])
        
        axis_2.move_to([-4, -2.85, 0])
        y_label=Tex('y', font_size=24)
        y_label.set_color(CHILL_BROWN)
        y_label.next_to(axis_2, RIGHT, buff=0.1)
        y_label.shift([0, -0.1, 0])

        sparse_probe_1=np.load(data_dir/'sparse_probe_1.npy')
        sparse_probe_2=np.load(data_dir/'sparse_probe_2.npy')
        sparse_probe_3=np.load(data_dir/'sparse_probe_3.npy')
        sparse_probe_4=np.load(data_dir/'sparse_probe_4.npy')

        pts_curve_1=[]
        for j in range(p):
            x = j / p
            y = sparse_probe_1[j]
            pts_curve_1.append(axis_1.c2p(x, y))

        curve_1 = VMobject(stroke_width=3)
        curve_1.set_points_smoothly(pts_curve_1)
        curve_1.set_color(YELLOW)

        pts_curve_2=[]
        for j in range(p):
            x = j / p
            y = sparse_probe_2[j]
            pts_curve_2.append(axis_1.c2p(x, y))

        curve_2 = VMobject(stroke_width=3)
        curve_2.set_points_smoothly(pts_curve_2)
        curve_2.set_color(MAGENTA)

        pts_curve_3=[]
        for j in range(p):
            x = j / p
            y = sparse_probe_3[j]
            pts_curve_3.append(axis_2.c2p(x, y))

        curve_3 = VMobject(stroke_width=3)
        curve_3.set_points_smoothly(pts_curve_3)
        curve_3.set_color(CYAN)

        pts_curve_4=[]
        for j in range(p):
            x = j / p
            y = sparse_probe_4[j]
            pts_curve_4.append(axis_2.c2p(x, y))

        curve_4 = VMobject(stroke_width=3)
        curve_4.set_points_smoothly(pts_curve_4)
        curve_4.set_color(RED)


        wave_label_1 =  Tex(r'\cos \big(\tfrac{8\pi}{113}x\big)')
        wave_label_1.set_color(YELLOW)
        wave_label_1.scale(0.45*1.5)
        wave_label_1.move_to([-0.9, 3.65, 0])


        wave_label_2 = Tex(r'\sin \big(\tfrac{8\pi}{113}x\big)')
        wave_label_2.set_color(MAGENTA)
        wave_label_2.scale(0.45*1.5)
        wave_label_2.move_to([-0.95, 2.65, 0])

        wave_label_3 = Tex(r'\cos \big(\tfrac{8\pi}{113}y\big)')
        wave_label_3.set_color(CYAN)
        wave_label_3.scale(0.45*1.5)
        wave_label_3.move_to([-0.9, -2.2, 0])

        wave_label_4 = Tex(r'\sin \big(\tfrac{8\pi}{113}y\big)')
        wave_label_4.set_color(RED)
        wave_label_4.scale(0.45*1.5)
        wave_label_4.move_to([-0.9, -3.2, 0])


        all_flat_objs=Group(all_svgs, axis_1, axis_2, curve_1, curve_2, curve_3, curve_4, x_label, y_label)
                            # wave_label_1, wave_label_2, wave_label_3, wave_label_4, )
        all_flat_objs.rotate(90*DEGREES, [1, 0, 0])
        self.frame.reorient(0, 90, 0, (0.00, 0.0, 0.00), 7.86)

        self.add(all_svgs[18])
        self.add(axis_1, axis_2, x_label, y_label)
        self.add(curve_1, curve_2, curve_3, curve_4)
        # self.add(wave_label_1, wave_label_2, wave_label_3, wave_label_4)
        self.remove(all_svgs[7]); self.add(all_svgs[7]) 
        self.remove(all_svgs[9]); self.add(all_svgs[9]) 
        

        self.add(all_svgs[19])
        nudge_group_1=Group(axis_2, curve_3, curve_4)
        nudge_group_1.shift([0, 0, -0.15])

        all_svgs[19].shift([0.05, 0, 0.2])
        all_svgs[19][24:].shift([0, 0, -0.18])


        # self.remove(all_svgs[19][24:])

        self.wait()





        #Ok so I think this is just like p30ish, where I fade out everything past the second MLP layer?
        mid_mlp_fade_group=Group(all_svgs[14][9],
                                all_svgs[14][:3], 
                                all_svgs[10],
                                all_svgs[11],
                                all_svgs[0][7:14],
                                all_svgs[0][-1],
                                all_svgs[8][-105:],
                                all_svgs[9][-14:],
                                all_svgs[14][-20:])

        self.wait()
        self.play(FadeOut(mid_mlp_fade_group), 
                 self.frame.animate.reorient(0, 0, 0, (1.32, 0.17, 0.0), 8.35), 
                 run_time=4)
        self.wait()


        # Hmmmm how do I want to build this surface?
        # Might be interesting to start will all just points the bulid the surface?
        # Trying to think through how we go from 2D to 3d here too...
        # Yeah kinda leaning towards setting up and explaining the x/y grid first maybe?
        # So, this will definitely need to be a separate scene that we bring together in premiere
        # In the 2d view I need to get the framing right, and s
        # Oh wait do I need to include probes here? Yes, right, like we're building.
        # Ok got em. 
        # Hmm if i want to "bring over" the probe plots that's might get tricky
        # ok one problem at a time tho
        #
        # Ok, I thought about the 2d vs 3d stuff, here's what I think: 
        # I do think, for some scenes at least, having the cosine and sine waves
        # interact with the 3D surfaces is going to be important. 
        # So I think It makes sense to "flip up" to network into 3D land
        # Now I don't think I actually do all my stuff in the same scene,
        # But I do think for a couple I can/should. 







        self.wait(20)
        self.embed()



class P41_43(InteractiveScene):
    def construct(self): 

        p=113

        svg_files=list(sorted(svg_dir.glob('*network_to_manim*')))

        with open(data_dir/'final_model_activations_sample.p', 'rb') as f:
            activations = pickle.load(f)

        all_svgs=Group()
        for svg_file in svg_files[1:20]: #Expand if I add more artboards
            svg_image=SVGMobject(str(svg_file))
            all_svgs.add(svg_image[1:]) #Thowout background

        all_svgs.scale(6.0) #Eh?

        # p41 - Ok need to pick up with my linear probes, 
        # then turn them into circles

        example_index=0
        self.frame.reorient(0, 0, 0, (0, 0, 0), 8.0)

        draw_inputs(self, activations, all_svgs, reset=False, example_index=example_index, wait=0)
        draw_embeddings(self, activations, all_svgs, reset=False, example_index=example_index, wait=0, colormap=black_to_tan_hex)
        draw_attention_values(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex)
        draw_attention_patterns(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex)
        draw_mlp_1(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex)
        draw_mlp_2(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex)
        draw_mlp_3(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex)
        draw_logits(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex, temperature=25.0)
        self.add(all_svgs[:15], all_svgs[16])
        self.remove(all_svgs[7]); self.add(all_svgs[7])        

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

        axis_2 = Axes(
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

        axis_1.move_to([-4, 3.05, 0])
        x_label=Tex('x', font_size=24)
        x_label.set_color(CHILL_BROWN)
        x_label.next_to(axis_1, RIGHT, buff=0.1)
        x_label.shift([0, -0.1, 0])
        
        axis_2.move_to([-4, -2.85, 0])
        y_label=Tex('y', font_size=24)
        y_label.set_color(CHILL_BROWN)
        y_label.next_to(axis_2, RIGHT, buff=0.1)
        y_label.shift([0, -0.1, 0])

        sparse_probe_1=np.load(data_dir/'sparse_probe_1.npy')
        sparse_probe_2=np.load(data_dir/'sparse_probe_2.npy')
        sparse_probe_3=np.load(data_dir/'sparse_probe_3.npy')
        sparse_probe_4=np.load(data_dir/'sparse_probe_4.npy')

        pts_curve_1=[]
        dots_curve_1=VGroup()
        for j in range(p):
            x = j / p
            y = sparse_probe_1[j]
            pts_curve_1.append(axis_1.c2p(x, y))

            pt = Dot(axis_1.c2p(x, y), radius=0.02, stroke_width=0)
            pt.set_color(WHITE)
            dots_curve_1.add(pt)

        curve_1 = VMobject(stroke_width=3)
        curve_1.set_points_smoothly(pts_curve_1)
        curve_1.set_color(YELLOW)

        pts_curve_2=[]
        dots_curve_2=VGroup()
        for j in range(p):
            x = j / p
            y = sparse_probe_2[j]
            pts_curve_2.append(axis_1.c2p(x, y))

            pt = Dot(axis_1.c2p(x, y), radius=0.02, stroke_width=0)
            pt.set_color(WHITE)
            dots_curve_2.add(pt)

        curve_2 = VMobject(stroke_width=3)
        curve_2.set_points_smoothly(pts_curve_2)
        curve_2.set_color(MAGENTA)

        pts_curve_3=[]
        dots_curve_3=VGroup()
        for j in range(p):
            x = j / p
            y = sparse_probe_3[j]
            pts_curve_3.append(axis_2.c2p(x, y))

            pt = Dot(axis_2.c2p(x, y), radius=0.02, stroke_width=0)
            pt.set_color(WHITE)
            dots_curve_3.add(pt)

        curve_3 = VMobject(stroke_width=3)
        curve_3.set_points_smoothly(pts_curve_3)
        curve_3.set_color(CYAN)

        pts_curve_4=[]
        dots_curve_4=VGroup()
        for j in range(p):
            x = j / p
            y = sparse_probe_4[j]
            pts_curve_4.append(axis_2.c2p(x, y))

            pt = Dot(axis_2.c2p(x, y), radius=0.02, stroke_width=0)
            pt.set_color(WHITE)
            dots_curve_4.add(pt)

        curve_4 = VMobject(stroke_width=3)
        curve_4.set_points_smoothly(pts_curve_4)
        curve_4.set_color(RED)


        wave_label_1 =  Tex(r'\cos \big(\tfrac{8\pi}{113}x\big)')
        wave_label_1.set_color(YELLOW)
        wave_label_1.scale(0.45*1.5)
        wave_label_1.move_to([-0.9, 3.65, 0])

        wave_label_2 = Tex(r'\sin \big(\tfrac{8\pi}{113}x\big)')
        wave_label_2.set_color(MAGENTA)
        wave_label_2.scale(0.45*1.5)
        wave_label_2.move_to([-0.95, 2.65, 0])

        wave_label_3 = Tex(r'\cos \big(\tfrac{8\pi}{113}y\big)')
        wave_label_3.set_color(CYAN)
        wave_label_3.scale(0.45*1.5)
        wave_label_3.move_to([-0.9, -2.2, 0])

        wave_label_4 = Tex(r'\sin \big(\tfrac{8\pi}{113}y\big)')
        wave_label_4.set_color(RED)
        wave_label_4.scale(0.45*1.5)
        wave_label_4.move_to([-0.9, -3.2, 0])


        self.add(all_svgs[18])
        self.add(axis_1, axis_2, x_label, y_label)
        self.add(curve_1, curve_2, curve_3, curve_4)
        self.add(wave_label_1, wave_label_2, wave_label_3, wave_label_4)
        self.wait()

        # Ok great now we can actually start p41
        # It's a little more complicated than I wanted, but that's ok.
        # So I think its fade out the same right half of the network (minus any parts of the emberdding layer)
        # Then move little points (which I should prep above) into two circle plots. 
        # And copies of labels move to each axis of the plots. 

        self.wait()
        self.play(
                  FadeOut(all_svgs[5:15]),
                  FadeOut(all_svgs[0][15:]),
                  FadeOut(all_svgs[0][5:14]), 
                  run_time=5)
        self.wait()


        axis_3 = Axes(
            x_range=[-1.1, 1.1, 1],
            y_range=[-1.1, 1.1, 1],
            width=3.5,
            height=3.5,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2.4,
                "tip_config": {"width":0.02, "length":0.02}
                }
            )
        axis_3[0].set_color(YELLOW)
        axis_3[1].set_color(MAGENTA)
        axis_3.move_to([3, 1.9, 0])
        
        axis_4 = Axes(
            x_range=[-1.1, 1.1, 1],
            y_range=[-1.1, 1.1, 1],
            width=3.5,
            height=3.5,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2.4,
                "tip_config": {"width":0.02, "length":0.02}
                }
            )
        axis_4[0].set_color(CYAN)
        axis_4[1].set_color(RED)
        axis_4.move_to([3, -1.9, 0])


        dots_curve_5=VGroup()
        for j in range(p):
            pt = Dot(axis_3.c2p(sparse_probe_1[j], sparse_probe_2[j]), 
                     radius=0.02, stroke_width=0)
            pt.set_color(WHITE)
            dots_curve_5.add(pt)

        dots_curve_6=VGroup()
        for j in range(p):
            pt = Dot(axis_4.c2p(sparse_probe_3[j], sparse_probe_4[j]), 
                     radius=0.02, stroke_width=0)
            pt.set_color(WHITE)
            dots_curve_6.add(pt)

        wave_label_1_copy=wave_label_1.copy()
        wave_label_1_copy.next_to(axis_3, RIGHT, buff=0.2)
        wave_label_2_copy=wave_label_2.copy()
        wave_label_2_copy.next_to(axis_3, TOP, buff=0)
        wave_label_2_copy.shift([0.9, -0.2, 0])
        wave_label_3_copy=wave_label_3.copy()
        wave_label_3_copy.next_to(axis_4, RIGHT, buff=0.2)
        wave_label_4_copy=wave_label_4.copy()
        wave_label_4_copy.next_to(axis_4, TOP, buff=0)
        wave_label_4_copy.shift([0.9, -0.2, 0])

        self.wait()
        self.play(ShowCreation(axis_3), run_time=2)
        self.play(ReplacementTransform(wave_label_1.copy(), wave_label_1_copy), run_time=2)
        self.play(ReplacementTransform(wave_label_2.copy(), wave_label_2_copy), run_time=2)
        self.wait()

        self.play(FadeIn(dots_curve_1), FadeIn(dots_curve_2), run_time=2)
        self.play(ReplacementTransform(dots_curve_1, dots_curve_5), 
                  ReplacementTransform(dots_curve_2, dots_curve_5), 
                  # lag_ratio=0.5, #This is fun but I think the other way is more clear. 
                  run_time=5)
        self.wait()

        self.play(ShowCreation(axis_4),
                ReplacementTransform(wave_label_3.copy(), wave_label_3_copy), 
                ReplacementTransform(wave_label_4.copy(), wave_label_4_copy), run_time=2)        
        self.play(ReplacementTransform(dots_curve_3, dots_curve_6), 
                  ReplacementTransform(dots_curve_4, dots_curve_6), 
                  # lag_ratio=0.5, #This is fun but I think the other way is more clear. 
                  run_time=5)
        self.wait()

        #P42 Fade out scatter plots and bring in network
        self.play(FadeOut(wave_label_1_copy),
                FadeOut(wave_label_2_copy),
                FadeOut(wave_label_3_copy),
                FadeOut(wave_label_4_copy),
                FadeOut(axis_3),
                FadeOut(axis_4),
                FadeOut(dots_curve_5),
                FadeOut(dots_curve_6),
                # FadeOut(),
                # FadeOut(),
                run_time=2)

        self.play(
                  FadeIn(all_svgs[5:15]),
                  FadeIn(all_svgs[0][15:]),
                  FadeIn(all_svgs[0][5:14]), 
                  run_time=2)        
        self.add(all_svgs[:15], all_svgs[16])
        self.remove(all_svgs[7]); self.add(all_svgs[7]) 

        #Ok now switch to #2 input. 
        self.wait()
        example_index=226
        draw_inputs(self, activations, all_svgs, reset=False, example_index=example_index, wait=0)
        draw_embeddings(self, activations, all_svgs, reset=False, example_index=example_index, wait=0, colormap=black_to_tan_hex)
        draw_attention_values(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex)
        draw_attention_patterns(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex)
        draw_mlp_1(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex)
        draw_mlp_2(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex)
        draw_mlp_3(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex)
        draw_logits(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex, temperature=25.0)
        self.add(all_svgs[:15], all_svgs[16])
        self.remove(all_svgs[7]); self.add(all_svgs[7])  
        self.wait()      

        #P43a - Ok this could be cool and kinda simple.
        #Fade out the guts of the attention layer, bring in cos(x) cos(y) on top of eachother with 
        # a plus sine - maybe elemtary school arithmetic style. 


        self.wait()

        # self.remove(all_svgs[5][10])
        # self.remove(all_svgs[5][1:10])
        # wave_label_1.move_to([-.45, 0.5, 0])
        # wave_label_3.move_to([-.45, 0., 0])

        addition_line=Line(start=[-1.13, -0.3, 0], end=[0.6, -0.3, 0])
        addition_line.set_stroke(width=2)
        addition_line.set_color(WHITE)
        plus_sign=Tex('+', font_size=28)
        plus_sign.set_color(WHITE)
        plus_sign.move_to([0.5, -0.1, 0])

        self.play(FadeOut(all_svgs[6:8]), 
                  FadeOut(all_svgs[12:14]), 
                  FadeOut(all_svgs[5][0]),
                  FadeOut(all_svgs[5][11:]),
                  run_time=2)
        self.play(wave_label_1.animate.move_to([-.45, 0.5, 0]),
                  wave_label_3.animate.move_to([-.45, 0., 0]),
                  run_time=3)
        self.play(ShowCreation(addition_line), ShowCreation(plus_sign), run_time=2)
        self.wait()

        #AH yeah draw nice horizontal line and plus sign let's go


        # self.add(wave_label_1_copy, wave_label_2_copy)
        # self.add(wave_label_3_copy, wave_label_4_copy)
        # self.add(axis_3)
        # self.add(axis_4)
        # self.add(dots_curve_5, dots_curve_6)

        #Ok cool now we cut to the clock again. Probably a good spot to start a new scene. 


        self.wait(20)
        self.embed()











