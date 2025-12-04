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


class P28_31f(InteractiveScene):
    def construct(self): 

        p=113

        svg_files=list(sorted(svg_dir.glob('*network_to_manim*')))

        with open(data_dir/'final_model_activations_sample.p', 'rb') as f:
            activations = pickle.load(f)

        all_svgs=Group()
        for svg_file in svg_files[1:17]: #Expand if I add more artboards
            svg_image=SVGMobject(str(svg_file))
            all_svgs.add(svg_image[1:]) #Thowout background

        all_svgs.scale(6.0) #Eh?

        draw_inputs(self, activations, all_svgs, reset=True, example_index=0)
        draw_attention_patterns(self, activations, all_svgs, reset=True, example_index=0)
        
        #MLP weights
        np.random.seed(5)
        R=np.random.uniform(0.3, 0.75, len(all_svgs[8]))
        for i in range(len(all_svgs[8])):
            all_svgs[8][i].set_opacity(R[i])


        #Ok want to pick up with the same activation structure we left off with!
        example_index=117
        self.frame.reorient(0, 0, 0, (0.02, -0.06, 0.0), 7.58)

        draw_inputs(self, activations, all_svgs, reset=False, example_index=example_index, wait=0)
        draw_embeddings(self, activations, all_svgs, reset=False, example_index=example_index, wait=0, colormap=black_to_tan_hex)
        draw_attention_values(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_attention_patterns(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_mlp_1(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_mlp_2(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_mlp_3(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_logits(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex, temperature=25.0)
        self.remove(all_svgs[7]); self.add(all_svgs[7])

        self.add(all_svgs[:15])
        self.wait()

        # Start p28
        # Ok ok so it would be dope to lower the opacity (or maybe just remove?) 
        # everything after the middle MLP layer. 

        p28_fade_group=Group(all_svgs[14][9],
                            all_svgs[14][:3], 
                            all_svgs[10],
                            all_svgs[11],
                            all_svgs[0][7:14],
                            all_svgs[0][-1],
                            all_svgs[8][-105:],
                            all_svgs[9][-14:],
                            all_svgs[14][-20:])

        # self.remove(all_svgs[14][9]) #MLP Block border
        # self.remove(all_svgs[14][:3]) #Elipses
        # self.remove(all_svgs[10])
        # self.remove(all_svgs[11])
        # self.remove(all_svgs[0][7:14])
        # self.remove(all_svgs[0][-1])
        # self.remove(all_svgs[8][-105:])
        # self.remove(all_svgs[9][-14:])
        # self.remove(all_svgs[14][-20:])

        self.wait()
        self.play(FadeOut(p28_fade_group), run_time=3.0)
        self.wait()

        #p29 new inpur
        example_index=0
        draw_inputs(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1)
        draw_embeddings(self, activations, all_svgs, reset=False, example_index=example_index, wait=0, colormap=black_to_tan_hex)
        draw_attention_values(self, activations, all_svgs, reset=False, example_index=example_index, wait=0, colormap=black_to_tan_hex)
        draw_attention_patterns(self, activations, all_svgs, reset=False, example_index=example_index, wait=0, colormap=black_to_tan_hex)
        self.remove(all_svgs[7]); self.add(all_svgs[7])
        draw_mlp_1(self, activations, all_svgs, reset=False, example_index=example_index, wait=0, colormap=black_to_tan_hex)
        draw_mlp_2(self, activations, all_svgs, reset=False, example_index=example_index, wait=0, colormap=black_to_tan_hex)
        self.wait()



        # activations['blocks.0.mlp.hook_pre'][example_index, 2, 0]

        activation_txt = VGroup()
        activation_arrows = VGroup()

        for i in range(7):
            val = activations['blocks.0.mlp.hook_pre'][example_index, 2, i]
            txt = Tex(f"{val:.2f}", font_size=24)
            activation_txt.add(txt)
        activation_txt.set_color(FRESH_TAN)
        activation_txt.arrange(DOWN, buff=0.128)
        activation_txt.move_to([3.6, 0.73, 0])

        for i in range(7):
            arr = Arrow(
                start=LEFT, end=RIGHT,
                buff=0.05,            # small spacing
                stroke_width=1.4,
                max_tip_length_to_length_ratio=0.3
            ).scale(0.2)
            activation_arrows.add(arr)
        activation_arrows.set_color(CHILL_BROWN)
        activation_arrows.arrange(DOWN, buff=0.16)
        activation_arrows.move_to([3.0, 0.73, 0])

        self.wait()
        self.play(FadeIn(activation_arrows), Write(activation_txt), lag_ratio=0.5)

        # Ok deeper into p30 here. 
        # Need 5-7 little plots 
        # And will probably want to zoom out a bit when I make these axes. 
        # The importantly I want to sweep through inptu values as I add points
        # Probably add some nice curvy arrows to each axis in illustrator
        # And I think use manim axes

        self.play(FadeOut(activation_arrows), FadeOut(activation_txt))
        self.wait()

        axes=VGroup()
        for i in range(7):
            a = Axes(
                x_range=[0, 1.0, 1],
                y_range=[-1.0, 1.0, 1],
                width=2.4,
                height=0.8,
                axis_config={
                    "color": CHILL_BROWN,
                    "include_ticks": False,
                    "include_numbers": False,
                    "include_tip": True,
                    "stroke_width":1.2,
                    "tip_config": {"width":0.02, "length":0.02}
                    }
                )
            axes.add(a)
        axes.arrange(DOWN, buff=0.2)
        axes.move_to([4.8, 0, 0])

        
        #Hmm probably makes sense to remove mean here and from scatter plots - not a big deal.

        all_pts=VGroup()
        neuron_indices=[0, 3, 4, 2, 11, 5, 9]
        for i, neuron_idx in enumerate(neuron_indices):
            neuron_average=activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx].mean()
            neuron_max=np.max(np.abs(activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx]-neuron_average))*1.0 #Might want to bring down
            dese_pts=VGroup()
            for j in range(p):
                x = j / p
                y = (activations['blocks.0.mlp.hook_pre'][j, 2, neuron_idx]-neuron_average)/neuron_max
                pt = Dot(axes[i].c2p(x, y), radius=0.02, color=FRESH_TAN, stroke_width=0)
                dese_pts.add(pt)
            all_pts.add(dese_pts)


        self.wait()
        self.play(Write(all_svgs[15]), Write(axes), run_time=4)
        self.wait()

        # Ok this is loking dope. 
        # Now how do I sweep through varoius inputs, while changing outputs and adding points to the scatter plot?

        #We're on 0,0 go ahead and add this sucker
        for i in range(7):
            self.add(all_pts[i][0])
            self.wait(0.1)
        for i in range(1, p):
            draw_inputs(self, activations, all_svgs, reset=False, example_index=i, wait=0)
            draw_embeddings(self, activations, all_svgs, reset=False, example_index=i, wait=0, colormap=black_to_tan_hex)
            draw_attention_values(self, activations, all_svgs, reset=False, example_index=i, wait=0, colormap=black_to_tan_hex)
            draw_attention_patterns(self, activations, all_svgs, reset=False, example_index=i, wait=0, colormap=black_to_tan_hex)
            self.remove(all_svgs[7]); self.add(all_svgs[7])
            draw_mlp_1(self, activations, all_svgs, reset=False, example_index=i, wait=0, colormap=black_to_tan_hex)
            draw_mlp_2(self, activations, all_svgs, reset=False, example_index=i, wait=0, colormap=black_to_tan_hex)
            for j in range(7):
                self.add(all_pts[j][i])
            self.wait(0.2)
        self.wait()

        # Ok this should be the most complicated part
        # First color points using viridis following p
        # Then fancy scatter plots. I think that by moving pairs of collections of poitns at the same to each 
        # of the 49 scatter plots, I can make it reasonably clear what I'm doing without adding a buynch of labelling. 
        # Ok but first, let's color!

        #My god this is slow lol - removing temporarily
        for i in range(p):
            for j in range(7):
                c=viridis_hex(i, 0, p)
                all_pts[j][i].set_color(c)
                self.wait(0.1)
        self.wait()


        # Ok now for axes here, I think start with just making a 7x7 grid of axes with the same spacing
        # Then I can figure out how to add little boxes around each scattter plot, I think that's the way to 
        # handle the organization
        # Then it's probably draw all the points in final locations, then to replacment transform?

        axes_2 = VGroup()
        for i in range(49):  # 7 × 7
            a = Axes(
                x_range=[-1.0, 1.0, 1],
                y_range=[-1.0, 1.0, 1],
                width=0.8,
                height=0.8,
                axis_config={
                    "color": CHILL_BROWN,
                    "include_ticks": False,
                    "include_numbers": False,
                    "include_tip": True,
                    "stroke_width": 1.2,
                    "tip_config": {"width": 0.02, "length": 0.02},
                },
            )
            axes_2.add(a)

        axes_2.arrange_in_grid(n_rows=7, n_cols=7, buff=0.2)
        axes_2.move_to([10, 0, 0])
        

        all_pts_2=VGroup()
        for i, neuron_idx_1 in enumerate(neuron_indices):
            for k, neuron_idx_2 in enumerate(neuron_indices):
                dese_pts=VGroup()
                neuron_average_x=activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx_1].mean()
                neuron_average_y=activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx_2].mean()
                neuron_max_x=np.max(np.abs(activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx_1]-neuron_average_x))*1.0 #Might want to bring down
                neuron_max_y=np.max(np.abs(activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx_2]-neuron_average_y))*1.0 #Might want to bring down
                for j in range(p):
                    x = (activations['blocks.0.mlp.hook_pre'][j, 2, neuron_idx_2]-neuron_average_y)/neuron_max_y
                    y = (activations['blocks.0.mlp.hook_pre'][j, 2, neuron_idx_1]-neuron_average_x)/neuron_max_x
                    pt = Dot(axes_2[len(neuron_indices)*i+k].c2p(x, y), radius=0.02, stroke_width=0)
                    pt.set_color(viridis_hex(j, 0, p))
                    dese_pts.add(pt)
                all_pts_2.add(dese_pts)

        self.wait()

        self.play(self.frame.animate.reorient(0, 0, 0, (7.62, 0.06, 0.0), 7.58),
                  Write(axes_2), 
                  run_time=4) #Pan over 
        self.wait()

        # self.add(all_pts_2[2])

        #P30 moves
        self.play(self.frame.animate.reorient(0, 0, 0, (6.27, 2.27, 0.0), 3.75), run_time=4)
        self.wait()
        self.play(ReplacementTransform(all_pts[0].copy(), all_pts_2[1]), 
                  ReplacementTransform(all_pts[1].copy(), all_pts_2[1]), lag_ratio=0.1, run_time=10)
        self.wait()

        #Hmm ok kinda thinkign about just fading in the rest actually? 
        #Would be cool to bring all the points over, but given the timing and vibe of 
        #this exploratory part of the script - feels overwhelming. 
        self.play(self.frame.animate.reorient(0, 0, 0, (7.62, 0.06, 0.0), 7.58), 
                  FadeIn(all_pts_2[0]), 
                  FadeIn(all_pts_2[2:]),
                  run_time=4) #Zoom back out

        self.wait()
        # self.add(all_pts_2)
        # self.remove(all_pts_2)

        #P31 - Ok, now roll back the clock on all these plots -> let's go! 500 or 1000 is probably good, let's target 500. 
        activation_history=np.load(str(data_dir/'time_history_mlp_hook_pre_1.npy'))
        #activation_history.shape (2000, 113, 3, 512)
        

        for step_count in tqdm(np.arange(999, -1, -1)):
            all_pts_old=all_pts
            all_pts_2_old=all_pts_2

            all_pts=VGroup()
            for i, neuron_idx in enumerate(neuron_indices):
                neuron_average=activation_history[step_count, :p, 2, neuron_idx].mean()
                neuron_max=np.max(np.abs(activation_history[step_count, :p, 2, neuron_idx]-neuron_average))*1.0 
                dese_pts=VGroup()
                for j in range(p):
                    x = j / p
                    y = (activation_history[step_count, j, 2, neuron_idx]-neuron_average)/neuron_max
                    pt = Dot(axes[i].c2p(x, y), radius=0.02, color=FRESH_TAN, stroke_width=0)
                    pt.set_color(viridis_hex(j, 0, p))
                    dese_pts.add(pt)
                all_pts.add(dese_pts)


            all_pts_2=VGroup()
            for i, neuron_idx_1 in enumerate(neuron_indices):
                for k, neuron_idx_2 in enumerate(neuron_indices):
                    dese_pts=VGroup()
                    neuron_average_x=activation_history[step_count, :p, 2, neuron_idx_1].mean()
                    neuron_average_y=activation_history[step_count, :p, 2, neuron_idx_2].mean()
                    neuron_max_x=np.max(np.abs(activation_history[step_count, :p, 2, neuron_idx_1]-neuron_average_x))*1.0 #Might want to bring down
                    neuron_max_y=np.max(np.abs(activation_history[step_count, :p, 2, neuron_idx_2]-neuron_average_y))*1.0 #Might want to bring down
                    for j in range(p):
                        x = (activation_history[step_count, j, 2, neuron_idx_2]-neuron_average_y)/neuron_max_y
                        y = (activation_history[step_count, j, 2, neuron_idx_1]-neuron_average_x)/neuron_max_x
                        pt = Dot(axes_2[len(neuron_indices)*i+k].c2p(x, y), radius=0.02, stroke_width=0)
                        pt.set_color(viridis_hex(j, 0, p))
                        dese_pts.add(pt)
                    all_pts_2.add(dese_pts)
            self.remove(all_pts_old, all_pts_2_old)
            self.add(all_pts, all_pts_2)
            self.wait(0.2)


        self.wait(20)
        self.embed()

            
            




















            # self.add(, all_pts)


        #Shoudl loopify this
        # neuron_1_pts = [axis_1.c2p(i/len(probe_1), probe_1[i,0]) for i in range(len(probe_1))]
        # probe_1a_line = VMobject(stroke_width=3)
        # probe_1a_line.set_points_smoothly(probe_1a_pts)
        # probe_1a_line.set_color(CYAN)



        #     # Small left arrow
        #     arr = Arrow(
        #         start=LEFT, end=RIGHT,
        #         buff=0.05,            # small spacing
        #         stroke_width=1.4,
        #         max_tip_length_to_length_ratio=0.3
        #     ).scale(0.2)

        #     # Put arrow to the left of the text
        #     row = VGroup(arr, txt)
        #     row.arrange(RIGHT, buff=0.1)

        #     activation_rows.add(row)

        # # Arrange rows vertically
        # activation_rows.arrange(DOWN, buff=0.128)

        # # Coloring and positioning
        # activation_rows.set_color(FRESH_TAN)
        # activation_rows.move_to([3.5, 0.73, 0])

        # self.add(activation_rows)

        # activation_texts = VGroup()
        # for i in range(7):
        #     val = round(activations['blocks.0.mlp.hook_pre'][example_index, 2, i], 2)
        #     activation_texts.add(Tex(str(val), font_size=24))

        # activation_texts.arrange(DOWN, buff=0.128)
        # activation_texts.set_color(FRESH_TAN)
        # activation_texts.move_to([3.5, 0.73, 0])
        # self.add(activation_texts)


        # Tex('')





