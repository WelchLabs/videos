from welch_paths import welch_path
import manimgx as gx
from activation_diagrams import black_to_tan_hex, draw_attention_patterns, draw_attention_values, draw_embeddings, draw_inputs, draw_logits, draw_mlp_1, draw_mlp_2, draw_mlp_3, softmax_with_temperature, viridis_hex
import numpy as np
from welch_geometry import diagram_arrow, diagram_axes
import pickle
from welch_scene import WelchScene
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

# svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/grokking/graphics/to_manim')
# data_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/grokking/from_linux/grok_1764706121')

svg_dir=Path(welch_path('grokking/graphics/to_manim'))
data_dir=Path(welch_path('grokking/from_linux/grok_1764706121'))

class P28_31h(WelchScene):
    def construct(self): 

        p=113

        svg_files=list(sorted(svg_dir.glob('*network_to_manim*')))

        with open(data_dir/'final_model_activations_sample.p', 'rb') as f:
            activations = pickle.load(f)

        all_svgs=gx.Group()
        for svg_file in svg_files[1:17]: #Expand if I add more artboards
            svg_image=gx.SVGMobject(str(svg_file))
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
        example_index=115
        self.set_view(0, 0, 0, (0.02, -0.06, 0.0), 7.58)

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

        p28_fade_group=gx.Group(all_svgs[14][9],
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
        self.play(gx.FadeOut(p28_fade_group), run_time=3.0)
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

        activation_txt = gx.VGroup()
        activation_arrows = gx.VGroup()

        for i in range(7):
            val = activations['blocks.0.mlp.hook_pre'][example_index, 2, i]
            txt = math_tex(f"{val:.2f}", font_size=24)
            activation_txt.add(txt)
        activation_txt.set_color(FRESH_TAN)
        activation_txt.arrange(gx.DOWN, buff=0.128)
        activation_txt.move_to([3.6, 0.73, 0])

        for i in range(7):
            arr = diagram_arrow(
                start=gx.LEFT, end=gx.RIGHT,
                buff=0.05,            # small spacing
                stroke_width=1.4,
                max_tip_length_to_length_ratio=0.3
            ).scale(0.2)
            activation_arrows.add(arr)
        activation_arrows.set_color(CHILL_BROWN)
        activation_arrows.arrange(gx.DOWN, buff=0.16)
        activation_arrows.move_to([3.0, 0.73, 0])

        self.wait()
        self.play(gx.FadeIn(activation_arrows), gx.Write(activation_txt), lag_ratio=0.5)

        # Ok deeper into p30 here. 
        # Need 5-7 little plots 
        # And will probably want to zoom out a bit when I make these axes. 
        # The importantly I want to sweep through inptu values as I add points
        # Probably add some nice curvy arrows to each axis in illustrator
        # And I think use manim axes

        self.play(gx.FadeOut(activation_arrows), gx.FadeOut(activation_txt))
        self.wait()

        axes=gx.VGroup()
        for i in range(7):
            a = diagram_axes(
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
        axes.arrange(gx.DOWN, buff=0.2)
        axes.move_to([4.8, 0, 0])

        
        #Hmm probably makes sense to remove mean here and from scatter plots - not a big deal.

        all_pts=gx.VGroup()
        neuron_indices=[0, 3, 4, 2, 11, 5, 9]
        for i, neuron_idx in enumerate(neuron_indices):
            neuron_average=activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx].mean()
            neuron_max=np.max(np.abs(activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx]-neuron_average))*1.0 #Might want to bring down
            dese_pts=gx.VGroup()
            for j in range(p):
                x = j / p
                y = (activations['blocks.0.mlp.hook_pre'][j, 2, neuron_idx]-neuron_average)/neuron_max
                pt = gx.Dot(axes[i].c2p(x, y), radius=0.02, color=FRESH_TAN, stroke_width=0)
                dese_pts.add(pt)
            all_pts.add(dese_pts)


        self.wait()
        self.play(gx.Write(all_svgs[15]), gx.Write(axes), run_time=4)
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

        axes_2 = gx.VGroup()
        for i in range(49):  # 7 × 7
            a = diagram_axes(
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

        axes_2.arrange_in_grid(rows=7, cols=7, buff=0.2)
        axes_2.move_to([10, 0, 0])
        

        all_pts_2=gx.VGroup()
        for i, neuron_idx_1 in enumerate(neuron_indices):
            for k, neuron_idx_2 in enumerate(neuron_indices):
                dese_pts=gx.VGroup()
                neuron_average_x=activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx_1].mean()
                neuron_average_y=activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx_2].mean()
                neuron_max_x=np.max(np.abs(activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx_1]-neuron_average_x))*1.0 #Might want to bring down
                neuron_max_y=np.max(np.abs(activations['blocks.0.mlp.hook_pre'][:p, 2, neuron_idx_2]-neuron_average_y))*1.0 #Might want to bring down
                for j in range(p):
                    x = (activations['blocks.0.mlp.hook_pre'][j, 2, neuron_idx_2]-neuron_average_y)/neuron_max_y
                    y = (activations['blocks.0.mlp.hook_pre'][j, 2, neuron_idx_1]-neuron_average_x)/neuron_max_x
                    pt = gx.Dot(axes_2[len(neuron_indices)*i+k].c2p(x, y), radius=0.02, stroke_width=0)
                    pt.set_color(viridis_hex(j, 0, p))
                    dese_pts.add(pt)
                all_pts_2.add(dese_pts)

        self.wait()

        self.play(self.animate_view(0, 0, 0, (7.62, 0.06, 0.0), 7.58),
                  gx.Write(axes_2),
                  run_time=4) #Pan over 
        self.wait()

        # self.add(all_pts_2[2])

        #P30 moves
        self.play(self.animate_view(0, 0, 0, (6.27, 2.27, 0.0), 3.75), run_time=4)
        self.wait()
        self.play(gx.ReplacementTransform(all_pts[0].copy(), all_pts_2[1]),
                  gx.ReplacementTransform(all_pts[1].copy(), all_pts_2[1]), lag_ratio=0.1, run_time=10)
        self.wait()

        #Hmm ok kinda thinkign about just fading in the rest actually? 
        #Would be cool to bring all the points over, but given the timing and vibe of 
        #this exploratory part of the script - feels overwhelming. 
        self.play(self.animate_view(0, 0, 0, (7.62, 0.06, 0.0), 7.58),
                  gx.FadeIn(all_pts_2[0]),
                  gx.FadeIn(all_pts_2[2:]),
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

            all_pts=gx.VGroup()
            for i, neuron_idx in enumerate(neuron_indices):
                neuron_average=activation_history[step_count, :p, 2, neuron_idx].mean()
                neuron_max=np.max(np.abs(activation_history[step_count, :p, 2, neuron_idx]-neuron_average))*1.0 
                dese_pts=gx.VGroup()
                for j in range(p):
                    x = j / p
                    y = (activation_history[step_count, j, 2, neuron_idx]-neuron_average)/neuron_max
                    pt = gx.Dot(axes[i].c2p(x, y), radius=0.02, color=FRESH_TAN, stroke_width=0)
                    pt.set_color(viridis_hex(j, 0, p))
                    dese_pts.add(pt)
                all_pts.add(dese_pts)


            all_pts_2=gx.VGroup()
            for i, neuron_idx_1 in enumerate(neuron_indices):
                for k, neuron_idx_2 in enumerate(neuron_indices):
                    dese_pts=gx.VGroup()
                    neuron_average_x=activation_history[step_count, :p, 2, neuron_idx_1].mean()
                    neuron_average_y=activation_history[step_count, :p, 2, neuron_idx_2].mean()
                    neuron_max_x=np.max(np.abs(activation_history[step_count, :p, 2, neuron_idx_1]-neuron_average_x))*1.0 #Might want to bring down
                    neuron_max_y=np.max(np.abs(activation_history[step_count, :p, 2, neuron_idx_2]-neuron_average_y))*1.0 #Might want to bring down
                    for j in range(p):
                        x = (activation_history[step_count, j, 2, neuron_idx_2]-neuron_average_y)/neuron_max_y
                        y = (activation_history[step_count, j, 2, neuron_idx_1]-neuron_average_x)/neuron_max_x
                        pt = gx.Dot(axes_2[len(neuron_indices)*i+k].c2p(x, y), radius=0.02, stroke_width=0)
                        pt.set_color(viridis_hex(j, 0, p))
                        dese_pts.add(pt)
                    all_pts_2.add(dese_pts)
            self.remove(all_pts_old, all_pts_2_old)
            self.add(all_pts, all_pts_2)
            self.wait(0.2)


        self.wait(20)
        # Scene authoring stops here in the interactive editor.

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





