from welch_paths import welch_path
import manimgx as gx
from activation_diagrams import black_to_tan_hex, draw_attention_patterns, draw_attention_values, draw_embeddings, draw_embeddings_2, draw_inputs, draw_logits, draw_mlp_1, draw_mlp_2, draw_mlp_3, softmax_with_temperature, viridis_hex
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


class P34_38(WelchScene):
    def construct(self): 

        #Pick up (mostly) where we left off on 31. 

        p=113

        svg_files=list(sorted(svg_dir.glob('*network_to_manim*')))

        with open(data_dir/'final_model_activations_sample.p', 'rb') as f:
            activations = pickle.load(f)

        all_svgs=gx.Group()
        for svg_file in svg_files[1:20]: #Expand if I add more artboards
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
        example_index=0
        self.set_view(0, 0, 0, (0, 0, 0), 8.0)

        draw_inputs(self, activations, all_svgs, reset=False, example_index=example_index, wait=0)
        draw_embeddings(self, activations, all_svgs, reset=False, example_index=example_index, wait=0, colormap=black_to_tan_hex)
        draw_attention_values(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_attention_patterns(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_mlp_1(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_mlp_2(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_mlp_3(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_logits(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex, temperature=25.0)
        

        self.add(all_svgs[:15], all_svgs[16])
        self.remove(all_svgs[7]); self.add(all_svgs[7])


        # Ok this should land us exactly wehre we left off. 
        # Now we can go analyze the embedding layer. 
        # Ok first column of embedding layer

        # self.remove(all_svgs[5:15])
        # self.remove(all_svgs[0][15:])
        # self.remove(all_svgs[0][5:14])
        # self.remove(all_svgs[4][25:])
        # self.remove(all_svgs[3])

        self.wait()
        self.play(self.animate_view(0, 0, 0, (-1.23, 0.07, 0.0), 5.51),
                  gx.FadeOut(all_svgs[5:15]),
                  gx.FadeOut(all_svgs[0][15:]),
                  gx.FadeOut(all_svgs[0][5:14]),
                  gx.FadeOut(all_svgs[4][25:]),
                  gx.FadeOut(all_svgs[3]),
                  run_time=5)

        #(113, 113, 3, 128)
        embedding_acts=np.load(data_dir/'hook_resid_pre.npy')
        
        axis_1 = diagram_axes(
            x_range=[0, 1.0, 1],
            y_range=[-1.0, 1.0, 1],
            width=2*2.4,
            height=2*0.8,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2.5,
                "tip_config": {"width":0.02, "length":0.02}
                }
            )

        axis_1.move_to([0, 0.58, 0])

        arrow_1 = diagram_arrow(
                start=gx.LEFT, end=gx.RIGHT,
                buff=0.05,            # small spacing
                stroke_width=1.0,
                max_tip_length_to_length_ratio=0.3
            ).scale(0.2)
        arrow_1.set_color(CHILL_BROWN)
        arrow_1.next_to(axis_1, gx.LEFT, buff=0.1)

        axis_1_x_label=math_tex('x', font_size=24)
        axis_1_x_label.set_color(CHILL_BROWN)
        axis_1_x_label.next_to(axis_1, gx.RIGHT, buff=0.1)
        axis_1_x_label.shift([0, -0.1, 0])
        

        self.play(gx.Write(axis_1), gx.Write(arrow_1), run_time=3)
        self.add(axis_1_x_label)
        self.wait()

        pts_1=gx.VGroup()
        neuron_idx=3
        neuron_average=embedding_acts[:, 0, 0, neuron_idx].mean()
        neuron_max=np.max(np.abs(embedding_acts[:, 0, 0, neuron_idx]-neuron_average))*1.0 #Might want to bring down
        for j in range(p):
            x = j / p
            y = (embedding_acts[j, 0, 0, neuron_idx]-neuron_average)/neuron_max
            pt = gx.Dot(axis_1.c2p(x, y), radius=0.02, color=FRESH_TAN, stroke_width=0)
            pt.set_color(FRESH_TAN)
            pts_1.add(pt)


    
        # I didn't cache the activations for sweeping x -> quickly hack it here 
        # Might need it later though -> i guess I could load up the full activations
        # Ok we'll see, one problem at a time. 
        activations_input_flipped={}
        activations_input_flipped['x']=np.zeros((565, 3))
        for i in range(p):
            activations_input_flipped['x'][i][0]=i
            activations_input_flipped['x'][i][1]=0
            activations_input_flipped['x'][i][2]=113

        self.wait()
        for i in range(p):
            draw_inputs(self, activations_input_flipped, all_svgs, reset=False, example_index=i, wait=0)
            draw_embeddings_2(self, activations, all_svgs, reset=False, example_index=i, wait=0, colormap=black_to_tan_hex)
            self.add(pts_1[i])
            self.wait(0.2)
        self.wait()

        xs=np.linspace(0, 1, p)
        ys=np.cos(np.arange(0, p)*8*np.pi/p-2.6)

        pts_curve_1 = [axis_1.c2p(xs[j], ys[j]) for j in range(p)]
        curve_1 = gx.VMobject(stroke_width=3)
        curve_1.set_points_smoothly(pts_curve_1)
        curve_1.set_color(YELLOW)

        wave_label_1 = math_tex(r'A_1 \cos \big( \frac{8 \pi }{113} x + \phi_1 \big)')
        wave_label_1.set_color(YELLOW)
        wave_label_1.scale(0.4)
        wave_label_1.move_to([-0.1, 1.7, 0])

        self.wait()
        self.play(gx.Create(curve_1), run_time=5)
        self.add(wave_label_1)
        self.wait()


        #P35 Hmm might be cool to contruct a "high frequency" curve that matches the data
        # and then replacement transform to it?
        pts_curve_2=[]
        for j in range(p):
            x = j / p
            y = (embedding_acts[j, 0, 0, neuron_idx]-neuron_average)/neuron_max
            pts_curve_2.append(axis_1.c2p(x, y))
        curve_2 = gx.VMobject(stroke_width=3)
        curve_2.set_points_smoothly(pts_curve_2)
        curve_2.set_color(YELLOW)

        self.play(gx.ReplacementTransform(curve_1, curve_2),
                  gx.FadeOut(wave_label_1), run_time=5)
        self.wait()

        # P36 - ok here we go. 
        # SO let me create 8 plots here, then I can do some transform action
        # Will also want to bring in some nice arrows again from ilustrator I think

        axes=gx.VGroup()
        for i in range(8):
            a = diagram_axes(
                x_range=[0, 1.0, 1],
                y_range=[-1.0, 1.0, 1],
                width=2.4,
                height=0.56,
                axis_config={
                    "color": CHILL_BROWN,
                    "include_ticks": False,
                    "include_numbers": False,
                    "include_tip": True,
                    "stroke_width":1.8,
                    "tip_config": {"width":0.02, "length":0.02}
                    }
                )
            axes.add(a)
        axes.arrange(gx.DOWN, buff=0.1)
        axes.move_to([-1., 0, 0])

        all_pts=gx.VGroup()
        neuron_indices=[ 3,  22,  70,  77,  97, 109, 110, 120]
        for i, neuron_idx in enumerate(neuron_indices):
            neuron_average=embedding_acts[:, 0, 0, neuron_idx].mean()
            neuron_max=np.max(np.abs(embedding_acts[:, 0, 0, neuron_idx]-neuron_average))*1.0 #Might want to bring down
            dese_pts=gx.VGroup()
            for j in range(p):
                x = j / p
                y = (embedding_acts[j, 0, 0, neuron_idx]-neuron_average)/neuron_max
                pt = gx.Dot(axes[i].c2p(x, y), radius=0.02, color=FRESH_TAN, stroke_width=0)
                pt.set_color(FRESH_TAN)
                dese_pts.add(pt)
            all_pts.add(dese_pts)

        self.wait()

        self.remove(curve_2, arrow_1, axis_1_x_label)
        self.play(gx.ReplacementTransform(axis_1, axes[0]),
                  gx.ReplacementTransform(pts_1, all_pts[0]),
                  run_time=3)
        self.play(gx.FadeIn(all_svgs[17]), gx.FadeIn(axes[1:]), gx.FadeIn(all_pts[1:]), run_time=3)

        self.wait()

        # Fancy veritasium sytle stacking would be cool here, but kinda think i should keep 
        # moving
        # Also I think load from disk vs compute dynamically. 
        sparse_probe_1=np.load(data_dir/'sparse_probe_1.npy')
        pts_probe_1=gx.VGroup()
        for j in range(p):
            x = j / p
            pt = gx.Dot(axes[0].c2p(x, sparse_probe_1[j]), radius=0.02, color=FRESH_TAN, stroke_width=0)
            pt.set_color(FRESH_TAN)
            pts_probe_1.add(pt)

        self.wait()
        self.play(*[gx.ReplacementTransform(all_pts[i], pts_probe_1) for i in range(8)],
                 gx.FadeOut(all_svgs[17]), gx.FadeOut(axes[1:]),
                 run_time=4)
        self.wait()

        #Now draw on cosine and label
        xs = np.linspace(0, 1.0, p) 
        pts = [axes[0].c2p(xs[j], np.cos(j*8*np.pi/p)) for j in range(p)]

        curve_3 = gx.VMobject(stroke_width=3)
        curve_3.set_points_smoothly(pts)
        curve_3.set_color(YELLOW)
        
        # wave_label_4 = Tex(r'\cos \big( \frac{8 \pi }{113} x \big)')
        wave_label_4 = math_tex(r'\cos \big(\tfrac{8\pi}{113}x\big)')
        wave_label_4.set_color(YELLOW)
        wave_label_4.scale(0.45)
        wave_label_4.move_to([0.9, 2.3, 0])

        self.wait()

        self.play(gx.Create(curve_3), run_time=3)
        self.add(wave_label_4)
        self.wait()

        # Ok actually not too far away from wrapping up p38 -> let's go!
        # So we're mid p36 here, I want to add back in the full network here
        # and move hte linear probe to it's "final" location, then add in the remaining


        probe_1_group=gx.VGroup(axes[0], curve_3)
        # probe_1_group.scale(2.0)
        # probe_1_group.move_to([-4, 3, 0])



        # wave_label_4.scale(1.5)
        # wave_label_4.move_to([-0.85, 3.6, 0])
        # Make logit match what we're currently showing
        draw_logits(self, activations, all_svgs, reset=False, example_index=111, wait=0, colormap=black_to_tan_hex, temperature=25.0)

        self.wait()
        self.remove(pts_probe_1)
        self.play(
                  self.animate_view(0, 0, 0, (0.0, 0.0, 0.0), 8.0),
                  probe_1_group.animate.scale(2.0).move_to([-4, 3, 0]),
                  wave_label_4.animate.scale(1.5).move_to([-0.85, 3.6, 0]),
                  gx.FadeIn(all_svgs[5:15]),
                  gx.FadeIn(all_svgs[0][15:]),
                  gx.FadeIn(all_svgs[0][5:14]),
                  gx.FadeIn(all_svgs[4][25:]),
                  gx.FadeIn(all_svgs[3]),
                  run_time=5)
        
        self.add(all_svgs[:15], all_svgs[16])
        self.remove(all_svgs[7]); self.add(all_svgs[7])
        self.add(all_svgs[18][:21])
        x_label_2=math_tex('x', font_size=24)
        x_label_2.set_color(CHILL_BROWN)
        x_label_2.next_to(axes[0], gx.RIGHT, buff=0.1)
        x_label_2.shift([0, -0.1, 0])
        self.add(x_label_2)
        self.wait()

        #Ok getting close to this milestone!!

        sparse_probe_2=np.load(data_dir/'sparse_probe_2.npy')
        sparse_probe_3=np.load(data_dir/'sparse_probe_3.npy')
        sparse_probe_4=np.load(data_dir/'sparse_probe_4.npy')

        pts_curve_2b=[]
        for j in range(p):
            x = j / p
            y = sparse_probe_2[j]
            pts_curve_2b.append(axes[0].c2p(x, y))
        curve_2b = gx.VMobject(stroke_width=3)
        curve_2b.set_points_smoothly(pts_curve_2b)
        curve_2b.set_color(MAGENTA)

        wave_label_2b = math_tex(r'\sin \big(\tfrac{8\pi}{113}x\big)')
        wave_label_2b.set_color(MAGENTA)
        wave_label_2b.scale(0.45*1.5)
        wave_label_2b.move_to([-0.9, 2.6, 0])

        self.play(gx.Create(curve_2b), run_time=3)
        self.add(wave_label_2b)
        self.wait()


        #Ok sparse probes for y and this scene will be done!

        # axis_2=axes[0].copy()
        axis_2 = diagram_axes(
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
        axis_2.move_to([-4, -2.85, 0])
        y_label=math_tex('y', font_size=24)
        y_label.set_color(CHILL_BROWN)
        y_label.next_to(axis_2, gx.RIGHT, buff=0.1)
        y_label.shift([0, -0.1, 0])


        pts_curve_3=[]
        for j in range(p):
            x = j / p
            y = sparse_probe_3[j]
            pts_curve_3.append(axis_2.c2p(x, y))
        curve_3 = gx.VMobject(stroke_width=3)
        curve_3.set_points_smoothly(pts_curve_3)
        curve_3.set_color(CYAN)

        wave_label_3 = math_tex(r'\cos \big(\tfrac{8\pi}{113}y\big)')
        wave_label_3.set_color(CYAN)
        wave_label_3.scale(0.45*1.5)
        wave_label_3.move_to([-0.9, -2.2, 0])


        pts_curve_4=[]
        for j in range(p):
            x = j / p
            y = sparse_probe_4[j]
            pts_curve_4.append(axis_2.c2p(x, y))
        curve_4 = gx.VMobject(stroke_width=3)
        curve_4.set_points_smoothly(pts_curve_4)
        curve_4.set_color(gx.RED)

        wave_label_4 = math_tex(r'\sin \big(\tfrac{8\pi}{113}y\big)')
        wave_label_4.set_color(gx.RED)
        wave_label_4.scale(0.45*1.5)
        wave_label_4.move_to([-0.9, -3.2, 0])


        self.wait()

        self.play(gx.Write(all_svgs[18][21:]), gx.Create(axis_2), run_time=2.5)
        self.play(gx.Create(curve_3), gx.Create(curve_4), run_time=5.0)
        self.add(wave_label_3, wave_label_4)
        self.wait()
    

        self.wait()



        self.wait(20)
        # Scene authoring stops here in the interactive editor.

class P33_34(WelchScene):
    def construct(self): 

        #Pick up (mostly) where we left off on 31. 

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
        example_index=0
        self.set_view(0, 0, 0, (0.02, -0.06, 0.0), 7.58)

        draw_inputs(self, activations, all_svgs, reset=False, example_index=example_index, wait=0)
        draw_embeddings(self, activations, all_svgs, reset=False, example_index=example_index, wait=0, colormap=black_to_tan_hex)
        draw_attention_values(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_attention_patterns(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_mlp_1(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_mlp_2(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_mlp_3(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_logits(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex, temperature=25.0)

        self.add(all_svgs[:15])
        self.remove(all_svgs[7]); self.add(all_svgs[7])

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
                pt.set_color(viridis_hex(j, 0, p))
                dese_pts.add(pt)
            all_pts.add(dese_pts)

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

        self.set_view(0, 0, 0, (7.62, 0.06, 0.0), 7.58)
        self.remove(p28_fade_group)



        self.add(axes, axes_2, all_pts, all_pts_2)
        self.add(all_svgs[15])
        self.wait()

        self.play(self.animate_view(0, 0, 0, (6.47, 1.97, 0.0), 3.63),
                  gx.FadeOut(all_pts_2), gx.FadeOut(axes_2), run_time=5)
        self.wait()



        svg_files=list(sorted(svg_dir.glob('*fft_plots*')))
        fft_svgs=gx.Group()
        for svg_file in svg_files[1:]: #Expand if I add more artboards
            svg_image=gx.SVGMobject(str(svg_file))
            fft_svgs.add(svg_image[1:]) #Throwout background

        fft_svgs.scale(1.8) 
        fft_svgs.move_to([7.7, 1.96, 0])

        self.wait()
        self.play(gx.Write(fft_svgs[0]))
        self.play(gx.Write(fft_svgs[1]))
        self.wait()

        self.play(gx.Write(fft_svgs[2]))
        self.wait()

        #Now load up numpy arrays so we can plot sine waves!
        fits=[]
        for i in range(3):
            fits.append(np.load(data_dir/'ffts_1'/('fit_'+str(i)+'.npy')))


        # Draw smooth YELLOW curves on the first 3 axes
        curves = gx.VGroup()
        N = len(fits[0])
        xs = np.linspace(0, 1.0, N)  # x in [0, 1]

        for i in range(3):  # first 3 axes
            y_vals = fits[i]
            if i==1: y_vals=y_vals/3.0 #No idea why I need this, going to keep moving for now. 

            ##normalize if needed to keep in [-1, 1]
            # y_mean = y_vals.mean()
            # y_max = np.max(np.abs(y_vals - y_mean))
            # y_vals = (y_vals - y_mean) / (y_max + 1e-8)

            pts = [axes[i].c2p(xs[j], y_vals[j]) for j in range(N)]

            curve = gx.VMobject(stroke_width=3)
            curve.set_points_smoothly(pts)
            curve.set_color(YELLOW)
            curves.add(curve)

        wave_label_1 = math_tex(r'A_1 \cos \big( \frac{8 \pi }{113} x + \phi_1 \big)')
        wave_label_1.set_color(YELLOW)
        wave_label_1.scale(0.28)
        wave_label_1.move_to([4.8, 3.45, 0])

        wave_label_2 = math_tex(r'A_2 \cos \big( \frac{8 \pi }{113} x + \phi_2 \big)')
        wave_label_2.set_color(YELLOW)
        wave_label_2.scale(0.28)
        wave_label_2.move_to([4.8, 2.4, 0])

        wave_label_3 = math_tex(r'A_3 \cos \big( \frac{6 \pi }{113} x + \phi_3 \big)')
        wave_label_3.set_color(YELLOW)
        wave_label_3.scale(0.28)
        wave_label_3.move_to([4.8, 1.45, 0])

        self.wait()
        self.play(gx.Create(curves), lag_ratio=0.8, run_time=4)
        self.add(wave_label_1, wave_label_2, wave_label_3) #Simple add here I think!
        

        # self.remove(wave_label_1, wave_label_2, wave_label_3)

        # self.add(curves)
        # self.remove(curves)

        # self.add(fft_svgs)
        # self.remove(fft_svgs)


        # self.set_view(0, 0, 0, (6.47, 1.97, 0.0), 3.63)

        #Ok now we need a nice transition to a "clean model", then smooth cut to new scene!
        self.wait()
        self.play(self.animate_view(0, 0, 0, (0, 0, 0), 8.0),
                 gx.FadeOut(wave_label_1),
                 gx.FadeOut(wave_label_2),
                 gx.FadeOut(wave_label_3),
                 gx.FadeOut(axes),
                 gx.FadeOut(all_svgs[15]),
                 gx.FadeOut(all_pts),
                 gx.FadeOut(fft_svgs[0]),
                 gx.FadeOut(fft_svgs[1]),
                 gx.FadeOut(curves),
                 gx.FadeOut(fft_svgs[2]),
                 gx.FadeIn(p28_fade_group),
                 run_time=6)

        
        # self.add(p28_fade_group); 
        self.remove(all_svgs[9]); self.add(all_svgs[9])
        self.remove(all_svgs[7]); self.add(all_svgs[7]) #Occlusions bro
        self.wait()

        self.wait(20)
        # Scene authoring stops here in the interactive editor.
