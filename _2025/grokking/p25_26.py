from welch_paths import welch_path
import manimgx as gx
from activation_diagrams import black_to_tan_hex, draw_attention_patterns, draw_attention_values, draw_embeddings, draw_inputs, draw_logits, draw_mlp_1, draw_mlp_2, draw_mlp_3, softmax_with_temperature, viridis_hex
import numpy as np
import pickle
from welch_scene import WelchScene

from functools import partial

from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as colors

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

svg_dir=Path(welch_path('grokking/graphics/to_manim'))
data_dir=Path(welch_path('grokking/from_linux/grok_1764602090'))


class P25_26(WelchScene):
    def construct(self):  


        svg_files=list(sorted(svg_dir.glob('*network_to_manim*')))

        with open(data_dir/'final_model_activations_sample.p', 'rb') as f:
            activations = pickle.load(f)

        all_svgs=gx.Group()
        for svg_file in svg_files[1:16]: #Expand if I add more artboards
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


        self.wait()
        self.set_view(0, 0, 0, (1.04, -0.06, 0.0), 5.63)
        self.play(*[gx.Write(all_svgs[i]) for i in [5, 6, 7, 8, 9, 12, 14]], run_time=7)
        
        self.wait()

        #Our data is fed into our model as...draw while mooving camera?

        self.play(self.animate_view(0, 0, 0, (-1.05, -0.07, 0.0), 6.12),
                  *[gx.Write(all_svgs[i]) for i in [1, 2]],
                 lag_ratio=0.5,
                 run_time=5) #not quite what I want but pretty close
        self.wait()

        # Now add some arrows/backet in illustrator in editing
        # Now add "one plus two" one step at a time
        example_index=115


        draw_inputs(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.2)
        self.wait()

        draw_embeddings(self, activations, all_svgs, reset=False, example_index=example_index, wait=0, colormap=black_to_tan_hex)
        self.wait()

        #Start paragrpah 26
        self.wait()
        self.play(gx.Write(all_svgs[0][:5]), gx.Write(all_svgs[0][14]))
        self.play(gx.FadeIn(all_svgs[3]), gx.FadeIn(all_svgs[4]))
        self.wait()


        self.play(gx.Write(all_svgs[0][5]))
        # self.wait()

        draw_attention_values(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)

        self.add(all_svgs[13])
        draw_attention_patterns(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)

        self.play(gx.Write(all_svgs[0][6]))
        draw_mlp_1(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_mlp_2(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        draw_mlp_3(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.1, colormap=black_to_tan_hex)
        self.wait()


        self.play(gx.Write(all_svgs[0][7:14]), gx.Write(all_svgs[0][15:]),
                 self.animate_view(0, 0, 0, (0.02, -0.06, 0.0), 7.58),
                 run_time=2)
        self.wait()


        #Add length 128 bracket in illustrator
        draw_logits(self, activations, all_svgs, reset=False, example_index=example_index, wait=0.0, colormap=black_to_tan_hex, temperature=25.0)


        
        self.play(gx.FadeIn(all_svgs[10]), gx.FadeIn(all_svgs[11]), run_time=2.0)



        self.wait(20)
        # Scene authoring stops here in the interactive editor.
