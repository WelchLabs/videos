from welch_paths import welch_path
import manimgx as gx
from deepseek_diagrams import get_attention_head
from welch_scene import WelchScene

from tqdm import tqdm
from pathlib import Path


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'


class poster_2(WelchScene):
    def construct(self):

        img_path=Path(welch_path('deepseek/hackin/linux_workdir/deepseek'))
        svg_path=Path(welch_path('deepseek/graphics/to_manim'))


        # a=get_attention_head(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1/0', 
        #                             svg_file='mha_2d_segments-')
        # self.add(a)
        # self.wait()

        # x=get_input_x(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # self.add(x)

        # self.set_view(0, 0, 0, (0.00, 0.00, 0.00), 2.80)
        # self.wait()

        # self.set_view(0, 86, 0, (-0.07, -0.06, 0.06), 2.39)

        # x=get_input_x(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # x.move_to([-1.85,0,0]) #Shift input data to the left
        # x.rotate([PI/2,0,0], axis=RIGHT)
        # self.add(x)        

        # self.remove(x)

        attention_heads=gx.Group()
        spacing=0.25
        for i in range(12): #Render in reverse order for occlusions
            a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                        img_path=img_path/'gpt_2_attention_viz_1'/str(i))
            a.rotate(gx.PI/2, axis=gx.RIGHT)
            a.move_to([0, spacing*i,0])
            # a.set_opacity(0.5)
            attention_heads.add(a)
        
        # attention_heads.set_opacity(0.9)

        for i in range(11, 0, -1):
            self.add(attention_heads[i][1][13].set_opacity(1.0)) #Arrows on right side
            # self.add(attention_heads[i][1][0].set_opacity(0.9)) #Outlines

            self.add(attention_heads[i][0][0].set_opacity(1.0)) #queries
            self.add(attention_heads[i][0][1].set_opacity(1.0)) #keys
            self.add(attention_heads[i][0][2].set_opacity(1.0)) #values

            ##should replace kt with same matrix in each head ->
            self.add(attention_heads[i][0][3:].set_opacity(1.0)) #attention patterns and output
            # self.add(attention_heads[i][2]) #Thick white lines

        # -- Now do first layer a little differently
        # self.add(attention_heads[0][0][0].set_opacity(0.4)) #Queries
        self.add(attention_heads[0][0][3:].set_opacity(1.0)) #attention patterns and output.
        self.add(attention_heads[0][1][13].set_opacity(1.0)) #Arrows on right side
        self.add(attention_heads[0][0][0].set_opacity(1.0)) #queries
        self.add(attention_heads[0][0][1].set_opacity(1.0)) #keys
        self.add(attention_heads[0][0][2].set_opacity(1.0)) #values
        self.add(attention_heads[0][1][3].set_opacity(1.0)) #Query Label
        self.add(attention_heads[0][1][4].set_opacity(1.0)) #Key Label
        self.add(attention_heads[0][1][5].set_opacity(1.0)) #Vakye Label

        #Opationally add labels/variable names on front head
        self.add(attention_heads[0][1][14].set_opacity(1.0)) #Yea I'm 50/50 on this - nice to have the option i think
        # self.add(attention_heads[0][2]) #Thick white lines

        ## -- So, I think a really good option would be showing the keys and values collapsing down into a single matrix
        ## -- and then add in the connector. 
        self.set_view(0, 82, 0, (0.45, -0.07, -0.02), 1.43)
        self.wait()
        # self.set_view(0, 83, 0, (0.37, -0.06, 0.01), 2.04)
        self.play(self.animate_view(0, 83, 0, (0.37, -0.06, 0.01), 2.04), run_time=3)
        self.wait()

        #Option with camera move - ok yeah this looks nice. 
        self.play(*[attention_heads[i][0][0].animate.set_opacity(0.0) for i in range(12)]+
	       [attention_heads[0][1][3].animate.set_opacity(0.0)]+
 		   [attention_heads[i][0][1].animate.move_to([-0.6 ,  1.391,  -0.08 ]) for i in range(12)]+ #Keys [-0.53 ,  1.391,  0.06 ]
 		   [attention_heads[0][1][4].animate.move_to([-0.6 ,  1.391,  -0.24 ])]+                   #[-0.53 ,  1.391,  -0.10 ]
 		   [attention_heads[i][0][2].animate.move_to([-0.6 ,  1.391, -0.48]) for i in range(12)]+ #Values [-0.53 ,  1.391, -0.34]
 		   [attention_heads[0][1][5].animate.move_to([-0.6 ,  1.391,  -0.65 ])]+                  #[-0.53 ,  1.391,  -0.51 ]
                   [self.animate_view(-53, 71, 0, (-0.36, 0.7, 0.07), 1.81)],
	       run_time=5)

        #Let's fade in connectors here. 

        connector_1=gx.SVGMobject(welch_path('deepseek/graphics/to_manim/thick_white_connector_1.svg'))
        connector_1.scale([1.0, 1.39, 1])
        connector_1.move_to([0.2, 1.389, -0.11]) #0.05 #[-0.12, 1.375, -0.08]
        # (np.array([0.2, 1.386, 0.05])-np.array([-0.12, 1.375, -0.08]))+np.array([-0.85, 1.38, -0.07])
        # self.add(connector_1) 

        connector_1b=gx.SVGMobject(welch_path('deepseek/graphics/to_manim/thick_white_connector_1.svg'))
        connector_1b.scale([1.0, 1.39, 1])
        connector_1b.move_to([ 0.14 ,  1.386, -0.499 ])
        # (np.array([0.2, 1.386, 0.05])-np.array([-0.12, 1.375, -0.08]))+np.array([-0.18, 1.375, -0.47])
        # self.add(connector_1b)

        # for i in range(12): self.add(attention_heads[i][2])
        white_arrows=gx.Group(*[attention_heads[i][2] for i in range(12)])

        self.play(gx.FadeIn(connector_1b), gx.FadeIn(connector_1), gx.FadeIn(white_arrows),
                         self.animate_view(-34, 69, 0, (0.04, 0.58, 0.09), 2.10), run_time=3)
        self.add(attention_heads[0][0][1]) #Occlusion
        self.add(attention_heads[0][0][2]) #Occlusion
        self.wait()

        #Can we get away with a little ambient rotation?
        # self.play(self.animate_view(-20, 66, 0, (0.19, 0.44, -0.01), 1.96), run_time=8, rate_func=linear)



        self.set_view(-38, 59, 0, (0.12, 0.93, -0.18), 2.93)
        


        self.wait(20)
        # Scene authoring stops here in the interactive editor.
