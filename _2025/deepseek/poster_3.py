from welch_paths import welch_path
import manimgx as gx
from deepseek_diagrams import get_attention_head
from welch_scene import WelchScene

from tqdm import tqdm
from pathlib import Path


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'


class poster_3(WelchScene):
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
        # self.wait()
        self.set_view(0, 82, 0, (0.45, -0.07, -0.02), 1.43)
        self.wait()
        # self.set_view(0, 83, 0, (0.37, -0.06, 0.01), 2.04)
        self.play(self.animate_view(0, 83, 0, (0.37, -0.06, 0.01), 2.04), run_time=3)
        self.wait()


        self.play(*[self.animate_view(-51, 66, 0, (-0.37, 0.6, -0.12), 2.37)]+
                [attention_heads[i][0][0].animate.set_opacity(0.0) for i in range(12)]+ #Turn down queries
                [attention_heads[0][1][3].animate.set_opacity(0.0)]+ #Query Labels
                [attention_heads[i][0][1].animate.move_to([-0.53,  0.39,  -0.1  ]) for i in range(4)]+ #First key grouping
                [attention_heads[i][0][1].animate.move_to([-0.53 ,  1.391,  -0.1 ]) for i in range(4,8)]+
                [attention_heads[i][0][1].animate.move_to([-0.53,  2.38,  -0.1  ]) for i in range(8,12)]+
                [attention_heads[0][1][4].animate.move_to([-0.53 ,  0.38,  -0.26 ])]+    #Move key labels 
                [attention_heads[i][0][2].animate.move_to([-0.53 ,  0.39,  -0.5 ]) for i in range(4)]+ #First value grouping
                [attention_heads[i][0][2].animate.move_to([-0.53 ,  1.38, -0.5 ]) for i in range(4,8)]+
                [attention_heads[i][0][2].animate.move_to([-0.53 ,  2.39, -0.5]) for i in range(8,12)]+
                [attention_heads[0][1][5].animate.move_to([-0.53 ,  0.38,  -0.66 ])], #Move value labels.
                run_time=5) 
        self.wait()


        ##Keys
        connector_1=gx.SVGMobject(welch_path('deepseek/graphics/to_manim/thick_small_white_connector.svg'))
        connector_1.scale([0.3, 0.39, 1])
        connector_1.move_to([0.18, 1.39, -0.11]) #[0.18, 1.38, 0.05]
        
        connector_1b=connector_1.copy()
        connector_1b.move_to([0.18, 2.39, -0.11]) #0.05

        connector_1c=connector_1.copy()
        connector_1c.move_to([0.18, 0.385, -0.11])

        ##Values
        connector_2=gx.SVGMobject(welch_path('deepseek/graphics/to_manim/thick_small_white_connector.svg'))
        connector_2.scale([0.3, 0.39, 1])
        connector_2.move_to([ 0.18 ,  1.38, -0.5 ])

        connector_2b=connector_2.copy()
        connector_2b.move_to([0.18, 2.39, -0.5])

        connector_2c=connector_2.copy()
        connector_2c.move_to([ 0.18 ,  0.39, -0.5 ])     

        white_arrows=gx.Group(*[attention_heads[i][2] for i in range(12)])

        self.play(gx.FadeIn(connector_1b), gx.FadeIn(connector_1), gx.FadeIn(white_arrows),
                  gx.FadeIn(connector_1c), gx.FadeIn(connector_2), gx.FadeIn(connector_2b),  gx.FadeIn(connector_2c),
                 self.animate_view(-38, 59, 0, (0.05, 0.56, 0.06), 2.54), run_time=2)
        self.add(attention_heads[4][0][1]) #Occlusion
        self.add(attention_heads[4][0][2]) #Occlusion
        self.add(attention_heads[0][0][1]) #Occlusion
        self.add(attention_heads[0][0][2]) #Occlusion
        self.wait()         

        self.set_view(-38, 59, 0, (0.12, 0.93, -0.18), 2.93)


        # self.play(self.animate_view(-18, 58, 0, (0.29, 0.47, 0.04), 2.54), run_time=8, rate_func=linear)
        self.wait()





        


        self.wait(20)
        # Scene authoring stops here in the interactive editor.
