from welch_paths import welch_path
import manimgx as gx
from deepseek_diagrams import get_attention_head, get_input_x
import numpy as np
from welch_scene import WelchScene, image_at_height

from tqdm import tqdm
from pathlib import Path


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'


class p23_28(WelchScene):
    def construct(self):

        #Start with previous 9 token input, and propaage 10 input thorugh
        img_path=Path(welch_path('deepseek/hackin/linux_workdir/deepseek'))
        svg_path=Path(welch_path('deepseek/graphics/to_manim'))

        i=0
        a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                    img_path=img_path/'gpt_2_attention_viz_5'/str(i))

        x=get_input_x(svg_path=welch_path('deepseek/graphics/to_manim'),
           svg_file='mha_2d_grouping_test.svg',
           img_path=img_path/'gpt_2_attention_viz_5')

        self.set_view(0, 0, 0, (-0.01, -0.04, 0.0), 2.01)
        self.wait()

        self.add(a[0], a[1][1:13], a[1][17], x[0])

        #Ok, first zoom in and change input text
        self.play(self.animate_view(0, 0, 0, (-1.06, -0.05, 0.0), 0.88), run_time=2)
        a[0][3:].set_opacity(0.2) #Lower opacity of stuff were not using yet. 
        a[1][7:14].set_opacity(0.2)

        i=0
        a2=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments_10_input-',
                                    img_path=img_path/'gpt_2_attention_viz_5'/str(i))

        self.wait()

        # a[1][1][:-10].scale([1, 0.9, 1]).shift(0.025*UP)
        # x[0].scale([1, 0.9, 1]).shift(0.025*UP)

        self.play(a[1][1][:-12].animate.scale([1, 0.9, 1]).shift(0.025*gx.UP),
                x[0].animate.scale([1, 0.9, 1]).shift(0.025*gx.UP))
        blue_word=a2[1][1][-20:-16].scale([1, 0.9, 1])
        self.add(blue_word.shift(0.005*gx.UP)) #"Blue"
        self.wait()

        x1n=image_at_height(str(img_path/'gpt_2_attention_viz_5'/'input_1_1n.png'))
        x1n.scale([0.0095,0.013, 1]) 
        x1n.move_to([-1.4492,-0.24,0])
        # self.add(x1n)

        x2n=image_at_height(str(img_path/'gpt_2_attention_viz_5'/'input_2_1n.png'))
        x2n.scale([0.0089,0.012, 1]) 
        x2n.move_to([-1.224,-0.239,0])
        # self.add(x2n)
        # self.remove(x2n)

        # self.wait()
        self.play(gx.FadeIn(x1n), gx.FadeIn(x2n), gx.FadeOut(a[1][1][-1]), gx.FadeIn(a2[1][1][-2:]))

        ## Now add row to Queries, keys, and values. 
        self.wait()
        self.play(self.animate_view(0, 0, 0, (-0.59, -0.07, 0.0), 1.36), run_time=2)
        self.wait()

        img_path_full=img_path/'gpt_2_attention_viz_5'/str(i)
        q1n=image_at_height(str(img_path_full/'q_1n.png'))
        q1n.scale([0.0127, 0.022, 1]) 
        q1n.move_to([-0.2,0.264,0])
        # self.add(q1n) 

        self.wait()
        self.play(a[0][0].animate.scale([1, 0.92, 1]).shift(0.012*gx.UP))
        self.play(gx.FadeIn(q1n), gx.Transform(a[1][3][-1], a2[1][3][-2:]))
        self.wait()

        img_path_full=img_path/'gpt_2_attention_viz_5'/str(i)
        k1n=image_at_height(str(img_path_full/'k_1n.png'))
        k1n.scale([0.0127, 0.022, 1]) 
        k1n.move_to([-0.2,-0.17,0])
        # self.add(k1n)  
        # self.remove(k1n)       
        
        self.wait()
        self.play(a[0][1].animate.scale([1, 0.92, 1]).shift(0.012*gx.UP))
        self.play(gx.FadeIn(k1n), gx.Transform(a[1][4][-1], a2[1][4][-2:]))
        self.wait()  
        

        img_path_full=img_path/'gpt_2_attention_viz_5'/str(i)
        v1n=image_at_height(str(img_path_full/'v_1n.png'))
        v1n.scale([0.0127, 0.022, 1]) 
        v1n.move_to([-0.2,-0.595,0])
        # self.add(v1n)  
        # self.remove(v1n)       
        
        self.wait()
        self.play(a[0][2].animate.scale([1, 0.92, 1]).shift(0.012*gx.UP))
        self.play(gx.FadeIn(v1n), gx.Transform(a[1][5][-1], a2[1][5][-2:]))
        self.wait()  


        self.play(self.animate_view(0, 0, 0, (0.02, 0.18, 0.0), 0.97), run_time=2)
        self.wait()

        full_k=gx.Group(a[0][1], k1n)
        kt=full_k.copy()
        # kt.scale(np.array([0.0215,0.035, 1])/np.array([0.0415, 0.08, 1])).rotate([0, 0, -PI/2]).rotate(PI,UP).move_to([0.405,0.305,0])
        # self.add(kt)
        # a[1][7].set_opacity(1.0)


        self.play(kt.animate.scale(np.array([0.0215,0.035, 1])/np.array([0.0415, 0.08, 1])).rotate(-gx.PI/2).rotate(gx.PI,gx.UP).move_to([0.405,0.305,0]),
                  run_time=2)
        a[1][7].set_opacity(1.0)
        self.wait()
        
        a2[1][9:13].set_opacity(0.2)
        # a2[1][5].set_opacity(0.2)
        # a2[1][16].set_opacity(0.2)
        a[0][2].set_opacity(0.2)
        v1n.set_opacity(0.2)
        self.add(a2[0][5].set_opacity(0.2)) #add in last two images we'll need later
        self.add(a2[0][6].set_opacity(0.2))
        self.play(gx.FadeIn(a2[0][4]), gx.FadeIn(a2[1][8:13]), gx.FadeOut(a[1][8:13]))
        self.remove(a[0][4])
        self.wait()

        #Let's do these recangles and the following ones in Illustrator/premier/post manim - I think that will be faster
        # r1=Rectangle(width=0.92, height=0.23)
        # r1.move_to([-0.2, 0.393, 0])
        # r1.set_stroke('#FF00FF')
        # self.add(r1)
        # self.wait()


        # r2=Rectangle(width=0.1, height=0.48)
        # r2.move_to([0.4, 0.305, 0])
        # r2.set_stroke('#FF00FF')
        # self.add(r2)
        # self.wait()

        # r3=Rectangle(width=0.185, height=0.185)
        # r3.move_to([0.649, 0.38, 0])
        # r3.set_stroke('#FF00FF')
        # self.add(r3)
        # self.wait()

        self.play(self.animate_view(0, 0, 0, (0.42, 0.2, 0.0), 1.32),
                a2[0][5].animate.set_opacity(1.0),
                a2[1][9:11].animate.set_opacity(1.0),
                run_time=2.0)
        # self.add(a2[0][5])
        # self.set_view(0, 0, 0, (0.27, -0.01, 0.0), 1.50)
        self.wait()

        self.play(self.animate_view(0, 0, 0, (0.31, -0.02, 0.0), 1.49),
                a[0][2].animate.set_opacity(1.0),
                v1n.animate.set_opacity(1.0),
                a2[0][-1].animate.set_opacity(1.0),
                a2[1][11:13].animate.set_opacity(1.0), run_time=2.0)
        self.wait()

        self.play(self.animate_view(0, 0, 0, (0.0, -0.03, 0.0), 2.03), run_time=2.0)
        self.wait()



        self.wait(20)
        # Scene authoring stops here in the interactive editor.

class p29(WelchScene):
    def construct(self):
        '''Dont need a bunch of the opening animations here, just cut them out in editing'''

        #Start with previous 9 token input, and propaage 10 input thorugh
        img_path=Path(welch_path('deepseek/hackin/linux_workdir/deepseek'))
        svg_path=Path(welch_path('deepseek/graphics/to_manim'))

        i=0
        a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                    img_path=img_path/'gpt_2_attention_viz_5'/str(i))

        x=get_input_x(svg_path=welch_path('deepseek/graphics/to_manim'),
           svg_file='mha_2d_grouping_test.svg',
           img_path=img_path/'gpt_2_attention_viz_5')


        self.add(a[0], a[1][1:13], a[1][17], x[0])

        #Ok, first zoom in and change input text
        self.play(self.animate_view(0, 0, 0, (-1.06, -0.05, 0.0), 0.88), run_time=2)
        a[0][3:].set_opacity(0.2) #Lower opacity of stuff were not using yet. 
        a[1][7:14].set_opacity(0.2)

        i=0
        a2=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments_10_input-',
                                    img_path=img_path/'gpt_2_attention_viz_5'/str(i))


        # a[1][1][:-10].scale([1, 0.9, 1]).shift(0.025*UP)
        # x[0].scale([1, 0.9, 1]).shift(0.025*UP)

        self.play(a[1][1][:-12].animate.scale([1, 0.9, 1]).shift(0.025*gx.UP),
                x[0].animate.scale([1, 0.9, 1]).shift(0.025*gx.UP))
        blue_word=a2[1][1][-20:-16].scale([1, 0.9, 1])
        self.add(blue_word.shift(0.005*gx.UP)) #"Blue"
        # self.wait()

        x1n=image_at_height(str(img_path/'gpt_2_attention_viz_5'/'input_1_1n.png'))
        x1n.scale([0.0095,0.013, 1]) 
        x1n.move_to([-1.4492,-0.24,0])
        # self.add(x1n)

        x2n=image_at_height(str(img_path/'gpt_2_attention_viz_5'/'input_2_1n.png'))
        x2n.scale([0.0089,0.012, 1]) 
        x2n.move_to([-1.224,-0.239,0])
        # self.add(x2n)
        # self.remove(x2n)

        # self.wait()
        self.play(gx.FadeIn(x1n), gx.FadeIn(x2n), gx.FadeOut(a[1][1][-1]), gx.FadeIn(a2[1][1][-2:]))

        ## Now add row to Queries, keys, and values. 
        # self.wait()
        # self.play(self.animate_view(0, 0, 0, (-0.59, -0.07, 0.0), 1.36), run_time=2)
        # self.wait()

        img_path_full=img_path/'gpt_2_attention_viz_5'/str(i)
        q1n=image_at_height(str(img_path_full/'q_1n.png'))
        q1n.scale([0.0127, 0.022, 1]) 
        q1n.move_to([-0.2,0.264,0])
        # self.add(q1n) 

        # self.wait()
        self.play(a[0][0].animate.scale([1, 0.92, 1]).shift(0.012*gx.UP))
        self.play(gx.FadeIn(q1n), gx.Transform(a[1][3][-1], a2[1][3][-2:]))
        # self.wait()

        img_path_full=img_path/'gpt_2_attention_viz_5'/str(i)
        k1n=image_at_height(str(img_path_full/'k_1n.png'))
        k1n.scale([0.0127, 0.022, 1]) 
        k1n.move_to([-0.2,-0.17,0])
        # self.add(k1n)  
        # self.remove(k1n)       
        
        # self.wait()
        self.play(a[0][1].animate.scale([1, 0.92, 1]).shift(0.012*gx.UP))
        self.play(gx.FadeIn(k1n), gx.Transform(a[1][4][-1], a2[1][4][-2:]))
        # self.wait()  
        

        img_path_full=img_path/'gpt_2_attention_viz_5'/str(i)
        v1n=image_at_height(str(img_path_full/'v_1n.png'))
        v1n.scale([0.0127, 0.022, 1]) 
        v1n.move_to([-0.2,-0.595,0])
        # self.add(v1n)  
        # self.remove(v1n)       
        
        # self.wait()
        self.play(a[0][2].animate.scale([1, 0.92, 1]).shift(0.012*gx.UP))
        self.play(gx.FadeIn(v1n), gx.Transform(a[1][5][-1], a2[1][5][-2:]))
        # self.wait()  


        self.play(self.animate_view(0, 0, 0, (0.02, 0.18, 0.0), 0.97), run_time=2)
        # self.wait()

        full_k=gx.Group(a[0][1], k1n)
        kt=full_k.copy()
        # kt.scale(np.array([0.0215,0.035, 1])/np.array([0.0415, 0.08, 1])).rotate([0, 0, -PI/2]).rotate(PI,UP).move_to([0.405,0.305,0])
        # self.add(kt)
        # a[1][7].set_opacity(1.0)


        self.play(kt.animate.scale(np.array([0.0215,0.035, 1])/np.array([0.0415, 0.08, 1])).rotate(-gx.PI/2).rotate(gx.PI,gx.UP).move_to([0.405,0.305,0]),
                  run_time=2)
        a[1][7].set_opacity(1.0)
        self.wait()
        
        a2[1][9:13].set_opacity(0.2)
        # a2[1][5].set_opacity(0.2)
        # a2[1][16].set_opacity(0.2)
        a[0][2].set_opacity(0.2)
        v1n.set_opacity(0.2)
        self.add(a2[0][5].set_opacity(0.2)) #add in last two images we'll need later
        self.add(a2[0][6].set_opacity(0.2))
        self.play(gx.FadeIn(a2[0][4]), gx.FadeIn(a2[1][8:13]), gx.FadeOut(a[1][8:13]))
        self.remove(a[0][4])
        self.wait()

        #Let's do these recangles and the following ones in Illustrator/premier/post manim - I think that will be faster
        # r1=Rectangle(width=0.92, height=0.23)
        # r1.move_to([-0.2, 0.393, 0])
        # r1.set_stroke('#FF00FF')
        # self.add(r1)
        # self.wait()


        # r2=Rectangle(width=0.1, height=0.48)
        # r2.move_to([0.4, 0.305, 0])
        # r2.set_stroke('#FF00FF')
        # self.add(r2)
        # self.wait()

        # r3=Rectangle(width=0.185, height=0.185)
        # r3.move_to([0.649, 0.38, 0])
        # r3.set_stroke('#FF00FF')
        # self.add(r3)
        # self.wait()

        self.play(self.animate_view(0, 0, 0, (0.42, 0.2, 0.0), 1.32),
                a2[0][5].animate.set_opacity(1.0),
                a2[1][9:11].animate.set_opacity(1.0),
                run_time=2.0)
        # self.add(a2[0][5])
        # self.set_view(0, 0, 0, (0.27, -0.01, 0.0), 1.50)
        # self.wait()

        self.play(self.animate_view(0, 0, 0, (0.31, -0.02, 0.0), 1.49),
                a[0][2].animate.set_opacity(1.0),
                v1n.animate.set_opacity(1.0),
                a2[0][-1].animate.set_opacity(1.0),
                a2[1][11:13].animate.set_opacity(1.0), run_time=2.0)
        self.wait()

        # self.play(self.animate_view(0, 0, 0, (0.0, -0.03, 0.0), 2.03), run_time=2.0)
        # self.wait()

        ## Ok start P29 here - Do i want to have the magenta rectangles as I zoom in? Maybe?
        full_panel=gx.Group(a[0][1],
                         a[0][2],                 
                         a[1][1:11],
                        a[0][0],
                        a[0][3:],
                        a2[0][3:],
                        a2[1][7:13],
                        a[1][17],
                        a2[1][16],
                        blue_word,
                        a2[1][1][-2:],
                        kt,
                        q1n,
                        k1n,
                        v1n,
                        x[0],
                        x1n,
                        x2n)

        full_panel.rotate(gx.PI/2, axis=gx.RIGHT)
        # self.wait()
        self.set_view(0, 90, 0, (-0.02, -0.02, -0.05), 2.03)
        self.wait()


        self.play(full_panel[2:].animate.set_opacity(0), 
                  # full_panel[0].animate.move_to([-0.2, -0.01976856, -0.1]),
                  # full_panel[1].animate.move_to([-0.2, -0.01976856, -0.37]),
                  self.animate_view(0, 89, 0, (-0.18, -0.02, -0.25), 1.55),
                  run_time=2)
        self.wait()


        self.play(full_panel[0].animate.move_to([-0.2, -0.01976856, -0.1]),
                  full_panel[1].animate.move_to([-0.2, -0.01976856, -0.37]),
                  run_time=1)
        self.wait()    

        attention_heads=gx.Group()
        spacing=0.1
        for i in range(1, 12): 
            a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                        img_path=img_path/'gpt_2_attention_viz_4'/str(i))
            a_select=gx.Group(a[0], a[1][0], a[1][3:6], a[1][6:13], a[1][17])
            a_select.rotate(gx.PI/2, axis=gx.RIGHT)
            a_select.move_to([0.27, spacing*i,0])

            a_select[0][1].scale([1, 1, 0.92])
            a_select[0][1].move_to([-0.2, spacing*i, -0.1])
            a_select[0][2].scale([1, 1, 0.92])
            a_select[0][2].move_to([-0.2, spacing*i, -0.37])

            attention_heads.add(a_select)

        for i in range(10, -1, -1):
            self.add(attention_heads[i][0][1])
            self.add(attention_heads[i][0][2])

        self.add(full_panel[:2])
        self.play(self.animate_view(-37, 71, 0, (-0.18, -0.02, -0.25), 1.55), run_time=2)

        self.wait()
        #Now add in other layers - getting there with full 3d view - gotta noodle a little more. 
        layer=gx.Group(*[gx.Group(attention_heads[i][0][1], attention_heads[i][0][2]) for i in range(10, -1, -1)])
        layer.add(full_panel[:2])

        layer_2=layer.copy()
        layer_2.rotate(5*gx.PI/180, gx.IN)
        layer_2.shift(1.25*gx.RIGHT+0.75*gx.DOWN)
        # self.add(layer_2)

        layer_3=layer.copy()
        layer_3.rotate(12*gx.PI/180, gx.IN)
        layer_3.shift(2.5*1.25*gx.RIGHT+2.5*0.75*gx.DOWN)
        # self.add(layer_3)

        # self.remove(layer_2)
        # self.remove(layer_3)
        # self.set_view(-34, 70, 0, (0.93, -1.16, -0.9), 3.39)
        self.play(gx.FadeIn(layer_2), gx.FadeIn(layer_3),self.animate_view(-34, 70, 0, (0.93, -1.16, -0.9), 3.39), run_time=2)


        self.wait()


        # full_panel[0].move_to([-0.2, -0.01976856, -0.1])
        # full_panel[1].move_to([-0.2, -0.01976856, -0.37])

        # full_panel[-3].set_opacity(1)

        # attention_heads=Group()
        # spacing=0.25
        # for i in range(1, 12): 
        #     a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
        #                                 img_path=img_path/'gpt_2_attention_viz_4'/str(i))
        #     a_select=Group(a[0], a[1][0], a[1][3:6], a[1][6:13], a[1][17])
        #     a_select.rotate([PI/2,0,0], axis=RIGHT)
        #     a_select.move_to([0.27, spacing*i,0])
        #     attention_heads.add(a_select)

        # self.
        # a[0][1].move_to([-0.2,-0.12,  0.0])
        # a[0][2].move_to([-0.2,-0.4,  0.0])

        # self.play(
        #         a[1][1:11].animate.set_opacity(0.0),
        #         a[0][0].animate.set_opacity(0),
        #         a[0][3:].animate.set_opacity(0),
        #         a2[0][3:].animate.set_opacity(0),
        #         a2[1][7:13].animate.set_opacity(0.0),
        #         a[1][17].animate.set_opacity(0.0),
        #         a2[1][16].animate.set_opacity(0.0),
        #         blue_word.animate.set_opacity(0.0),
        #         a2[1][1][-2:].animate.set_opacity(0),
        #         kt.animate.set_opacity(0),
        #         q1n.animate.set_opacity(0),
        #         k1n.animate.set_opacity(0),
        #         v1n.animate.set_opacity(0),
        #         x[0].animate.set_opacity(0),
        #         x1n.animate.set_opacity(0),
        #         x2n.animate.set_opacity(0),
        #         self.animate_view(0, 0, 0, (-0.17, -0.24, 0.0), 1.93),
        #         run_time=2)




        self.wait(20)
        # Scene authoring stops here in the interactive editor.
