from welch_paths import welch_path
import manimgx as gx
from deepseek_diagrams import get_attention_head, get_input_x
import numpy as np
from welch_scene import WelchScene, image_at_height
from welch_text import math_tex

from tqdm import tqdm
from pathlib import Path


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'


class p12_20c(WelchScene):
    def construct(self):

        img_path=Path(welch_path('deepseek/hackin/linux_workdir/deepseek'))
        svg_path=Path(welch_path('deepseek/graphics/to_manim'))

        i=0
        a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                    img_path=img_path/'gpt_2_attention_viz_4'/str(i))

        x=get_input_x(svg_path=welch_path('deepseek/graphics/to_manim'),
           svg_file='mha_2d_grouping_test.svg',
           img_path=img_path/'gpt_2_attention_viz_4')

        self.set_view(0, 0, 0, (-1.39, -0.06, -0.0), 1.16)
        self.wait()
        # self.add(x[0][0])
        # self.add(x[0][1])

        self.play(gx.FadeIn(x[0]))
        self.add(a[1][1]) #xlabels
        self.add(a[1][2]) #xlabels
        self.wait()

        self.add(a[1][16]) #deepseek dim
        self.wait()
        self.remove(a[1][16])
        self.wait()

        self.play(gx.FadeIn(a[1][6]), self.animate_view(0, 0, 0, (-0.65, 0.02, 0.0), 1.36), run_time=2)
        self.wait()

        # self.play(FadeIn(a[1][3]), FadeIn(a[1][4]), FadeIn(a[0][0]), FadeIn(a[0][1]))
        # self.wait()

        # self.add(a[1][3])
        # self.add(a[1][4])
        # Can i, withoug going insane, actually have these matrices be made up of rows and 
        # then break them apart and pull one out of each?

        #Ok hack on row by row version here then break into subfuncrion


        separate_row_im_path=welch_path('deepseek/hackin/linux_workdir/deepseek/gpt_2_attention_viz_4')
        q_rows=gx.Group()
        for row_id in range(9):
            q=image_at_height(separate_row_im_path+'/query_row_'+str(row_id)+'.png')
            q.scale([0.0127, 0.024, 1]) 
            q.move_to([-0.2,0.492-0.028*row_id,0]) 
            q_rows.add(q)
        # self.add(q_rows)
        # self.remove(q_rows)

        k_rows=gx.Group()
        for row_id in range(9):
            k=image_at_height(separate_row_im_path+'/key_row_'+str(row_id)+'.png')
            k.scale([0.0127, 0.024, 1]) 
            k.move_to([-0.2,0.05-0.028*row_id,0]) 
            k_rows.add(k)
        # self.add(k_rows)
        # self.remove(k_rows)

        self.wait()
        self.play(gx.FadeIn(a[1][3]), gx.FadeIn(a[1][4]), gx.FadeIn(k_rows), gx.FadeIn(q_rows)) #Queries and Keys
        self.wait()

        #Pan over, create space for captions, while expanding all rows
        self.play(*[q_rows[row_id].animate.shift(0.01*(8-row_id)*gx.UP) for row_id in range(9)]+
                   [k_rows[row_id].animate.shift(0.01*(8-row_id)*gx.UP) for row_id in range(9)]+
                   [self.animate_view(0, 0, 0, (-0.23, 0.18, 0.0), 1.20)], lag_ratio=0.2, run_time=3)

        self.wait()

        self.play(gx.FadeOut(x[0]), gx.FadeOut(a[1][1]), gx.FadeOut(a[1][2]), gx.FadeOut(a[1][6]))
        a[1][18].scale(0.595)
        a[1][18].move_to([-0.75,0.2,0])
        self.add(a[1][18]) #Add american flag text
        self.wait()

        a[1][19].scale(0.6)
        a[1][19].move_to([0.49,0.37,0])
        self.add(a[1][19][:30]) #Key and Query Questions
        self.wait()
        self.add(a[1][19][30:])
        self.wait()

        #Multistep option
        self.play(gx.FadeOut(a[1][18][:11]), gx.FadeOut(a[1][18][14:32]), gx.FadeOut(a[1][18][40:]), gx.FadeOut(a[1][19]),
                  gx.FadeOut(a[1][3]), gx.FadeOut(a[1][4]), gx.FadeOut(q_rows[:2]), gx.FadeOut(q_rows[3:]), gx.FadeOut(k_rows[0]),
                  gx.FadeOut(k_rows[2:]), run_time=2)
        # self.wait()
        self.play(q_rows[2].animate.shift(0.12*gx.DOWN), k_rows[1].animate.shift(0.12*gx.UP),
                  a[1][18][11:14].animate.shift(0.12*gx.DOWN), a[1][18][32:40].animate.shift(0.12*gx.UP))
        self.wait()

        #All at once option - feels like to much
        # self.play(FadeOut(a[1][18][:11]), FadeOut(a[1][18][14:32]), FadeOut(a[1][18][40:]), FadeOut(a[1][19]), 
        #       FadeOut(a[1][3]), FadeOut(a[1][4]), FadeOut(q_rows[:2]), FadeOut(q_rows[3:]), FadeOut(k_rows[0]), 
        #       FadeOut(k_rows[2:]), q_rows[2].animate.shift(0.15*DOWN), k_rows[1].animate.shift(0.15*UP), run_time=2)     
        # self.wait()

        #Dot Product Dot
        d=gx.Dot(stroke_color=None, fill_color=gx.WHITE)
        d.scale(0.15)
        d.move_to([-0.2, 0.295, 0])
        # d.set_color(WHITE)
        self.add(d)

        t=math_tex("=524.2").set_color(gx.WHITE)
        t.scale(0.1)
        t.move_to([-0.2, 0.12, 0])
        self.add(t)
        self.wait()

        self.remove(d,t)

        # So here's just a pure reversal and everything back, but I wonder if I want to actually return to non-broken rows
        # so I can go straight into the transpose?
        # self.play(q_rows[2].animate.shift(0.12*UP), k_rows[1].animate.shift(0.12*DOWN),
        #           a[1][18][11:14].animate.shift(0.12*UP), a[1][18][32:40].animate.shift(0.12*DOWN))
        # self.wait()

        # self.add(a[1][18][:11], a[1][18][14:32], a[1][18][40:])
        # self.add(a[1][3], a[1][4])
        # self.add(q_rows[:2], q_rows[3:], k_rows[0], k_rows[2:])
        # self.wait()

        #Ok try to smoothly move back to full matrix

        # self.wait()
        self.play(q_rows[2].animate.shift(0.059*gx.UP), k_rows[1].animate.shift(0.188*gx.DOWN),
                  gx.FadeOut(a[1][18][11:14]), gx.FadeOut(a[1][18][32:40]))
        
        # self.add(a[0][0].set_opacity(0.5)) #Query image
        # q_rows[2].shift(0.059*UP)
        # self.wait()
        # self.add(a[0][1].set_opacity(0.5)) #Key image
        # k_rows[1].shift(0.188*DOWN)

        self.play(gx.FadeIn(a[0][0]), gx.FadeIn(a[0][1]))
        self.remove(q_rows[2], k_rows[1])
        self.add(a[1][3][:10], a[1][3][-1:]) #Queryt labels
        self.add(a[1][4][:4], a[1][4][-4:]) #Key labels
        self.wait()

        #Ok time to transpose a copy of the keys and add next set of labels and results -> how can i do this smoothy?
        # self.add(a[0][3])

        kt=a[0][1].copy()
        self.play(kt.animate.scale(np.array([0.0215,0.035, 1])/np.array([0.0415, 0.08, 1])).rotate(-gx.PI/2).rotate(gx.PI,gx.UP).move_to([0.405,0.305,0]), run_time=2)
        # self.play(kt.animate.rotate(PI,RIGHT).rotate([0, 0, -PI/2]))
        self.add(a[1][7])
        self.wait()
        # self.remove(kt)


        self.play(gx.FadeIn(a[1][8]), gx.FadeIn(a[0][4]), self.animate_view(0, 0, 0, (0.16, 0.17, 0.0), 1.20), run_time=1.2)
        self.wait()

        self.add(a[1][9])
        self.wait()
        self.play(gx.FadeIn(a[0][5]), self.animate_view(0, 0, 0, (0.46, 0.18, 0.0), 1.43))
        self.wait()
        self.add(a[1][10])
        self.wait()

        #P18 - move then add option
        self.play(self.animate_view(0, 0, 0, (0, 0, 0.0), 2.00))
        self.remove(a[1][3][:10], a[1][3][-1:]) #Remove partial labels, add back in full. 
        self.remove(a[1][4][:4], a[1][4][-4:])
        self.add(a[1][3], a[1][4]) #Key query value labels
        self.add(x[0], a[1][1], a[1][2], a[1][6]) #X inputs and intiial arrows
        self.wait()

        self.play(gx.FadeIn(a[1][17]), gx.FadeIn(a[1][5]), gx.FadeIn(a[0][2])) #Value Stuff
        self.wait()

        ## Zoom in on values
        self.play(self.animate_view(0, 0, 0, (-0.22, -0.49, -0.0), 1.15))
        self.wait()

        ## Zoom out partially to make comparison to keys
        self.play(self.animate_view(0, 0, 0, (-0.16, -0.09, 0.0), 1.35))
        self.wait()

        ## Fully Zoom out
        self.play(self.animate_view(0, 0, 0, (0, 0, 0.0), 2.00))
        self.wait()

        #P19
        self.add(a[1][11])
        self.wait()

        self.play(gx.FadeIn(a[0][6]), gx.FadeIn(a[1][12]))
        self.wait()

        #Attention head border 
        self.play(gx.Write(a[1][0]), run_time=1) #kind cool kind cheesy, switch to fade in if I hate it
        # self.play(FadeIn(a[1][0]))
        self.wait()

        # Ok now we're getting serious. I need to load up the other 11 heads, bring them in, and maybe like
        # pan to the side as I do it. Maybe I can do a lag ratio thing. Oh yeah shoot and my camera is 
        # top down right now, and for decent pan I need to rotate everything up. Hmm. 
        # Would it be insane to rotate the attention head up like as I did it?

        #Quick hacky test - a little weird but might work?
        #Oh man can my patters slot in from the left one after eachother as the camera moves alittle?
        #That would be dope. 
        head_0_3d_images=gx.Group(a[0], kt) #a[0][:2], a[3:], kt)
        head_0_3d_vectors=gx.Group(a[1][0], a[1][3:6], a[1][6:13], a[1][17])
        head_0_3d=gx.Group(head_0_3d_images, head_0_3d_vectors)

        # Ok here's my alternative solution - there should not appear to be a camera jump in theory, we'll see - 
        # If there is a little one, by be able to fix in Premier
        self.remove(x, a[1][1], a[1][2])
        head_0_3d.rotate(gx.PI/2, axis=gx.RIGHT)
        self.set_view(0, 90, 0, (0, 0, 0.0), 2.0)
        self.wait()

        # self.play(head_0_3d.animate.rotate([PI/2,0,0], axis=RIGHT), 
        #          self.animate_view(0, 90, 0, (0, 0, 0.0), 2.0), run_time=2)

        #Ok now the cool stuff, can I slot in new heads from the left while panning to left?!
        attention_heads=gx.Group()
        spacing=0.25
        for i in range(1, 12): 
            a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                        img_path=img_path/'gpt_2_attention_viz_4'/str(i))
            a_select=gx.Group(a[0], a[1][0], a[1][3:6], a[1][6:13], a[1][17])
            a_select.rotate(gx.PI/2, axis=gx.RIGHT)
            a_select.move_to([0.27, spacing*i,0])
            attention_heads.add(a_select)

        # self.wait()
        # self.set_view(-27, 74, 0, (-0.29, 0.14, -0.1), 2.56)

        #What if I add heads and then set all their opacities to zero? Is ordering preserved then?
        for i in range(10, -1, -1): #-1):
            # print(i)
            self.add(attention_heads[i])
        self.add(head_0_3d)

        for i in range(10, -1, -1): 
            attention_heads[i].shift(6*gx.LEFT)

        # self.wait()
        # self.set_view(-38, 72, 0, (-0.37, 0.29, 0.02), 2.56)
        self.play(self.animate_view(-38, 72, 0, (-0.37, 0.29, 0.02), 2.56), run_time=3)

        # self.play(attention_heads[0].animate.shift(6*RIGHT)) #Works
        # self.play(attention_heads[1].animate.shift(6*RIGHT)) #Works

        self.wait(0)
        for i in range(11): #not my favorite, but gets the job done. 
            self.play(attention_heads[i].animate.shift(6*gx.RIGHT), run_time=1, rate_func=gx.linear)

        self.wait()

        self.play(self.animate_view(36, 83, 0, (0.8, 0.62, -0.11), 2.65), run_time=6)
        self.wait()

        #P20 -> pull out output matrices
        out_matrices=[attention_heads[i][0][-1] for i in range(10, -1, -1)] + [head_0_3d[0][0][-1]]

        # for i,o in enumerate(out_matrices):
        #     o.scale(0.2).move_to([3+i*0.2, 0, 0])

        self.play(*[o.animate.scale(0.2).move_to([3+i*0.2, 0, 0]) for i,o in enumerate(out_matrices)]+
                  [self.animate_view(-1, 86, 0, (4.32, 0.84, -0.21), 3.29)], run_time=3)
        self.wait()

        p20_overlay=gx.SVGMobject(welch_path('deepseek/graphics/to_manim/p20_svg_overay_1.svg'))[1:]
        p20_overlay.rotate(gx.PI/2, axis=gx.RIGHT)
        p20_overlay.scale(1.3)
        p20_overlay.move_to([4.62,0,-0.26])
        self.add(p20_overlay)

        # self.remove(p20_overlay)


        # for i in range(10, -1, -1):
            # out_matrix=attention_heads[i][0][-1]
            # out_matrix.scale(0.2).move_to([2.25+i*0.25, 0, 0])
        # head_0_3d[0][0][-1].scale(0.25).move_to([2.25+(i-1)*0.25, 0, 0])




        # self.play(*[attention_heads[i].animate.shift(6*RIGHT) for i in range(11)], lag_ratio=1.2) #Breaks things down wrong 

        # animations = [attention_heads[i].animate.shift(6*RIGHT) for i in range(11)]
        # staggered_animations = AnimationGroup(*animations, lag_ratio=1.5)
        # self.play(staggered_animations, run_time=2)


        # self.play(attention_heads[i].animate.shift(6*RIGHT))
        # self.set_view(-27, 74, 0, (-0.29, 0.14, -0.1), 2.56)


        # attention_heads.set_opacity(0.0) #Ok ordering seems to survive this simple test
        # attention_heads.set_opacity(1.0)

        #Ok seems like the animation group was actually fucking me in terms of occlusions???
        # animations = [attention_heads[i].animate.set_opacity(0.85) for i in range(11)]
        # staggered_animations = AnimationGroup(*animations, lag_ratio=1.5)
        # self.play(staggered_animations, run_time=2)

        #Well shit I may just have to call this one a loss for now and try later if I have time
        #The behavior I can't seem to create is bringin in the heads on at a time, front to back, without fucking occlusions. 


        #Hmm I'm going to possibly run into some rendering order stuff? 
        #maybe not if I render them off screen in the right order and then bring them in?
        # for i in range(10, -1, -1):
        # for i in range(10,-1,-1):
        #     attention_heads[i].set_opacity(0.0)

        #     # attention_heads[i].shift(6*LEFT) #Not sure what starting position is?
        #     self.add(attention_heads[i])
        # Set z_index based on depth - higher values render on top
        # for i in range(11):
        #     # Calculate z_index based on depth (negative z_coordinate)
        #     z_coord = attention_heads[i].get_center()[2]
        #     # Convert z coordinate to z_index - smaller z should have higher z_index to be on top
        #     z_index = -int(z_coord * 100)  # Scale appropriately for your scene
        #     attention_heads[i].set_z_index(z_index)
        #     attention_heads[i].set_opacity(1.0)
        #     self.add(attention_heads[i])

        # self.add(head_0_3d) #Re add as top layer
        # self.wait()

        # attention_heads.shift(0.25*RIGHT)

        # self.set_view(-27, 74, 0, (-0.29, 0.14, -0.1), 2.56)

        # # animations = [attention_heads[i].animate.shift(6*RIGHT) for i in range(11)]
        # animations = [FadeIn(attention_heads[i]) for i in range(11)]
        # staggered_animations = AnimationGroup(*animations, lag_ratio=1.5)
        # self.play(staggered_animations, run_time=2)

        # head_0_3d.set_z_index(100) #Doesn't seem eo work. 
        # self.add(head_0_3d)

        #Man this is tricky! No matter how I introduce the additional layers, I can't seem to get the occlusion 
        #ordering right, even with seeing z order manually. Am I missing something here?



        # self.play(*[attention_heads[i].animate.shift(6*RIGHT) for i in range(11)], run_time=3, lag_ratio=1.5)


        # self.wait()






        self.wait(20)
        # Scene authoring stops here in the interactive editor.

class p21_pan(WelchScene):
    '''BUnch of extra stuff in here but meh'''
    def construct(self):

        img_path=Path(welch_path('deepseek/hackin/linux_workdir/deepseek'))
        svg_path=Path(welch_path('deepseek/graphics/to_manim'))

        i=0
        a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                    img_path=img_path/'gpt_2_attention_viz_4'/str(i))

        x=get_input_x(svg_path=welch_path('deepseek/graphics/to_manim'),
           svg_file='mha_2d_grouping_test.svg',
           img_path=welch_path('deepseek/hackin/linux_workdir/deepseek/flowchart_graphics'))

        self.set_view(0, 0, 0, (-1.39, -0.06, -0.0), 1.16)
        self.wait()
        # self.add(x[0][0])
        # self.add(x[0][1])

        self.play(gx.FadeIn(x[0]))
        self.add(a[1][1]) #xlabels
        self.add(a[1][2]) #xlabels
        self.wait()

        self.add(a[1][16]) #deepseek dim
        self.wait()
        self.remove(a[1][16])
        self.wait()

        self.play(gx.FadeIn(a[1][6]), self.animate_view(0, 0, 0, (-0.65, 0.02, 0.0), 1.36), run_time=2)
        self.wait()

        # self.play(FadeIn(a[1][3]), FadeIn(a[1][4]), FadeIn(a[0][0]), FadeIn(a[0][1]))
        # self.wait()

        # self.add(a[1][3])
        # self.add(a[1][4])
        # Can i, withoug going insane, actually have these matrices be made up of rows and 
        # then break them apart and pull one out of each?

        #Ok hack on row by row version here then break into subfuncrion


        separate_row_im_path=welch_path('deepseek/hackin/linux_workdir/deepseek/gpt_2_attention_viz_4')
        q_rows=gx.Group()
        for row_id in range(9):
            q=image_at_height(separate_row_im_path+'/query_row_'+str(row_id)+'.png')
            q.scale([0.0127, 0.024, 1]) 
            q.move_to([-0.2,0.492-0.028*row_id,0]) 
            q_rows.add(q)
        # self.add(q_rows)
        # self.remove(q_rows)

        k_rows=gx.Group()
        for row_id in range(9):
            k=image_at_height(separate_row_im_path+'/key_row_'+str(row_id)+'.png')
            k.scale([0.0127, 0.024, 1]) 
            k.move_to([-0.2,0.05-0.028*row_id,0]) 
            k_rows.add(k)
        # self.add(k_rows)
        # self.remove(k_rows)

        self.wait()
        self.play(gx.FadeIn(a[1][3]), gx.FadeIn(a[1][4]), gx.FadeIn(k_rows), gx.FadeIn(q_rows)) #Queries and Keys
        self.wait()

        #Pan over, create space for captions, while expanding all rows
        self.play(*[q_rows[row_id].animate.shift(0.01*(8-row_id)*gx.UP) for row_id in range(9)]+
                   [k_rows[row_id].animate.shift(0.01*(8-row_id)*gx.UP) for row_id in range(9)]+
                   [self.animate_view(0, 0, 0, (-0.23, 0.18, 0.0), 1.20)], lag_ratio=0.2, run_time=3)

        self.wait()

        self.play(gx.FadeOut(x[0]), gx.FadeOut(a[1][1]), gx.FadeOut(a[1][2]), gx.FadeOut(a[1][6]))
        a[1][18].scale(0.595)
        a[1][18].move_to([-0.75,0.2,0])
        self.add(a[1][18]) #Add american flag text
        self.wait()

        a[1][19].scale(0.6)
        a[1][19].move_to([0.49,0.37,0])
        self.add(a[1][19][:30]) #Key and Query Questions
        self.wait()
        self.add(a[1][19][30:])
        self.wait()

        #Multistep option
        self.play(gx.FadeOut(a[1][18][:11]), gx.FadeOut(a[1][18][14:32]), gx.FadeOut(a[1][18][40:]), gx.FadeOut(a[1][19]),
                  gx.FadeOut(a[1][3]), gx.FadeOut(a[1][4]), gx.FadeOut(q_rows[:2]), gx.FadeOut(q_rows[3:]), gx.FadeOut(k_rows[0]),
                  gx.FadeOut(k_rows[2:]), run_time=2)
        # self.wait()
        self.play(q_rows[2].animate.shift(0.12*gx.DOWN), k_rows[1].animate.shift(0.12*gx.UP),
                  a[1][18][11:14].animate.shift(0.12*gx.DOWN), a[1][18][32:40].animate.shift(0.12*gx.UP))
        self.wait()

        #All at once option - feels like to much
        # self.play(FadeOut(a[1][18][:11]), FadeOut(a[1][18][14:32]), FadeOut(a[1][18][40:]), FadeOut(a[1][19]), 
        #       FadeOut(a[1][3]), FadeOut(a[1][4]), FadeOut(q_rows[:2]), FadeOut(q_rows[3:]), FadeOut(k_rows[0]), 
        #       FadeOut(k_rows[2:]), q_rows[2].animate.shift(0.15*DOWN), k_rows[1].animate.shift(0.15*UP), run_time=2)     
        # self.wait()

        #Dot Product Dot
        d=gx.Dot(stroke_color=None, fill_color=gx.WHITE)
        d.scale(0.15)
        d.move_to([-0.2, 0.295, 0])
        # d.set_color(WHITE)
        self.add(d)

        t=math_tex("=524.2").set_color(gx.WHITE)
        t.scale(0.1)
        t.move_to([-0.2, 0.12, 0])
        self.add(t)
        self.wait()

        self.remove(d,t)

        # So here's just a pure reversal and everything back, but I wonder if I want to actually return to non-broken rows
        # so I can go straight into the transpose?
        # self.play(q_rows[2].animate.shift(0.12*UP), k_rows[1].animate.shift(0.12*DOWN),
        #           a[1][18][11:14].animate.shift(0.12*UP), a[1][18][32:40].animate.shift(0.12*DOWN))
        # self.wait()

        # self.add(a[1][18][:11], a[1][18][14:32], a[1][18][40:])
        # self.add(a[1][3], a[1][4])
        # self.add(q_rows[:2], q_rows[3:], k_rows[0], k_rows[2:])
        # self.wait()

        #Ok try to smoothly move back to full matrix

        # self.wait()
        self.play(q_rows[2].animate.shift(0.059*gx.UP), k_rows[1].animate.shift(0.188*gx.DOWN),
                  gx.FadeOut(a[1][18][11:14]), gx.FadeOut(a[1][18][32:40]))
        
        # self.add(a[0][0].set_opacity(0.5)) #Query image
        # q_rows[2].shift(0.059*UP)
        # self.wait()
        # self.add(a[0][1].set_opacity(0.5)) #Key image
        # k_rows[1].shift(0.188*DOWN)

        self.play(gx.FadeIn(a[0][0]), gx.FadeIn(a[0][1]))
        self.remove(q_rows[2], k_rows[1])
        self.add(a[1][3][:10], a[1][3][-1:]) #Queryt labels
        self.add(a[1][4][:4], a[1][4][-4:]) #Key labels
        self.wait()

        #Ok time to transpose a copy of the keys and add next set of labels and results -> how can i do this smoothy?
        # self.add(a[0][3])

        kt=a[0][1].copy()
        self.play(kt.animate.scale(np.array([0.0215,0.035, 1])/np.array([0.0415, 0.08, 1])).rotate(-gx.PI/2).rotate(gx.PI,gx.UP).move_to([0.405,0.305,0]), run_time=2)
        self.add(a[1][7])
        self.wait()


        self.play(gx.FadeIn(a[1][8]), gx.FadeIn(a[0][4]), self.animate_view(0, 0, 0, (0.16, 0.17, 0.0), 1.20), run_time=1.2)
        self.wait()

        self.add(a[1][9])
        self.wait()
        self.play(gx.FadeIn(a[0][5]), self.animate_view(0, 0, 0, (0.46, 0.18, 0.0), 1.43))
        self.wait()
        self.add(a[1][10])
        self.wait()

        #P18 - move then add option
        self.play(self.animate_view(0, 0, 0, (0, 0, 0.0), 2.00))
        self.remove(a[1][3][:10], a[1][3][-1:]) #Remove partial labels, add back in full. 
        self.remove(a[1][4][:4], a[1][4][-4:])
        self.add(a[1][3], a[1][4]) #Key query value labels
        self.add(x[0], a[1][1], a[1][2], a[1][6]) #X inputs and intiial arrows
        self.wait()

        self.play(gx.FadeIn(a[1][17]), gx.FadeIn(a[1][5]), gx.FadeIn(a[0][2])) #Value Stuff
        self.wait()

        #P19
        self.add(a[1][11])
        self.wait()

        self.play(gx.FadeIn(a[0][6]), gx.FadeIn(a[1][12]))
        self.wait()

        #Attention head border 
        self.play(gx.Write(a[1][0]), run_time=1) #kind cool kind cheesy, switch to fade in if I hate it
        # self.play(FadeIn(a[1][0]))
        self.wait()

        # Ok now we're getting serious. I need to load up the other 11 heads, bring them in, and maybe like
        # pan to the side as I do it. Maybe I can do a lag ratio thing. Oh yeah shoot and my camera is 
        # top down right now, and for decent pan I need to rotate everything up. Hmm. 
        # Would it be insane to rotate the attention head up like as I did it?

        #Quick hacky test - a little weird but might work?
        #Oh man can my patters slot in from the left one after eachother as the camera moves alittle?
        #That would be dope. 
        head_0_3d_images=gx.Group(a[0], kt) #a[0][:2], a[3:], kt)
        head_0_3d_vectors=gx.Group(a[1][0], a[1][3:6], a[1][6:13], a[1][17])
        head_0_3d=gx.Group(head_0_3d_images, head_0_3d_vectors)

        # Ok here's my alternative solution - there should not appear to be a camera jump in theory, we'll see - 
        # If there is a little one, by be able to fix in Premier
        self.remove(x, a[1][1], a[1][2])
        head_0_3d.rotate(gx.PI/2, axis=gx.RIGHT)
        self.set_view(0, 90, 0, (0, 0, 0.0), 2.0)
        self.wait()

        # self.play(head_0_3d.animate.rotate([PI/2,0,0], axis=RIGHT), 
        #          self.animate_view(0, 90, 0, (0, 0, 0.0), 2.0), run_time=2)

        #Ok now the cool stuff, can I slot in new heads from the left while panning to left?!
        attention_heads=gx.Group()
        spacing=0.25
        for i in range(1, 12): 
            a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                        img_path=img_path/'gpt_2_attention_viz_4'/str(i))
            a_select=gx.Group(a[0], a[1][0], a[1][3:6], a[1][6:13], a[1][17])
            a_select.rotate(gx.PI/2, axis=gx.RIGHT)
            a_select.move_to([0.27, spacing*i,0])
            attention_heads.add(a_select)

        # self.wait()
        # self.set_view(-27, 74, 0, (-0.29, 0.14, -0.1), 2.56)

        #What if I add heads and then set all their opacities to zero? Is ordering preserved then?
        for i in range(10, -1, -1): #-1):
            # print(i)
            self.add(attention_heads[i])
        self.add(head_0_3d)

        for i in range(10, -1, -1): 
            attention_heads[i].shift(6*gx.LEFT)

        self.wait()
        # self.set_view(-38, 72, 0, (-0.37, 0.29, 0.02), 2.56)
        self.play(self.animate_view(-38, 72, 0, (-0.37, 0.29, 0.02), 2.56), run_time=3)


        for i in range(11): #not my favorite, but gets the job done. 
            self.play(attention_heads[i].animate.shift(6*gx.RIGHT), run_time=1, rate_func=gx.linear)

        self.wait()

        ### --- Do fun pannning here ---- ###
        self.set_view(51, 74, 0, (0.05, 0.82, -0.29), 3.44)

        self.play(self.animate_view(-48, 70, 0, (0.05, 0.82, -0.29), 3.44), run_time=10, rate_func=gx.linear)
        self.wait()



        self.wait(20)
        # Scene authoring stops here in the interactive editor.

        # q1=ImageMobject(str(img_path/'q_1.png'))
        # q1.scale([0.0415, 0.08, 1]) 
        # q1.move_to([-0.2,0.38,0]) 
        # # self.add(q1)
        # # self.remove(q1)      

        # k1=ImageMobject(str(img_path/'k_1.png'))
        # k1.scale([0.0415, 0.08, 1]) 
        # k1.move_to([-0.2,-0.06,0]) 



        # self.wait()
        # self.embed()

        #         attention_heads=Group()
        # spacing=0.25
        # for i in range(12): #Render in reverse order for occlusions
        #     a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
        #                                 img_path=img_path/'gpt_2_attention_viz_1'/str(i))
        #     a.rotate([PI/2,0,0], axis=RIGHT)
        #     a.move_to([0, spacing*i,0])
        #     # a.set_opacity(0.5)
        #     attention_heads.add(a)


        # a=get_attention_head_course(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1/0', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # self.add(a)

        # x=get_input_x(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # self.add(x)

        # self.set_view(0, 0, 0, (0.00, 0.00, 0.00), 2.80)
        # # self.wait()



