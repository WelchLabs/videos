from manimlib import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b'
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
MAGENTA='#FF00FF'

#ulimit -n 4096

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics/p34_48_to_manim')
img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics'


class p43_48(InteractiveScene):
    def construct(self):
        '''Might end up needing to break this scene up, we'll see'''

        svgs_to_skip=[]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])

        mushroom=ImageMobject(str(img_dir+'/AdobeStock_93352397.jpeg'))
        two_cats=ImageMobject(str(img_dir+'/AdobeStock_240682884.jpeg'))
        gqa_image=ImageMobject(str(img_dir+'/gqa_img.jpg'))
        gqa_table=ImageMobject(str(img_dir+'/vl_jepa_gqa_plot'))
        v_jepa_cats=ImageMobject(str(img_dir+'/P19_22_3db.mp4.00_00_10_29.Still001 copy.jpg'))
        v_jepa_cats_2=ImageMobject(str(img_dir+'/P19_22_3db.mp4.00_00_13_27.Still002 copy.jpg'))
        zucchini = ImageMobject(str(img_dir+'/Pi07VFINAL.mp4.00_00_44_07.Still001 copy 2.jpg'))

        mushroom_copy=mushroom.copy() 
        mushroom_copy.scale(0.28)
        mushroom_copy.move_to([-1.55, -0.05, 0])
        mushroom.scale(0.35).move_to([-4.7, -1.8, 0])

        self.add(all_svgs[46], all_svgs[47], all_svgs[48], all_svgs[49])
        self.add(mushroom_copy)
        self.add(mushroom)
        self.wait()

        self.play(Write(all_svgs[14][:60]), run_time=4)
        self.wait()
        self.play(Write(all_svgs[14][60:]), run_time=2)

        self.wait()
        self.play(Write(all_svgs[12]), run_time=4)
        self.wait()
        self.play(Write(all_svgs[13]), run_time=3)
        self.add(all_svgs[15])

        #P44 Let's go
        fade_group=Group(all_svgs[46], all_svgs[48], all_svgs[49], all_svgs[12], all_svgs[13],  all_svgs[14], all_svgs[15], mushroom)
        # self.remove(all_svgs[46], all_svgs[48], all_svgs[49], all_svgs[12], all_svgs[13],  all_svgs[14], all_svgs[15], mushroom)
        self.play(FadeOut(fade_group))

        jepa_and_mushroom=Group(all_svgs[47], mushroom_copy)
        self.wait()
        self.play(jepa_and_mushroom.animate.move_to([-4.1, -0.3, 0]),
                  Write(all_svgs[16]), run_time=4)
        self.add(all_svgs[18][-22:]) #Do not eat this mushroom
        
      
        self.wait()
        self.play(Write(all_svgs[20]), Write(all_svgs[21]), run_time=5)

        self.wait()
        self.play(ShowCreation(all_svgs[22]), run_time=2)
        self.play(ShowCreation(all_svgs[23]), run_time=2)

        gqa_table.scale(0.9)
        gqa_table.move_to([3, 1.5, 0])
        gqa_image.scale(0.65)
        gqa_image.move_to([1.5, -1.8, 0])

        self.wait()
        self.play(FadeOut(all_svgs[20]), 
                  FadeOut(all_svgs[21]),
                  FadeOut(all_svgs[22]),
                  FadeOut(all_svgs[23]))

        self.play(FadeIn(gqa_table),
                  FadeIn(gqa_image), 
                  FadeIn(all_svgs[24]))


        self.wait()
        self.remove(gqa_table, gqa_image, all_svgs[24])

        # self.remove(all_svgs[47][49]) #minus sign

        right_network=Group(all_svgs[47][47:49], all_svgs[47][:45])
        left_network=all_svgs[47][50:-19]
        vl_jepa_label=all_svgs[16][-7:]
        two_cats.scale(0.5)
        two_cats.move_to([-1.7, -1.5, 0])

        self.wait()
        self.remove(all_svgs[47][45:47], all_svgs[47][-19:], mushroom_copy, all_svgs[16][:-7], all_svgs[18][-22:], all_svgs[47][49])
        self.play(right_network.animate.scale(1.1).move_to([2.8, -1.1, 0]),
                  left_network.animate.scale(1.1).move_to([-0.6, 1.1, 0]),
                  vl_jepa_label.animate.scale(1.1).move_to([1, 3.2, 0]),
                  self.frame.animate.reorient(0, 0, 0, (0.64, -0.15, 0.0)),
                  run_time=4)
        self.add(two_cats)
        self.add(all_svgs[25], all_svgs[26])

        self.wait()
        self.play(LaggedStart([Write(all_svgs[27]), Write(all_svgs[28]),Write(all_svgs[29])], lag_ratio=0.9), 
                  run_time=6)


        self.wait()
        all_svgs[28].set_color(GREEN)

        self.wait()
        self.play(FadeOut(all_svgs[27]), FadeOut(all_svgs[28]), FadeOut(all_svgs[29]))
        self.play(right_network.animate.shift([0, 0.6, 0]), 
                  Write(all_svgs[30]),
                  run_time=4)

        self.wait()

        #P47

        self.wait()

        self.remove(all_svgs[30], two_cats, all_svgs[25], all_svgs[26],left_network[54:57], right_network[0])
        # self.add(left_network)

        # self.add(all_svgs[31], all_svgs[32], all_svgs[33], all_svgs[34], all_svgs[35], all_svgs[36], all_svgs[37], all_svgs[38])
        mushroom_copy.move_to([2.55, -3.1, 0])

        v_jepa_cats.scale(0.35)
        v_jepa_cats.move_to([-5.4, -2.9, 0])
        v_jepa_cats_2.scale(0.35)
        v_jepa_cats_2.move_to([-3.4, -2.9, 0])

        self.wait()
        self.play(ReplacementTransform(left_network[57:], all_svgs[36]), #, run_time=3)
                  ReplacementTransform(left_network[:54], all_svgs[37][:-2]), 
                  ReplacementTransform(right_network[1], all_svgs[38][:45]),
                  ReplacementTransform(vl_jepa_label, all_svgs[35][-7:]),
                  self.frame.animate.reorient(0, 0, 0, (0.03, -0.17, 0.0)),
                  run_time=5)
        self.add(all_svgs[35], all_svgs[36], all_svgs[37], all_svgs[38], mushroom_copy)

        self.play(FadeIn(all_svgs[39]), FadeIn(v_jepa_cats), FadeIn(v_jepa_cats_2),
                  FadeIn(all_svgs[31]), FadeIn(all_svgs[32]), FadeIn(all_svgs[33]))
        self.play(Write(all_svgs[34][:2]),run_time=3)
        self.wait()
        self.play(Write(all_svgs[34][2:]),run_time=3)

        # self.wait()
        # self.play(Write(all_svgs[31]), Write(all_svgs[32]), Write(all_svgs[33]), Write(all_svgs[34]), run_time=4)

        # P48 - Hmm there's going to be a hard cut here, could do a new 
        # scene, let's see here tho. 
        group_to_remove_2=Group(v_jepa_cats, v_jepa_cats_2, mushroom_copy, all_svgs[36], all_svgs[37], all_svgs[38], all_svgs[35], all_svgs[34], all_svgs[39])
        vla_group=Group(all_svgs[31], all_svgs[32], all_svgs[33])
        vla_group_copy=vla_group.copy()
        vla_group_copy.move_to([0.585, 0.055, 0])
        vla_group_copy[0].set_color(FRESH_TAN)
        vla_group_copy[2].set_color(FRESH_TAN)
        
        zucchini.scale(0.58)
        zucchini.move_to([-4.4, 0.5, 0])
        self.wait()
        self.play(FadeOut(group_to_remove_2), run_time=3)
        self.play(ReplacementTransform(vla_group, vla_group_copy),
                  self.frame.animate.reorient(0, 0, 0, (-0.25, 0.07, 0.0), 6.78),
                  # all_svgs[31].animate.set_color(FRESH_TAN),
                  # all_svgs[33].animate.set_color(FRESH_TAN),
                  run_time=3
                  )
        self.play(
                  Write(all_svgs[40]), 
                  Write(all_svgs[41]),
                  FadeIn(zucchini),
                  run_time=4
                 )


        full_vla_group=Group(all_svgs[42], all_svgs[43])
        # self.add(full_vla_group) #Bigger border VLA and action expert
        
        full_vla_group.move_to([0.6, -0.68, 0])

        # self.remove(all_svgs[42][0]) #Border
        # self.remove(all_svgs[40][0])
        # self.remove(all_svgs[41][-36:])
        output=all_svgs[41][-34:]
        # output.move_to([4.5, -2.15, 0])

        self.wait()
        self.play(ReplacementTransform(all_svgs[40], all_svgs[42]),
                  self.frame.animate.reorient(0, 0, 0, (-0.11, -0.46, 0.0), 6.78),
                  output.animate.move_to([4.5, -2.15, 0]),
                  run_time=4)
        self.play(Write(all_svgs[43]), run_time=3)

        self.wait()


        


        self.wait(20)
        self.embed()






class p34_40(InteractiveScene):
    def construct(self):
        '''Might end up needing to break this scene up, we'll see'''

        svgs_to_skip=[]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])


        mushroom=ImageMobject(str(img_dir+'/AdobeStock_93352397.jpeg'))
        two_cats=ImageMobject(str(img_dir+'/AdobeStock_240682884.jpeg'))
        gqa_image=ImageMobject(str(img_dir+'/gqa_img.jpg'))
        
        #P34
        self.frame.reorient(0, 0, 0, (-0.32, 0.29, 0.0), 6.71)
        self.wait()

        self.add(all_svgs[0], all_svgs[1], all_svgs[2])

        all_svgs[4][0].set_opacity(0.2)
        all_svgs[3][0].set_opacity(0.15)

        self.wait()
        self.play(Write(all_svgs[4]), run_time=3)
        self.wait()

        self.play(Write(all_svgs[3]), run_time=3)
        self.wait()

        #P35 question, little zoom out to get read
        self.play(self.frame.animate.reorient(0, 0, 0, (-0.32, 0.21, 0.0), 7.64), 
                  FadeOut(all_svgs[3]), 
                  FadeOut(all_svgs[4]),
                  run_time=4)

        

        all_svgs[5].move_to([-0.2, -0.25, 0])
        mushroom.scale(0.47)
        mushroom.move_to([-5.45, 0.6, 0])

        #P36
        self.wait()
        self.play(Write(all_svgs[5][17:19]),
                  FadeIn(mushroom),
                  run_time=3)
        self.wait()
        self.play(Write(all_svgs[5][:17]),
                  Write(all_svgs[5][19:-16]),
                  run_time=3)
        self.wait()
        self.play(Write(all_svgs[5][-16:]),
                  run_time=3)


        #P37
        vla_bottom=Group(all_svgs[0], all_svgs[1], all_svgs[5], mushroom)
        vlm_text=all_svgs[2][3:6]

        self.wait()
        self.remove(all_svgs[2]) #Remove border
        self.play(vla_bottom.animate.scale(0.86).move_to([0.2, -2.3, 0]),
                  ReplacementTransform(vlm_text, all_svgs[7]),
                  self.frame.animate.reorient(0, 0, 0, (-0.01, 0.14, 0.0), 7.98),
                  run_time=4)
        self.play(Write(all_svgs[6]), 
                  Write(all_svgs[8]),
                  Write(all_svgs[9]),
                  Write(all_svgs[10]), 
                  run_time=4)

        self.play(Write(all_svgs[11]), run_time=3)

        self.wait()
        self.add(all_svgs[10])


        right_side_copy=Group(all_svgs[9].copy(), all_svgs[10][:-1].copy())
        definitely_not=all_svgs[5][-16:-2].copy()


        self.wait()
        self.add(right_side_copy, definitely_not)
        self.play(right_side_copy.animate.move_to([4.8, 1.7, 0]), 
                  run_time=4)
        self.play(definitely_not.animate.move_to([5.1, 0, 0]), run_time=3)


        jepa_left_side_copy=all_svgs[8][:53].copy()
        self.add(jepa_left_side_copy)
        
        mushroom_copy = mushroom.copy()
        self.add(mushroom_copy)

        self.wait()
        self.play(jepa_left_side_copy.animate.move_to([1.8, 1.43, 0]), 
                  run_time=4)
        self.play(mushroom_copy.animate.scale(0.5).move_to([2.05, -0.18, 0]),
                  jepa_left_side_copy[:2].animate.shift([0, 0.06, 0]), #Scrooch up arrow
                  run_time=4)
        

        lil_arrow=jepa_left_side_copy[:2].copy()
        lil_arrow.move_to([3.63, 1.82, 0])

        eat_this=all_svgs[5][:17].copy()
        self.add(eat_this)

        self.wait()
        self.play(eat_this.animate.scale(0.9).move_to([3.63, 1.47, 0]),
                 FadeIn(lil_arrow), 
                 FadeOut(right_side_copy[0][-9:]),
                 run_time=4)


        #P40
        # self.add(all_svgs[11]) #JEPA VLM title
        group_to_fade = Group(all_svgs[0], all_svgs[1], all_svgs[5], all_svgs[6], all_svgs[7], all_svgs[8], all_svgs[9], all_svgs[10], mushroom)


        self.wait()
        self.play(FadeOut(group_to_fade), 
                  self.frame.animate.reorient(0, 0, 0, (3.7, 1.45, 0.0), 5.53),
                  run_time=5)

        self.wait()

        # Ok things are getting unusably slow here -> i think asking Claude for 
        # a fresh start on this scene will speed thigns up, and I have a hard
        # cut before P43, so a little change is no big deal. 


        
        



        self.embed()
        self.wait(20)
