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

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/graphics/p28_33/')
img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/hacking/overhead_ball_1a'




class P28_33c(InteractiveScene):
    def construct(self):

        imgs=Group()
        for i in range(10, 146):
            imgs.add(ImageMobject(str(img_dir+'/overhead_ball_1'+str(i).zfill(3)+'.jpg')))
        # imgs.rotate(90*DEGREES, [1, 0, 0])

        svgs_to_skip=[0]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(5.0)
            all_svgs.add(svg_image[1:])


        # self.frame.reorient(0, 0, 0, (0.22, -0.06, 0.0), 3.92)
        self.frame.reorient(0, 0, 0, (-0.0, -0.0, 0.0), 3.50)
        self.wait()
        for i in range(90):
            if i>0: self.remove(imgs[i-1])
            self.add(imgs[i])
            self.wait(0.05)

        #First blur frame is like 1108-10
        #might need subtle chill brown frame around these guys

        input_video=Group()
        input_borders = Group()
        spacing=0.6
        # indices_to_show=[-40, -35, -30, -25, -20, -15, -10, -5]
        indices_to_show=[79, 81, 83, 85, 87, 89]
        for count, i in enumerate(indices_to_show):
            imgs[i].move_to([0, 0, -spacing*(len(indices_to_show)-1)+spacing*count])
            imgs[i].set_opacity(0.5)
            border = SurroundingRectangle(imgs[i], color=CHILL_BROWN, buff=0)
            border.set_stroke(width=2, opacity=0.0)
            input_video.add(imgs[i])
            # border.set_opacity(0.0)
            input_borders.add(border)
        
        input_video_and_borders=Group(input_video, input_borders)

        self.add(input_video_and_borders)
        imgs[89].set_opacity(1.0)
        self.remove(imgs[89]); self.add(imgs[89])

        

        # imgs[89].set_opacity(0.5)
        # self.add(input_borders)
        # self.frame.reorient(0, 0, 0, (2.78, -0.65, 0.0), 8.60)
        # input_video_and_borders.rotate(65*DEGREES, [0, 1, 0]).rotate(35*DEGREES, [1, 0, 0])

        # input_video_and_borders.rotate(5*DEGREES, [0, 1, 0])

        
        target = input_video_and_borders.copy()
        for b in target[1]:
            b.set_stroke(opacity=0.8)

        # input_borders.set_stroke(opacity=0.0)
        target.rotate(72*DEGREES, [0, 1, 0]).rotate(35*DEGREES, [1, 0, 0])
        # input_video_and_borders[0] is input_video; [5] is imgs[89] (last in indices_to_show)
        target[0][5].set_opacity(0.5)

        self.wait()
        self.play(
            # self.frame.animate.reorient(0, 0, 0, (2.78, -0.65, 0.0), 8.60),
            self.frame.animate.reorient(0, 0, 0, (4.78, -0.39, 0.0), 10.52),
            Transform(input_video_and_borders, target),
            run_time=5,
        )

        self.wait()
        all_svgs[0].move_to([4.8, -0.7, 0])

        # self.add(all_svgs[0])
        self.play(FadeIn(all_svgs[0]), run_time=3)

        imgs[110].scale(0.95)
        imgs[110].move_to([10, -0.5, 0])
        border_110 = SurroundingRectangle(imgs[110], color=CHILL_BROWN, buff=0)
        border_110.set_stroke(width=2, opacity=0.5)
        group_110=Group(imgs[110], border_110)
        self.wait()
        self.add(group_110)
        
        # self.play(FadeIn(group_110), run_time=2)
        # self.remove(border_110); self.add(border_110)


        # Ah when we get to the langauge bit, the network ican can presist, 
        # and we can just temporarily lose the video inputs/outputs!
        self.wait()
        self.play(FadeOut(input_video_and_borders), FadeOut(group_110))

        t_final=Text("The capital of France is Paris", font_size=45)
        t_final.move_to([0, -0.72 , 0])
        # self.add(t_final)

        t1=Text("The", font_size=45)
        t1.move_to([2, -0.72 , 0])
        self.add(t1)

        t2=Text("capital", font_size=45)
        t2.set_color(YELLOW)
        t2.move_to([7.8, -0.7 , 0])
        self.wait()
        self.play(Write(t2), run_time=2)

        self.wait()
        t3=Text("The capital", font_size=45)
        t3.move_to([1.3, -0.73 , 0])
        # self.add(t3)

        self.play(ReplacementTransform(t1, t3[:3]), 
                  ReplacementTransform(t2, t3[3:]),
                  run_time=3)

        t4=Text("of", font_size=45)
        t4.set_color(YELLOW)
        t4.move_to([7.8, -0.7 , 0])
        self.wait()
        self.play(Write(t4), run_time=2)        

        self.wait()
        t5=Text("The capital of ", font_size=45)
        t5.move_to([1.1, -0.73 , 0])
        # self.add(t5)

        self.play(ReplacementTransform(t3, t5[:-2]), 
                  ReplacementTransform(t4, t5[-2:]),
                  run_time=3)


        t6=Text("France", font_size=45)
        t6.set_color(YELLOW)
        t6.move_to([7.8, -0.7 , 0])
        self.wait()
        self.play(Write(t6), run_time=2)        

        self.wait()
        t7=Text("The capital of France", font_size=45)
        t7.move_to([0.7, -0.73 , 0])
        # self.add(t7)

        self.play(ReplacementTransform(t5, t7[:-6]), 
                  ReplacementTransform(t6, t7[-6:]),
                  run_time=3)


        t8=Text("is", font_size=45)
        t8.set_color(YELLOW)
        t8.move_to([7.8, -0.7 , 0])
        self.wait()
        self.play(Write(t8), run_time=2)        

        self.wait()
        t9=Text("The capital of France is", font_size=45)
        t9.move_to([0.3, -0.73 , 0])
        # self.add(t9)

        self.play(ReplacementTransform(t7, t9[:-2]), 
                  ReplacementTransform(t8, t9[-2:]),
                  run_time=3)

        t10=Text("Paris", font_size=45)
        t10.set_color(YELLOW)
        t10.move_to([7.8, -0.7 , 0])
        self.wait()
        self.play(Write(t10), run_time=2)    

        # self.play(ReplacementTransform(t9, t_final[:-5]), 
        #           ReplacementTransform(t10, t_final[-5:]),
        #           run_time=3)


        self.wait()
        self.play(FadeOut(t9), FadeOut(t10))

        self.wait()
        self.play(FadeIn(input_video_and_borders), FadeIn(group_110))

        # Ok Claude, how do I animate group_110 coming over and getting added
        # to the stack of frames in input_video_and_borders?


        # --- compute where the new frame should land in the rotated stack ---
        top_frame  = input_video_and_borders[0][5]   # imgs[89], current top
        prev_frame = input_video_and_borders[0][4]   # imgs[87], one below
        stack_offset  = top_frame.get_center() - prev_frame.get_center()
        target_center = top_frame.get_center() + stack_offset

        # --- build target state: same orientation + look as the other stack frames ---
        target_110 = group_110.copy()
        target_110.scale(1.0 / 0.95)                      # undo the 0.95 scale from earlier
        target_110.rotate(72 * DEGREES, [0, 1, 0]).rotate(35 * DEGREES, [1, 0, 0])
        target_110[0].set_opacity(0.5)                    # match other background frames
        target_110[1].set_stroke(width=2, opacity=0.8)    # match target[1] borders
        target_110.move_to(target_center)

        # --- animate slide + rotate + settle into the stack ---
        self.wait()
        self.play(Transform(group_110, target_110), 
                  all_svgs[0].animate.shift([0.35, 0, 0]), 
                  self.frame.animate.reorient(0, 0, 0, (5.06, -0.56, 0.0), 10.73),
                  run_time=3)

        # (optional) register it as part of the stack so later ops treat it uniformly
        input_video_and_borders[0].add(imgs[110])
        input_video_and_borders[1].add(border_110)


        imgs[120].scale(0.95)
        imgs[120].move_to([10.3, -0.5, 0])
        border_120 = SurroundingRectangle(imgs[120], color=CHILL_BROWN, buff=0)
        border_120.set_stroke(width=2, opacity=0.5)
        group_120=Group(imgs[120], border_120)

        self.wait()
        self.add(group_120)


       # --- compute where the new frame should land in the rotated stack ---
        top_frame  = input_video_and_borders[0][6]   # imgs[89], current top
        prev_frame = input_video_and_borders[0][5]   # imgs[87], one below
        stack_offset  = top_frame.get_center() - prev_frame.get_center()
        target_center = top_frame.get_center() + stack_offset

        # --- build target state: same orientation + look as the other stack frames ---
        target_120 = group_120.copy()
        target_120.scale(1.0 / 0.95)                      # undo the 0.95 scale from earlier
        target_120.rotate(72 * DEGREES, [0, 1, 0]).rotate(35 * DEGREES, [1, 0, 0])
        target_120[0].set_opacity(0.5)                    # match other background frames
        target_120[1].set_stroke(width=2, opacity=0.8)    # match target[1] borders
        target_120.move_to(target_center)

        # --- animate slide + rotate + settle into the stack ---
        self.wait()
        self.play(Transform(group_120, target_120), 
                  all_svgs[0].animate.shift([0.45, 0, 0]), 
                  self.frame.animate.reorient(0, 0, 0, (5.22, -0.73, 0.0), 10.94),
                  run_time=3)

        # (optional) register it as part of the stack so later ops treat it uniformly
        input_video_and_borders[0].add(imgs[120])
        input_video_and_borders[1].add(border_120)


        imgs[133].scale(0.95)
        imgs[133].move_to([10.8, -0.5, 0])
        border_130 = SurroundingRectangle(imgs[133], color=CHILL_BROWN, buff=0)
        border_130.set_stroke(width=2, opacity=0.5)
        group_130=Group(imgs[133], border_130)

        self.wait()
        self.add(group_130)



        # self.frame.reorient(0, 0, 0, (4.78, -0.39, 0.0), 10.52)

        # target.rotate(5*DEGREES, [0, 1, 0])


        # self.play(imgs[89].animate.set_opacity(0.5), run_time=2)
        # imgs[89].set_opacity(0.7)
        # self.play(
        #     self.frame.animate.reorient(0, 0, 0, (2.78, -0.65, 0.0), 8.60),
        #     input_video_and_borders.animate.rotate(65*DEGREES, [0, 1, 0]).rotate(35*DEGREES, [1, 0, 0]),
        #     # imgs[89].animate.set_opacity(0.5),
        #     # input_video_and_borders.animate,
        #     # imgs[89].animate.set_opacity(0.5),
        #     run_time=5
        # )
        # imgs[89].set_opacity(0.5)

        # input_video.set_opacity(0.7)


        # imgs[89].set_opacity(0.5)

        # input_video_and_borders.rotate(5*DEGREES, [1, 0, 0])
        
        # self.wait()



        # self.remove(input_video)

        # self.remove(imgs[0])




        self.embed()
        self.wait()





class P28_33a(InteractiveScene):
    def construct(self):

        imgs=Group()
        for i in range(48):
            imgs.add(ImageMobject(str(img_dir+'/a_roll_random'+str(i).zfill(3)+'.jpg')))
        imgs.rotate(90*DEGREES, [1, 0, 0])


        #Play dat videer
        self.frame.reorient(0, 86, 0, (0.18, 0.31, 0.15), 8.00)
        self.wait()
        for i in range(len(imgs)):
            if i>1: self.remove(imgs[i-1])
            self.add(imgs[i])
            self.wait(0.1)

        self.wait()

        #Pan to side reveal depth kidna thing?
        input_video=Group()
        spacing=0.6
        # indices_to_show=[-40, -35, -30, -25, -20, -15, -10, -5]
        indices_to_show=[-45, -40, -35, -30, -25, -20, -15, -10, -5]
        for count, i in enumerate(indices_to_show):
            imgs[i].move_to([0, -spacing*count+spacing*len(indices_to_show), 0])
            imgs[i].set_opacity(0.5)
            input_video.add(imgs[i])

        
        self.wait()
        self.remove(imgs[:-1]); 
        self.add(input_video)
        self.add(imgs[-1])

        self.wait()


        self.frame.reorient(-69, 51, 0, (0.31, -0.2, -0.49), 10.21)
        # self.remove(input_video[-1])
        # self.remove(imgs[-1]); self.add(imgs[-1])


        # Hmm maybe the blurry one is like me walking around?

        

        # self.remove(imgs[-1])

        self.wait()


        self.embed(20)
        self.wait()