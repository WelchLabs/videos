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

NUM_FRAME_TO_RENDER=100 #CRANK UP FOR FINAL VIZ
svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/graphics/p75_80_manim/')
img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/hacking/overhead_ball_3'


class P75_80(InteractiveScene):
    def construct(self):


        svgs_to_skip=[0, 1, 2, 3, 4, 5]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])

        self.frame.reorient(0, 0, 0, (-0.08, 0.21, 0.0), 7.11)
        all_svgs[0].shift([0, -0.2, 0]) #Scrooch down title a smidge

        # self.add(all_svgs[3][4:6]) #Action arrow
        # self.add(all_svgs[4][22:29])
        # self.add(all_svgs[4][:22])
        # self.add(all_svgs[4][36:45])

        # self.wait()
        # self.add(all_svgs[0]) #Title
        # self.add(all_svgs[1]) #Encoder 1
        # self.add(all_svgs[17]) #Predictor
        # self.add(all_svgs[18]) #Encoder 2
        # self.add(all_svgs[19]) #Error box
        # self.add(all_svgs[3]) #Arrows
        # self.add(all_svgs[4]) #Labels
        # self.add(all_svgs[5]) #xs

        #Paragraph 75
        self.wait()
        self.play(Write(all_svgs[0]), run_time=4)

        self.wait()
        self.play(Write(all_svgs[5][::-1]), 
                  Write(all_svgs[4][:11]), 
                  Write(all_svgs[4][11:22]), 
                  run_time=4)
        self.wait()

        self.play(Write(all_svgs[1]),
                  Write(all_svgs[18]),
                  Write(all_svgs[4][22:29]),
                  Write(all_svgs[4][29:36]),
                  run_time=4)
        self.add(all_svgs[3][:4])


        self.wait()
        self.play(Write(all_svgs[17]),
                  Write(all_svgs[3][6:]),
                  Write(all_svgs[19]),
                  Write(all_svgs[4][36:45]),
                  run_time=5)

        self.wait()
        self.play(Write(all_svgs[4][45:]),
                  Write(all_svgs[3][4:6]),
                  run_time=3)

        # Paragraph 76
        # Let me get everything on the canvas here and then
        # figure out how I want to bring it in/transition

        imgs=Group()
        for i in range(NUM_FRAME_TO_RENDER):
            imgs.add(ImageMobject(str(img_dir+'/overhead_ball_3'+str(i).zfill(3)+'.jpg')))


        self.remove(all_svgs[5])
        self.remove(all_svgs[4][:22])
        self.remove(all_svgs[4][45:])
        self.remove(all_svgs[3][4:6])
        self.remove(all_svgs[19])
        self.remove(all_svgs[3][6:10])

        # Add other stuff, I think I'l want to scale down 
        # the models a bit
 
        self.frame.reorient(0, 0, 0, (0.0, -0.17, 0.0), 7.17)

        all_svgs[0].move_to([0, 2.85, 0]) #Move down title a little

        all_svgs[1].scale(0.8) #Encoder 1
        all_svgs[17].scale(0.8) #Predictor
        all_svgs[18].scale(0.8) #Encoder 2
        all_svgs[3].scale(0.9) #Arrows
        all_svgs[4].scale(0.9) #Labels

        # self.add(all_svgs[7])
        embedding_1=Tex(r'\begin{bmatrix} 0.21, \ -0.11, \ \dots, \ 0.32 \end{bmatrix}')
        embedding_1.set_color(BLUE)
        embedding_1.scale(0.65)
        embedding_1.move_to([-2.68, 1.35, 0])

        embedding_2=Tex(r'\begin{bmatrix} 0.22, \ -0.13, \ \dots, \ 0.31 \end{bmatrix}')
        embedding_2.set_color(BLUE)
        embedding_2.scale(0.65)
        embedding_2.move_to([2.6, 1.6, 0])

        embedding_3=Tex(r'\begin{bmatrix} 0.21, \ -0.10, \ \dots, \ 0.35 \end{bmatrix}')
        embedding_3.set_color(YELLOW)
        embedding_3.scale(0.65)
        embedding_3.move_to([2.6, 1.0, 0])

        self.add(embedding_1, embedding_2, embedding_3)

        # self.add(all_svgs[8])
        self.add(all_svgs[9])

        self.add(all_svgs[6][0]) #Little arrows
        self.add(all_svgs[6][1]) #Little arrows
        self.add(all_svgs[6][3]) #ittle arrows

        all_svgs[1].move_to([-2.68, -0.2, 0]) #Encoder 1
        # all_svgs[4][22:29].next_to(all_svgs[1], LEFT, buff=0.5) #Encoder 1 label
        all_svgs[4][22:29].move_to([-4.19,  -0.2,  0. ]) #Encoder 1 label
        all_svgs[3][:2].set_color(CHILL_BROWN).move_to([-2.68, -1.0, 0]) #Encoder 1 arrow in
        all_svgs[6][1].move_to([-2.68, 0.69, 0]) #Encoder 1 arrow out


        all_svgs[18].move_to([2.72, -0.2, 0]) #Encoder 2
        all_svgs[4][29:36].move_to([4.19,  -0.2,  0. ]) #Encoder 2 label
        all_svgs[3][2:4].set_color(CHILL_BROWN).move_to([2.72, -1.0, 0]) #Encoder 2 arrow in
        all_svgs[3][10:12].set_color(CHILL_BROWN).move_to([2.72, 0.55, 0])

        all_svgs[9].move_to([5.0, 1.3, 0]) #Minimze Prediction error


        all_svgs[17].move_to([-0.14, 1.35, 0]) #Predictor
        all_svgs[6][3].move_to([0.7, 1.5, 0]) #Predictor arrow out
        all_svgs[4][36:46].move_to([-0.14, 1.1, 0]) #Predictor label 
        all_svgs[6][0].move_to([-0.85, 1.34, 0]) #Predcitor Arrow in 

        self.add(all_svgs[8][:5]) #Video label
        self.add(all_svgs[8][5:14]) #Next frame label
        self.add(all_svgs[8][14:]) #Embedding label

        all_svgs[8][14:].move_to([-5.1, 1.33, 0]) #Embedding label
        all_svgs[8][5:14].move_to([2.75, -3.26, 0]) #Next frame label
        all_svgs[8][:5].move_to([-5.0, -2.52, 0]) #Bideo label




        # input_video=Group()
        # input_borders = Group()
        # spacing=0.6
        # # indices_to_show=[-40, -35, -30, -25, -20, -15, -10, -5]
        # indices_to_show=[79, 81, 83, 85, 87, 89]
        # for count, i in enumerate(indices_to_show):
        #     imgs[i].move_to([0, 0, -spacing*(len(indices_to_show)-1)+spacing*count])
        #     imgs[i].set_opacity(0.5)
        #     border = SurroundingRectangle(imgs[i], color=CHILL_BROWN, buff=0)
        #     border.set_stroke(width=2, opacity=0.0)
        #     input_video.add(imgs[i])
        #     # border.set_opacity(0.0)
        #     input_borders.add(border)
        
        # input_video_and_borders=Group(input_video, input_borders)

        # self.add(input_video_and_borders)
        # imgs[89].set_opacity(1.0)
        # self.remove(imgs[89]); self.add(imgs[89])

        

        # # imgs[89].set_opacity(0.5)
        # # self.add(input_borders)
        # # self.frame.reorient(0, 0, 0, (2.78, -0.65, 0.0), 8.60)
        # # input_video_and_borders.rotate(65*DEGREES, [0, 1, 0]).rotate(35*DEGREES, [1, 0, 0])

        # # input_video_and_borders.rotate(5*DEGREES, [0, 1, 0])

        
        # target = input_video_and_borders.copy()
        # for b in target[1]:
        #     b.set_stroke(opacity=0.8)

        # # input_borders.set_stroke(opacity=0.0)
        # target.rotate(72*DEGREES, [0, 1, 0]).rotate(35*DEGREES, [1, 0, 0])
        # # input_video_and_borders[0] is input_video; [5] is imgs[89] (last in indices_to_show)
        # target[0][5].set_opacity(0.5)



        self.wait()




        self.wait(20)
        self.embed()















