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

        input_video=Group()
        input_borders = Group()
        spacing=0.6
        indices_to_show=[0, 1, 2, 3, 4]
        for count, i in enumerate(indices_to_show):
            imgs[i].move_to([0, 0, -spacing*(len(indices_to_show)-1)+spacing*count])
            imgs[i].set_opacity(0.5)
            border = SurroundingRectangle(imgs[i], color=CHILL_BROWN, buff=0)
            border.set_stroke(width=2, opacity=0.5)
            input_video.add(imgs[i])
            input_borders.add(border)
        
        input_video_and_borders=Group(input_video, input_borders)

        input_video_and_borders.scale(0.38)
        input_video_and_borders.move_to([-2.5, -2.1, 0])
        input_video_and_borders.rotate(70*DEGREES, [0, 1, 0])

        
        imgs[5].scale(0.45)
        imgs[5].move_to([2.72, -2.15, 0])
        border_next_frame = SurroundingRectangle(imgs[5], color=CHILL_BROWN, buff=0)
        border_next_frame.set_stroke(width=2, opacity=0.5)


        group_to_remove=Group(all_svgs[5],
                              all_svgs[4][:22],
                              all_svgs[4][45:],
                              # all_svgs[4][:22],#Do all labels and bring back selectively
                              # all_svgs[4][45:],
                              # all_svgs[4],
                              # all_svgs[3][4:6], #Let's do all arrows and bring back selectively
                              all_svgs[3],
                              all_svgs[19],
                              all_svgs[3][6:10])
        

        self.wait()
        self.play(FadeOut(group_to_remove), 
                 self.frame.animate.reorient(0, 0, 0, (0.0, -0.17, 0.0), 7.17),
                 all_svgs[0].animate.move_to([0, 2.85, 0]), #Move down title a little
                 all_svgs[1].animate.scale(0.8).move_to([-2.68, -0.2, 0]), #Encoder 1
                 all_svgs[17].animate.scale(0.8).move_to([-0.14, 1.35, 0]), #Predictor
                 all_svgs[18].animate.scale(0.8).move_to([2.72, -0.2, 0]), #Encoder 2
                 all_svgs[4][22:29].animate.scale(0.9).move_to([-4.19,  -0.2,  0. ]), #Encoder 1 label
                 all_svgs[4][36:45].animate.scale(0.9).move_to([-0.14, 2.2, 0]), 
                 all_svgs[4][29:36].animate.scale(0.9).move_to([4.19,  -0.2,  0. ]), #Encoder 2 label
                 run_time=2
                )


        all_svgs[3].scale(0.9) #Arrows
        all_svgs[3][:2].set_color(CHILL_BROWN).move_to([-2.68, -1.0, 0]) #Encoder 1 arrow in
        all_svgs[3][2:4].set_color(CHILL_BROWN).move_to([2.72, -1.0, 0]) #Encoder 2 arrow in
        all_svgs[3][10:12].set_color(CHILL_BROWN).move_to([2.72, 0.55, 0])

        all_svgs[6][1].move_to([-2.68, 0.69, 0]), #Encoder 1 arrow out
        all_svgs[6][3].move_to([0.7, 1.5, 0]), #Predictor arrow out
        all_svgs[6][0].move_to([-0.85, 1.34, 0]), #Predcitor Arrow in 

        all_svgs[8][14:].move_to([-5.1, 1.33, 0]) #Embedding label
        all_svgs[8][5:14].move_to([2.8, -3.26, 0]) #Next frame label
        all_svgs[8][:5].move_to([-5.0, -2.52, 0]) #Video label

        all_svgs[9].move_to([5.0, 1.3, 0]), #Minimze Prediction error


        self.wait()
        self.play(FadeIn(input_video_and_borders),
                  FadeIn(border_next_frame),
                  FadeIn(imgs[5]),
                  FadeIn(all_svgs[3][:2]), 
                  FadeIn(all_svgs[3][2:4]),
                  run_time=3)
        self.add(all_svgs[8][:5], #Video label
                  all_svgs[8][5:14]) #Next frame label


        self.wait()
        self.play(Write(embedding_1),
                  Write(embedding_3),
                  Write(all_svgs[3][10:12]),
                  Write(all_svgs[6][1]),
                  FadeIn(all_svgs[8][14:]),
                  run_time=3)

        self.wait()
        self.play(Write(embedding_2),
                  Write(all_svgs[6][0]),
                  Write(all_svgs[6][3]),
                  # Write(all_svgs[9]),
                  # Write(all_svgs[8][14:]),
                  run_time=4
                  )
        self.add(all_svgs[9])

        '''
        Ok Claude, could use some help here. At this point I want to animate this 
        scene in the following way. I want to step through all imgs, and each step 
        replacing the current img[5] with the latest image, moving img[5] to where
        img[4] is, img[4] to where img[3] is and so on. While doing that I also 
        want to make updates the the 3 numbers in each the three embedding vectors. 

        To keep this realistic, I don't want the numbers to change too quickly. I think 
        the ideal thing would be to send all three vectors on the same random walk, and 
        add a little noise to each vector after the main walk step to the resultin vectors
        are not identical. 
        
        '''













        # self.add(all_svgs[6][0], #Little arrows
                 # all_svgs[6][1], #Little arrows
                 # all_svgs[6][3], #ittle arrows
                 # embedding_1, 
                 # embedding_2, 
                 # embedding_3,
                 # all_svgs[9],
                 # all_svgs[8][:5], #Video label
                 # all_svgs[8][5:14], #Next frame label
                 # all_svgs[8][14:],
                 # all_svgs[3][:2],
                 # all_svgs[3][2:4],
                 # all_svgs[3][10:12],
                 # )


        # self.wait()



        # self.add(all_svgs[8])
        # self.add(all_svgs[9])

        # all_svgs[1].move_to([-2.68, -0.2, 0]) #Encoder 1
        # all_svgs[4][22:29].next_to(all_svgs[1], LEFT, buff=0.5) #Encoder 1 label
        #all_svgs[4][22:29].move_to([-4.19,  -0.2,  0. ]) #Encoder 1 label
        #all_svgs[3][:2].set_color(CHILL_BROWN).move_to([-2.68, -1.0, 0]) #Encoder 1 arrow in
        # all_svgs[6][1].move_to([-2.68, 0.69, 0]) #Encoder 1 arrow out


        # all_svgs[18].move_to([2.72, -0.2, 0]) #Encoder 2
        # all_svgs[4][29:36].move_to([4.19,  -0.2,  0. ]) #Encoder 2 label
        # all_svgs[3][2:4].set_color(CHILL_BROWN).move_to([2.72, -1.0, 0]) #Encoder 2 arrow in
        # all_svgs[3][10:12].set_color(CHILL_BROWN).move_to([2.72, 0.55, 0])

        # all_svgs[9].move_to([5.0, 1.3, 0]) #Minimze Prediction error


        # all_svgs[17].move_to([-0.14, 1.35, 0]) #Predictor
        # all_svgs[6][3].move_to([0.7, 1.5, 0]) #Predictor arrow out
        # all_svgs[4][36:46].move_to([-0.14, 1.1, 0]) #Predictor label 
        # all_svgs[6][0].move_to([-0.85, 1.34, 0]) #Predcitor Arrow in 

        # self.add(all_svgs[8][:5]) #Video label
        # self.add(all_svgs[8][5:14]) #Next frame label
        # self.add(all_svgs[8][14:]) #Embedding label

        # all_svgs[8][14:].move_to([-5.1, 1.33, 0]) #Embedding label
        # all_svgs[8][5:14].move_to([2.75, -3.26, 0]) #Next frame label
        # all_svgs[8][:5].move_to([-5.0, -2.52, 0]) #Video label








        self.wait()




        self.wait(20)
        self.embed()















