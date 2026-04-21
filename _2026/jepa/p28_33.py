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


class P28_33b(InteractiveScene):
    def construct(self):

        imgs=Group()
        for i in range(10, 140):
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

        self.add(all_svgs[0])

        all_svgs[0].move_to([5.5, -0.7, 0])














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