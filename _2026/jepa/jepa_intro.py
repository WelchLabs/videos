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

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/graphics/jepa_intro_to_manim/')
img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/graphics/img_pairs'


class Intro(InteractiveScene):
    def construct(self):

        svgs_to_skip=[0, 1, 2]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])

        cat_img=ImageMobject(img_dir+'/img_pairs-03.png')

        
        self.frame.reorient(0, 0, 0, (0.53, 1.33, 0.0), 5.77)

        all_svgs[2].move_to(all_svgs[4][0])
        all_svgs[2].shift([0.02, 0, 0])

        all_svgs[0][1].scale(0.8) #y
        all_svgs[0][1].next_to(all_svgs[4][-1], RIGHT, buff=0.45)
        all_svgs[0][1].shift([0, -0.06, 0])


        all_svgs[0][0].scale(0.8) #x
        all_svgs[0][0].next_to(all_svgs[4][-2], LEFT, buff=0.5)
        # all_svgs[0][9].shift([0, -0.06, 0])

        self.wait()
        self.play(Write(all_svgs[4][0]), #Berx
                  Write(all_svgs[4][-2:]), #Arows
                  Write(all_svgs[2]), #Model
                  run_time=3)
        self.wait()
        self.add(all_svgs[0][1]) #y
        self.wait()
        self.add(all_svgs[0][0]) #x


        self.wait()
        self.remove(all_svgs[2])
        self.play(ReplacementTransform(all_svgs[0][0], all_svgs[3][0]), #X
                  # ReplacementTransform(all_svgs[0][1], all_svgs[3][1]), #y
                  FadeIn(all_svgs[4][1:-2]),
                  Write(all_svgs[5][:-5]),
                  self.frame.animate.reorient(0, 0, 0, (-0.35, 1.35, 0.0), 6.07),
                  run_time=4)

        # self.add(all_svgs[5][:-5])



        # self.add(all_svgs[4][0]) #Berx
        # self.add(all_svgs[4][-2:]) #Arrows

        # self.add(all_svgs[2])



        # self.play(Write(all_svgs[1]), 
        #           Write(all_svgs[2]),
        #           run_time=3)

        # self.wait()
        # self.add(all_svgs[0][1]) #y
        # self.wait()
        # self.add(all_svgs[0][0]) #x

        # self.add(all_svgs[1][0])





        self.wait()



        self.wait(20)
        self.embed()

