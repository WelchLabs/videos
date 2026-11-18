from welch_paths import welch_path
import manimgx as gx
from welch_scene import WelchScene, image_at_height
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

svg_dir=Path(welch_path('jepa/graphics/jepa_intro_to_manim/'))
img_dir=welch_path('jepa/graphics/img_pairs')


class IntroC(WelchScene):
    def construct(self):

        svgs_to_skip=[0, 1, 2]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=gx.Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=gx.SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])

        cat_img=image_at_height(img_dir+'/img_pairs-03.png')

        
        self.set_view(0, 0, 0, (0.53, 1.33, 0.0), 5.77)

        all_svgs[2].move_to(all_svgs[4][0])
        all_svgs[2].shift([0.02, 0, 0])

        all_svgs[0][1].scale(0.8) #y
        all_svgs[0][1].next_to(all_svgs[4][-1], gx.RIGHT, buff=0.45)
        all_svgs[0][1].shift([0, -0.06, 0])


        all_svgs[0][0].scale(0.8) #x
        all_svgs[0][0].next_to(all_svgs[4][-2], gx.LEFT, buff=0.5)
        # all_svgs[0][9].shift([0, -0.06, 0])

        self.wait()
        self.play(gx.Write(all_svgs[4][0]), #Berx
                  gx.Write(all_svgs[4][-2:]), #Arows
                  gx.Write(all_svgs[2]), #Model
                  run_time=3)
        self.wait()
        self.add(all_svgs[0][1]) #y
        self.wait()
        self.add(all_svgs[0][0]) #x


        self.wait()
        self.remove(all_svgs[2])
        self.play(gx.ReplacementTransform(all_svgs[0][0], all_svgs[3][0]), #X
                  # ReplacementTransform(all_svgs[0][1], all_svgs[3][1]), #y
                  gx.FadeIn(all_svgs[4][1:-2]),
                  gx.Write(all_svgs[5][5:]),
                  self.animate_view(0, 0, 0, (-0.32, 1.35, 0.0), 6.20),
                  run_time=4)

        self.wait()
        self.play(gx.ReplacementTransform(all_svgs[0][1], all_svgs[3][1]),
                  gx.Write(all_svgs[5][:5]),
                  run_time=3)


        self.wait()
        self.play(gx.Write(all_svgs[7]),
                  self.animate_view(0, 0, 0, (-0.24, -0.02, 0.0), 6.87),
                  run_time=4)

        cat_img.scale(0.52)
        cat_img.move_to([-3.4, -1.3, 0])
        self.play(gx.FadeIn(cat_img), gx.FadeIn(all_svgs[6][0]))

        self.play(gx.Write(all_svgs[8]),
                  gx.FadeIn(all_svgs[6][1]),
                  run_time=2)



        self.wait()
        self.remove(all_svgs[3], all_svgs[4], all_svgs[5], all_svgs[7], cat_img, all_svgs[8])
        self.play(gx.Write(all_svgs[9]),
                  self.animate_view(0, 0, 0, (-0.07, 0.23, 0.0), 7.27),
                  gx.ReplacementTransform(all_svgs[6][0], all_svgs[10][0]),
                  gx.ReplacementTransform(all_svgs[6][1], all_svgs[10][1]),
                  run_time=4)

        self.wait()
        self.play(gx.Write(all_svgs[11]),run_time=3)

        self.wait()
        self.play(gx.Write(all_svgs[12]),run_time=3)

        self.wait()
        self.play(gx.Write(all_svgs[13]),run_time=3)

        self.wait()
        self.play(self.animate_view(0, 0, 0, (0.01, 0.3, 0.0), 7.57), run_time=4)



        # self.add(all_svgs[6])

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




        self.wait(20)

