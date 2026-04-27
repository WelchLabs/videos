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

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/graphics/p75_80_manim/')
# img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/graphics/img_pairs'


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





        self.wait()


        self.wait(20)
        self.embed()