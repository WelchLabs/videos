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

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/graphics/p67/')
# img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/graphics/img_pairs'


class P67(InteractiveScene):
    def construct(self):

        svgs_to_skip=[0]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])


        self.add(all_svgs[0])

        # self.add(all_svgs[1][6])

        self.wait()

        rect = all_svgs[1][6]
        target = rect.copy()
        rect.stretch(0.001, dim=1, about_edge=DOWN)

        self.wait()
        self.play(Transform(rect, target), 
                  Write(all_svgs[1][:6]),
                  Write(all_svgs[1][7:]),
                  run_time=5)


        self.wait(20)
        self.embed()