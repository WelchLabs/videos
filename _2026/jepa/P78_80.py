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

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/graphics/p78_80')
img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/hacking/v_jepa_ac_1'


class P78_80(InteractiveScene):
    def construct(self):

        svgs_to_skip=[0, 1, 2]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])


        self.frame.reorient(0, 0, 0, (-0.1, 0.75, 0.0), 6.92)
        self.add(all_svgs[0], all_svgs[1], all_svgs[2], all_svgs[3],
                 all_svgs[4], all_svgs[5])

        self.wait()
        self.play(Write(all_svgs[6]), run_time=5)


        # Ok getting close here, goog next step is probably to go
        # figure out the matpotlib graph stuff
        # Hmm do I want to actually tackon/render in premier I guess?
        # I do think that would be easier. 
        # Ok improtant thing for now then is figuring out a reasonable
        # set of curves in matplotlib!



        self.embed()
        self.wait(20)

