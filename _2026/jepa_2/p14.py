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

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics/p14_to_manim')
img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics'


class P14(InteractiveScene):
    def construct(self):

        svgs_to_skip=[0]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])


        dog_2=ImageMobject(str(img_dir+'/u9595213284_A_golden_retriever_outdoors_photographed_close-up_96fc86ba-49c9-4551-8d90-b93843923024_1.png'))
        dog_1=ImageMobject(str(img_dir+'/imgs/n02099601_7101.jpg'))

        dog_1.scale(0.85)
        dog_1.next_to(all_svgs[0], LEFT, buff=0.9)
        dog_1.shift([0, 0.1, 0])

        dog_2.scale(0.85)
        dog_2.next_to(all_svgs[-1], RIGHT, buff=0.5)
        dog_2.shift([0, 0.3, 0])

        all_svgs[1].scale(1.15)
        all_svgs[1].shift([0, 0.1, 0])

        all_svgs[2].scale(1.15)

        all_svgs[3].shift([0.2, 0, 0])
        dog_2.shift([0.2, 0, 0])


        self.frame.reorient(0, 0, 0, (-0.03, -0.02, 0.0))
        self.wait()

        self.add(dog_1); self.add(all_svgs[0])
        

        self.play(Write(all_svgs[1]),
                  run_time=5)

        self.wait()
        self.play(ReplacementTransform(all_svgs[1].copy(), all_svgs[2]),
                  run_time=4)
        self.add(all_svgs[3])

        self.play(FadeIn(dog_2), run_time=2)

        # self.add(all_svgs[0])
        # self.add(dog_1, dog_2)
        # self.add(all_svgs)



        self.embed()
        self.wait(20)
