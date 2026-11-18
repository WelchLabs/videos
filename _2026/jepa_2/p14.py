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

#ulimit -n 4096

svg_dir=Path(welch_path('jepa_2/graphics/p14_to_manim'))
img_dir=welch_path('jepa_2/graphics')


class P14b(WelchScene):
    def construct(self):

        svgs_to_skip=[0]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=gx.Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=gx.SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])


        dog_2=image_at_height(str(img_dir+'/u9595213284_A_golden_retriever_outdoors_photographed_close-up_96fc86ba-49c9-4551-8d90-b93843923024_1.png'))
        dog_1=image_at_height(str(img_dir+'/imgs/n02099601_7101.jpg'))

        dog_1.scale(0.85)
        dog_1.next_to(all_svgs[0], gx.LEFT, buff=0.9)
        dog_1.shift([0, 0.1, 0])

        dog_2.scale(0.85)
        dog_2.next_to(all_svgs[-1], gx.RIGHT, buff=0.5)
        dog_2.shift([0, 0.3, 0])

        all_svgs[1].scale(1.15)
        all_svgs[1].shift([0, 0.1, 0])

        all_svgs[2].scale(1.15)

        all_svgs[3].shift([0.2, 0, 0])
        dog_2.shift([0.2, 0, 0])


        self.set_view(0, 0, 0, (-0.03, -0.02, 0.0))
        self.wait()

        # self.add(dog_1); self.add(all_svgs[0])
        self.play(gx.FadeIn(dog_1), gx.FadeIn(all_svgs[0]), run_time=3)
        self.wait()
        

        self.play(gx.Write(all_svgs[1]),
                  run_time=5)

        self.wait()
        self.play(gx.ReplacementTransform(all_svgs[1].copy(), all_svgs[2]),
                  run_time=4)
        self.add(all_svgs[3])

        self.play(gx.FadeIn(dog_2), run_time=2)

        # self.add(all_svgs[0])
        # self.add(dog_1, dog_2)
        # self.add(all_svgs)



        self.wait(20)
