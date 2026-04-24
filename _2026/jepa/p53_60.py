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

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/graphics/p53_60/')
img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/graphics/img_pairs'




class P53_60(InteractiveScene):
    def construct(self):


        imgs=Group()
        for i in range(1,21):
            imgs.add(ImageMobject(str(img_dir+'/img_pairs-'+str(i).zfill(2)+'.png')))


        svgs_to_skip=[0, 1, 2, 3, 4, 5]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])

        

        embedding_network_1=all_svgs[0]
        embedding_network_2=embedding_network_1.copy()

        embedding_network_1.move_to([-3, -0.2, 0 ])
        embedding_network_2.move_to([3, -0.2, 0 ])

        imgs[0].scale(0.5)
        imgs[0].next_to(embedding_network_1, DOWN, buff=0.2)
        border_0 = SurroundingRectangle(imgs[0], color=CHILL_BROWN, buff=0)
        border_0.set_stroke(width=2, opacity=1.0)

        imgs[1].scale(0.5)
        imgs[1].next_to(embedding_network_2, DOWN, buff=0.2)
        border_1 = SurroundingRectangle(imgs[1], color=CHILL_BROWN, buff=0)
        border_1.set_stroke(width=2, opacity=1.0)

        self.wait()
        self.play(Write(embedding_network_1), 
                  Write(embedding_network_2),
                  FadeIn(imgs[0]),
                  FadeIn(border_0),
                  FadeIn(imgs[1]),
                  FadeIn(border_1),
                  run_time=7)

        # self.add(embedding_network_1, embedding_network_2)
        # self.add(imgs[0], imgs[1])
        # self.add(border_0, border_1)


        self.wait()


        self.wait(20)
        self.embed()



