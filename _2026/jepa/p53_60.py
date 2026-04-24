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


        svgs_to_skip=[0, 1, 2, 3, 4]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])

        

        embedding_network_1=all_svgs[1]
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


        # self.add(all_svgs[2])
        all_svgs[2].move_to([-3.01, 1.7, 0])
        all_svgs[3].next_to(all_svgs[2], UP, buff=0.2)
        all_svgs[4].move_to([-4.5, 0.65, 0]) 

        self.wait()
        self.play(FadeIn(all_svgs[2]),
                  embedding_network_1[75].animate.set_color(YELLOW),
                  embedding_network_1[83].animate.set_color(YELLOW),
                  embedding_network_1[87].animate.set_color(YELLOW),
                )

        self.wait()
        self.play(Write(all_svgs[3]), run_time=3)

        self.wait()
        self.play(Write(all_svgs[4]), run_time=3)

        

        image_border_group_0=Group(imgs[0], border_0)
        image_border_group_1=Group(imgs[1], border_1)

        embedding_network_1a=embedding_network_1.copy()
        embedding_network_1a[75].set_color(CHILL_BROWN)
        embedding_network_1a[83].set_color(CHILL_BROWN)
        # embedding_network_1a[87].set_color(CHILL_BROWN)

        embedding_network_1a.rotate(-90*DEGREES, [0, 0, 1])
        embedding_network_1a.move_to([-3, 1.5, 0])
        embedding_network_1a.scale(0.85)

        embedding_network_2b=embedding_network_2.copy()
        embedding_network_2b.rotate(-90*DEGREES, [0, 0, 1])
        embedding_network_2b.move_to([-3, -1.5, 0])
        embedding_network_2b.scale(0.85)

        self.wait()
        self.remove(all_svgs[4], all_svgs[3], all_svgs[2])
        self.play(
                  #FadeOut(all_svgs[4]),
                  #FadeOut(all_svgs[3]),
                  #FadeOut(all_svgs[2]),
                  Transform(embedding_network_1, embedding_network_1a),
                  Transform(embedding_network_2, embedding_network_2b),
                  image_border_group_0.animate.move_to([-5, 1.5, 0]),
                  image_border_group_1.animate.move_to([-5, -1.5, 0]),
                  run_time=5)

        # self.remove(all_svgs[4], all_svgs[3], all_svgs[2])
        # embedding_network_1[75].set_color(CHILL_BROWN)
        # embedding_network_1[83].set_color(CHILL_BROWN)
        # embedding_network_1[87].set_color(CHILL_BROWN)

        # embedding_network_1.rotate(-90*DEGREES, [0, 0, 1])
        # embedding_network_1.move_to([-3, 1.5, 0])
        # embedding_network_1.scale(0.85)

        # embedding_network_2.rotate(-90*DEGREES, [0, 0, 1])
        # embedding_network_2.move_to([-3, -1.5, 0])
        # embedding_network_2.scale(0.85)

        # image_border_group_0.move_to([-5, 1.5, 0])
        # image_border_group_1.move_to([-5, -1.5, 0])

        # image_border_group_0.next_to(embedding_network_1, LEFT, buff=0.2)
        # image_border_group_1.next_to(embedding_network_2, LEFT, buff=0.2)

        all_svgs[8].move_to([-1.65, 0.2, 0])

        self.wait()
        self.add(all_svgs[8][0])

        axes = Axes(
            x_range=(0, 21, 5),
            y_range=(-1.2, 1.2, 0.5),
            width=7.0,
            height=2.0,
            axis_config={
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_ticks": False,
                "include_tip": True,
                "tip_config": {
                    "width": 0.15,
                    "length": 0.15,
                },
            }
        )
        axes.move_to([2.4, 1.7, 0])
        # self.add(axes)

        # Explicit but semi-random y values
        y_values = [0.9, -0.8, 0.2, 0.5, -0.3, 0.7, 0.6, 0.1, 0.4, -0.5,
                    0.75, -0.25, 0.55, 0.7, 0.3, -0.45, -0.6, -0.15, 0.8, 0.35]
        x_values = list(range(1, 21))

        # Build dots and connecting lines up front
        dots = VGroup()
        lines = VGroup()
        for i, (x, y) in enumerate(zip(x_values, y_values)):
            dot = Dot(axes.c2p(x, y), radius=0.06)
            dot.set_color(YELLOW)
            dots.add(dot)
            if i > 0:
                line = Line(
                    axes.c2p(x_values[i-1], y_values[i-1]),
                    axes.c2p(x, y),
                    stroke_width=2,
                )
                line.set_color(YELLOW)
                lines.add(line)

        self.wait()

        self.add(axes)
        self.add(dots, lines)







        self.wait(20)
        self.embed()



