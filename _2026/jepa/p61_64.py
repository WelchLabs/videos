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

svg_dir=Path(welch_path('jepa/graphics/p61_64/'))
# img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/graphics/img_pairs'


class P60_64(WelchScene):
    def construct(self):

        svgs_to_skip=[0, 1]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=gx.Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=gx.SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])

        img_1=image_at_height(welch_path('jepa/graphics/imgs_p61_64/ILSVRC2012_val_00029839.JPEG'))

        self.set_view(0, 0, 0, (2.87, 0.26, 0.0), 5.92)

        self.wait()
        self.play(gx.Create(all_svgs[4]),
                  run_time=4)

        all_svgs[9].move_to([6.6, -2.4, 0])

        rect = all_svgs[5][0]
        target = rect.copy()
        rect.stretch(0.001, dim=1, about_edge=gx.DOWN)  # squish to ~0 height, bottom pinned
        
        self.wait()
        self.add(rect)
        self.play(gx.Transform(rect, target), run_time=2.5)
        self.play(gx.Write(all_svgs[5][1:]),
                  gx.FadeIn(all_svgs[9]))


        # self.add(all_svgs[4])
        # self.add(all_svgs[5])
        self.wait()
        self.play(self.animate_view(0, 0, 0, (0.5, 0.01, 0.0), 7.24),
                  gx.Write(all_svgs[0][2:]),
                  gx.FadeOut(all_svgs[9]),
                  run_time=4)

        # self.remove(all_svgs[0][:2])

        self.wait()
        self.play(gx.Write(all_svgs[1]),
                  gx.Write(all_svgs[2]),
                  run_time=3)
        self.remove(all_svgs[0][2:]); self.add(all_svgs[0][2:]) #Occlusions


        img_1.scale(0.5)
        img_1.next_to(all_svgs[0], gx.DOWN, buff=0.1)

        self.wait()
        self.play(gx.FadeIn(all_svgs[3]),
                  gx.FadeIn(img_1),
                  gx.FadeIn(all_svgs[0][:2]),
                  run_time=3)
        
        rect = all_svgs[6][11]
        target = rect.copy()
        rect.stretch(0.001, dim=1, about_edge=gx.DOWN)

        self.wait()
        self.play(gx.Transform(rect, target),
                  gx.Write(all_svgs[6][:11]),
                  gx.Write(all_svgs[6][12:]),
                  run_time=2.5)

        # self.add(all_svgs[6][11])

        # self.add(all_svgs[7][0])

        rect = all_svgs[7][0]
        target = rect.copy()
        rect.stretch(0.001, dim=1, about_edge=gx.DOWN)

        self.wait()
        self.play(gx.Transform(rect, target),
                  gx.Write(all_svgs[7][1:]),
                  gx.Write(all_svgs[8]),
                  run_time=2.5)



        self.wait(20)



















