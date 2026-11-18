from welch_paths import welch_path
import manimgx as gx
from welch_scene import WelchScene, image_at_height
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import json

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

svg_dir=Path(welch_path('jepa_2/graphics/p84_85_to_manim'))
img_dir=welch_path('jepa_2/graphics')
hackin_dir=welch_path('jepa_2/hackin')

class p85(WelchScene):
    def construct(self):

        svgs_to_skip=[]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=gx.Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=gx.SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])


        self.wait()
        self.play(gx.LaggedStart([gx.Write(all_svgs[5]), gx.Write(all_svgs[10])], lag_ratio=0.8), run_time=4)

        self.wait()
        self.play(gx.ReplacementTransform(all_svgs[5], all_svgs[11]),
                  gx.ReplacementTransform(all_svgs[10], all_svgs[12]), run_time=3)
        self.play(
                  gx.Write(all_svgs[6]),
                  run_time=10
                  )

        self.wait()
        self.play(gx.Write(all_svgs[7]), run_time=6)

        self.wait()
        self.play(gx.Write(all_svgs[8]), run_time=6)

        self.wait()
        self.play(gx.Write(all_svgs[9]), run_time=6)


        self.wait(20)



class p84(WelchScene):
    def construct(self):

        svgs_to_skip=[]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=gx.Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=gx.SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])

        start_im=image_at_height(hackin_dir+'/p71/ep_10738/real/frame_0000.png')
        subgoal_im=image_at_height(hackin_dir+'/p71/ep_10738/real/frame_0035.png')
        goal_im=image_at_height(hackin_dir+'/p71/ep_10738/real/frame_0095.png')


        start_im.scale(0.5)
        start_im.move_to([-5.35, -2.5, 0])

        subgoal_im.scale(0.5)
        subgoal_im.move_to([2.0, -0.2, 0])

        goal_im.scale(0.5)
        goal_im.move_to([5.4, -2.45, 0])

        self.add(start_im, goal_im, subgoal_im)
        self.add(all_svgs[0], all_svgs[1], all_svgs[2], all_svgs[3], all_svgs[4])

        self.set_view(0, 0, 0, (-4.64, 2.58, 0.0), 2.71)
        self.wait()
        self.play(self.animate_view(0, 0, 0, (0, 0, 0), 8), run_time=8)


        # self.wait()
        # self.play(FadeIn(all_svgs[0]),
        #           FadeIn(all_svgs[2]),
        #           run_time=3)

        # self.wait()
        # self.play(FadeIn(all_svgs[3]),
        #       FadeIn(all_svgs[4]),
        #       run_time=3)  
        # self.add(subgoal_im) 
        # self.add(all_svgs[4])











        self.wait(20)
