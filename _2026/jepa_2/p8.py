from manimlib import *
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
svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics/p8_to_manim')
img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics'
hackin_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/hackin'



class p75b(InteractiveScene):
    def construct(self):


        svgs_to_skip=[]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])

        v_jepa_cats=ImageMobject(str(img_dir+'/P19_22_3db.mp4.00_00_10_29.Still001 copy.jpg'))
        v_jepa_cats_2=ImageMobject(str(img_dir+'/P19_22_3db.mp4.00_00_13_27.Still002 copy.jpg'))
        mushroom=ImageMobject(str(img_dir+'/AdobeStock_93352397.jpeg'))
        pusht_1=ImageMobject(str(img_dir+'/sample_rollout/frame_0000.png'))
        pusht_2=ImageMobject(str(img_dir+'/sample_rollout/frame_0015.png'))
        pusht_3=ImageMobject(str(img_dir+'/sample_rollout/frame_0030.png'))
        pusht_4=ImageMobject(str(img_dir+'/sample_rollout/frame_0045.png'))
        pusht_5=ImageMobject(str(img_dir+'/sample_rollout/frame_0065.png'))
        pusht_6=ImageMobject(str(img_dir+'/sample_rollout/frame_0080.png'))
        pusht_group=Group(pusht_1, pusht_2, pusht_3, pusht_4, pusht_5, pusht_6)

        # self.wait()

        #Get images positioned then work on animation. 
        all_svgs[6].set_opacity(0.4)
        

        v_jepa_cats.scale(0.25)
        v_jepa_cats.move_to([-5.9, 0.1, 0])

        v_jepa_cats_2.scale(0.25)
        v_jepa_cats_2.move_to([-4.1, 0.1, 0])

        mushroom.scale(0.22)
        mushroom.move_to([3.95, 0.0, 0])

        pusht_group.scale(0.25)
        pusht_1.move_to([-3.7, -3.1, 0])
        pusht_2.move_to([-2.2, -3.1, 0])
        pusht_3.move_to([-0.6, -3.1, 0])
        pusht_4.move_to([0.95, -3.1, 0])
        pusht_5.move_to([2.5, -3.1, 0])
        pusht_6.move_to([4.05, -3.1, 0])


        # self.add(v_jepa_cats)
        # self.add(v_jepa_cats_2)
        # self.add(mushroom)
        # self.add(pusht_group)
        # self.add(all_svgs)

        self.wait()




        self.wait(20)
        self.embed()










