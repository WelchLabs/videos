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

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics/p34_48_to_manim')
img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics'



class p34_48(InteractiveScene):
    def construct(self):
        '''Might end up needing to break this scene up, we'll see'''

        svgs_to_skip=[]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])


        mushroom=ImageMobject(str(img_dir+'/Omphalotus_olearius_Mallorca.jpg'))
        two_cats=ImageMobject(str(img_dir+'/AdobeStock_240682884.jpeg'))
        gqa_image=ImageMobject(str(img_dir+'/gqa_img.jpg'))
        
        #P34
        self.frame.reorient(0, 0, 0, (-0.32, 0.29, 0.0), 6.71)
        self.wait()

        self.add(all_svgs[0], all_svgs[1], all_svgs[2])

        all_svgs[4][0].set_opacity(0.2)
        all_svgs[3][0].set_opacity(0.15)

        self.wait()
        self.play(Write(all_svgs[4]), run_time=3)
        self.wait()

        self.play(Write(all_svgs[3]), run_time=3)
        self.wait()

        #P35 question, little zoom out to get read
        self.play(self.frame.animate.reorient(0, 0, 0, (-0.32, 0.21, 0.0), 7.64), 
                  FadeOut(all_svgs[3]), 
                  FadeOut(all_svgs[4]),
                  run_time=4)

        

        all_svgs[5].move_to([-0.2, -0.25, 0])
        mushroom.scale(0.5)
        mushroom.move_to([-5.4, 0.6, 0])

        #P36
        self.wait()
        self.play(Write(all_svgs[5][17:19]),
                  FadeIn(mushroom),
                  run_time=3)
        self.wait()
        self.play(Write(all_svgs[5][:17]),
                  Write(all_svgs[5][19:-16]),
                  run_time=3)
        self.wait()
        self.play(Write(all_svgs[5][-16:]),
                  run_time=3)


        self.wait()









        self.embed()
        self.wait(20)
