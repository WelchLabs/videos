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

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/vla/graphics/to_manim/')
hacking_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/vla/hackin')

class P31_61_1(InteractiveScene):
    def construct(self): 
        '''
        Ok not sure how I want to break stuff up just yet - lets start hacking
        and see where we end up. 
        '''

        svgs_to_skip=[0, 2]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(3.9)
            all_svgs.add(svg_image[1:]) #Thowout background

        

        #P31
        final_time_series=ImageMobject(str(hacking_dir/'p31/000/pred_tall/299.png'))
        final_time_series.scale(0.7)
        final_time_series.move_to([2.3, -2.1, 0])

        legend=ImageMobject(str(hacking_dir/'p31/legend_2.png'))
        legend.scale(0.6)
        legend.move_to([6.1, -2.1, 0])
        legend.set_opacity(0.8)

        final_image_overhead=ImageMobject(str(hacking_dir/'p31/000/high/299.jpg'))
        final_image_overhead.scale(0.78)
        final_image_overhead.move_to([-4.55, 1.83, 0])

        final_image_left=ImageMobject(str(hacking_dir/'p31/000/left_wrist/299.jpg'))
        final_image_left.scale(0.78)
        final_image_left.move_to([-0.15, 1.83, 0])

        final_image_right=ImageMobject(str(hacking_dir/'p31/000/right_wrist/299.jpg'))
        final_image_right.scale(0.78)
        final_image_right.move_to([4.21, 1.83, 0])

        prompt=Text('"Uncap the pen"', font="Myriad Pro", weight='bold', font_size=28)
        prompt.move_to([-5.6, -2.08, 0])

        pi0_box = RoundedRectangle(
            width=1.85,
            height=1.55,
            corner_radius=0.2,
            stroke_color=FRESH_TAN,
            stroke_width=2,
            fill_opacity=0,
        )
        pi0_box.move_to([-2.76, -2.1, 0])       

        pi0_logo=Tex(r'\pi_0', font_size=60)
        pi0_logo.set_color(FRESH_TAN)
        pi0_logo.move_to(pi0_box)
        pi0_logo.shift([0, -0.05, 0])


        self.add(final_time_series, legend, final_image_overhead, final_image_left, final_image_right)
        self.add(prompt, pi0_box, pi0_logo)

        self.add(all_svgs[0])



        self.wait()


        self.wait(20)
        self.embed()














