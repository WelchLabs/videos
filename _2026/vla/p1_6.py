from manimlib import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import colorsys
# import gc
# import resource
import shutil
import tempfile

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
BLUE2='#00AEEF'
GREEN='#00a14b' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
MAGENTA='#FF00FF'
PINK='#FAD0E2'

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/vla/graphics/p1_6_to_manim/')
graphics_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/vla/graphics')

class P1_6a(InteractiveScene):
    def construct(self): 

        svgs_to_skip=[0, 1]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(3.9)
            all_svgs.add(svg_image[1:])

        coke_start_img=ImageMobject(str(graphics_dir / 'coke_can_start_frame.png'))
        coke_end_img=ImageMobject(str(graphics_dir / 'coke_can_end.png'))
        keyring_start_img=ImageMobject(str(graphics_dir / 'Keyring.mp4.00_00_22_03.Still001.png'))
        room_end_img=ImageMobject(str(graphics_dir / 'processed_5xspeed_maybe_best_fully_autonomous_bedroom_lowres.mp4.00_00_44_02.Still001.png'))


        self.frame.reorient(0, 0, 0, (-3.14, 1.79, 0.0), 3.60)
        
        all_svgs[0][1:].set_opacity(0.0)
        coke_start_img.scale(0.745)
        coke_start_img.move_to([-3.145, 1.83, 0])
        coke_end_img.scale(0.745)
        coke_end_img.move_to([-3.145, 1.83, 0])

        manual_box = RoundedRectangle(
            width=all_svgs[0][0].get_width(),
            height=all_svgs[0][0].get_height(),
            corner_radius=0.025,
            stroke_color=CHILL_BROWN,
            stroke_width=4,
            fill_opacity=0,
        )
        manual_box.move_to(all_svgs[0][0].get_center())


        self.wait()
        self.play(FadeIn(coke_start_img),
                  # Write(all_svgs[0][0]),
                  ShowCreation(manual_box),
                  all_svgs[0][1:].animate.set_opacity(0.5),
                  run_time=6)


        self.wait()
        self.play(Write(all_svgs[1]), 
                  Write(all_svgs[2]),
                  self.frame.animate.reorient(0, 0, 0, (-1.36, 1.84, 0.0), 5.51),
                  run_time=5)

        self.wait()
        self.remove(coke_start_img)
        self.add(coke_end_img)
        self.remove(manual_box); self.add(manual_box)
        self.remove(all_svgs[1]); self.add(all_svgs[1])


        quote_group=Group(all_svgs[3], all_svgs[4], all_svgs[5])
        quote_group.scale(1.15)
        quote_group.shift([0, 0.3, 0])
        quote_group[1:].shift([0, 0.2, 0])


        self.wait()
        self.play(Write(all_svgs[3]),
                  self.frame.animate.reorient(0, 0, 0, (-1.24, 0.78, 0.0), 5.74),
                  run_time=6)
        self.play(FadeIn(quote_group[1:]), run_time=2)




        self.wait()













        self.wait(20)
        self.embed()