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

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics/p19_22_to_manim')
img_dir_1='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics/fish_stock_video'




import numpy as np

class P19_22_3da(InteractiveScene):
    def construct(self):

        RANDOM_SEED=5

        imgs = Group()
        for i in range(0, 72, 3):
            imgs.add(ImageMobject(str(img_dir_1 + '/fish_stock_video' + str(i).zfill(2) + '.jpg')))

        video = Group()
        borders = Group()
        masks = Group()
        spacing = 1.65
        indices_to_show = [0, 1, 2, 3]

        # --- Tube mask: pick once, reuse across frames ---
        n_px, n_py = 4, 3                  # patch grid (try 16x16 for a more ViT-like look)
        mask_ratio = 0.4                  # V-JEPA goes as high as ~0.9
        total = n_px * n_py
        n_masked = int(mask_ratio * total)
        rng = np.random.default_rng(RANDOM_SEED)     # tweak seed until the pattern feels right
        masked_ids = rng.choice(total, n_masked, replace=False)

        for count, i in enumerate(indices_to_show):
            z = -spacing * (len(indices_to_show) - 1) + spacing * count
            imgs[i].move_to([0, 0, z])

            border = SurroundingRectangle(imgs[i], color=CHILL_BROWN, buff=0)
            border.set_stroke(width=2)
            video.add(imgs[i])
            borders.add(border)

            # Patch geometry, derived from this image's actual bounds
            w, h = imgs[i].get_width(), imgs[i].get_height()
            pw, ph = w / n_px, h / n_py
            cx, cy, _ = imgs[i].get_center()
            left, bottom = cx - w / 2, cy - h / 2

            for idx in masked_ids:
                row, col = divmod(idx, n_px)
                x = left + (col + 0.5) * pw
                y = bottom + (row + 0.5) * ph
                patch = Rectangle(width=pw, height=ph)
                patch.set_fill(BLACK, opacity=1.0)
                patch.set_stroke(width=0)
                # Tiny z nudge so patches sit just in front of the image plane
                patch.move_to([x, y, z + 1e-3])
                masks.add(patch)

        video_and_borders = Group(video, borders, masks)
        video_and_borders.rotate(90 * DEGREES, [1, 0, 0])

        self.frame.reorient(-53, 59, 0, (-1.77, -1.52, -3.17), 10.06)
        self.wait()

        self.add(borders)
        self.add(video)
        self.add(masks)


        self.wait()




        self.wait(20)
        self.embed()






# class P19_22_3da(InteractiveScene):
#     def construct(self):

#         imgs=Group()
#         for i in range(0, 72, 3):
#             imgs.add(ImageMobject(str(img_dir_1+'/fish_stock_video'+str(i).zfill(2)+'.jpg'))) 

#         video=Group()
#         borders = Group()
#         spacing=1.5
#         indices_to_show=[0, 1, 2, 3]
#         for count, i in enumerate(indices_to_show):
#             imgs[i].move_to([0, 0, -spacing*(len(indices_to_show)-1)+spacing*count])
#             # imgs[i].set_opacity(0.5)
#             border = SurroundingRectangle(imgs[i], color=CHILL_BROWN, buff=0)
#             border.set_stroke(width=e) #, opacity=0.0)
#             video.add(imgs[i])
#             # border.set_opacity(0.0)
#             borders.add(border)
        
#         video_and_borders=Group(video, borders)
#         video_and_borders.rotate(90*DEGREES, [1, 0, 0])
#         # self.remove(video[-1]); self.add(video[-1])

#         self.frame.reorient(-53, 59, 0, (-1.77, -1.52, -3.17), 10.06)
#         self.wait()

#         self.add(borders)
#         self.add(video)





#         self.wait(20)
#         self.embed()


class P19_22_2d(InteractiveScene):
    def construct(self):

        svgs_to_skip=[0]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])


        self.wait()
        self.play(Write(all_svgs[0]),
                  Write(all_svgs[1]),
                  run_time=5)

        self.wait()
        self.play(Write(all_svgs[2]), run_time=4)
        self.wait()

        self.play(Write(all_svgs[3]), run_time=4)


        # self.add(all_svgs)




        self.wait(20)
        self.embed()