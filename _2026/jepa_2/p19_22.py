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
img_dir_1_masked='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics/fish_stock_video_masked'



class P19_22_3da(InteractiveScene):
    def construct(self):

        imgs=Group()
        imgs_masked=Group()
        for i in range(0, 72):
            imgs.add(ImageMobject(str(img_dir_1+'/fish_stock_video'+str(i).zfill(2)+'.jpg'))) 
            imgs_masked.add(ImageMobject(str(img_dir_1_masked+'/fish_stock_video'+str(i).zfill(2)+'.png'))) 

        imgs_copy=imgs.copy()
        imgs_copy.rotate(90*DEGREES, [1, 0, 0])

        video=Group()
        video_masked=Group()
        borders = Group()
        spacing=1.2
        indices_to_show=[0, 10, 20, 30, 40]
        for count, i in enumerate(indices_to_show):
            imgs[i].move_to([0, 0, -spacing*(len(indices_to_show)-1)+spacing*count])
            imgs_masked[i].move_to([0, 0, -spacing*(len(indices_to_show)-1)+spacing*count])
            # imgs[i].set_opacity(0.5)
            border = SurroundingRectangle(imgs[i], color=CHILL_BROWN, buff=0)
            border.set_stroke(width=3) #, opacity=0.0)
            video.add(imgs[i])
            video_masked.add(imgs_masked[i])
            # border.set_opacity(0.0)
            borders.add(border)
        
        video_and_borders=Group(video_masked, video, borders)
        video_and_borders.rotate(90*DEGREES, [1, 0, 0])
        # self.remove(video[-1]); self.add(video[-1])

        # self.frame.reorient(0, 87, 0, (0.07, 0.54, -2.47), 6.95)
        self.frame.reorient(0, 90, 0, (0.01, 0.54, -2.4), 6.95)

        self.add(borders[-1])
        self.wait()
        #Play video 
        for i in range(0, indices_to_show[-1]+1):
            if i>0: self.remove(imgs_copy[i-1])
            imgs_copy[i].move_to(video[-1])
            self.remove(borders[-1])
            self.add(imgs_copy[i])
            self.add(borders[-1])
            self.wait(0.1)

        self.wait()

        
        self.add(video_masked)
        self.add(borders)
        self.add(video_masked[-1])
        self.remove(imgs_copy[i])

        self.play( 
                  # self.frame.animate.reorient(-50, 62, 0, (0.39, -0.12, -3.32), 10.36),
                  self.frame.animate.reorient(36, 65, 0, (0.57, -0.32, -3.12), 8.53),
                  run_time=6)
        self.wait()

        # Ok not perfect but I think workable
        # Now in Premiere I'll split into two and move to the right spots in the 2d deal. 

        #Here's the non-corrupted version: 
        self.remove(video_masked)
        self.add(video)
        self.wait()


        self.wait(20)
        self.embed()

img_dir_2='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics/cat_stock_video'
img_dir_2_masked='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics/cat_stock_video_masked'


class P19_22_3db(InteractiveScene):
    def construct(self):

        imgs=Group()
        imgs_masked=Group()
        for i in range(0, 28):
            imgs.add(ImageMobject(str(img_dir_2+'/cat_stock_video'+str(i).zfill(2)+'.jpg'))) 
            imgs_masked.add(ImageMobject(str(img_dir_2_masked+'/cat_stock_video'+str(i).zfill(2)+'.png'))) 

        imgs_copy=imgs.copy()
        imgs_copy.rotate(90*DEGREES, [1, 0, 0])

        video=Group()
        video_masked=Group()
        borders = Group()
        spacing=1.2
        indices_to_show=[5, 10, 15, 20, 25]
        for count, i in enumerate(indices_to_show):
            imgs[i].move_to([0, 0, -spacing*(len(indices_to_show)-1)+spacing*count])
            imgs_masked[i].move_to([0, 0, -spacing*(len(indices_to_show)-1)+spacing*count])
            # imgs[i].set_opacity(0.5)
            border = SurroundingRectangle(imgs[i], color=CHILL_BROWN, buff=0)
            border.set_stroke(width=3) #, opacity=0.0)
            video.add(imgs[i])
            video_masked.add(imgs_masked[i])
            # border.set_opacity(0.0)
            borders.add(border)
        
        video_and_borders=Group(video_masked, video, borders)
        video_and_borders.rotate(90*DEGREES, [1, 0, 0])
        # self.remove(video[-1]); self.add(video[-1])

        # self.frame.reorient(0, 87, 0, (0.07, 0.54, -2.47), 6.95)
        self.frame.reorient(0, 90, 0, (0.01, 0.54, -2.4), 6.95)

        self.add(borders[-1])
        self.wait()
        #Play video 
        for i in range(0, indices_to_show[-1]+1):
            if i>0: self.remove(imgs_copy[i-1])
            imgs_copy[i].move_to(video[-1])
            self.remove(borders[-1])
            self.add(imgs_copy[i])
            self.add(borders[-1])
            self.wait(0.1)

        self.wait()

        
        self.add(video_masked)
        self.add(borders)
        self.add(video_masked[-1])
        self.remove(imgs_copy[i])

        self.play( 
                  # self.frame.animate.reorient(-50, 62, 0, (0.39, -0.12, -3.32), 10.36),
                  self.frame.animate.reorient(36, 65, 0, (0.57, -0.32, -3.12), 8.53),
                  run_time=6)
        self.wait()

        # Ok not perfect but I think workable
        # Now in Premiere I'll split into two and move to the right spots in the 2d deal. 

        #Here's the non-corrupted version: 
        self.remove(video_masked)
        self.add(video)
        self.wait()


        self.wait(20)
        self.embed()


img_dir_3='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics/a_roll_clips'
img_dir_3_masked='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics/a_roll_clips_masked'


class P19_22_3dc(InteractiveScene):
    def construct(self):

        imgs=Group()
        imgs_masked=Group()
        for i in range(0, 300):
            imgs.add(ImageMobject(str(img_dir_3+'/a_roll_clips_from_jepa_1'+str(i).zfill(3)+'.jpg'))) 
            imgs_masked.add(ImageMobject(str(img_dir_3_masked+'/a_roll_clips_from_jepa_1'+str(i).zfill(3)+'.png'))) 

        imgs_copy=imgs.copy()
        imgs_copy.rotate(90*DEGREES, [1, 0, 0])

        video=Group()
        video_masked=Group()
        borders = Group()
        spacing=1.2
        indices_to_show=[0, 10, 20, 30, 40]
        for count, i in enumerate(indices_to_show):
            imgs[i].move_to([0, 0, -spacing*(len(indices_to_show)-1)+spacing*count])
            imgs_masked[i].move_to([0, 0, -spacing*(len(indices_to_show)-1)+spacing*count])
            # imgs[i].set_opacity(0.5)
            border = SurroundingRectangle(imgs[i], color=CHILL_BROWN, buff=0)
            border.set_stroke(width=3) #, opacity=0.0)
            video.add(imgs[i])
            video_masked.add(imgs_masked[i])
            # border.set_opacity(0.0)
            borders.add(border)
        
        video_and_borders=Group(video_masked, video, borders)
        video_and_borders.rotate(90*DEGREES, [1, 0, 0])
        # self.remove(video[-1]); self.add(video[-1])

        # self.frame.reorient(0, 87, 0, (0.07, 0.54, -2.47), 6.95)
        self.frame.reorient(0, 90, 0, (0.01, 0.54, -2.4), 6.95)

        self.add(borders[-1])
        self.wait()
        #Play video 
        for i in range(0, indices_to_show[-1]+1):
            if i>0: self.remove(imgs_copy[i-1])
            imgs_copy[i].move_to(video[-1])
            self.remove(borders[-1])
            self.add(imgs_copy[i])
            self.add(borders[-1])
            self.wait(0.1)

        self.wait()

        
        self.add(video_masked)
        self.add(borders)
        self.add(video_masked[-1])
        self.remove(imgs_copy[i])

        self.play( 
                  # self.frame.animate.reorient(-50, 62, 0, (0.39, -0.12, -3.32), 10.36),
                  self.frame.animate.reorient(36, 65, 0, (0.57, -0.32, -3.12), 8.53),
                  run_time=6)
        self.wait()

        # Ok not perfect but I think workable
        # Now in Premiere I'll split into two and move to the right spots in the 2d deal. 

        #Here's the non-corrupted version: 
        self.remove(video_masked)
        self.add(video)
        self.wait()


        self.wait(20)
        self.embed()




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





# class P19_22_3da(InteractiveScene):
#     def construct(self):

#         RANDOM_SEED=5

#         imgs = Group()
#         for i in range(0, 72, 3):
#             imgs.add(ImageMobject(str(img_dir_1 + '/fish_stock_video' + str(i).zfill(2) + '.jpg')))

#         video = Group()
#         borders = Group()
#         masks = Group()
#         spacing = 1.65
#         indices_to_show = [0, 1, 2, 3]

#         # --- Tube mask: pick once, reuse across frames ---
#         n_px, n_py = 4, 3                  # patch grid (try 16x16 for a more ViT-like look)
#         mask_ratio = 0.4                  # V-JEPA goes as high as ~0.9
#         total = n_px * n_py
#         n_masked = int(mask_ratio * total)
#         rng = np.random.default_rng(RANDOM_SEED)     # tweak seed until the pattern feels right
#         masked_ids = rng.choice(total, n_masked, replace=False)

#         for count, i in enumerate(indices_to_show):
#             z = -spacing * (len(indices_to_show) - 1) + spacing * count
#             imgs[i].move_to([0, 0, z])

#             border = SurroundingRectangle(imgs[i], color=CHILL_BROWN, buff=0)
#             border.set_stroke(width=2)
#             video.add(imgs[i])
#             borders.add(border)

#             # Patch geometry, derived from this image's actual bounds
#             w, h = imgs[i].get_width(), imgs[i].get_height()
#             pw, ph = w / n_px, h / n_py
#             cx, cy, _ = imgs[i].get_center()
#             left, bottom = cx - w / 2, cy - h / 2

#             for idx in masked_ids:
#                 row, col = divmod(idx, n_px)
#                 x = left + (col + 0.5) * pw
#                 y = bottom + (row + 0.5) * ph
#                 patch = Rectangle(width=pw, height=ph)
#                 patch.set_fill(BLACK, opacity=1.0)
#                 patch.set_stroke(width=0)
#                 # Tiny z nudge so patches sit just in front of the image plane
#                 patch.move_to([x, y, z + 1e-3])
#                 masks.add(patch)

#         video_and_borders = Group(video, borders, masks)
#         video_and_borders.rotate(90 * DEGREES, [1, 0, 0])

#         self.frame.reorient(-53, 59, 0, (-1.77, -1.52, -3.17), 10.06)
#         self.wait()

#         self.add(borders)
#         self.add(video)
#         self.add(masks)


#         self.wait()




#         self.wait(20)
#         self.embed()









