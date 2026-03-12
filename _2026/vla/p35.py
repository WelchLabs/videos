from manimlib import *
import numpy as np
import tempfile, os
from PIL import Image


class P35(InteractiveScene):
    def construct(self):

        img = Image.open('pre_split_im_right.png')
        img_array = np.array(img)
        height, width = img_array.shape[:2]

        grid_n = 16
        patch_h = height // grid_n
        patch_w = width // grid_n
        total_height = 3.0
        patch_size = total_height / grid_n

        patch_dir = tempfile.mkdtemp()
        pixel_squares = Group()
        for i in range(grid_n):
            for j in range(grid_n):
                patch_img = img.crop((j*patch_w, i*patch_h, (j+1)*patch_w, (i+1)*patch_h))
                patch_path = os.path.join(patch_dir, f'patch_{i}_{j}.png')
                patch_img.save(patch_path)
                patch_mob = ImageMobject(patch_path)
                patch_mob.set_height(patch_size)
                patch_mob.set_width(patch_size, stretch=True)

                x_pos = (j - grid_n/2 + 0.5) * patch_size
                y_pos = -(i - grid_n/2 + 0.5) * patch_size

                patch_mob.move_to([x_pos, y_pos, 0])
                pixel_squares.add(patch_mob)

        self.wait()
        self.play(FadeIn(pixel_squares))
        self.wait()

        animations = []
        center = pixel_squares.get_center()
        gap_factor = 0.4

        for pixel in pixel_squares:
            pixel_pos = pixel.get_center()
            direction_vector = pixel_pos - center
            distance = np.linalg.norm(direction_vector)

            if distance > 0:
                unit_vector = direction_vector / distance
                displacement = unit_vector * distance * gap_factor
                new_position = pixel_pos + displacement
                animations.append(ApplyMethod(pixel.move_to, new_position))

        self.play(*animations, run_time=2)
        self.wait()

        self.embed()
