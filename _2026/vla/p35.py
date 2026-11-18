import manimgx as gx
from welch_scene import WelchScene, image_at_height
import numpy as np
from pathlib import Path
from PIL import Image


class P35(WelchScene):
    def construct(self):

        source = Path(__file__).resolve().parent / "pre_split_im_right.png"
        with Image.open(source) as img:
            img_array = np.asarray(img.convert("RGBA"))
        height, width = img_array.shape[:2]

        grid_n = 16
        patch_h = height // grid_n
        patch_w = width // grid_n
        total_height = 3.0
        patch_size = total_height / grid_n

        pixel_squares = gx.Group()
        for i in range(grid_n):
            for j in range(grid_n):
                patch = img_array[i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]
                patch_mob = image_at_height(patch, height=patch_size)
                patch_mob.stretch_to_fit_width(patch_size)

                x_pos = (j - grid_n/2 + 0.5) * patch_size
                y_pos = -(i - grid_n/2 + 0.5) * patch_size

                patch_mob.move_to([x_pos, y_pos, 0])
                pixel_squares.add(patch_mob)

        self.wait()
        self.play(gx.FadeIn(pixel_squares))
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
                animations.append(pixel.animate.move_to(new_position))

        self.play(*animations, run_time=2)
        self.wait()
