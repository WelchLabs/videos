from welch_paths import welch_path
import os
from pathlib import Path

import manimgx as m
import numpy as np

from welch_style import BLUE, YELLOW


svg_dir = Path(os.environ.get(
    "WORLD_MODELS_JEPA_SVGS",
    welch_path('jepa/graphics/p78_80'),
))
img_dir = Path(os.environ.get(
    "WORLD_MODELS_JEPA_FRAMES",
    welch_path('jepa/hacking/v_jepa_ac_1'),
))


def make_emb_tex(vals, color, position, scale=0.65):
    s = (r'\begin{bmatrix} '
         + f'{vals[0]:.2f}' + r', \ '
         + f'{vals[1]:.2f}' + r', \ \dots, \ '
         + f'{vals[2]:.2f}'
         + r' \end{bmatrix}')
    e = m.MathTex(s)
    e.set_color(color)
    e.scale(scale)
    e.move_to(position)
    return e


class P78_80B(m.Scene):
    def construct(self) -> None:
        svg_files = sorted(svg_dir.glob("*.svg"))[3:]
        if len(svg_files) < 13:
            raise FileNotFoundError(f"Expected at least 16 storyboard SVGs in {svg_dir}")
        all_svgs = [m.SVGMobject(path).scale(4.0)[1:] for path in svg_files]

        frame = self.camera.frame
        frame.scale_to_fit_height(6.92).move_to([-0.1, 0.75, 0])
        self.add(all_svgs[0], all_svgs[1], all_svgs[2], all_svgs[3],
                 all_svgs[4], all_svgs[5])

        self.wait()
        self.play(m.Write(all_svgs[6]), run_time=5)

        imgs = m.Group()
        num_frames = 124
        for i in range(num_frames):
            imgs.add(m.ImageMobject(img_dir / f"VJEPA_AC_CLIPS{i:03d}.jpg").scale_to_fit_height(4))

        imgs_left = imgs.copy()
        imgs_right = imgs.copy()

        imgs[0].scale(0.42)
        imgs[0].move_to([-3.2, -1.7, 0])
        imgs[0].set_opacity(0.5)
        imgs[1].scale(0.42)
        imgs[1].move_to([-3.1, -1.8, 0])
        imgs[1].set_opacity(0.5)
        imgs[2].scale(0.42)
        imgs[2].move_to([-3.0, -1.9, 0])

        imgs[3].scale(0.47)
        imgs[3].move_to([2.45, -1.77, 0])

        self.wait()
        # Bring in the actuator signals.
        self.remove(all_svgs[6][1:])
        self.play(m.Write(all_svgs[8][1:]),
                 all_svgs[6][0].animate.move_to([-0.233, 0.95, 0]),
                 run_time=3)

        self.play(frame.animate.scale_to_fit_height(7.43).move_to([-0.19, 0.27, 0]),
                  m.FadeOut(all_svgs[1]),
                  run_time=2)

        # Stack the video frames.
        self.add(imgs[0])
        self.wait(0.2)
        self.add(imgs[1])
        self.wait(0.2)
        self.add(imgs[2])
        self.wait(0.2)
        self.add(imgs[3])
        self.add(all_svgs[7])

        for im in imgs_left:
            im.scale(0.42).move_to([-3.0, -1.9, 0])

        for im in imgs_right:
            im.scale(0.47).move_to([2.45, -1.77, 0])

        self.wait()
        self.remove(imgs[2], imgs[3])
        for i in range(4, num_frames):
            self.remove(imgs_left[i-2], imgs_right[i-1])
            self.add(imgs_left[i-1], imgs_right[i])
            self.wait(0.1)

        self.play(m.FadeOut(imgs[0]),
                  m.FadeOut(imgs[1]),
                  m.FadeOut(imgs_left[i-1]),
                  m.FadeOut(imgs_right[i]),
                  m.FadeOut(all_svgs[7]),
                  run_time=2)

        all_svgs[10].move_to([-0.15, -2.9, 0])
        self.play(m.FadeIn(imgs_right[i]),
                  m.Write(all_svgs[10][5:]),
                  m.FadeIn(imgs[0]),
                  m.FadeIn(imgs[1]),
                  m.FadeIn(imgs[2]),
                  m.Write(all_svgs[10][:5]),
                  run_time=2)

        embedding_1 = make_emb_tex([0.22, -0.13, 0.31], BLUE, [2.3, 2.0, 0], scale=0.5)
        embedding_1_og = embedding_1.copy()
        embedding_2 = make_emb_tex([0.05, -0.02, 0.50], YELLOW, [2.3, 1.4, 0], scale=0.5)

        self.wait()
        self.play(m.FadeOut(all_svgs[5][1:]))
        self.play(m.Write(embedding_2),
                  m.FadeIn(all_svgs[11][:13]),
                  run_time=2)

        self.wait()
        all_svgs[11][13:].set_color(BLUE)
        self.play(m.Write(embedding_1),
                  m.FadeIn(all_svgs[11][13:]),
                  m.Write(all_svgs[9]),
                  run_time=2)

        num_steps = num_frames
        s1 = np.linspace(0.22, 0.05, num_steps)
        s2 = np.linspace(-0.13, -0.02, num_steps)
        s3 = np.linspace(0.31, 0.5, num_steps)

        all_svgs[12].move_to([-5.4, -3.15, 0])
        self.wait()
        self.add(all_svgs[12])
        for n1, n2, n3 in zip(s1, s2, s3):
            self.remove(embedding_1)
            embedding_1 = make_emb_tex([n1, n2, n3], color=BLUE, position=[2.3, 2.0, 0], scale=0.5)
            self.add(embedding_1)
            self.wait(0.1)

        self.wait()
        self.remove(imgs[2])
        self.remove(all_svgs[12])
        self.remove(embedding_1)
        self.add(embedding_1_og)
        self.wait()
        self.remove(embedding_1_og)
        for i in range(4, num_frames):
            self.remove(imgs_left[i-2])
            self.add(imgs_left[i-1])
            self.remove(embedding_1)
            embedding_1 = make_emb_tex([s1[i], s2[i], s3[i]], color=BLUE, position=[2.3, 2.0, 0], scale=0.5)
            self.add(embedding_1)
            self.wait(0.1)

        self.wait(20)
