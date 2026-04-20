from manimlib import *
from tqdm import tqdm
import re
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


def wire_prism(dims, color=CHILL_BROWN, stroke_width=10.0):
    """Wireframe rectangular prism centered at origin.
    dims = [x_len, y_len, z_len] -> VGroup of 12 edges."""
    w, h, d = dims
    corners = {
        (sx, sy, sz): np.array([sx * w / 2, sy * h / 2, sz * d / 2])
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    }
    edges = VGroup()
    keys = list(corners.keys())
    overshoot = stroke_width / 400
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            if sum(a != b for a, b in zip(k1, k2)) == 1:
                p1, p2 = corners[k1], corners[k2]
                v = normalize(p2 - p1)
                edges.add(Line(
                    p1 - v * overshoot, p2 + v * overshoot,
                    color=color, stroke_width=stroke_width,
                ))
    return edges


class LeNetSketch2(InteractiveScene):
    def construct(self):
        # Tilted 3D view so the prisms read as boxes
        self.frame.reorient(-25, 65, 0)

        # Input image: thin slab, large spatial face
        img = wire_prism([0.3, 3.0, 3.0])
        img.move_to([-3.6, 0, 0])

        # Intermediate conv layer: roughly cubic, ~1/3 the spatial size
        l2 = wire_prism([0.7, 2.0, 2.0])
        l2.move_to([-2.6, 0, 0])

        # Deeper conv layer: elongated prism (lots of channels, small face)
        l3 = wire_prism([1.2, 0.7, 0.7])
        l3.move_to([-1.3, 0, 0])

        layers = VGroup(img, l2, l3)
        # self.add(layers)

        # self.frame.reorient(30, 64, 0, (-2.25, -0.05, 0.28), 6.55)
        # self.frame.reorient(31, 55, 0, (-2.27, -0.11, 0.21), 6.55)
        self.frame.reorient(24, 64, 0, (-2.27, -0.11, 0.21), 6.55)

        self.play(Write(layers), run_time=5)



        self.embed()



class AlexNetSketch3(InteractiveScene):
    def construct(self):
        # Tilted 3D view so the prisms read as boxes
        self.frame.reorient(-25, 65, 0)

        # Input image: thin slab, large spatial face
        img = wire_prism([0.3, 3.0, 3.0])
        img.move_to([-3.6, 0, 0])

        # Intermediate conv layer: roughly cubic, ~1/3 the spatial size
        l2 = wire_prism([0.8, 1.3, 1.3])
        l2.move_to([-2.8, 0, 0])

        # Deeper conv layer: elongated prism (lots of channels, small face)
        l3 = wire_prism([1.0, 0.7, 0.7])
        l3.move_to([-1.6, 0, 0])

        l4 = wire_prism([1.0, 0.7, 0.7])
        l4.move_to([-0.3, 0, 0])

        l5 = wire_prism([0.7, 0.7, 0.7])
        l5.move_to([0.8, 0, 0])

        layers = VGroup(img, l2, l3, l4, l5)
        # self.add(layers)

        # self.frame.reorient(24, 64, 0, (-2.27, -0.11, 0.21), 6.55)
        self.frame.reorient(20, 63, 0, (-1.7, 0.08, -0.03), 6.55)

        self.play(Write(layers), run_time=5)


        # self.frame.reorient(3, 65, 0, (-1.44, -0.02, -0.24), 7.21)
        self.wait()
        self.play(self.frame.animate.reorient(1, 62, 0, (-1.39, 0.05, -0.11), 6.55),
                  run_time=5)

        self.embed()












