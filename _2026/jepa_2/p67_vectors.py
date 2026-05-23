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




class p67_vectors(InteractiveScene):
    def construct(self):

        n_steps=100
        dt=0.2
        vec_len=5
        step_size=0.35
        seed=25
        position=ORIGIN
        font_size=48

        np.random.seed(seed)
        # Keep values in [0.0, 0.94] so they always round to "0.X" (stable width)
        values = np.clip(np.random.normal(0.5, 0.2, size=vec_len), 0.0, 0.94)

        self.frame.reorient(0, 0, 0, (-0.06, 0.0, 0.0), 5.24)

        current = None
        for _ in range(n_steps):
            s = f"[ {values[0]:.1f}, {values[1]:.1f}, ... , {values[-1]:.1f} ]"
            new_t = Text(s, color=FRESH_TAN, font_size=font_size).move_to(position)
            new_t.set_color(FRESH_TAN)

            if current is not None:
                self.remove(current)
            self.add(new_t)
            current = new_t

            self.wait(dt)

            # Random walk step, clipped to keep one-decimal formatting stable
            values = np.clip(
                values + np.random.uniform(-step_size, step_size, size=vec_len),
                0.0, 0.94,
            )

        # self.add(new_t)

        self.wait()


        self.wait(20)
        self.embed()