from manimlib import *
import numpy as np
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


class P30(InteractiveScene):
    def construct(self):
        # Line 1: title in Myriad Pro Bold
        title = Text(
            "CROSS-ENTROPY LOSS",
            font="Myriad Pro",
            weight=BOLD,
            font_size=32,
        )
        title.set_color(CHILL_BROWN)

        # Line 2: loss = -ln(P(Screw Driver))
        loss_eq = Tex(
            r"\text{loss} = -\ln\big(P(\text{Screw Driver})\big)",
            font_size=52,
        )
        # loss_eq.set_color(FRESH_TAN)

        # Line 3: -ln(1.0) = 0
        example_eq = Tex(
            r"-\ln(1.0) = 0",
            font_size=52,
        )
        # example_eq.set_color(FRESH_TAN)

        # Stack and center
        lines = VGroup(title, loss_eq, example_eq)
        lines.arrange(DOWN, buff=0.7)
        lines.move_to(ORIGIN)

        example_eq.shift([-0.35, 0, 0])
        example_eq.shift([0, 0.4, 0])


        # self.add(title, loss_eq, example_eq)
        # self.remove(title, loss_eq, example_eq)

        # Write in one at a time
        self.wait(1)
        self.play(Write(title), run_time=2.5)
        self.wait(0.5)
        self.play(Write(loss_eq), run_time=2.5)
        self.wait(0.5)
        self.play(Write(example_eq), run_time=2.0)



        self.wait(20)
        self.embed()