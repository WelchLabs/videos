import manimgx as gx
from welch_style import BLUE, CHILL_BLUE, CHILL_BROWN, CHILL_GREEN, CYAN, FRESH_TAN, GREEN, MAGENTA, YELLOW, YELLOW_FADE
import numpy as np
import matplotlib.pyplot as plt


class P30(gx.Scene):
    def construct(self):
        # Line 1: title in Myriad Pro Bold
        title = gx.Text(
            "CROSS-ENTROPY LOSS",
            font="Myriad Pro",
            weight=gx.BOLD,
            font_size=32,
        )
        title.set_color(CHILL_BROWN)

        # Line 2: loss = -ln(P(Screw Driver))
        loss_eq = gx.MathTex(
            r"\text{loss} = -\ln\big(P(\text{Screw Driver})\big)",
            font_size=52,
        )
        # loss_eq.set_color(FRESH_TAN)

        # Line 3: -ln(1.0) = 0
        example_eq = gx.MathTex(
            r"-\ln(1.0) = 0",
            font_size=52,
        )
        # example_eq.set_color(FRESH_TAN)

        # Stack and center
        lines = gx.VGroup(title, loss_eq, example_eq)
        lines.arrange(gx.DOWN, buff=0.7)
        lines.move_to(gx.ORIGIN)

        example_eq.shift([-0.35, 0, 0])
        example_eq.shift([0, 0.4, 0])


        # self.add(title, loss_eq, example_eq)
        # self.remove(title, loss_eq, example_eq)

        # Write in one at a time
        self.wait(1)
        self.play(gx.Write(title), run_time=2.5)
        self.wait(0.5)
        self.play(gx.Write(loss_eq), run_time=2.5)
        self.wait(0.5)
        self.play(gx.Write(example_eq), run_time=2.0)


        self.wait(20)
