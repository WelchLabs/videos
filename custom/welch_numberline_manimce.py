"""ManimGX example for the shared WelchNumberLine construct."""

import manimgx as m

from welch_axes import WelchNumberLine


class WelchNumberLineExample(m.Scene):
    def construct(self):
        welch_number_line = WelchNumberLine()
        welch_number_line2 = WelchNumberLine(
            stroke_width=5,
            x_range=[-2, 3, 1],
            include_tip=True,
        ).scale(1.23)
        welch_number_line2.remove_tip("start")
        welch_number_line3 = WelchNumberLine(
            stroke_width=5,
            x_range=[-3, 12, 1],
            include_tip=True,
        )
        welch_number_line3.remove_tip("end").rotate(1, about_point=m.ORIGIN)
        welch_number_line4 = WelchNumberLine(include_tip = False).rotate(-0.23).scale(0.93)
        welch_number_line5 = WelchNumberLine()
        self.play(m.Create(welch_number_line))
        self.play(m.FadeOut(welch_number_line))
        self.play(m.DrawBorderThenFill(welch_number_line))
        self.play(m.FadeOut(welch_number_line))
        self.play(m.GrowFromEdge(welch_number_line, m.LEFT))
        self.wait()
        self.play(m.ReplacementTransform(welch_number_line, welch_number_line2))
        self.wait()
        self.play(m.ReplacementTransform(welch_number_line2, welch_number_line3))
        self.wait()
        self.play(m.ReplacementTransform(welch_number_line3, welch_number_line4))
        self.wait()
        self.play(m.ReplacementTransform(welch_number_line4, welch_number_line5))
        self.wait()
