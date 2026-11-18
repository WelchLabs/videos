"""ManimGX example for the shared WelchTip construct."""

import manimgx as m

from welch_axes import WelchTip


class WelchTipExample(m.Scene):
    def construct(self):
        welch_tip = WelchTip()
        welch_tip.move_to(m.ORIGIN)
        self.play(m.Create(welch_tip))
        self.play(m.FadeOut(welch_tip))
        self.play(m.DrawBorderThenFill(welch_tip))
        self.play(m.FadeOut(welch_tip))
        self.play(m.GrowFromEdge(welch_tip, m.LEFT))
        self.play(m.FadeOut(welch_tip))
