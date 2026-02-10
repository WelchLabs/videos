from manim import *

class WelchTip(VMobject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        points = [
            [32.33, 13.21, 0],
            [0, 26.42, 0],
            [7.67, 13.21, 0],
            [0, 0, 0],
            [32.33, 13.21, 0]
        ]

        scaled_points = [np.array(p) * 0.1 for p in points]

        polygon = Polygon(*scaled_points, color="#948979", fill_color="#948979", fill_opacity=1)

        self.add(polygon)

class WelchTipExample(Scene):
    def construct(self):
        welch_tip = WelchTip()
        welch_tip.move_to(ORIGIN)
        self.play(Create(welch_tip))
        self.play(FadeOut(welch_tip))
        self.play(DrawBorderThenFill(welch_tip))
        self.play(FadeOut(welch_tip))
        self.play(GrowFromEdge(welch_tip, LEFT))
        self.play(FadeOut(welch_tip))
