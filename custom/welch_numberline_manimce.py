from manim import *
from custom.welch_tip_manimce import WelchTip

class WelchNumberLine(NumberLine):
    def __init__(self, **kwargs):
        kwargs.setdefault("color", "#948979")
        kwargs.setdefault("stroke_color", "#948979")
        kwargs.setdefault("stroke_width", 3)
        kwargs.setdefault("include_numbers", True)
        kwargs.setdefault("include_ticks", True)
        kwargs.setdefault("x_range", [0, 6])

        kwargs.pop("include_tip", None)  
        
        include_tip = kwargs.get("include_tip", True)
        
        super().__init__(**kwargs)
        
        self.starting_tip = None
        self.ending_tip = None
        
        if include_tip:
            tick_marks = self.get_tick_marks()
            if tick_marks and len(tick_marks.submobjects) > 1:
                tick_marks.submobjects = tick_marks.submobjects[1:-1]
                
            self.starting_tip = WelchTip().rotate(PI)
            self.ending_tip = WelchTip()

            if tick_marks and len(tick_marks.submobjects) > 0:
                tick_height = tick_marks.submobjects[0].get_height() if tick_marks.submobjects else 0.1
                self.starting_tip.scale_to_fit_height(tick_height)
                self.ending_tip.scale_to_fit_height(tick_height)

                self.starting_tip.move_to(self.get_start()) 
                self.ending_tip.move_to(self.get_end())

                self.add(self.starting_tip, self.ending_tip)

    def remove_tip(self, position):
        if position == "start" and self.starting_tip:
            self.remove(self.starting_tip)
            self.starting_tip = None
            tick = self.get_tick(self.x_range[0])
            self.add(tick)
        elif position == "end" and self.ending_tip:
            self.remove(self.ending_tip)
            self.ending_tip = None
            tick = self.get_tick(self.x_range[1])
            self.add(tick)
        
        return self     
        
        
class WelchNumberLineExample(Scene):
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
        welch_number_line3.remove_tip("end").rotate_about_origin(1)
        welch_number_line4 = WelchNumberLine(include_tip = False).rotate(-0.23).scale(0.93)
        welch_number_line5 = WelchNumberLine()
        self.play(Create(welch_number_line))
        self.play(FadeOut(welch_number_line))
        self.play(DrawBorderThenFill(welch_number_line))
        self.play(FadeOut(welch_number_line))
        self.play(GrowFromEdge(welch_number_line, LEFT))
        self.wait()
        self.play(ReplacementTransform(welch_number_line, welch_number_line2))
        self.wait()
        self.play(ReplacementTransform(welch_number_line2, welch_number_line3))
        self.wait()
        self.play(ReplacementTransform(welch_number_line3, welch_number_line4))
        self.wait()
        self.play(ReplacementTransform(welch_number_line4, welch_number_line5))
        self.wait()