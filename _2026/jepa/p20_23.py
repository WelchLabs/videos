from manimlib import *
from tqdm import tqdm
import re
import random
import math
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

SVG_DIR = '/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/jepa/graphics/p20_23_to_manim'

svg_01 = SVGMobject(f'{SVG_DIR}/p20_23_to_manim-01.svg')[1:].scale(4)
svg_02 = SVGMobject(f'{SVG_DIR}/p20_23_to_manim-02.svg')[1:].scale(4)
svg_03 = SVGMobject(f'{SVG_DIR}/p20_23_to_manim-03.svg')[1:].scale(4)
svg_04 = SVGMobject(f'{SVG_DIR}/p20_23_to_manim-04.svg')[1:].scale(4)
svg_05 = SVGMobject(f'{SVG_DIR}/p20_23_to_manim-05.svg')[1:].scale(4)
svg_06 = SVGMobject(f'{SVG_DIR}/p20_23_to_manim-06.svg')[1:].scale(4)
svg_07 = SVGMobject(f'{SVG_DIR}/p20_23_to_manim-07.svg')[1:].scale(4)
svg_08 = SVGMobject(f'{SVG_DIR}/p20_23_to_manim-08.svg')[1:].scale(4)
svg_09 = SVGMobject(f'{SVG_DIR}/p20_23_to_manim-09.svg')[1:].scale(4)
svg_10 = SVGMobject(f'{SVG_DIR}/p20_23_to_manim-10.svg')[1:].scale(4)
svg_11 = SVGMobject(f'{SVG_DIR}/p20_23_to_manim-11.svg')[1:].scale(4)
svg_12 = SVGMobject(f'{SVG_DIR}/p20_23_to_manim-12.svg')[1:].scale(4)

class P20(InteractiveScene):
    def construct(self):
        svg_05.move_to(ORIGIN)
        self.play(Write(svg_05))

class P20_21(InteractiveScene):
    def construct(self):
        german_text  = VGroup(*[svg_03[i] for i in list(range(0,20))  + list(range(20,38)) + list(range(38,59)) + list(range(59,78)) + list(range(78,94))])
        english_text = VGroup(*[svg_03[i] for i in list(range(94,117)) + list(range(117,142)) + list(range(142,164)) + list(range(164,198))])
        german_label = VGroup(*[svg_03[i] for i in range(198, 204)])
        english_label= VGroup(*[svg_03[i] for i in range(204, 211)])

        german_text.to_edge(LEFT)
        german_label.next_to(german_text, DOWN, buff=0.2)
        english_text.to_edge(RIGHT)
        english_label.next_to(english_text, DOWN, buff=0.2)

        svg_05.move_to(ORIGIN)

        arrow_len = 0.4
        left_mid  = (german_text.get_right() + svg_05.get_left()) / 2
        right_mid = (svg_05.get_right() + english_text.get_left()) / 2
        left_mid[1] = right_mid[1] = 0  # keep vertical centered

        left_arrow  = Arrow(left_mid  + LEFT * arrow_len / 2, left_mid  + RIGHT * arrow_len / 2, buff=0).set_color(CHILL_BROWN)
        right_arrow = Arrow(right_mid + LEFT * arrow_len / 2, right_mid + RIGHT * arrow_len / 2, buff=0).set_color(CHILL_BROWN)

        capital_text = Text(
            '"The capital of France is"',
            font="Georgia", slant=ITALIC, color=WHITE, font_size=32,
        ).set_width(german_text.get_width()).move_to([german_text.get_center()[0], 0, 0])

        paris_text = Text(
            '"Paris"',
            font="Georgia", slant=ITALIC, font_size=32,
        ).move_to([english_text.get_center()[0], 0, 0]).set_color(YELLOW)

        self.play(Write(svg_05), run_time=4)

        self.play(Write(german_text), run_time=4)
        self.play(FadeIn(german_label), run_time=2)
        self.play(GrowArrow(left_arrow))
        self.play(GrowArrow(right_arrow))
        self.play(ReplacementTransform(german_text.copy(), english_text), run_time=3)
        self.play(FadeIn(english_label), run_time=2)

        self.play(
            ReplacementTransform(german_text, capital_text),
            FadeOut(german_label),
            run_time=2,
        )
        self.play(
            ReplacementTransform(capital_text.copy(), paris_text),
            FadeOut(english_text),
            FadeOut(english_label),
            run_time=3,
        )

        self.embed()

CURATED_COMPLETIONS = [
    ("The capital of France is",         "Paris"),
    ("The largest ocean is the",          "Pacific"),
    ("Mount Everest is the world's tallest", "mountain"),
    ("A triangle has three",              "sides"),
    ("Two plus two equals",               "four"),
    ("Half of ten is",                    "five"),
    ("The earth orbits the",              "sun"),
    ("The sun is a",                      "star"),
    ("The heart pumps",                   "blood"),
    ("Light travels faster than",         "sound"),
    ("The brain controls the",            "body"),
    ("Humans breathe",                    "air"),
    ("Plants need sunlight to make",      "food"),
    ("Birds can",                         "fly"),
    ("Fish can",                          "swim"),
    ("Sharks live in the",                "ocean"),
    ("Bees make",                         "honey"),
    ("Fire is",                           "hot"),
    ("Ice is",                            "cold"),
    ("Honey is",                          "sweet"),
    ("The opposite of left is",           "right"),
    ("The opposite of day is",            "night"),
    ("The opposite of war is",            "peace"),
    ("Curiosity killed the",              "cat"),
    ("Knowledge is",                      "power"),
    ("Practice makes",                    "perfect"),
    ("Time is",                           "money"),
    ("Rome was not built in a",           "day"),
    ("Dogs are man's best",               "friend"),
    ("The pen is mightier than the",      "sword"),
    ("Actions speak louder than",         "words"),
    ("Every cloud has a silver",          "lining"),
    ("Spring follows",                    "winter"),
    ("A year has twelve",                 "months"),
    ("The sun rises in the",              "east"),
    ("Water flows",                       "down"),
    ("The sky is",                        "blue"),
    ("Gravity keeps us on the",           "ground"),
    ("Sound travels slower than",         "light"),
    ("The Beatles were a famous",         "band"),
]

def _load_wikitext_completions(n=100, seed=42):
    import re as _re, random as _random
    _random.seed(seed)
    _word_token = _re.compile(r"[A-Za-z]+")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        raw_texts = [t.strip() for t in ds["text"] if t.strip() and not t.strip().startswith("=")]
    except Exception as e:
        print(f"wikitext load failed: {e}")
        return []

    pairs = []
    for text in raw_texts:
        # Keep only plain alphabetic words so rendered examples have no punctuation.
        words = _word_token.findall(text)
        if 6 <= len(words) <= 12:
            mid = max(3, len(words) // 2)
            inp = " ".join(words[:mid])
            out = " ".join(words[mid:])
            pairs.append((inp, out))

    _random.shuffle(pairs)
    return pairs[:n]

WIKITEXT_COMPLETIONS = _load_wikitext_completions()


class P20_22a(InteractiveScene):
    def construct(self):
        german_text  = VGroup(*[svg_03[i] for i in list(range(0,20))  + list(range(20,38)) + list(range(38,59)) + list(range(59,78)) + list(range(78,94))])
        english_text = VGroup(*[svg_03[i] for i in list(range(94,117)) + list(range(117,142)) + list(range(142,164)) + list(range(164,198))])
        german_label = VGroup(*[svg_03[i] for i in range(198, 204)])
        english_label= VGroup(*[svg_03[i] for i in range(204, 211)])

        german_text.to_edge(LEFT)
        german_label.next_to(german_text, DOWN, buff=0.2)
        english_text.to_edge(RIGHT)
        english_label.next_to(english_text, DOWN, buff=0.2)

        svg_05.move_to(ORIGIN)

        left_mid  = (german_text.get_right() + svg_05.get_left()) / 2
        right_mid = (svg_05.get_right() + english_text.get_left()) / 2
        left_mid[1] = right_mid[1] = 0  # keep vertical centered

        io_arrows = svg_07.copy().set_color(CHILL_BROWN)
        arrows_mid = (left_mid + right_mid) / 2
        io_arrows.move_to(arrows_mid)

        capital_text = Text(
            '"The capital of France is"',
            font="Georgia", slant=ITALIC, color=WHITE, font_size=32,
        ).set_width(german_text.get_width()).move_to([german_text.get_center()[0], 0, 0])

        paris_text = Text(
            '"Paris"',
            font="Georgia", slant=ITALIC, font_size=32,
        ).move_to([english_text.get_center()[0], 0, 0]).set_color(YELLOW)

        self.play(Write(svg_05), run_time=4)

        self.play(Write(german_text), run_time=4)
        self.play(FadeIn(german_label), run_time=2)
        self.play(GrowArrow(left_arrow))
        self.play(GrowArrow(right_arrow))
        self.play(ReplacementTransform(german_text.copy(), english_text), run_time=3)
        self.play(FadeIn(english_label), run_time=2)

        self.play(
            ReplacementTransform(german_text, capital_text),
            FadeOut(german_label),
            run_time=2,
        )
        self.play(
            ReplacementTransform(capital_text.copy(), paris_text),
            FadeOut(english_text),
            FadeOut(english_label),
            run_time=3,
        )


        svg_06.move_to([-3.44073984e-02,  2.49935210e-01, -3.29861139e-18])
        for mob in svg_06:
            mob.set_fill(opacity=0.1)

        # Clamp each side to both available panel space and the original side text width.
        MARGIN   = 0.8
        left_gap_w = abs(svg_05.get_left()[0] - (-7.1)) - MARGIN
        right_gap_w = abs(7.1 - svg_05.get_right()[0]) - MARGIN
        LEFT_MAX_TEXT_W = min(max(left_gap_w, 0.1), german_text.get_width())
        RIGHT_MAX_TEXT_W = min(max(right_gap_w, 0.1), english_text.get_width())
        LEFT_X   = (-7.1 + svg_05.get_left()[0]) / 2
        RIGHT_X  = (svg_05.get_right()[0] + 7.1) / 2



        self.play(FadeIn(svg_06))

        def make_input(txt):
            t = Text(f'"{txt}"', font="Georgia", slant=ITALIC, color=WHITE, font_size=22,
                     line_spacing_height=0.05)
            if t.get_width() > LEFT_MAX_TEXT_W:
                t.set_width(LEFT_MAX_TEXT_W)
            return t.move_to([LEFT_X, 0, 0])

        def make_output(txt):
            t = Text(f'"{txt}"', font="Georgia", slant=ITALIC, color=YELLOW, font_size=22,
                     line_spacing_height=0.05)
            if t.get_width() > RIGHT_MAX_TEXT_W:
                t.set_width(RIGHT_MAX_TEXT_W)
            return t.move_to([RIGHT_X, 0, 0])

        def rand_fill_anims():
            return [mob.animate.set_fill(opacity=random.uniform(0.05, 1.0)) for mob in svg_06]

        examples = WIKITEXT_COMPLETIONS
        cur_in  = make_input(examples[0][0])
        cur_out = make_output(examples[0][1])

        self.play(FadeOut(capital_text), FadeOut(paris_text))

        self.play(FadeIn(cur_in), FadeIn(cur_out), *rand_fill_anims(), run_time=0.5)

        for inp, out in examples[1:]:
            new_in  = make_input(inp)
            new_out = make_output(out)
            self.play(
                FadeTransform(cur_in, new_in),
                FadeTransform(cur_out, new_out),
                *rand_fill_anims(),
                run_time=0.35,
            )
            cur_in, cur_out = new_in, new_out

        self.embed()

class P20_22a(InteractiveScene):
    def construct(self):
        german_text  = VGroup(*[svg_03[i] for i in list(range(0,20))  + list(range(20,38)) + list(range(38,59)) + list(range(59,78)) + list(range(78,94))])
        english_text = VGroup(*[svg_03[i] for i in list(range(94,117)) + list(range(117,142)) + list(range(142,164)) + list(range(164,198))])
        german_label = VGroup(*[svg_03[i] for i in range(198, 204)])
        english_label= VGroup(*[svg_03[i] for i in range(204, 211)])

        german_text.to_edge(LEFT)
        german_label.next_to(german_text, DOWN, buff=0.2)
        english_text.to_edge(RIGHT)
        english_label.next_to(english_text, DOWN, buff=0.2)

        svg_05.move_to(ORIGIN)

        arrow_len = 0.4
        left_mid  = (german_text.get_right() + svg_05.get_left()) / 2
        right_mid = (svg_05.get_right() + english_text.get_left()) / 2
        left_mid[1] = right_mid[1] = 0  # keep vertical centered

        left_arrow  = Arrow(left_mid  + LEFT * arrow_len / 2, left_mid  + RIGHT * arrow_len / 2, buff=0).set_color(CHILL_BROWN)
        right_arrow = Arrow(right_mid + LEFT * arrow_len / 2, right_mid + RIGHT * arrow_len / 2, buff=0).set_color(CHILL_BROWN)

        capital_text = Text(
            '"The capital of France is"',
            font="Georgia", slant=ITALIC, color=WHITE, font_size=32,
        ).set_width(german_text.get_width()).move_to([german_text.get_center()[0], 0, 0])

        paris_text = Text(
            '"Paris"',
            font="Georgia", slant=ITALIC, font_size=32,
        ).move_to([english_text.get_center()[0], 0, 0]).set_color(YELLOW)

        self.play(Write(svg_05), run_time=4)

        self.play(Write(german_text), run_time=4)
        self.play(FadeIn(german_label), run_time=2)
        self.play(GrowArrow(left_arrow))
        self.play(GrowArrow(right_arrow))
        self.play(ReplacementTransform(german_text.copy(), english_text), run_time=3)
        self.play(FadeIn(english_label), run_time=2)

        self.play(
            ReplacementTransform(german_text, capital_text),
            FadeOut(german_label),
            run_time=2,
        )
        self.play(
            ReplacementTransform(capital_text.copy(), paris_text),
            FadeOut(english_text),
            FadeOut(english_label),
            run_time=3,
        )


        svg_06.move_to([-3.44073984e-02,  2.49935210e-01, -3.29861139e-18])
        for mob in svg_06:
            mob.set_fill(opacity=0.1)

        # Clamp each side to both available panel space and the original side text width.
        MARGIN   = 0.8
        left_gap_w = abs(svg_05.get_left()[0] - (-7.1)) - MARGIN
        right_gap_w = abs(7.1 - svg_05.get_right()[0]) - MARGIN
        LEFT_MAX_TEXT_W = min(max(left_gap_w, 0.1), german_text.get_width())
        RIGHT_MAX_TEXT_W = min(max(right_gap_w, 0.1), english_text.get_width())
        LEFT_X   = (-7.1 + svg_05.get_left()[0]) / 2
        RIGHT_X  = (svg_05.get_right()[0] + 7.1) / 2



        self.play(FadeIn(svg_06))

        def make_input(txt):
            t = Text(f'"{txt}"', font="Georgia", slant=ITALIC, color=WHITE, font_size=22,
                     line_spacing_height=0.05)
            if t.get_width() > LEFT_MAX_TEXT_W:
                t.set_width(LEFT_MAX_TEXT_W)
            return t.move_to([LEFT_X, 0, 0])

        def make_output(txt):
            t = Text(f'"{txt}"', font="Georgia", slant=ITALIC, color=YELLOW, font_size=22,
                     line_spacing_height=0.05)
            if t.get_width() > RIGHT_MAX_TEXT_W:
                t.set_width(RIGHT_MAX_TEXT_W)
            return t.move_to([RIGHT_X, 0, 0])

        def rand_fill_anims():
            return [mob.animate.set_fill(opacity=random.uniform(0.05, 1.0)) for mob in svg_06]

        examples = WIKITEXT_COMPLETIONS
        cur_in  = make_input(examples[0][0])
        cur_out = make_output(examples[0][1])

        self.play(FadeOut(capital_text), FadeOut(paris_text))

        self.play(FadeIn(cur_in), FadeIn(cur_out), *rand_fill_anims(), run_time=0.5)

        for inp, out in examples[1:]:
            new_in  = make_input(inp)
            new_out = make_output(out)
            self.play(
                FadeTransform(cur_in, new_in),
                FadeTransform(cur_out, new_out),
                *rand_fill_anims(),
                run_time=0.35,
            )
            cur_in, cur_out = new_in, new_out

        self.play(FadeOut(cur_in), FadeOut(cur_out), FadeOut(left_arrow), FadeOut(right_arrow), FadeOut(svg_06))

        self.play(
            self.frame.animate.move_to(svg_05.get_center()).set_width(svg_05.get_width() + 2.0),
            run_time=2.0,
        )

        self.embed()

class P20_22(InteractiveScene):
    def construct(self):
        german_text  = VGroup(*[svg_03[i] for i in list(range(0,20))  + list(range(20,38)) + list(range(38,59)) + list(range(59,78)) + list(range(78,94))])
        english_text = VGroup(*[svg_03[i] for i in list(range(94,117)) + list(range(117,142)) + list(range(142,164)) + list(range(164,198))])
        german_label = VGroup(*[svg_03[i] for i in range(198, 204)])
        english_label= VGroup(*[svg_03[i] for i in range(204, 211)])

        german_text.to_edge(LEFT)
        german_label.next_to(german_text, DOWN, buff=0.2)
        english_text.to_edge(RIGHT)
        english_label.next_to(english_text, DOWN, buff=0.2)

        svg_05.move_to(ORIGIN)

        arrow_len = 0.4
        left_mid  = (german_text.get_right() + svg_05.get_left()) / 2
        right_mid = (svg_05.get_right() + english_text.get_left()) / 2
        left_mid[1] = right_mid[1] = 0  # keep vertical centered

        left_arrow  = Arrow(left_mid  + LEFT * arrow_len / 2, left_mid  + RIGHT * arrow_len / 2, buff=0).set_color(CHILL_BROWN)
        right_arrow = Arrow(right_mid + LEFT * arrow_len / 2, right_mid + RIGHT * arrow_len / 2, buff=0).set_color(CHILL_BROWN)

        capital_text = Text(
            '"The capital of France is"',
            font="Georgia", slant=ITALIC, color=WHITE, font_size=32,
        ).set_width(german_text.get_width()).move_to([german_text.get_center()[0], 0, 0])

        paris_text = Text(
            '"Paris"',
            font="Georgia", slant=ITALIC, font_size=32,
        ).move_to([english_text.get_center()[0], 0, 0]).set_color(YELLOW)

        self.play(Write(svg_05), run_time=4)

        self.play(Write(german_text), run_time=4)
        self.play(FadeIn(german_label), run_time=2)
        self.play(GrowArrow(left_arrow))
        self.play(GrowArrow(right_arrow))
        self.play(ReplacementTransform(german_text.copy(), english_text), run_time=3)
        self.play(FadeIn(english_label), run_time=2)

        self.play(
            ReplacementTransform(german_text, capital_text),
            FadeOut(german_label),
            run_time=2,
        )
        self.play(
            ReplacementTransform(capital_text.copy(), paris_text),
            FadeOut(english_text),
            FadeOut(english_label),
            run_time=3,
        )


        svg_06.move_to([-3.44073984e-02,  2.49935210e-01, -3.29861139e-18])
        for mob in svg_06:
            mob.set_fill(opacity=0.1)

        # Clamp each side to both available panel space and the original side text width.
        MARGIN   = 0.8
        left_gap_w = abs(svg_05.get_left()[0] - (-7.1)) - MARGIN
        right_gap_w = abs(7.1 - svg_05.get_right()[0]) - MARGIN
        LEFT_MAX_TEXT_W = min(max(left_gap_w, 0.1), german_text.get_width())
        RIGHT_MAX_TEXT_W = min(max(right_gap_w, 0.1), english_text.get_width())
        LEFT_X   = (-7.1 + svg_05.get_left()[0]) / 2
        RIGHT_X  = (svg_05.get_right()[0] + 7.1) / 2



        self.play(FadeIn(svg_06))

        def make_input(txt):
            t = Text(f'"{txt}"', font="Georgia", slant=ITALIC, color=WHITE, font_size=22,
                     line_spacing_height=0.05)
            if t.get_width() > LEFT_MAX_TEXT_W:
                t.set_width(LEFT_MAX_TEXT_W)
            return t.move_to([LEFT_X, 0, 0])

        def make_output(txt):
            t = Text(f'"{txt}"', font="Georgia", slant=ITALIC, color=YELLOW, font_size=22,
                     line_spacing_height=0.05)
            if t.get_width() > RIGHT_MAX_TEXT_W:
                t.set_width(RIGHT_MAX_TEXT_W)
            return t.move_to([RIGHT_X, 0, 0])

        def rand_fill_anims():
            return [mob.animate.set_fill(opacity=random.uniform(0.05, 1.0)) for mob in svg_06]

        examples = WIKITEXT_COMPLETIONS
        cur_in  = make_input(examples[0][0])
        cur_out = make_output(examples[0][1])

        self.play(FadeOut(capital_text), FadeOut(paris_text))

        self.play(FadeIn(cur_in), FadeIn(cur_out), *rand_fill_anims(), run_time=0.5)

        for inp, out in examples[1:]:
            new_in  = make_input(inp)
            new_out = make_output(out)

            self.play(
                FadeTransform(cur_in, new_in),
                FadeTransform(cur_out, new_out),
                *rand_fill_anims(),
                run_time=0.35,
            )
            cur_in, cur_out = new_in, new_out


        self.play(FadeOut(cur_in), FadeOut(cur_out), FadeOut(left_arrow), FadeOut(right_arrow), FadeOut(svg_06))

        pan_right = (svg_05.get_right()[0] - self.frame.get_left()[0]) + 0.5
        self.play(self.frame.animate.shift(RIGHT * pan_right), run_time=2.0)

        svg_10.move_to([3.83159839e-01, 1.26843078e-01, 4.00618036e-18]).shift(RIGHT * pan_right)
        svg_09.shift(RIGHT * pan_right)
        self.play(Write(svg_09))

        self.wait()

        self.play(Write(svg_10))
        focus = VGroup(svg_05, svg_09, svg_10)
        self.play(
            self.frame.animate.move_to(focus.get_center()).set_width(focus.get_width() + 2.0),
            run_time=2.0,
        )

        self.embed()


class P20_23(InteractiveScene):
    def construct(self):
        german_text  = VGroup(*[svg_03[i] for i in list(range(0,20))  + list(range(20,38)) + list(range(38,59)) + list(range(59,78)) + list(range(78,94))])
        english_text = VGroup(*[svg_03[i] for i in list(range(94,117)) + list(range(117,142)) + list(range(142,164)) + list(range(164,198))])
        german_label = VGroup(*[svg_03[i] for i in range(198, 204)])
        english_label= VGroup(*[svg_03[i] for i in range(204, 211)])

        german_text.to_edge(LEFT)
        german_label.next_to(german_text, DOWN, buff=0.2)
        english_text.to_edge(RIGHT)
        english_label.next_to(english_text, DOWN, buff=0.2)

        svg_05.move_to(ORIGIN)

        left_mid  = (german_text.get_right() + svg_05.get_left()) / 2
        right_mid = (svg_05.get_right() + english_text.get_left()) / 2
        left_mid[1] = right_mid[1] = 0  # keep vertical centered

        io_arrows = svg_07.copy().set_color(CHILL_BROWN)
        arrows_mid = (left_mid + right_mid) / 2
        io_arrows.move_to(arrows_mid)

        capital_text = Text(
            '"The capital of France is"',
            font="Georgia", slant=ITALIC, color=WHITE, font_size=32,
        ).set_width(german_text.get_width()).move_to([german_text.get_center()[0], 0, 0])

        paris_text = Text(
            '"Paris"',
            font="Georgia", slant=ITALIC, font_size=32,
        ).move_to([english_text.get_center()[0], 0, 0]).set_color(YELLOW)

        self.play(Write(svg_05), run_time=4.0)

        self.play(Write(german_text), run_time=3.0)
        self.play(FadeIn(german_label), run_time=2.0)
        self.play(FadeIn(io_arrows), run_time=2.0)
        self.play(FadeIn(english_text), run_time=2.0)
        self.play(FadeIn(english_label), run_time=2.0)

        self.wait()

        self.play(
            FadeOut(german_text),
            FadeOut(german_label),
            FadeOut(english_text),
            FadeOut(english_label),
            run_time=2.0,
        )
        self.wait(2.0)
        self.play(Write(capital_text), run_time=2.0)
        self.wait(2.0)
        self.play(FadeIn(paris_text), run_time=2.0)

        svg_06.move_to([-3.44073984e-02,  2.32035210e-01, -3.29861139e-18])
        for mob in svg_06:
            mob.set_fill(opacity=0.1)

        # Clamp each side to both available panel space and the original side text width.
        MARGIN   = 0.8
        left_gap_w = abs(svg_05.get_left()[0] - (-7.1)) - MARGIN
        right_gap_w = abs(7.1 - svg_05.get_right()[0]) - MARGIN
        LEFT_MAX_TEXT_W = min(max(left_gap_w, 0.1), german_text.get_width())
        RIGHT_MAX_TEXT_W = min(max(right_gap_w, 0.1), english_text.get_width())
        LEFT_X   = (-7.1 + svg_05.get_left()[0]) / 2
        RIGHT_X  = (svg_05.get_right()[0] + 7.1) / 2



        self.play(FadeIn(svg_06), run_time=2.0)

        def make_input(txt):
            t = Text(f'"{txt}"', font="Georgia", slant=ITALIC, color=WHITE, font_size=22,
                     line_spacing_height=0.05)
            if t.get_width() > LEFT_MAX_TEXT_W:
                t.set_width(LEFT_MAX_TEXT_W)
            return t.move_to([LEFT_X, 0, 0])

        def make_output(txt):
            t = Text(f'"{txt}"', font="Georgia", slant=ITALIC, color=YELLOW, font_size=22,
                     line_spacing_height=0.05)
            if t.get_width() > RIGHT_MAX_TEXT_W:
                t.set_width(RIGHT_MAX_TEXT_W)
            return t.move_to([RIGHT_X, 0, 0])

        def rand_fill_anims(run_time):
            return [mob.animate(run_time=run_time).set_fill(opacity=random.uniform(0.05, 1.0)) for mob in svg_06]

        examples = CURATED_COMPLETIONS[1:]  # skip France/Paris, already shown above
        cur_in  = make_input(examples[0][0])
        cur_out = make_output(examples[0][1])

        self.wait()

        self.play(FadeOut(capital_text), FadeOut(paris_text), run_time=2.0)

        self.play(FadeIn(cur_in), FadeIn(cur_out), *rand_fill_anims(1.0), run_time=1.0)

        n_swaps = max(1, len(examples) - 1)
        hold_count = min(20, n_swaps)
        ramp_steps = max(1, n_swaps - hold_count)
        for i, (inp, out) in enumerate(examples[1:]):
            new_in  = make_input(inp)
            new_out = make_output(out)
            if i >= ramp_steps:
                hold_t = 0.15
            else:
                progress = i / max(1, ramp_steps - 1)
                hold_t = 0.15 + 0.85 * math.exp(-2.0 * progress)
            self.remove(cur_in, cur_out)
            self.add(new_in, new_out)
            self.play(*rand_fill_anims(hold_t), run_time=hold_t)
            cur_in, cur_out = new_in, new_out

        self.wait()

        self.play(FadeOut(cur_in), FadeOut(cur_out), FadeOut(io_arrows), FadeOut(svg_06), run_time=2.0)

        pan_right = (svg_05.get_right()[0] - self.frame.get_left()[0]) + 0.5
        self.play(self.frame.animate.shift(RIGHT * pan_right), run_time=5.0)

        svg_10.move_to([3.83159839e-01, 1.26843078e-01, 4.00618036e-18]).shift(RIGHT * pan_right)
        svg_09.shift(RIGHT * pan_right)
        self.play(Write(svg_09), run_time=6.0)

        self.wait(2.0)

        for bar in svg_10:
            bar.save_state()
            bottom = bar.get_bottom()
            bar.stretch(0, 1)
            bar.move_to(bottom, aligned_edge=DOWN)
        self.play(*[bar.animate.restore() for bar in svg_10], run_time=2.0)
        self.wait(3.0)
        focus = VGroup(svg_05, svg_09, svg_10)
        self.play(
            self.frame.animate.move_to(focus.get_center()).set_width(focus.get_width() + 2.0),
            run_time=2.0,
        )

        self.wait()

        self.play(FadeOut(svg_10), FadeOut(svg_09), run_time=2.0)

        svg_11_target = svg_11.copy().move_to(svg_05.get_center())

        def _cx(m): return m.get_center()[0]
        def _sort(mobs): return sorted(mobs, key=lambda m: (-round(_cx(m), 1), m.get_center()[1]))
        def _sample(lst, n):
            if len(lst) <= n: return lst
            step = len(lst) / n
            return [lst[int(i * step)] for i in range(n)]

        tgt_left  = _sort([m for m in svg_11_target if _cx(m) < -0.05])
        tgt_mid   = _sort([m for m in svg_11_target if -0.05 <= _cx(m) <= 0.15])
        tgt_right = _sort([m for m in svg_11_target if _cx(m) > 0.15])

        src_left  = _sample(_sort([m for m in svg_05 if _cx(m) < -0.2]),  len(tgt_left))
        src_mid   = _sample(_sort([m for m in svg_05 if -0.2 <= _cx(m) <= 0.2]), len(tgt_mid))
        src_right = _sample(_sort([m for m in svg_05 if _cx(m) > 0.2]),   len(tgt_right))

        selected = {id(m) for m in src_left + src_mid + src_right}
        fade_rest = VGroup(*[m for m in svg_05 if id(m) not in selected])

        self.play(
            ReplacementTransform(VGroup(*src_left),  VGroup(*tgt_left)),
            ReplacementTransform(VGroup(*src_mid),   VGroup(*tgt_mid)),
            ReplacementTransform(VGroup(*src_right), VGroup(*tgt_right)),
            FadeOut(fade_rest),
            self.frame.animate.move_to(svg_05.get_center()).set_width(svg_11_target.get_width() + 4),
            run_time=2.0,
        )

        self.embed()
