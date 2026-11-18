from welch_paths import welch_path
import manimgx as gx
from welch_scene import WelchScene
from tqdm import tqdm
import re
import random
import math
from functools import lru_cache
from os import environ
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

SVG_DIR = Path(environ.get(
    "WELCH_JEPA_P20_SVGS",
    welch_path('jepa/graphics/p20_23_to_manim'),
)).expanduser()


def load_panel(number):
    return gx.SVGMobject(SVG_DIR / f"p20_23_to_manim-{number:02d}.svg")[1:].scale(4)


class P20(WelchScene):
    def construct(self):
        svg_05 = load_panel(5)

        svg_05.move_to(gx.ORIGIN)
        self.play(gx.Write(svg_05))

class P20_21(WelchScene):
    def construct(self):
        svg_03 = load_panel(3)
        svg_05 = load_panel(5)

        german_text  = gx.VGroup(*[svg_03[i] for i in list(range(0,20))  + list(range(20,38)) + list(range(38,59)) + list(range(59,78)) + list(range(78,94))])
        english_text = gx.VGroup(*[svg_03[i] for i in list(range(94,117)) + list(range(117,142)) + list(range(142,164)) + list(range(164,198))])
        german_label = gx.VGroup(*[svg_03[i] for i in range(198, 204)])
        english_label= gx.VGroup(*[svg_03[i] for i in range(204, 211)])

        german_text.to_edge(gx.LEFT)
        german_label.next_to(german_text, gx.DOWN, buff=0.2)
        english_text.to_edge(gx.RIGHT)
        english_label.next_to(english_text, gx.DOWN, buff=0.2)

        svg_05.move_to(gx.ORIGIN)

        arrow_len = 0.4
        left_mid  = (german_text.get_right() + svg_05.get_left()) / 2
        right_mid = (svg_05.get_right() + english_text.get_left()) / 2
        left_mid[1] = right_mid[1] = 0  # keep vertical centered

        left_arrow  = gx.Arrow(left_mid  + gx.LEFT * arrow_len / 2, left_mid  + gx.RIGHT * arrow_len / 2, buff=0).set_color(CHILL_BROWN)
        right_arrow = gx.Arrow(right_mid + gx.LEFT * arrow_len / 2, right_mid + gx.RIGHT * arrow_len / 2, buff=0).set_color(CHILL_BROWN)

        capital_text = gx.Text(
            '"The capital of France is"',
            font="Georgia", slant=gx.ITALIC, color=gx.WHITE, font_size=32,
        ).scale_to_fit_width(german_text.width).move_to([german_text.get_center()[0], 0, 0])

        paris_text = gx.Text(
            '"Paris"',
            font="Georgia", slant=gx.ITALIC, font_size=32,
        ).move_to([english_text.get_center()[0], 0, 0]).set_color(YELLOW)

        self.play(gx.Write(svg_05), run_time=4)

        self.play(gx.Write(german_text), run_time=4)
        self.play(gx.FadeIn(german_label), run_time=2)
        self.play(gx.GrowArrow(left_arrow))
        self.play(gx.GrowArrow(right_arrow))
        self.play(gx.ReplacementTransform(german_text.copy(), english_text), run_time=3)
        self.play(gx.FadeIn(english_label), run_time=2)

        self.play(
            gx.ReplacementTransform(german_text, capital_text),
            gx.FadeOut(german_label),
            run_time=2,
        )
        self.play(
            gx.ReplacementTransform(capital_text.copy(), paris_text),
            gx.FadeOut(english_text),
            gx.FadeOut(english_label),
            run_time=3,
        )


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

@lru_cache(maxsize=1)
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


class P20_22a(WelchScene):
    def construct(self):
        svg_03 = load_panel(3)
        svg_05 = load_panel(5)
        svg_06 = load_panel(6)
        svg_07 = load_panel(7)

        german_text  = gx.VGroup(*[svg_03[i] for i in list(range(0,20))  + list(range(20,38)) + list(range(38,59)) + list(range(59,78)) + list(range(78,94))])
        english_text = gx.VGroup(*[svg_03[i] for i in list(range(94,117)) + list(range(117,142)) + list(range(142,164)) + list(range(164,198))])
        german_label = gx.VGroup(*[svg_03[i] for i in range(198, 204)])
        english_label= gx.VGroup(*[svg_03[i] for i in range(204, 211)])

        german_text.to_edge(gx.LEFT)
        german_label.next_to(german_text, gx.DOWN, buff=0.2)
        english_text.to_edge(gx.RIGHT)
        english_label.next_to(english_text, gx.DOWN, buff=0.2)

        svg_05.move_to(gx.ORIGIN)

        left_mid  = (german_text.get_right() + svg_05.get_left()) / 2
        right_mid = (svg_05.get_right() + english_text.get_left()) / 2
        left_mid[1] = right_mid[1] = 0  # keep vertical centered

        io_arrows = svg_07.copy().set_color(CHILL_BROWN)
        arrows_mid = (left_mid + right_mid) / 2
        io_arrows.move_to(arrows_mid)

        capital_text = gx.Text(
            '"The capital of France is"',
            font="Georgia", slant=gx.ITALIC, color=gx.WHITE, font_size=32,
        ).scale_to_fit_width(german_text.width).move_to([german_text.get_center()[0], 0, 0])

        paris_text = gx.Text(
            '"Paris"',
            font="Georgia", slant=gx.ITALIC, font_size=32,
        ).move_to([english_text.get_center()[0], 0, 0]).set_color(YELLOW)

        self.play(gx.Write(svg_05), run_time=4)

        self.play(gx.Write(german_text), run_time=4)
        self.play(gx.FadeIn(german_label), run_time=2)
        self.play(gx.Create(io_arrows))
        self.play(gx.ReplacementTransform(german_text.copy(), english_text), run_time=3)
        self.play(gx.FadeIn(english_label), run_time=2)

        self.play(
            gx.ReplacementTransform(german_text, capital_text),
            gx.FadeOut(german_label),
            run_time=2,
        )
        self.play(
            gx.ReplacementTransform(capital_text.copy(), paris_text),
            gx.FadeOut(english_text),
            gx.FadeOut(english_label),
            run_time=3,
        )


        svg_06.move_to([-3.44073984e-02,  2.49935210e-01, -3.29861139e-18])
        for mob in svg_06:
            mob.set_fill(opacity=0.1)

        # Clamp each side to both available panel space and the original side text width.
        MARGIN   = 0.8
        left_gap_w = abs(svg_05.get_left()[0] - (-7.1)) - MARGIN
        right_gap_w = abs(7.1 - svg_05.get_right()[0]) - MARGIN
        LEFT_MAX_TEXT_W = min(max(left_gap_w, 0.1), german_text.width)
        RIGHT_MAX_TEXT_W = min(max(right_gap_w, 0.1), english_text.width)
        LEFT_X   = (-7.1 + svg_05.get_left()[0]) / 2
        RIGHT_X  = (svg_05.get_right()[0] + 7.1) / 2



        self.play(gx.FadeIn(svg_06))

        def make_input(txt):
            t = gx.Text(f'"{txt}"', font="Georgia", slant=gx.ITALIC, color=gx.WHITE, font_size=22,
                     line_spacing=0.05)
            if t.width > LEFT_MAX_TEXT_W:
                t.scale_to_fit_width(LEFT_MAX_TEXT_W)
            return t.move_to([LEFT_X, 0, 0])

        def make_output(txt):
            t = gx.Text(f'"{txt}"', font="Georgia", slant=gx.ITALIC, color=YELLOW, font_size=22,
                     line_spacing=0.05)
            if t.width > RIGHT_MAX_TEXT_W:
                t.scale_to_fit_width(RIGHT_MAX_TEXT_W)
            return t.move_to([RIGHT_X, 0, 0])

        def rand_fill_anims():
            return [mob.animate.set_fill(opacity=random.uniform(0.05, 1.0)) for mob in svg_06]

        examples = _load_wikitext_completions() or CURATED_COMPLETIONS
        cur_in  = make_input(examples[0][0])
        cur_out = make_output(examples[0][1])

        self.play(gx.FadeOut(capital_text), gx.FadeOut(paris_text))

        self.play(gx.FadeIn(cur_in), gx.FadeIn(cur_out), *rand_fill_anims(), run_time=0.5)

        for inp, out in examples[1:]:
            new_in  = make_input(inp)
            new_out = make_output(out)
            self.play(
                gx.FadeTransform(cur_in, new_in),
                gx.FadeTransform(cur_out, new_out),
                *rand_fill_anims(),
                run_time=0.35,
            )
            cur_in, cur_out = new_in, new_out


class P20_22a(WelchScene):
    def construct(self):
        svg_03 = load_panel(3)
        svg_05 = load_panel(5)
        svg_06 = load_panel(6)

        german_text  = gx.VGroup(*[svg_03[i] for i in list(range(0,20))  + list(range(20,38)) + list(range(38,59)) + list(range(59,78)) + list(range(78,94))])
        english_text = gx.VGroup(*[svg_03[i] for i in list(range(94,117)) + list(range(117,142)) + list(range(142,164)) + list(range(164,198))])
        german_label = gx.VGroup(*[svg_03[i] for i in range(198, 204)])
        english_label= gx.VGroup(*[svg_03[i] for i in range(204, 211)])

        german_text.to_edge(gx.LEFT)
        german_label.next_to(german_text, gx.DOWN, buff=0.2)
        english_text.to_edge(gx.RIGHT)
        english_label.next_to(english_text, gx.DOWN, buff=0.2)

        svg_05.move_to(gx.ORIGIN)

        arrow_len = 0.4
        left_mid  = (german_text.get_right() + svg_05.get_left()) / 2
        right_mid = (svg_05.get_right() + english_text.get_left()) / 2
        left_mid[1] = right_mid[1] = 0  # keep vertical centered

        left_arrow  = gx.Arrow(left_mid  + gx.LEFT * arrow_len / 2, left_mid  + gx.RIGHT * arrow_len / 2, buff=0).set_color(CHILL_BROWN)
        right_arrow = gx.Arrow(right_mid + gx.LEFT * arrow_len / 2, right_mid + gx.RIGHT * arrow_len / 2, buff=0).set_color(CHILL_BROWN)

        capital_text = gx.Text(
            '"The capital of France is"',
            font="Georgia", slant=gx.ITALIC, color=gx.WHITE, font_size=32,
        ).scale_to_fit_width(german_text.width).move_to([german_text.get_center()[0], 0, 0])

        paris_text = gx.Text(
            '"Paris"',
            font="Georgia", slant=gx.ITALIC, font_size=32,
        ).move_to([english_text.get_center()[0], 0, 0]).set_color(YELLOW)

        self.play(gx.Write(svg_05), run_time=4)

        self.play(gx.Write(german_text), run_time=4)
        self.play(gx.FadeIn(german_label), run_time=2)
        self.play(gx.GrowArrow(left_arrow))
        self.play(gx.GrowArrow(right_arrow))
        self.play(gx.ReplacementTransform(german_text.copy(), english_text), run_time=3)
        self.play(gx.FadeIn(english_label), run_time=2)

        self.play(
            gx.ReplacementTransform(german_text, capital_text),
            gx.FadeOut(german_label),
            run_time=2,
        )
        self.play(
            gx.ReplacementTransform(capital_text.copy(), paris_text),
            gx.FadeOut(english_text),
            gx.FadeOut(english_label),
            run_time=3,
        )


        svg_06.move_to([-3.44073984e-02,  2.49935210e-01, -3.29861139e-18])
        for mob in svg_06:
            mob.set_fill(opacity=0.1)

        # Clamp each side to both available panel space and the original side text width.
        MARGIN   = 0.8
        left_gap_w = abs(svg_05.get_left()[0] - (-7.1)) - MARGIN
        right_gap_w = abs(7.1 - svg_05.get_right()[0]) - MARGIN
        LEFT_MAX_TEXT_W = min(max(left_gap_w, 0.1), german_text.width)
        RIGHT_MAX_TEXT_W = min(max(right_gap_w, 0.1), english_text.width)
        LEFT_X   = (-7.1 + svg_05.get_left()[0]) / 2
        RIGHT_X  = (svg_05.get_right()[0] + 7.1) / 2



        self.play(gx.FadeIn(svg_06))

        def make_input(txt):
            t = gx.Text(f'"{txt}"', font="Georgia", slant=gx.ITALIC, color=gx.WHITE, font_size=22,
                     line_spacing=0.05)
            if t.width > LEFT_MAX_TEXT_W:
                t.scale_to_fit_width(LEFT_MAX_TEXT_W)
            return t.move_to([LEFT_X, 0, 0])

        def make_output(txt):
            t = gx.Text(f'"{txt}"', font="Georgia", slant=gx.ITALIC, color=YELLOW, font_size=22,
                     line_spacing=0.05)
            if t.width > RIGHT_MAX_TEXT_W:
                t.scale_to_fit_width(RIGHT_MAX_TEXT_W)
            return t.move_to([RIGHT_X, 0, 0])

        def rand_fill_anims():
            return [mob.animate.set_fill(opacity=random.uniform(0.05, 1.0)) for mob in svg_06]

        examples = _load_wikitext_completions() or CURATED_COMPLETIONS
        cur_in  = make_input(examples[0][0])
        cur_out = make_output(examples[0][1])

        self.play(gx.FadeOut(capital_text), gx.FadeOut(paris_text))

        self.play(gx.FadeIn(cur_in), gx.FadeIn(cur_out), *rand_fill_anims(), run_time=0.5)

        for inp, out in examples[1:]:
            new_in  = make_input(inp)
            new_out = make_output(out)
            self.play(
                gx.FadeTransform(cur_in, new_in),
                gx.FadeTransform(cur_out, new_out),
                *rand_fill_anims(),
                run_time=0.35,
            )
            cur_in, cur_out = new_in, new_out

        self.play(gx.FadeOut(cur_in), gx.FadeOut(cur_out), gx.FadeOut(left_arrow), gx.FadeOut(right_arrow), gx.FadeOut(svg_06))

        self.play(
            self.camera.frame.animate.move_to(svg_05.get_center()).scale_to_fit_width(svg_05.width + 2.0),
            run_time=2.0,
        )


class P20_22(WelchScene):
    def construct(self):
        svg_03 = load_panel(3)
        svg_05 = load_panel(5)
        svg_06 = load_panel(6)
        svg_09 = load_panel(9)
        svg_10 = load_panel(10)

        german_text  = gx.VGroup(*[svg_03[i] for i in list(range(0,20))  + list(range(20,38)) + list(range(38,59)) + list(range(59,78)) + list(range(78,94))])
        english_text = gx.VGroup(*[svg_03[i] for i in list(range(94,117)) + list(range(117,142)) + list(range(142,164)) + list(range(164,198))])
        german_label = gx.VGroup(*[svg_03[i] for i in range(198, 204)])
        english_label= gx.VGroup(*[svg_03[i] for i in range(204, 211)])

        german_text.to_edge(gx.LEFT)
        german_label.next_to(german_text, gx.DOWN, buff=0.2)
        english_text.to_edge(gx.RIGHT)
        english_label.next_to(english_text, gx.DOWN, buff=0.2)

        svg_05.move_to(gx.ORIGIN)

        arrow_len = 0.4
        left_mid  = (german_text.get_right() + svg_05.get_left()) / 2
        right_mid = (svg_05.get_right() + english_text.get_left()) / 2
        left_mid[1] = right_mid[1] = 0  # keep vertical centered

        left_arrow  = gx.Arrow(left_mid  + gx.LEFT * arrow_len / 2, left_mid  + gx.RIGHT * arrow_len / 2, buff=0).set_color(CHILL_BROWN)
        right_arrow = gx.Arrow(right_mid + gx.LEFT * arrow_len / 2, right_mid + gx.RIGHT * arrow_len / 2, buff=0).set_color(CHILL_BROWN)

        capital_text = gx.Text(
            '"The capital of France is"',
            font="Georgia", slant=gx.ITALIC, color=gx.WHITE, font_size=32,
        ).scale_to_fit_width(german_text.width).move_to([german_text.get_center()[0], 0, 0])

        paris_text = gx.Text(
            '"Paris"',
            font="Georgia", slant=gx.ITALIC, font_size=32,
        ).move_to([english_text.get_center()[0], 0, 0]).set_color(YELLOW)

        self.play(gx.Write(svg_05), run_time=4)

        self.play(gx.Write(german_text), run_time=4)
        self.play(gx.FadeIn(german_label), run_time=2)
        self.play(gx.GrowArrow(left_arrow))
        self.play(gx.GrowArrow(right_arrow))
        self.play(gx.ReplacementTransform(german_text.copy(), english_text), run_time=3)
        self.play(gx.FadeIn(english_label), run_time=2)

        self.play(
            gx.ReplacementTransform(german_text, capital_text),
            gx.FadeOut(german_label),
            run_time=2,
        )
        self.play(
            gx.ReplacementTransform(capital_text.copy(), paris_text),
            gx.FadeOut(english_text),
            gx.FadeOut(english_label),
            run_time=3,
        )


        svg_06.move_to([-3.44073984e-02,  2.49935210e-01, -3.29861139e-18])
        for mob in svg_06:
            mob.set_fill(opacity=0.1)

        # Clamp each side to both available panel space and the original side text width.
        MARGIN   = 0.8
        left_gap_w = abs(svg_05.get_left()[0] - (-7.1)) - MARGIN
        right_gap_w = abs(7.1 - svg_05.get_right()[0]) - MARGIN
        LEFT_MAX_TEXT_W = min(max(left_gap_w, 0.1), german_text.width)
        RIGHT_MAX_TEXT_W = min(max(right_gap_w, 0.1), english_text.width)
        LEFT_X   = (-7.1 + svg_05.get_left()[0]) / 2
        RIGHT_X  = (svg_05.get_right()[0] + 7.1) / 2



        self.play(gx.FadeIn(svg_06))

        def make_input(txt):
            t = gx.Text(f'"{txt}"', font="Georgia", slant=gx.ITALIC, color=gx.WHITE, font_size=22,
                     line_spacing=0.05)
            if t.width > LEFT_MAX_TEXT_W:
                t.scale_to_fit_width(LEFT_MAX_TEXT_W)
            return t.move_to([LEFT_X, 0, 0])

        def make_output(txt):
            t = gx.Text(f'"{txt}"', font="Georgia", slant=gx.ITALIC, color=YELLOW, font_size=22,
                     line_spacing=0.05)
            if t.width > RIGHT_MAX_TEXT_W:
                t.scale_to_fit_width(RIGHT_MAX_TEXT_W)
            return t.move_to([RIGHT_X, 0, 0])

        def rand_fill_anims():
            return [mob.animate.set_fill(opacity=random.uniform(0.05, 1.0)) for mob in svg_06]

        examples = _load_wikitext_completions() or CURATED_COMPLETIONS
        cur_in  = make_input(examples[0][0])
        cur_out = make_output(examples[0][1])

        self.play(gx.FadeOut(capital_text), gx.FadeOut(paris_text))

        self.play(gx.FadeIn(cur_in), gx.FadeIn(cur_out), *rand_fill_anims(), run_time=0.5)

        for inp, out in examples[1:]:
            new_in  = make_input(inp)
            new_out = make_output(out)

            self.play(
                gx.FadeTransform(cur_in, new_in),
                gx.FadeTransform(cur_out, new_out),
                *rand_fill_anims(),
                run_time=0.35,
            )
            cur_in, cur_out = new_in, new_out


        self.play(gx.FadeOut(cur_in), gx.FadeOut(cur_out), gx.FadeOut(left_arrow), gx.FadeOut(right_arrow), gx.FadeOut(svg_06))

        pan_right = (svg_05.get_right()[0] - self.camera.frame.get_left()[0]) + 0.5
        self.play(self.camera.frame.animate.shift(gx.RIGHT * pan_right), run_time=2.0)

        svg_10.move_to([3.83159839e-01, 1.26843078e-01, 4.00618036e-18]).shift(gx.RIGHT * pan_right)
        svg_09.shift(gx.RIGHT * pan_right)
        self.play(gx.Write(svg_09))

        self.wait()

        self.play(gx.Write(svg_10))
        focus = gx.VGroup(svg_05, svg_09, svg_10)
        self.play(
            self.camera.frame.animate.move_to(focus.get_center()).scale_to_fit_width(focus.width + 2.0),
            run_time=2.0,
        )



class P20_23(WelchScene):
    def construct(self):
        svg_03 = load_panel(3)
        svg_05 = load_panel(5)
        svg_06 = load_panel(6)
        svg_07 = load_panel(7)
        svg_09 = load_panel(9)
        svg_10 = load_panel(10)
        svg_11 = load_panel(11)

        german_text  = gx.VGroup(*[svg_03[i] for i in list(range(0,20))  + list(range(20,38)) + list(range(38,59)) + list(range(59,78)) + list(range(78,94))])
        english_text = gx.VGroup(*[svg_03[i] for i in list(range(94,117)) + list(range(117,142)) + list(range(142,164)) + list(range(164,198))])
        german_label = gx.VGroup(*[svg_03[i] for i in range(198, 204)])
        english_label= gx.VGroup(*[svg_03[i] for i in range(204, 211)])

        german_text.to_edge(gx.LEFT)
        german_label.next_to(german_text, gx.DOWN, buff=0.2)
        english_text.to_edge(gx.RIGHT)
        english_label.next_to(english_text, gx.DOWN, buff=0.2)

        svg_05.move_to(gx.ORIGIN)

        left_mid  = (german_text.get_right() + svg_05.get_left()) / 2
        right_mid = (svg_05.get_right() + english_text.get_left()) / 2
        left_mid[1] = right_mid[1] = 0  # keep vertical centered

        io_arrows = svg_07.copy().set_color(CHILL_BROWN)
        arrows_mid = (left_mid + right_mid) / 2
        io_arrows.move_to(arrows_mid)

        capital_text = gx.Text(
            '"The capital of France is"',
            font="Georgia", slant=gx.ITALIC, color=gx.WHITE, font_size=32,
        ).scale_to_fit_width(german_text.width).move_to([german_text.get_center()[0], 0, 0])

        paris_text = gx.Text(
            '"Paris"',
            font="Georgia", slant=gx.ITALIC, font_size=32,
        ).move_to([english_text.get_center()[0], 0, 0]).set_color(YELLOW)

        self.play(gx.Write(svg_05), run_time=10.0)

        self.wait(1)

        self.play(
            gx.Write(german_text),
            gx.Write(german_label),
            gx.Write(io_arrows),
            gx.Write(english_text),
            gx.Write(english_label),
        )

        self.wait()

        self.play(
            gx.FadeOut(german_text),
            gx.FadeOut(german_label),
            gx.FadeOut(english_text),
            gx.FadeOut(english_label),
            run_time=2.0,
        )
        self.wait(2.0)
        self.play(gx.Write(capital_text), run_time=2.0)
        self.wait(2.0)
        self.play(gx.FadeIn(paris_text), run_time=2.0)

        svg_06.move_to([-3.44073984e-02,  2.32035210e-01, -3.29861139e-18])
        for mob in svg_06:
            mob.set_fill(opacity=0.1)

        # Clamp each side to both available panel space and the original side text width.
        MARGIN   = 0.8
        left_gap_w = abs(svg_05.get_left()[0] - (-7.1)) - MARGIN
        right_gap_w = abs(7.1 - svg_05.get_right()[0]) - MARGIN
        LEFT_MAX_TEXT_W = min(max(left_gap_w, 0.1), german_text.width)
        RIGHT_MAX_TEXT_W = min(max(right_gap_w, 0.1), english_text.width)
        LEFT_X   = (-7.1 + svg_05.get_left()[0]) / 2
        RIGHT_X  = (svg_05.get_right()[0] + 7.1) / 2



        self.play(gx.FadeIn(svg_06), run_time=2.0)

        def make_input(txt):
            t = gx.Text(f'"{txt}"', font="Georgia", slant=gx.ITALIC, color=gx.WHITE, font_size=22,
                     line_spacing=0.05)
            if t.width > LEFT_MAX_TEXT_W:
                t.scale_to_fit_width(LEFT_MAX_TEXT_W)
            return t.move_to([LEFT_X, 0, 0])

        def make_output(txt):
            t = gx.Text(f'"{txt}"', font="Georgia", slant=gx.ITALIC, color=YELLOW, font_size=22,
                     line_spacing=0.05)
            if t.width > RIGHT_MAX_TEXT_W:
                t.scale_to_fit_width(RIGHT_MAX_TEXT_W)
            return t.move_to([RIGHT_X, 0, 0])

        def rand_fill_anims(run_time):
            return [mob.animate(run_time=run_time).set_fill(opacity=random.uniform(0.05, 1.0)) for mob in svg_06]

        examples = CURATED_COMPLETIONS[1:]  # skip France/Paris, already shown above
        cur_in  = make_input(examples[0][0])
        cur_out = make_output(examples[0][1])

        self.wait()

        self.play(gx.FadeOut(capital_text), gx.FadeOut(paris_text), run_time=2.0)

        self.play(gx.FadeIn(cur_in), gx.FadeIn(cur_out), *rand_fill_anims(1.0), run_time=1.0)

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

        self.play(gx.FadeOut(cur_in), gx.FadeOut(cur_out), gx.FadeOut(io_arrows), gx.FadeOut(svg_06), run_time=2.0)

        pan_right = (svg_05.get_right()[0] - self.camera.frame.get_left()[0]) + 0.5
        self.play(self.camera.frame.animate.shift(gx.RIGHT * pan_right), run_time=5.0)

        svg_10.move_to([3.83159839e-01, 1.26843078e-01, 4.00618036e-18]).shift(gx.RIGHT * pan_right)
        svg_09.shift(gx.RIGHT * pan_right)
        self.play(gx.Write(svg_09), run_time=6.0)

        self.wait(2.0)

        for bar in svg_10:
            bar.save_state()
            bottom = bar.get_bottom()
            bar.stretch(0, 1)
            bar.move_to(bottom, aligned_edge=gx.DOWN)
        self.play(*[bar.animate.restore() for bar in svg_10], run_time=2.0)
        self.wait(3.0)
        focus = gx.VGroup(svg_05, svg_09, svg_10)
        self.play(
            self.camera.frame.animate.move_to(focus.get_center()).scale_to_fit_width(focus.width + 2.0),
            run_time=2.0,
        )

        self.wait()

        self.play(gx.FadeOut(svg_10), gx.FadeOut(svg_09), run_time=2.0)

        svg_11_target = svg_11.copy().move_to(svg_05.get_center())

        def pair_group(svg_obj, pairs):
            return [gx.VGroup(svg_obj[a], svg_obj[b]) for a, b in pairs]

        # 8 circles (inner + outer ring pairs), spread across the full transformer.
        src_circle_pairs = [
            (219, 220),
            (217, 218),
            (225, 226),
            (211, 212),
            (675, 676),
            (673, 674),
            (681, 682),
            (667, 668),
        ]
        tgt_circle_pairs = [
            (12, 13),
            (14, 15),
            (16, 17),
            (38, 39),
            (28, 29),
            (26, 27),
            (36, 37),
            (34, 35),
        ]

        # 12 line pairs chosen for angle/shape similarity to target edges.
        src_line_pairs = [
            (530, 531),  # ~155 deg
            (587, 588),  # ~25 deg
            (131, 132),  # ~25 deg
            (631, 632),  # ~-25 deg
            (364, 365),  # ~121 deg
            (368, 369),  # ~59 deg
            (74, 75),    # ~155 deg
            (591, 592),  # ~-156 deg
            (486, 487),  # ~-156 deg
            (824, 825),  # ~-121 deg
            (179, 180),  # ~-24 deg
            (820, 821),  # ~121 deg
        ]
        tgt_line_pairs = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9),
            (10, 11),
            (18, 19),
            (20, 21),
            (22, 23),
            (24, 25),
            (30, 31),
            (32, 33),
        ]

        src_circles = pair_group(svg_05, src_circle_pairs)
        tgt_circles = pair_group(svg_11_target, tgt_circle_pairs)
        src_lines = pair_group(svg_05, src_line_pairs)
        tgt_lines = pair_group(svg_11_target, tgt_line_pairs)

        selected_indices = {
            idx
            for pair in (src_circle_pairs + src_line_pairs)
            for idx in pair
        }
        fade_rest = gx.VGroup(*[m for i, m in enumerate(svg_05) if i not in selected_indices])

        self.play(
            self.camera.frame.animate.move_to(svg_05.get_center()).scale_to_fit_width(svg_05.width + 2.0),
            run_time=1.5,
        )
        self.play(
            gx.FadeOut(fade_rest),
            *[gx.ReplacementTransform(src, tgt) for src, tgt in zip(src_lines, tgt_lines)],
            *[gx.ReplacementTransform(src, tgt) for src, tgt in zip(src_circles, tgt_circles)],
            run_time=2.0,
        )
        self.play(
            self.camera.frame.animate.move_to(svg_05.get_center()).scale_to_fit_width(svg_11_target.width + 4),
            run_time=1.5,
        )
