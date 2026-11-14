"""
P36 - LLM vs WORLD MODEL: text completion on top, novel-view "completion" below. One continuous take (ManimGL):

    manimgl <path to this file> P36

  beat 1  LLM row builds left to right: prompt -> arrow -> LLM box -> arrow -> completion
  beat 2  WORLD MODEL row builds the same way: three input views -> arrow -> box -> arrow -> the fourth view
  beat 3  flip through the examples, both rows changing together (input and output swap at once)

Line art is redrawn from p36.svg using its pixel coordinates (1920x1080 viewBox mapped onto manim's 8-unit-tall
frame). The mockup's hand-placed images / text are tidied onto slots: prompts are right-aligned to the LLM input
arrow, the three input views sit in equal slots, and every photo is fit inside its slot without cropping.

Photos: each folder in p36_multiview is one example; sorted by filename, the first three are the input views and the
last is the output. They're EXIF-rotated and downsized once into _p36_cache/ next to this file.

Examples, order, and timings are class attributes on P36.
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from manimlib import *

# ----------------------------------------------------------------------------- config
HERE = Path(__file__).resolve().parent
GRAPHICS = Path(os.environ.get(
    "WM_GRAPHICS", "/Users/stephen/Library/CloudStorage/Dropbox-Stephencwelch/welch_labs/world_models/graphics"))
MULTIVIEW = GRAPHICS / "p36_multiview"
CACHE = HERE / "_p36_cache"
CACHE_PX = 1400                     # long side of the cached photos

FRESH_TAN = "#f4ebd9"               # everything in this mockup is this color
LABEL_FONT = "Open Sans"            # LLM / WORLD MODEL (bold)
TEXT_FONT = "STIX Two Text"         # prompts + completions (italic)
CAP = {LABEL_FONT: 0.714, TEXT_FONT: 0.656}     # cap height / em, to match Illustrator's point sizes

# ----------------------------------------------------------------------------- mockup coordinates
PX = FRAME_HEIGHT / 1080            # manim units per mockup pixel
STROKE_PER_PX = 1 / 2.03            # manim stroke_width that renders 1 mockup pixel wide (measured at 1080p)


def P(x, y):
    """Mockup pixel (1920x1080, y down) -> manim point."""
    return np.array([(x - 960) * PX, (540 - y) * PX, 0.0])


def px_line(a, b, stroke_px, color=FRESH_TAN):
    return Line(P(*a), P(*b)).set_stroke(color, stroke_px * STROKE_PER_PX)


def arrow(tail, tip, stroke_px=2.0, head=(19.9, 8.13, 15.18), color=FRESH_TAN):
    """Mockup-style arrow: shaft + notched head. head = (length, half width, notch depth) in pixels."""
    tail, tip = np.array(tail, float), np.array(tip, float)
    u = (tip - tail) / np.linalg.norm(tip - tail)
    n = np.array([-u[1], u[0]])
    length, half, notch = head
    shaft = px_line(tail, tip - notch * u, stroke_px, color)
    barb = Polygon(P(*tip), P(*(tip - length * u + half * n)), P(*(tip - notch * u)), P(*(tip - length * u - half * n)))
    barb.set_fill(color, 1).set_stroke(color, 0)
    return VGroup(shaft, barb)


def box(x, y, w, h, r, stroke_px=2.5, color=FRESH_TAN):
    rect = RoundedRectangle(width=w * PX, height=h * PX, corner_radius=r * PX)
    rect.move_to(P(x + w / 2, y + h / 2))
    return rect.set_stroke(color, stroke_px * STROKE_PER_PX).set_fill(opacity=0)


def px_text(string, size_px, x, y, font=LABEL_FONT, bold=False, italic=False, align="left", color=FRESH_TAN):
    """Text at mockup size with its baseline at y. x is its left edge (align="left"), right edge ("right") or middle.

    A throwaway "H" is set in front of the string: its height fixes the scale and its bottom is the baseline, so
    strings with descenders (g, p, y, commas) sit correctly without per-string fudge factors.
    """
    full = Text("H" + string, font=font, weight=BOLD if bold else NORMAL, slant=ITALIC if italic else NORMAL,
                font_size=48)
    full.scale(CAP.get(font, 0.7) * size_px * PX / full[0].get_height())
    t = full[1:]
    edge_x = {"left": t.get_left()[0], "right": t.get_right()[0], "center": t.get_center()[0]}[align]
    target = P(x, y)
    t.shift((target[0] - edge_x) * RIGHT + (target[1] - full[0].get_bottom()[1]) * UP)
    return t.set_color(color)


# ----------------------------------------------------------------------------- photos
def cached_photo(path):
    """EXIF-upright, downsized copy of a photo (iPhone shots are stored rotated; textures don't need 12 MP)."""
    CACHE.mkdir(exist_ok=True)
    out = CACHE / f"{path.parent.name}_{path.stem}.jpg"
    if not out.exists() or out.stat().st_mtime < path.stat().st_mtime:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        im.thumbnail((CACHE_PX, CACHE_PX))
        im.save(out, quality=92)
    return out


def photo_in_slot(path, slot):
    """ImageMobject fit inside slot = (center x, center y, width, height) in pixels, aspect preserved, centered."""
    cx, cy, w, h = slot
    im = ImageMobject(str(cached_photo(path)))
    try:
        im.image.load()             # don't keep the file handle open
    except Exception:
        pass
    im.set_height(h * PX)
    if im.get_width() > w * PX:
        im.set_width(w * PX)
    return im.move_to(P(cx, cy))


def example_photos(name):
    """Sorted photo paths of one example folder: first three = input views, last = output view."""
    files = sorted(p for p in (MULTIVIEW / name).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(files) < 4:
        raise FileNotFoundError(f"{MULTIVIEW / name}: need 4 photos, found {len(files)}")
    return files[:3], files[-1]


# ----------------------------------------------------------------------------- layout (pixels)
LLM_Y, WM_Y = 344.0, 723.0                      # arrow heights of the two rows
TEXT_BASELINE = 356.0                           # prompt + completion baseline (x-height centered on the LLM arrows)
PROMPT_RIGHT, COMPLETION_LEFT = 1040.0, 1432.0  # prompts end here / completions start here
IN_W, IN_H, IN_GAP, IN_LEFT = 316.0, 237.0, 24.0, 27.0     # three input-view slots
IN_SLOTS = [(IN_LEFT + IN_W / 2 + k * (IN_W + IN_GAP), WM_Y, IN_W, IN_H) for k in range(3)]
OUT_SLOT = (1615.7, WM_Y, 348.0, 261.0)


# ----------------------------------------------------------------------------- the scene
class P36(InteractiveScene):
    # --- content -------------------------------------------------------------------------------------------
    texts = [("The capital of France is", "Paris"),
             ("The Great Wall is located in", "China"),
             ("The chemical symbol for gold is", "Au"),
             ("Ice is the solid form of", "water"),
             ("The Berlin Wall fell in", "1989")]
    image_sets = ["office", "car", "house", "playset", "turtle", "book"]     # folders in p36_multiview
    # Each flip shows texts[i % len(texts)] with image_sets[i % len(image_sets)], for num_examples flips in total
    # (None = one pass through the longer list, so with 5 texts and 6 image sets the first text comes round again).
    num_examples = None
    # --- timings (seconds) ---------------------------------------------------------------------------------
    build_time = 1.0          # each piece of a row during the build-in
    row_hold = 1.5            # after the LLM row is complete, before the world model row starts
    example_hold = 2.5        # each example on screen
    flip_time = 0.3           # crossfade between examples; 0 = hard cut
    end_hold = 5.0

    def construct(self):
        self.build()
        self.beat_llm_row()
        self.beat_world_model_row()
        self.beat_flip()
        self.wait(self.end_hold)

    # ------------------------------------------------------------------------------------------- pieces
    def build(self):
        self.llm_box = box(1148.15, 272.61, 172.97, 146.51, 11.73)
        self.wm_box = box(1148.15, 648.38, 172.97, 146.51, 11.73)
        box_cx = 1148.15 + 172.97 / 2
        self.llm_label = px_text("LLM", 36.58, box_cx, 361.73, bold=True, align="center")
        self.wm_label = VGroup(px_text("WORLD", 32, box_cx, 713.31, bold=True, align="center"),
                               px_text("MODEL", 32, box_cx, 751.71, bold=True, align="center"))
        self.llm_in = arrow((1064.52, LLM_Y), (1119.15, LLM_Y))
        self.llm_out = arrow((1346.36, LLM_Y), (1400.99, LLM_Y))
        self.wm_in = arrow((1064.52, WM_Y), (1119.15, WM_Y))
        self.wm_out = arrow((1346.36, WM_Y), (1400.99, WM_Y))

        n = self.num_examples or max(len(self.texts), len(self.image_sets))
        self.examples = []
        for i in range(n):
            prompt, completion = self.texts[i % len(self.texts)]
            inputs, output = example_photos(self.image_sets[i % len(self.image_sets)])
            self.examples.append({
                "prompt": px_text(prompt, 36, PROMPT_RIGHT, TEXT_BASELINE, TEXT_FONT, italic=True, align="right"),
                "completion": px_text(completion, 36, COMPLETION_LEFT, TEXT_BASELINE, TEXT_FONT, italic=True),
                "inputs": Group(*[photo_in_slot(p, s) for p, s in zip(inputs, IN_SLOTS)]),
                "output": photo_in_slot(output, OUT_SLOT),
            })

    # ------------------------------------------------------------------------------------------- beat 1
    def beat_llm_row(self):
        ex, t = self.examples[0], self.build_time
        self.play(Write(ex["prompt"]), run_time=t)
        self.play(ShowCreation(self.llm_in[0]), FadeIn(self.llm_in[1]),
                  ShowCreation(self.llm_box), Write(self.llm_label), run_time=t)
        self.play(ShowCreation(self.llm_out[0]), FadeIn(self.llm_out[1]), run_time=t / 2)
        self.play(Write(ex["completion"]), run_time=t)
        self.wait(self.row_hold)

    # ------------------------------------------------------------------------------------------- beat 2
    def beat_world_model_row(self):
        ex, t = self.examples[0], self.build_time
        self.play(LaggedStart(*[FadeIn(im) for im in ex["inputs"]], lag_ratio=0.4), run_time=1.5 * t)
        self.play(ShowCreation(self.wm_in[0]), FadeIn(self.wm_in[1]),
                  ShowCreation(self.wm_box), Write(self.wm_label), run_time=t)
        self.play(ShowCreation(self.wm_out[0]), FadeIn(self.wm_out[1]), run_time=t / 2)
        self.play(FadeIn(ex["output"]), run_time=t)
        self.wait(self.example_hold)

    # ------------------------------------------------------------------------------------------- beat 3
    def beat_flip(self):
        def parts(ex):
            return [ex["prompt"], ex["completion"], ex["inputs"], ex["output"]]

        for prev, ex in zip(self.examples, self.examples[1:]):
            if self.flip_time > 0:
                self.play(*[FadeOut(m) for m in parts(prev)], *[FadeIn(m) for m in parts(ex)], run_time=self.flip_time)
            else:
                self.remove(*parts(prev))
                self.add(*parts(ex))
            self.wait(self.example_hold)
