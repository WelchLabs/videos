"""
P36 - LLM vs WORLD MODEL: text completion on top, novel-view "completion" below. One continuous take (ManimGX):

    uv run manimgx render p36.py P36

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
from pathlib import Path

import manimgx as m

from assets import GRAPHICS
from welch_scene import cache_upright_photo
from welch_storyboard import (
    PX, STORYBOARD_BACKGROUND, TEXT_FONT, load_image, px_arrow as arrow, px_box as box,
    px_text, storyboard_point as P,
)

# ----------------------------------------------------------------------------- config
HERE = Path(__file__).resolve().parent
MULTIVIEW = GRAPHICS / "p36_multiview"
CACHE = HERE / "_p36_cache"
CACHE_PX = 1400                     # long side of the cached photos

# ----------------------------------------------------------------------------- photos
def cached_photo(path):
    """EXIF-upright, downsized copy of a photo (iPhone shots are stored rotated; textures don't need 12 MP)."""
    out = CACHE / f"{path.parent.name}_{path.stem}.jpg"
    return cache_upright_photo(path, out, CACHE_PX, quality=92)


def photo_in_slot(path, slot):
    """ImageMobject fit inside slot = (center x, center y, width, height) in pixels, aspect preserved, centered."""
    cx, cy, w, h = slot
    im = load_image(path if path.parent == CACHE else cached_photo(path))
    im.scale_to_fit_height(h * PX)
    if im.width > w * PX:
        im.scale_to_fit_width(w * PX)
    return im.move_to(P(cx, cy))


def example_photos(name):
    """Sorted photo paths of one example folder: first three = input views, last = output view."""
    source = MULTIVIEW / name
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    else:
        files = sorted(CACHE.glob(f"{name}_*.jpg"))
    if len(files) < 4:
        raise FileNotFoundError(f"{source}: need 4 photos, found {len(files)}")
    return files[:3], files[-1]


# ----------------------------------------------------------------------------- layout (pixels)
LLM_Y, WM_Y = 344.0, 723.0                      # arrow heights of the two rows
TEXT_BASELINE = 356.0                           # prompt + completion baseline (x-height centered on the LLM arrows)
PROMPT_RIGHT, COMPLETION_LEFT = 1040.0, 1432.0  # prompts end here / completions start here
IN_W, IN_H, IN_GAP, IN_LEFT = 316.0, 237.0, 24.0, 27.0     # three input-view slots
IN_SLOTS = [(IN_LEFT + IN_W / 2 + k * (IN_W + IN_GAP), WM_Y, IN_W, IN_H) for k in range(3)]
OUT_SLOT = (1615.7, WM_Y, 348.0, 261.0)


# ----------------------------------------------------------------------------- the scene
class P36(m.Scene):
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

    def construct(self) -> None:
        self.camera.background_color = STORYBOARD_BACKGROUND
        self.build()
        self.beat_llm_row()
        self.beat_world_model_row()
        self.beat_flip()
        self.wait(self.end_hold)

    # ------------------------------------------------------------------------------------------- pieces
    def build(self):
        box_cx = 1148.15 + 172.97 / 2
        self.llm_box = box(box_cx, 272.61 + 146.51 / 2)
        self.wm_box = box(box_cx, 648.38 + 146.51 / 2)
        self.llm_label = px_text("LLM", 36.58, box_cx, 361.73, bold=True, align="center")
        self.wm_label = m.VGroup(px_text("WORLD", 32, box_cx, 713.31, bold=True, align="center"),
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
                "prompt": px_text(prompt, 36, PROMPT_RIGHT, TEXT_BASELINE, font=TEXT_FONT, italic=True, align="right"),
                "completion": px_text(completion, 36, COMPLETION_LEFT, TEXT_BASELINE, font=TEXT_FONT, italic=True),
                "inputs": m.Group(*[photo_in_slot(p, s) for p, s in zip(inputs, IN_SLOTS)]),
                "output": photo_in_slot(output, OUT_SLOT),
            })

    # ------------------------------------------------------------------------------------------- beat 1
    def beat_llm_row(self):
        ex, t = self.examples[0], self.build_time
        self.play(m.Write(ex["prompt"]), run_time=t)
        self.play(m.Create(self.llm_in[0]), m.FadeIn(self.llm_in[1]),
                  m.Create(self.llm_box), m.Write(self.llm_label), run_time=t)
        self.play(m.Create(self.llm_out[0]), m.FadeIn(self.llm_out[1]), run_time=t / 2)
        self.play(m.Write(ex["completion"]), run_time=t)
        self.wait(self.row_hold)

    # ------------------------------------------------------------------------------------------- beat 2
    def beat_world_model_row(self):
        ex, t = self.examples[0], self.build_time
        self.play(m.LaggedStart(*[m.FadeIn(im) for im in ex["inputs"]], lag_ratio=0.4), run_time=1.5 * t)
        self.add(ex["inputs"])
        self.play(m.Create(self.wm_in[0]), m.FadeIn(self.wm_in[1]),
                  m.Create(self.wm_box), m.Write(self.wm_label), run_time=t)
        self.play(m.Create(self.wm_out[0]), m.FadeIn(self.wm_out[1]), run_time=t / 2)
        self.play(m.FadeIn(ex["output"]), run_time=t)
        self.wait(self.example_hold)

    # ------------------------------------------------------------------------------------------- beat 3
    def beat_flip(self):
        def parts(ex):
            return [ex["prompt"], ex["completion"], ex["inputs"], ex["output"]]

        for prev, ex in zip(self.examples, self.examples[1:]):
            if self.flip_time > 0:
                self.play(*[m.FadeOut(part) for part in parts(prev)], *[m.FadeIn(part) for part in parts(ex)], run_time=self.flip_time)
            else:
                self.remove(*parts(prev))
                self.add(*parts(ex))
            self.wait(self.example_hold)
