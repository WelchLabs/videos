"""
P63-64 - Atlas: prompt -> image + depth -> autoregressive new views -> edited prompt -> summary card  (ManimGL)

    manimgl <path to this file> P63_64

One continuous take, storyboards in p63_64_sketches/:

  beat 1  -01  prompt -> ATLAS -> (empty) fade in. The right side stays clear: the 360 video goes on in Premiere
  beat 2  -02  everything slides over so ATLAS is centered while the first image + depth map come out
  beat 3  -03  autoregression: image + depth slide over to the input side (prompt tucks up above them), a pose icon
               draws under them, the target pose draws on the output side, and the new view + depth appear
  beat 4  -04  again: inputs shrink into column 1, the generated view (and its pose) becomes column 2, new target
               pose, third view
  beat 5       everything fades except prompt -> ATLAS; they settle into the -05 layout and the extra clause
               ("with an absurd number of books") writes on in yellow
  beat 6  -05  out arrow + the new image, depth map and pose
  beat 7  -06  output slides to the input side (prompt goes up top), target pose, new view
  beat 8  -07  everything fades but the word ATLAS, which becomes the title; columns 1 + 2 write on while both
               dividers draw (leaving an obvious hole in the third spot), pause, then the training column writes on

All artwork is redrawn from the storyboards' pixel coordinates (1920x1080 viewBox mapped onto manim's 8-unit-tall
frame). Images come from Dropbox .../world_models/graphics/p60_64_assets (set WM_P60_64_ASSETS to point somewhere
else); a missing image is replaced by a grey placeholder so the scene still runs.

Timings are class attributes on P63_64.
"""
import os
from pathlib import Path

import numpy as np
from manimlib import *

# ----------------------------------------------------------------------------- config
ASSETS = Path(os.environ.get(
    "WM_P60_64_ASSETS",
    "/Users/stephen/Library/CloudStorage/Dropbox-Stephencwelch/welch_labs/world_models/graphics/p60_64_assets"))
BOOKS = ASSETS / "moar_books"
BOOKS_DEPTH = BOOKS / "depth_renders" / "magma_linear"

# (image, depth map) for each view, in order of appearance
VIEW_FILES = {
    "A": (ASSETS / "u9595213284_A_home_office_with_large_wooden_desk_and_extensiv_a5ac2ae3-d0fa-41ff-b743-eda479a3046c_2.png",
          ASSETS / "depth_magma_linear.png"),                                   # beat 2
    "B": (ASSETS / "download (1).jpeg", ASSETS / "02-depth.png"),               # beat 3
    "C": (ASSETS / "download (2).jpeg", ASSETS / "04-depth.png"),               # beat 4
    "D": (BOOKS / "01-u9595213284-a-grand-home-office-with-large-wooden-desk-and-ex-d0f7e423-1ec8-49d7-94cf-bd7944664d79-1.png",
          BOOKS_DEPTH / "01-depth.png"),                                        # beat 6 (more books)
    "E": (BOOKS / "02-view.png", BOOKS_DEPTH / "02-depth.png"),                 # beat 7
}

FRESH_TAN = "#f4ebd9"       # line art, prompt, ATLAS
CHILL_BROWN = "#948678"     # summary title, dividers, column labels
YELLOW_HL = "#fcc947"       # the added prompt clause
LABEL_FONT = "Open Sans"
TEXT_FONT = "STIX Two Text"
CAP = {LABEL_FONT: 0.714, TEXT_FONT: 0.656}     # cap height / em, to match Illustrator's point sizes

LINE_1 = "“A home office with wooden desk and"
LINE_2 = "floor-to-ceiling dark green bookshelves"
TAIL = ".”"                                       # closes line 2 until the extra clause arrives
LINE_3 = "with an absurd number of books.”"

# ----------------------------------------------------------------------------- storyboard coordinates
PX = FRAME_HEIGHT / 1080            # manim units per storyboard pixel
STROKE_PER_PX = 1 / 2.03            # manim stroke_width that renders 1 storyboard pixel wide (measured at 1080p)


def P(x, y):
    """Storyboard pixel (1920x1080, y down) -> manim point."""
    return np.array([(x - 960) * PX, (540 - y) * PX, 0.0])


def V(dx, dy):
    """Storyboard pixel offset -> manim vector."""
    return np.array([dx * PX, -dy * PX, 0.0])


def px_line(a, b, stroke_px, color=FRESH_TAN):
    return Line(P(*a), P(*b)).set_stroke(color, stroke_px * STROKE_PER_PX)


def arrow(tail, tip, stroke_px=2.0, head=(19.9, 8.13, 15.18), color=FRESH_TAN):
    """Storyboard-style arrow: shaft + notched head. head = (length, half width, notch depth) in pixels."""
    tail, tip = np.array(tail, float), np.array(tip, float)
    u = (tip - tail) / np.linalg.norm(tip - tail)
    n = np.array([-u[1], u[0]])
    length, half, notch = head
    shaft = px_line(tail, tip - notch * u, stroke_px, color)
    barb = Polygon(P(*tip), P(*(tip - length * u + half * n)), P(*(tip - notch * u)), P(*(tip - length * u - half * n)))
    barb.set_fill(color, 1).set_stroke(color, 0)
    return VGroup(shaft, barb)


def px_text(string, size_px, x, y, font=LABEL_FONT, bold=False, italic=False, color=FRESH_TAN):
    """Text at storyboard size with its left edge at x and its baseline at y (pixels).

    A throwaway "H" is set in front of the string: its height fixes the scale and its bottom is the baseline, so
    strings with descenders sit correctly without per-string fudge factors.
    """
    full = Text("H" + string, font=font, weight=BOLD if bold else NORMAL, slant=ITALIC if italic else NORMAL,
                font_size=48)
    full.scale(CAP.get(font, 0.7) * size_px * PX / full[0].get_height())
    t = full[1:]
    t.shift(P(x, y) - np.array([t.get_left()[0], full[0].get_bottom()[1], 0.0]))
    return t.set_color(color)


# ----------------------------------------------------------------------------- ATLAS box + arrows
# box centers per storyboard
C1 = (860.16, 538.78)       # -01
C2 = (960.00, 554.66)       # -02
C3 = (960.00, 553.39)       # -03
C4 = (1041.68, 537.50)      # -04
C5 = (970.61, 524.12)       # -05
C7 = (938.05, 522.55)       # -06


def atlas_box(c):
    box = RoundedRectangle(width=172.97 * PX, height=146.51 * PX, corner_radius=11.73 * PX)
    box.set_fill(opacity=0).set_stroke(FRESH_TAN, 2.5 * STROKE_PER_PX)
    return box.move_to(P(*c))


def atlas_label(c):
    return px_text("ATLAS", 36.58, c[0] - 56.24, c[1] + 14.42, bold=True)


def in_arrow(c):
    return arrow((c[0] - 168.16, c[1]), (c[0] - 113.53, c[1]))


def out_arrow(c):
    return arrow((c[0] + 121.40, c[1]), (c[0] + 176.03, c[1]))


# ----------------------------------------------------------------------------- prompt layouts
# (left edge of line 1, baseline of line 1, font size, leading, x offset of line 2, x offset of line 3)
L1 = (137.82, 533.77, 30.0, 36.0, -16.02, 0.0)       # -01
L2 = (237.66, 549.65, 30.0, 36.0, -16.02, 0.0)       # -02
L3 = (214.05, 142.25, 30.0, 36.0, -16.02, 0.0)       # -03
L4 = (96.97, 201.84, 22.41, 26.89, -11.97, 0.0)      # -04
L5 = (214.05, 497.87, 30.0, 36.0, -6.86, 20.44)      # -05 (three lines)
L7 = (155.25, 95.33, 30.0, 36.0, -1.48, 31.18)       # -06 (three lines, quote marks kept)


def prompt_at(layout, three_lines=False):
    """-> (line 1, line 2 without its closing '.”', the '.”' tail, line 3 or None), all in STIX italic."""
    x, y, size, lead, dx2, dx3 = layout
    kw = dict(font=TEXT_FONT, italic=True)
    l1 = px_text(LINE_1, size, x, y, **kw)
    full2 = px_text(LINE_2 + TAIL, size, x + dx2, y + lead, **kw)
    l2, tail = full2[:-2], full2[-2:]
    l3 = px_text(LINE_3, size, x + dx3, y + 2 * lead, color=YELLOW_HL, **kw) if three_lines else None
    return l1, l2, tail, l3


# ----------------------------------------------------------------------------- images
# top-left x, top-left y, height (pixels) for the image and for its depth map
RECTS = {
    "A_out": ((1228.64, 144.74, 410.83), (1228.64, 555.55, 409.03)),     # -02
    "A_in1": ((257.67, 234.10, 287.68), (257.67, 523.66, 288.00)),       # -03
    "A_in2": ((129.55, 284.25, 213.44), (129.55, 500.54, 216.00)),       # -04
    "B_out": ((1277.91, 160.26, 381.02), (1277.91, 541.28, 381.02)),     # -03
    "B_in": ((505.78, 285.99, 216.00), (505.78, 500.54, 216.00)),        # -04
    "C_out": ((1291.52, 156.16, 383.26), (1291.52, 539.44, 383.26)),     # -04
    "D_out": ((1231.10, 55.12, 431.14), (1231.10, 486.29, 431.14)),      # -05
    "D_in": ((156.53, 198.46, 360.00), (156.53, 559.65, 360.00)),        # -06
    "E_out": ((1246.52, 58.35, 440.21), (1246.52, 493.88, 440.21)),      # -06
}


def load_image(path):
    """ImageMobject that doesn't keep its file handle open; a grey 4:3 placeholder if the file is missing."""
    path = Path(path)
    if not path.exists():
        print(f"[P63_64] image not found: {path}")
        return Rectangle(4.0, 3.0).set_fill(CHILL_BROWN, 0.35).set_stroke(CHILL_BROWN, 1)
    im = ImageMobject(str(path))
    try:
        im.image.load()
    except Exception:
        pass
    return im


def place_px(im, x, y, h):
    """Top-left corner at pixel (x, y), height h pixels."""
    im.set_height(h * PX)
    im.move_to(P(x, y), aligned_edge=UL)
    return im


def make_view(key, rect_key):
    """Group(image, depth map) placed at RECTS[rect_key]."""
    g = Group(*[load_image(f) for f in VIEW_FILES[key]])
    for im, r in zip(g, RECTS[rect_key]):
        place_px(im, *r)
    return g


def view_to(view, rect_key):
    """Animations sliding + resizing a view to RECTS[rect_key]."""
    return [im.animate.set_height(r[2] * PX).move_to(P(r[0], r[1]), aligned_edge=UL)
            for im, r in zip(view, RECTS[rect_key])]


# ----------------------------------------------------------------------------- pose icons
# Each: frustum polygon, apex (camera center), dot radius, stroke, POSE label (size, x, y) - at its first storyboard spot
POSES = {
    # looking straight in (under the input in -06)
    "front": dict(poly=[(363.36, 950.91), (443.89, 950.91), (446.37, 995.19), (360.74, 995.19)],
                  apex=(406.73, 1030.51), dot=4.74, stroke=2.74, label=(16.89, 385.32, 1060.63)),
    # turned to the side (target pose under the output in -03)
    "side": dict(poly=[(1520.85, 954.34), (1576.24, 940.80), (1576.24, 981.32), (1520.85, 997.71)],
                 apex=(1593.18, 1014.46), dot=4.34, stroke=2.51, label=(20.90, 1532.69, 1052.43)),
    # third target (under the output in -04)
    "down": dict(poly=[(1527.88, 962.58), (1595.28, 962.47), (1598.90, 991.34), (1525.78, 991.34)],
                 apex=(1562.66, 929.45), dot=2.97, stroke=2.76, label=(16.99, 1538.47, 1017.78)),
}


def pose(kind, dx=0.0, dy=0.0):
    """Camera frustum + POSE label, shifted (dx, dy) pixels from its home spot. -> VGroup with .wire .dot .label"""
    spec = POSES[kind]
    w = spec["stroke"] * STROKE_PER_PX
    rays = VGroup(*[px_line(spec["apex"], c, spec["stroke"]) for c in spec["poly"]])
    face = Polygon(*[P(*c) for c in spec["poly"]]).set_fill(opacity=0).set_stroke(FRESH_TAN, w)
    dot = Dot(P(*spec["apex"]), radius=spec["dot"] * PX).set_fill(FRESH_TAN, 1).set_stroke(width=0)
    size, lx, ly = spec["label"]
    label = px_text("POSE", size, lx, ly, bold=True)
    g = VGroup(VGroup(rays, face), dot, label)
    g.shift(V(dx, dy))
    g.wire, g.dot, g.label = g[0], g[1], g[2]
    return g


# ----------------------------------------------------------------------------- the scene
class P63_64(InteractiveScene):
    # --- timings (seconds) ---------------------------------------------------------------------------------
    fade_time = 1.5           # beat 1 fade in
    intro_hold = 4.0          # prompt -> ATLAS -> [360 video in Premiere]
    slide_time = 1.5          # beat 1 -> 2 slide
    image_time = 1.2          # a generated image + depth appearing
    image_lag = 0.35          # depth map trails the image by this fraction
    move_time = 2.0           # outputs travelling over to the input side
    pose_time = 1.0           # a pose icon drawing in
    beat_hold = 2.0           # after each generation
    clear_time = 1.0          # beat 5 fade out
    prompt_only_hold = 0.75   # prompt -> ATLAS alone, before the extra clause
    clause_time = 1.5         # writing "with an absurd number of books"
    clause_hold = 1.5
    title_time = 2.0          # beat 8: ATLAS -> title
    write_time = 2.5          # columns 1 + 2 and the dividers
    hole_hold = 3.0           # the empty third spot
    train_time = 1.5          # column 3
    end_hold = 5.0

    def construct(self):
        self.beat_intro()
        self.beat_first_image()
        self.beat_new_view()
        self.beat_new_view_again()
        self.beat_new_prompt()
        self.beat_books_image()
        self.beat_books_view()
        self.beat_summary()
        self.wait(self.end_hold)

    # ------------------------------------------------------------------------------------------- helpers
    def core_to(self, c, with_out=True):
        """Animations sliding box + label + arrows so the box is centered on pixel c."""
        v = V(c[0] - self.c[0], c[1] - self.c[1])
        mobs = [self.box, self.label, self.arrow_in] + ([self.arrow_out] if with_out else [])
        self.c = c
        return [m.animate.shift(v) for m in mobs]

    def prompt_to(self, layout):
        """Animations moving the current prompt lines into a layout."""
        three = self.l3 is not None
        t1, t2, tail, t3 = prompt_at(layout, three_lines=three)
        anims = [Transform(self.l1, t1), Transform(self.l2, t2)]
        if three:
            anims.append(Transform(self.l3, t3))
        else:
            anims.append(Transform(self.tail, tail))
        return anims

    def show_view(self, view):
        self.play(LaggedStart(*[FadeIn(im, shift=0.25 * RIGHT) for im in view], lag_ratio=self.image_lag),
                  run_time=self.image_time)

    def draw_pose(self, p):
        self.play(ShowCreation(p.wire, lag_ratio=0.15), FadeIn(p.dot), FadeIn(p.label), run_time=self.pose_time)

    # ------------------------------------------------------------------------------------------- beat 1  (-01)
    def beat_intro(self):
        self.c = C1
        self.box, self.label = atlas_box(C1), atlas_label(C1)
        self.arrow_in, self.arrow_out = in_arrow(C1), out_arrow(C1)
        self.l1, self.l2, self.tail, self.l3 = prompt_at(L1)
        self.play(*[FadeIn(m) for m in (self.l1, self.l2, self.tail, self.arrow_in, self.box, self.label,
                                        self.arrow_out)], run_time=self.fade_time)
        self.wait(self.intro_hold)          # <- the 360 video sits to the right of the out arrow (Premiere)

    # ------------------------------------------------------------------------------------------- beat 2  (-02)
    def beat_first_image(self):
        # the slide and the first image + depth happen together
        self.A = make_view("A", "A_out")
        self.play(AnimationGroup(*self.core_to(C2), *self.prompt_to(L2), run_time=self.slide_time),
                  LaggedStart(*[FadeIn(im, shift=0.25 * RIGHT) for im in self.A], lag_ratio=self.image_lag,
                              run_time=self.image_time))
        self.wait(self.beat_hold)

    # ------------------------------------------------------------------------------------------- beat 3  (-03)
    def beat_new_view(self):
        self.play(*self.core_to(C3), *self.prompt_to(L3), *view_to(self.A, "A_in1"), run_time=self.move_time)
        self.pose_1 = pose("front", 56.30, -102.27)       # under the input image + depth
        self.draw_pose(self.pose_1)
        self.pose_2 = pose("side")                        # the view we're asking for
        self.draw_pose(self.pose_2)
        self.B = make_view("B", "B_out")
        self.show_view(self.B)
        self.wait(self.beat_hold)

    # ------------------------------------------------------------------------------------------- beat 4  (-04)
    def beat_new_view_again(self):
        self.play(*self.core_to(C4), *self.prompt_to(L4),
                  *view_to(self.A, "A_in2"), self.pose_1.animate.shift(V(-169.10, -89.18)),
                  *view_to(self.B, "B_in"), self.pose_2.animate.shift(V(-897.77, -195.64)),
                  run_time=self.move_time)
        self.pose_3 = pose("down")
        self.draw_pose(self.pose_3)
        self.C = make_view("C", "C_out")
        self.show_view(self.C)
        self.wait(self.beat_hold)

    # ------------------------------------------------------------------------------------------- beat 5
    def beat_new_prompt(self):
        gone = [*self.A, *self.B, *self.C, self.pose_1, self.pose_2, self.pose_3, self.arrow_out]
        self.play(*[FadeOut(m) for m in gone], run_time=self.clear_time)

        # prompt -> ATLAS settle into the -05 layout; the closing '.”' fades as line 2 opens up for the new clause
        t1, t2, tail, l3 = prompt_at(L5, three_lines=True)
        tail.set_opacity(0)
        self.play(*self.core_to(C5, with_out=False),
                  Transform(self.l1, t1), Transform(self.l2, t2), Transform(self.tail, tail),
                  run_time=self.slide_time)
        self.remove(self.tail)
        self.wait(self.prompt_only_hold)
        self.l3 = l3
        self.play(Write(self.l3), run_time=self.clause_time)
        self.wait(self.clause_hold)

    # ------------------------------------------------------------------------------------------- beat 6  (-05)
    def beat_books_image(self):
        self.arrow_out = out_arrow(C5)
        self.D = make_view("D", "D_out")
        self.pose_4 = pose("front", 1127.06, -14.10)      # under the new image + depth
        self.play(FadeIn(self.arrow_out), run_time=0.5)
        self.show_view(self.D)
        self.draw_pose(self.pose_4)
        self.wait(self.beat_hold)

    # ------------------------------------------------------------------------------------------- beat 7  (-06)
    def beat_books_view(self):
        self.play(*self.core_to(C7), *self.prompt_to(L7),
                  *view_to(self.D, "D_in"), self.pose_4.animate.shift(V(-1127.06, 14.10)),
                  run_time=self.move_time)
        self.pose_5 = pose("side", 3.47, 10.18)
        self.draw_pose(self.pose_5)
        self.E = make_view("E", "E_out")
        self.show_view(self.E)
        self.wait(self.beat_hold)

    # ------------------------------------------------------------------------------------------- beat 8  (-07)
    def beat_summary(self):
        title = px_text("ATLAS", 42, 894.73, 114.18, bold=True, color=CHILL_BROWN)
        gone = [self.l1, self.l2, self.l3, self.arrow_in, self.arrow_out, self.box,
                *self.D, *self.E, self.pose_4, self.pose_5]
        self.play(*[FadeOut(m) for m in gone], Transform(self.label, title), run_time=self.title_time)

        kw = dict(font=TEXT_FONT, italic=True)
        tag = dict(bold=True, color=CHILL_BROWN)
        target = VGroup(px_text("New View Prediction", 36.58, 140.18, 509.18, **kw),
                        px_text("PREDICTION TARGET", 24, 185.55, 674.08, **tag))
        arch = VGroup(px_text("Multimodal Autoregressive", 36.58, 740.59, 486.83, **kw),
                      px_text("Diffusion Transformer", 36.58, 775.14, 530.73, **kw),
                      px_text("ARCHITECTURE", 24, 859.68, 672.29, **tag))
        training = VGroup(px_text("From Scratch Pre-Train", 36.58, 1425.13, 497.76, **kw),
                          px_text("TRAINING", 24, 1563.88, 674.08, **tag))
        dividers = VGroup(px_line((590.75, 264.0), (590.75, 821.12), 1.0, CHILL_BROWN),
                          px_line((1286.06, 264.0), (1286.06, 821.12), 1.0, CHILL_BROWN))

        self.play(*[Write(m) for m in (*target, *arch)], *[ShowCreation(d) for d in dividers],
                  run_time=self.write_time)
        self.wait(self.hole_hold)           # <- obvious hole in the third spot
        self.play(*[Write(m) for m in training], run_time=self.train_time)
