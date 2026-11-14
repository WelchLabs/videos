"""
P24 - V-JEPA: 1x3 grid -> zoom to V-JEPA -> labels -> cats out -> action-conditioned layout -> robot clip.
One continuous take (ManimGL):

    manimgl <path to this file> P24VJEPA

  beat 1  storyboard -01: SORA | V-JEPA | GENIE grid, small flowchart with the two cat images in the middle column
  beat 2  zoom in on the flowchart (-02 framing); SORA / GENIE / dividers fade, the V-JEPA title settles lower
  beat 3  labels write on (ENCODER x2, PREDICTOR, Minimize Prediction Error, citation)
  beat 4  cat images fade out (-03)                       <- HOLD: Premiere takes over here (video clips + masks)
  beat 5  flowchart rearranges into the action-conditioned layout (-04), actuator axes + arrow draw in
                                                          <- HOLD: the actuator time series go on in Premiere
  beat 6  robot clip: 3-frame stack on the left, next frame on the right, one pass through the sequence

All artwork is redrawn from the p24_to_manim-0x.svg storyboards using their pixel coordinates (1920x1080 viewBox
mapped onto manim's 8-unit-tall frame). The three networks are one template (the -02 left encoder) placed with a
center + scale (+ a quarter turn for the predictor), so the same pieces can morph between layouts.
The camera never moves: the "zoom" scales the flowchart about the point that maps -01 onto -02 (strokes included).

Timings and toggles are class attributes on P24VJEPA.
"""
import os
import re
from pathlib import Path

import numpy as np
from manimlib import *

# ----------------------------------------------------------------------------- config
GRAPHICS = Path(os.environ.get(
    "WM_GRAPHICS", "/Users/stephen/Library/CloudStorage/Dropbox-Stephencwelch/welch_labs/world_models/graphics"))
ROBOT_DIR = GRAPHICS / "v_jepa_ac_1"                     # VJEPA_AC_CLIPS000.jpg ... 123.jpg
ROBOT_PATTERN = "VJEPA_AC_CLIPS{:03d}.jpg"
CAT_GLOB = "Screenshot 2026-11-11 at 9.59.17*AM.png"     # (macOS puts a narrow no-break space before "AM")

LINE_COLOR = "#948678"      # strokes, dates, citations
LABEL_COLOR = "#948979"     # ENCODER / PREDICTOR / Minimize ... (CHILL_BROWN)
TITLE_COLOR = "#f4ebd9"     # SORA / V-JEPA / GENIE
BG_FALLBACK = "#031e1e"     # node fill = scene background (read from the camera at runtime when possible)
FONT = "Open Sans"

# ----------------------------------------------------------------------------- storyboard coordinates
PX = FRAME_HEIGHT / 1080            # manim units per storyboard pixel
STROKE_PER_PX = 1 / 2.03            # manim stroke_width that renders 1 storyboard pixel wide (measured at 1080p)


def P(x, y):
    """Storyboard pixel (1920x1080, y down) -> manim point."""
    return np.array([(x - 960) * PX, (540 - y) * PX, 0.0])


def svg_path(d, stroke_px=2.0, color=LINE_COLOR):
    """Tiny SVG path reader (M L H V C, absolute or relative) -> VMobject in manim coordinates."""
    tokens = re.findall(r"[MmLlHhVvCc]|-?(?:\d+\.?\d*|\.\d+)", d)
    mob = VMobject()
    pos, cmd, i = np.zeros(2), None, 0

    def nums(n):
        nonlocal i
        vals = [float(t) for t in tokens[i:i + n]]
        i += n
        return vals

    while i < len(tokens):
        if re.match(r"[A-Za-z]", tokens[i]):
            cmd = tokens[i]; i += 1
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            x, y = nums(2); pos = pos + [x, y] if rel else np.array([x, y])
            mob.start_new_path(P(*pos)); cmd = "l" if rel else "L"
        elif c == "L":
            x, y = nums(2); pos = pos + [x, y] if rel else np.array([x, y]); mob.add_line_to(P(*pos))
        elif c == "H":
            x, = nums(1); pos = np.array([pos[0] + x if rel else x, pos[1]]); mob.add_line_to(P(*pos))
        elif c == "V":
            y, = nums(1); pos = np.array([pos[0], pos[1] + y if rel else y]); mob.add_line_to(P(*pos))
        elif c == "C":
            v = np.array(nums(6)).reshape(3, 2)
            pts = v + pos if rel else v
            mob.add_cubic_bezier_curve_to(P(*pts[0]), P(*pts[1]), P(*pts[2])); pos = pts[2]
    mob.set_stroke(color, stroke_px * STROKE_PER_PX).set_fill(opacity=0)
    return mob


def px_line(a, b, stroke_px, color=LINE_COLOR, opacity=1.0):
    return Line(P(*a), P(*b)).set_stroke(color, stroke_px * STROKE_PER_PX, opacity)


def arrow(tail, tip, stroke_px, head=(16.69, 6.82, 12.73), color=LINE_COLOR):
    """Storyboard-style arrow: shaft + notched head. head = (length, half width, notch depth) in pixels."""
    tail, tip = np.array(tail, float), np.array(tip, float)
    u = (tip - tail) / np.linalg.norm(tip - tail)
    n = np.array([-u[1], u[0]])
    length, half, notch = head
    shaft = px_line(tail, tip - notch * u, stroke_px, color)
    barb = Polygon(P(*tip), P(*(tip - length * u + half * n)), P(*(tip - notch * u)), P(*(tip - length * u - half * n)))
    barb.set_fill(color, 1).set_stroke(color, 0)
    return VGroup(shaft, barb)


def elbow_arrow(d, tip, stroke_px, head, color=LINE_COLOR):
    """Rounded-corner path that ends in a rightward arrowhead at `tip`."""
    length, half, notch = head
    x, y = tip
    barb = Polygon(P(x, y), P(x - length, y + half), P(x - notch, y), P(x - length, y - half))
    barb.set_fill(color, 1).set_stroke(color, 0)
    return VGroup(svg_path(d, stroke_px, color), barb)


# --- text -------------------------------------------------------------------------------------------------
_CAP_REF = {}


def svg_text(string, size_px, bold=True, color=LINE_COLOR):
    """Text set so its cap height matches Open Sans at `size_px` in the storyboard (cap height = 0.714 em)."""
    weight = BOLD if bold else NORMAL
    if weight not in _CAP_REF:
        _CAP_REF[weight] = Text("H", font=FONT, weight=weight, font_size=48).get_height()
    t = Text(string, font=FONT, weight=weight, font_size=48, alignment="LEFT")
    t.scale(0.714 * size_px * PX / _CAP_REF[weight])
    return t.set_color(color)


def at_baseline(mob, x, y, descent_px=0.0):
    """Put a text's left edge at x and its baseline at y (pixels). descent_px: how far its glyphs dip below it."""
    return mob.move_to(P(x, y + descent_px), aligned_edge=DL)


def text_lines(lines, size_px, x, y, leading_px, bold=True, color=LINE_COLOR, top_aligned=False):
    """Left-aligned lines with exact storyboard leading; (x, y) is the first baseline."""
    g = VGroup()
    for k, s in enumerate(lines):
        t = svg_text(s, size_px, bold, color)
        if top_aligned:   # mixed-case small print: hang each line from its ascender line instead (0.76 em above)
            t.move_to(P(x, y + k * leading_px - 0.76 * size_px), aligned_edge=UL)
        else:
            at_baseline(t, x, y + k * leading_px)
        g.add(t)
    return g


# --- images -----------------------------------------------------------------------------------------------
def load_image(path):
    """ImageMobject that doesn't keep its file handle open (matters with a hundred-odd frames)."""
    im = ImageMobject(str(path))
    try:
        im.image.load()
    except Exception:
        pass
    return im


def place_px(im, x, y, h):
    """Top-left corner at pixel (x, y), height h pixels. Idempotent, so frames can be re-placed freely."""
    im.set_height(h * PX)
    im.move_to(P(x, y), aligned_edge=UL)
    return im


# ----------------------------------------------------------------------------- the network template
# -02 left encoder: 18 connections (x1, y1, x2, y2, opacity) and 9 nodes, around its middle node.
NET_C0 = np.array([698.59, 442.93])
NET_R, NET_LINE_PX, NET_NODE_PX = 20.39, 8.0, 6.0 # 4.99, 4.31
NET_LINES = [
    (709.06, 392.18, 659.55, 427.28, .60), (669.57, 492.85, 654.36, 460.88, .60), (710.93, 497.10, 661.31, 456.62, .60),
    (638.84, 500.55, 731.27, 451.89, .60), (680.69, 492.02, 692.52, 461.42, .60), (719.83, 492.41, 705.55, 462.10, .60),
    (770.28, 490.56, 757.63, 462.20, .82), (731.74, 493.17, 744.31, 462.21, .60), (735.16, 456.98, 687.82, 497.18, .60),
    (761.52, 496.12, 713.00, 456.60, .75), (731.62, 398.41, 744.06, 424.75, .80), (717.08, 397.22, 705.07, 423.60, .60),
    (628.43, 491.52, 640.42, 462.23, .85), (683.87, 455.76, 635.07, 494.42, .60), (679.61, 397.22, 691.32, 422.54, .60),
    (689.03, 392.38, 736.35, 428.70, .82), (664.68, 397.22, 652.04, 423.92, .60), (759.42, 501.55, 663.08, 451.71, .60),
]
NET_NODES = [(674.34, 511.40), (725.46, 511.40), (750.95, 442.93), (725.46, 376.83), (621.58, 511.40),
             (672.70, 376.83), (644.67, 442.93), (698.59, 442.93), (777.52, 510.95)]


def network(center, scale=1.0, quarter_turn=False, bg=BG_FALLBACK):
    """The 2-3-4 network glyph. `center` = pixel position of its middle node. quarter_turn: the predictor's orientation."""
    center = np.array(center, float)

    def place(x, y):
        d = (np.array([x, y]) - NET_C0) * scale
        if quarter_turn:
            d = np.array([-d[1], d[0]])
        return center + d

    lines = VGroup(*[px_line(place(x1, y1), place(x2, y2), NET_LINE_PX * scale, opacity=op)
                     for x1, y1, x2, y2, op in NET_LINES])
    nodes = VGroup()
    for x, y in NET_NODES:
        c = Circle(radius=NET_R * scale * PX).move_to(P(*place(x, y)))
        c.set_stroke(LINE_COLOR, NET_NODE_PX * scale * STROKE_PER_PX).set_fill(bg, 1)
        nodes.add(c)
    return VGroup(lines, nodes)


def minus_node(center, bg=BG_FALLBACK):
    cx, cy = center
    c = Circle(radius=26.9 * PX).move_to(P(cx, cy)).set_stroke(LINE_COLOR, 5.69 * STROKE_PER_PX).set_fill(bg, 1)
    bar = px_line((cx - 13.05, cy + 0.88), (cx + 13.05, cy + 0.88), 5.93)
    return VGroup(c, bar)


# ----------------------------------------------------------------------------- layouts
HEAD_M = (17.93, 7.33, 13.67)     # -02 arrowheads
HEAD_S = (16.58, 6.77, 12.64)     # -04 small arrowheads
HEAD_L = (19.01, 7.77, 14.50)     # -04 predictor in / out
HEAD_AX = (13.76, 5.62, 10.49)    # -04 actuator axes


def flow_closeup(bg):
    """Flowchart line art at the -02 / -03 framing. Same keys + structure as flow_action(), so they can morph."""
    return {
        "enc_l": network((698.59, 442.93), 1.0, bg=bg),
        "enc_r": network((1179.53, 443.87), 1.0, bg=bg),
        "pred": network((970.64, 211.68), 1.0, quarter_turn=True, bg=bg),
        "in_l": arrow((698.58, 591.92), (698.58, 551.69), 2.58, (16.69, 6.82, 12.73)),
        "in_r": arrow((1183.59, 591.92), (1183.59, 548.70), 2.77, HEAD_M),
        "to_pred": elbow_arrow("M697.03,321.72v-98.67c0-7.34,5.95-13.29,13.29-13.29h122.85", (849.87, 209.75), 2.36,
                               (23.52, 9.61, 17.94)),
        "pred_out": arrow((1078.67, 214.84), (1121.89, 214.84), 2.77, HEAD_M),
        "enc_out": arrow((1179.04, 331.76), (1179.04, 288.54), 2.77, HEAD_M),
        "minus": minus_node((1180.42, 214.34), bg),
    }


def flow_action(bg):
    """Flowchart line art in the action-conditioned layout (-04)."""
    s_enc, s_pred = 16.65 / NET_R, 17.19 / NET_R
    return {
        "enc_l": network((634.74, 414.01), s_enc, bg=bg),
        "enc_r": network((1240.60, 417.42), s_enc, bg=bg),
        "pred": network((936.34, 217.73), s_pred, quarter_turn=True, bg=bg),
        "in_l": arrow((634.84, 541.26), (634.84, 501.31), 2.56, HEAD_S),
        "in_r": arrow((1246.53, 545.79), (1246.53, 505.83), 2.56, HEAD_S),
        "to_pred": elbow_arrow("M631.8,325.29v-91.21c0-6.79,5.5-12.29,12.29-12.29h159.29", (816.86, 221.80), 2.94, HEAD_L),
        "pred_out": arrow((1038.34, 217.62), (1108.50, 217.62), 2.94, HEAD_L),
        "enc_out": arrow((1239.76, 331.98), (1239.76, 292.03), 2.56, HEAD_S),
        "minus": minus_node((1241.45, 211.18), bg),
    }


def labels_closeup():
    return {
        "enc_l": at_baseline(svg_text("ENCODER", 27, color=LABEL_COLOR), 463.19, 450.30),
        "enc_r": at_baseline(svg_text("ENCODER", 27, color=LABEL_COLOR), 1291.69, 445.43),
        "pred": at_baseline(svg_text("PREDICTOR", 27, color=LABEL_COLOR), 905.44, 84.64),
        "minimize": text_lines(["Minimize", "Prediction", "Error"], 28, 1234.23, 195.42, 29, color=LABEL_COLOR),
    }


def labels_action():
    return {
        "enc_l": at_baseline(svg_text("ENCODER", 32.93, color=LABEL_COLOR), 398.20, 429.42),
        "enc_r": at_baseline(svg_text("ENCODER", 32.93, color=LABEL_COLOR), 1321.89, 429.43),
        "pred": at_baseline(svg_text("PREDICTOR", 32.93, color=LABEL_COLOR), 843.44, 97.09),
        "minimize": text_lines(["Minimize", "Prediction", "Error"], 28, 1295.26, 192.26, 29, color=LABEL_COLOR),
    }


def zoomed(mob, factor, about):
    """Copy of a VMobject scaled about a point, stroke widths scaled with it (what a camera zoom would show)."""
    m = mob.copy()
    parts = m.family_members_with_points()
    widths = [float(np.max(p.get_stroke_width())) for p in parts]
    m.scale(factor, about_point=about)
    for p, w in zip(parts, widths):
        p.set_stroke(width=w * factor)
    return m


# -01 -> -02 is a pure zoom: middle node of the left encoder goes (819.28, 522.41) r 13.16 -> (698.59, 442.93) r 20.39
ZOOM = 20.39 / 13.16
ZOOM_FIXED = (np.array([698.59, 442.93]) - ZOOM * np.array([819.28, 522.41])) / (1 - ZOOM)   # pixel that doesn't move


# ----------------------------------------------------------------------------- the scene
class P24VJEPA(InteractiveScene):
    # --- toggles / timings (seconds) -----------------------------------------------------------------------
    show_text = True          # titles, labels, citations as manim Text (Open Sans). False: add them in post
    grid_hold = 2.0           # on the 1x3 grid before the zoom
    zoom_time = 3.0
    label_time = 2.0
    cat_fade_time = 1.0
    premiere_hold = 4.0       # empty labelled flowchart (-03): the Premiere section cuts in here
    rearrange_time = 3.0
    axes_time = 2.0
    axes_hold = 3.0           # actuator axes on screen before the robot frames arrive (time series go here in post)
    stack_dt = 0.2            # between the robot frames popping in
    num_frames = 124
    frame_dt = 0.1            # per robot frame
    end_hold = 5.0

    def construct(self):
        try:
            self.bg = rgb_to_hex(self.camera.background_rgba[:3])
        except Exception:
            self.bg = BG_FALLBACK
        self.build()
        self.beat_grid()
        self.beat_zoom()
        self.beat_labels()
        self.beat_cats_out()
        self.beat_rearrange()
        self.beat_robot()
        self.wait(self.end_hold)

    # ------------------------------------------------------------------------------------------- pieces
    def build(self):
        fixed = P(*ZOOM_FIXED)
        self.big = flow_closeup(self.bg)                                    # -02 targets
        self.flow = {k: zoomed(m, 1 / ZOOM, fixed) for k, m in self.big.items()}   # start at the -01 size
        self.act = flow_action(self.bg)                                     # -04 targets

        # cat images + the mask on the right one (placed at -02, then shrunk to -01)
        cat_files = sorted(GRAPHICS.glob(CAT_GLOB))
        if cat_files:
            cat_l, cat_r = load_image(cat_files[0]), load_image(cat_files[0])
        else:   # keep the scene runnable without the screenshot
            print(f"[P24] cat image not found: {GRAPHICS / CAT_GLOB}")
            cat_l, cat_r = [Rectangle(3526, 1972).set_fill(LABEL_COLOR, 0.35).set_stroke(LINE_COLOR, 1) for _ in range(2)]
        CAT_H = 225.0     # pixels at the -02 framing, measured off the exported PNG (the placed link is ~4% over its .11 scale)
        place_px(cat_l, 501.63, 622.0, CAT_H)
        place_px(cat_r, 966.84, 622.0, CAT_H)
        mask = Rectangle(228.88 * PX, 190.12 * PX).set_fill(self.bg, 1).set_stroke(self.bg, 0)
        mask.move_to(P(986.64, 639.18), aligned_edge=UL)
        self.cats = Group(cat_l, cat_r, mask)
        self.cats.scale(1 / ZOOM, about_point=fixed)

        # grid furniture (-01)
        self.dividers = VGroup(px_line((629.78, 113.05), (629.78, 946.71), 1.0),
                               px_line((1310.32, 113.05), (1310.32, 946.71), 1.0))
        self.t = {}
        if self.show_text:
            t = self.t
            t["sora"] = at_baseline(svg_text("SORA", 38, color=TITLE_COLOR), 257.87, 948.56)
            t["genie"] = at_baseline(svg_text("GENIE", 38, color=TITLE_COLOR), 1541.90, 948.46)
            t["sora_date"] = at_baseline(svg_text("February 15, 2024", 22, bold=False), 216.86, 986.78, 5.3)
            t["genie_date"] = at_baseline(svg_text("February 23, 2024", 22, bold=False), 1512.46, 981.62, 5.3)
            # "J" hangs ~7 px below the baseline in Open Sans Bold
            t["title"] = VGroup(at_baseline(svg_text("V-JEPA", 38, color=TITLE_COLOR), 899.37, 948.46, 7.0),
                                at_baseline(svg_text("February 15, 2024", 22, bold=False), 861.92, 987.56, 5.3))
            self.labels = labels_closeup()
            self.labels_act = labels_action()
            t["cite_1"] = text_lines(['Bardes, Adrien, et al. "Revisiting feature prediction for learning visual',
                                      'representations from video." arXiv preprint arXiv:2404.08471 (2024).'],
                                     16, 1370.14, 1032.96, 19.2, bold=False, top_aligned=True)
            t["cite_2"] = text_lines(['Robotics examples from: Assran, Mido, et al. "V-jepa 2: Self-supervised video models',
                                      'enable understanding, prediction and planning." arXiv preprint arXiv:2506.09985 (2025).'],
                                     16, 1239.08, 1032.58, 19.2, bold=False, top_aligned=True)
            t["video"] = at_baseline(svg_text("VIDEO", 32.93, color=LABEL_COLOR), 615.67, 877.96)
            t["next_frame"] = at_baseline(svg_text("NEXT FRAME", 32.56, color=LABEL_COLOR), 1145.54, 877.34)

        # action-conditioning extras (-04): arrow into the predictor + the axes the time series will sit on
        self.act_arrow = arrow((939.24, 349.26), (939.24, 309.30), 2.56, HEAD_S)
        self.axis_v = arrow((817.57, 478.67), (817.57, 313.77), 2.13, HEAD_AX)
        self.axis_h = arrow((817.66, 408.80), (1100.14, 408.80), 2.13, HEAD_AX)
        self.act_label = None
        if self.show_text:
            a = Tex(r"\textbf{\textit{Actuators}}").set_color(LABEL_COLOR)
            a.set_width(152 * PX).rotate(PI / 2)
            a.move_to(P(801.5, 478.8), aligned_edge=DR)     # reads bottom-to-top, just left of the vertical axis
            self.act_label = a

    def flow_group(self):
        return VGroup(*self.flow.values())

    # ------------------------------------------------------------------------------------------- beat 1
    def beat_grid(self):
        self.add(self.dividers, self.cats, self.flow_group())
        for k in ("sora", "genie", "sora_date", "genie_date", "title"):
            if k in self.t:
                self.add(self.t[k])
        self.wait(self.grid_hold)

    # ------------------------------------------------------------------------------------------- beat 2
    def beat_zoom(self):
        fixed = P(*ZOOM_FIXED)
        anims = [Transform(self.flow[k], self.big[k]) for k in self.flow]
        anims.append(self.cats.animate.scale(ZOOM, about_point=fixed))
        anims.append(FadeOut(self.dividers))
        anims += [FadeOut(self.t[k]) for k in ("sora", "genie", "sora_date", "genie_date") if k in self.t]
        if "title" in self.t:   # -01 (899.37, 948.46) -> -02 (886.74, 975.77)
            anims.append(self.t["title"].animate.shift((886.74 - 899.37) * PX * RIGHT + (975.77 - 948.46) * PX * DOWN))
        self.play(*anims, run_time=self.zoom_time)
        self.wait()

    # ------------------------------------------------------------------------------------------- beat 3
    def beat_labels(self):
        if not self.show_text:
            return
        self.play(*[Write(m) for m in self.labels.values()], FadeIn(self.t["cite_1"]), run_time=self.label_time)
        self.wait()

    # ------------------------------------------------------------------------------------------- beat 4
    def beat_cats_out(self):
        self.play(FadeOut(self.cats), run_time=self.cat_fade_time)
        self.wait(self.premiere_hold)       # <- storyboard -03: clean plate for the Premiere section

    # ------------------------------------------------------------------------------------------- beat 5
    def beat_rearrange(self):
        anims = [Transform(self.flow[k], self.act[k]) for k in self.flow]
        if self.show_text:
            anims += [Transform(self.labels[k], self.labels_act[k]) for k in self.labels]
            anims += [FadeOut(self.t["cite_1"]), FadeIn(self.t["cite_2"])]
        self.play(*anims, run_time=self.rearrange_time)

        draw = [ShowCreation(self.axis_v[0]), ShowCreation(self.axis_h[0]), ShowCreation(self.act_arrow[0]),
                FadeIn(self.axis_v[1]), FadeIn(self.axis_h[1]), FadeIn(self.act_arrow[1])]
        if self.act_label is not None:
            draw.append(Write(self.act_label))
        self.play(*draw, run_time=self.axes_time)
        self.wait(self.axes_hold)           # <- actuator time series go on top of these axes in Premiere

    # ------------------------------------------------------------------------------------------- beat 6
    def beat_robot(self):
        LEFT_H, RIGHT_H = 2048 * 0.12, 2048 * 0.1283            # storyboard frame sizes (square), pixels
        GHOSTS = [(515.60, 551.80), (533.07, 572.35)]            # two faded frames at the back of the stack
        FRONT, RIGHT_XY = (548.02, 592.31), (1117.77, 568.04)

        n = self.num_frames
        paths = [ROBOT_DIR / ROBOT_PATTERN.format(i) for i in range(n)]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"[P24] {len(missing)} robot frames missing (first: {missing[0]}), playing what exists")
            paths = [p for p in paths if p.exists()]
            n = len(paths)
        if n < 4:
            return
        imgs = [load_image(p) for p in paths]

        # frames 0, 1 stay as the faded history; 2 is the newest "video" frame; 3 is the "next frame"
        place_px(imgs[0], *GHOSTS[0], LEFT_H).set_opacity(0.5)
        place_px(imgs[1], *GHOSTS[1], LEFT_H).set_opacity(0.5)
        place_px(imgs[2], *FRONT, LEFT_H)
        place_px(imgs[3], *RIGHT_XY, RIGHT_H)

        for k in range(3):
            self.add(imgs[k])
            self.wait(self.stack_dt)
        if "video" in self.t:
            self.play(FadeIn(imgs[3]), Write(self.t["video"]), Write(self.t["next_frame"]), run_time=1.0)
        else:
            self.play(FadeIn(imgs[3]), run_time=1.0)
        self.wait()

        # each frame plays on the right, then one step later becomes the front of the stack on the left
        for i in range(4, n):
            self.remove(imgs[i - 2])
            place_px(imgs[i - 1], *FRONT, LEFT_H)
            place_px(imgs[i], *RIGHT_XY, RIGHT_H)
            self.add(imgs[i])               # imgs[i - 1] is already on screen (it was the right-hand frame)
            self.wait(self.frame_dt)
