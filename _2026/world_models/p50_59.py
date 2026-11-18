"""
P50-59 - LLM training / teacher forcing -> ... (ManimGL)

    manimgl <path to this file> P50_59

One scene for the whole sequence, built a paragraph at a time. So far:

  P50   storyboards p43_59-11, -12
    beat_intro          title, the sentence, its arrow and the transformer graphic all draw in together
    beat_isolate        everything about the transformer fades except one attention-pattern icon (top of the first
                        column); the arrow and "Paris" fade with it
    beat_zoom           the icon grows into the 5 x 5 attention pattern in the middle of the screen; as it grows, the
                        five remaining words fly to the left of their rows, and a copy of each swings round to sit
                        under its column
  P51   storyboard p43_59-12
    beat_shift_pattern  the pattern and its labels slide left to where storyboard -12 has them
    beat_targets        INPUT/CONTEXT and TARGET (NEXT TOKEN) headers, the five arrows and the five targets come in
                        (plus the small <bos> footnote)
    beat_highlight_row  yellow boxes land on one row's cells one at a time, left to right (only the cells the row
                        can see: 5 on the bottom row, 4 on the one above), then a box round that row's target
    beat_clear_highlight  those boxes fade
  ... storyboard p43_59-13
    beat_teacher_forcing  the pattern fades and the layout turns into the teacher forcing table: each row label
                        becomes the last word of its row's context, the earlier words of every row swing up out of
                        the column labels, arrows / targets / headers slide to their new places and the title
                        crossfades to TEACHER FORCING
  P52   storyboard p43_59-14: an LLM generating one word at a time
    beat_clear_all      the teacher forcing table fades; the title crossfades to AUTOREGRESSIVE INFERENCE
    beat_llm_intro      the LLM box, its two arrows and the first input word ("The") come in, mid-screen
    beat_llm_step       a pulse runs in -> box -> out and the next word appears on the output side
    beat_llm_feed       that word drops under the box and comes up at the end of the input, the words already
                        there shuffling left to make room   (step / feed repeat until "... is" -> "Paris")
  ... storyboard p43_59-15: the same loop with video frames
    beat_llm_up         the LLM row moves up (and right) to where storyboard -15 has it, and stays as it is
    beat_wm_intro       WORLD MODEL box + arrows, one context frame next to the input arrow, CONTEXT bracket
    beat_wm_step        pulse through the model, the next frame appears on the right (PREDICTION label the first time)
    beat_wm_feed        the prediction shrinks, passes under the box and lands as the newest context frame; the
                        others shift left and the bracket stretches. Stops with three context frames + the fourth
                        as the prediction, i.e. storyboard -15.

The icon in storyboard -11 already has the same shading as the big pattern in -12, so the zoom is a straight
enlargement: each little cell / grid line becomes its big counterpart (snapped onto a clean grid, with the
storyboard's line weight). In P50 the pattern is centered on screen (PATTERN_CENTER); storyboard -12 has it
14.5 units further left, which is where it goes in P51 to make room for the targets.

Artwork comes from the Illustrator storyboards in p50_59_storyboard/ (192 x 108 viewBox = one unit per 10 px at
1080p, mapped onto manim's 8-unit-tall frame). The transformer graphic is read straight out of p43_59-11.svg at
runtime; everything else is redrawn from the storyboard's coordinates.

  class P53   (its own scene)   a flashback to storyboard p43_59-02 (in p43_59_sketches_1/), then training
    beat_intro          title BIDIRECTIONAL DIFFUSION, prompt + transformer on the left, 12 tiles of noise (3 x 4) on
                        the right, all fading in together
    beat_denoise        the diffusion steps run on all 12 frames at once, noise -> the red ball video (the real
                        Wan2.2 steps, as in P44's beat_grid_denoise), with the step counter under the transformer
    beat_to_real        the tiles crossfade to 12 frames of real tennis ball video; the prompt and its arrow fade;
                        the title becomes BIDIRECTIONAL DIFFUSION - TRAINING
    beat_noise_up       noise is added to all 12 real frames together until they are pure noise
    beat_denoise        ... and the steps run forward again, back to the real frames
  ... storyboard p43_59-16: teacher forcing on the real video
    beat_teacher_layout the grid breaks up into three rows: row 1 keeps real frame 1 as its input, row 2 frames 1-2,
                        row 3 frames 1-3 (each frame leaves its grid tile, splitting into as many copies as it
                        needs); the other tiles, the transformer and the counter fade; the title becomes TEACHER
                        FORCING - TRAINING
    beat_teacher_models a VIDEO MODEL box with its arrows draws in on every row, the three outputs appear as pure
                        noise, plus the REAL VIDEO bracket and PREDICTION TARGET label
    beat_denoise        the three outputs denoise together into the real next frame of each row
  P54 (same scene)   storyboard p43_59-17: the trained model running on its own outputs
    beat_rollout_layout the bottom row's VIDEO MODEL (box + arrows) slides to the middle of the screen; everything
                        else from the teacher forcing rows fades; the title becomes TEACHER FORCING + AUTOREGRESSIVE
                        INFERENCE
    beat_rollout_start  the first, clean frame of the red ball clip appears as the model's input
    beat_rollout_step   a pulse runs through the model and the next frame appears on the output side
    beat_rollout_feed   that output shrinks, passes under the box and becomes the newest input, the others shifting
                        left   (step / feed repeat; ends with three inputs + the fourth frame as the output)
  P55 (same scene)   storyboard p43_59-18: training and inference side by side
    beat_compare_layout the rollout row slides down to the lower half; the teacher forcing training row from before
                        (three real tennis frames -> VIDEO MODEL -> the real next frame) fades in above it, on the
                        same columns
    beat_compare_labels TEACHER FORCING (TRAINING) / AUTOREGRESSIVE INFERENCE headings, REAL VIDEO and MODEL OUTPUTS
                        brackets, PREDICTION TARGET under the training output and GENERATED NEXT FRAME under the
                        inference output
    beat_exposure_title after a hold, the title becomes EXPOSURE BIAS
  ... storyboard p43_59-19: the same two rows for an LLM, then back
    beat_compare_swap("llm")    all at once: the images fade, sentences write in where they were ("The capital of
                        France is" -> "Paris" on top, "The capital of Germany is" -> "Berlin" below), VIDEO MODEL
                        becomes LLM, and the labels switch to REAL TEXT / GENERATED TEXT / GENERATED NEXT TOKEN
    beat_compare_swap("video")  ... and all back to the video version
  The rollout frames are synthetic: _p54_drift/mild/drift_K_frame_FFF.png, made by p54_drift.py from the clean
  frames of the Wan2.2 clip (K = 0 is clean, 1-3 drift further and further). DRIFT_DIR picks the folder.

  The diffusion on the real frames is faked: each tile is a pixel blend of the real frame and one of the Wan2.2
  run's pure-noise frames, with the blend weight per step (NOISE_CURVE) measured from the real run so it clears at
  the same pace. Every tile has its own noise level (RealTile.set_level), all driven together here.

  class P56_58   (its own scene)   storyboard p43_59-21: bidirectional diffusion and teacher forcing side by side
    beat_intro          the dividing line draws down from the top while the two titles write in
    beat_tiles_in       left: four frames of the real video as pure noise. right: the first real frame, an arrow,
                        and the next frame as pure noise
    beat_run            right: the output frame denoises, then slides over to join the inputs (which shuffle left),
                        a new noise frame takes its place and denoises, and so on: 1 input -> 2 -> 3, ending on
                        storyboard -21 (three clean inputs -> the fourth frame). left: the four frames denoise
                        together in ONE pass spread over that whole time, so both sides finish on the same frame
                        (left_sync = False: the left pass runs at the right side's pace and finishes early).
  Same faked diffusion on the real frames as P53 (RealTile + NOISE_CURVE).

The ball frames in P52 are stills from the finished Wan2.2 clip, read from the same notebook output folder
p43_59.py uses (<RUN>/p43/xt/step_NN_final/frame_FFF.png; set WM_P43_59 to point somewhere else).

Self-contained: the storyboard helpers (P, u_text, arrow, the SVG reader) are copies of the ones in p43_59.py.

Timings and toggles are class attributes on the scene class; the beat methods also take run_time arguments.
"""
import os
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from manimlib import *

# ----------------------------------------------------------------------------- config
HERE = Path(__file__).resolve().parent
SKETCHES = HERE / "p50_59_storyboard"
SKETCHES_43 = HERE / "p43_59_sketches_1"        # P53's flashback uses the transformer from storyboard p43_59-01
RUN = Path(os.environ.get(
    "WM_P43_59",
    "/Users/stephen/Library/CloudStorage/Dropbox-Stephencwelch/welch_labs/world_models/hackin/Wan2.2/"
    "attn_runs/ball_drop_s252/p43_59_3"))
TENNIS = Path(os.environ.get(                   # the 12 frames of real video for P53 (sorted by file name)
    "WM_TENNIS",
    "/Users/stephen/Library/CloudStorage/Dropbox-Stephencwelch/welch_labs/world_models/hackin/tennis_ball_2"))

FRESH_TAN = "#f4ebd9"       # line art, text, attention cells
CHILL_BROWN = "#948678"     # title, headers
YELLOW = "#fcc947"          # cell / target highlights
BG_FALLBACK = "#031e1e"     # node fill = scene background (read from the camera at runtime when possible)
SVG_BG = "#031e1e"          # the background color used inside the storyboards (swapped for the scene's)
LABEL_FONT = "Open Sans"
TEXT_FONT = "STIX Two Text"
CAP = {LABEL_FONT: 0.714, TEXT_FONT: 0.656}     # cap height / em, to match Illustrator's point sizes

WORDS = ["The", "capital", "of", "France", "is", "Paris"]      # the last one is the final target, never a row

# ----------------------------------------------------------------------------- storyboard coordinates
U = FRAME_HEIGHT / 108              # manim units per storyboard unit (192 x 108 viewBox)
STROKE_PER_U = 10 / 2.03            # manim stroke_width that renders 1 storyboard unit wide (measured at 1080p)


def P(x, y):
    """Storyboard point (192 x 108, y down) -> manim point."""
    return np.array([(x - 96) * U, (54 - y) * U, 0.0])


def u_line(a, b, stroke_u=0.2, color=FRESH_TAN, opacity=1.0):
    return Line(P(*a), P(*b)).set_stroke(color, stroke_u * STROKE_PER_U, opacity)


def arrow(tail, tip, stroke_u=0.2, head=(1.99, 0.815, 1.52), color=FRESH_TAN):
    """Storyboard-style arrow: shaft + notched head. head = (length, half width, notch depth). -> VGroup(shaft, barb)"""
    tail, tip = np.array(tail, float), np.array(tip, float)
    u = (tip - tail) / np.linalg.norm(tip - tail)
    n = np.array([-u[1], u[0]])
    length, half, notch = head
    shaft = u_line(tail, tip - notch * u, stroke_u, color)
    barb = Polygon(P(*tip), P(*(tip - length * u + half * n)), P(*(tip - notch * u)), P(*(tip - length * u - half * n)))
    barb.set_fill(color, 1).set_stroke(color, 0)
    return VGroup(shaft, barb)


def u_text(string, size_u, x, y, font=LABEL_FONT, bold=False, italic=False, align="left", color=FRESH_TAN,
           weight=None):
    """Text at storyboard size with its baseline at y. x is its left edge ("left"), right edge ("right") or middle.

    A throwaway "H" is set in front of the string: its height fixes the scale and its bottom is the baseline, so
    strings with descenders sit correctly without per-string fudge factors.
    """
    full = Text("H" + string, font=font, weight=weight or (BOLD if bold else NORMAL),
                slant=ITALIC if italic else NORMAL, font_size=48)
    full.scale(CAP.get(font, 0.7) * size_u * U / full[0].get_height())
    t = full[1:]
    baseline = full[0].get_bottom()[1]
    edge_x = {"left": t.get_left()[0], "right": t.get_right()[0]}.get(align, t.get_center()[0])
    t.shift(P(x, y) - np.array([edge_x, baseline, 0.0]))
    return t.set_color(color)


# ----------------------------------------------------------------------------- reading storyboard SVGs
_NUM = r"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"


def _css_classes(svg_text):
    """{class name: {property: value}} from the <style> block Illustrator writes."""
    styles = {}
    m = re.search(r"<style[^>]*>(.*?)</style>", svg_text, flags=re.S)
    for selectors, body in re.findall(r"([^{}]+)\{([^}]*)\}", m.group(1) if m else ""):
        props = dict((k.strip(), v.strip()) for k, v in (kv.split(":", 1) for kv in body.split(";") if ":" in kv))
        for sel in selectors.split(","):
            styles.setdefault(sel.strip().lstrip("."), {}).update(props)
    return styles


def _matrix(transform):
    """SVG transform attribute (translate / rotate / scale) -> 3x3 matrix."""
    M = np.eye(3)
    for name, args in re.findall(r"(\w+)\(([^)]*)\)", transform or ""):
        a = [float(v) for v in re.findall(_NUM, args)]
        T = np.eye(3)
        if name == "translate":
            T[0, 2], T[1, 2] = a[0], (a[1] if len(a) > 1 else 0.0)
        elif name == "rotate":
            c, s = np.cos(np.radians(a[0])), np.sin(np.radians(a[0]))
            T[:2, :2] = [[c, -s], [s, c]]
        elif name == "scale":
            T[0, 0], T[1, 1] = a[0], (a[1] if len(a) > 1 else a[0])
        M = M @ T
    return M


def _path_mob(d, M):
    """SVG path data (M L H V C S Z, absolute or relative, implicit repeats) -> VMobject."""
    tokens = re.findall(r"[A-Za-z]|" + _NUM, d)
    mob = VMobject()
    pos, start, last_ctrl, cmd, i = np.zeros(2), np.zeros(2), None, None, 0

    def pt(xy):
        v = M @ np.array([xy[0], xy[1], 1.0])
        return P(v[0], v[1])

    def nums(n):
        nonlocal i
        vals = [float(t) for t in tokens[i:i + n]]
        i += n
        return vals

    while i < len(tokens):
        if re.match(r"[A-Za-z]", tokens[i]):
            cmd = tokens[i]
            i += 1
        rel, c = cmd.islower(), cmd.upper()
        if c == "Z":
            mob.add_line_to(pt(start))
            pos, last_ctrl = start.copy(), None
            continue
        if c == "M":
            xy = np.array(nums(2))
            pos = pos + xy if rel else xy
            start = pos.copy()
            mob.start_new_path(pt(pos))
            cmd, last_ctrl = ("l" if rel else "L"), None
        elif c == "L":
            xy = np.array(nums(2))
            pos = pos + xy if rel else xy
            mob.add_line_to(pt(pos))
            last_ctrl = None
        elif c == "H":
            x, = nums(1)
            pos = np.array([pos[0] + x if rel else x, pos[1]])
            mob.add_line_to(pt(pos))
            last_ctrl = None
        elif c == "V":
            y, = nums(1)
            pos = np.array([pos[0], pos[1] + y if rel else y])
            mob.add_line_to(pt(pos))
            last_ctrl = None
        elif c in "CS":
            if c == "C":
                v = np.array(nums(6)).reshape(3, 2)
                c1, c2, end = (v + pos if rel else v)
            else:
                v = np.array(nums(4)).reshape(2, 2)
                c2, end = (v + pos if rel else v)
                c1 = 2 * pos - last_ctrl if last_ctrl is not None else pos
            mob.add_cubic_bezier_curve_to(pt(c1), pt(c2), pt(end))
            pos, last_ctrl = np.array(end), np.array(c2)
        else:
            raise ValueError(f"SVG path command {cmd!r} not handled")
    return mob


def read_svg_shapes(path, region=None, bg=BG_FALLBACK):
    """Vector shapes of a storyboard SVG (line, rect, circle, ellipse, polygon, polyline, path) as styled VMobjects, in
    document order. region = (x0, y0, x1, y1) in storyboard units keeps only shapes that lie entirely inside it.

    Each mobject gets .svg_tag, .svg_class and .svg (its resolved style: fill, stroke, stroke_u, opacity, rounded).
    """
    svg_text = Path(path).read_text()
    css = _css_classes(svg_text)
    root = ET.fromstring(svg_text)
    out = []

    def walk(node, M):
        tag = node.tag.split("}")[-1]
        if tag in ("defs", "style", "text", "use", "image"):
            return
        M = M @ _matrix(node.get("transform"))
        if tag in ("svg", "g"):
            for child in node:
                walk(child, M)
            return
        A = lambda k, default=0.0: float(node.get(k, default))
        tp = lambda x, y: P(*(M @ np.array([x, y, 1.0]))[:2])
        rounded = False
        if tag == "line":
            mob = Line(tp(A("x1"), A("y1")), tp(A("x2"), A("y2")))
        elif tag == "rect":
            x, y, w, h = A("x"), A("y"), A("width"), A("height")
            corners = np.array([tp(x, y), tp(x + w, y), tp(x + w, y + h), tp(x, y + h)])
            ext = corners.max(0) - corners.min(0)
            rx = A("rx")
            rounded = rx > 0
            mob = RoundedRectangle(ext[0], ext[1], corner_radius=rx * U) if rounded else Rectangle(ext[0], ext[1])
            mob.move_to(corners.mean(0))
        elif tag == "circle":
            mob = Circle(radius=A("r") * U).move_to(tp(A("cx"), A("cy")))
        elif tag == "ellipse":
            mob = Circle(radius=A("rx") * U)
            if A("ry") != A("rx"):
                mob.stretch(A("ry") / A("rx"), 1)
            mob.move_to(tp(A("cx"), A("cy")))
        elif tag in ("polygon", "polyline"):
            v = np.array([float(t) for t in re.findall(_NUM, node.get("points", ""))]).reshape(-1, 2)
            pts = [tp(*q) for q in v]
            mob = Polygon(*pts) if tag == "polygon" else VMobject().set_points_as_corners(pts)
        elif tag == "path":
            mob = _path_mob(node.get("d", ""), M)
        else:
            return
        if region is not None:
            x0, x1 = mob.get_left()[0] / U + 96, mob.get_right()[0] / U + 96
            y0, y1 = 54 - mob.get_top()[1] / U, 54 - mob.get_bottom()[1] / U
            if not (region[0] <= x0 and x1 <= region[2] and region[1] <= y0 and y1 <= region[3]):
                return      # (whole shape must be inside: the full-frame background shapes share the same center)
        st = {}
        for name in (node.get("class") or "").split():
            st.update(css.get(name, {}))
        fill, stroke = st.get("fill", "#000000"), st.get("stroke", "none")
        opacity = float(st.get("opacity", 1))
        stroke_u = float(st.get("stroke-width", "1").replace("px", ""))
        swap = lambda col: bg if col.lower() == SVG_BG else col
        if fill == "none" or tag in ("line", "polyline"):
            mob.set_fill(opacity=0)
        else:
            mob.set_fill(swap(fill), opacity)
        if stroke == "none":
            mob.set_stroke(width=0)
        else:
            mob.set_stroke(swap(stroke), stroke_u * STROKE_PER_U, opacity)
        mob.svg_tag, mob.svg_class = tag, node.get("class", "")
        mob.svg = dict(fill=fill, stroke=stroke, stroke_u=stroke_u, opacity=opacity, rounded=rounded)
        out.append(mob)

    walk(root, np.eye(3))
    return out


# ----------------------------------------------------------------------------- transformer graphic
TF_SVG = SKETCHES / "p43_59-11.svg"
TF_REGION = (65.5, 52.3, 128.9, 80.3)       # the transformer's box in p43_59-11.svg
TF_LAYERS = ("border", "columns", "cells", "grid", "wires", "nodes", "dots")


def transformer_graphic(bg=BG_FALLBACK, label=True, svg=None, region=None, label_spec=None, thin_u=0.15,
                        grid_stroke_u=None):
    """The TRANSFORMER box from storyboard -11 (or from another storyboard: svg, region, label_spec = (size, x,
    baseline), thin_u = strokes thinner than this are icon grid lines, grid_stroke_u = force their width / color),
    at its storyboard position.

    -> VGroup with named layers (also in this order, back to front):
       .border  .columns  .cells  .grid  .wires  .nodes  .dots  .label
    .grid holds, for each of the nine attention icons, its outline square and its eight inner lines.
    """
    layers = {k: VGroup() for k in TF_LAYERS}
    for m in read_svg_shapes(svg or TF_SVG, region or TF_REGION, bg):
        s = m.svg
        thin = s["stroke"] != "none" and s["stroke_u"] < thin_u
        if m.svg_tag == "rect":
            key = "columns" if s["rounded"] else ("grid" if s["fill"] == "none" else "cells")
        elif m.svg_tag in ("circle", "ellipse"):
            key = "nodes"
        elif m.svg_tag in ("line", "polyline"):
            key = "grid" if thin else "wires"
        elif m.svg_tag == "path":
            key = "border" if s["fill"] == "none" else ("nodes" if s["fill"].lower() == SVG_BG else "dots")
        else:
            continue
        if key == "grid" and grid_stroke_u:
            m.set_stroke(FRESH_TAN, grid_stroke_u * STROKE_PER_U, 1)
        layers[key].add(m)
    g = VGroup(*layers.values())
    for z, (k, v) in enumerate(layers.items()):
        setattr(g, k, v)
        for m in v:     # manimgl draws a batch's fills, then its strokes; a per-layer tag keeps the layers in separate
            m.uniforms["z_layer"] = float(z)    # batches so e.g. the node fills really do cover the wires behind them
    g.label = VGroup()
    if label:
        size, x, y = label_spec or (3.58, 83.43, 81.04)
        g.label = u_text("TRANSFORMER", size, x, y, bold=True)
        g.add(g.label)
    return g


def storyboard_xy(point):
    """manim point -> storyboard (x, y)."""
    return point[0] / U + 96, 54 - point[1] / U


def sentence_words(words, size_u, x, y, align="center"):
    """Words set as one line of italic text with its baseline at y (x = its middle, or its left / right edge with
    align) -> one VGroup per word (set together first, so the spacing is the font's own)."""
    line = u_text(" ".join(words), size_u, x, y, font=TEXT_FONT, italic=True, align=align)
    counts = [len(w) for w in words]
    if len(line) == sum(counts):                        # the usual case: one submobject per visible character
        out, k = [], 0
        for n in counts:
            out.append(VGroup(*line[k:k + n]))
            k += n
        return out
    out = []                                            # otherwise place each word on its own, left to right
    gaps = len(words) - 1
    singles = [u_text(w, size_u, 0, y, font=TEXT_FONT, italic=True) for w in words]
    space = (line.get_width() - sum(s.get_width() for s in singles)) / max(gaps, 1)
    left = line.get_left()[0]
    for s in singles:
        s.shift(RIGHT * (left - s.get_left()[0]))
        left = s.get_right()[0] + space
        out.append(s)
    return out


def clip_frame_path(frame):
    """One frame of the finished ball clip: <RUN>/p43/xt/<last step folder>/frame_FFF.png"""
    root = RUN / "p43" / "xt"
    dirs = sorted(d for d in root.glob("step_*") if d.is_dir())
    if not dirs:
        raise FileNotFoundError(
            f"No step folders in {root}\n  -> point WM_P43_59 at the notebook's p43_59_3 output folder")
    path = dirs[-1] / f"frame_{frame:03d}.png"
    if not path.exists():
        raise FileNotFoundError(f"Ball clip frame not found: {path}")
    return path


def under_bump(alpha):
    """0 -> 1 -> 0 over a move, with a flat stretch in the middle (down, along, back up)."""
    return float(np.sin(PI / 2 * min(1.0, 1.5 * np.sin(PI * alpha))))


def fly_under(mob, end, dip_u, mid_scale=1.0, **kwargs):
    """Animation: mob travels to the point `end`, dipping dip_u storyboard units on the way (to pass under a model
    box) and shrinking to mid_scale of its size while it is down there."""
    start, width = mob.get_center().copy(), mob.get_width()
    end = np.array(end, float)

    def update(m, alpha):
        b = under_bump(alpha)
        if mid_scale != 1.0:
            m.set_width(width * (1 - (1 - mid_scale) * b))
        m.move_to(interpolate(start, end, alpha) + dip_u * U * b * DOWN)

    return UpdateFromAlphaFunc(mob, update, **kwargs)


# ----------------------------------------------------------------------------- scene
class P50_59(InteractiveScene):
    # --- toggles -------------------------------------------------------------------------------------------
    show_text = True            # title / TRANSFORMER label as manim Text. False: add them in post
    column_labels = True        # copies of the words also go under the pattern's columns

    # --- timings (seconds) ---------------------------------------------------------------------------------
    intro_time = 5.0            # title + sentence + arrow + transformer, all at once
    intro_hold = 2.0
    isolate_time = 1.5          # transformer (all but one icon), arrow and "Paris" fade
    isolate_hold = 0.5
    zoom_time = 3.5             # icon -> full pattern, words -> labels
    word_lag = 0.15             # stagger between the words as they leave (LaggedStart lag_ratio)
    word_arc = 0.0              # path_arc for the row labels' flight (0 = straight lines)
    p50_end_hold = 4.0

    # P51
    shift_time = 1.5            # pattern + labels slide left
    targets_time = 2.5          # headers, arrows and targets come in
    targets_hold = 2.0
    hl_draw_time = 0.3          # each yellow box
    hl_gap = 0.25               # pause after each cell's box ("The" . "capital" . "of" ...)
    hl_target_gap = 0.4         # extra pause before the target's box
    hl_hold = 1.5               # whole row + target highlighted
    hl_fade_time = 0.6
    hl_between_rows = 0.4
    teacher_pre_hold = 2.0      # the pause before rearranging into the teacher forcing table
    teacher_time = 3.5
    teacher_lag = 0.08          # stagger between rows as the words fly (LaggedStart lag_ratio)
    p51_end_hold = 4.0
    show_bos_note = True        # the small "<bos>" footnote in the lower right

    # P52
    clear_time = 1.5            # teacher forcing table fades, title crossfades
    llm_intro_time = 2.0        # LLM box + arrows + "The"
    llm_intro_hold = 0.8
    llm_step_time = 0.9         # pulse through the model, next word appears
    llm_out_hold = 0.5          # the new word sits on the output side
    llm_feed_time = 1.2         # ... then goes under the box to the end of the input
    llm_feed_hold = 0.2
    llm_end_hold = 2.5          # "The capital of France is" -> "Paris"
    llm_up_time = 1.5           # LLM row moves up for the image row
    wm_intro_time = 2.0         # WORLD MODEL box + arrows + first context frame + bracket
    wm_intro_hold = 0.8
    wm_step_time = 1.0
    wm_out_hold = 0.7
    wm_feed_time = 1.6
    wm_feed_hold = 0.3
    p52_end_hold = 4.0

    # --- layout (storyboard units) -------------------------------------------------------------------------
    TITLE = ("LLM TRAINING", 4.2, 80.72, 9.88)
    SENTENCE = (4.2, 96.0, 37.92)               # size, center x, baseline y  (storyboard -11)
    PROMPT_ARROW = ((96.0, 42.86), (96.0, 47.74))
    ICON = (0, 0)                               # which icon in the transformer: (column 0-2, row 0-2 from the top)

    PATTERN_N = 5
    PATTERN_SIZE = 37.24                        # storyboard -12
    PATTERN_CENTER = (96.0, 55.57)              # centered for P50; -12 has it at x = 81.51 (P51 moves it there)
    PATTERN_STROKE = 0.31
    LABEL_SIZE = 3.8
    ROW_LABEL_GAP = 1.85                        # pattern's left edge -> right edge of the row labels
    TEXT_DROP = 1.3                             # row center -> baseline of that row's text (labels, targets)
    COL_LABEL_GAP = 2.6                         # pattern's bottom edge -> top end of the rotated column labels
    COL_LABEL_SHIFT = 1.2                       # column center -> label baseline (labels read bottom to top)

    # P51 (storyboard -12)
    PATTERN_CENTER_51 = (81.51, 55.57)
    HEADER_SIZE, HEADER_LEADING = 3.2, 3.84
    HEADER_Y = 28.12                            # baseline of the headers' first line
    TARGET_HEADER_X = 126.0                     # center of TARGET / (NEXT TOKEN)
    ARROW_X = (106.34, 114.16)                  # tail, tip
    ARROW_STROKE = 0.3
    ARROW_HEAD = (2.99, 1.22, 2.28)             # length, half width, notch depth
    TARGET_X = 120.83                           # left edge of the targets
    BOS_NOTE = ("Note that we’re ignoring the <bos> token before “The”", 1.8, 137.67, 104.24)
    HL_STROKE = 1.0
    HL_TARGET_PAD = (1.3, 5.6)                  # target's box: padding each side of the word, box height

    # teacher forcing table (storyboard -13)
    TF_TITLE = ("TEACHER FORCING", 4.2, 77.0, 9.88)
    TF_ROW0 = 46.6                              # center of the first row (rows keep the pattern's spacing)
    TF_CONTEXT_X = 78.9                         # right edge of the context column
    TF_ARROW_X = 98.15                          # arrow tails
    TF_TARGET_X = 122.49
    TF_HEADER_Y = 34.55
    TF_INPUT_HEADER_X = 73.3                    # center of INPUT/ CONTEXT
    TF_TARGET_HEADER_X = 127.66

    # P52: LLM row (storyboard -14)
    AR_TITLE = ("AUTOREGRESSIVE INFERENCE", 4.2, 65.24, 9.88)
    MODEL_ARROW = (0.3, (2.99, 1.22, 2.28))     # stroke, head (length, half width, notch depth)
    MODEL_STROKE = 0.2
    MODEL_R = 1.17                              # corner radius of the model boxes
    LLM_CENTER = (97.77, 50.86)
    LLM_BOX = 14.32
    LLM_LABEL = ("LLM", 4.2, 52.44)             # string, size, baseline
    LLM_IN_ARROW = (79.1, 86.93)                # tail x, tip x
    LLM_OUT_ARROW = (107.9, 115.72)
    LLM_INPUT_X = 75.6                          # right edge of the input words
    LLM_OUTPUT_X = 118.69                       # left edge of the output word
    LLM_DIP = 10.5                              # how far a word drops to pass under the box
    LLM_UP = (13.48, -13.74)                    # the row's move to its storyboard -15 position

    # P52: world model row (storyboard -15)
    WM_FRAMES = (0, 4, 8, 12)                   # frames of the 48-frame clip: 3 context frames + the last prediction
    WM_Y = 76.4                                 # the row's center line
    WM_IMG = (33.28, 19.2)
    WM_CONTEXT_X = (8.33, 43.39, 78.41)         # left edges of the three context slots
    WM_PRED_X = 150.35
    WM_CENTER_X = 132.0
    WM_BOX = 15.13
    WM_LABEL = (("WORLD", "MODEL"), 3.2, 3.84)  # lines, size, leading
    WM_IN_ARROW = (114.73, 122.56)
    WM_OUT_ARROW = (141.04, 148.23)
    WM_BRACKET = (89.61, 91.83, 1.07, 1.6)      # top y, bottom y, corner radius, gap either side of the label
    WM_NOTE_SIZE, WM_NOTE_Y = 3.2, 92.64        # CONTEXT / PREDICTION
    WM_DIP = 14.0                               # how far a frame drops to pass under the box ...
    WM_FLY_SCALE = 0.45                         # ... and how small it gets while it is down there

    def construct(self):
        self.setup_scene()
        # P50
        self.beat_intro()
        self.wait(self.intro_hold)
        self.beat_isolate()
        self.wait(self.isolate_hold)
        self.beat_zoom()
        self.wait(self.p50_end_hold)
        # P51
        self.beat_shift_pattern()
        self.beat_targets()
        self.wait(self.targets_hold)
        self.beat_highlight_row(4)              # "The capital of France is" -> "Paris"
        self.beat_clear_highlight()
        self.wait(self.hl_between_rows)
        self.beat_highlight_row(3)              # "The capital of France" -> "is"
        self.beat_clear_highlight()
        self.wait(self.teacher_pre_hold)
        self.beat_teacher_forcing()
        self.wait(self.p51_end_hold)
        # P52
        self.beat_clear_all()
        self.beat_llm_intro()
        self.wait(self.llm_intro_hold)
        for k in range(self.PATTERN_N):         # "The" -> capital, "The capital" -> of, ... -> Paris
            self.beat_llm_step()
            if k < self.PATTERN_N - 1:
                self.wait(self.llm_out_hold)
                self.beat_llm_feed()
                self.wait(self.llm_feed_hold)
        self.wait(self.llm_end_hold)
        self.beat_llm_up()
        self.beat_wm_intro()
        self.wait(self.wm_intro_hold)
        n_context = len(self.WM_CONTEXT_X)
        for k in range(n_context):              # 1 context frame -> 2 -> 3, then one last prediction
            self.beat_wm_step()
            if k < n_context - 1:
                self.wait(self.wm_out_hold)
                self.beat_wm_feed()
                self.wait(self.wm_feed_hold)
        self.wait(self.p52_end_hold)

    # ------------------------------------------------------------------------------------------- setup
    def setup_scene(self):
        try:
            self.bg = rgb_to_hex(self.camera.background_rgba[:3])
        except Exception:
            self.bg = BG_FALLBACK

        string, size, x, y = self.TITLE
        self.title = u_text(string, size, x, y, bold=True, color=CHILL_BROWN)
        self.tf = transformer_graphic(self.bg, label=self.show_text)
        self.words = sentence_words(WORDS, *self.SENTENCE)     # six mobjects, one per word
        self.prompt_arrow = arrow(*self.PROMPT_ARROW)
        self.pattern_center = self.PATTERN_CENTER       # where the big pattern currently is

    # ------------------------------------------------------------------------------------------- helpers
    def settle(self, group):
        """After animating a group's pieces one by one, put the group itself back on screen as a single mobject."""
        self.remove(*group.get_family())
        self.add(group)

    def tf_icon(self, col, row):
        """One little attention-pattern icon in the transformer graphic: its filled cells, outline and inner lines.

        -> VGroup(*cells, outline, *h_lines, *v_lines) with .cells (each has .ij = (row, column) in the 5 x 5
           pattern), .outline, .h_lines (top to bottom) and .v_lines (left to right).
        """
        tf, n = self.tf, self.PATTERN_N
        boxes = [m for m in tf.grid if m.svg_tag == "rect"]
        xs = sorted(b.get_center()[0] for b in boxes)
        col_x = xs[len(xs) // 3 * col + 1]                      # the middle x of that column's three icons
        in_col = sorted([b for b in boxes if abs(b.get_center()[0] - col_x) < 2 * U], key=lambda b: -b.get_center()[1])
        outline = in_col[row]
        pad = 0.3 * U
        x0, x1 = outline.get_left()[0] - pad, outline.get_right()[0] + pad
        y0, y1 = outline.get_bottom()[1] - pad, outline.get_top()[1] + pad
        inside = lambda m: x0 <= m.get_center()[0] <= x1 and y0 <= m.get_center()[1] <= y1
        cell = outline.get_width() / n
        cells = [c for c in tf.cells if inside(c)]
        for c in cells:
            j = int((c.get_center()[0] - outline.get_left()[0]) / cell)
            i = int((outline.get_top()[1] - c.get_center()[1]) / cell)
            c.ij = (min(max(i, 0), n - 1), min(max(j, 0), n - 1))
        lines = [l for l in tf.grid if l.svg_tag == "line" and inside(l)]
        h_lines = sorted([l for l in lines if l.get_width() > l.get_height()], key=lambda l: -l.get_center()[1])
        v_lines = sorted([l for l in lines if l.get_width() <= l.get_height()], key=lambda l: l.get_center()[0])
        icon = VGroup(*cells, outline, *h_lines, *v_lines)
        icon.cells, icon.outline, icon.h_lines, icon.v_lines = cells, outline, h_lines, v_lines
        return icon

    def pattern_box(self, center=None):
        """(left, top, size) of the big pattern in storyboard units."""
        cx, cy = center or self.pattern_center
        s = self.PATTERN_SIZE
        return cx - s / 2, cy - s / 2, s

    def row_y(self, k, center=None):
        """Storyboard y of the middle of pattern row k."""
        left, top, s = self.pattern_box(center)
        return top + (k + 0.5) * s / self.PATTERN_N

    def tf_row_y(self, k):
        """... and of row k in the teacher forcing table."""
        return self.TF_ROW0 + k * self.PATTERN_SIZE / self.PATTERN_N

    def header(self, lines, x, y):
        """Two-line brown header centered on x, first baseline at y."""
        return VGroup(*[u_text(s, self.HEADER_SIZE, x, y + k * self.HEADER_LEADING, bold=True, align="center",
                               color=CHILL_BROWN) for k, s in enumerate(lines)])

    def pattern_target(self, icon, center=None):
        """What the icon becomes: the same pieces, in the same order, laid out as the big pattern on a clean grid."""
        left, top, s = self.pattern_box(center)
        n = self.PATTERN_N
        c = s / n
        width = self.PATTERN_STROKE * STROKE_PER_U
        out = []
        for m in icon.cells:
            i, j = m.ij
            t = m.copy()
            t.set_width(c * U, stretch=True).set_height(c * U, stretch=True)
            t.move_to(P(left + (j + 0.5) * c, top + (i + 0.5) * c))
            out.append(t)
        t = icon.outline.copy()
        t.set_width(s * U, stretch=True).set_height(s * U, stretch=True)
        t.move_to(P(left + s / 2, top + s / 2)).set_stroke(width=width)
        out.append(t)

        def line_to(m, a, b):
            t = m.copy()
            if np.dot(m.get_end() - m.get_start(), b - a) < 0:      # keep the line's own direction (no flip mid-move)
                a, b = b, a
            t.put_start_and_end_on(a, b).set_stroke(width=width)
            return t

        for k, m in enumerate(icon.h_lines):        # inner lines, evenly spaced however many the icon has
            y = top + s * (k + 1) / (len(icon.h_lines) + 1)
            out.append(line_to(m, P(left, y), P(left + s, y)))
        for k, m in enumerate(icon.v_lines):
            x = left + s * (k + 1) / (len(icon.v_lines) + 1)
            out.append(line_to(m, P(x, top), P(x, top + s)))
        return VGroup(*out)

    def row_label(self, k, word, center=None):
        left, top, s = self.pattern_box(center)
        c = s / self.PATTERN_N
        return u_text(word, self.LABEL_SIZE, left - self.ROW_LABEL_GAP, top + (k + 0.5) * c + self.TEXT_DROP,
                      font=TEXT_FONT, italic=True, align="right")

    def col_label(self, k, word, center=None):
        """Rotated a quarter turn, reading bottom to top, its end just under the pattern."""
        left, top, s = self.pattern_box(center)
        c = s / self.PATTERN_N
        x, y = left + (k + 0.5) * c + self.COL_LABEL_SHIFT, top + s + self.COL_LABEL_GAP
        t = u_text(word, self.LABEL_SIZE, x, y, font=TEXT_FONT, italic=True, align="right")
        return t.rotate(PI / 2, about_point=P(x, y))

    # ------------------------------------------------------------------------------------------- P50
    def beat_intro(self, run_time=None):
        """Title, sentence, arrow and transformer draw in together (storyboard -11)."""
        tf = self.tf
        sentence = VGroup(*self.words)
        anims = [
            LaggedStart(
                AnimationGroup(ShowCreation(tf.border), FadeIn(tf.columns, lag_ratio=0.3)),
                AnimationGroup(FadeIn(tf.cells, lag_ratio=0.02), ShowCreation(tf.grid, lag_ratio=0.02), FadeIn(tf.dots)),
                AnimationGroup(ShowCreation(tf.wires, lag_ratio=0.01), FadeIn(tf.nodes, lag_ratio=0.02)),
                *([Write(tf.label)] if len(tf.label) else []),
                lag_ratio=0.35),
            Write(sentence, lag_ratio=0.05),
            ShowCreation(self.prompt_arrow[0]),
            FadeIn(self.prompt_arrow[1]),
        ]
        if self.show_text:
            anims.append(Write(self.title))
        self.play(*anims, run_time=run_time or self.intro_time)
        self.settle(tf)
        self.remove(*sentence.get_family())     # from here on each word is its own mobject
        self.add(*self.words)

    def beat_isolate(self, run_time=None):
        """The transformer fades except for one attention-pattern icon; the arrow and "Paris" go with it."""
        tf = self.tf
        self.icon = icon = self.tf_icon(*self.ICON)
        mine = set(id(m) for m in icon)
        self.tf_rest = VGroup(*[m for key in TF_LAYERS for m in getattr(tf, key) if id(m) not in mine],
                              *([tf.label] if len(tf.label) else []))
        self.remove(*tf.get_family())           # the transformer is now two mobjects: the icon and the rest
        self.add(self.tf_rest, icon)
        self.play(FadeOut(self.tf_rest), FadeOut(self.prompt_arrow), FadeOut(self.words[-1]),
                  run_time=run_time or self.isolate_time)

    def beat_zoom(self, run_time=None, word_lag=None):
        """The icon grows into the full pattern, centered; the words become its row (and column) labels."""
        icon, words = self.icon, self.words[:self.PATTERN_N]
        lag = self.word_lag if word_lag is None else word_lag
        rows = [self.row_label(k, w) for k, w in enumerate(WORDS[:self.PATTERN_N])]
        anims = [Transform(icon, self.pattern_target(icon))]
        cols = []
        if self.column_labels:
            cols = [w.copy() for w in words]
            targets = [self.col_label(k, w) for k, w in enumerate(WORDS[:self.PATTERN_N])]
            # (path_arc = the quarter turn itself, so each copy swings round rigidly instead of squashing)
            anims.append(LaggedStart(*[Transform(m, t, path_arc=PI / 2) for m, t in zip(cols, targets)],
                                     lag_ratio=lag))
        anims.append(LaggedStart(*[Transform(w, t, path_arc=self.word_arc) for w, t in zip(words, rows)],
                                 lag_ratio=lag))
        self.add(*cols)
        self.bring_to_front(*words)             # the words stay above the growing pattern
        self.play(*anims, run_time=run_time or self.zoom_time)
        self.pattern = icon
        self.row_labels = VGroup(*words)
        self.col_labels = VGroup(*cols)

    # ------------------------------------------------------------------------------------------- P51
    def beat_shift_pattern(self, run_time=None):
        """The pattern and its labels slide left, to where storyboard -12 has them."""
        (x0, y0), (x1, y1) = self.pattern_center, self.PATTERN_CENTER_51
        delta = P(x1, y1) - P(x0, y0)
        self.play(*[m.animate.shift(delta) for m in (self.pattern, self.row_labels, self.col_labels)],
                  run_time=run_time or self.shift_time)
        self.pattern_center = self.PATTERN_CENTER_51

    def beat_targets(self, run_time=None):
        """Headers, arrows and targets come in to the right of the pattern (storyboard -12)."""
        n = self.PATTERN_N
        x0, x1 = self.ARROW_X
        self.input_header = self.header(["INPUT/", "CONTEXT"], self.pattern_center[0], self.HEADER_Y)
        self.target_header = self.header(["TARGET", "(NEXT TOKEN)"], self.TARGET_HEADER_X, self.HEADER_Y)
        self.arrows = VGroup(*[arrow((x0, self.row_y(k)), (x1, self.row_y(k)), self.ARROW_STROKE, self.ARROW_HEAD)
                               for k in range(n)])
        self.targets = VGroup(*[
            u_text(w, self.LABEL_SIZE, self.TARGET_X, self.row_y(k) + self.TEXT_DROP, font=TEXT_FONT, italic=True)
            for k, w in enumerate(WORDS[1:n + 1])])
        string, size, x, y = self.BOS_NOTE
        self.bos_note = u_text(string, size, x, y, weight="SEMIBOLD", color=CHILL_BROWN)
        anims = [
            LaggedStart(*[AnimationGroup(ShowCreation(a[0]), FadeIn(a[1])) for a in self.arrows], lag_ratio=0.15),
            LaggedStart(*[Write(t) for t in self.targets], lag_ratio=0.15),
        ]
        if self.show_text:
            anims += [Write(self.input_header), Write(self.target_header)]
            if self.show_bos_note:
                anims.append(FadeIn(self.bos_note))
        self.play(*anims, run_time=run_time or self.targets_time)

    def cell_box(self, i, j):
        """Yellow outline on cell (row i, column j) of the pattern."""
        left, top, s = self.pattern_box()
        c = s / self.PATTERN_N
        box = Rectangle(c * U, c * U).move_to(P(left + (j + 0.5) * c, top + (i + 0.5) * c))
        return box.set_stroke(YELLOW, self.HL_STROKE * STROKE_PER_U).set_fill(opacity=0)

    def target_box(self, i):
        """Yellow outline round row i's target word."""
        word = self.targets[i]
        pad, height = self.HL_TARGET_PAD
        box = Rectangle(word.get_width() + 2 * pad * U, height * U)
        box.move_to(np.array([word.get_center()[0], P(0, self.row_y(i))[1], 0.0]))
        return box.set_stroke(YELLOW, self.HL_STROKE * STROKE_PER_U).set_fill(opacity=0)

    def beat_highlight_row(self, row, draw_time=None, gap=None, target_gap=None, hold=None):
        """Yellow boxes on row `row`: its cells one at a time, left to right (just the ones it can see, up to the
        diagonal), then its target. gap can be one number or a list with one pause per cell, to follow the VO."""
        draw_time = self.hl_draw_time if draw_time is None else draw_time
        gap = self.hl_gap if gap is None else gap
        gaps = list(gap) if isinstance(gap, (list, tuple)) else [gap] * (row + 1)
        self.highlights = VGroup()
        for j in range(row + 1):
            box = self.cell_box(row, j)
            self.highlights.add(box)
            self.play(ShowCreation(box), run_time=draw_time)
            self.wait(gaps[min(j, len(gaps) - 1)])
        self.wait(self.hl_target_gap if target_gap is None else target_gap)
        box = self.target_box(row)
        self.highlights.add(box)
        self.play(ShowCreation(box), run_time=draw_time)
        self.wait(self.hl_hold if hold is None else hold)

    def beat_clear_highlight(self, run_time=None):
        self.play(FadeOut(self.highlights), run_time=run_time or self.hl_fade_time)

    def beat_teacher_forcing(self, run_time=None, lag=None):
        """The pattern goes; what is left rearranges into the teacher forcing table (storyboard -13)."""
        n = self.PATTERN_N
        lag = self.teacher_lag if lag is None else lag
        # where every word of every context row ends up: row i = words 0 .. i, right-aligned
        slots = [sentence_words(WORDS[:i + 1], self.LABEL_SIZE, self.TF_CONTEXT_X, self.tf_row_y(i) + self.TEXT_DROP,
                                align="right") for i in range(n)]
        rows, fly = [], []
        for i in range(n):
            # the row's own label becomes its last word; the earlier words swing up out of the column labels
            copies = [self.col_labels[j].copy() for j in range(i)]
            self.add(*copies)
            fly.append(AnimationGroup(
                *[Transform(m, slots[i][j], path_arc=-PI / 2) for j, m in enumerate(copies)],
                Transform(self.row_labels[i], slots[i][i])))
            rows.append(VGroup(*copies, self.row_labels[i]))
        dy = [P(0, self.tf_row_y(k))[1] - P(0, self.row_y(k))[1] for k in range(n)]
        anims = [
            FadeOut(self.pattern), FadeOut(self.col_labels),
            LaggedStart(*fly, lag_ratio=lag),
            *[a.animate.shift(np.array([(self.TF_ARROW_X - self.ARROW_X[0]) * U, dy[k], 0.0]))
              for k, a in enumerate(self.arrows)],
            *[t.animate.shift(np.array([(self.TF_TARGET_X - self.TARGET_X) * U, dy[k], 0.0]))
              for k, t in enumerate(self.targets)],
        ]
        if self.show_text:
            string, size, x, y = self.TF_TITLE
            self.tf_title = u_text(string, size, x, y, bold=True, color=CHILL_BROWN)
            hy = (self.HEADER_Y - self.TF_HEADER_Y) * U
            anims += [
                FadeOut(self.title), FadeIn(self.tf_title),
                self.input_header.animate.shift(
                    np.array([(self.TF_INPUT_HEADER_X - self.pattern_center[0]) * U, hy, 0.0])),
                self.target_header.animate.shift(
                    np.array([(self.TF_TARGET_HEADER_X - self.TARGET_HEADER_X) * U, hy, 0.0])),
            ]
        self.play(*anims, run_time=run_time or self.teacher_time)
        self.context_rows = rows                # row i: VGroup of its words, left to right
        if self.show_text:
            self.title = self.tf_title

    # ------------------------------------------------------------------------------------------- P52
    @staticmethod
    def pulse(vmob, width_factor=3.0, time_width=0.6, **kwargs):
        """A brighter, thicker flash running along a stroke."""
        hot = vmob.copy().set_fill(opacity=0).set_stroke(FRESH_TAN, float(vmob.get_stroke_width()) * width_factor, 1)
        return VShowPassingFlash(hot, time_width=time_width, **kwargs)

    @staticmethod
    def fade_image(image, start=0.0, end=1.0, **kwargs):
        return UpdateFromAlphaFunc(image, lambda m, a: m.set_opacity(interpolate(start, end, a)), **kwargs)

    def model_box(self, cx, cy, size):
        box = RoundedRectangle(size * U, size * U, corner_radius=self.MODEL_R * U).move_to(P(cx, cy))
        return box.set_stroke(FRESH_TAN, self.MODEL_STROKE * STROKE_PER_U).set_fill(opacity=0)

    def model_arrow(self, xs, y):
        stroke, head = self.MODEL_ARROW
        return arrow((xs[0], y), (xs[1], y), stroke, head)

    def forward_pulse(self, in_arrow, box, out_arrow, appear, run_time):
        """in arrow -> box -> out arrow light up in turn, then the output appears."""
        return LaggedStart(self.pulse(in_arrow[0]), self.pulse(box, time_width=0.4), self.pulse(out_arrow[0]),
                           appear, lag_ratio=0.3, run_time=run_time)

    def beat_clear_all(self, run_time=None):
        """Everything from the teacher forcing table fades; the title crossfades to AUTOREGRESSIVE INFERENCE."""
        gone = [*self.context_rows, self.arrows, self.targets]
        anims = []
        if self.show_text:
            gone += [self.input_header, self.target_header, self.title]
            if self.show_bos_note:
                gone.append(self.bos_note)
            string, size, x, y = self.AR_TITLE
            new_title = u_text(string, size, x, y, bold=True, color=CHILL_BROWN)
            anims.append(FadeIn(new_title))
        self.play(*[FadeOut(m) for m in gone], *anims, run_time=run_time or self.clear_time)
        if self.show_text:
            self.title = new_title

    def llm_input_slots(self, n):
        """Where the first n words sit as the LLM's input: one line, right-aligned next to the input arrow."""
        return sentence_words(WORDS[:n], self.LABEL_SIZE, self.LLM_INPUT_X, self.LLM_CENTER[1] + self.TEXT_DROP,
                              align="right")

    def beat_llm_intro(self, run_time=None):
        """LLM box, arrows and the first input word (storyboard -14, minus the output)."""
        cx, cy = self.LLM_CENTER
        string, size, base = self.LLM_LABEL
        self.llm_box = self.model_box(cx, cy, self.LLM_BOX)
        self.llm_label = u_text(string, size, cx, base, bold=True, align="center")
        self.llm_in = self.model_arrow(self.LLM_IN_ARROW, cy)
        self.llm_out = self.model_arrow(self.LLM_OUT_ARROW, cy)
        self.llm_input = self.llm_input_slots(1)        # list of word mobjects, left to right
        self.llm_output = None                          # the word currently on the output side
        self.play(ShowCreation(self.llm_box), Write(self.llm_label),
                  *[AnimationGroup(ShowCreation(a[0]), FadeIn(a[1])) for a in (self.llm_in, self.llm_out)],
                  Write(self.llm_input[0]),
                  run_time=run_time or self.llm_intro_time)

    def beat_llm_step(self, run_time=None):
        """The LLM produces the next word."""
        k = len(self.llm_input)                         # index of the word about to come out
        word = u_text(WORDS[k], self.LABEL_SIZE, self.LLM_OUTPUT_X, self.LLM_CENTER[1] + self.TEXT_DROP,
                      font=TEXT_FONT, italic=True)
        self.llm_output = word
        self.play(self.forward_pulse(self.llm_in, self.llm_box, self.llm_out, FadeIn(word, shift=0.15 * RIGHT),
                                     run_time or self.llm_step_time))

    def beat_llm_feed(self, run_time=None):
        """The output word goes under the box and joins the end of the input; the input shuffles left."""
        word = self.llm_output
        slots = self.llm_input_slots(len(self.llm_input) + 1)
        anims = [m.animate.shift(slot.get_center() - m.get_center()) for m, slot in zip(self.llm_input, slots)]
        anims.append(fly_under(word, slots[-1].get_center(), self.LLM_DIP))
        self.play(*anims, run_time=run_time or self.llm_feed_time)
        self.llm_input.append(word)
        self.llm_output = None

    def beat_llm_up(self, run_time=None):
        """The whole LLM row moves to its place in storyboard -15 and stays as it is."""
        dx, dy = self.LLM_UP
        self.llm_row = VGroup(self.llm_box, self.llm_label, self.llm_in, self.llm_out, *self.llm_input,
                              *([self.llm_output] if self.llm_output is not None else []))
        self.play(self.llm_row.animate.shift(np.array([dx * U, -dy * U, 0.0])), run_time=run_time or self.llm_up_time)

    # ... the same loop with video frames
    def wm_image(self, frame, x):
        img = ImageMobject(str(clip_frame_path(frame)))
        try:
            img.image.load()
        except Exception:
            pass
        w, h = self.WM_IMG
        img.set_width(w * U, stretch=True).set_height(h * U, stretch=True)
        return img.move_to(self.wm_slot_point(x))

    def wm_slot_point(self, x):
        """Center of the image slot whose left edge is x."""
        return P(x + self.WM_IMG[0] / 2, self.WM_Y)

    def wm_bracket(self, n):
        """CONTEXT bracket under the last n context slots -> VGroup(left line, right line, label)."""
        y0, y1, r, gap = self.WM_BRACKET
        x0, x1 = self.WM_CONTEXT_X[-n], self.WM_CONTEXT_X[-1] + self.WM_IMG[0]
        mid = (x0 + x1) / 2
        label = u_text("CONTEXT", self.WM_NOTE_SIZE, mid, self.WM_NOTE_Y, bold=True, align="center", color=CHILL_BROWN)
        half = label.get_width() / U / 2 + gap

        def side(x, sign):      # down from the corner of the images, round the corner, in toward the label
            t = np.linspace(PI, PI / 2, 8)
            corner = [(x + sign * (r + r * np.cos(a)), y1 - r + r * np.sin(a)) for a in t]
            pts = [(x, y0), *corner, (mid - sign * half, y1)]
            line = VMobject().set_points_as_corners([P(*q) for q in pts])
            return line.set_stroke(CHILL_BROWN, 0.2 * STROKE_PER_U).set_fill(opacity=0)

        return VGroup(side(x0, 1), side(x1, -1), label)

    def beat_wm_intro(self, run_time=None):
        """WORLD MODEL box, arrows, the first context frame and the CONTEXT bracket."""
        y = self.WM_Y
        lines, size, leading = self.WM_LABEL
        self.wm_box = self.model_box(self.WM_CENTER_X, y, self.WM_BOX)
        base = y + CAP[LABEL_FONT] * size / 2 - leading * (len(lines) - 1) / 2      # block centered on the box
        self.wm_label = VGroup(*[u_text(s, size, self.WM_CENTER_X, base + k * leading, bold=True, align="center")
                                 for k, s in enumerate(lines)])
        self.wm_in = self.model_arrow(self.WM_IN_ARROW, y)
        self.wm_out = self.model_arrow(self.WM_OUT_ARROW, y)
        self.wm_context = [self.wm_image(self.WM_FRAMES[0], self.WM_CONTEXT_X[-1]).set_opacity(0)]
        self.wm_pred = None
        self.wm_bracket_mob = self.wm_bracket(1)
        self.wm_pred_label = u_text("PREDICTION", self.WM_NOTE_SIZE, self.WM_PRED_X + self.WM_IMG[0] / 2,
                                    self.WM_NOTE_Y, bold=True, align="center", color=CHILL_BROWN)
        self.add(self.wm_context[0])
        left, right, label = self.wm_bracket_mob
        anims = [ShowCreation(self.wm_box),
                 *[AnimationGroup(ShowCreation(a[0]), FadeIn(a[1])) for a in (self.wm_in, self.wm_out)],
                 self.fade_image(self.wm_context[0]),
                 ShowCreation(left), ShowCreation(right)]
        if self.show_text:
            anims += [Write(self.wm_label), Write(label)]
        self.play(*anims, run_time=run_time or self.wm_intro_time)

    def beat_wm_step(self, run_time=None):
        """The world model produces the next frame."""
        k = len(self.wm_context)
        self.wm_pred = img = self.wm_image(self.WM_FRAMES[k], self.WM_PRED_X).set_opacity(0)
        self.add(img)
        anims = [self.forward_pulse(self.wm_in, self.wm_box, self.wm_out, self.fade_image(img),
                                    run_time or self.wm_step_time)]
        if self.show_text and k == 1:
            anims.append(FadeIn(self.wm_pred_label, run_time=run_time or self.wm_step_time))
        self.play(*anims)

    def beat_wm_feed(self, run_time=None):
        """The prediction passes under the box and becomes the newest context frame; the rest shift left."""
        img = self.wm_pred
        n = len(self.wm_context) + 1
        xs = self.WM_CONTEXT_X[-n:]                     # where the n context frames sit afterwards
        anims = [m.animate.move_to(self.wm_slot_point(x)) for m, x in zip(self.wm_context, xs)]
        anims.append(Transform(self.wm_bracket_mob, self.wm_bracket(n)))
        self.bring_to_front(img)
        anims.append(fly_under(img, self.wm_slot_point(xs[-1]), self.WM_DIP, self.WM_FLY_SCALE))
        self.play(*anims, run_time=run_time or self.wm_feed_time)
        self.wm_context.append(img)
        self.wm_pred = None


# ============================================================================= P53
PROMPT = [
    "“A single red rubber ball is dropped from the top of the frame onto a wooden",
    "floor and bounces several times, each bounce lower than the last, until it comes",
    "to rest. Static camera, soft daylight, plain white wall behind, realistic physics.”",
]
TILE_SIZE = (832, 480)      # pixels of every tile's texture (the Wan2.2 frames' own size)

# How much of a frame is still noise after each diffusion step (0 = clean): measured on the Wan2.2 ball run by
# projecting every step's frame onto the line from its pure-noise frame to its finished frame. 41 values, step 0-40.
NOISE_CURVE = [
    1.0, 0.994, 0.995, 0.993, 0.991, 0.989, 0.986, 0.983, 0.98, 0.977, 0.974, 0.971, 0.967, 0.963, 0.959, 0.955,
    0.95, 0.945, 0.939, 0.933, 0.927, 0.92, 0.912, 0.903, 0.894, 0.883, 0.871, 0.857, 0.841, 0.822, 0.8, 0.774,
    0.742, 0.704, 0.657, 0.598, 0.522, 0.423, 0.296, 0.152, 0.0]


def run_step_dirs():
    """Sorted step folders of the Wan2.2 run: step_00 ... step_40_final."""
    root = RUN / "p43" / "xt"
    dirs = sorted(d for d in root.glob("step_*") if d.is_dir())
    if not dirs:
        raise FileNotFoundError(
            f"No step folders in {root}\n  -> point WM_P43_59 at the notebook's p43_59_3 output folder")
    return dirs


@lru_cache(maxsize=64)
def _rgb(path, crop=False):
    """Image file -> uint8 array (480, 832, 3). crop: center-crop to the tile's shape first (the real video is 16:9)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = TILE_SIZE
        if crop:
            keep = min(im.width, im.height * w / h), min(im.height, im.width * h / w)
            x0, y0 = (im.width - keep[0]) / 2, (im.height - keep[1]) / 2
            im = im.crop((round(x0), round(y0), round(x0 + keep[0]), round(y0 + keep[1])))
        if im.size != TILE_SIZE:
            im = im.resize(TILE_SIZE, Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def tennis_paths():
    paths = sorted(p for p in TENNIS.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not paths:
        raise FileNotFoundError(f"No frames in {TENNIS}\n  -> point WM_TENNIS at the folder of real video frames")
    return paths


class Tile(ImageMobject):
    """A still with a private texture that gets rewritten in place: tile.show(array) with a uint8 (480, 832, 3)
    array. (Same trick as StepVideo in p43_59.py: one mobject and one texture however often its picture changes.)"""
    _count = 0

    def __init__(self, path, pixels):
        self.pixels = pixels
        super().__init__(str(path))
        try:
            self.image.load()               # don't hold the file handle open
        except Exception:
            pass
        Tile._count += 1
        self.uniforms["video_slot"] = float(Tile._count)    # keeps two tiles from being batched onto one texture

    @staticmethod
    def _bytes(pixels):
        return np.dstack([pixels, np.full(pixels.shape[:2], 255, np.uint8)]).tobytes()

    def init_shader_wrapper(self, ctx):
        super().init_shader_wrapper(ctx)
        # a private texture (the stock one is cached per file path and shared between mobjects)
        self.shader_wrapper.textures[0] = ctx.texture(size=TILE_SIZE, components=4, data=self._bytes(self.pixels))

    def show(self, pixels):
        self.pixels = pixels
        wrapper = getattr(self, "shader_wrapper", None)
        if wrapper is not None and wrapper.textures:
            wrapper.textures[0].write(self._bytes(pixels))
        return self

    def place(self, x, y, w, h):
        """Fill the storyboard slot with top-left (x, y) and size (w, h)."""
        self.set_width(w * U, stretch=True).set_height(h * U, stretch=True)
        return self.move_to(P(x + w / 2, y + h / 2))


class RunTile(Tile):
    """One frame of the Wan2.2 run at any diffusion step: tile.set_step(12)"""

    def __init__(self, frame, step=0):
        self.frame, self.dirs = frame, run_step_dirs()
        self.step = int(step)
        super().__init__(self.path(), _rgb(self.path()))

    def path(self):
        return self.dirs[int(np.clip(self.step, 0, len(self.dirs) - 1))] / f"frame_{self.frame:03d}.png"

    def set_step(self, step):
        self.step = int(step)
        return self.show(_rgb(self.path()))


class RealTile(Tile):
    """A real frame with faked diffusion noise: tile.set_level(0.3) blends in 30% of a pure-noise frame from the
    Wan2.2 run; tile.set_step(12) uses the level the real run had after that step (NOISE_CURVE)."""

    def __init__(self, path, noise_path, level=0.0):
        self.clean = _rgb(path, True).astype(np.float32)
        self.noise = _rgb(noise_path).astype(np.float32)
        self.level = float(level)
        super().__init__(path, self.blend())

    def blend(self):
        w = float(np.clip(self.level, 0, 1))
        return np.clip(self.clean * (1 - w) + self.noise * w, 0, 255).astype(np.uint8)

    def set_level(self, level):
        self.level = float(level)
        return self.show(self.blend())

    def set_step(self, step):
        return self.set_level(NOISE_CURVE[int(np.clip(step, 0, len(NOISE_CURVE) - 1))])


class P53(InteractiveScene):
    # --- toggles -------------------------------------------------------------------------------------------
    show_text = True            # title / prompt / TRANSFORMER as manim Text. False: add them in post
    show_step_counter = True    # small "STEP 12 / 40" under the transformer

    # --- timings (seconds) ---------------------------------------------------------------------------------
    intro_time = 2.0            # everything fades in, tiles as noise
    intro_hold = 1.0
    step_time = 0.15            # per diffusion step, generated video (40 steps)
    step_times = None           # or a list of per-step seconds (e.g. linger on the late steps, where it resolves)
    gen_hold = 3.0              # finished red ball frames
    to_real_time = 1.5          # crossfade to the real frames, prompt fades, title changes
    real_hold = 2.0             # clean real frames
    noise_up_time = 3.0         # noise added to the real frames, all the way
    noise_hold = 1.0
    real_step_time = 0.15       # per (faked) diffusion step on the real frames
    real_step_times = None
    grid_end_hold = 2.0         # clean real frames again, before the teacher forcing layout
    teacher_layout_time = 2.5   # grid -> three rows of context frames
    teacher_models_time = 2.0   # boxes, arrows, noise outputs, labels
    teacher_noise_hold = 1.5
    teacher_step_time = 0.15    # per (faked) diffusion step on the three outputs
    teacher_step_times = None
    end_hold = 4.0              # end of P53: the three denoised outputs

    # P54
    rollout_layout_time = 2.0   # one model slides to the middle, the rest fades, title changes
    rollout_start_time = 1.0    # first clean frame fades in as the input
    rollout_start_hold = 1.0
    rollout_step_time = 1.0     # pulse through the model, output fades in
    rollout_out_hold = 0.8      # output sits on the right
    rollout_feed_time = 1.6     # ... then goes under the box to the input side
    rollout_feed_hold = 0.3
    p54_end_hold = 4.0

    # P55
    compare_slide_time = 1.5    # rollout row slides down
    compare_fade_time = 1.5     # training row fades in above it
    compare_labels_time = 2.0   # headings, brackets, labels
    compare_hold = 3.0          # the beat before the title changes
    exposure_title_time = 1.5
    exposure_hold = 3.0         # EXPOSURE BIAS title up, before the LLM version
    llm_swap_time = 2.0         # video rows -> LLM rows, everything at once
    llm_hold = 4.0
    video_swap_time = 1.5       # ... and back
    p55_end_hold = 5.0

    # --- layout: storyboard p43_59-02 (192 x 108 units) ----------------------------------------------------
    TF_SVG = SKETCHES_43 / "p43_59-01.svg"
    TF_REGION = (69.0, 43.0, 124.0, 67.6)       # the transformer's box in p43_59-01.svg ...
    TF_LABEL = (3.06, 84.77, 67.49)
    TF_SHIFT = (-59.44, 0.29)                   # ... and how far left it sits in -02
    TITLE = ("BIDIRECTIONAL DIFFUSION", 4.2, 95.84, 6.89)                   # string, size, center x, baseline
    TITLE_TRAINING = "BIDIRECTIONAL DIFFUSION - TRAINING"
    PROMPT_LINES = (2.2, 96.0, 28.42, 2.64)     # size, center x, first baseline, leading (before the shift)
    PROMPT_SCALE = 0.72
    PROMPT_ARROW = ((95.95, 35.86), (95.95, 40.76))
    OUT_ARROW = ((124.42, 55.41), (129.33, 55.41))
    GRID_X, GRID_Y = (77.20, 112.74, 148.27), (11.00, 31.71, 52.57, 73.54)     # tile left edges / top edges
    GRID_W, GRID_H = 33.28, 19.20
    GRID_FRAMES = [4 * f + 3 for f in range(12)]    # video frame shown per tile: the last of each latent frame's 4
    COUNTER = (37.08, 73.5)                     # step counter under the transformer

    # --- layout: storyboard p43_59-16 ----------------------------------------------------------------------
    TITLE_TEACHER = "TEACHER FORCING - TRAINING"
    TEACHER_FRAMES = (0, 1, 2, 3)               # which of the 12 real frames: three inputs + the last row's target
    ROW_TOPS = (24.3, 47.72, 71.15)             # top edge of each row's images (row 1 = one input frame, on top)
    ROW_W, ROW_H = 30.76, 17.75                 # image size in the rows
    CONTEXT_X = (6.47, 38.22, 69.98)            # left edges of the three input slots (rows fill from the right)
    OUTPUT_X = 146.7
    MODEL_X, MODEL_BOX, MODEL_R = 125.0, 17.3, 1.17
    MODEL_LABEL = (("VIDEO", "MODEL"), 3.6, 4.39)       # lines, size, leading
    MODEL_IN_ARROW = (105.4, 114.3)             # tail x, tip x
    MODEL_OUT_ARROW = (135.5, 143.7)
    MODEL_ARROW = (0.3, (2.99, 1.22, 2.28))     # stroke, head (length, half width, notch depth)
    REAL_BRACKET = (93.61, 95.83, 1.07, 1.6)    # top y, bottom y, corner radius, gap either side of the label
    NOTE_SIZE = 2.8
    REAL_NOTE_Y = 96.69                         # baseline of REAL VIDEO
    GEN_NOTE = (("PREDICTION", "TARGET"), 95.21, 3.1)       # label under the outputs: lines, first baseline, leading

    # --- layout: storyboard p43_59-17 (P54) ----------------------------------------------------------------
    TITLE_ROLLOUT = "TEACHER FORCING + AUTOREGRESSIVE INFERENCE"
    DRIFT_DIR = HERE / "_p54_drift" / "mild"    # drift_0_... (clean) .. drift_3_... : first input + three outputs
    ROLLOUT_MODEL = (131.71, 54.41)             # where the model box ends up (center)
    ROLLOUT_ROW = 2                             # which teacher forcing row's model is kept (2 = bottom)
    ROLLOUT_IMG = (33.28, 19.2)
    ROLLOUT_CONTEXT_X = (4.76, 40.22, 75.68)    # left edges of the three input slots (inputs fill from the right)
    ROLLOUT_OUTPUT_X = 151.56
    ROLLOUT_DIP = 15.0                          # how far a frame drops to pass under the box ...
    ROLLOUT_FLY_SCALE = 0.45                    # ... and how small it gets while it is down there

    # --- layout: storyboard p43_59-18 (P55). Both rows use the rollout row's columns. -----------------------
    TITLE_EXPOSURE = "EXPOSURE BIAS"
    COMPARE_Y = (33.5, 78.0)                    # center line of the training row / the inference row
    COMPARE_FRAMES = (0, 1, 2, 3)               # real frames in the training row: three inputs + the output
    COMPARE_HEADINGS = ("TEACHER FORCING (TRAINING)", "AUTOREGRESSIVE INFERENCE")
    HEADING_SIZE, HEADING_GAP = 3.6, 3.4        # heading baseline sits this far above its row's images
    COMPARE_BRACKETS = ("REAL VIDEO", "MODEL OUTPUTS")
    COMPARE_BRACKET_GAP = 2.5                   # images' bottom edge -> top of the bracket
    COMPARE_NOTE_SIZE = 2.67
    COMPARE_NOTES = (("PREDICTION", "TARGET"), ("GENERATED NEXT", "FRAME"))    # under the training / inference output

    # the LLM version of the two rows (storyboard p43_59-19)
    LLM_ROWS = (("The capital of France is", "Paris"), ("The capital of Germany is", "Berlin"))     # input, output
    LLM_TEXT_SIZE, LLM_TEXT_DROP = 4.37, 1.0    # sentence size; row center line -> text baseline
    LLM_MODEL_LABEL = ("LLM", 4.83)
    LLM_BRACKETS = ("REAL TEXT", "GENERATED TEXT")
    LLM_NOTES = (("PREDICTION", "TARGET"), ("GENERATED NEXT", "TOKEN"))

    def construct(self):
        self.setup_scene()
        self.beat_intro()
        self.wait(self.intro_hold)
        self.beat_denoise(self.tiles, step_times=self.step_times)           # noise -> the generated ball video
        self.wait(self.gen_hold)
        self.beat_to_real()
        self.wait(self.real_hold)
        self.beat_noise_up()
        self.wait(self.noise_hold)
        self.beat_denoise(self.real_tiles, step_time=self.real_step_time, step_times=self.real_step_times)
        self.wait(self.grid_end_hold)
        # teacher forcing (storyboard -16)
        self.beat_teacher_layout()
        self.beat_teacher_models()
        self.wait(self.teacher_noise_hold)
        self.beat_denoise(self.out_tiles, step_time=self.teacher_step_time, step_times=self.teacher_step_times,
                          counter=False)
        self.wait(self.end_hold)
        # P54: autoregressive rollout (storyboard -17)
        self.beat_rollout_layout()
        self.beat_rollout_start()
        self.wait(self.rollout_start_hold)
        n_inputs = len(self.ROLLOUT_CONTEXT_X)
        for k in range(n_inputs):               # 1 input -> 2 -> 3, then one last output
            self.beat_rollout_step()
            if k < n_inputs - 1:
                self.wait(self.rollout_out_hold)
                self.beat_rollout_feed()
                self.wait(self.rollout_feed_hold)
        self.wait(self.p54_end_hold)
        # P55: training vs inference (storyboard -18)
        self.beat_compare_layout()
        self.beat_compare_labels()
        self.wait(self.compare_hold)
        self.beat_exposure_title()
        self.wait(self.exposure_hold)
        self.beat_compare_swap("llm")           # storyboard -19
        self.wait(self.llm_hold)
        self.beat_compare_swap("video")         # ... and back to -18
        self.wait(self.p55_end_hold)

    # ------------------------------------------------------------------------------------------- setup
    def grid_slot(self, k):
        return (self.GRID_X[k % 3], self.GRID_Y[k // 3], self.GRID_W, self.GRID_H)

    def setup_scene(self):
        try:
            self.bg = rgb_to_hex(self.camera.background_rgba[:3])
        except Exception:
            self.bg = BG_FALLBACK
        self.n_steps = len(run_step_dirs()) - 1                 # 40 model calls -> 41 step folders
        d = np.array([self.TF_SHIFT[0] * U, -self.TF_SHIFT[1] * U, 0.0])

        string, size, x, y = self.TITLE
        self.title = u_text(string, size, x, y, bold=True, align="center", color=CHILL_BROWN)
        self.tf = transformer_graphic(self.bg, label=self.show_text, svg=self.TF_SVG, region=self.TF_REGION,
                                      label_spec=self.TF_LABEL, thin_u=0.1, grid_stroke_u=0.07).shift(d)
        size, x, y, leading = self.PROMPT_LINES
        self.prompt = VGroup(*[u_text(line, size, x, y + leading * k, font=TEXT_FONT, italic=True, align="center")
                               for k, line in enumerate(PROMPT)])
        self.prompt.scale(self.PROMPT_SCALE, about_point=self.prompt.get_bottom()).shift(d)
        self.prompt_arrow = arrow(*self.PROMPT_ARROW).shift(d)
        self.out_arrow = arrow(*self.OUT_ARROW).shift(d)

        self.tiles = [RunTile(f, 0).place(*self.grid_slot(k)) for k, f in enumerate(self.GRID_FRAMES)]
        self.real_tiles = []
        self.counter = VGroup()
        self._counter_cache = {}

    def set_counter(self, n=None):
        """STEP n / 40 under the transformer (n = None clears it)."""
        if not (self.show_step_counter and self.show_text):
            return
        if n is None:
            self.counter.set_submobjects([])
            return
        if n not in self._counter_cache:
            x, y = self.COUNTER
            self._counter_cache[n] = u_text(f"STEP {n} / {self.n_steps}", 2.0, x, y, bold=True, align="center",
                                            color=CHILL_BROWN)
        self.counter.set_submobjects([self._counter_cache[n]])
        if self.counter not in self.mobjects:
            self.add(self.counter)

    @staticmethod
    def fade(image, start=0.0, end=1.0, **kwargs):
        """Opacity animation for a tile (keeps the mobject itself, and so its texture, on screen)."""
        return UpdateFromAlphaFunc(image, lambda m, a: m.set_opacity(interpolate(start, end, a)), **kwargs)

    # ------------------------------------------------------------------------------------------- beats
    def beat_intro(self, run_time=None):
        """Storyboard -02 as it was in P44, fading in with the 12 tiles as pure noise."""
        for t in self.tiles:
            t.set_opacity(0)
        self.add(*self.tiles)
        anims = [FadeIn(self.tf), FadeIn(self.prompt_arrow), FadeIn(self.out_arrow),
                 *[self.fade(t) for t in self.tiles]]
        if self.show_text:
            anims += [FadeIn(self.title), FadeIn(self.prompt)]
        self.play(*anims, run_time=run_time or self.intro_time)

    def beat_denoise(self, tiles, run_time=None, step_time=None, step_times=None, counter=True):
        """The diffusion steps run on all the tiles together, from pure noise (step 0) to step 40.
        run_time: total seconds, spread evenly, instead of step_time per step; step_times: one duration per step.
        counter=False leaves the step counter off."""
        n = self.n_steps
        per = run_time / n if run_time else (step_time or self.step_time)
        times = list((None if run_time else step_times) or [per] * n)[:n]
        times += [per] * (n - len(times))
        for t in tiles:
            t.set_step(0)
        if counter:
            self.set_counter(0)
        for k in range(n):
            self.wait(times[k])
            for t in tiles:
                t.set_step(k + 1)
            if counter:
                self.set_counter(k + 1)

    def beat_to_real(self, run_time=None):
        """The generated frames crossfade to 12 frames of real video; the prompt goes; the title gains - TRAINING."""
        paths = tennis_paths()[:len(self.tiles)]
        noise_dir = run_step_dirs()[0]
        self.real_tiles = [
            RealTile(p, noise_dir / f"frame_{self.GRID_FRAMES[k]:03d}.png").place(*self.grid_slot(k)).set_opacity(0)
            for k, p in enumerate(paths)]
        self.add(*self.real_tiles)
        anims = [self.fade(t) for t in self.real_tiles]
        anims += [FadeOut(self.prompt_arrow)]
        if len(self.counter):
            anims.append(FadeOut(self.counter))
        if self.show_text:
            _, size, x, y = self.TITLE
            new_title = u_text(self.TITLE_TRAINING, size, x, y, bold=True, align="center", color=CHILL_BROWN)
            anims += [FadeOut(self.prompt), FadeOut(self.title), FadeIn(new_title)]
        self.play(*anims, run_time=run_time or self.to_real_time)
        self.remove(*self.tiles)
        self.set_counter(None)
        self._counter_cache = {}                # (the old labels were faded out; fresh ones get made)
        if self.show_text:
            self.title = new_title

    def beat_noise_up(self, run_time=None):
        """Noise is added to every real frame together, level 0 -> 1 (ease in and out)."""
        run_time = run_time or self.noise_up_time

        def update(m, alpha):
            for t in self.real_tiles:
                t.set_level(alpha)

        self.play(UpdateFromAlphaFunc(Mobject(), update), run_time=run_time)
        for t in self.real_tiles:
            t.set_level(1.0)

    # ------------------------------------------------------------------------------------------- teacher forcing
    def row_y(self, r):
        """Storyboard y of the middle of row r (0 = top)."""
        return self.ROW_TOPS[r] + self.ROW_H / 2

    def context_slot(self, r, j):
        """Slot of input j (0 = oldest) on row r, which has r + 1 inputs packed against the model's side."""
        return (self.CONTEXT_X[len(self.CONTEXT_X) - (r + 1) + j], self.ROW_TOPS[r], self.ROW_W, self.ROW_H)

    def output_slot(self, r):
        return (self.OUTPUT_X, self.ROW_TOPS[r], self.ROW_W, self.ROW_H)

    def real_tile(self, index, level=0.0):
        """A fresh tile of real frame `index` (0-11), with that frame's own noise."""
        noise = run_step_dirs()[0] / f"frame_{self.GRID_FRAMES[index]:03d}.png"
        return RealTile(tennis_paths()[index], noise, level)

    @staticmethod
    def move_slot(tile, start, end, **kwargs):
        """Animate a tile from one (x, y, w, h) slot to another."""
        a0, a1 = np.array(start, float), np.array(end, float)
        return UpdateFromAlphaFunc(tile, lambda m, a: m.place(*interpolate(a0, a1, a)), **kwargs)

    def bracket(self, x0, x1, spec, string, baseline, size=None):
        """Brown bracket under x0 .. x1 with a label in its middle -> VGroup(left line, right line, label)."""
        y0, y1, r, gap = spec
        mid = (x0 + x1) / 2
        label = u_text(string, size or self.NOTE_SIZE, mid, baseline, bold=True, align="center", color=CHILL_BROWN)
        half = label.get_width() / U / 2 + gap

        def side(x, sign):      # down from the corner of the images, round the corner, in toward the label
            corner = [(x + sign * (r + r * np.cos(a)), y1 - r + r * np.sin(a)) for a in np.linspace(PI, PI / 2, 8)]
            line = VMobject().set_points_as_corners([P(*q) for q in [(x, y0), *corner, (mid - sign * half, y1)]])
            return line.set_stroke(CHILL_BROWN, 0.2 * STROKE_PER_U).set_fill(opacity=0)

        return VGroup(side(x0, 1), side(x1, -1), label)

    def beat_teacher_layout(self, run_time=None):
        """Grid -> three rows of real input frames; everything else from the diffusion view fades."""
        frames, n_rows = self.TEACHER_FRAMES, len(self.ROW_TOPS)
        self.context_tiles = [[] for _ in range(n_rows)]        # per row, oldest input first
        moves = []
        for j in range(n_rows):                                 # input j is used by rows j .. n_rows - 1
            index = frames[j]
            source = self.real_tiles[index]
            for r in range(j, n_rows):
                # the grid tile itself goes to the bottom row; the rows above get fresh copies that start on top of it
                tile = source if r == n_rows - 1 else self.real_tile(index).place(*self.grid_slot(index))
                self.context_tiles[r].append(tile)
                moves.append(self.move_slot(tile, self.grid_slot(index), self.context_slot(r, j)))
        used = [t for row in self.context_tiles for t in row]
        going = [t for t in self.real_tiles if not any(t is u for u in used)]
        self.remove(*used)
        self.add(*used)                                         # movers on top of the tiles that fade
        anims = [*moves, *[self.fade(t, 1.0, 0.0) for t in going], FadeOut(self.tf), FadeOut(self.out_arrow)]
        if len(self.counter):
            anims.append(FadeOut(self.counter))
        if self.show_text:
            _, size, x, y = self.TITLE
            new_title = u_text(self.TITLE_TEACHER, size, x, y, bold=True, align="center", color=CHILL_BROWN)
            anims += [FadeOut(self.title), FadeIn(new_title)]
        self.play(*anims, run_time=run_time or self.teacher_layout_time)
        self.remove(*going)
        self.set_counter(None)
        self._counter_cache = {}
        if self.show_text:
            self.title = new_title

    def beat_teacher_models(self, run_time=None):
        """A VIDEO MODEL on every row, its output starting as pure noise; bracket and label underneath."""
        frames, n_rows = self.TEACHER_FRAMES, len(self.ROW_TOPS)
        stroke, head = self.MODEL_ARROW
        lines, size, leading = self.MODEL_LABEL
        self.models, self.out_tiles = [], []
        anims = []
        for r in range(n_rows):
            y = self.row_y(r)
            box = RoundedRectangle(self.MODEL_BOX * U, self.MODEL_BOX * U, corner_radius=self.MODEL_R * U)
            box.move_to(P(self.MODEL_X, y)).set_stroke(FRESH_TAN, 0.2 * STROKE_PER_U).set_fill(opacity=0)
            base = y + CAP[LABEL_FONT] * size / 2 - leading * (len(lines) - 1) / 2      # block centered on the box
            label = VGroup(*[u_text(s, size, self.MODEL_X, base + k * leading, bold=True, align="center")
                             for k, s in enumerate(lines)])
            arrows = [arrow((xs[0], y), (xs[1], y), stroke, head) for xs in (self.MODEL_IN_ARROW, self.MODEL_OUT_ARROW)]
            out = self.real_tile(frames[r + 1], level=1.0).place(*self.output_slot(r)).set_opacity(0)
            self.add(out)
            self.models.append(VGroup(box, label, *arrows))
            self.out_tiles.append(out)
            row = [ShowCreation(box), *[AnimationGroup(ShowCreation(a[0]), FadeIn(a[1])) for a in arrows],
                   self.fade(out)]
            if self.show_text:
                row.append(Write(label))
            anims.append(AnimationGroup(*row))
        x0, x1 = self.CONTEXT_X[0], self.CONTEXT_X[-1] + self.ROW_W
        self.real_bracket = self.bracket(x0, x1, self.REAL_BRACKET, "REAL VIDEO", self.REAL_NOTE_Y)
        lines, base, leading = self.GEN_NOTE
        self.gen_note = VGroup(*[u_text(s, self.NOTE_SIZE, self.OUTPUT_X + self.ROW_W / 2, base + k * leading,
                                        bold=True, align="center", color=CHILL_BROWN) for k, s in enumerate(lines)])
        left, right, label = self.real_bracket
        notes = [ShowCreation(left), ShowCreation(right)]
        if self.show_text:
            notes += [Write(label), Write(self.gen_note)]
        self.play(LaggedStart(*anims, lag_ratio=0.2), *notes, run_time=run_time or self.teacher_models_time)

    # ------------------------------------------------------------------------------------------- P54: rollout
    @staticmethod
    def pulse(vmob, width_factor=3.0, time_width=0.6, **kwargs):
        """A brighter, thicker flash running along a stroke."""
        hot = vmob.copy().set_fill(opacity=0).set_stroke(FRESH_TAN, float(vmob.get_stroke_width()) * width_factor, 1)
        return VShowPassingFlash(hot, time_width=time_width, **kwargs)

    def rollout_slot_point(self, x):
        """Center of the rollout image slot whose left edge is x."""
        return P(x + self.ROLLOUT_IMG[0] / 2, self.ROLLOUT_MODEL[1])

    def rollout_image(self, k, x):
        """Rollout frame k (0 = the clean first frame) as a still in the slot with left edge x."""
        paths = sorted(Path(self.DRIFT_DIR).glob("drift_*.png"))
        if len(paths) <= k:
            raise FileNotFoundError(f"Need drift_0 .. drift_{k} in {self.DRIFT_DIR}\n  -> run p54_drift.py, or point "
                                    f"DRIFT_DIR at one of its output folders")
        img = ImageMobject(str(paths[k]))
        try:
            img.image.load()
        except Exception:
            pass
        w, h = self.ROLLOUT_IMG
        img.set_width(w * U, stretch=True).set_height(h * U, stretch=True)
        return img.move_to(self.rollout_slot_point(x))

    def beat_rollout_layout(self, run_time=None):
        """One VIDEO MODEL slides to the middle; the rest of the teacher forcing rows fades; new title."""
        keep = self.models[self.ROLLOUT_ROW]
        self.rollout_model = keep                               # VGroup(box, label, in arrow, out arrow)
        target = P(*self.ROLLOUT_MODEL)
        delta = target - keep[0].get_center()
        tiles = [t for row in self.context_tiles for t in row] + list(self.out_tiles)
        left, right, label = self.real_bracket
        anims = [keep.animate.shift(delta),
                 *[self.fade(t, 1.0, 0.0) for t in tiles],
                 *[FadeOut(m) for r, m in enumerate(self.models) if r != self.ROLLOUT_ROW],
                 FadeOut(left), FadeOut(right)]
        if self.show_text:
            _, size, x, y = self.TITLE
            new_title = u_text(self.TITLE_ROLLOUT, size, x, y, bold=True, align="center", color=CHILL_BROWN)
            anims += [FadeOut(label), FadeOut(self.gen_note), FadeOut(self.title), FadeIn(new_title)]
        self.play(*anims, run_time=run_time or self.rollout_layout_time)
        self.remove(*tiles)
        if self.show_text:
            self.title = new_title

    def beat_rollout_start(self, run_time=None):
        """The clean first frame appears as the model's (only) input."""
        self.rollout_context = [self.rollout_image(0, self.ROLLOUT_CONTEXT_X[-1]).set_opacity(0)]
        self.rollout_output = None
        self.add(self.rollout_context[0])
        self.play(self.fade(self.rollout_context[0]), run_time=run_time or self.rollout_start_time)

    def beat_rollout_step(self, run_time=None):
        """The model produces the next frame (each one a little further gone than the last)."""
        k = len(self.rollout_context)
        box, label, in_arrow, out_arrow = self.rollout_model
        self.rollout_output = img = self.rollout_image(k, self.ROLLOUT_OUTPUT_X).set_opacity(0)
        self.add(img)
        self.play(LaggedStart(self.pulse(in_arrow[0]), self.pulse(box, time_width=0.4), self.pulse(out_arrow[0]),
                              self.fade(img), lag_ratio=0.3, run_time=run_time or self.rollout_step_time))

    def beat_rollout_feed(self, run_time=None):
        """The output passes under the box and becomes the newest input; the earlier inputs shift left."""
        img = self.rollout_output
        n = len(self.rollout_context) + 1
        xs = self.ROLLOUT_CONTEXT_X[-n:]                        # where the n inputs sit afterwards
        anims = [m.animate.move_to(self.rollout_slot_point(x)) for m, x in zip(self.rollout_context, xs)]
        self.bring_to_front(img)
        anims.append(fly_under(img, self.rollout_slot_point(xs[-1]), self.ROLLOUT_DIP, self.ROLLOUT_FLY_SCALE))
        self.play(*anims, run_time=run_time or self.rollout_feed_time)
        self.rollout_context.append(img)
        self.rollout_output = None

    # ------------------------------------------------------------------------------------------- P55: side by side
    def video_model(self, x, y):
        """VIDEO MODEL box with its label and two arrows, centered on storyboard (x, y)
        -> VGroup(box, label, in arrow, out arrow), the same pieces as the teacher forcing rows' models."""
        stroke, head = self.MODEL_ARROW
        lines, size, leading = self.MODEL_LABEL
        box = RoundedRectangle(self.MODEL_BOX * U, self.MODEL_BOX * U, corner_radius=self.MODEL_R * U)
        box.move_to(P(x, y)).set_stroke(FRESH_TAN, 0.2 * STROKE_PER_U).set_fill(opacity=0)
        base = y + CAP[LABEL_FONT] * size / 2 - leading * (len(lines) - 1) / 2
        label = VGroup(*[u_text(s, size, x, base + k * leading, bold=True, align="center")
                         for k, s in enumerate(lines)])
        if not self.show_text:
            label = VGroup()
        dx = x - self.MODEL_X
        arrows = [arrow((xs[0] + dx, y), (xs[1] + dx, y), stroke, head)
                  for xs in (self.MODEL_IN_ARROW, self.MODEL_OUT_ARROW)]
        return VGroup(box, label, *arrows)

    def compare_slot(self, x, row):
        """(x, y, w, h) of the image slot with left edge x on compare row `row` (0 = training, 1 = inference)."""
        w, h = self.ROLLOUT_IMG
        return (x, self.COMPARE_Y[row] - h / 2, w, h)

    def beat_compare_layout(self, slide_time=None, fade_time=None):
        """The rollout row slides down; the teacher forcing training row fades in above it."""
        down = (self.ROLLOUT_MODEL[1] - self.COMPARE_Y[1]) * U * UP         # (negative: downwards)
        images = [*self.rollout_context, *([self.rollout_output] if self.rollout_output is not None else [])]
        self.play(self.rollout_model.animate.shift(down), *[m.animate.shift(down) for m in images],
                  run_time=slide_time or self.compare_slide_time)

        frames = self.COMPARE_FRAMES
        xs = [*self.ROLLOUT_CONTEXT_X, self.ROLLOUT_OUTPUT_X]
        self.train_tiles = [self.real_tile(f).place(*self.compare_slot(x, 0)).set_opacity(0)
                            for f, x in zip(frames, xs)]
        self.train_model = self.video_model(self.ROLLOUT_MODEL[0], self.COMPARE_Y[0])
        self.add(*self.train_tiles)
        self.play(*[self.fade(t) for t in self.train_tiles], FadeIn(self.train_model),
                  run_time=fade_time or self.compare_fade_time)

    def compare_row_parts(self, row, kind="video"):
        """The pieces of compare row `row` that differ between the video and the LLM version, freshly made:
        dict(model_label, bracket = VGroup(left line, right line, label), note, strings = (bracket string, note lines))"""
        w, h = self.ROLLOUT_IMG
        x0, x1 = self.ROLLOUT_CONTEXT_X[0], self.ROLLOUT_CONTEXT_X[-1] + w
        y = self.COMPARE_Y[row]
        mx = self.ROLLOUT_MODEL[0]
        if kind == "llm":
            string, size = self.LLM_MODEL_LABEL
            model_label = VGroup(u_text(string, size, mx, y + CAP[LABEL_FONT] * size / 2, bold=True, align="center"))
            bracket_string, note_lines = self.LLM_BRACKETS[row], self.LLM_NOTES[row]
        else:
            lines, size, leading = self.MODEL_LABEL
            base = y + CAP[LABEL_FONT] * size / 2 - leading * (len(lines) - 1) / 2
            model_label = VGroup(*[u_text(s, size, mx, base + k * leading, bold=True, align="center")
                                   for k, s in enumerate(lines)])
            bracket_string, note_lines = self.COMPARE_BRACKETS[row], self.COMPARE_NOTES[row]
        y0 = y + h / 2 + self.COMPARE_BRACKET_GAP
        bracket = self.bracket(x0, x1, (y0, y0 + 2.22, 1.07, 1.6), bracket_string, y0 + 3.1,
                               size=self.COMPARE_NOTE_SIZE)
        leading = self.COMPARE_NOTE_SIZE * 1.2
        note = VGroup(*[u_text(s, self.COMPARE_NOTE_SIZE, self.ROLLOUT_OUTPUT_X + w / 2, y0 + 1.65 + k * leading,
                               bold=True, align="center", color=CHILL_BROWN) for k, s in enumerate(note_lines)])
        return dict(model_label=model_label, bracket=bracket, note=note, strings=(bracket_string, tuple(note_lines)))

    def beat_compare_labels(self, run_time=None):
        """Headings over both rows, REAL VIDEO / MODEL OUTPUTS brackets under the inputs, PREDICTION TARGET /
        GENERATED NEXT FRAME under the outputs."""
        w, h = self.ROLLOUT_IMG
        x0 = self.ROLLOUT_CONTEXT_X[0]
        self.compare_headings = VGroup(*[
            u_text(self.COMPARE_HEADINGS[row], self.HEADING_SIZE, x0, y - h / 2 - self.HEADING_GAP, bold=True,
                   color=CHILL_BROWN) for row, y in enumerate(self.COMPARE_Y)])
        # the current state of each row's swappable pieces (beat_compare_swap replaces them)
        self.compare_rows = [self.compare_row_parts(row, "video") for row in range(len(self.COMPARE_Y))]
        for row, model in enumerate((self.train_model, self.rollout_model)):
            self.compare_rows[row]["model_label"] = model[1]            # the label already on screen
        self.compare_kind = "video"
        self.compare_text = []
        anims = [ShowCreation(side) for parts in self.compare_rows for side in parts["bracket"][:2]]
        if self.show_text:
            anims += [Write(self.compare_headings),
                      *[Write(parts["note"]) for parts in self.compare_rows],
                      *[Write(parts["bracket"][2]) for parts in self.compare_rows]]
        self.play(*anims, run_time=run_time or self.compare_labels_time)

    def beat_compare_swap(self, kind, run_time=None):
        """Everything at once: both rows switch between the video version and the LLM version (kind = "llm" or
        "video"). To LLM: images fade, sentences write in, VIDEO MODEL -> LLM, labels switch. To video: the reverse."""
        if kind == self.compare_kind:
            return
        images = [*self.train_tiles, *self.rollout_context,
                  *([self.rollout_output] if self.rollout_output is not None else [])]
        to_llm = kind == "llm"
        anims = [self.fade(m, 1.0, 0.0) if to_llm else self.fade(m, 0.0, 1.0) for m in images]
        w = self.ROLLOUT_IMG[0]
        in_x = (self.ROLLOUT_CONTEXT_X[0] + self.ROLLOUT_CONTEXT_X[-1] + w) / 2        # middle of the inputs
        out_x = self.ROLLOUT_OUTPUT_X + w / 2
        new_text = []
        for row, old in enumerate(self.compare_rows):
            new = self.compare_row_parts(row, kind)
            # bracket lines reshape (the gap follows the new label); the labels themselves crossfade
            anims += [Transform(old["bracket"][0], new["bracket"][0]), Transform(old["bracket"][1], new["bracket"][1])]
            if self.show_text:
                anims += [FadeOut(old["model_label"]), FadeIn(new["model_label"]),
                          FadeOut(old["bracket"][2]), FadeIn(new["bracket"][2])]
                if old["strings"][1] != new["strings"][1]:
                    anims += [FadeOut(old["note"]), FadeIn(new["note"])]
                else:
                    new["note"] = old["note"]
                if to_llm:
                    base = self.COMPARE_Y[row] + self.LLM_TEXT_DROP
                    for string, x in zip(self.LLM_ROWS[row], (in_x, out_x)):
                        t = u_text(string, self.LLM_TEXT_SIZE, x, base, font=TEXT_FONT, italic=True, align="center")
                        new_text.append(t)
                        anims.append(Write(t))
            else:
                new["model_label"], new["note"] = old["model_label"], old["note"]
            new["bracket"] = VGroup(old["bracket"][0], old["bracket"][1], new["bracket"][2])    # lines stay the same mobjects
            self.compare_rows[row] = new
        anims += [FadeOut(t) for t in self.compare_text]
        default = self.llm_swap_time if to_llm else self.video_swap_time
        self.play(*anims, run_time=run_time or default)
        self.compare_text = new_text
        self.compare_kind = kind

    def beat_exposure_title(self, run_time=None):
        """The title becomes EXPOSURE BIAS."""
        if not self.show_text:
            return
        _, size, x, y = self.TITLE
        new_title = u_text(self.TITLE_EXPOSURE, size, x, y, bold=True, align="center", color=CHILL_BROWN)
        self.play(FadeOut(self.title), Write(new_title), run_time=run_time or self.exposure_title_time)
        self.title = new_title


# ============================================================================= P56-58
class P56_58(InteractiveScene):
    # --- toggles -------------------------------------------------------------------------------------------
    show_text = True            # titles as manim Text. False: add them in post
    left_sync = True            # left side's single pass is stretched so both sides finish together

    # --- timings (seconds) ---------------------------------------------------------------------------------
    intro_time = 2.0            # dividing line draws down, titles write in
    tiles_in_time = 1.5         # the noise frames (and the first real frame + arrow) fade in
    tiles_hold = 1.0
    step_time = 0.1             # per (faked) diffusion step; 40 steps per pass, same pace on both sides
    pass_hold = 0.6             # a freshly denoised output sits there before it moves over
    shuffle_time = 1.0          # output slides over to the inputs, inputs shuffle left
    next_noise_time = 0.5       # the next noise frame fades in on the output side
    next_noise_hold = 0.3
    p56_end_hold = 5.0

    # --- layout: storyboard p43_59-21 (192 x 108 units) ----------------------------------------------------
    DIVIDER = (96.0, 7.49, 103.99)              # x, top y, bottom y
    TITLES = (("BIDIRECTIONAL DIFFUSION", 48.0), ("TEACHER FORCING", 144.0))        # string, center x
    TITLE_SIZE, TITLE_Y = 3.6, 9.9
    FRAMES = (0, 1, 2, 3)                       # which of the 12 real frames
    ROW_Y = 54.3                                # center line of both rows
    LEFT_X = (6.39, 27.21, 48.02, 68.95)        # left edges of the four frames on the left
    LEFT_W, LEFT_H = 20.15, 11.62
    RIGHT_X = (101.91, 121.14, 140.36)          # left edges of the three input slots on the right
    RIGHT_OUT_X = 169.03
    RIGHT_W, RIGHT_H = 18.55, 10.7
    RIGHT_ARROW = (161.35, 166.77)              # tail x, tip x
    N_STEPS = len(NOISE_CURVE) - 1              # 40

    def construct(self):
        self.setup_scene()
        # P56
        self.beat_intro()
        self.beat_tiles_in()
        self.wait(self.tiles_hold)
        self.beat_run()
        self.wait(self.p56_end_hold)

    # ------------------------------------------------------------------------------------------- helpers
    def left_slot(self, k):
        return (self.LEFT_X[k], self.ROW_Y - self.LEFT_H / 2, self.LEFT_W, self.LEFT_H)

    def right_slot(self, x):
        return (x, self.ROW_Y - self.RIGHT_H / 2, self.RIGHT_W, self.RIGHT_H)

    def real_tile(self, index, level=0.0, noise_shift=0):
        """A tile of real frame `index` (0-11). noise_shift picks a different noise frame, so the two sides of the
        screen don't show the very same noise."""
        dirs = run_step_dirs()
        noise = dirs[0] / f"frame_{(4 * index + 3 + noise_shift) % 48:03d}.png"
        return RealTile(tennis_paths()[index], noise, level)

    @staticmethod
    def fade(image, start=0.0, end=1.0, **kwargs):
        return UpdateFromAlphaFunc(image, lambda m, a: m.set_opacity(interpolate(start, end, a)), **kwargs)

    @staticmethod
    def move_slot(tile, start, end, **kwargs):
        a0, a1 = np.array(start, float), np.array(end, float)
        return UpdateFromAlphaFunc(tile, lambda m, a: m.place(*interpolate(a0, a1, a)), **kwargs)

    # ------------------------------------------------------------------------------------------- setup
    def setup_scene(self):
        x, y0, y1 = self.DIVIDER
        self.divider = u_line((x, y0), (x, y1), 0.2, CHILL_BROWN)                  # (drawn top to bottom)
        self.titles = VGroup(*[u_text(s, self.TITLE_SIZE, cx, self.TITLE_Y, bold=True, align="center",
                                      color=CHILL_BROWN) for s, cx in self.TITLES])
        self.left_tiles = [self.real_tile(f, level=1.0).place(*self.left_slot(k)) for k, f in enumerate(self.FRAMES)]
        self.right_arrow = arrow((self.RIGHT_ARROW[0], self.ROW_Y), (self.RIGHT_ARROW[1], self.ROW_Y))
        # right side: frame 0 is given; the output slot starts as frame 1 in pure noise
        self.right_context = [self.real_tile(self.FRAMES[0]).place(*self.right_slot(self.RIGHT_X[-1]))]
        self.right_output = self.new_right_output()

    def new_right_output(self):
        f = self.FRAMES[len(self.right_context)]
        return self.real_tile(f, level=1.0, noise_shift=2).place(*self.right_slot(self.RIGHT_OUT_X))

    # ------------------------------------------------------------------------------------------- P56
    def beat_intro(self, run_time=None):
        """The dividing line draws down from the top while the titles come in."""
        anims = [ShowCreation(self.divider)]
        if self.show_text:
            anims.append(Write(self.titles))
        self.play(*anims, run_time=run_time or self.intro_time)

    def beat_tiles_in(self, run_time=None):
        """Left: four noise frames. Right: the first real frame, the arrow, and a noise frame as the output."""
        tiles = [*self.left_tiles, *self.right_context, self.right_output]
        for t in tiles:
            t.set_opacity(0)
        self.add(*tiles)
        self.play(*[self.fade(t) for t in tiles], ShowCreation(self.right_arrow[0]), FadeIn(self.right_arrow[1]),
                  run_time=run_time or self.tiles_in_time)

    def denoise_pass(self, tiles, step_time=None):
        """All the given tiles step from pure noise to clean together."""
        per = step_time or self.step_time
        for t in tiles:
            t.set_step(0)
        for k in range(self.N_STEPS):
            self.wait(per)
            for t in tiles:
                t.set_step(k + 1)

    def beat_right_shuffle(self, run_time=None):
        """The clean output slides over to become the newest input (inputs shift left); then the next frame fades
        in as noise on the output side."""
        out = self.right_output
        n = len(self.right_context) + 1
        xs = self.RIGHT_X[-n:]
        anims = [self.move_slot(m, self.right_slot(x_old), self.right_slot(x_new))
                 for m, x_old, x_new in zip(self.right_context, self.RIGHT_X[-(n - 1):], xs)]
        anims.append(self.move_slot(out, self.right_slot(self.RIGHT_OUT_X), self.right_slot(xs[-1])))
        self.bring_to_front(out)
        self.play(*anims, run_time=run_time or self.shuffle_time)
        self.right_context.append(out)
        self.right_output = new = self.new_right_output().set_opacity(0)
        self.add(new)
        self.play(self.fade(new), run_time=self.next_noise_time)

    def beat_run(self, step_time=None):
        """Right: one denoising pass per generated frame, with the output shuffled over to the inputs in between.
        Left: one joint pass on all four frames, spread over the whole of that (left_sync) so both sides end
        together, or run alongside the right side's first pass."""
        per = step_time or self.step_time
        n_passes = len(self.RIGHT_X)
        ticker = None
        if self.left_sync:
            total = (n_passes * self.N_STEPS * per
                     + (n_passes - 1) * (self.pass_hold + self.shuffle_time + self.next_noise_time + self.next_noise_hold))
            state = dict(t=0.0, step=0)
            for t in self.left_tiles:
                t.set_step(0)

            def tick(mob, dt):          # runs every frame, through the waits and the shuffles alike
                state["t"] += dt
                step = min(self.N_STEPS, int(self.N_STEPS * state["t"] / total + 1e-6))
                if step != state["step"]:
                    state["step"] = step
                    for t in self.left_tiles:
                        t.set_step(step)

            ticker = Mobject()
            ticker.add_updater(tick)
            self.add(ticker)
        for p in range(n_passes):
            tiles = [self.right_output] + (self.left_tiles if (p == 0 and not self.left_sync) else [])
            self.denoise_pass(tiles, per)
            if p < n_passes - 1:
                self.wait(self.pass_hold)
                self.beat_right_shuffle()
                self.wait(self.next_noise_hold)
        if ticker is not None:
            ticker.clear_updaters()
            self.remove(ticker)
            for t in self.left_tiles:   # land exactly on the last step, whatever the frame rounding did
                t.set_step(self.N_STEPS)
