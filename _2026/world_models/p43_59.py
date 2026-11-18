"""
P43-59 - bidirectional diffusion -> attention patterns -> causal masking -> exposure bias  (ManimGL)

    manimgl <path to this file> P43

One growing file for the whole sequence; scenes get split out when it makes sense. So far:

  P43   storyboard p43_59-01
    beat_intro          title, transformer graphic, and the text prompt + its arrow all draw in together
    beat_io             noisy video on the left (already playing), in / out arrows
    beat_teach_steps    the first few denoising steps, slowly: a pulse runs through the model, the slightly less
                        noisy video appears on the right, then travels around the loop and becomes the next input
                        (the feedback line + plus node draw in during the first trip)
    beat_fast_steps     the remaining steps tick by (left = step s, right = step s + 1) while the ball keeps bouncing
  ... continuing into P44, storyboard p43_59-02
    beat_rearrange      prompt + transformer slide left, the finished video (paused) moves to the right of the model;
                        only the prompt -> transformer and transformer -> video arrows stay
    beat_play_once      the video plays through once, start to end, and stops on its last frame
    beat_split          it breaks apart into 12 frames, 3 across x 4 down (one per latent frame)
    beat_grid_denoise   the 12 frames drop back to noise, then the diffusion steps run on all of them together
  ... storyboard p43_59-03
    beat_icon_highlight one of the little attention-pattern icons inside the transformer turns pink
    beat_clear_left     the rest of the transformer, the prompt and the arrows fade; the frame grid shrinks to the
                        right half of the screen
    beat_icon_to_bracket  the pink icon drops to the bottom of the screen and a line runs out of each side of it,
                        forming a bracket under where the full pattern goes
    beat_pattern        the full attention pattern grows up out of the icon; dimensions (18,720 x 18,720) + label
  ... storyboard p43_59-04: one row / column of the pattern = one spacetime patch of video
    beat_stack          the 12 frames stack into a block seen at an angle (last frame of the video in front)
    beat_rule_patches   the outline of every patch is ruled onto the block (52 x 30 x 12)
    beat_patch_zoom     the block scales up, in place on the right half, onto one spacetime patch (16 x 16 pixels x
                        4 frames); then it fills in: pixel grid, its dimensions and a SPACETIME PATCH title
    beat_pull_back      the labels clear and the block pulls back out while the video dims
    beat_unstack        video back to full brightness, block flattens to the 3 x 4 grid again; every frame keeps
                        its own patch grid (52 x 30) as it goes

  ... the pattern's 12 x 12 block structure
    beat_block_lines    tan lines draw down the pattern, splitting it into 12 columns (one per frame); then across,
                        splitting it into 12 rows
    beat_frames_to_blocks  each frame's patch grid lifts off with a pale pink panel under it and flies to that
                        frame's block on the top row of the pattern, squashing into it (the lines crowd together and
                        become the panel)
  ... storyboard p43_59-05: one block of the pattern = one pair of frames
    beat_block_highlight  a yellow border draws around the block in row 2, column 9 (the top-row panels fade)
    beat_frame_link     ("row") frame 2 gets a yellow border and its number; a dashed arrow runs from it to the block's
                        side. ("col") the same for frame 9, its arrow coming down into the block from above.
  P45   storyboards p43_59-06, -07: one row of the pattern = one patch looking at every other patch
    beat_clear_block_links  the P44 highlights go (block border, frame borders / numbers / arrows, the 12 x 12 lines,
                        and the left-hand dimension, which is where the row's number goes)
    beat_row_highlight  a yellow line draws across the pattern at the query token's row, with its number (15,322)
    beat_patch_link     a yellow square marks that token's spacetime patch on its frame; a dashed two-headed
                        connector joins the row to the patch
    beat_drop_query     the heatmap, patch square, connector and row line all fade
    beat_swap_pattern   the attention pattern (and its LAYER / HEAD label) crossfades to another head
    beat_query_with_heat  for that head's query token: row line + number, then the patch square and connector draw
                        while its heatmap fades in over the frames (no flying dots this time)
  P49   storyboards p43_59-08, -09: which blocks a causal model is allowed to use
    beat_causal_mask    back on the first pattern, the blocks above the diagonal (a frame looking at a later frame) dim
    beat_block_pair     one block and its two frames in one color: border round the block, borders + numbers on the
                        frames (the rest dim), a dashed arrow from each frame to the block (row frame into its side,
                        column frame into its top), and a heavy arrow from the frame being looked at to the frame doing
                        the looking, with a label. (2, 9, RED, "NON CAUSAL") then (9, 2, GREEN, "CAUSAL")
    beat_crossfade_block_pair  one quick crossfade from the current highlight to another (red -> green)
    beat_clear_block_pair  that highlight fades and the frames come back; the dimmed blocks stay dimmed
  P47
    beat_title_highlight  after the query is dropped and a pause: the letters of BIDIRECTIONAL in the title pop and
                        turn yellow one after another, hold, then return to the title color
    beat_row_to_heat    all 18,720 values in the row leave it as colored dots, in order, and fly to their own patch
                        on their own frame; when they land they are the heatmap, which then sits over the frames

The block is drawn with a flat (orthographic) projection computed here, not a 3D camera: nothing else on screen
tilts, and "zooming" is the block scaling about a fixed point behind two background-colored masks (left half +
title strip), so the attention pattern stays put.

Artwork comes from the Illustrator storyboards in p43_59_sketches_1/ (192 x 108 viewBox = one unit per 10 px at
1080p, mapped onto manim's 8-unit-tall frame). The transformer graphic is read straight out of p43_59-01.svg at
runtime (it's ~550 little pieces); everything else is redrawn from the storyboard's coordinates.

Video frames come from the Wan2.2 notebook p43_59_3.ipynb:  <RUN>/p43/xt/step_SS/frame_FFF.png  (41 step folders
x 48 frames; 12 latent frames x 4 video frames each). Step and frame counts are read off the folder, so the older
Wan2.1 output (p43_59_2: 31 x 45) still runs. Set WM_P43_59 to point somewhere else (e.g. on the Linux box).

Timings and toggles are class attributes on the scene classes.
"""
import json
import os
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from manimlib import *

try:
    import moderngl
except Exception:      # only needed for hard-edged pixels in the patch zoom
    moderngl = None

# ----------------------------------------------------------------------------- config
HERE = Path(__file__).resolve().parent
SKETCHES = HERE / "p43_59_sketches_1"
RUN = Path(os.environ.get(
    "WM_P43_59",
    "/Users/stephen/Library/CloudStorage/Dropbox-Stephencwelch/welch_labs/world_models/hackin/Wan2.2/"
    "attn_runs/ball_drop_s252/p43_59_3"))

FRESH_TAN = "#f4ebd9"       # line art, prompt, TRANSFORMER
RED = "#d73b2f"             # NON CAUSAL highlights (storyboard -08)
GREEN = "#419c52"           # CAUSAL highlights (storyboard -09)
YELLOW = "#fcc947"          # block / frame highlights and their dashed arrows (storyboard -05)
FRAME_NUM = "#063336"       # the big frame numbers set over the highlighted frames
PINK = "#ed5e78"            # the highlighted attention-pattern icon + its bracket
CHILL_BROWN = "#948678"     # title, feedback loop
BG_FALLBACK = "#031e1e"     # node fill = scene background (read from the camera at runtime when possible)
SVG_BG = "#031e1e"          # the background color used inside the storyboards (swapped for the scene's)
LABEL_FONT = "Open Sans"
TEXT_FONT = "STIX Two Text"
DIM_FONT = "Latin Modern Roman"                 # the 18,720 dimension labels
CAP = {LABEL_FONT: 0.714, TEXT_FONT: 0.656, DIM_FONT: 0.683}    # cap height / em, to match Illustrator's point sizes

PROMPT = [
    "“A single red rubber ball is dropped from the top of the frame onto a wooden",
    "floor and bounces several times, each bounce lower than the last, until it comes",
    "to rest. Static camera, soft daylight, plain white wall behind, realistic physics.”",
]

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
    """Vector shapes of a storyboard SVG (line, rect, circle, polygon, polyline, path) as styled VMobjects, in
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
TF_SVG = SKETCHES / "p43_59-01.svg"
TF_REGION = (69.0, 43.0, 124.0, 67.6)       # the transformer's box in p43_59-01.svg
TF_CENTER = (96.52, 54.93)                  # center of its border there


def transformer_graphic(bg=BG_FALLBACK, label=True):
    """The TRANSFORMER box from storyboard -01, at its storyboard position.

    -> VGroup with named layers (also in this order, back to front):
       .border  .columns  .cells  .grid  .wires  .nodes  .dots  .label
    Move / scale it as one piece (use scale_with_strokes for a "camera zoom" look).
    """
    layers = {k: VGroup() for k in ("border", "columns", "cells", "grid", "wires", "nodes", "dots")}
    for m in read_svg_shapes(TF_SVG, TF_REGION, bg):
        s = m.svg
        thin = s["stroke"] != "none" and s["stroke_u"] < 0.1
        if m.svg_tag == "rect":
            key = "columns" if s["rounded"] else "cells"
        elif m.svg_tag == "circle":
            key = "nodes"
        elif m.svg_tag in ("line", "polyline"):
            key = "grid" if thin else "wires"
        elif m.svg_tag == "path":
            key = "border" if s["fill"] == "none" else ("nodes" if s["fill"].lower() == SVG_BG else "dots")
        else:
            continue
        if key == "grid":       # one of the nine little grids is a different color in the mockup; make them match
            m.set_stroke(FRESH_TAN, 0.07 * STROKE_PER_U, 1)
        layers[key].add(m)
    g = VGroup(*layers.values())
    for z, (k, v) in enumerate(layers.items()):
        setattr(g, k, v)
        for m in v:     # manimgl draws a batch's fills, then its strokes; a per-layer tag keeps the layers in separate
            m.uniforms["z_layer"] = float(z)    # batches so e.g. the node fills really do cover the wires behind them
    g.label = VGroup()
    if label:
        g.label = u_text("TRANSFORMER", 3.06, 84.77, 67.49, bold=True)
        g.add(g.label)
    return g


def scale_with_strokes(mob, factor, about_point=None):
    """Scale a VMobject and its stroke widths together (what a camera zoom would show)."""
    mob.scale(factor, about_point=about_point)
    for sm in mob.family_members_with_points():
        sm.set_stroke(width=float(sm.get_stroke_width()) * factor)
    return mob


# ----------------------------------------------------------------------------- video frames
def step_dirs(kind="xt"):
    """Sorted step folders: step_00 ... step_29, step_30_final."""
    root = RUN / "p43" / kind
    dirs = sorted(d for d in root.glob("step_*") if d.is_dir())
    if not dirs:
        raise FileNotFoundError(
            f"No step folders in {root}\n  -> point WM_P43_59 at the notebook's p43_59_3 output folder")
    return dirs


@lru_cache(maxsize=160)
def _rgba(path):
    with Image.open(path) as im:
        im = im.convert("RGBA")
        return im.size, im.tobytes()


class Clock:
    """Shared playback time for every StepVideo, so they all show the same video frame."""

    def __init__(self, fps, n_frames):
        self.t, self.fps, self.n = 0.0, fps, n_frames
        self.fixed = None           # set to a frame number to stop looping and hold / scrub

    def frame(self):
        return int(self.fixed) if self.fixed is not None else int(self.t * self.fps) % self.n


class StepVideo(ImageMobject):
    """One quad showing <RUN>/p43/<kind>/step_SS/frame_FFF.png. Its texture is rewritten in place, so a whole
    video (at any diffusion step) costs one mobject and one texture instead of one ImageMobject per frame.

        v = StepVideo(clock, step=0)        v.set_step(12)        v.sync()   # scene calls sync() every frame
        StepVideo(clock, step=40, frame=47)                                  # a still: always video frame 47
    """
    _count = 0

    def __init__(self, clock, step=0, kind="xt", height=2.0, frame=None):
        self.clock, self.step, self.kind, self.frame = clock, step, kind, frame
        self.pixelated = False
        self.dirs = step_dirs(kind)
        self._shown = None
        super().__init__(str(self.frame_path()), height=height)
        try:
            self.image.load()               # don't hold the file handle open
        except Exception:
            pass
        StepVideo._count += 1
        self.uniforms["video_slot"] = float(StepVideo._count)   # keeps two videos from being batched onto one texture

    def frame_path(self):
        frame = self.clock.frame() if self.frame is None else self.frame
        return self.dirs[int(np.clip(self.step, 0, len(self.dirs) - 1))] / f"frame_{frame:03d}.png"

    def set_step(self, step):
        self.step = int(step)
        return self

    def init_shader_wrapper(self, ctx):
        super().init_shader_wrapper(ctx)
        path = self.frame_path()
        size, data = _rgba(path)
        # a private texture (the stock one is cached per file path and shared between mobjects)
        self.shader_wrapper.textures[0] = ctx.texture(size=size, components=4, data=data)
        self._shown = path
        self.set_pixelated(self.pixelated)

    def set_pixelated(self, on=True):
        """Hard-edged pixels when the image is blown up (instead of smooth interpolation)."""
        self.pixelated = on
        if moderngl is not None and self.shader_wrapper is not None and self.shader_wrapper.textures:
            self.shader_wrapper.textures[0].filter = (moderngl.LINEAR, moderngl.NEAREST if on else moderngl.LINEAR)
        return self

    def set_quad(self, ul, dl, ur, dr):
        """Put the image's four corners anywhere (used for the tilted block)."""
        self.set_points(np.array([ul, dl, ur, dr, ur, dl]))
        return self

    def sync(self):
        path = self.frame_path()
        if path == self._shown or self.shader_wrapper is None or not self.shader_wrapper.textures:
            return self
        self.shader_wrapper.textures[0].write(_rgba(path)[1])
        self._shown = path
        return self

    def place(self, x, y, w, h):
        """Fill the storyboard slot with top-left (x, y) and size (w, h)."""
        return place_rect(self, x, y, w, h)


def place_rect(mob, x, y, w, h):
    """Stretch a mobject to fill the storyboard rectangle with top-left (x, y) and size (w, h)."""
    mob.set_width(w * U, stretch=True).set_height(h * U, stretch=True)
    mob.move_to(P(x + w / 2, y + h / 2))
    return mob


def polyline_point(points, alpha):
    """Point a fraction alpha of the way along a polyline, by arc length."""
    pts = np.array(points, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    d = np.clip(alpha, 0, 1) * cum[-1]
    k = int(np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(seg) - 1))
    return pts[k] + (pts[k + 1] - pts[k]) * ((d - cum[k]) / seg[k] if seg[k] > 0 else 0)


# ============================================================================= P43
class P43(InteractiveScene):
    # --- toggles -------------------------------------------------------------------------------------------
    show_text = True            # title / prompt / TRANSFORMER as manim Text. False: add them in post
    show_step_counter = True    # small "STEP 12 / 40" under the plus node (not in the storyboard)
    video_fps = 16              # playback rate of the 48-frame ball clip (Wan2.2's native rate)

    # --- timings (seconds) ---------------------------------------------------------------------------------
    intro_time = 6.0            # title + transformer + prompt, all at once
    io_time = 1.5
    pre_steps_hold = 1.0
    teach_steps = 2             # how many steps get the slow treatment (pulse through the model + trip round the loop)
    forward_time = 1.4          # pulse through the model, output fades in
    output_hold = 0.6           # output sits on the right before it heads round the loop
    first_feedback_time = 2.5   # the first trip: the feedback line + plus node draw in as the output travels
    feedback_time = 1.8         # output travels round the loop to become the next input
    fast_step_time = 0.3        # per step once they just tick by
    fast_step_times = None      # or a list of per-step seconds for the fast steps (e.g. linger on the late ones)
    final_feedback = True       # after the last step, send the finished video round the loop so it ends on the left
    end_hold = 6.0              # finished video, still looping (end of P43)

    # P44
    rearrange_time = 2.5        # prompt + transformer slide left, video grows on the right
    play_once_pre_hold = 0.5    # paused on the first frame before the single play-through
    play_once_post_hold = 1.0   # ... and on the last frame after it
    split_time = 2.5            # video breaks apart into the 12-frame grid
    split_hold = 2.0
    renoise_time = 1.0          # the 12 clean frames run back to noise (steps played in reverse)
    renoise_hold = 1.0
    grid_step_time = 0.3        # per diffusion step on the grid
    grid_step_times = None      # or a list of per-step seconds (one per step: 40)
    p44_end_hold = 5.0          # finished grid, before the attention pattern section

    # P44, attention pattern (-03)
    icon_color_time = 1.0       # one attention icon in the transformer turns pink
    icon_hold = 0.5
    clear_left_time = 2.0       # rest of the transformer fades, frame grid shrinks to the right half
    icon_move_time = 1.5        # icon drops to the bottom of the screen
    bracket_time = 1.0          # lines run out of each side of it
    pattern_time = 2.0          # full pattern grows out of the icon
    dims_time = 1.5             # dimension lines + numbers + label
    pattern_hold = 5.0

    # P44, spacetime patch (-04)
    stack_time = 2.5            # grid -> tilted block
    stack_hold = 1.0
    rule_patches_time = 2.0     # every patch's outline is ruled onto the block
    rule_hold = 1.0
    patch_zoom_time = 3.0       # block scales up onto the patch
    patch_box_time = 1.5        # the patch fills in
    patch_labels_time = 1.5     # pixel grid, dimensions, title
    patch_hold = 4.0
    pull_back_time = 4.0        # labels clear, block pulls back out, video dims
    tiled_hold = 4.0
    unstack_time = 3.0          # block flattens back to the grid, each frame keeping its patch grid
    unstack_hold = 4.0

    # P44, block structure of the pattern
    col_lines_time = 2.0        # 11 lines draw down, splitting the pattern into 12 columns
    row_lines_time = 2.0        # 11 lines draw across, splitting it into 12 rows
    block_lines_hold = 1.5
    frames_to_blocks_time = 5.0  # all 12 panels (with the lag below), frame -> top-row block
    frames_to_blocks_hold = 5.0

    # P44, one block <-> two frames (-05)
    block_hl_time = 1.0         # yellow border around the block (panels on the top row fade with it)
    block_hl_hold = 1.0
    frame_hl_time = 1.0         # yellow border + number on a frame
    link_arrow_time = 2.0       # its dashed arrow to the block
    link_hold = 1.5             # after each frame's link
    p44_final_hold = 5.0

    # P45, one row -> heatmap over the frames (-06, -07)
    clear_links_time = 1.0      # P44 highlights fade
    row_hl_time = 1.5           # yellow row line + its number
    row_hl_hold = 1.0
    patch_hl_time = 1.0         # yellow square on the query patch
    patch_link_time = 2.0       # dashed connector, row <-> patch
    patch_link_hold = 2.0
    row_to_heat_time = 7.0      # the row's values fly to the frames
    heat_settle_time = 1.0      # dots hand over to the heatmap images
    heat_hold = 6.0

    # P45, second head: swap the pattern, then row + patch + connector with the heatmap fading in
    drop_query_time = 1.0       # heatmap, patch square, connector, row line + number fade
    swap_pattern_time = 1.5     # pattern (and its label) crossfade to the next head
    swap_pattern_hold = 1.5
    query_heat_time = 3.0       # patch square + connector draw while the heatmap fades in
    query_heat_hold = 6.0

    # P47: clear the query, pause, flash BIDIRECTIONAL in the title
    p47_pause = 2.0             # clean pattern + frames before the title lights up
    title_hl_time = 1.2         # letters pop and turn yellow, left to right
    title_hl_hold = 2.5         # ... stay yellow this long
    title_restore_time = 0.8    # ... and go back to the title color
    p47_end_hold = 4.0

    # P49: causal masking (-08, -09)
    causal_mask_time = 1.5      # the blocks above the diagonal dim
    causal_mask_hold = 2.0
    pair_block_time = 1.0       # border round the block
    pair_frames_time = 1.2      # borders + numbers on its two frames, the other frames dim
    pair_links_time = 2.5       # both dashed arrows, frames -> block
    pair_arrow_time = 1.2       # the heavy arrow between the frames + its label
    block_pair_hold = 4.0
    pair_crossfade_time = 0.6   # NON CAUSAL highlight -> CAUSAL highlight
    clear_pair_time = 1.0
    p49_end_hold = 5.0

    # --- storyboard -01 geometry (192 x 108 units) ---------------------------------------------------------
    SLOT_L = (11.99, 41.03, 49.92, 28.80)       # x, y, w, h of the input video
    SLOT_R = (131.03, 41.03, 49.92, 28.80)      # ... and the output video
    LOOP_START, LOOP_END = (156.86, 71.25), (35.38, 70.97)
    LOOP_Y, LOOP_R = 78.41, 1.44                # the loop's horizontal run and corner radius
    PLUS = (96.0, 78.69, 1.22)                  # center x, y and radius of the plus node
    TRAVEL_SCALE = 0.22                         # size of the video while it rides the loop

    # --- storyboard -02 geometry ---------------------------------------------------------------------------
    TF_SHIFT = (-59.44, 0.29)                   # how far the transformer moves from -01 to -02
    TITLE_SHIFT = (-0.17, -2.97)
    PROMPT_SCALE = 0.72                         # the prompt shrinks to the transformer's width so it fits on the left
    GRID_X, GRID_Y = (77.20, 112.74, 148.27), (11.00, 31.71, 52.57, 73.54)     # tile left edges / top edges
    GRID_W, GRID_H = 33.28, 19.20
    GRID_FRAMES = [4 * f + 3 for f in range(12)]           # video frame shown per tile: the last of each latent frame's 4
    BIG = (77.20, 25.61, 104.35, 60.20)         # the whole video before it splits: grid width, centered on the out arrow
    COUNTER_44 = (37.08, 73.5)                  # step counter under the transformer

    # --- storyboard -03 geometry ---------------------------------------------------------------------------
    ATTN_NAME = "step_05_layer_09_head_26_mean_f20_log"     # <RUN>/p44/attn/<ATTN_NAME>.png
    ATTN = (14.94, 19.33, 74.88, 74.88)         # x, y, w, h of the full attention pattern
    DIM_TOP_Y, DIM_LEFT_X = 15.97, 10.84        # where the two dimension lines run
    ICON = (1, 0)                               # which icon in the transformer: (column 0-2, row 0-2 from the top)
    ICON_SCALE = 1.0                            # how much it grows on its way down (1 = stays the same size)
    BRACKET_Y, BRACKET_TOP = 97.8, 95.6         # the bracket's horizontal run, and how far up its end ticks reach
    BRACKET_R = 0.8                             # its corner radius
    ATTN_LABEL_Y = 103.8                        # baseline of "LAYER 9 HEAD 26", centered under the icon
    GRID3_X, GRID3_Y = (100.49, 129.04, 157.56), (22.19, 38.83, 55.55, 72.42)      # the half-screen frame grid
    GRID3_W, GRID3_H = 24.96, 14.40
    TITLE_SHIFT_3 = (0.0, -0.73)

    # --- spacetime patch (-04) -----------------------------------------------------------------------------
    BLOCK_BOX = (101.0, 17.0, 86.0, 80.0)       # x, y, w, h of the area the tilted block is fit into
    BLOCK_ANGLES = (30.0, 22.0)                 # degrees: turn about the vertical axis, tilt toward the viewer
    BLOCK_SPACING = 0.03                        # gap between frames in the stack, as a fraction of a frame's width
    BLOCK_LAST_IN_FRONT = True                  # the last frame of the video is the front sheet (False: the first)
    PATCH_TOKEN = (11, 0, 0)                    # (frame 0-11, row 0-29, col 0-51) of the patch to zoom on; this is the
                                                # upper-left corner of the front sheet
    PATCH_PX, PATCH_FRAMES = 16, 4              # a token covers 16 x 16 pixels x 4 video frames
    PATCH_VIEW = (150.0, 62.0)                  # where the patch's front face sits on screen when zoomed in
    PATCH_ZOOM = 18.0                           # how much the block scales up
    PATCH_COLOR = PINK                          # patch outlines, the highlighted patch, the SPACETIME PATCH title
    RULE_WIDTHS = (0.08, 0.25)                  # patch outline width (storyboard units) pulled back / zoomed in
    QUERY_NAME = "step_05_layer_09_head_26_any_07_f09_tok15322"     # <RUN>/p45_48/data/<QUERY_NAME>.npz
    MASK_OPACITY = 0.75                         # how much the non-causal blocks are covered (0.75 -> they show at 25%)
    PAIR_DIM = 0.5                              # opacity of the frames that aren't part of the highlighted pair
    LINK_BOTTOM_Y = 99.0                        # where a row frame's arrow runs when it has to go round underneath
    LABEL_PLATE = 0.85                          # opacity of the dark plate behind NON CAUSAL / CAUSAL (0 = none)
    PAIR_BORDER = 0.8                           # border width round the highlighted block and its two frames
    PAIR_STROKE = 0.4                           # the heavy arrow between the two frames
    TITLE_WORD = 13                             # how many letters of the title light up (BIDIRECTIONAL)
    TITLE_POP = 0.18                            # how much each letter swells as it lights up
    ATTN_NAME_2 = "step_29_layer_03_head_02_max_f10_log"            # the second head's pattern ...
    QUERY_NAME_2 = "step_29_layer_03_head_02_ball_f02_tok03564"     # ... and the query token shown on it
    ATTN_NAME_3 = "step_05_layer_00_head_38_max_f10_log"            # the third head's pattern ...
    QUERY_NAME_3 = "step_05_layer_00_head_38_ball_02_f04_tok06997"  # ... and its query token
    ROW_STROKE = 0.3                            # yellow row line / patch square / connector width
    PATCH_BOX_STROKE = 0.65                      # ... and the weight of the yellow square round the query patch
    ROW_LINK_X = 95.0                           # the connector drops down between the pattern and the frames here
    PATCH_BOX = 1.5                             # side of the yellow square drawn round the query patch (the patch is 0.48)
    HEAT_CMAP, HEAT_ALPHA = "magma", 0.5        # the heatmap's colormap, and its opacity over the frames
    HEAT_SCALE = "log"                          # "log": color by log10(weight) over the top HEAT_DECADES decades below the
    HEAT_DECADES = 6.0                          #        row's max (like the pattern image, which is log-scaled too)
    HEAT_GAMMA = 0.5                            # "gamma": color by (w / max) ** HEAT_GAMMA (the notebook's overlays; mostly black)
    HEAT_LAG = 0.65                             # how spread out the departures are (0 = all at once, -> 1 = strictly one by one)
    DOT_MIN_ALPHA = 0.9                        # opacity of the dots carrying the smallest weights (the largest are solid)
    DOT_RADII = (0.14, 0.24)                    # dot radius leaving the row / landing on its patch (0.24 = half a patch)
    DOT_FADE_IN = 0.0                           # fraction of its trip over which a dot fades in (0 = solid the moment it leaves)
    ROW_MARKER = 0.5                            # radius of the bright marker that runs along the row showing where values
                                                # are leaving from right now (0 = no marker)
    HL_ROW_FRAME, HL_COL_FRAME = 2, 9           # (1-based) the block in row 2, column 9 <-> frames 2 and 9
    HL_STROKE = 0.9                             # yellow border width (storyboard units)
    LINK_DASH = (0.6, 0.6)                      # dash, gap of the connecting arrows
    LINK_R = 1.45                               # their corner radius
    LINK_TOP_Y, LINK_SIDE_X = 12.6, 187.6       # where the column frame's arrow runs: above the pattern, right of the frames
    LINK_MID_X, LINK_OVER_Y = 95.4, 18.3        # where the row frame's arrow runs: between pattern and frames, above the frames
    BLOCK_LINE = (0.15, 0.9)                    # width and opacity of the tan lines splitting the pattern into 12 x 12
    PANEL_OPACITY = 0.35                        # fill of the pink panels that go from the frames to the pattern
    PANEL_LAG = 0.15                            # lag ratio between one frame's panel leaving and the next
    PANEL_ROW = 0                               # which row of the pattern the panels land on (0 = top)
    FLAT_RULE = (0.05, 0.8)                     # width and opacity of the patch grid on each frame once they're flat again
    DIM_OPACITY = 0.3                           # video opacity while the patches are ruled in
    MASK_X, MASK_Y = 95.5, 11.5                 # the zoomed block is hidden left of / above these

    def construct(self):
        self.setup_scene()
        self.beat_intro()
        self.beat_io()
        self.wait(self.pre_steps_hold)
        self.beat_teach_steps()
        self.beat_fast_steps()
        self.wait(self.end_hold)
        # P44
        self.beat_rearrange()
        self.beat_play_once()
        self.beat_split()
        self.wait(self.split_hold)
        self.beat_grid_denoise()
        self.wait(self.p44_end_hold)
        # P44, attention pattern
        self.beat_icon_highlight()
        self.wait(self.icon_hold)
        self.beat_clear_left()
        self.beat_icon_to_bracket()
        self.beat_pattern()
        self.wait(self.pattern_hold)
        # P44, spacetime patch
        self.beat_stack()
        self.wait(self.stack_hold)
        self.beat_rule_patches()
        self.wait(self.rule_hold)
        self.beat_patch_zoom()
        self.wait(self.patch_hold)
        self.beat_pull_back()
        self.wait(self.tiled_hold)
        self.beat_unstack()
        self.wait(self.unstack_hold)
        # P44, block structure of the pattern
        self.beat_block_lines()
        self.wait(self.block_lines_hold)
        self.beat_frames_to_blocks()
        self.wait(self.frames_to_blocks_hold)
        # P44, one block <-> two frames
        self.beat_block_highlight()
        self.wait(self.block_hl_hold)
        self.beat_frame_link("row")
        self.wait(self.link_hold)
        self.beat_frame_link("col")
        self.wait(self.p44_final_hold)
        # P45
        self.beat_clear_block_links()
        self.beat_row_highlight()
        self.wait(self.row_hl_hold)
        self.beat_patch_link()
        self.wait(self.patch_link_hold)
        self.beat_row_to_heat()
        self.wait(self.heat_hold)
        # P45, second head
        self.beat_drop_query()
        self.beat_swap_pattern(self.ATTN_NAME_2)
        self.wait(self.swap_pattern_hold)
        self.beat_query_with_heat(self.QUERY_NAME_2, heat_alpha=0.7)
        self.wait(self.query_heat_hold)
        # P45, third head
        self.beat_drop_query()
        self.beat_swap_pattern(self.ATTN_NAME_3)
        self.wait(self.swap_pattern_hold)
        self.beat_query_with_heat(self.QUERY_NAME_3, heat_alpha=0.6, scale="linear")
        self.wait(self.query_heat_hold)
        # P47
        self.beat_drop_query()
        self.wait(self.p47_pause)
        self.beat_title_highlight()
        self.wait(self.p47_end_hold)
        # P49
        self.beat_swap_pattern(self.ATTN_NAME)
        self.wait(self.swap_pattern_hold)
        self.beat_causal_mask()
        self.wait(self.causal_mask_hold)
        self.beat_block_pair(2, 9, RED, "NON CAUSAL")
        self.wait(self.block_pair_hold)
        self.beat_crossfade_block_pair(9, 2, GREEN, "CAUSAL")
        self.wait(self.block_pair_hold)
        self.beat_clear_block_pair()
        self.wait(self.p49_end_hold)

    # ------------------------------------------------------------------------------------------- pieces
    def setup_scene(self):
        try:
            self.bg = rgb_to_hex(self.camera.background_rgba[:3])
        except Exception:
            self.bg = BG_FALLBACK

        # everything that plays video hangs off one clock, advanced by an invisible mobject's updater
        self.n_steps = len(step_dirs("xt")) - 1                         # 40 model calls -> 41 step folders
        n_frames = len(list(step_dirs("xt")[0].glob("frame_*.png")))    # 48
        self.clock = Clock(self.video_fps, n_frames)
        self.videos, self.on_tick = [], []
        self.ticker = Mobject()
        self.ticker.add_updater(self._tick)
        self.add(self.ticker)

        self.title = u_text("BIDIRECTIONAL DIFFUSION", 4.2, 68.07, 9.86, bold=True, color=CHILL_BROWN)
        self.tf = transformer_graphic(self.bg, label=self.show_text)
        self.prompt = VGroup(*[
            u_text(s, 2.2, 96.0, 28.42 + 2.64 * k, font=TEXT_FONT, italic=True, align="center")
            for k, s in enumerate(PROMPT)])
        self.prompt_arrow = arrow((95.95, 35.86), (95.95, 40.76))
        self.in_arrow = arrow((63.28, 55.05), (68.18, 55.05))
        self.out_arrow = arrow((124.42, 55.41), (129.33, 55.41))

        # feedback loop: down from under the output, along the bottom, back up to the input
        (sx, sy), (ex, ey), ly, r = self.LOOP_START, self.LOOP_END, self.LOOP_Y, self.LOOP_R
        # (built from many equal-length pieces: manim draws partial paths by piece count, so this makes
        #  ShowCreation / passing flashes move along it at an even speed)
        t = np.linspace(0, PI / 2, 12)
        corner_r = [(sx - r + r * np.cos(a), ly - r + r * np.sin(a)) for a in t]
        corner_l = [(ex + r + r * np.cos(a), ly - r + r * np.sin(a)) for a in t + PI / 2]
        outline = np.array([(sx, sy), *corner_r, *corner_l, (ex, ey + 1.52)])
        n_pieces = int(np.linalg.norm(np.diff(outline, axis=0), axis=1).sum() / 0.25)
        loop = VMobject().set_points_as_corners(
            [P(*polyline_point(outline, k / n_pieces)) for k in range(n_pieces + 1)])
        loop.set_stroke(CHILL_BROWN, 0.2 * STROKE_PER_U).set_fill(opacity=0)
        barb = Polygon(P(ex, ey), P(ex + 0.815, ey + 1.99), P(ex, ey + 1.52), P(ex - 0.815, ey + 1.99))
        barb.set_fill(CHILL_BROWN, 1).set_stroke(CHILL_BROWN, 0)
        self.loop = VGroup(loop, barb)
        px, py, pr = self.PLUS
        self.plus = VGroup(
            Circle(radius=pr * U).move_to(P(px, py)).set_fill(self.bg, 1).set_stroke(CHILL_BROWN, 0.2 * STROKE_PER_U),
            u_line((px - 0.62, py), (px + 0.62, py), 0.26, CHILL_BROWN),
            u_line((px, py - 0.62), (px, py + 0.62), 0.26, CHILL_BROWN))

        self.left = self.new_video(0).place(*self.SLOT_L)
        self.right = self.new_video(1).place(*self.SLOT_R)
        self.counter = VGroup()
        self._counter_cache = {}

    def new_video(self, step, kind="xt", frame=None):
        v = StepVideo(self.clock, step, kind, frame=frame)
        self.videos.append(v)
        return v

    def _tick(self, mob, dt):
        self.clock.t += dt
        for f in list(self.on_tick):
            f()
        for v in self.videos:
            v.sync()

    def settle(self, group):
        """After animating a group's pieces one by one, put the group itself back on screen as a single mobject."""
        self.remove(*group.get_family())
        self.add(group)

    def set_counter(self, n):
        if not (self.show_step_counter and self.show_text):
            return
        x, y = getattr(self, "counter_pos", (self.PLUS[0], 85.2))
        if (n, x, y) not in self._counter_cache:
            self._counter_cache[(n, x, y)] = u_text(f"STEP {n} / {self.n_steps}", 2.0, x, y, bold=True,
                                                    align="center", color=CHILL_BROWN)
        self.counter.set_submobjects([self._counter_cache[(n, x, y)]])
        if self.counter not in self.mobjects:
            self.add(self.counter)

    @staticmethod
    def move_slot(video, start, end, **kwargs):
        """Animate a StepVideo from one (x, y, w, h) slot to another."""
        a0, a1 = np.array(start, float), np.array(end, float)
        return UpdateFromAlphaFunc(video, lambda m, a: m.place(*interpolate(a0, a1, a)), **kwargs)

    @staticmethod
    def fade(image, start=0.0, end=1.0, **kwargs):
        """Opacity animation for a StepVideo (keeps the mobject itself, and so its texture, on screen)."""
        return UpdateFromAlphaFunc(image, lambda m, a: m.set_opacity(interpolate(start, end, a)), **kwargs)

    @staticmethod
    def pulse(vmob, width_factor=3.0, time_width=0.6, **kwargs):
        """A brighter, thicker flash running along a stroke."""
        hot = vmob.copy().set_fill(opacity=0).set_stroke(FRESH_TAN, float(vmob.get_stroke_width()) * width_factor, 1)
        return VShowPassingFlash(hot, time_width=time_width, **kwargs)

    # ------------------------------------------------------------------------------------------- beats
    def beat_intro(self, run_time=None):
        """Title, transformer and prompt draw in together."""
        tf = self.tf
        anims = [
            LaggedStart(
                AnimationGroup(ShowCreation(tf.border), FadeIn(tf.columns, lag_ratio=0.3)),
                AnimationGroup(FadeIn(tf.cells, lag_ratio=0.02), ShowCreation(tf.grid, lag_ratio=0.02), FadeIn(tf.dots)),
                AnimationGroup(ShowCreation(tf.wires, lag_ratio=0.01), FadeIn(tf.nodes, lag_ratio=0.02)),
                *([Write(tf.label)] if len(tf.label) else []),
                lag_ratio=0.35),
            ShowCreation(self.prompt_arrow[0]),
            FadeIn(self.prompt_arrow[1]),
        ]
        if self.show_text:
            anims += [Write(self.title), Write(self.prompt, lag_ratio=0.05)]
        self.play(*anims, run_time=run_time or self.intro_time)
        self.settle(tf)

    def beat_io(self, run_time=None):
        self.left.set_opacity(0)
        self.add(self.left)
        self.play(
            self.fade(self.left),
            LaggedStart(
                AnimationGroup(ShowCreation(self.in_arrow[0]), FadeIn(self.in_arrow[1])),
                AnimationGroup(ShowCreation(self.out_arrow[0]), FadeIn(self.out_arrow[1])),
                lag_ratio=0.6),
            run_time=run_time or self.io_time)

    def forward_pass(self, step, run_time=None):
        """Input (left, at `step`) runs through the model; the next, slightly cleaner video fades in on the right."""
        self.right.set_step(step + 1).place(*self.SLOT_R).set_opacity(0)
        self.add(self.right)
        nodes = sorted(self.tf.nodes, key=lambda m: m.get_center()[0])
        wave = LaggedStart(*[
            UpdateFromAlphaFunc(n, lambda m, a: m.set_fill(interpolate_color(self.bg, FRESH_TAN, there_and_back(a)), 1))
            for n in nodes], lag_ratio=0.03)
        self.play(
            LaggedStart(self.pulse(self.in_arrow[0]), wave, self.pulse(self.out_arrow[0]), self.fade(self.right),
                        lag_ratio=0.45),
            run_time=run_time or self.forward_time)
        for n in nodes:
            n.set_fill(self.bg, 1)
        self.set_counter(step + 1)
        if self.counter not in self.mobjects and len(self.counter):
            self.add(self.counter)

    def feedback(self, run_time=None):
        """The output shrinks, rides the loop past the plus node, and lands on the left as the next input.

        The first time through, the loop isn't on screen yet: its line draws in behind the travelling video, the
        plus node appears as the video reaches it, and the arrowhead lands as the video arrives.
        """
        (lx, ly, lw, lh), (rx, ry, rw, rh) = self.SLOT_L, self.SLOT_R
        (sx, sy), (ex, ey), loop_y = self.LOOP_START, self.LOOP_END, self.LOOP_Y
        route = [P(rx + rw / 2, ry + rh / 2), P(sx, sy), P(sx, loop_y), P(ex, loop_y), P(ex, ey), P(lx + lw / 2, ly + lh / 2)]
        full_h, small = rh * U, self.TRAVEL_SCALE

        def ride(m, a):
            grow = max(smooth(1 - a / 0.2), smooth((a - 0.8) / 0.2))      # 1 at both ends, 0 through the middle
            h = interpolate(small, 1.0, np.clip(grow, 0, 1)) * full_h
            m.set_width(h * rw / rh, stretch=True).set_height(h, stretch=True)
            m.move_to(polyline_point(route, smooth(a)))

        if not getattr(self, "loop_drawn", False):
            self.loop_drawn = True
            seg = np.linalg.norm(np.diff(np.array(route), axis=0), axis=1)
            total, d0, loop_len = seg.sum(), seg[0], seg[1:4].sum()
            pos = lambda a: smooth(a) * total                                   # how far along the route the video is
            d_plus = d0 + seg[1] + (sx - self.PLUS[0]) * U                      # ... and where the plus node sits on it
            ramp = lambda a, start, length: float(np.clip((pos(a) - start) / length, 0, 1))
            line, barb = self.loop
            self.plus.set_opacity(0)
            barb.set_opacity(0)
            self.add(line, self.plus, barb)         # under the video, so the line appears to trail out from behind it
            self.bring_to_front(self.right)
            self.play(UpdateFromAlphaFunc(self.right, ride, rate_func=linear),
                      ShowCreation(line, rate_func=lambda a: ramp(a, d0, loop_len)),     # line tip stays under the video
                      UpdateFromAlphaFunc(self.plus, lambda m, a: m.set_opacity(ramp(a, d_plus - 10 * U, 8 * U)),
                                          rate_func=linear),
                      UpdateFromAlphaFunc(barb, lambda m, a: m.set_opacity(ramp(a, d0 + loop_len - 2 * U, 4 * U)),
                                          rate_func=linear),
                      run_time=run_time or self.first_feedback_time)
        else:
            self.bring_to_front(self.right)
            self.play(UpdateFromAlphaFunc(self.right, ride, rate_func=linear),
                      self.pulse(self.loop[0], time_width=0.25, rate_func=smooth),
                      run_time=run_time or self.feedback_time)
        self.left.set_step(self.right.step)
        self.remove(self.right)

    def beat_teach_steps(self, forward_time=None, feedback_time=None):
        """forward_time / feedback_time: seconds per pass through the model / per trip round the loop."""
        for s in range(min(self.teach_steps, self.n_steps)):
            self.forward_pass(s, forward_time)
            self.wait(self.output_hold)
            self.feedback(feedback_time)

    def beat_fast_steps(self, run_time=None):
        """run_time: total seconds for the ticking steps (spread evenly), instead of fast_step_time per step."""
        first = min(self.teach_steps, self.n_steps)
        if first >= self.n_steps:
            return
        self.forward_pass(first)                 # left = first, right = first + 1, both stay up from here on
        steps = list(range(first + 1, self.n_steps))
        per = run_time / (len(steps) + 1) if run_time else self.fast_step_time
        times = list((None if run_time else self.fast_step_times) or [per] * len(steps))[:len(steps)]
        times += [per] * (len(steps) - len(times))
        edges = np.cumsum([per] + times)       # the hold on `first`, then one entry per later step
        t0 = self.clock.t

        def tick():
            k = int(np.searchsorted(edges, self.clock.t - t0, side="right"))
            s = min(first + k, self.n_steps - 1)
            self.left.set_step(s)
            self.right.set_step(s + 1)
            self.set_counter(s + 1)

        self.on_tick.append(tick)
        self.wait(float(edges[-1]))
        self.on_tick.remove(tick)
        self.left.set_step(self.n_steps - 1)
        self.right.set_step(self.n_steps)
        self.set_counter(self.n_steps)
        if self.final_feedback:                  # the finished video takes one last trip round the loop to the input side
            self.wait(self.output_hold)
            self.feedback()

    # ------------------------------------------------------------------------------------------- P44 beats
    def beat_rearrange(self, run_time=None):
        """-01 -> -02: prompt + transformer to the left, finished video to the right of the model. The input arrow,
        feedback loop, plus node and step counter go; prompt -> transformer and transformer -> video arrows stay.
        The video is paused on whatever frame it's showing while it moves."""
        video, other = (self.left, self.right) if self.final_feedback else (self.right, self.left)
        start = self.SLOT_L if self.final_feedback else self.SLOT_R
        d = np.array([self.TF_SHIFT[0] * U, -self.TF_SHIFT[1] * U, 0.0])
        dt = np.array([self.TITLE_SHIFT[0] * U, -self.TITLE_SHIFT[1] * U, 0.0])
        self.clock.fixed = self.clock.frame()           # pause playback

        going = [self.in_arrow, self.plus]
        going += [self.loop] if getattr(self, "loop_drawn", False) else []
        going += [self.counter] if len(self.counter) else []
        anims = [FadeOut(m) for m in going]
        if other in self.mobjects:
            anims.append(self.fade(other, 1.0, 0.0))
        anims += [self.tf.animate.shift(d), self.prompt_arrow.animate.shift(d), self.out_arrow.animate.shift(d)]
        if self.show_text:
            anims += [self.title.animate.shift(dt),
                      self.prompt.animate.scale(self.PROMPT_SCALE, about_point=self.prompt.get_bottom()).shift(d)]

        self.remove(video)      # so it's re-added last, on top of the transformer it passes over
        self.play(*anims, self.move_slot(video, start, self.BIG), run_time=run_time or self.rearrange_time)
        for m in going:
            self.remove(*m.get_family())
        self.remove(other)
        self.bring_to_front(video)
        self.counter.set_submobjects([])
        self.counter_pos = self.COUNTER_44
        self.big = video

    def beat_play_once(self, run_time=None):
        """Back to the first frame, a short pause, then the clip plays through once and stops on its last frame.
        run_time: seconds for the play-through (default: real time at video_fps)."""
        c = self.clock
        if c.fixed is None:                                  # still looping: let the current loop finish first
            self.wait((c.n - (c.t * c.fps) % c.n) / c.fps)
        c.fixed = 0
        self.wait(self.play_once_pre_hold)
        t0, dur = c.t, run_time or c.n / c.fps

        def tick():
            c.fixed = min(c.n - 1, int((c.t - t0) / dur * c.n))

        self.on_tick.append(tick)
        self.wait(dur)
        self.on_tick.remove(tick)
        c.fixed = c.n - 1
        self.wait(self.play_once_post_hold)

    def grid_slot(self, k):
        return (self.GRID_X[k % 3], self.GRID_Y[k // 3], self.GRID_W, self.GRID_H)

    def beat_split(self, run_time=None):
        """The video (stopped on its last frame) deals out into 12 stills, one per latent frame, 3 across x 4 down."""
        self.tiles = [self.new_video(self.n_steps, frame=min(f, self.clock.n - 1)).place(*self.BIG)
                      for f in self.GRID_FRAMES]
        self.add(*self.tiles)               # last tile on top: it shows the frame the video stopped on
        self.remove(self.big)
        self.play(LaggedStart(*[self.move_slot(t, self.BIG, self.grid_slot(k)) for k, t in enumerate(self.tiles)],
                              lag_ratio=0.08),
                  run_time=run_time or self.split_time)

    def beat_grid_denoise(self, run_time=None, renoise_time=None):
        """All 12 frames run back to noise, then the diffusion steps play forward on all of them at once.
        run_time: total seconds for the forward steps (spread evenly), instead of grid_step_time per step.
        renoise_time: seconds for the run back to noise."""
        n, t0 = self.n_steps, self.clock.t
        renoise_time = renoise_time or self.renoise_time

        def set_all(step):
            for t in self.tiles:
                t.set_step(step)

        def back():
            set_all(n - int(np.clip((self.clock.t - t0) / renoise_time, 0, 1) * n))

        self.on_tick.append(back)
        self.wait(renoise_time)
        self.on_tick.remove(back)
        set_all(0)
        self.wait(self.renoise_hold)

        per = run_time / n if run_time else self.grid_step_time
        times = list((None if run_time else self.grid_step_times) or [per] * n)[:n]
        times += [per] * (n - len(times))
        edges = np.cumsum(times)
        t1 = self.clock.t

        def forward():
            k = int(np.searchsorted(edges, self.clock.t - t1, side="right"))      # steps finished so far
            set_all(min(k + 1, n))
            self.set_counter(min(k + 1, n))

        self.on_tick.append(forward)
        self.wait(float(edges[-1]))
        self.on_tick.remove(forward)
        set_all(n)
        self.set_counter(n)

    # ------------------------------------------------------------------------------------------- P44: attention pattern
    def tf_icon(self, col, row):
        """The pieces (16 cells + grid lines) of one little attention-pattern icon in the transformer graphic."""
        box = sorted(self.tf.columns, key=lambda m: m.get_center()[0])[col]
        cells = [c for c in self.tf.cells if box.get_left()[0] <= c.get_center()[0] <= box.get_right()[0]]
        cells = sorted(cells, key=lambda c: -c.get_center()[1])[16 * row:16 * (row + 1)]
        xs, ys = [c.get_center()[0] for c in cells], [c.get_center()[1] for c in cells]
        pad = 0.8 * U
        lines = [l for l in self.tf.grid
                 if min(xs) - pad <= l.get_center()[0] <= max(xs) + pad and min(ys) - pad <= l.get_center()[1] <= max(ys) + pad]
        icon = VGroup(*cells, *lines)
        icon.cells, icon.lines = cells, lines
        return icon

    def small_grid_slot(self, k):
        return (self.GRID3_X[k % 3], self.GRID3_Y[k // 3], self.GRID3_W, self.GRID3_H)

    def beat_icon_highlight(self, run_time=None):
        """One attention-pattern icon inside the transformer turns pink."""
        tf = self.tf
        self.icon = icon = self.tf_icon(*self.ICON)
        mine = set(id(m) for m in icon)
        layers = (tf.border, tf.columns, tf.cells, tf.grid, tf.wires, tf.nodes, tf.dots)
        self.tf_rest = VGroup(*[m for layer in layers for m in layer if id(m) not in mine],
                              *([tf.label] if len(tf.label) else []))
        self.remove(*tf.get_family())           # from here on the transformer is two mobjects: the icon and the rest
        self.add(self.tf_rest, icon)
        self.play(*[c.animate.set_fill(PINK) for c in icon.cells],
                  *[l.animate.set_stroke(PINK) for l in icon.lines],
                  run_time=run_time or self.icon_color_time)

    def beat_clear_left(self, run_time=None):
        """Everything on the left except the pink icon fades; the frame grid shrinks to the right half (-03)."""
        going = [self.tf_rest, self.prompt_arrow, self.out_arrow]
        going += [self.prompt] if self.show_text else []
        going += [self.counter] if len(self.counter) else []
        anims = [FadeOut(m) for m in going]
        anims += [self.move_slot(t, self.grid_slot(k), self.small_grid_slot(k)) for k, t in enumerate(self.tiles)]
        if self.show_text:
            anims.append(self.title.animate.shift(np.array([self.TITLE_SHIFT_3[0] * U, -self.TITLE_SHIFT_3[1] * U, 0.0])))
        self.play(*anims, run_time=run_time or self.clear_left_time)
        for m in going:
            self.remove(*m.get_family())
        self.counter.set_submobjects([])

    def beat_icon_to_bracket(self, run_time=None, bracket_time=None):
        """The icon drops to the bottom of the screen, centered under where the pattern goes; then a line runs out
        of each side of it to the pattern's edges and turns up: a bracket saying "this is a blow-up of that"."""
        ax, ay, aw, ah = self.ATTN
        cx, by = ax + aw / 2, self.BRACKET_Y
        target = self.icon.copy()
        scale_with_strokes(target, self.ICON_SCALE)
        target.move_to(P(cx, by))
        self.play(Transform(self.icon, target), run_time=run_time or self.icon_move_time)

        half = self.icon.get_width() / U / 2 + 0.9          # where the lines start, in storyboard units from center
        r = self.BRACKET_R
        arms = VGroup()
        for sgn, edge in ((-1, ax), (1, ax + aw)):
            # icon side -> out along the bottom -> round the corner -> short tick up (in equal-length pieces, so it
            # draws at an even speed)
            corner = [(edge - sgn * r + sgn * r * np.sin(a), by - r + r * np.cos(a)) for a in np.linspace(0, PI / 2, 10)]
            outline = np.array([(cx + sgn * half, by), *corner, (edge, self.BRACKET_TOP)])
            n = int(np.linalg.norm(np.diff(outline, axis=0), axis=1).sum() / 0.25)
            arms.add(VMobject().set_points_as_corners([P(*polyline_point(outline, k / n)) for k in range(n + 1)]))
        arms.set_stroke(PINK, 0.2 * STROKE_PER_U).set_fill(opacity=0)
        self.bracket = arms
        self.play(*[ShowCreation(a) for a in arms], run_time=bracket_time or self.bracket_time)

    def dimension(self, a, b, string, gap_pad=1.6):
        """A storyboard-style dimension: a line from a to b with the number sitting in a gap at its middle."""
        a, b = np.array(a, float), np.array(b, float)
        mid, u = (a + b) / 2, (b - a) / np.linalg.norm(b - a)
        text = u_text(string, 3.2, 0, 0, font=DIM_FONT, align="center", color=CHILL_BROWN)
        gap = text.get_width() / U / 2 + gap_pad
        if abs(u[1]) > abs(u[0]):
            text.rotate(PI / 2)
        text.move_to(P(*mid))
        lines = VGroup(u_line(mid - gap * u, a, 0.1, CHILL_BROWN), u_line(mid + gap * u, b, 0.1, CHILL_BROWN))
        return VGroup(lines, text)

    def beat_pattern(self, run_time=None, dims_time=None):
        """The full attention pattern grows up from just above the icon; then its dimensions and label come in."""
        path = RUN / "p44" / "attn" / f"{self.ATTN_NAME}.png"
        if not path.exists():
            raise FileNotFoundError(f"Attention pattern not found: {path}")
        self.pattern = img = ImageMobject(str(path))
        try:
            img.image.load()
        except Exception:
            pass
        end = np.array(self.ATTN, float)
        w = self.icon.get_width() / U           # starts icon-sized, sitting on the pattern's bottom edge above the icon,
        start = np.array([end[0] + end[2] / 2 - w / 2, end[1] + end[3] - w, w, w])     # so it never covers the icon

        def grow(m, a):
            place_rect(m, *interpolate(start, end, a))
            m.set_opacity(float(np.clip(a / 0.25, 0, 1)))

        grow(img, 0)
        self.add(img)
        self.play(UpdateFromAlphaFunc(img, grow), run_time=run_time or self.pattern_time)

        ax, ay, aw, ah = self.ATTN
        try:
            n_tokens = json.loads((RUN / "frame_map.json").read_text())["n_tokens"]
        except Exception:
            n_tokens = 18720
        number = f"{n_tokens:,}"
        self.dims = VGroup(self.dimension((ax, self.DIM_TOP_Y), (ax + aw, self.DIM_TOP_Y), number),
                           self.dimension((self.DIM_LEFT_X, ay + ah), (self.DIM_LEFT_X, ay), number))
        anims = [ShowCreation(l) for d in self.dims for l in d[0]]
        if self.show_text:
            m = re.search(r"layer_(\d+)_head_(\d+)", self.ATTN_NAME)
            self.attn_label = u_text(f"LAYER {int(m.group(1))} HEAD {int(m.group(2))}", 2.2, ax + aw / 2,
                                     self.ATTN_LABEL_Y, weight="SEMIBOLD", align="center", color=CHILL_BROWN)
            anims += [Write(d[1]) for d in self.dims] + [Write(self.attn_label)]
        self.play(*anims, run_time=dims_time or self.dims_time)

    # ------------------------------------------------------------------------------------------- P44: spacetime patch
    # Block coordinates: storyboard units, origin at the block's center, x right, y up, z toward the viewer.
    # The frames are sheets one spacing apart, the front one at z = 5.5 * spacing. With BLOCK_LAST_IN_FRONT the last
    # frame is in front and time runs toward the viewer; otherwise frame 0 is in front and time runs back.
    def blk_z(self, k):
        """Block-space depth of frame k's sheet, in units of the sheet spacing."""
        return (k - 5.5) if self.BLOCK_LAST_IN_FRONT else (5.5 - k)

    def blk_proj(self, pts, frac=1.0):
        """Block-space points [N, 3] -> on-screen offsets (manim units), turned by `frac` of the block's angles."""
        th, ph = np.radians(self.BLOCK_ANGLES[0]) * frac, np.radians(self.BLOCK_ANGLES[1]) * frac
        ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        rx = np.array([[1, 0, 0], [0, np.cos(ph), -np.sin(ph)], [0, np.sin(ph), np.cos(ph)]])
        q = np.atleast_2d(np.array(pts, float)) @ (rx @ ry).T
        return np.stack([q[:, 0] * U, q[:, 1] * U, np.zeros(len(q))], axis=1)

    def blk_screen(self, pts):
        """Block-space points -> manim points, at the block's current zoom."""
        base = self.blk_center + self.blk_proj(pts)
        return self.blk_fixed + self.blk_zoom * (base - self.blk_fixed)

    def blk_setup(self):
        ratio = self.GRID3_H / self.GRID3_W
        sp = self.BLOCK_SPACING

        def box(w):     # corners of the whole ruled volume (12 patches deep) for a frame width w
            return np.array([[x * w / 2, y * w * ratio / 2, z * w * sp]
                             for x in (-1, 1) for y in (-1, 1) for z in (5.5, -6.5)])

        bx, by, bw, bh = self.BLOCK_BOX
        unit = self.blk_proj(box(1.0)) / U
        ext = unit.max(0) - unit.min(0)
        self.blk_w = W = min(bw / ext[0], bh / ext[1])
        self.blk_h, self.blk_s = W * ratio, W * sp
        pr = self.blk_proj(box(W))
        self.blk_center = P(bx + bw / 2, by + bh / 2) - (pr.max(0) + pr.min(0)) / 2
        self.blk_t, self.blk_zoom, self.blk_fixed = 0.0, 1.0, np.zeros(3)
        # zooming = scaling about the one point that carries the patch's front face to PATCH_VIEW
        (x0, x1), (y0, y1), (zf, zb) = self.patch_bounds(*self.PATCH_TOKEN)
        front = self.blk_center + self.blk_proj([[(x0 + x1) / 2, (y0 + y1) / 2, zf]])[0]
        Z = self.PATCH_ZOOM
        self.blk_fixed = (Z * front - P(*self.PATCH_VIEW)) / (Z - 1)
        self.blk_lines = []
        self.blk_driver = Mobject()

    def patch_bounds(self, f, r, c):
        """Block-space extent of token (frame, row, col): (x0, x1), (y0, y1), (z front, z back)."""
        W, H, s = self.blk_w, self.blk_h, self.blk_s
        wp, hp = W * self.PATCH_PX / 832, H * self.PATCH_PX / 480
        x0, y1, zf = -W / 2 + c * wp, H / 2 - r * hp, self.blk_z(f) * s
        return (x0, x0 + wp), (y1 - hp, y1), (zf, zf - s)

    def blk_update(self):
        """Put every frame (and every line riding on the block) where the block's current state says."""
        W, H, s, t = self.blk_w, self.blk_h, self.blk_s, self.blk_t
        local = np.array([[-W / 2, H / 2, 0], [-W / 2, -H / 2, 0], [W / 2, H / 2, 0], [W / 2, -H / 2, 0]])   # UL DL UR DR
        for k, tile in enumerate(self.tiles):
            depth = np.array([0, 0, self.blk_z(k) * s])
            if t >= 1:
                corners = self.blk_screen(local + depth)
            else:       # part-way between its flat grid slot and its place in the block
                gx, gy, gw, gh = self.small_grid_slot(k)
                center = interpolate(P(gx + gw / 2, gy + gh / 2), self.blk_center + self.blk_proj([depth])[0], t)
                corners = center + interpolate(gw / W, 1.0, t) * self.blk_proj(local, frac=t)
            tile.set_quad(*corners)
        for m in self.blk_lines:
            m.set_points_as_corners(self.blk_screen(m.bpts))
        self.sync_tile_grids()

    def make_tile_grid(self, color, stroke_u, opacity):
        """A patch grid (52 x 30 cells) for one frame, as a single mobject. Its points are kept in the frame's own
        coordinates (.uv: 0-1 across, 0-1 down) so it can be laid onto the frame wherever and however it sits."""
        nx, ny = 832 // self.PATCH_PX, 480 // self.PATCH_PX
        g = VMobject()
        for u in np.linspace(0, 1, nx + 1):
            g.start_new_path(np.array([u, 0.0, 0.0]))
            g.add_line_to(np.array([u, 1.0, 0.0]))
        for v in np.linspace(0, 1, ny + 1):
            g.start_new_path(np.array([0.0, v, 0.0]))
            g.add_line_to(np.array([1.0, v, 0.0]))
        g.uv = g.get_points().copy()
        g.set_stroke(color, stroke_u * STROKE_PER_U, opacity).set_fill(opacity=0)
        return g

    def sync_tile_grids(self):
        """Lay each frame's patch grid onto its frame (call after moving the frames once the grids exist)."""
        for tile, g in zip(self.tiles, getattr(self, "tile_grids", [])):
            ul, dl, ur = tile.get_points()[:3]
            g.set_points(ul + g.uv[:, :1] * (ur - ul) + g.uv[:, 1:2] * (dl - ul))

    def blk_line(self, pts, color, stroke_u, opacity=1.0, fill=None):
        """A polyline (or filled polygon) that lives in block space and follows the block."""
        m = VMobject()
        m.bpts = np.array(pts, float)
        m.set_points_as_corners(self.blk_screen(m.bpts))
        m.set_stroke(color, stroke_u * STROKE_PER_U, opacity)
        m.set_fill(fill[0], fill[1]) if fill else m.set_fill(opacity=0)
        self.blk_lines.append(m)
        return m

    def blk_restack(self):
        """Layer order while the block is up: frames, anything drawn on the block, the masks, then everything that
        must stay visible on the left and along the top."""
        self.remove(*self.blk_masks.get_family())
        self.add(self.blk_masks)
        for name in ("pattern", "dims", "bracket", "icon", "attn_label", "title"):
            m = getattr(self, name, None)
            if m is not None and (name != "title" or self.show_text):
                self.remove(*m.get_family())        # (some of these went on screen piece by piece)
                self.add(m)

    def blk_run(self, func, run_time, **kwargs):
        """Play an animation that drives the block: func(alpha) sets its state, then everything is re-placed."""
        def step(m, a):
            func(a)
            self.blk_update()
        self.play(UpdateFromAlphaFunc(self.blk_driver, step, **kwargs), run_time=run_time)

    def beat_stack(self, run_time=None):
        """The 12 frames lift off the grid into a tilted block, the last frame of the video in front."""
        self.blk_setup()
        self.remove(*self.tiles)
        self.add(*sorted(self.tiles, key=lambda t: self.blk_z(self.tiles.index(t))))    # back to front, nearer covers farther
        # background-colored masks: once the block is scaled up it only shows on the right half, below the title
        fw, fh = 192.0, 108.0
        self.blk_masks = VGroup(
            place_rect(Rectangle(), -10, -10, self.MASK_X + 10, fh + 20),
            place_rect(Rectangle(), -10, -10, fw + 20, self.MASK_Y + 10))
        self.blk_masks.set_fill(self.bg, 1).set_stroke(width=0)
        for m in self.blk_masks:        # tagged so manimgl never batches the masks with the lines under them (within
            m.uniforms["z_layer"] = 99.0    # a batch all fills are drawn before all strokes, which would let lines through)
        self.blk_restack()

        def state(a):
            self.blk_t = a
        self.blk_run(state, run_time or self.stack_time)
        self.blk_t = 1.0
        self.blk_update()

    def beat_rule_patches(self, run_time=None):
        """Rule the outline of every patch onto the block: 52 across x 30 down x 12 deep. Done before the zoom so
        there is always something to follow on the way in."""
        W, H, s, col = self.blk_w, self.blk_h, self.blk_s, self.PATCH_COLOR
        nx, ny, nz = 832 // self.PATCH_PX, 480 // self.PATCH_PX, len(self.tiles)
        xs, ys = np.linspace(-W / 2, W / 2, nx + 1), np.linspace(H / 2, -H / 2, ny + 1)
        zf, zb = 5.5 * s, (5.5 - nz) * s
        zs = np.linspace(zf, zb, nz + 1)
        x0, y1 = -W / 2, H / 2
        rules = [[(x, y1, zf), (x, -y1, zf)] for x in xs] + [[(x0, y, zf), (-x0, y, zf)] for y in ys]     # front
        rules += [[(x, y1, zf), (x, y1, zb)] for x in xs] + [[(x0, y1, z), (-x0, y1, z)] for z in zs]     # top
        rules += [[(x0, y, zf), (x0, y, zb)] for y in ys] + [[(x0, y1, z), (x0, -y1, z)] for z in zs]     # left side
        self.token_grid = VGroup(*[self.blk_line(r, col, self.RULE_WIDTHS[0], 0.9) for r in rules])
        self.add(self.token_grid)
        self.blk_restack()
        self.play(ShowCreation(self.token_grid, lag_ratio=0.01), run_time=run_time or self.rule_patches_time)

    def beat_patch_zoom(self, run_time=None, box_time=None, labels_time=None):
        """Scale the block up onto one spacetime patch (its outline is already there), then fill it in: tinted
        faces and a heavier outline, pixel grid, dimensions and title."""
        Z, col, dark = self.PATCH_ZOOM, self.PATCH_COLOR, self.bg
        for t in self.tiles:
            t.set_pixelated(True)

        def state(a):
            self.blk_zoom = Z ** a
            self.token_grid.set_stroke(width=float(interpolate(*self.RULE_WIDTHS, a) * STROKE_PER_U))
        self.blk_restack()
        self.blk_run(state, run_time or self.patch_zoom_time)

        (x0, x1), (y0, y1), (zf, zb) = self.patch_bounds(*self.PATCH_TOKEN)
        n, nf = self.PATCH_PX, self.PATCH_FRAMES
        # the three faces we can see: front, top, left
        front = [(x0, y1, zf), (x1, y1, zf), (x1, y0, zf), (x0, y0, zf), (x0, y1, zf)]
        top = [(x0, y1, zf), (x1, y1, zf), (x1, y1, zb), (x0, y1, zb), (x0, y1, zf)]
        left = [(x0, y1, zf), (x0, y0, zf), (x0, y0, zb), (x0, y1, zb), (x0, y1, zf)]
        self.patch_faces = VGroup(self.blk_line(front, col, 0, fill=(col, 0.12)),
                                  self.blk_line(top, col, 0, fill=(col, 0.45)),
                                  self.blk_line(left, col, 0, fill=(col, 0.30)))
        edges = [front[:-1] + front[:1],
                 [(x0, y1, zf), (x0, y1, zb), (x1, y1, zb), (x1, y1, zf)],
                 [(x0, y1, zb), (x0, y0, zb), (x0, y0, zf)]]
        self.patch_edges = VGroup(*[self.blk_line(e, col, 0.35) for e in edges])
        # pixel grid: 16 x 16 on the front, 16 pixels x 4 frames on the top and the side
        xs, ys = np.linspace(x0, x1, n + 1)[1:-1], np.linspace(y1, y0, n + 1)[1:-1]
        zs = np.linspace(zf, zb, nf + 1)[1:-1]
        grid = [[(x, y1, zf), (x, y0, zf)] for x in xs] + [[(x0, y, zf), (x1, y, zf)] for y in ys]
        grid += [[(x, y1, zf), (x, y1, zb)] for x in xs] + [[(x0, y1, z), (x1, y1, z)] for z in zs]
        grid += [[(x0, y, zf), (x0, y, zb)] for y in ys] + [[(x0, y1, z), (x0, y0, z)] for z in zs]
        self.patch_grid = VGroup(*[self.blk_line(g, dark, 0.07, 0.45) for g in grid])

        self.add(self.patch_faces, self.patch_edges)
        self.blk_restack()
        self.play(FadeIn(self.patch_faces), ShowCreation(self.patch_edges, lag_ratio=0.3),
                  run_time=box_time or self.patch_box_time)
        self.blk_restack()

        anims = [FadeIn(self.patch_grid, lag_ratio=0.02)]
        self.patch_labels = VGroup()
        if self.show_text:
            def at(block_pt, dx=0.0, dy=0.0):       # block point -> storyboard (x, y), nudged by (dx, dy) units
                q = self.blk_screen([block_pt])[0]
                return q[0] / U + 96 + dx, 54 - q[1] / U + dy
            corners = self.blk_screen([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (zf, zb)])
            top_y = 54 - corners[:, 1].max() / U
            mid_x = (corners[:, 0].max() + corners[:, 0].min()) / 2 / U + 96
            px, fr = f"{n} pixels", f"{nf} frames"
            dim = dict(font=DIM_FONT, color=CHILL_BROWN)
            self.patch_labels = VGroup(
                u_text("SPACETIME PATCH", 3.4, mid_x, top_y - 4.0, bold=True, align="center", color=col),
                u_text(px, 3.2, *at(((x0 + x1) / 2, y0, zf), 0, 4.2), align="center", **dim),
                u_text(px, 3.2, *at((x1, (y0 + y1) / 2, zf), 1.8, 1.1), align="left", **dim),
                u_text(fr, 3.2, *at((x0, y0, (zf + zb) / 2), -1.8, 3.0), align="right", **dim))
            anims.append(Write(self.patch_labels))
        self.play(*anims, run_time=labels_time or self.patch_labels_time)

    def beat_pull_back(self, run_time=None):
        """Labels and pixel grid clear, then the block pulls back out to the whole ruled volume while the video dims,
        so it reads as a pile of these boxes."""
        Z = self.PATCH_ZOOM
        grid0 = float(self.patch_grid[0].get_stroke_opacity()) if len(self.patch_grid) else 0.0

        def state(a):
            self.blk_zoom = Z ** (1 - a)
            early = float(np.clip(a / 0.15, 0, 1))                 # labels + pixel grid clear out first
            self.patch_labels.set_opacity(float(1 - early))
            self.patch_grid.set_stroke(opacity=grid0 * (1 - early))
            self.token_grid.set_stroke(width=float(interpolate(*self.RULE_WIDTHS[::-1], a) * STROKE_PER_U))
            self.patch_edges.set_stroke(width=float(interpolate(0.35, 0.12, a) * STROKE_PER_U))
            for t in self.tiles:
                t.set_opacity(float(interpolate(1.0, self.DIM_OPACITY, a)))
        self.blk_run(state, run_time or self.pull_back_time)
        self.blk_zoom = 1.0
        self.blk_update()
        self.remove(self.patch_grid, self.patch_labels)
        self.blk_lines = [m for m in self.blk_lines if m not in set(self.patch_grid)]
        for t in self.tiles:
            t.set_pixelated(False)

    def beat_unstack(self, run_time=None):
        """The highlighted patch and the rules on the block's top and side fade while the video comes back up to
        full brightness; then the block flattens to the grid again, every frame carrying its own patch grid."""
        run_time = run_time or self.unstack_time
        col, (flat_w, flat_op) = self.PATCH_COLOR, self.FLAT_RULE
        rule_op = float(self.token_grid[0].get_stroke_opacity())
        edge_op = float(self.patch_edges[0].get_stroke_opacity())
        fills = [float(f.get_fill_opacity()) for f in self.patch_faces]

        # one grid per frame, each drawn just above its own frame (so nearer frames still cover farther grids)
        self.tile_grids = [self.make_tile_grid(col, self.RULE_WIDTHS[0], 0.0) for _ in self.tiles]
        self.sync_tile_grids()
        self.remove(*self.tiles)
        for k in sorted(range(len(self.tiles)), key=self.blk_z):
            self.add(self.tiles[k], self.tile_grids[k])
        self.blk_restack()

        def fade(m, a):     # block rules -> per-frame grids (the front one sits exactly on the block's front rules)
            self.token_grid.set_stroke(opacity=float(rule_op * (1 - a)))
            self.patch_edges.set_stroke(opacity=float(edge_op * (1 - a)))
            for f, o in zip(self.patch_faces, fills):
                f.set_fill(opacity=float(o * (1 - a)))
            for g in self.tile_grids:
                g.set_stroke(opacity=float(interpolate(0.0, rule_op, a)))
            for t in self.tiles:
                t.set_opacity(float(interpolate(self.DIM_OPACITY, 1.0, a)))
        self.play(UpdateFromAlphaFunc(self.blk_driver, fade), run_time=run_time * 0.35)
        self.remove(self.token_grid, self.patch_edges, self.patch_faces)
        self.blk_lines = []

        def state(a):
            self.blk_t = 1 - a
            for g in self.tile_grids:
                g.set_stroke(width=float(interpolate(self.RULE_WIDTHS[0], flat_w, a) * STROKE_PER_U),
                             opacity=float(interpolate(rule_op, flat_op, a)))
        self.blk_run(state, run_time * 0.65)
        self.remove(self.blk_masks)
        for k, t in enumerate(self.tiles):
            t.place(*self.small_grid_slot(k))
        self.sync_tile_grids()

    # ------------------------------------------------------------------------------------------- P44: 12 x 12 blocks
    def beat_block_lines(self, run_time=None, row_time=None):
        """Tan lines split the pattern into its 12 natural columns (drawn top to bottom, left to right), then into
        12 rows (drawn left to right, top to bottom). One column / row per latent frame: 1,560 tokens each."""
        ax, ay, aw, ah = self.ATTN
        n = len(self.tiles)
        w, op = self.BLOCK_LINE
        xs = [ax + k * aw / n for k in range(1, n)]
        ys = [ay + k * ah / n for k in range(1, n)]
        self.col_lines = VGroup(*[u_line((x, ay), (x, ay + ah), w, FRESH_TAN, op) for x in xs])
        self.row_lines = VGroup(*[u_line((ax, y), (ax + aw, y), w, FRESH_TAN, op) for y in ys])
        self.play(ShowCreation(self.col_lines, lag_ratio=0.15), run_time=run_time or self.col_lines_time)
        self.play(ShowCreation(self.row_lines, lag_ratio=0.15), run_time=row_time or self.row_lines_time)

    def beat_frames_to_blocks(self, run_time=None):
        """Frame k's patch grid -> block k on one row of the pattern.

        Each frame's grid gets a pale pink panel under it, then grid + panel lift off the frame together and are
        squashed onto the block (the grid is laid onto the moving panel, so its 52 x 30 cells crowd into the block
        and fade as the panel takes over): the frame's 1,560 patches are that block's 1,560 columns.
        """
        ax, ay, aw, ah = self.ATTN
        n, col = len(self.tiles), self.PATCH_COLOR
        cw, ch = aw / n, ah / n
        grids = list(getattr(self, "tile_grids", []))
        if len(grids) != n:         # (run without the spacetime-patch section: make the grids here)
            grids = [self.make_tile_grid(col, self.FLAT_RULE[0], 0.0) for _ in self.tiles]
            self.tile_grids = grids
            self.sync_tile_grids()
            self.add(*grids)
        grid_op = [float(g.get_stroke_opacity()) for g in grids]
        self.panels = VGroup()
        anims = []
        for k, (tile, g) in enumerate(zip(self.tiles, grids)):
            ul, dl, ur = [np.array(q) for q in tile.get_points()[:3]]
            y0 = ay + self.PANEL_ROW * ch
            cul, cdl, cur = P(ax + k * cw, y0), P(ax + k * cw, y0 + ch), P(ax + (k + 1) * cw, y0)
            panel = VMobject().set_points_as_corners([ul, ur, ur + dl - ul, dl, ul])
            panel.set_fill(col, 0.0).set_stroke(col, 0.12 * STROKE_PER_U, 0.0)
            self.panels.add(panel)

            def fly(m, a, g=g, panel=panel, src=(ul, dl, ur), dst=(cul, cdl, cur), op=grid_op[k]):
                rise = float(np.clip(a / 0.2, 0, 1))                    # panel fades up on the frame ...
                go = smooth(float(np.clip((a - 0.2) / 0.8, 0, 1)))      # ... then it and the grid head for the block
                p_ul, p_dl, p_ur = [interpolate(s0, d0, go) for s0, d0 in zip(src, dst)]
                panel.set_points_as_corners([p_ul, p_ur, p_ur + p_dl - p_ul, p_dl, p_ul])
                panel.set_fill(opacity=self.PANEL_OPACITY * rise).set_stroke(opacity=rise)
                g.set_points(p_ul + g.uv[:, :1] * (p_ur - p_ul) + g.uv[:, 1:2] * (p_dl - p_ul))
                g.set_stroke(opacity=float(op * (1 - go)))

            anims.append(UpdateFromAlphaFunc(Group(panel, g), fly, rate_func=linear))
        self.add(self.panels)
        self.bring_to_front(*grids)
        self.play(LaggedStart(*anims, lag_ratio=self.PANEL_LAG), run_time=run_time or self.frames_to_blocks_time)
        self.remove(*grids)
        self.tile_grids = []            # the grids are gone: they turned into the panels
        self.add(self.panels)

    # ------------------------------------------------------------------------------------------- P44: block <-> frames (-05)
    def hl_block_rect(self):
        """(x, y, w, h) of the highlighted block of the pattern."""
        ax, ay, aw, ah = self.ATTN
        n = len(self.tiles)
        return (ax + (self.HL_COL_FRAME - 1) * aw / n, ay + (self.HL_ROW_FRAME - 1) * ah / n, aw / n, ah / n)

    def dashed_arrow(self, corners, color=YELLOW, stroke_u=0.2, start_head=False):
        """Storyboard-style connector: a dashed line through `corners` (storyboard points) with rounded corners and
        a notched arrowhead at the end (and at the start too with start_head). -> VGroup(dashes, head[, start head]);
        the dashes are in order, so ShowCreation(dashes, lag_ratio=1) draws the line from start to end."""
        pts = [np.array(c, float) for c in corners]
        r = self.LINK_R
        u0 = (pts[1] - pts[0]) / np.linalg.norm(pts[1] - pts[0])
        outline = [pts[0] + 1.52 * u0 if start_head else pts[0]]
        for a, b, c in zip(pts, pts[1:], pts[2:]):          # round each corner with a quarter turn
            u, v = (b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)
            rr = min(r, np.linalg.norm(b - a) / 2, np.linalg.norm(c - b) / 2)
            p0, center = b - rr * u, b - rr * u + rr * v
            outline += [center - rr * v * np.cos(t) + rr * u * np.sin(t) for t in np.linspace(0, PI / 2, 8)]
        tip = pts[-1]
        u = (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2])
        nrm = np.array([-u[1], u[0]])
        outline.append(tip - 1.52 * u)                      # the line stops at the head's notch
        outline = np.array(outline)
        total = np.linalg.norm(np.diff(outline, axis=0), axis=1).sum()
        dash, gap = self.LINK_DASH
        dashes, d = VGroup(), 0.0
        while d < total:
            e = min(d + dash, total)
            seg = [P(*polyline_point(outline, q / total)) for q in np.linspace(d, e, 4)]
            dashes.add(VMobject().set_points_as_corners(seg))
            d += dash + gap
        dashes.set_stroke(color, stroke_u * STROKE_PER_U).set_fill(opacity=0)
        head = Polygon(P(*tip), P(*(tip - 1.99 * u + 0.815 * nrm)), P(*(tip - 1.52 * u)), P(*(tip - 1.99 * u - 0.815 * nrm)))
        head.set_fill(color, 1).set_stroke(color, 0)
        out = VGroup(dashes, head)
        if start_head:
            n0 = np.array([-u0[1], u0[0]])
            a = pts[0]
            head0 = Polygon(P(*a), P(*(a + 1.99 * u0 + 0.815 * n0)), P(*(a + 1.52 * u0)), P(*(a + 1.99 * u0 - 0.815 * n0)))
            out.add(head0.set_fill(color, 1).set_stroke(color, 0))
        return out

    def beat_block_highlight(self, run_time=None):
        """Yellow border around one block of the pattern (row = HL_ROW_FRAME, column = HL_COL_FRAME). The pink
        panels from the last beat fade out with it."""
        bx, by, bw, bh = self.hl_block_rect()
        self.block_hl = place_rect(Rectangle(), bx, by, bw, bh)
        self.block_hl.set_stroke(YELLOW, self.HL_STROKE * STROKE_PER_U).set_fill(opacity=0)
        anims = [ShowCreation(self.block_hl)]
        panels = getattr(self, "panels", None)
        if panels is not None and len(panels):
            anims.append(FadeOut(panels))
        self.play(*anims, run_time=run_time or self.block_hl_time)
        if panels is not None:
            self.remove(*panels.get_family())

    def beat_frame_link(self, kind="row", run_time=None, arrow_time=None):
        """Highlight one of the block's two frames and connect it to the block.

        kind="row": the row's frame (HL_ROW_FRAME); its arrow leaves the top of the frame, runs round between the
                    pattern and the frames, and comes into the block from the side, along its row.
        kind="col": the column's frame (HL_COL_FRAME); its arrow leaves the frame's right side, runs up and over the
                    top of the pattern, and comes down into the block from above, along its column.
        """
        bx, by, bw, bh = self.hl_block_rect()
        num = self.HL_ROW_FRAME if kind == "row" else self.HL_COL_FRAME
        fx, fy, fw, fh = self.small_grid_slot(num - 1)
        gap = 1.2                                   # between an arrow's ends and the borders
        border = place_rect(Rectangle(), fx, fy, fw, fh)
        border.set_stroke(YELLOW, self.HL_STROKE * STROKE_PER_U).set_fill(opacity=0)
        if kind == "row":
            cy = by + bh / 2
            route = [(fx + fw / 2, fy - gap), (fx + fw / 2, self.LINK_OVER_Y), (self.LINK_MID_X, self.LINK_OVER_Y),
                     (self.LINK_MID_X, cy), (bx + bw + gap, cy)]
        else:
            cx = bx + bw / 2
            route = [(fx + fw + gap, fy + fh / 2), (self.LINK_SIDE_X, fy + fh / 2), (self.LINK_SIDE_X, self.LINK_TOP_Y),
                     (cx, self.LINK_TOP_Y), (cx, by - gap)]
        arrow = self.dashed_arrow(route)
        anims = [ShowCreation(border)]
        label = VGroup()
        if self.show_text:
            label = u_text(str(num), 4.8, fx + fw / 2, fy + fh / 2 + 0.5 * 0.714 * 4.8, bold=True, align="center",
                           color=FRAME_NUM)
            anims.append(Write(label))
        self.play(*anims, run_time=run_time or self.frame_hl_time)
        t = arrow_time or self.link_arrow_time
        self.play(ShowCreation(arrow[0], lag_ratio=1), run_time=t * 0.9, rate_func=smooth)
        self.play(FadeIn(arrow[1]), run_time=t * 0.1)
        if not hasattr(self, "links"):
            self.links = {}
        self.links[kind] = VGroup(border, label, arrow)

    # ============================================================================================= P45 (-06, -07)
    def load_query(self, name=None, scale=None):
        """The query token's attention row: weights [12, 30, 52] + where the token is.
        scale: how this row is colored ("log", "gamma" or "linear"; default HEAT_SCALE)."""
        self.q_name = name or self.QUERY_NAME
        self.heat_scale = scale or self.HEAT_SCALE
        path = RUN / "p45_48" / "data" / f"{self.q_name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Attention row not found: {path}")
        d = np.load(path, allow_pickle=True)
        self.q_row = np.array(d["row"], dtype=np.float64)
        self.q_tok = int(d["q_tok"])
        self.q_fhw = tuple(int(v) for v in d["q_fhw"])
        ax, ay, aw, ah = self.ATTN
        self.q_y = ay + (self.q_tok + 0.5) / self.q_row.size * ah          # the row's height on the pattern
        f, r, c = self.q_fhw
        gx, gy, gw, gh = self.small_grid_slot(f)
        nf, nr, nc = self.q_row.shape
        self.q_patch = (gx + (c + 0.5) * gw / nc, gy + (r + 0.5) * gh / nr)      # center of its patch on its frame

    def beat_clear_block_links(self, run_time=None):
        """Fade the P44 highlights, and the left-hand dimension (the row's number goes there)."""
        going = [getattr(self, n, None) for n in ("block_hl", "col_lines", "row_lines", "panels")]
        going += list(getattr(self, "links", {}).values())
        if hasattr(self, "dims"):
            going.append(self.dims[1])
        going = [m for m in going if m is not None and len(m.get_family()) > 0]
        if going:
            self.play(*[FadeOut(m) for m in going], run_time=run_time or self.clear_links_time)
        for m in going:
            self.remove(*m.get_family())

    def beat_row_highlight(self, run_time=None):
        """A yellow line across the pattern at the query token's row, and the token's number beside it."""
        self.load_query()
        ax, ay, aw, ah = self.ATTN
        self.row_line = u_line((ax, self.q_y), (ax + aw, self.q_y), self.ROW_STROKE, YELLOW)
        anims = [ShowCreation(self.row_line)]
        self.row_label = VGroup()
        if self.show_text:
            self.row_label = u_text(f"{self.q_tok:,}", 3.2, ax - 1.6, self.q_y + 0.5 * 0.683 * 3.2, font=DIM_FONT,
                                    align="right", color=YELLOW)
            anims.append(Write(self.row_label))
        self.play(*anims, run_time=run_time or self.row_hl_time)

    def beat_patch_link(self, run_time=None, link_time=None):
        """A yellow square round the query token's patch on its frame, then a dashed two-headed connector from the
        end of the row, down between the pattern and the frames, along under the patch's frame and up to the patch."""
        ax, ay, aw, ah = self.ATTN
        px, py = self.q_patch
        half = self.PATCH_BOX / 2
        self.patch_box = place_rect(Rectangle(), px - half, py - half, 2 * half, 2 * half)
        self.patch_box.set_stroke(YELLOW, self.PATCH_BOX_STROKE * STROKE_PER_U).set_fill(opacity=0)
        self.play(ShowCreation(self.patch_box), run_time=run_time or self.patch_hl_time)

        f = self.q_fhw[0]
        gx, gy, gw, gh = self.small_grid_slot(f)
        last_row = f // 3 == len(self.GRID3_Y) - 1
        under = gy + gh + (4.2 if last_row else (self.GRID3_Y[f // 3 + 1] - gy - gh) / 2)     # a clear run under that frame
        route = [(ax + aw + 0.8, self.q_y), (self.ROW_LINK_X, self.q_y), (self.ROW_LINK_X, under), (px, under),
                 (px, py + half + 0.9)]
        self.row_link = self.dashed_arrow(route, stroke_u=self.ROW_STROKE, start_head=True)
        t = link_time or self.patch_link_time
        self.play(FadeIn(self.row_link[2]), run_time=t * 0.1)
        self.play(ShowCreation(self.row_link[0], lag_ratio=1), run_time=t * 0.8)
        self.play(FadeIn(self.row_link[1]), run_time=t * 0.1)

    def heat_strength(self):
        """The row mapped to 0-1 [12, 30, 52] for coloring, by the loaded query's scale (see load_query)."""
        w = self.q_row / self.q_row.max()
        scale = getattr(self, "heat_scale", self.HEAT_SCALE)
        if scale == "log":
            return np.clip(np.log10(np.maximum(w, 1e-30)) / self.HEAT_DECADES + 1, 0, 1)
        if scale == "linear":
            return w
        return w ** self.HEAT_GAMMA

    def heat_tag(self):
        """Short name of the current color scale, for the cached heatmap files."""
        scale = getattr(self, "heat_scale", self.HEAT_SCALE)
        return {"log": f"log{self.HEAT_DECADES}", "linear": "lin"}.get(scale, f"g{self.HEAT_GAMMA}")

    def heat_colors(self):
        """RGB [12, 30, 52, 3] for the row."""
        import matplotlib
        try:
            cmap = matplotlib.colormaps[self.HEAT_CMAP]
        except AttributeError:
            from matplotlib import cm
            cmap = cm.get_cmap(self.HEAT_CMAP)
        return cmap(self.heat_strength())[..., :3]

    def beat_row_to_heat(self, run_time=None, settle_time=None):
        """Every value in the row (all 18,720) leaves it as a colored dot and flies to its own patch on its own frame:
        token i = (frame, row, col) starts at its place along the row line and lands on patch (row, col) of that
        frame. They go in token order, so the frames fill one after another, top row to bottom row. Once they have
        landed they are the heatmap; crisp heatmap images fade in over the frames and the dots fade away."""
        if not hasattr(self, "q_row"):
            self.load_query()
        ax, ay, aw, ah = self.ATTN
        nf, nr, nc = self.q_row.shape
        n = self.q_row.size
        rgb = self.heat_colors()

        # where each value starts (along the row) and lands (its patch)
        i = np.arange(n)
        start = np.stack([(ax + (i + 0.5) / n * aw - 96) * U, np.full(n, (54 - self.q_y) * U), np.zeros(n)], axis=1)
        end = np.zeros((nf, nr, nc, 3))
        for f in range(nf):
            gx, gy, gw, gh = self.small_grid_slot(f)
            xs = (gx + (np.arange(nc) + 0.5) * gw / nc - 96) * U
            ys = (54 - (gy + (np.arange(nr) + 0.5) * gh / nr)) * U
            end[f, :, :, 0], end[f, :, :, 1] = xs[None, :], ys[:, None]
        end = end.reshape(n, 3)
        r0, r1 = self.DOT_RADII[0] * U, self.DOT_RADII[1] * U
        rgba = np.concatenate([rgb.reshape(n, 3), np.zeros((n, 1))], axis=1)

        dots = DotCloud(start.copy(), radius=r0)
        dots.set_rgba_array(rgba)
        self.heat_dots = dots
        lag, order = self.HEAT_LAG, i / (n - 1)
        strength = self.heat_strength().reshape(n)
        dot_alpha = self.DOT_MIN_ALPHA + (1 - self.DOT_MIN_ALPHA) * strength

        def fly(m, a):
            s = np.clip((a - lag * order) / (1 - lag), 0, 1)       # each dot's own progress, staggered by token index
            s = s * s * (3 - 2 * s)
            m.set_points(start + (end - start) * s[:, None])
            m.set_radii(r0 + (r1 - r0) * s)
            left = (a - lag * order) > 0                             # has this dot left the row yet?
            fade = np.clip(s / self.DOT_FADE_IN, 0, 1) if self.DOT_FADE_IN > 0 else 1.0
            rgba[:, 3] = left * fade * dot_alpha
            m.set_rgba_array(rgba)
            if marker is not None:      # rides the row at the token that is leaving right now
                front = float(np.clip(a / lag, 0, 1)) if lag > 0 else 1.0
                marker.move_to(P(ax + front * aw, self.q_y))
                marker.set_opacity(float(np.clip((1 - front) / 0.03, 0, 1)) if a > 0 else 0.0)

        marker = None
        if self.ROW_MARKER > 0:
            marker = Circle(radius=self.ROW_MARKER * U).set_fill("#ffffff", 1).set_stroke(YELLOW, self.ROW_STROKE * STROKE_PER_U)
        fly(dots, 0)
        self.add(dots)
        if marker is not None:
            self.add(marker)
        self.play(UpdateFromAlphaFunc(dots, fly, rate_func=linear), run_time=run_time or self.row_to_heat_time)
        if marker is not None:
            self.remove(marker)

        # the landed dots hand over to one heatmap image per frame (square patches, HEAT_ALPHA over the video)
        cache = HERE / "_p43_59_cache"
        cache.mkdir(exist_ok=True)
        self.heat_images = Group()
        for f in range(nf):
            tag = self.heat_tag()
            path = cache / f"heat_{self.QUERY_NAME}_{self.HEAT_CMAP}_{tag}_f{f:02d}.png"
            if not path.exists():
                im = Image.fromarray((rgb[f] * 255).round().astype(np.uint8))
                im.resize((nc * 16, nr * 16), Image.NEAREST).save(path)
            img = ImageMobject(str(path))
            try:
                img.image.load()
            except Exception:
                pass
            place_rect(img, *self.small_grid_slot(f)).set_opacity(0)
            self.heat_images.add(img)
        self.add(self.heat_images)
        self.bring_to_front(dots)
        if hasattr(self, "patch_box"):
            self.bring_to_front(self.patch_box, self.row_link)

        def settle(m, a):
            for img in self.heat_images:
                img.set_opacity(float(self.HEAT_ALPHA * a))
            rgba[:, 3] = (1 - a) * dot_alpha
            dots.set_rgba_array(rgba)
        self.play(UpdateFromAlphaFunc(dots, settle), run_time=settle_time or self.heat_settle_time)
        self.remove(dots)
        self.heat_alpha_now = self.HEAT_ALPHA

    # ------------------------------------------------------------------------------------------- P45: next head
    def make_heat_images(self):
        """One heatmap image per frame for the loaded query (see load_query), placed over the frames, invisible."""
        nf, nr, nc = self.q_row.shape
        rgb = self.heat_colors()
        cache = HERE / "_p43_59_cache"
        cache.mkdir(exist_ok=True)
        tag = self.heat_tag()
        images = Group()
        for f in range(nf):
            path = cache / f"heat_{self.q_name}_{self.HEAT_CMAP}_{tag}_f{f:02d}.png"
            if not path.exists():
                im = Image.fromarray((rgb[f] * 255).round().astype(np.uint8))
                im.resize((nc * 16, nr * 16), Image.NEAREST).save(path)
            img = ImageMobject(str(path))
            try:
                img.image.load()
            except Exception:
                pass
            images.add(place_rect(img, *self.small_grid_slot(f)).set_opacity(0))
        return images

    def beat_drop_query(self, run_time=None):
        """Fade everything belonging to the current query: heatmap, patch square, connector, row line and number."""
        heat = getattr(self, "heat_images", None)
        going = [getattr(self, n, None) for n in ("patch_box", "row_link", "row_line", "row_label")]
        going = [m for m in going if m is not None and len(m.get_family()) > 0]
        anims = [FadeOut(m) for m in going]
        if heat is not None and len(heat):
            now = getattr(self, "heat_alpha_now", self.HEAT_ALPHA)      # whatever opacity the heatmap went up at
            anims.append(UpdateFromAlphaFunc(
                heat, lambda m, a: [img.set_opacity(float(now * (1 - a))) for img in m]))
        if anims:
            self.play(*anims, run_time=run_time or self.drop_query_time)
        for m in going:
            self.remove(*m.get_family())
        if heat is not None:
            self.remove(*heat)
            self.heat_images = Group()

    def beat_swap_pattern(self, name, run_time=None):
        """Crossfade the attention pattern to <RUN>/p44/attn/<name>.png, and its LAYER / HEAD label with it."""
        path = RUN / "p44" / "attn" / f"{name}.png"
        if not path.exists():
            raise FileNotFoundError(f"Attention pattern not found: {path}")
        old = self.pattern
        new = ImageMobject(str(path))
        try:
            new.image.load()
        except Exception:
            pass
        place_rect(new, *self.ATTN).set_opacity(0)
        self.add(new)
        anims = [UpdateFromAlphaFunc(new, lambda m, a: m.set_opacity(float(a)))]     # new one fades in over the old
        old_label, new_label = getattr(self, "attn_label", None), None
        if self.show_text:
            ax, ay, aw, ah = self.ATTN
            m = re.search(r"layer_(\d+)_head_(\d+)", name)
            new_label = u_text(f"LAYER {int(m.group(1))} HEAD {int(m.group(2))}", 2.2, ax + aw / 2,
                               self.ATTN_LABEL_Y, weight="SEMIBOLD", align="center", color=CHILL_BROWN)
            anims.append(FadeIn(new_label))
            if old_label is not None:
                anims.append(FadeOut(old_label))
        self.play(*anims, run_time=run_time or self.swap_pattern_time)
        self.remove(old)
        if old_label is not None:
            self.remove(*old_label.get_family())
        self.pattern = new
        if new_label is not None:
            self.attn_label = new_label

    def beat_query_with_heat(self, name, run_time=None, link_time=None, heat_alpha=None, scale=None):
        """Row line + number for query `name`, then its patch square and connector draw (as in beat_patch_link)
        while its heatmap fades in over the frames. heat_alpha: this heatmap's opacity (default HEAT_ALPHA).
        scale: this heatmap's color scale, "log", "gamma" or "linear" (default HEAT_SCALE)."""
        self.heat_alpha_now = alpha = self.HEAT_ALPHA if heat_alpha is None else heat_alpha
        self.load_query(name, scale)
        ax, ay, aw, ah = self.ATTN
        self.row_line = u_line((ax, self.q_y), (ax + aw, self.q_y), self.ROW_STROKE, YELLOW)
        anims = [ShowCreation(self.row_line)]
        self.row_label = VGroup()
        if self.show_text:
            self.row_label = u_text(f"{self.q_tok:,}", 3.2, ax - 1.6, self.q_y + 0.5 * 0.683 * 3.2, font=DIM_FONT,
                                    align="right", color=YELLOW)
            anims.append(Write(self.row_label))
        self.play(*anims, run_time=run_time or self.row_hl_time)

        px, py = self.q_patch
        half = self.PATCH_BOX / 2
        self.patch_box = place_rect(Rectangle(), px - half, py - half, 2 * half, 2 * half)
        self.patch_box.set_stroke(YELLOW, self.PATCH_BOX_STROKE * STROKE_PER_U).set_fill(opacity=0)
        f = self.q_fhw[0]
        gx, gy, gw, gh = self.small_grid_slot(f)
        last_row = f // 3 == len(self.GRID3_Y) - 1
        under = gy + gh + (4.2 if last_row else (self.GRID3_Y[f // 3 + 1] - gy - gh) / 2)
        route = [(ax + aw + 0.8, self.q_y), (self.ROW_LINK_X, self.q_y), (self.ROW_LINK_X, under), (px, under),
                 (px, py + half + 0.9)]
        self.row_link = self.dashed_arrow(route, stroke_u=self.ROW_STROKE, start_head=True)
        box, (dashes, head, head0) = self.patch_box, self.row_link

        self.heat_images = self.make_heat_images()
        self.add(self.heat_images)              # under the square and the connector
        t = link_time or self.query_heat_time
        # one play, each piece confined to its own slice of it (and invisible until its slice starts)
        self.play(
            UpdateFromAlphaFunc(self.heat_images, lambda m, a: [img.set_opacity(float(alpha * a)) for img in m]),
            ShowCreation(box, rate_func=squish_rate_func(smooth, 0, 0.25)),
            FadeIn(head0, rate_func=squish_rate_func(smooth, 0, 0.25)),
            ShowCreation(dashes, lag_ratio=1, rate_func=squish_rate_func(smooth, 0.25, 0.9)),
            FadeIn(head, rate_func=squish_rate_func(smooth, 0.9, 1)),
            run_time=t)

    # ============================================================================================= P47
    def beat_title_highlight(self, run_time=None, hold=None, restore_time=None):
        """BIDIRECTIONAL in the title: each letter swells a little and turns yellow, running left to right; they
        stay yellow for `hold`, then fade back to the title color."""
        if not self.show_text:
            return
        letters = list(self.title[:self.TITLE_WORD])
        base = [(g.get_points().copy(), g.get_center().copy()) for g in letters]

        def pop(g, pts, c):
            def f(m, a):
                m.set_points(c + (pts - c) * (1 + self.TITLE_POP * there_and_back(a)))
                m.set_fill(interpolate_color(CHILL_BROWN, YELLOW, smooth(a)), 1)
            return UpdateFromAlphaFunc(g, f, rate_func=linear)

        self.play(LaggedStart(*[pop(g, *b) for g, b in zip(letters, base)], lag_ratio=0.12),
                  run_time=run_time or self.title_hl_time)
        for g, (pts, c) in zip(letters, base):
            g.set_points(pts)
        self.wait(self.title_hl_hold if hold is None else hold)
        self.play(*[UpdateFromAlphaFunc(g, lambda m, a: m.set_fill(interpolate_color(YELLOW, CHILL_BROWN, a), 1))
                    for g in letters], run_time=restore_time or self.title_restore_time)

    # ============================================================================================= P49 (-08, -09)
    def beat_causal_mask(self, run_time=None):
        """Dim every block above the diagonal: row frame earlier than column frame, i.e. looking at the future."""
        ax, ay, aw, ah = self.ATTN
        n = len(self.tiles)
        c = aw / n
        pts = [(ax + c, ay), (ax + aw, ay), (ax + aw, ay + (n - 1) * c)]
        for r in range(n - 2, -1, -1):                      # back up the staircase along the diagonal
            pts += [(ax + (r + 1) * c, ay + (r + 1) * c), (ax + (r + 1) * c, ay + r * c)]
        self.causal_mask = Polygon(*[P(*q) for q in pts]).set_fill(self.bg, 0.0).set_stroke(width=0)
        self.add(self.causal_mask)
        self.play(self.causal_mask.animate.set_fill(self.bg, self.MASK_OPACITY),
                  run_time=run_time or self.causal_mask_time)

    def make_block_pair(self, row_frame, col_frame, color, label):
        """Everything for one block + its two frames, in `color` (nothing is added to the scene).
        -> dict: block, borders, numbers, links (two dashed arrows), arrow (shaft, head), text, group, others."""
        ax, ay, aw, ah = self.ATTN
        n = len(self.tiles)
        c = aw / n
        bx, by = ax + (col_frame - 1) * c, ay + (row_frame - 1) * c
        gap, w = 1.2, self.PAIR_BORDER * STROKE_PER_U

        block = place_rect(Rectangle(), bx, by, c, c).set_stroke(color, w).set_fill(opacity=0)
        slots = {f: self.small_grid_slot(f - 1) for f in (row_frame, col_frame)}
        borders = VGroup(*[place_rect(Rectangle(), *slots[f]).set_stroke(color, w).set_fill(opacity=0)
                           for f in (row_frame, col_frame)])
        numbers = VGroup()
        if self.show_text:
            numbers = VGroup(*[u_text(str(f), 4.8, slots[f][0] + slots[f][2] / 2,
                                      slots[f][1] + slots[f][3] / 2 + 0.5 * 0.714 * 4.8, bold=True, align="center",
                                      color=FRAME_NUM) for f in (row_frame, col_frame)])

        # dashed arrows, frame -> block: the row frame comes in from the side, the column frame from the top
        fx, fy, fw, fh = slots[row_frame]
        cy = by + c / 2
        if (row_frame - 1) // 3 == 0:       # top row of the grid: up and over, down between pattern and frames
            row_route = [(fx + fw / 2, fy - gap), (fx + fw / 2, self.LINK_OVER_Y), (self.LINK_MID_X, self.LINK_OVER_Y),
                         (self.LINK_MID_X, cy), (bx + c + gap, cy)]
        else:                               # otherwise out the right side, round underneath, back up
            row_route = [(fx + fw + gap, fy + fh / 2), (self.LINK_SIDE_X, fy + fh / 2),
                         (self.LINK_SIDE_X, self.LINK_BOTTOM_Y), (self.LINK_MID_X, self.LINK_BOTTOM_Y),
                         (self.LINK_MID_X, cy), (bx + c + gap, cy)]
        fx, fy, fw, fh = slots[col_frame]
        cx = bx + c / 2
        if (col_frame - 1) // 3 == 0:       # top row: straight up, across above the pattern, down into the block
            col_route = [(fx + fw / 2, fy - gap), (fx + fw / 2, self.LINK_TOP_Y), (cx, self.LINK_TOP_Y), (cx, by - gap)]
        else:
            col_route = [(fx + fw + gap, fy + fh / 2), (self.LINK_SIDE_X, fy + fh / 2),
                         (self.LINK_SIDE_X, self.LINK_TOP_Y), (cx, self.LINK_TOP_Y), (cx, by - gap)]
        links = [self.dashed_arrow(r, color=color) for r in (row_route, col_route)]

        # heavy arrow from the frame being looked at (column) to the frame doing the looking (row), and the label
        def edge_point(slot, toward):       # where the line from this frame's center toward `toward` leaves the frame
            x, y, sw, sh = slot
            ctr = np.array([x + sw / 2, y + sh / 2])
            d = np.array(toward) - ctr
            k = min((sw / 2) / abs(d[0]) if d[0] else np.inf, (sh / 2) / abs(d[1]) if d[1] else np.inf)
            return ctr + d * k, d / np.linalg.norm(d)
        c_row = (slots[row_frame][0] + slots[row_frame][2] / 2, slots[row_frame][1] + slots[row_frame][3] / 2)
        c_col = (slots[col_frame][0] + slots[col_frame][2] / 2, slots[col_frame][1] + slots[col_frame][3] / 2)
        (a0, u), (b0, _) = edge_point(slots[col_frame], c_row), edge_point(slots[row_frame], c_col)
        tail, tip = a0 + 2.0 * u, b0 - 2.0 * u
        nrm = np.array([-u[1], u[0]])
        k = 2.2                              # head size relative to the small connector heads
        shaft = u_line(tail, tip - 1.52 * k * u, self.PAIR_STROKE, color)
        head = Polygon(P(*tip), P(*(tip - 1.99 * k * u + 0.815 * k * nrm)), P(*(tip - 1.52 * k * u)),
                       P(*(tip - 1.99 * k * u - 0.815 * k * nrm))).set_fill(color, 1).set_stroke(color, 0)
        text = VGroup()
        if self.show_text and label:
            mid = (tail + tip) / 2
            word = u_text(label, 3.6, mid[0] + 2.6, mid[1] + 1.3, bold=True, color=color)
            plate = RoundedRectangle(width=word.get_width() + 1.6 * U, height=word.get_height() + 1.6 * U,
                                     corner_radius=0.5 * U).move_to(word)
            plate.set_fill(self.bg, self.LABEL_PLATE).set_stroke(width=0)      # so it reads over the frame behind it
            text = VGroup(plate, word)
        arrow = VGroup(shaft, head)
        return dict(block=block, borders=borders, numbers=numbers, links=links, arrow=arrow, text=text,
                    group=VGroup(block, borders, numbers, *links, arrow, text),
                    others=[t for i, t in enumerate(self.tiles) if i + 1 not in (row_frame, col_frame)])

    def beat_block_pair(self, row_frame, col_frame, color, label, run_time=None):
        """Highlight the block in row `row_frame`, column `col_frame` (1-based) with its two frames, all in `color`,
        drawing it in piece by piece.

        The row frame is the one doing the looking, the column frame the one being looked at, so the heavy arrow
        runs from the column frame to the row frame. run_time: total for the whole build (default: the pair_* times).
        """
        d = self.make_block_pair(row_frame, col_frame, color, label)
        parts = [self.pair_block_time, self.pair_frames_time, self.pair_links_time, self.pair_arrow_time]
        if run_time:
            parts = [p * run_time / sum(parts) for p in parts]

        self.play(ShowCreation(d["block"]), run_time=parts[0])
        anims = [ShowCreation(d["borders"], lag_ratio=0.3),
                 UpdateFromAlphaFunc(Group(*d["others"]),
                                     lambda m, a: [t.set_opacity(float(interpolate(1.0, self.PAIR_DIM, a))) for t in m])]
        if len(d["numbers"]):
            anims.append(Write(d["numbers"]))
        self.play(*anims, run_time=parts[1])
        self.bring_to_front(d["borders"], d["numbers"])
        links = d["links"]
        self.play(*[ShowCreation(l[0], lag_ratio=1, rate_func=squish_rate_func(smooth, 0, 0.9)) for l in links],
                  *[FadeIn(l[1], rate_func=squish_rate_func(smooth, 0.9, 1)) for l in links],
                  run_time=parts[2])
        shaft, head = d["arrow"]
        anims = [ShowCreation(shaft, rate_func=squish_rate_func(smooth, 0, 0.7)),
                 FadeIn(head, rate_func=squish_rate_func(smooth, 0.6, 1))]
        if len(d["text"]):
            anims += [FadeIn(d["text"][0]), Write(d["text"][1])]
        self.play(*anims, run_time=parts[3])
        self.block_pair, self.pair_dimmed = d["group"], d["others"]

    def beat_crossfade_block_pair(self, row_frame, col_frame, color, label, run_time=None):
        """Swap the current block / frame highlight for another one in a single quick crossfade (nothing is drawn
        in). Frames that are in one pair but not the other fade to / from their dimmed level along the way."""
        old, old_others = getattr(self, "block_pair", None), getattr(self, "pair_dimmed", [])
        d = self.make_block_pair(row_frame, col_frame, color, label)
        dim = lambda t, group: self.PAIR_DIM if any(t is o for o in group) else 1.0
        levels = [(t, dim(t, old_others) if old is not None else 1.0, dim(t, d["others"])) for t in self.tiles]
        # (the frames' animation goes first: play() re-adds what it animates in order, and the highlights must end
        #  up above the frames)
        anims = [UpdateFromAlphaFunc(Group(*self.tiles),
                                     lambda m, a: [t.set_opacity(float(interpolate(o0, o1, a))) for t, o0, o1 in levels])]
        if old is not None:
            anims.append(FadeOut(old))
        anims.append(FadeIn(d["group"]))
        self.play(*anims, run_time=run_time or self.pair_crossfade_time)
        if old is not None:
            self.remove(*old.get_family())
        self.bring_to_front(d["group"])
        self.block_pair, self.pair_dimmed = d["group"], d["others"]

    def beat_clear_block_pair(self, run_time=None):
        """Fade the current block / frame highlight and bring the other frames back up. The causal mask stays."""
        pair, others = getattr(self, "block_pair", None), getattr(self, "pair_dimmed", [])
        if pair is None:
            return
        self.play(FadeOut(pair),
                  UpdateFromAlphaFunc(Group(*others), lambda m, a: [t.set_opacity(float(interpolate(self.PAIR_DIM, 1.0, a)))
                                                                    for t in m]),
                  run_time=run_time or self.clear_pair_time)
        self.remove(*pair.get_family())
        self.block_pair, self.pair_dimmed = None, []
