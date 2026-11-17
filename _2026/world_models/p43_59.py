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
