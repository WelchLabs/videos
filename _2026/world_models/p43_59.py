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

Artwork comes from the Illustrator storyboards in p43_59_sketches_1/ (192 x 108 viewBox = one unit per 10 px at
1080p, mapped onto manim's 8-unit-tall frame). The transformer graphic is read straight out of p43_59-01.svg at
runtime (it's ~550 little pieces); everything else is redrawn from the storyboard's coordinates.

Video frames come from the Wan2.1 notebook p43_59_2.ipynb:  <RUN>/p43/xt/step_SS/frame_FFF.png  (31 steps x 45
frames). Set WM_P43_59 to point somewhere else (e.g. on the Linux box).

Timings and toggles are class attributes on the scene classes.
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
SKETCHES = HERE / "p43_59_sketches_1"
RUN = Path(os.environ.get(
    "WM_P43_59",
    "/Users/stephen/Library/CloudStorage/Dropbox-Stephencwelch/welch_labs/world_models/hackin/Wan2.1/"
    "attn_runs/ball_drop_01/p43_59_2"))

FRESH_TAN = "#f4ebd9"       # line art, prompt, TRANSFORMER
CHILL_BROWN = "#948678"     # title, feedback loop
BG_FALLBACK = "#031e1e"     # node fill = scene background (read from the camera at runtime when possible)
SVG_BG = "#031e1e"          # the background color used inside the storyboards (swapped for the scene's)
LABEL_FONT = "Open Sans"
TEXT_FONT = "STIX Two Text"
CAP = {LABEL_FONT: 0.714, TEXT_FONT: 0.656}     # cap height / em, to match Illustrator's point sizes

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


def u_text(string, size_u, x, y, font=LABEL_FONT, bold=False, italic=False, align="left", color=FRESH_TAN):
    """Text at storyboard size with its baseline at y. x is its left edge ("left"), right edge ("right") or middle.

    A throwaway "H" is set in front of the string: its height fixes the scale and its bottom is the baseline, so
    strings with descenders sit correctly without per-string fudge factors.
    """
    full = Text("H" + string, font=font, weight=BOLD if bold else NORMAL, slant=ITALIC if italic else NORMAL,
                font_size=48)
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
    document order. region = (x0, y0, x1, y1) in storyboard units keeps only shapes whose center is inside it.

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
            cx, cy = mob.get_center()[0] / U + 96, 54 - mob.get_center()[1] / U
            if not (region[0] <= cx <= region[2] and region[1] <= cy <= region[3]):
                return
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
            f"No step folders in {root}\n  -> point WM_P43_59 at the notebook's p43_59_2 output folder")
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

    def frame(self):
        return int(self.t * self.fps) % self.n


class StepVideo(ImageMobject):
    """One quad showing <RUN>/p43/<kind>/step_SS/frame_FFF.png. Its texture is rewritten in place, so a whole
    video (at any diffusion step) costs one mobject and one texture instead of one ImageMobject per frame.

        v = StepVideo(clock, step=0)        v.set_step(12)        v.sync()   # scene calls sync() every frame
    """
    _count = 0

    def __init__(self, clock, step=0, kind="xt", height=2.0):
        self.clock, self.step, self.kind = clock, step, kind
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
        return self.dirs[int(np.clip(self.step, 0, len(self.dirs) - 1))] / f"frame_{self.clock.frame():03d}.png"

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

    def sync(self):
        path = self.frame_path()
        if path == self._shown or self.shader_wrapper is None or not self.shader_wrapper.textures:
            return self
        self.shader_wrapper.textures[0].write(_rgba(path)[1])
        self._shown = path
        return self

    def place(self, x, y, w, h):
        """Fill the storyboard slot with top-left (x, y) and size (w, h)."""
        self.set_width(w * U, stretch=True).set_height(h * U, stretch=True)
        self.move_to(P(x + w / 2, y + h / 2))
        return self


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
    show_step_counter = True    # small "STEP 12 / 30" under the plus node (not in the storyboard)
    video_fps = 16              # playback rate of the 45-frame ball clip (Wan2.1's native rate)

    # --- timings (seconds) ---------------------------------------------------------------------------------
    intro_time = 6.0            # title + transformer + prompt, all at once
    io_time = 1.5
    pre_steps_hold = 1.0
    teach_steps = 3             # how many steps get the slow treatment (pulse through the model + trip round the loop)
    forward_time = 1.4          # pulse through the model, output fades in
    output_hold = 0.6           # output sits on the right before it heads round the loop
    first_feedback_time = 2.5   # the first trip: the feedback line + plus node draw in as the output travels
    feedback_time = 1.8         # output travels round the loop to become the next input
    fast_step_time = 0.3        # per step once they just tick by
    fast_step_times = None      # or a list of per-step seconds for the fast steps (e.g. linger on the late ones)
    final_feedback = True       # after the last step, send the finished video round the loop so it ends on the left
    end_hold = 6.0              # finished video on the right, still looping

    # --- storyboard -01 geometry (192 x 108 units) ---------------------------------------------------------
    SLOT_L = (11.99, 41.03, 49.92, 28.80)       # x, y, w, h of the input video
    SLOT_R = (131.03, 41.03, 49.92, 28.80)      # ... and the output video
    LOOP_START, LOOP_END = (156.86, 71.25), (35.38, 70.97)
    LOOP_Y, LOOP_R = 78.41, 1.44                # the loop's horizontal run and corner radius
    PLUS = (96.0, 78.69, 1.22)                  # center x, y and radius of the plus node
    TRAVEL_SCALE = 0.22                         # size of the video while it rides the loop

    def construct(self):
        self.setup_scene()
        self.beat_intro()
        self.beat_io()
        self.wait(self.pre_steps_hold)
        self.beat_teach_steps()
        self.beat_fast_steps()
        self.wait(self.end_hold)

    # ------------------------------------------------------------------------------------------- pieces
    def setup_scene(self):
        try:
            self.bg = rgb_to_hex(self.camera.background_rgba[:3])
        except Exception:
            self.bg = BG_FALLBACK

        # everything that plays video hangs off one clock, advanced by an invisible mobject's updater
        self.n_steps = len(step_dirs("xt")) - 1                         # 30 model calls -> 31 step folders
        n_frames = len(list(step_dirs("xt")[0].glob("frame_*.png")))    # 45
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

    def new_video(self, step, kind="xt"):
        v = StepVideo(self.clock, step, kind)
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
        if n not in self._counter_cache:
            self._counter_cache[n] = u_text(f"STEP {n} / {self.n_steps}", 2.0, self.PLUS[0], 85.2, bold=True,
                                            align="center", color=CHILL_BROWN)
        self.counter.set_submobjects([self._counter_cache[n]])

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
    def beat_intro(self):
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
        self.play(*anims, run_time=self.intro_time)
        self.settle(tf)

    def beat_io(self):
        self.left.set_opacity(0)
        self.add(self.left)
        self.play(
            self.fade(self.left),
            LaggedStart(
                AnimationGroup(ShowCreation(self.in_arrow[0]), FadeIn(self.in_arrow[1])),
                AnimationGroup(ShowCreation(self.out_arrow[0]), FadeIn(self.out_arrow[1])),
                lag_ratio=0.6),
            run_time=self.io_time)

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

    def beat_teach_steps(self):
        for s in range(min(self.teach_steps, self.n_steps)):
            self.forward_pass(s)
            self.wait(self.output_hold)
            self.feedback()

    def beat_fast_steps(self):
        first = min(self.teach_steps, self.n_steps)
        if first >= self.n_steps:
            return
        self.forward_pass(first)                 # left = first, right = first + 1, both stay up from here on
        steps = list(range(first + 1, self.n_steps))
        times = list(self.fast_step_times or [self.fast_step_time] * len(steps))[:len(steps)]
        times += [self.fast_step_time] * (len(steps) - len(times))
        edges = np.cumsum([self.fast_step_time] + times)       # the hold on `first`, then one entry per later step
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
