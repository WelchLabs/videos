"""Shared artwork and data for the transformer storyboard scenes."""
from welch_paths import welch_path

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import manimgx as gx
import numpy as np

from welch_scene import set_image_pixels
from welch_storyboard import (
    CAP_HEIGHT, FRESH_TAN, LABEL_FONT, PX, STORYBOARD_BACKGROUND, STROKE_PER_PX,
    SVG_NUMBER, baseline_text, px_arrow, px_line, storyboard_point, svg_path,
)

HERE = Path(__file__).resolve().parent
RUN = Path(os.environ.get(
    "WM_P43_59",
    welch_path('world_models/hackin/Wan2.2/attn_runs/ball_drop_s252/p43_59_3')))

BG_FALLBACK = STORYBOARD_BACKGROUND
SVG_BG = "#031e1e"  # background color encoded in the exported artwork
DIM_FONT = "Latin Modern Roman"
CAP = {**CAP_HEIGHT, DIM_FONT: 0.683}
PIXELS_PER_UNIT = 10
U = PIXELS_PER_UNIT * PX
STROKE_PER_U = PIXELS_PER_UNIT * STROKE_PER_PX


def P(x, y):
    """Convert a point from the 192 by 108 storyboard to scene coordinates."""
    return storyboard_point(x, y, width=192, height=108)


def u_line(a, b, stroke_u=0.2, color=FRESH_TAN, opacity=1.0):
    return px_line(np.asarray(a) * PIXELS_PER_UNIT, np.asarray(b) * PIXELS_PER_UNIT,
                   stroke_u * PIXELS_PER_UNIT, color, opacity)


def arrow(tail, tip, stroke_u=0.2, head=(1.99, 0.815, 1.52), color=FRESH_TAN):
    return px_arrow(np.asarray(tail) * PIXELS_PER_UNIT, np.asarray(tip) * PIXELS_PER_UNIT,
                    stroke_u * PIXELS_PER_UNIT, tuple(length * PIXELS_PER_UNIT for length in head), color)


def u_text(string, size_u, x, y, font=LABEL_FONT, bold=False, italic=False,
           align="left", color=FRESH_TAN, weight=None):
    return baseline_text(
        string, CAP.get(font, 0.7) * size_u * U, P(x, y), font,
        bold=bold, italic=italic, align=align, color=color, weight=weight,
    )


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
        a = [float(v) for v in re.findall(SVG_NUMBER, args)]
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
            mob = gx.Line(tp(A("x1"), A("y1")), tp(A("x2"), A("y2")))
        elif tag == "rect":
            x, y, w, h = A("x"), A("y"), A("width"), A("height")
            corners = np.array([tp(x, y), tp(x + w, y), tp(x + w, y + h), tp(x, y + h)])
            ext = corners.max(0) - corners.min(0)
            rx = A("rx")
            rounded = rx > 0
            mob = gx.RoundedRectangle(width=ext[0], height=ext[1], corner_radius=rx * U) if rounded else gx.Rectangle(width=ext[0], height=ext[1])
            mob.move_to(corners.mean(0))
        elif tag == "circle":
            mob = gx.Circle(radius=A("r") * U).move_to(tp(A("cx"), A("cy")))
        elif tag == "ellipse":
            mob = gx.Circle(radius=A("rx") * U)
            if A("ry") != A("rx"):
                mob.stretch(A("ry") / A("rx"), 1)
            mob.move_to(tp(A("cx"), A("cy")))
        elif tag in ("polygon", "polyline"):
            v = np.array([float(t) for t in re.findall(SVG_NUMBER, node.get("points", ""))]).reshape(-1, 2)
            pts = [tp(*q) for q in v]
            mob = gx.Polygon(*pts) if tag == "polygon" else gx.VMobject().set_points_as_corners(pts)
        elif tag == "path":
            mob = svg_path(node.get("d", ""), tp)
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
TF_LAYERS = ("border", "columns", "cells", "grid", "wires", "nodes", "dots")


def transformer_graphic(svg, region, label_spec, bg=BG_FALLBACK, label=True,
                        thin_u=0.15, grid_stroke_u=None, outline_grid=True):
    """The TRANSFORMER box from storyboard -11 (or from another storyboard: svg, region, label_spec = (size, x,
    baseline), thin_u = strokes thinner than this are icon grid lines, grid_stroke_u = force their width / color),
    at its storyboard position.

    -> VGroup with named layers (also in this order, back to front):
       .border  .columns  .cells  .grid  .wires  .nodes  .dots  .label
    .grid holds, for each of the nine attention icons, its outline square and its eight inner lines.
    """
    layers = {k: gx.VGroup() for k in TF_LAYERS}
    for m in read_svg_shapes(svg, region, bg):
        s = m.svg
        thin = s["stroke"] != "none" and s["stroke_u"] < thin_u
        if m.svg_tag == "rect":
            key = "columns" if s["rounded"] else ("grid" if outline_grid and s["fill"] == "none" else "cells")
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
    g = gx.VGroup(*layers.values())
    for name, layer in layers.items():
        setattr(g, name, layer)
    g.label = gx.VGroup()
    if label:
        size, x, y = label_spec
        g.label = u_text("TRANSFORMER", size, x, y, bold=True)
        g.add(g.label)
    return g


def place_rect(mob, x, y, w, h):
    """Stretch a mobject to fill the storyboard rectangle with top-left (x, y) and size (w, h)."""
    mob.stretch_to_fit_width(w * U).stretch_to_fit_height(h * U)
    mob.move_to(P(x + w / 2, y + h / 2))
    return mob


def step_dirs(kind="xt"):
    """Sorted step folders: step_00 ... step_29, step_30_final."""
    root = RUN / "p43" / kind
    dirs = sorted(d for d in root.glob("step_*") if d.is_dir())
    if not dirs:
        raise FileNotFoundError(
            f"No step folders in {root}\n  -> point WM_P43_59 at the notebook's p43_59_3 output folder")
    return dirs


def wan_frame_path(dirs, step, frame):
    """Return the image for a frame at a diffusion step, clamped to the available run."""
    index = max(0, min(int(step), len(dirs) - 1))
    return dirs[index] / f"frame_{frame:03d}.png"


class StoryboardTile(gx.ImageMobject):
    """An image that can swap pixels and move between storyboard slots."""

    def show(self, pixels):
        return set_image_pixels(self, pixels)

    def place(self, x, y, width, height):
        return place_rect(self, x, y, width, height)


def settle_group(scene, group):
    """Replace individually animated members with the group that owns them."""
    scene.remove(*group.get_family())
    scene.add(group)


def move_slot(image, start, end, **kwargs):
    """Animate an image between two storyboard rectangles."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    return gx.UpdateFromAlphaFunc(
        image, lambda mob, alpha: mob.place(*gx.interpolate(start, end, alpha)), **kwargs,
    )


def fade_image(image, start=0.0, end=1.0, **kwargs):
    """Animate opacity while keeping the same live image on screen."""
    return gx.UpdateFromAlphaFunc(
        image, lambda mob, alpha: mob.set_opacity(gx.interpolate(start, end, alpha)), **kwargs,
    )


def pulse(path, width_factor=3.0, time_width=0.6, **kwargs):
    """Run a brighter, thicker stroke along a path."""
    highlight = path.copy().set_fill(opacity=0)
    highlight.set_stroke(FRESH_TAN, float(path.get_stroke_width()) * width_factor, 1)
    return gx.ShowPassingFlash(highlight, time_width=time_width, **kwargs)
