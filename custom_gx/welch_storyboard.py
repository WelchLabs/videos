"""Geometry, text and image helpers for scenes traced from storyboards."""

import re

import manimgx as m
import numpy as np


SVG_NUMBER = r"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"


def storyboard_point(x, y, width=1920, height=1080):
    """Convert a top-left-origin storyboard point into scene coordinates."""
    units = m.config.frame_height / height
    return np.array([(x - width / 2) * units, (height / 2 - y) * units, 0.0])


def baseline_text(string, height, anchor, font, bold=False, italic=False,
                  align="left", color=m.WHITE, weight=None):
    """Place text using its capital height and baseline as layout references."""
    full = m.Text(
        "H" + string, font=font, weight=weight or (m.BOLD if bold else m.NORMAL),
        slant=m.ITALIC if italic else m.NORMAL, disable_ligatures=True,
    )
    full.scale(height / full[0].height)
    label = full[1:]
    edge = {"left": label.get_left, "right": label.get_right, "center": label.get_center}
    label.shift(np.asarray(anchor) - [edge[align]()[0], full[0].get_bottom()[1], 0.0])
    return label.set_color(color)


def svg_path(d, point):
    """SVG path data (M L H V C S Z, absolute or relative, implicit repeats) -> VMobject."""
    tokens = re.findall(r"[A-Za-z]|" + SVG_NUMBER, d)
    mob = m.VMobject()
    pos, start, last_ctrl, cmd, i = np.zeros(2), np.zeros(2), None, None, 0

    def pt(xy):
        return point(*xy)

    def nums(n):
        nonlocal i
        vals = [float(t) for t in tokens[i:i + n]]
        i += n
        return vals

    while i < len(tokens):
        if re.match(r"[A-Za-z]", tokens[i]):
            cmd = tokens[i]
            i += 1
        if cmd is None:
            raise ValueError("SVG path coordinates must follow a command")
        rel, c = cmd.islower(), cmd.upper()
        if c == "Z":
            mob.close_path()
            pos, last_ctrl, cmd = start.copy(), None, None
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


STORYBOARD_BACKGROUND = "#031e1e"
FRESH_TAN = "#f4ebd9"
LABEL_FONT = "Open Sans"
TEXT_FONT = "STIX Two Text"
CAP_HEIGHT = {LABEL_FONT: 0.714, TEXT_FONT: 0.656}
PX = m.config.frame_height / 1080
# GX measures strokes in hundredths of a scene unit.
STROKE_PER_PX = 100 * PX


def storyboard_offset(dx, dy):
    """Convert an offset in 1920 × 1080 storyboard pixels to a scene vector."""
    return np.array([dx * PX, -dy * PX, 0.0])


def px_line(a, b, stroke_px, color=FRESH_TAN, opacity=1.0):
    return m.Line(storyboard_point(*a), storyboard_point(*b), stroke_color=color,
                  stroke_width=stroke_px * STROKE_PER_PX, stroke_opacity=opacity)


def px_arrowhead(tip, direction, head, color):
    """Notched arrowhead in pixels. Head dimensions are length, half width, and notch depth."""
    length, half, notch = head
    side = half * np.array([-direction[1], direction[0]])
    back = tip - length * direction
    return m.Polygon(*(storyboard_point(*p) for p in (tip, back + side, tip - notch * direction, back - side)),
                     fill_color=color, fill_opacity=1, stroke_width=0)


def px_arrow(tail, tip, stroke_px=2.0, head=(19.9, 8.13, 15.18), color=FRESH_TAN):
    tail, tip = np.asarray(tail, dtype=float), np.asarray(tip, dtype=float)
    direction = tip - tail
    direction /= np.linalg.norm(direction)
    shaft = px_line(tail, tip - head[2] * direction, stroke_px, color)
    return m.VGroup(shaft, px_arrowhead(tip, direction, head, color))


def px_box(cx, cy, width=172.97, height=146.51, radius=11.73, stroke_px=2.5, color=FRESH_TAN):
    return m.RoundedRectangle(
        width=width * PX, height=height * PX, corner_radius=radius * PX,
        stroke_color=color, stroke_width=stroke_px * STROKE_PER_PX, fill_opacity=0,
    ).move_to(storyboard_point(cx, cy))


def px_text(string, size_px, x, y, font=LABEL_FONT, bold=False, italic=False,
            align="left", color=FRESH_TAN):
    """Place a line of text at an Illustrator point size and baseline in storyboard pixels."""
    height = CAP_HEIGHT.get(font, 0.7) * size_px * PX
    return baseline_text(string, height, storyboard_point(x, y), font, bold, italic, align, color)


def load_image(path):
    """Load a local texture. GX decodes it into an array during construction."""
    return m.ImageMobject(str(path))


def place_px(image, x, y, height):
    return image.scale_to_fit_height(height * PX).move_to(storyboard_point(x, y), aligned_edge=m.UL)
