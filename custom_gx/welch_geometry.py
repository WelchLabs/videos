"""Geometry constructors shared by the Welch Labs diagrams."""

import manimgx as m
import numpy as np


def diagram_arrow(start=m.LEFT, end=m.RIGHT, *, thickness=None, tip_width_ratio=None,
                  max_width_to_length_ratio=None, **options):
    """Make an arrow with a shaft width and a tip sized relative to the shaft."""
    if thickness is not None:
        options.setdefault("stroke_width", thickness)
    if max_width_to_length_ratio is not None:
        # GL measured both widths and lengths in scene units. GX counts stroke
        # widths in hundredths of a scene unit.
        options.setdefault("max_stroke_width_to_length_ratio", 100 * max_width_to_length_ratio)
    fill = options.get("fill_color")
    if fill is not None:
        options.setdefault("stroke_color", fill)
    arrow = m.Arrow(start, end, **options)
    if tip_width_ratio == 0:
        arrow.pop_tips()
    elif tip_width_ratio is not None:
        tip = arrow.get_tip()
        if isinstance(tip, m.ArrowTriangleTip):
            # Resize across the existing triangle's base. Replacing the tip would
            # lose the 3D orientation GX gives the first tip it attaches.
            _, a, b = tip.get_vertices()[:3]
            side = b - a
            width = np.linalg.norm(side)
            if width > 0:
                unit_side = side / width
                ratio = tip_width_ratio * arrow.stroke_width / (100 * width)
                tip.apply_matrix(np.eye(3) + (ratio - 1) * np.outer(unit_side, unit_side),
                                 about_point=tip.base)
    return arrow


def _axis_style(style):
    """Translate authored tip dimensions to the NumberLine options used by GX."""
    if style is None:
        return None
    style = dict(style)
    tip = style.pop("tip_config", None)
    if tip is not None:
        if "length" in tip:
            style.setdefault("tip_height", tip["length"])
        if "width" in tip:
            style.setdefault("tip_width", tip["width"])
        style.setdefault("tip_style", {k: v for k, v in tip.items() if k not in {"length", "width"}})
    return style


def diagram_axes(*args, width=None, height=None, axis_config=None, x_axis_config=None, y_axis_config=None, **options):
    """Place 2D axes using the requested lengths in scene units."""
    if width is not None:
        options.setdefault("x_length", width)
    if height is not None:
        options.setdefault("y_length", height)
    return m.Axes(*args, axis_config=_axis_style(axis_config), x_axis_config=_axis_style(x_axis_config),
                  y_axis_config=_axis_style(y_axis_config), **options)


def diagram_axes_3d(*args, width=None, height=None, depth=None, axis_config=None,
                    x_axis_config=None, y_axis_config=None, z_axis_config=None, **options):
    """Place 3D axes using the requested lengths in scene units."""
    if width is not None:
        options.setdefault("x_length", width)
    if height is not None:
        options.setdefault("y_length", height)
    if depth is not None:
        options.setdefault("z_length", depth)
    return m.ThreeDAxes(*args, axis_config=_axis_style(axis_config), x_axis_config=_axis_style(x_axis_config),
                       y_axis_config=_axis_style(y_axis_config), z_axis_config=_axis_style(z_axis_config), **options)


class DiagramMatrix(m.Matrix):
    """Matrix with omitted entries marked by ellipses and accessible by row or column."""

    def __init__(self, entries, *, ellipses_row=None, ellipses_col=None, v_buff=0.5, h_buff=0.5, **options):
        rows = [list(row) for row in entries]
        nr, nc = len(rows), len(rows[0]) if rows else 0
        ellipses_row = self._resolve_index(ellipses_row, nr)
        ellipses_col = self._resolve_index(ellipses_col, nc)
        for r, row in enumerate(rows):
            for c in range(len(row)):
                if r == ellipses_row or c == ellipses_col:
                    row[c] = r"\ddots" if r == ellipses_row and c == ellipses_col else (
                        r"\vdots" if r == ellipses_row else r"\cdots")
        factory = options.pop("element_to_mobject", m.MathTex)
        config = options.pop("element_to_mobject_config", {}) or {}
        cells = [[entry if isinstance(entry, m.Mobject) else factory(entry, **config)
                  for entry in row] for row in rows]
        flat = [entry for row in cells for entry in row]
        options.setdefault("element_alignment_corner", m.DOWN)
        options.setdefault("bracket_h_buff", 0.2)
        super().__init__(cells, element_to_mobject=lambda entry: entry,
                         v_buff=v_buff + max((entry.height for entry in flat), default=0),
                         h_buff=h_buff + max((entry.width for entry in flat), default=0), **options)
        placed = list(super().get_entries())
        # Scene layouts slice the authored entries directly in row-major order.
        self.remove(self.elements, *self.brackets)
        self.add(*placed, *self.brackets)
        missing = {r * nc + c for r in range(nr) for c in range(nc) if r == ellipses_row or c == ellipses_col}
        self.ellipses = [entry for i, entry in enumerate(placed) if i in missing]
        self.visible_entries = [entry for i, entry in enumerate(placed) if i not in missing]

    @staticmethod
    def _resolve_index(index, count):
        return index % count if index is not None and -count <= index < count else None

    def get_entries(self):
        return m.VGroup(*self.visible_entries)

    def get_column(self, index):
        return self.get_columns()[index]

    def get_row(self, index):
        return self.get_rows()[index]
