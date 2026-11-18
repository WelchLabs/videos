"""Welch Labs axes with explicit tick values and in-scene coordinate mapping."""

import manimgx as m
import numpy as np

from welch_style import CHILL_BROWN


def generate_nice_ticks(min_val, max_val, min_ticks=3, max_ticks=16, ignore=(0,)):
    low, high = sorted((min_val, max_val))
    if np.isclose(low, high, rtol=0, atol=1e-10):
        low, high = low - 1, high + 1
    power = np.floor(np.log10(high - low))
    steps = 10.0**power * np.array([1, 0.5, 0.2, 0.1])
    for step in steps:
        ticks = step * np.arange(np.ceil(low / step), np.floor(high / step) + 1)
        ticks = [float(t) for t in ticks if not any(np.isclose(t, v, atol=1e-10, rtol=0) for v in ignore)]
        if len(ticks) >= min_ticks:
            break
    if len(ticks) > max_ticks:
        stride = int(np.ceil(len(ticks) / max_ticks))
        ticks = ticks[::stride]
    return ticks, float(np.floor(low / step) * step - step), float(np.ceil(high / step) * step + step)


def axis_tip(point, direction, size, color=CHILL_BROWN):
    """Welch's notched arrow tip, with the point centered on the end of the axis."""
    side = np.cross(m.OUT, direction)
    left = point - 3.233 * size * direction
    return m.Polygon(point, left + 1.321 * size * side, left + 0.767 * size * direction,
                     left - 1.321 * size * side, fill_color=color, fill_opacity=1, stroke_width=0)


class _WelchAxis(m.VGroup):
    def __init__(self, minimum, maximum, ticks, tick_size, label_size, stroke_width,
                 color, tip_scale, length, include_tip, direction, tick_direction, digits, **kwargs):
        if minimum == maximum:
            raise ValueError("Axis minimum and maximum must differ")
        self.minimum, self.maximum = minimum, maximum
        self.axis_length_on_canvas = length
        self.axis_to_canvas_scale = (maximum - minimum) / length
        end = length * direction
        line = m.Line(m.ORIGIN, end, color=color, stroke_width=stroke_width)
        self.axis_line = m.VGroup(line, axis_tip(end, direction, tip_scale, color)) if include_tip else line
        self.ticks, self.labels = m.VGroup(), m.VGroup()
        for value in ticks:
            start = self.map_to_canvas(value) * direction
            end = start + tick_size * tick_direction
            self.ticks.add(m.Line(start, end, color=color, stroke_width=stroke_width))
            self.labels.add(m.MathTex(str(round(value, digits)), font_size=label_size, color=color)
                            .next_to(end, tick_direction, buff=0.1))
        super().__init__(self.axis_line, self.ticks, self.labels, **kwargs)

    def map_to_canvas(self, value, axis_start=0):
        fraction = (value - self.minimum) / (self.maximum - self.minimum)
        return (fraction + axis_start) * self.axis_length_on_canvas

    def get_axis_line(self):
        return self.axis_line

    def get_ticks(self):
        return self.ticks

    def get_labels(self):
        return self.labels


class WelchXAxis(_WelchAxis):
    def __init__(self, x_min=0, x_max=6, x_ticks=(1, 2, 3, 4, 5), x_tick_height=0.15,
                 x_label_font_size=24, stroke_width=3, color=CHILL_BROWN, arrow_tip_scale=0.1,
                 axis_length_on_canvas=5, include_tip=True, **kwargs):
        super().__init__(x_min, x_max, x_ticks, x_tick_height, x_label_font_size, stroke_width,
                         color, arrow_tip_scale, axis_length_on_canvas, include_tip,
                         m.RIGHT, m.DOWN, 4, **kwargs)
        self.x_min, self.x_max, self.x_ticks = x_min, x_max, x_ticks
        self.x_ticks_scaled = self.map_to_canvas(np.asarray(x_ticks))


class WelchYAxis(_WelchAxis):
    def __init__(self, y_min=0, y_max=6, y_ticks=(1, 2, 3, 4, 5), y_tick_width=0.15,
                 y_label_font_size=24, stroke_width=3, color=CHILL_BROWN, arrow_tip_scale=0.1,
                 axis_length_on_canvas=5, include_tip=True, **kwargs):
        super().__init__(y_min, y_max, y_ticks, y_tick_width, y_label_font_size, stroke_width,
                         color, arrow_tip_scale, axis_length_on_canvas, include_tip,
                         m.UP, m.LEFT, 5, **kwargs)
        self.y_min, self.y_max, self.y_ticks = y_min, y_max, y_ticks
        self.y_ticks_scaled = self.map_to_canvas(np.asarray(y_ticks))


class WelchTip(m.VGroup):
    """The Welch Labs notched tip as a standalone mobject."""

    def __init__(self, color=CHILL_BROWN, **kwargs):
        super().__init__(axis_tip(np.zeros(3), m.RIGHT, 1, color), **kwargs)
        self.center()


class WelchNumberLine(m.NumberLine):
    """Number line with notched tips at both ends."""

    def __init__(self, x_range=(0, 6, 1), include_tip=True, **style):
        style.setdefault("color", CHILL_BROWN)
        style.setdefault("stroke_width", 3)
        style.setdefault("include_numbers", True)
        self._welch_ticks = style.pop("include_ticks", True)
        super().__init__(x_range=x_range, include_tip=False, include_ticks=False, **style)
        self.starting_tip = self.ending_tip = None
        direction = m.normalize(self.get_end() - self.get_start())
        if self._welch_ticks:
            low, high = self.x_range[:2]
            ticks = [self.get_tick(t) for t in self.get_tick_range()
                     if not include_tip or not (np.isclose(t, low) or np.isclose(t, high))]
            self.add(m.VGroup(*ticks))
        if include_tip:
            size = 2 * self.tick_size / 2.642
            color = style["color"]
            self.starting_tip = axis_tip(self.get_start(), -direction, size, color)
            self.ending_tip = axis_tip(self.get_end(), direction, size, color)
            self.add(self.starting_tip, self.ending_tip)

    def remove_tip(self, position):
        attribute, index = {"start": ("starting_tip", 0), "end": ("ending_tip", 1)}[position]
        tip = getattr(self, attribute)
        if tip is not None:
            self.remove(tip)
            setattr(self, attribute, None)
            if self._welch_ticks:
                self.add(self.get_tick(self.x_range[index]))
        return self
