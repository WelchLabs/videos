"""Shared animation helpers for Welch Labs scenes."""

import manimgx as m


def flash_under(mobject, color=m.YELLOW, buff=0.1, stroke_width=4, **kwargs):
    line = m.Line(mobject.get_corner(m.DL), mobject.get_corner(m.DR),
                  color=color, stroke_width=stroke_width)
    return m.ShowPassingFlash(line.shift(buff * m.DOWN), **kwargs)


def rate_func_with_early_end(t, complete_at=1):
    return m.smooth(min(t / complete_at, 1))


def apply_method(method, *args, **options):
    """Animate a bound mobject method using GX's method animation builder."""
    return getattr(method.__self__.animate(**options), method.__name__)(*args)


class TransformByGlyphMap(m.AnimationGroup):
    """Animate authored glyph correspondences, using native shape matching for incompatible layouts."""

    def __init__(self, source, target, *mappings, **kwargs):
        old, new = self._glyphs(source), self._glyphs(target)
        paths = [(rule[0], rule[1], rule[2] if len(rule) > 2 else {}) for rule in mappings]
        valid = all(0 <= index < len(glyphs) for left, right, _ in paths
                    for ref, glyphs in ((left, old), (right, new)) if not callable(ref)
                    for index in ref)
        if not valid:
            animations = [m.TransformMatchingShapes(source, target)]
        else:
            animations = self._glyph_animations(old, new, paths)
        self.source, self.target = source, target
        super().__init__(*animations, **kwargs)

    @staticmethod
    def _glyphs(formula):
        return [glyph for part in formula.submobjects
                for glyph in (part.submobjects if isinstance(part, m.MathTexPart) else (part,))]

    @staticmethod
    def _glyph_animations(old, new, paths):
        animations, old_used, new_used = [], set(), set()
        for left, right, options in paths:
            if callable(left):
                animations.append(left(m.VGroup(*(new[i] for i in right)), **options))
                new_used.update(right)
            elif callable(right):
                animations.append(right(m.VGroup(*(old[i] for i in left)), **options))
                old_used.update(left)
            elif left and right:
                before = m.VGroup(*(old[i].copy() if i in old_used else old[i] for i in left))
                after = m.VGroup(*(new[i] for i in right))
                animations.append(m.ReplacementTransform(before, after, **options))
                old_used.update(left)
                new_used.update(right)
            elif left:
                animations.append(m.FadeOut(m.VGroup(*(old[i] for i in left)), **options))
                old_used.update(left)
            elif right:
                animations.append(m.FadeIn(m.VGroup(*(new[i] for i in right)), **options))
                new_used.update(right)
        animations.extend(m.FadeOut(glyph) for i, glyph in enumerate(old) if i not in old_used)
        animations.extend(m.FadeIn(glyph) for i, glyph in enumerate(new) if i not in new_used)
        return animations

    def clean_up_from_scene(self, scene):
        scene.remove(self.source)
        scene.add(self.target)
