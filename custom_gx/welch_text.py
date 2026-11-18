"""Text constructors shared by Welch Labs scenes."""

import manimgx as m


def markup_text(text, *, alignment=None, **kwargs):
    """Place annotated plain text and preserve the requested paragraph alignment."""
    if alignment is not None:
        return m.Paragraph(*text.split("\n"), alignment=alignment, **kwargs)
    return m.Text(text, **kwargs)


class _SinglePartGlyphIndexing:
    """Keep authored glyph indices for formulas with one semantic part."""

    def _single_part(self):
        parts = self.submobjects
        if len(parts) == 1 and isinstance(parts[0], m.MathTexPart) and len(parts[0]):
            return parts[0]
        return None

    def __getitem__(self, index):
        part = self._single_part()
        return part[index] if part is not None else super().__getitem__(index)

    def __iter__(self):
        part = self._single_part()
        return iter(part) if part is not None else super().__iter__()

    def __len__(self):
        part = self._single_part()
        return len(part) if part is not None else super().__len__()


class WelchMathTex(_SinglePartGlyphIndexing, m.MathTex):
    pass


class WelchTex(_SinglePartGlyphIndexing, m.Tex):
    pass


def math_tex(*strings, isolate=None, t2c=None, **kwargs):
    """Typeset math with named pieces available to animation and coloring."""
    return _tex(WelchMathTex, strings, isolate, t2c, kwargs)


def tex_text(*strings, isolate=None, t2c=None, **kwargs):
    """Typeset LaTeX prose with named pieces available to animation and coloring."""
    return _tex(WelchTex, strings, isolate, t2c, kwargs)


def _tex(constructor, strings, isolate, colors, options):
    if isolate is not None:
        options.setdefault("substrings_to_isolate", [isolate] if isinstance(isolate, str) else isolate)
    if colors is not None:
        options["tex_to_color_map"] = {**colors, **(options.get("tex_to_color_map") or {})}
    return constructor(*strings, **options)
