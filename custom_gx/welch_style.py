"""The Welch Labs look for manimgx scenes: palette, font, background.

    import manimgx as m
    from welch_style import *

Importing sets Myriad Pro as every Text's default font (an explicit `font=` still wins)
and the background on `manimgx.config`. Sizes and stroke widths are named here but not
forced: they were tuned per scene in manimgl and vary a lot.

Values are the ones repeated across the 2025-2026 manimgl scenes.
"""

import os
from collections.abc import Sequence
from pathlib import Path

import manimgx as _m

__all__ = [
    'CHILL_BROWN', 'YELLOW', 'BLUE', 'CHILL_GREEN', 'CHILL_BLUE',
    'YELLOW_FADE', 'FRESH_TAN', 'GREEN', 'CYAN', 'MAGENTA',
    'FONT', 'BACKGROUND', 'FONT_SIZE', 'STROKE_WIDTH', 'THIN_STROKE',
    'font_installed',
]

# Core palette
CHILL_BROWN = '#948979'
YELLOW = '#ffd35a'
BLUE = '#65c8d0'
CHILL_GREEN = '#6c946f'
CHILL_BLUE = '#3d5c6f'
YELLOW_FADE = '#7f6a2d'
FRESH_TAN = '#dfd0b9'
GREEN = '#00a14b'
CYAN = '#00ffff'
MAGENTA = '#ff00ff'

FONT = 'Myriad Pro'
BACKGROUND = '#000000'

# Common sizes, for passing explicitly
FONT_SIZE = 24
STROKE_WIDTH = 3
THIN_STROKE = 2

_FONT_DIRS = [
    Path.home() / 'Library/Fonts',
    Path('/Library/Fonts'),
    Path('/System/Library/Fonts'),
    Path('/usr/share/fonts'),
    Path.home() / '.local/share/fonts',
]
_FONT_EXTS = {'.otf', '.ttf', '.ttc'}


def _squash(name: str) -> str:
    return name.lower().replace(' ', '').replace('-', '').replace('_', '')


def font_installed(name: str, dirs: Sequence[Path] | None = None) -> bool:
    """Whether a font file for family `name` is in `dirs` (default: the system's font
    folders), matched by file name: 'Myriad Pro' finds MyriadPro-Regular.otf."""
    want = _squash(name)
    for d in dirs if dirs is not None else _FONT_DIRS:
        for root, _, files in os.walk(d):
            for f in files:
                p = Path(f)
                if p.suffix.lower() in _FONT_EXTS and _squash(p.stem).startswith(want):
                    return True
    return False


_m.Text.set_default(font=FONT)
_m.config.background_color = BACKGROUND

if not font_installed(FONT):
    print(f"welch_style: {FONT} not installed; text will use manimgx's fallback font")
