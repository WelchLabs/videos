"""Local and external asset locations for the Subliminal Learning scenes."""
from welch_paths import welch_path

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRAPHICS_DIR = Path(
    os.environ.get(
        "WELCH_SUBLIMINAL_GRAPHICS",
        welch_path('subliminal_learning/graphics/to_manim'),
    )
).expanduser()


def scene_asset(name):
    local_file = HERE / name
    return local_file if local_file.is_file() else GRAPHICS_DIR / name
