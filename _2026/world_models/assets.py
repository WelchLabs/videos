"""Project-specific asset locations for World Models."""
from welch_paths import welch_path

import os
from pathlib import Path

DEFAULT_GRAPHICS = Path(
    welch_path('world_models/graphics')
)


def graphics_root(default=DEFAULT_GRAPHICS):
    """Locate the shared graphics, including assets for an external CameraCtrl checkout."""
    return Path(os.environ.get("WM_GRAPHICS") or default).expanduser()


GRAPHICS = graphics_root()
