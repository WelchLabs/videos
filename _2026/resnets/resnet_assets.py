"""Asset roots shared by the ResNet animation scenes."""

import os
from pathlib import Path

from welch_paths import welch_path


LOCAL_DATA = Path(__file__).resolve().parent / "data"
ACTIVATION_ROOT = Path(os.environ.get(
    "WELCH_RESNET_ASSETS", welch_path("resnet/hackin"),
)).expanduser()
LANDSCAPE_ROOT = ACTIVATION_ROOT / "P33_landscapes_v7" / "hires_option1"
