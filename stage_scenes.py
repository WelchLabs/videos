#!/usr/bin/env python
"""Collect existing ManimGX clips in a numbered directory for editing."""

import argparse
from itertools import count
from pathlib import Path

from manimgx.cli.scenes import load, scenes


def get_sorted_scene_classes(source):
    module = load(Path(source))
    return module.SCENES_IN_ORDER if hasattr(module, "SCENES_IN_ORDER") else scenes(module)


def stage_scenes(source, animation_dir=None):
    source = Path(source).resolve()
    animation_dir = Path(animation_dir).resolve() if animation_dir else source.parent
    rendered = sorted(p for p in animation_dir.iterdir() if p.is_file() and p.suffix in {".mp4", ".mov"})
    clips = [clip for scene in get_sorted_scene_classes(source) for clip in rendered
             if clip.stem == scene.__name__]
    if not clips:
        return None

    for number in count():
        directory = animation_dir / f"staged_scenes_{number}"
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        break
    for number, clip in reversed(list(enumerate(clips))):
        (directory / f"Scene_{number:03}_{clip.name}").symlink_to(clip)
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Python file defining the scenes")
    parser.add_argument("--animation-dir", type=Path, help="Folder containing already rendered clips")
    args = parser.parse_args()
    directory = stage_scenes(args.source, args.animation_dir)
    print(directory if directory is not None else "No matching rendered clips.")
