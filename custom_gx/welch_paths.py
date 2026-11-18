"""Workspace configuration for shared Welch Labs assets and research data."""

import argparse
from functools import cache
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import tomllib


_WORKSPACE = Path(__file__).resolve().parents[1]
_CONFIG = _WORKSPACE / "welch.local.toml"
_ENV = "WELCH_LABS_ROOT"


@cache
def _workspace_root():
    if _CONFIG.is_file():
        with _CONFIG.open("rb") as stream:
            setting = tomllib.load(stream).get("welch_labs_root")
        if not isinstance(setting, str) or not setting.strip():
            raise ValueError(f"Set welch_labs_root to a directory in {_CONFIG}")
        return (_WORKSPACE / Path(setting).expanduser()).resolve()
    for ancestor in _WORKSPACE.parents:
        if ancestor.name == "welch_labs":
            return ancestor
        candidate = ancestor / "welch_labs"
        if candidate.is_dir():
            return candidate
    return _WORKSPACE.parent


def welch_root():
    """Return the asset root for this checkout, with an environment override."""
    override = os.environ.get(_ENV)
    if override:
        return (_WORKSPACE / Path(override).expanduser()).resolve()
    return _workspace_root()


def welch_path(relative):
    """Join an asset path to the chosen root, preserving an authored trailing slash."""
    relative = os.fspath(relative)
    if Path(relative).is_absolute():
        raise ValueError(f"Expected a path relative to the Welch Labs root: {relative}")
    path = str(welch_root() / relative)
    return f"{path}/" if relative.endswith("/") and not path.endswith("/") else path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show", help="print the effective asset root")
    set_root = commands.add_parser("set", help="store the asset root for this checkout")
    set_root.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "show":
        print(welch_root())
        return

    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        parser.error(f"not a directory: {directory}")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=_CONFIG.parent, prefix=".welch.local.", delete=False) as stream:
        stream.write(f"welch_labs_root = {json.dumps(str(directory), ensure_ascii=False)}\n")
        temp = Path(stream.name)
    temp.replace(_CONFIG)
    _workspace_root.cache_clear()
    print(f"Saved Welch Labs asset root: {directory}")


if __name__ == "__main__":
    main()
