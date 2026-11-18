# AGENTS.md

## Repository rules

- Use [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages, e.g. `docs(agents): document Git conventions`.
- Use [Conventional Branch](https://conventionalbranch.org/) for new branch names, e.g. `chore/update-dependencies` or `codex/update-agent-guidance`.
- Run render or inspect commands only when requested. `inspect` draws frames and writes storyboard PNGs, even though it does not write video.

## What this repo is

Supporting code for Welch Labs YouTube videos. Per-video folders (`_2025/<video>/`, `_2026/<video>/`) contain research scripts and notebooks that train models, export data, and prepare assets, alongside Python scenes that use ManimGX.

Use `import manimgx as m` (or the alias already used by the file) and explicit engine symbols such as `m.Scene`. Use the installed ManimGX source or the official [API index](https://manimgx.academa.ai/llms.txt) and [full reference](https://manimgx.academa.ai/llms-full.txt) before assuming that an API from another Manim version is available. Keep `manimlib` and `manim_imports_ext` out of active scene code.

There is no repo build step. Match neighboring code, put helpers shared across videos in `custom_gx/`, and keep helpers specific to one video in its project folder.

## Setup and commands

Run these commands from the `videos/` repository root. The root `.venv` and `uv.lock` serve all active projects, including World Models and the `custom_gx` workspace member (`welch-style`). `.python-version` pins Python 3.13. Use `.venv/bin/python` for editors and notebook kernels. Keep the workspace on one environment.

```bash
uv sync --all-groups
uv run jupyter lab
uv run python run_scene.py render path/to/scene.py SceneName
uv run python run_scene.py inspect path/to/scene.py SceneName
uv run python stage_scenes.py path/to/scene.py
uv add --group ml <package>
uv lock --check
```

- The default dependency groups are `dev` and `ml`. Use `--all-groups` to install the research groups (`interp`, `rl`, `jepa`, `tf`, `vla`, `tooling`) alongside them in the same environment.
- `run_scene.py` puts the repository root on the Python import path. ManimGX also adds the scene directory for sibling imports. Anchor files bundled with a scene to `Path(__file__).resolve().parent` so they resolve independently of the working directory.
- `render` writes `<Scene>.mp4` and a storyboard beside the scene by default. Options include `-r 1280x720`, `--fps`, and `--preset ultrafast`.
- `stage_scenes.py` symlinks existing clips in scene order, honoring `SCENES_IN_ORDER` when defined. Use `--animation-dir` for clips stored elsewhere.
- When requested, run the shared style tests with `uv run --with pytest pytest custom_gx/tests` or target `custom_gx/tests/test_welch_style.py::test_palette_values`.
- The `rl` group installs `_2026/rl/references/reasoning-from-scratch` as an editable dependency. The vendored `nano-aha-moment` reference keeps its original CUDA pins for a separate Linux GPU machine.
- On Dropbox, uv uses `link-mode = "copy"`. Keep the environment local with `xattr -w com.dropbox.ignored 1 .venv` on macOS.

## Workspace asset root

Select the external asset directory that contains project folders such as `world_models`, `jepa`, and `resnet`:

```bash
uv run python -m welch_paths set "/path/to/welch_labs"
uv run python -m welch_paths show
```

`set` writes the checkout's Git-ignored `welch.local.toml`. `WELCH_LABS_ROOT` takes precedence for a shell or a single command. Without either setting, `welch_paths` looks for a nearby `welch_labs` directory and falls back to the checkout's parent. Restart a running notebook kernel after changing the saved root.

Use `from welch_paths import welch_path` and `welch_path("project/path/to/asset")` for paths under the external data root. Use `welch_root()` when a `Path` for the root is needed. Preserve existing project-specific environment overrides. If assets are missing, check the effective root and project override before editing scene code.

## Shared animation library

- `welch_style` defines the Welch palette, Myriad Pro font, and black background. Importing it sets the default `Text` font and `config.background_color`; an explicit font still wins. Its star import exports only style names. Update `custom_gx/tests` if intentionally changing those values.
- `welch_scene` provides `WelchScene`, camera view conversion, animation and save/restore helpers, clipboard formatting via `copy_view_positioning`, image planes, frustums, and texture swaps. `set_view(theta, phi, gamma, center, height)` uses degrees and storyboard frame height. `camera_view()` applies the theta offset and negates gamma for GX.
- `welch_storyboard` provides 1920×1080 storyboard coordinates, baseline text, shapes, and SVG paths. GX measures stroke widths in hundredths of a scene unit. Use `STROKE_PER_PX` when working from storyboard pixels.
- `welch_text`, `welch_axes`, `welch_surface`, `welch_geometry`, and `welch_animation` provide shared math text, axes, surfaces, shapes, and animation primitives. Import just the names the file uses.
- `custom/` contains older entry points and wrappers, and `custom_config.yml` is historical ManimGL configuration. Use `custom_gx` for new code.

## Conventions

- Keep comments short, usually at most two lines. Prefer `#` comments for local explanations and short docstrings for reusable APIs.
- Preserve authored scene names and variant entry points. Numbered takes, `*_book.py`, `*_old.py`, and copy variants can encode distinct edits.
- Keep generated media, run data, local asset configuration, and `.venv` out of Git. Preserve stored notebook outputs unless the task calls for changing them.
