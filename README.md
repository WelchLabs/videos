This project contains the supporting code used to develop the math and science videos found on [Welch Labs](https://www.youtube.com/@WelchLabs/).

This consists of two major components: the hacking scripts we use to generate data, images, etc., and the core animation used to make the final scenes. The hacking scripts are primarily done through notebooks, with a good example being `_2025/backprop_1/Loss Landscape While Learning.ipynb`. The core animation consists of scene files like `_2025/backprop_2/llama_learning_animation_1.py`.

The animation consists of scenes generated using the library [Manim](https://github.com/3b1b/manim); note that this is the 3Blue1Brown version of Manim, not Manim Community. However, this is not the only method used for the animations present in videos.

## Setup

The Python environment is managed by [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync                       # core (manimgl, numpy, ...) + jupyterlab + the torch/HF stack
uv sync --all-groups          # everything, including interp, rl, jepa, tf, vla, tooling
uv sync --group jepa          # defaults plus one extra group
```

Then run things through the environment:

```bash
uv run manimgl _2025/backprop_2/llama_learning_animation_1.py SceneName
uv run jupyter lab
```

Or point your editor / notebook kernel at `.venv/bin/python`. LaTeX (e.g. MacTeX) must be installed separately for `Tex` mobjects.

The modules in `custom/` (`welch_axes`, `welch_tools`, ...) are on the path automatically, so `from welch_axes import *` works from any file.

Dependency groups (see `pyproject.toml`): `ml` (torch, transformers, diffusers, ...), `interp` (transformer-lens, sae-lens), `rl` (gymnasium, ale-py, wandb, the vendored `reasoning-from-scratch`), `jepa` (stable-worldmodel, stable-pretraining), `tf`, `vla` (jax), `tooling` (notion-client, reportlab), `dev` (jupyterlab). Add a dependency with `uv add <pkg>` or `uv add --group ml <pkg>`.

`_2026/rl/references/nano-aha-moment` is a separate uv project with pinned CUDA-only dependencies (vllm, deepspeed, flash-attn) for a Linux GPU box; run `uv sync` inside that folder there.

The repo lives in Dropbox, so mark `.venv` as ignored or Dropbox will try to sync it: `xattr -w com.dropbox.ignored 1 .venv`.

## License

The contents of this repository are available under the [MIT License](LICENSE).

Copyright © 2026 Welch Labs
