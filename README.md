This project contains the supporting code used to develop the math and science videos found on [Welch Labs](https://www.youtube.com/@WelchLabs/).

This consists of two major components: the hacking scripts we use to generate data, images, etc., and the core animation used to make the final scenes. The hacking scripts are primarily done through notebooks, with a good example being `_2025/backprop_1/Loss Landscape While Learning.ipynb`. The core animation consists of scene files like `_2025/backprop_2/llama_learning_animation_1.py`.

The animation uses [ManimGX](https://manimgx.academa.ai/). Shared geometry, camera, text, surface, and style helpers live in `custom_gx/`.

## Setup

Use [uv](https://docs.astral.sh/uv/) from the repository root. Scenes and notebooks share the root `.venv` on Python 3.13, including the `custom_gx` package and the World Models scenes.

```bash
uv sync --all-groups
uv run python run_scene.py render _2025/backprop_2/llama_learning_animation_1.py SceneName
uv run jupyter lab
```

Point your editor or notebook kernel at `.venv/bin/python`. `run_scene.py` makes cross-project imports resolve from the repository root. `stage_scenes.py` can collect already rendered clips in source order.

### Shared Welch Labs assets

Choose the folder that contains the project asset directories, such as `world_models`, `jepa`, and `resnet`. This is the data folder, which can live outside the `videos` checkout.

```bash
uv run python -m welch_paths set "/path/to/welch_labs"
uv run python -m welch_paths show
uv run python run_scene.py render _2026/world_models/p20.py P20a_PhotosTo3D
```

`set` saves the location to this checkout’s ignored `welch.local.toml`. `WELCH_LABS_ROOT` overrides that setting for a shell or command. With neither set, the resolver looks for a nearby `welch_labs` directory and uses the checkout’s parent as a fallback. Scenes and notebooks call `welch_path("project/path/to/asset")` from the shared package. Existing project-specific environment overrides still work. Restart a running notebook kernel after changing the saved location.

Dependency groups (see `pyproject.toml`): `ml` (torch, transformers, diffusers, ...), `interp` (transformer-lens, sae-lens), `rl` (gymnasium, ale-py, wandb, the vendored `reasoning-from-scratch`), `jepa` (stable-worldmodel, stable-pretraining), `tf`, `vla` (jax), `tooling` (notion-client, reportlab), `dev` (jupyterlab). Add a dependency with `uv add <pkg>` or `uv add --group ml <pkg>`.

`_2026/rl/references/nano-aha-moment` is a vendored reference with its original CUDA pins for a separate Linux GPU machine. The root environment does not install that reference stack.

The repo lives in Dropbox, so mark `.venv` as ignored or Dropbox will try to sync it: `xattr -w com.dropbox.ignored 1 .venv`.

## License

The contents of this repository are available under the [MIT License](LICENSE).

Copyright © 2026 Welch Labs
