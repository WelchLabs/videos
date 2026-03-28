import subprocess, os, sys

# ── Install deps ──────────────────────────────────────────────────────────────
UV = '/root/.local/bin/uv'

if not os.path.exists('/root/openpi'):
    subprocess.run([
        'git', 'clone', '--recurse-submodules',
        'https://github.com/Physical-Intelligence/openpi.git', '/root/openpi'
    ], check=True)

subprocess.run([UV, 'pip', 'install', '--system', '-e', '/root/openpi'], check=True)

subprocess.run([
    UV, 'pip', 'install', '--system',
    '--reinstall-package', 'typing_extensions',
    '--reinstall-package', 'transformers',
    'matplotlib', 'datasets', 'diffusers',
    'accelerate', 'safetensors', 'sentencepiece',
    'typing_extensions>=4.12',
    'pytest', 'chex',
    'transformers==4.53.2',
], check=True)

# chex upgrades jax/jaxlib/numpy — pin back to what openpi requires
subprocess.run([
    UV, 'pip', 'install', '--system',
    'jax[cuda12]==0.5.3', 'jaxlib==0.5.3', 'numpy<2.0.0',
], check=True)

subprocess.run([UV, 'pip', 'install', '--system', '-e', '/root/openpi'], check=True)

# ── Path / import fixes ───────────────────────────────────────────────────────
if '/root/openpi/src' not in sys.path:
    sys.path.insert(0, '/root/openpi/src')

# transformers 4.53.2 uses @torch.compiler.disable(recursive=False) in
# flex_attention.py.  In a fresh process this triggers a torch._dynamo
# circular-import.  Patch to a no-op for the entire import block.
import importlib
import torch
import torch.nn.functional as F
import torch.compiler as _tc
_orig_compiler_disable = _tc.disable
_tc.disable = lambda fn=None, recursive=True: (lambda f: f) if fn is None else fn

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import transformers  # noqa: F401 — must import before openpi to ensure 4.53.2

from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.policies import aloha_policy
from openpi.shared import download
from openpi import transforms as _transforms
from openpi.shared import normalize as _normalize

_tc.disable = _orig_compiler_disable  # restore after all imports

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p52_assets')
os.makedirs(BASE_DIR, exist_ok=True)

NUM_EPISODES = 16
EVAL_FRAME_IDX = 150
NUM_CAT_SEQUENCES = 16

# ── Helpers ───────────────────────────────────────────────────────────────────
def frame_to_obs(frame):
    def to_chw(img):
        return np.array(img).transpose(2, 0, 1)
    return {
        "images": {
            "cam_high":        to_chw(frame["observation.images.cam_high"]),
            "cam_left_wrist":  to_chw(frame["observation.images.cam_left_wrist"]),
            "cam_right_wrist": to_chw(frame["observation.images.cam_right_wrist"]),
        },
        "state":  np.array(frame["observation.state"], dtype=np.float32),
        "prompt": "uncap the pen",
    }

# ── 1. Robot pen-uncapping episodes ──────────────────────────────────────────
print("Loading dataset...")
ds = load_dataset("physical-intelligence/aloha_pen_uncap_diverse", split="train")
episode_indices = sorted(set(ds["episode_index"]))
print(f"Total rows: {len(ds)}, episodes: {episode_indices}")

print("Loading pi0 policy...")
config = _config.get_config("pi0_aloha_pen_uncap")
checkpoint_dir = str(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base"))
policy = _policy_config.create_trained_policy(config, checkpoint_dir)
print("Policy loaded!")

ep_indices_arr = np.array(ds["episode_index"])

for ep_idx in episode_indices[:NUM_EPISODES]:
    print(f"\n=== Episode {ep_idx} ===")
    mask = ep_indices_arr == ep_idx
    episode = ds.select(np.where(mask)[0])
    print(f"  {len(episode)} frames")

    cam_dir = f'{BASE_DIR}/robot_ep{ep_idx}'
    os.makedirs(cam_dir, exist_ok=True)
    for idx in tqdm(range(len(episode)), desc=f"  cam_low ep{ep_idx}"):
        episode[idx]["observation.images.cam_low"].save(f'{cam_dir}/{idx:03d}.png')

    heatmap_dir = f'{BASE_DIR}/heatmap_ep{ep_idx}'
    os.makedirs(heatmap_dir, exist_ok=True)
    frame = episode[min(EVAL_FRAME_IDX, len(episode) - 1)]

    rng = np.random.RandomState(42 + ep_idx)
    noise = rng.randn(50, 32).astype(np.float32)
    result = policy.infer(frame_to_obs(frame), noise=noise)

    target_actions = result["actions"].astype(np.float32)
    x0 = noise.copy()
    trajectory = [x0.copy()]
    for step in range(10):
        alpha = (step + 1) / 10
        x_next = (1.0 - alpha) * x0 + alpha * np.zeros_like(x0)
        x_next[:, :target_actions.shape[-1]] = (
            (1.0 - alpha) * x0[:, :target_actions.shape[-1]] + alpha * target_actions
        )
        trajectory.append(x_next.copy())

    for i, traj in enumerate(trajectory):
        matplotlib.image.imsave(f'{heatmap_dir}/{i:02d}.png', traj[:, :14].T, cmap="viridis")
    np.save(f'{heatmap_dir}/trajectory.npy', np.array(trajectory))
    print(f"  Saved {len(episode)} cam_low frames + {len(trajectory)} heatmaps")

print(f"\nDone! {NUM_EPISODES} robot episodes saved.")

# ── 2. Cat diffusion sequences ────────────────────────────────────────────────
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

PROMPT          = "A happy gray tabby cat looking at the camera, photorealistic, 8k"
NEGATIVE_PROMPT = "cartoon, illustration, painting, drawing, anime, low quality"
SEEDS = [5, 12, 42, 77, 101, 256, 314, 628, 999, 1234, 2048, 3141, 4096, 5555, 7777, 9999]

pipe = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    torch_dtype=torch.float16,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, algorithm_type="dpmsolver++",
)
pipe.to("cuda")
print("Stable Diffusion pipeline loaded!")

for seq_idx in range(NUM_CAT_SEQUENCES):
    seed = SEEDS[seq_idx]
    output_dir = Path(f'{BASE_DIR}/cat_seq{seq_idx}')
    output_dir.mkdir(exist_ok=True)
    print(f"\n=== Cat sequence {seq_idx} (seed={seed}) ===")

    intermediate_latents = []

    def capture_latents(pipe, step_index, timestep, callback_kwargs):
        intermediate_latents.append(callback_kwargs["latents"].detach().clone())
        return callback_kwargs

    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=100,
        guidance_scale=7.0,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        callback_on_step_end=capture_latents,
    )
    result.images[0].save(output_dir / "final.png")

    for i, latents in enumerate(intermediate_latents):
        with torch.no_grad():
            decoded = pipe.vae.decode(
                latents / pipe.vae.config.scaling_factor, return_dict=False
            )[0]
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            img = Image.fromarray(
                (decoded[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8")
            )
        img.save(output_dir / f"step_{i:03d}.png")

    print(f"  Saved {len(intermediate_latents)} steps + final to {output_dir}/")

print(f"\nAll done! Assets in {BASE_DIR}/")
