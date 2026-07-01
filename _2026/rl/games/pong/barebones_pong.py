import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

ENV_ID = "ALE/Pong-v5"
ACTIONS = [2, 3]
ACTION_MEANINGS = ["UP", "DOWN"]
H = 200
D = 80 * 80
LEARNING_RATE = 3e-4
SEED = 0
RESUME = True
RENDER = False
SAVE_EVERY = 25
PROGRESS_EVERY = 200

ALGO = "barebones"
GAME = "pong"

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs", ALGO)
os.makedirs(RUNS_DIR, exist_ok=True)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def new_run_dir():
    path = os.path.join(RUNS_DIR, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(path)
    return path


def latest_run_dir():
    runs = sorted(r for r in os.listdir(RUNS_DIR)
                  if os.path.isdir(os.path.join(RUNS_DIR, r)))
    return os.path.join(RUNS_DIR, runs[-1]) if runs else None


resume_dir = latest_run_dir() if RESUME else None
have_ckpt = resume_dir and os.path.exists(os.path.join(resume_dir, "checkpoint.p"))
run_dir = resume_dir if have_ckpt else new_run_dir()
print(f"Run: {run_dir}")

checkpoint_file = os.path.join(run_dir, "checkpoint.p")
best_checkpoint_file = os.path.join(run_dir, "best_checkpoint.p")
run_file = os.path.join(run_dir, "run.json")
plot_file = os.path.join(run_dir, "plot.png")
progress_dir = os.path.join(run_dir, "progress")
os.makedirs(progress_dir, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
# MPS is ~10x slower than CPU here: per-step kernel-launch/sync overhead dwarfs
# the tiny 6400->200->n matmuls. Use CUDA if present, else CPU (skip MPS).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, len(ACTIONS))).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

running_reward = None
best_reward = -21.0
episode_number = 0
total_steps = 0
elapsed_offset = 0.0
episodes = []

meta = {
    "game": GAME,
    "algo": ALGO,
    "framework": "pytorch",
    "description": "vanilla REINFORCE, Adam, whole-episode return, "
                   "running-reward baseline, no discount",
    "env_id": ENV_ID,
    "actions": ACTIONS,
    "action_meanings": ACTION_MEANINGS,
    "D": D,
    "H": H,
    "learning_rate": LEARNING_RATE,
    "optimizer": "adam",
    "seed": SEED,
    "device": str(device),
    "started_at": datetime.now().isoformat(timespec="seconds"),
    "git_commit": git_commit(),
    "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                 "torch": torch.__version__, "gymnasium": gym.__version__,
                 "ale_py": ale_py.__version__},
}

if have_ckpt:
    ckpt = pickle.load(open(checkpoint_file, "rb"))
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    running_reward = ckpt["running_reward"]
    best_reward = ckpt.get("best_reward", -21.0)
    episode_number = ckpt["episode_number"]
    total_steps = ckpt.get("total_steps", 0)
    elapsed_offset = ckpt.get("elapsed_s", 0.0)
    if os.path.exists(run_file):
        prev = json.load(open(run_file))
        episodes = prev.get("episodes", [])
        meta = prev.get("meta", meta)
    print(f"Resumed ep {episode_number}, running reward {running_reward:.2f}")


def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float32).ravel()


def rolling(x, k=100):
    out, s, q = [], 0.0, []
    for v in x:
        q.append(v); s += v
        if len(q) > k:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def summary(elapsed):
    rewards = [e["reward"] for e in episodes]
    recent = rewards[-100:]
    eps_s = episode_number / elapsed if elapsed > 0 else 0.0
    steps_s = total_steps / elapsed if elapsed > 0 else 0.0
    return {
        "episodes": episode_number,
        "total_steps": total_steps,
        "running_reward": running_reward,
        "best_reward": best_reward,
        "mean_reward_last_100": float(np.mean(recent)) if recent else None,
        "max_reward": float(np.max(rewards)) if rewards else None,
        "min_reward": float(np.min(rewards)) if rewards else None,
        "elapsed_s": elapsed,
        "episodes_per_sec": eps_s,
        "steps_per_sec": steps_s,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_run(elapsed):
    json.dump({"meta": meta, "summary": summary(elapsed), "episodes": episodes},
              open(run_file, "w"), indent=2)


def save_plot():
    if not episodes:
        return
    ep = [e["episode"] for e in episodes]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax = axes[0, 0]
    ax.plot(ep, [e["reward"] for e in episodes], alpha=0.25, color="teal", label="reward")
    ax.plot(ep, [e["running_reward"] for e in episodes], color="teal", lw=2, label="running reward")
    ax.set_title("episode reward"); ax.set_xlabel("episode"); ax.legend()
    ax = axes[0, 1]
    ax.plot(ep, rolling([e["loss"] for e in episodes]), color="indianred")
    ax.set_title("loss (rolling-100)"); ax.set_xlabel("episode")
    ax = axes[1, 0]
    ax.plot(ep, rolling([e["entropy"] for e in episodes]), color="darkorange")
    ax.set_title("policy entropy (rolling-100)"); ax.set_xlabel("episode")
    ax = axes[1, 1]
    ax.plot(ep, rolling([e["ep_steps"] for e in episodes]), color="slateblue")
    ax.set_title("episode length (rolling-100)"); ax.set_xlabel("episode")
    fig.suptitle("Pong - barebones REINFORCE (PyTorch, Adam, no discount)")
    fig.tight_layout()
    fig.savefig(plot_file, dpi=120)
    plt.close(fig)


def save_checkpoint(elapsed, is_best=False):
    data = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "running_reward": running_reward, "best_reward": best_reward,
        "episode_number": episode_number, "total_steps": total_steps,
        "elapsed_s": elapsed,
        "algo": ALGO, "env_id": ENV_ID, "H": H, "D": D, "actions": ACTIONS,
    }
    pickle.dump(data, open(checkpoint_file, "wb"))
    if is_best:
        pickle.dump(data, open(best_checkpoint_file, "wb"))


_progress_env = None


def render_progress():
    global _progress_env
    import imageio.v2 as imageio
    if _progress_env is None:
        _progress_env = gym.make(ENV_ID, render_mode="rgb_array")
    obs, _ = _progress_env.reset()
    prev = None
    frames = [_progress_env.render()]
    steps = 0
    while True:
        cur = prepro(obs)
        xx = cur - prev if prev is not None else np.zeros(D, dtype=np.float32)
        prev = cur
        with torch.no_grad():
            a = torch.distributions.Categorical(
                logits=model(torch.from_numpy(xx).to(device))).sample().item()
        obs, _, term, trunc, _ = _progress_env.step(ACTIONS[a])
        steps += 1
        if steps % 3 == 0:
            frames.append(_progress_env.render())
        if term or trunc or steps > 4000:
            break
    imageio.mimsave(os.path.join(progress_dir, f"ep_{episode_number:06d}.gif"),
                    frames, fps=30)


env = gym.make(ENV_ID, render_mode="human" if RENDER else None)
observation, _ = env.reset(seed=SEED)
prev_x = None
log_probs, entropies = [], []
reward_sum = 0.0
ep_steps = 0
start_time = time.time()

print(f"{ENV_ID} | H={H} D={D} lr={LEARNING_RATE} | device={device} | Adam, no discount")

try:
    while True:
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D, dtype=np.float32)
        prev_x = cur_x

        logits = model(torch.from_numpy(x).to(device))
        dist = torch.distributions.Categorical(logits=logits)
        action_idx = dist.sample()
        log_probs.append(dist.log_prob(action_idx))
        entropies.append(dist.entropy())

        observation, reward, terminated, truncated, _ = env.step(ACTIONS[action_idx.item()])
        reward_sum += reward
        ep_steps += 1
        total_steps += 1

        if terminated or truncated:
            episode_number += 1

            baseline = running_reward if running_reward is not None else 0.0
            advantage = reward_sum - baseline
            loss = -torch.stack(log_probs).sum() * advantage
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
            optimizer.step()

            mean_entropy = float(torch.stack(entropies).mean().item())
            log_probs, entropies = [], []

            running_reward = reward_sum if running_reward is None \
                else 0.99 * running_reward + 0.01 * reward_sum
            is_best = running_reward > best_reward
            if is_best:
                best_reward = running_reward

            elapsed = elapsed_offset + (time.time() - start_time)
            eps_s = episode_number / elapsed if elapsed > 0 else 0.0
            steps_s = total_steps / elapsed if elapsed > 0 else 0.0

            episodes.append({
                "episode": episode_number, "total_steps": total_steps,
                "ep_steps": ep_steps, "reward": reward_sum,
                "running_reward": running_reward, "best_reward": best_reward,
                "loss": loss.item(), "grad_norm": grad_norm,
                "entropy": mean_entropy, "elapsed_s": elapsed,
                "eps_per_sec": eps_s, "steps_per_sec": steps_s,
            })

            print(f"ep {episode_number:5d} | reward {reward_sum:+5.0f} | "
                  f"running {running_reward:6.2f} | best {best_reward:6.2f} | "
                  f"H {mean_entropy:.3f} | {steps_s:5.0f} steps/s")

            if episode_number % SAVE_EVERY == 0:
                save_checkpoint(elapsed, is_best=is_best)
                save_run(elapsed)
                save_plot()
            elif is_best:
                save_checkpoint(elapsed, is_best=True)

            if episode_number % PROGRESS_EVERY == 0:
                render_progress()

            reward_sum = 0.0
            ep_steps = 0
            observation, _ = env.reset()
            prev_x = None

except KeyboardInterrupt:
    print(f"\nInterrupted at episode {episode_number}")
finally:
    elapsed = elapsed_offset + (time.time() - start_time)
    save_checkpoint(elapsed)
    save_run(elapsed)
    save_plot()
    render_progress()
    env.close()
    if _progress_env is not None:
        _progress_env.close()
    print(f"Saved -> {run_dir}")
