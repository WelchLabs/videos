import argparse
import glob
import os
import pickle

import numpy as np
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

ENV_ID = "ALE/Pong-v5"
H = 200
D = 80 * 80
ACTIONS = [2, 3]
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")


def find_checkpoint():
    cands = glob.glob(os.path.join(RUNS_DIR, "*", "*", "best_checkpoint.p"))
    cands += glob.glob(os.path.join(RUNS_DIR, "*", "*", "checkpoint.p"))
    return max(cands, key=os.path.getmtime) if cands else None


def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float32).ravel()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def load_policy(path):
    data = pickle.load(open(path, "rb"))
    m = data["model"]
    if isinstance(m, dict) and "W1" in m:
        def act(x):
            h = np.dot(m["W1"], x); h[h < 0] = 0
            p = sigmoid(np.dot(m["W2"], h))
            return 2 if np.random.uniform() < p else 3
        return "karpathy", act, data
    import torch
    import torch.nn as nn
    net = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, len(ACTIONS)))
    net.load_state_dict(m)
    net.eval()

    def act(x):
        import torch
        with torch.no_grad():
            logits = net(torch.from_numpy(x))
            return ACTIONS[int(logits.argmax())]
    return "pytorch", act, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default=None)
    ap.add_argument("--record", metavar="PATH")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    path = args.checkpoint or find_checkpoint()
    if path is None:
        print("No checkpoint found under runs/. Train first.")
        return

    kind, act, data = load_policy(path)
    rr = data.get("running_reward")
    print(f"Loading {kind} checkpoint: {path}")
    if rr is not None:
        print(f"  episode {data.get('episode_number', '?')} | running reward {rr:.2f}")

    writer = None
    if args.record:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.record, fps=args.fps, macro_block_size=1)
        env = gym.make(ENV_ID, render_mode="rgb_array")
    else:
        env = gym.make(ENV_ID, render_mode="human")

    try:
        for ep in range(args.episodes):
            observation, _ = env.reset()
            prev_x = None
            score = 0.0
            while True:
                cur_x = prepro(observation)
                x = cur_x - prev_x if prev_x is not None else np.zeros(D, dtype=np.float32)
                prev_x = cur_x
                observation, reward, terminated, truncated, _ = env.step(act(x))
                score += reward
                if writer is not None:
                    writer.append_data(env.render())
                if terminated or truncated:
                    print(f"episode {ep + 1} | score {score:+.0f}")
                    break
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        env.close()
        if writer is not None:
            writer.close()
            print(f"saved -> {args.record}")


if __name__ == "__main__":
    main()
