import os
import pickle
import sys

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

gym.register_envs(ale_py)

D = 80 * 80
H = 200
n_actions = 4
action_repeat = 4

if len(sys.argv) > 1:
    model_file = sys.argv[1]
else:
    runs_dir = os.path.join('runs', 'v6')
    runs = sorted(r for r in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, r))) if os.path.exists(runs_dir) else []
    model_file = os.path.join(runs_dir, runs[-1], 'best_checkpoint.p') if runs else 'best_checkpoint.p'

if not os.path.exists(model_file):
    print(f"Model file not found: {model_file}")
    sys.exit(1)

print(f"Loading model from {model_file}")
checkpoint = pickle.load(open(model_file, 'rb'))
print(f"Model from episode {checkpoint['episode_number']}, running reward: {checkpoint['running_reward']:.2f}")

model = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, n_actions))
model.load_state_dict(checkpoint['model'])
model.eval()


def prepro(I):
    I = I[32:192]
    I = I[::2, ::2, 0]
    I[I != 0] = 1
    return I.astype(np.float32).ravel()


def step_repeat(env, action):
    total_reward = 0.0
    frames = []
    info = {}
    terminated = truncated = False

    for _ in range(action_repeat):
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frames.append(observation)
        if terminated or truncated:
            break

    observation = np.maximum(frames[-1], frames[-2]) if len(frames) >= 2 else frames[-1]
    return observation, total_reward, terminated, truncated, info


env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
observation, info = env.reset()
prev_x = None
total_reward = 0.0
games = 0

print('Watching agent play... Press Ctrl+C to stop')

try:
    while True:
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D, dtype=np.float32)
        prev_x = cur_x

        with torch.no_grad():
            logits = model(torch.from_numpy(x))
            aprob = torch.softmax(logits, dim=0).numpy()
        action = np.random.choice(n_actions, p=aprob)

        observation, reward, terminated, truncated, info = step_repeat(env, action)
        total_reward += reward

        if terminated or truncated:
            games += 1
            print(f"Game {games} finished. Score this game: {total_reward:+.0f}")
            total_reward = 0.0
            observation, info = env.reset()
            prev_x = None

except KeyboardInterrupt:
    print(f"\nWatched {games} games.")
    env.close()
