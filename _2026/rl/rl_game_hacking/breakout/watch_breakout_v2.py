import numpy as np
import pickle
import gymnasium as gym
import ale_py
import os
import sys
import torch
import torch.nn as nn

gym.register_envs(ale_py)

model_file = sys.argv[1] if len(sys.argv) > 1 else 'breakout_model_best.p'

if not os.path.exists(model_file):
    print(f"Model file not found: {model_file}")
    sys.exit(1)

print(f"Loading model from {model_file}")
checkpoint = pickle.load(open(model_file, 'rb'))
print(f"Model from episode {checkpoint['episode_number']}, running reward: {checkpoint['running_reward']:.2f}")

D = 80 * 80
H = 200
n_actions = 4

model = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, n_actions))
model.load_state_dict(checkpoint['model'])
model.eval()


def prepro(I):
    I = I[32:192]
    I = I[::2, ::2, 0]
    I[I != 0] = 1
    return I.astype(np.float32).ravel()


def policy_forward(x):
    with torch.no_grad():
        logits = model(torch.tensor(x))
        p = torch.softmax(logits, dim=0).numpy()
    return p


env = gym.make("ALE/Breakout-v5", render_mode="human")
observation, info = env.reset()
prev_x = None
total_reward = 0
games = 0

print("Watching agent play... Press Ctrl+C to stop")

try:
    while True:
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D, dtype=np.float32)
        prev_x = cur_x

        aprob = policy_forward(x)
        action = np.random.choice(n_actions, p=aprob)

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            games += 1
            print(f"Game {games} finished. Score this game: {total_reward:+.0f}")
            total_reward = 0
            observation, info = env.reset()
            prev_x = None

except KeyboardInterrupt:
    print(f"\nWatched {games} games.")
    env.close()
