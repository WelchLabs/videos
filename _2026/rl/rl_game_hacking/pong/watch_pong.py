import numpy as np
import pickle
import gymnasium as gym
import ale_py
import os
import sys

gym.register_envs(ale_py)

model_file = sys.argv[1] if len(sys.argv) > 1 else 'pong_model_best.p'

if not os.path.exists(model_file):
    model_file = 'pong_model.p'
if not os.path.exists(model_file):
    print("No saved model found. Train first with pong_pg.py")
    sys.exit(1)

print(f"Loading model from {model_file}")
checkpoint = pickle.load(open(model_file, 'rb'))
model = checkpoint['model']
print(f"Model from episode {checkpoint['episode_number']}, running reward: {checkpoint['running_reward']:.2f}")

D = 80 * 80


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float64).ravel()


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p


env = gym.make("ALE/Pong-v5", render_mode="human")
observation, info = env.reset()
prev_x = None
total_reward = 0
games = 0

print("Watching agent play... Press Ctrl+C to stop")

try:
    while True:
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        aprob = policy_forward(x)
        action = 2 if np.random.uniform() < aprob else 3

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
