# SGD, no discounting, update every episode — absolute minimum policy gradient

import csv
import os
import pickle
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

H = 200
learning_rate = 1e-4
resume = True
render = False

D = 80 * 80
n_actions = 4
VERSION = 'v3'

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, 'runs', VERSION)
os.makedirs(RUNS_DIR, exist_ok=True)


def new_run_dir():
    name = datetime.now().strftime('run_%Y%m%d_%H%M%S')
    path = os.path.join(RUNS_DIR, name)
    os.makedirs(path)
    return path


def latest_run_dir():
    runs = sorted(r for r in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, r)))
    return os.path.join(RUNS_DIR, runs[-1]) if runs else None


run_dir = latest_run_dir() if (resume and latest_run_dir()) else new_run_dir()
print(f"Run: {run_dir}")

checkpoint_file = os.path.join(run_dir, 'checkpoint.p')
best_checkpoint_file = os.path.join(run_dir, 'best_checkpoint.p')
log_file = os.path.join(run_dir, 'log.csv')
plot_file = os.path.join(run_dir, 'plot.png')

model = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, n_actions))
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

running_reward = None
episode_number = 0
best_reward = 0

if resume and os.path.exists(checkpoint_file):
    checkpoint = pickle.load(open(checkpoint_file, 'rb'))
    model.load_state_dict(checkpoint['model'])
    running_reward = checkpoint['running_reward']
    episode_number = checkpoint['episode_number']
    best_reward = checkpoint.get('best_reward', 0)
    print(f"Resumed at episode {episode_number}, running reward: {running_reward:.2f}")

log_episodes, log_rewards, log_running = [], [], []
if os.path.exists(log_file):
    with open(log_file) as f:
        for row in csv.DictReader(f):
            log_episodes.append(int(row['episode']))
            log_rewards.append(float(row['reward']))
            log_running.append(float(row['running_reward']))

csv_f = open(log_file, 'a', newline='')
csv_writer = csv.writer(csv_f)
if os.path.getsize(log_file) == 0:
    csv_writer.writerow(['episode', 'reward', 'running_reward'])


def prepro(I):
    I = I[32:192]
    I = I[::2, ::2, 0]
    I[I != 0] = 1
    return I.astype(np.float32).ravel()


def save_plot():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(log_episodes, log_rewards, alpha=0.3, color='seagreen', label='reward')
    ax.plot(log_episodes, log_running, color='seagreen', linewidth=2, label='running reward')
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    ax.set_title('v3 — SGD, no discount, update every episode')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_file, dpi=120)
    plt.close(fig)


def save_checkpoint(is_best=False):
    data = {
        'model': model.state_dict(),
        'running_reward': running_reward,
        'episode_number': episode_number,
        'best_reward': best_reward,
    }
    pickle.dump(data, open(checkpoint_file, 'wb'))
    if is_best:
        pickle.dump(data, open(best_checkpoint_file, 'wb'))


env = gym.make("ALE/Breakout-v5", render_mode="human" if render else None)
observation, info = env.reset()
prev_x = None
log_probs = []
reward_sum = 0

print(f"H={H}, lr={learning_rate}, SGD+no_discount")

try:
    while True:
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D, dtype=np.float32)
        prev_x = cur_x

        logits = model(torch.from_numpy(x))
        dist = torch.distributions.Categorical(F.softmax(logits, dim=0))
        action = dist.sample()
        log_probs.append(dist.log_prob(action))

        observation, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        reward_sum += reward

        if done:
            episode_number += 1

            loss = -(torch.stack(log_probs).sum() * reward_sum)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_probs = []

            if running_reward is None:
                running_reward = reward_sum
            else:
                running_reward = 0.99 * running_reward + 0.01 * reward_sum

            is_best = running_reward > best_reward
            if is_best:
                best_reward = running_reward

            log_episodes.append(episode_number)
            log_rewards.append(reward_sum)
            log_running.append(running_reward)
            csv_writer.writerow([episode_number, reward_sum, f'{running_reward:.4f}'])
            csv_f.flush()

            print(f'Ep {episode_number} | Reward: {reward_sum:+.0f} | Running: {running_reward:.2f} | Best: {best_reward:.2f}')

            if episode_number % 50 == 0:
                save_checkpoint(is_best=is_best)
                save_plot()
            elif is_best:
                save_checkpoint(is_best=True)

            reward_sum = 0
            observation, info = env.reset()
            prev_x = None

except KeyboardInterrupt:
    print(f'\nInterrupted at episode {episode_number}')
    save_checkpoint()
    save_plot()
    csv_f.close()
