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

from breakout import Breakout

H = 200
learning_rate = 3e-4
resume = True

DOWN = 2
D = (84 // DOWN) * (84 // DOWN)
n_actions = 3

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, 'runs', 'breakout')
os.makedirs(RUNS_DIR, exist_ok=True)


def new_run_dir():
    path = os.path.join(RUNS_DIR, datetime.now().strftime('run_%Y%m%d_%H%M%S'))
    os.makedirs(path)
    return path


def latest_run_dir():
    runs = sorted(r for r in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, r)))
    return os.path.join(RUNS_DIR, runs[-1]) if runs else None


run_dir = latest_run_dir() if (resume and latest_run_dir()) else new_run_dir()
print(f"Run: {run_dir}")

checkpoint_file = os.path.join(run_dir, 'checkpoint.p')
log_file = os.path.join(run_dir, 'log.csv')
plot_file = os.path.join(run_dir, 'plot.png')

model = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, n_actions))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

running_reward = None
episode_number = 0

if resume and os.path.exists(checkpoint_file):
    ckpt = pickle.load(open(checkpoint_file, 'rb'))
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    running_reward = ckpt['running_reward']
    episode_number = ckpt['episode_number']
    print(f"Resumed ep {episode_number}, running reward: {running_reward:.2f}")

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


def prepro(obs):
    I = obs[::DOWN, ::DOWN, 0]
    return (I > 0).astype(np.float32).ravel()


def save_plot():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(log_episodes, log_rewards, alpha=0.3, color='teal', label='reward')
    ax.plot(log_episodes, log_running, color='teal', linewidth=2, label='running reward')
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    ax.set_title('Breakout - PyTorch Adam, no discount')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_file, dpi=120)
    plt.close(fig)


def save_checkpoint():
    pickle.dump({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'running_reward': running_reward,
        'episode_number': episode_number,
    }, open(checkpoint_file, 'wb'))


env = Breakout()
observation, info = env.reset()
prev_x = None
log_probs = []
reward_sum = 0.0

print(f"H={H}, D={D}, lr={learning_rate}, Adam, no discount")

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
        reward_sum += reward

        if terminated or truncated:
            episode_number += 1

            baseline = running_reward if running_reward is not None else 0.0
            loss = -torch.stack(log_probs).sum() * (reward_sum - baseline)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log_probs = []

            running_reward = reward_sum if running_reward is None else 0.99 * running_reward + 0.01 * reward_sum

            log_episodes.append(episode_number)
            log_rewards.append(reward_sum)
            log_running.append(running_reward)
            csv_writer.writerow([episode_number, reward_sum, f'{running_reward:.4f}'])
            csv_f.flush()

            print(f'Ep {episode_number} | Reward: {reward_sum:+.0f} | Running: {running_reward:.2f}')

            if episode_number % 50 == 0:
                save_checkpoint()
                save_plot()

            reward_sum = 0.0
            observation, info = env.reset()
            prev_x = None

except KeyboardInterrupt:
    print(f'\nInterrupted at episode {episode_number}')
    save_checkpoint()
    save_plot()
    csv_f.close()
    env.close()
