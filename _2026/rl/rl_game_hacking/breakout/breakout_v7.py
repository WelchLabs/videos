# PyTorch policy gradient - MinAtar Breakout (proper collision, no ROM phasing bug)

import csv
import os
import pickle
from datetime import datetime

import matplotlib
import minatar
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = 200
learning_rate = 3e-4
action_repeat = 4
resume = True

# MinAtar Breakout: 10x10x4 state, minimal actions [0=noop, 1=left, 3=right]
D = 10 * 10 * 4
MINIMAL_ACTIONS = [0, 1, 3]
n_actions = len(MINIMAL_ACTIONS)
VERSION = 'v7'

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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

running_reward = None
episode_number = 0
best_reward = 0

if resume and os.path.exists(checkpoint_file):
    checkpoint = pickle.load(open(checkpoint_file, 'rb'))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
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


def prepro(state):
    return state.astype(np.float32).ravel()


def step_repeat(env, action):
    total_reward = 0
    frames = []
    terminated = False

    for _ in range(action_repeat):
        reward, terminated = env.act(MINIMAL_ACTIONS[action])
        total_reward += reward
        frames.append(env.state().copy())
        if terminated:
            break

    # max-pool last 2 frames so ball stays visible at any position
    obs = np.maximum(frames[-1], frames[-2]) if len(frames) >= 2 else frames[-1]
    return obs, total_reward, terminated


def save_plot():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(log_episodes, log_rewards, alpha=0.3, color='teal', label='reward')
    ax.plot(log_episodes, log_running, color='teal', linewidth=2, label='running reward')
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    ax.set_title('v7 - MinAtar Breakout, Adam, raw return baseline')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_file, dpi=120)
    plt.close(fig)


def save_checkpoint(is_best=False):
    data = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'running_reward': running_reward,
        'episode_number': episode_number,
        'best_reward': best_reward,
        'env_id': 'minatar-breakout',
    }
    pickle.dump(data, open(checkpoint_file, 'wb'))
    if is_best:
        pickle.dump(data, open(best_checkpoint_file, 'wb'))


env = minatar.Environment('breakout')
env.reset()
observation = env.state().copy()
prev_x = None
log_probs = []
reward_sum = 0.0

print(f"H={H}, lr={learning_rate}, Adam, raw return baseline, repeat={action_repeat}, MinAtar Breakout")

try:
    while True:
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D, dtype=np.float32)
        prev_x = cur_x

        logits = model(torch.from_numpy(x))
        dist = torch.distributions.Categorical(F.softmax(logits, dim=0))
        action_idx = dist.sample()
        log_probs.append(dist.log_prob(action_idx))

        observation, reward, terminated = step_repeat(env, action_idx.item())
        reward_sum += reward

        if terminated:
            episode_number += 1

            baseline = 0.0 if running_reward is None else running_reward
            loss = -torch.stack(log_probs).sum() * (reward_sum - baseline)
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

            reward_sum = 0.0
            env.reset()
            observation = env.state().copy()
            prev_x = None

except KeyboardInterrupt:
    print(f'\nInterrupted at episode {episode_number}')
    save_checkpoint()
    save_plot()
    csv_f.close()
