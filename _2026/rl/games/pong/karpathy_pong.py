import csv
import os
import pickle
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

H = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99
resume = True
render = False

D = 80 * 80

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, 'runs', 'karpathy')
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
best_checkpoint_file = os.path.join(run_dir, 'best_checkpoint.p')
log_file = os.path.join(run_dir, 'log.csv')
plot_file = os.path.join(run_dir, 'plot.png')

if resume and os.path.exists(checkpoint_file):
    data = pickle.load(open(checkpoint_file, 'rb'))
    model = data['model']
    rmsprop_cache = data['rmsprop_cache']
    running_reward = data['running_reward']
    episode_number = data['episode_number']
    best_reward = data.get('best_reward', -21)
    print(f"Resumed ep {episode_number}, running reward: {running_reward:.2f}")
else:
    model = {
        'W1': np.random.randn(H, D) / np.sqrt(D),
        'W2': np.random.randn(H) / np.sqrt(H),
    }
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}
    running_reward = None
    episode_number = 0
    best_reward = -21

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}

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
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float64).ravel()


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h


def policy_backward(eph, epdlogp, epx):
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


def discount_rewards(r):
    discounted = np.zeros_like(r)
    running_add = 0.0
    for t in reversed(range(len(r))):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted[t] = running_add
    return discounted


def save_plot():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(log_episodes, log_rewards, alpha=0.3, color='steelblue', label='reward')
    ax.plot(log_episodes, log_running, color='steelblue', linewidth=2, label='running reward')
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    ax.set_title('Karpathy Pong — numpy RMSProp + discounted per-step returns')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_file, dpi=120)
    plt.close(fig)


def save_checkpoint(is_best=False):
    data = {
        'model': model,
        'rmsprop_cache': rmsprop_cache,
        'running_reward': running_reward,
        'episode_number': episode_number,
        'best_reward': best_reward,
    }
    pickle.dump(data, open(checkpoint_file, 'wb'))
    if is_best:
        pickle.dump(data, open(best_checkpoint_file, 'wb'))


env = gym.make('PongNoFrameskip-v4', render_mode='human' if render else None)
observation, _ = env.reset()
prev_x = None
xs, hs, dlogps, drs = [], [], [], []
reward_sum = 0

print(f"H={H}, D={D}, lr={learning_rate}, batch={batch_size}, gamma={gamma}, RMSProp+discount")

try:
    while True:
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        aprob, h = policy_forward(x)
        action = 2 if np.random.uniform() < aprob else 3

        xs.append(x)
        hs.append(h)
        y = 1 if action == 2 else 0
        dlogps.append(y - aprob)

        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward_sum += reward
        drs.append(reward)

        if done:
            episode_number += 1

            epx = np.array(xs, dtype=np.float32)
            eph = np.array(hs, dtype=np.float32)
            epdlogp = np.array(dlogps, dtype=np.float32).ravel()
            epr = np.array(drs, dtype=np.float32).ravel()
            xs, hs, dlogps, drs = [], [], [], []

            discounted_epr = discount_rewards(epr)
            discounted_epr -= discounted_epr.mean()
            discounted_epr /= discounted_epr.std() + 1e-8

            epdlogp *= discounted_epr
            grad = policy_backward(eph, epdlogp, epx)
            for k in model:
                grad_buffer[k] += grad[k]

            if episode_number % batch_size == 0:
                for k, v in model.items():
                    g = grad_buffer[k]
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v)

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
            observation, _ = env.reset()
            prev_x = None

except KeyboardInterrupt:
    print(f'\nInterrupted at episode {episode_number}')
    save_checkpoint()
    save_plot()
    csv_f.close()
    env.close()
