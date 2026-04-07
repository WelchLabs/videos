import numpy as np
import pickle
import gymnasium as gym
import ale_py
import os
import torch
import torch.nn.functional as F

gym.register_envs(ale_py)

H = 200
n_envs = 8           # parallel environments — main speedup
batch_size = 10      # gradient update every N episodes (per env)
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99
resume = False
render = False

D = 80 * 80
n_actions = 4
save_file = 'breakout_model_gpu.p'
best_save_file = 'breakout_model_gpu_best.p'

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

if resume and os.path.exists(save_file):
    print(f"Loading model from {save_file}")
    checkpoint = pickle.load(open(save_file, 'rb'))
    W1 = torch.tensor(checkpoint['W1'], dtype=torch.float32, device=device)
    W2 = torch.tensor(checkpoint['W2'], dtype=torch.float32, device=device)
    rmsprop_cache_W1 = torch.tensor(checkpoint['rms_W1'], dtype=torch.float32, device=device)
    rmsprop_cache_W2 = torch.tensor(checkpoint['rms_W2'], dtype=torch.float32, device=device)
    running_reward = checkpoint['running_reward']
    episode_number = checkpoint['episode_number']
    best_reward = checkpoint.get('best_reward', 0)
    print(f"Resumed at episode {episode_number}, running reward: {running_reward:.2f}")
else:
    W1 = torch.randn(H, D, device=device) / np.sqrt(D)
    W2 = torch.randn(n_actions, H, device=device) / np.sqrt(H)
    rmsprop_cache_W1 = torch.zeros_like(W1)
    rmsprop_cache_W2 = torch.zeros_like(W2)
    running_reward = None
    episode_number = 0
    best_reward = 0

grad_buffer_W1 = torch.zeros_like(W1)
grad_buffer_W2 = torch.zeros_like(W2)


def prepro(I):
    I = I[32:192]
    I = I[::2, ::2, 0]
    I[I != 0] = 1
    return I.astype(np.float32).ravel()


def discount_rewards(r):
    # r is a list of rewards for one episode
    r = np.array(r, dtype=np.float32)
    discounted = np.zeros_like(r)
    running_add = 0.0
    for t in reversed(range(len(r))):
        if r[t] != 0:
            running_add = 0  # reset on life loss
        running_add = running_add * gamma + r[t]
        discounted[t] = running_add
    return discounted


def policy_forward_batch(xs):
    # xs: (N, D) tensor
    h = torch.mm(xs, W1.T)       # (N, H)
    h = torch.clamp(h, min=0)    # ReLU
    logits = torch.mm(h, W2.T)   # (N, n_actions)
    probs = F.softmax(logits, dim=1)
    return probs, h


def policy_backward_batch(eph, epdlogp, epx):
    # eph: (T, H), epdlogp: (T, n_actions), epx: (T, D)
    dW2 = torch.mm(epdlogp.T, eph)      # (n_actions, H)
    dh = torch.mm(epdlogp, W2)          # (T, H)
    dh[eph <= 0] = 0                     # ReLU backprop
    dW1 = torch.mm(dh.T, epx)           # (H, D)
    return dW1, dW2


def rmsprop_update(W, grad, cache):
    cache = decay_rate * cache + (1 - decay_rate) * grad ** 2
    W = W + learning_rate * grad / (torch.sqrt(cache) + 1e-5)
    return W, cache


def save_checkpoint(filename, is_best=False):
    checkpoint = {
        'W1': W1.cpu().numpy(),
        'W2': W2.cpu().numpy(),
        'rms_W1': rmsprop_cache_W1.cpu().numpy(),
        'rms_W2': rmsprop_cache_W2.cpu().numpy(),
        'running_reward': running_reward,
        'episode_number': episode_number,
        'best_reward': best_reward,
    }
    pickle.dump(checkpoint, open(filename, 'wb'))
    if is_best:
        pickle.dump(checkpoint, open(best_save_file, 'wb'))


# Parallel environments
envs = [gym.make("ALE/Breakout-v5") for _ in range(n_envs)]
observations = [env.reset()[0] for env in envs]
prev_xs = [None] * n_envs

# Per-env episode buffers
buffers = [{'xs': [], 'hs': [], 'dlogps': [], 'drs': []} for _ in range(n_envs)]
reward_sums = [0.0] * n_envs

print(f"Starting training... H={H}, lr={learning_rate}, batch={batch_size}, n_envs={n_envs}, device={device}")

try:
    while True:
        # Build input batch for all envs
        cur_xs = [prepro(observations[i]) for i in range(n_envs)]
        xs = np.stack([
            cur_xs[i] - prev_xs[i] if prev_xs[i] is not None else np.zeros(D, dtype=np.float32)
            for i in range(n_envs)
        ])  # (n_envs, D)

        xs_t = torch.tensor(xs, dtype=torch.float32, device=device)

        # Single batched forward pass for all envs
        with torch.no_grad():
            aprobs, hs = policy_forward_batch(xs_t)  # (n_envs, n_actions), (n_envs, H)

        aprobs_np = aprobs.cpu().numpy()

        for i in range(n_envs):
            prev_xs[i] = cur_xs[i]

            action = np.random.choice(n_actions, p=aprobs_np[i])

            buffers[i]['xs'].append(xs[i])
            buffers[i]['hs'].append(hs[i].cpu().numpy())

            dlogp = -aprobs_np[i].copy()
            dlogp[action] += 1.0
            buffers[i]['dlogps'].append(dlogp)

            obs, reward, terminated, truncated, info = envs[i].step(action)
            done = terminated or truncated
            reward_sums[i] += reward
            buffers[i]['drs'].append(reward)
            observations[i] = obs

            if done:
                episode_number += 1

                epx = torch.tensor(np.array(buffers[i]['xs'], dtype=np.float32), device=device)
                eph = torch.tensor(np.array(buffers[i]['hs'], dtype=np.float32), device=device)
                epdlogp = torch.tensor(np.array(buffers[i]['dlogps'], dtype=np.float32), device=device)
                epr = np.array(buffers[i]['drs'], dtype=np.float32)

                buffers[i] = {'xs': [], 'hs': [], 'dlogps': [], 'drs': []}

                discounted = discount_rewards(epr)
                discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)
                discounted_t = torch.tensor(discounted, dtype=torch.float32, device=device)

                epdlogp *= discounted_t[:, None]
                dW1, dW2 = policy_backward_batch(eph, epdlogp, epx)

                grad_buffer_W1 += dW1
                grad_buffer_W2 += dW2

                if episode_number % batch_size == 0:
                    W1, rmsprop_cache_W1 = rmsprop_update(W1, grad_buffer_W1, rmsprop_cache_W1)
                    W2, rmsprop_cache_W2 = rmsprop_update(W2, grad_buffer_W2, rmsprop_cache_W2)
                    grad_buffer_W1.zero_()
                    grad_buffer_W2.zero_()

                if running_reward is None:
                    running_reward = reward_sums[i]
                else:
                    running_reward = 0.99 * running_reward + 0.01 * reward_sums[i]

                is_best = running_reward > best_reward
                if is_best:
                    best_reward = running_reward

                print(f'Ep {episode_number} | Reward: {reward_sums[i]:+.0f} | Running: {running_reward:.2f} | Best: {best_reward:.2f}')

                if episode_number % 50 == 0:
                    save_checkpoint(save_file, is_best=is_best)
                    print(f'Checkpoint saved at episode {episode_number}')
                elif is_best:
                    save_checkpoint(save_file, is_best=True)
                    print(f'New best model saved! Running reward: {running_reward:.2f}')

                reward_sums[i] = 0.0
                observations[i] = envs[i].reset()[0]
                prev_xs[i] = None

except KeyboardInterrupt:
    print(f'\nInterrupted! Saving checkpoint at episode {episode_number}...')
    save_checkpoint(save_file)
    print('Checkpoint saved. Exiting.')
finally:
    for env in envs:
        env.close()
