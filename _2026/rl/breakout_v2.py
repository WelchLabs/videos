import numpy as np
import pickle
import gymnasium as gym
import ale_py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

gym.register_envs(ale_py)

# Auto-detect best available device
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

H = 200
num_envs = 32
batch_size = 10
learning_rate = 1e-4
resume = True
render = False

D = 80 * 80
n_actions = 4
save_file = 'breakout_v2.p'
best_save_file = 'breakout_v2_best.p'

model = nn.Sequential(
    nn.Linear(D, H),
    nn.ReLU(),
    nn.Linear(H, n_actions)
).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

running_reward = None
episode_number = 0
best_reward = 0

if resume and os.path.exists(save_file):
    print(f"Loading model from {save_file}")
    checkpoint = pickle.load(open(save_file, 'rb'))
    model.load_state_dict(checkpoint['model'])
    running_reward = checkpoint['running_reward']
    episode_number = checkpoint['episode_number']
    best_reward = checkpoint.get('best_reward', 0)
    print(f"Resumed at episode {episode_number}, running reward: {running_reward:.2f}")


def prepro(I):
    I = I[32:192]
    I = I[::2, ::2, 0]
    I[I != 0] = 1
    return I.astype(np.float32).ravel()


def save_checkpoint(filename, is_best=False):
    checkpoint = {
        'model': {k: v.cpu() for k, v in model.state_dict().items()},
        'running_reward': running_reward,
        'episode_number': episode_number,
        'best_reward': best_reward,
    }
    pickle.dump(checkpoint, open(filename, 'wb'))
    if is_best:
        pickle.dump(checkpoint, open(best_save_file, 'wb'))


envs = gym.make_vec("ALE/Breakout-v5",
                    num_envs=num_envs,
                    vectorization_mode="async",
                    render_mode="human" if render else None)

observations, infos = envs.reset()
prev_x = [None] * num_envs
# Per-env episode buffers
env_log_probs = [[] for _ in range(num_envs)]
env_rewards = [0.0] * num_envs

# Accumulate episode losses across a batch before stepping optimizer
batch_losses = []
episodes_in_batch = 0

zero_input = np.zeros(D, dtype=np.float32)

print(f"Starting training... H={H}, lr={learning_rate}, batch={batch_size}, num_envs={num_envs}, device={device}")

try:
    while True:
        # Preprocess all observations
        cur_xs = [prepro(observations[i]) for i in range(num_envs)]
        diffs = []
        for i in range(num_envs):
            if prev_x[i] is not None:
                diffs.append(cur_xs[i] - prev_x[i])
            else:
                diffs.append(zero_input)
            prev_x[i] = cur_xs[i]

        # Batched forward pass
        x_batch = torch.as_tensor(np.stack(diffs), device=device)
        logits = model(x_batch)
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs_batch = dist.log_prob(actions)

        # Store per-env log probs
        for i in range(num_envs):
            env_log_probs[i].append(log_probs_batch[i])

        # Step all envs at once
        observations, rewards, terminateds, truncateds, infos = envs.step(actions.cpu().numpy())
        dones = terminateds | truncateds

        for i in range(num_envs):
            env_rewards[i] += rewards[i]

        # Handle finished episodes
        for i in range(num_envs):
            if not dones[i]:
                continue

            episode_number += 1

            # Loss: scale all log probs by total episode reward (no discounting)
            total_reward = env_rewards[i]
            ep_log_probs = torch.stack(env_log_probs[i])
            loss = -(ep_log_probs * total_reward).sum()
            batch_losses.append(loss)
            episodes_in_batch += 1

            if running_reward is None:
                running_reward = env_rewards[i]
            else:
                running_reward = 0.99 * running_reward + 0.01 * env_rewards[i]

            is_best = running_reward > best_reward
            if is_best:
                best_reward = running_reward

            print(f'Ep {episode_number} | Reward: {env_rewards[i]:+.0f} | Running: {running_reward:.2f} | Best: {best_reward:.2f}')

            if episode_number % 50 == 0:
                save_checkpoint(save_file, is_best=is_best)
                print(f'Checkpoint saved at episode {episode_number}')
            elif is_best:
                save_checkpoint(save_file, is_best=True)
                print(f'New best model saved! Running reward: {running_reward:.2f}')

            # Reset this env's buffers
            env_log_probs[i] = []
            env_rewards[i] = 0.0
            prev_x[i] = None

        # Update weights after batch_size episodes complete
        if episodes_in_batch >= batch_size:
            total_loss = torch.stack(batch_losses).sum()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            batch_losses = []
            episodes_in_batch = 0

except KeyboardInterrupt:
    print(f'\nInterrupted! Saving checkpoint at episode {episode_number}...')
    save_checkpoint(save_file)
    print('Checkpoint saved. Exiting.')
