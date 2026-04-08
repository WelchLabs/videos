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

# Enable TF32 on CUDA for faster matmuls
if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

H = 200
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


env = gym.make("ALE/Breakout-v5", render_mode="human" if render else None)
observation, info = env.reset()
prev_x = None
log_probs, drs = [], []
reward_sum = 0

# Pre-allocate reusable tensors on device
zero_input = torch.zeros(D, dtype=torch.float32, device=device)

print(f"Starting training... H={H}, lr={learning_rate}, batch={batch_size}, device={device}")

try:
    while True:
        cur_x = prepro(observation)
        if prev_x is not None:
            x_t = torch.as_tensor(cur_x - prev_x, device=device)
        else:
            x_t = zero_input
        prev_x = cur_x

        logits = model(x_t)
        probs = F.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))

        observation, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        reward_sum += reward
        drs.append(reward)

        if done:
            episode_number += 1

            # Loss: scale all log probs by total episode reward (no discounting)
            total_reward = reward_sum
            loss = -(torch.stack(log_probs) * total_reward).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_probs, drs = [], []

            if running_reward is None:
                running_reward = reward_sum
            else:
                running_reward = 0.99 * running_reward + 0.01 * reward_sum

            is_best = running_reward > best_reward
            if is_best:
                best_reward = running_reward

            print(f'Ep {episode_number} | Reward: {reward_sum:+.0f} | Running: {running_reward:.2f} | Best: {best_reward:.2f} | Loss: {loss.item():.2f}')

            if episode_number % 50 == 0:
                save_checkpoint(save_file, is_best=is_best)
                print(f'Checkpoint saved at episode {episode_number}')
            elif is_best:
                save_checkpoint(save_file, is_best=True)
                print(f'New best model saved! Running reward: {running_reward:.2f}')

            reward_sum = 0
            observation, info = env.reset()
            prev_x = None

except KeyboardInterrupt:
    print(f'\nInterrupted! Saving checkpoint at episode {episode_number}...')
    save_checkpoint(save_file)
    print('Checkpoint saved. Exiting.')
