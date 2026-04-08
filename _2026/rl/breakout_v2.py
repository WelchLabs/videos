import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py
import pickle
import os

gym.register_envs(ale_py)

H = 200
n_envs = 8
learning_rate = 1e-4
render = False

D = 80 * 80
n_actions = 4
save_file = 'breakout_v2.p'
best_save_file = 'breakout_v2_best.p'

device = torch.device("mps") if torch.backends.mps.is_available() else \
         torch.device("cuda") if torch.cuda.is_available() else \
         torch.device("cpu")
print(f"Using device: {device}")

# Model
model = nn.Sequential(
    nn.Linear(D, H),
    nn.ReLU(),
    nn.Linear(H, n_actions)
).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

running_reward = None
episode_number = 0
best_reward = 0


def prepro(I):
    I = I[32:192]
    I = I[::2, ::2, 0]
    I[I != 0] = 1
    return I.astype(np.float32).ravel()


# Parallel environments
envs = [gym.make("ALE/Breakout-v5") for _ in range(n_envs)]
observations = [env.reset()[0] for env in envs]
prev_xs = [None] * n_envs

# Per-env buffers: store log probs and raw rewards
buffers = [{'log_probs': [], 'rewards': []} for _ in range(n_envs)]
reward_sums = [0.0] * n_envs

print(f"Starting training... H={H}, lr={learning_rate}, n_envs={n_envs}, device={device}")
print("No discounting, no RMSProp — plain SGD on raw episode reward")

try:
    while True:
        # Build input batch
        cur_xs = [prepro(observations[i]) for i in range(n_envs)]
        xs = np.stack([
            cur_xs[i] - prev_xs[i] if prev_xs[i] is not None else np.zeros(D, dtype=np.float32)
            for i in range(n_envs)
        ])
        xs_t = torch.tensor(xs, device=device)

        # Forward pass
        logits = model(xs_t)                              # (n_envs, n_actions)
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()                           # (n_envs,)
        log_probs = dist.log_prob(actions)                # (n_envs,)

        for i in range(n_envs):
            prev_xs[i] = cur_xs[i]

            buffers[i]['log_probs'].append(log_probs[i])

            obs, reward, terminated, truncated, info = envs[i].step(actions[i].item())
            done = terminated or truncated
            reward_sums[i] += reward
            buffers[i]['rewards'].append(reward)
            observations[i] = obs

            if done:
                episode_number += 1

                # Loss: scale all log probs by total episode reward (no discounting)
                total_reward = reward_sums[i]
                ep_log_probs = torch.stack(buffers[i]['log_probs'])
                loss = -(ep_log_probs * total_reward).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                buffers[i] = {'log_probs': [], 'rewards': []}

                if running_reward is None:
                    running_reward = reward_sums[i]
                else:
                    running_reward = 0.99 * running_reward + 0.01 * reward_sums[i]

                is_best = running_reward > best_reward
                if is_best:
                    best_reward = running_reward

                print(f'Ep {episode_number} | Reward: {reward_sums[i]:+.0f} | Running: {running_reward:.2f} | Best: {best_reward:.2f} | Loss: {loss.item():.2f}')

                if episode_number % 50 == 0 or is_best:
                    checkpoint = {'model': model.state_dict(), 'running_reward': running_reward, 'episode_number': episode_number}
                    pickle.dump(checkpoint, open(save_file, 'wb'))
                    if is_best:
                        pickle.dump(checkpoint, open(best_save_file, 'wb'))
                        print(f'New best model saved! Running reward: {running_reward:.2f}')
                    else:
                        print(f'Checkpoint saved at episode {episode_number}')

                reward_sums[i] = 0.0
                observations[i] = envs[i].reset()[0]
                prev_xs[i] = None

except KeyboardInterrupt:
    print(f'\nInterrupted at episode {episode_number}')
    checkpoint = {'model': model.state_dict(), 'running_reward': running_reward, 'episode_number': episode_number, 'best_reward': best_reward}
    pickle.dump(checkpoint, open(save_file, 'wb'))
    print('Checkpoint saved.')
finally:
    for env in envs:
        env.close()
