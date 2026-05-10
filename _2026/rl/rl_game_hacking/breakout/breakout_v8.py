# PyTorch policy gradient - custom pygame Breakout (no ALE, proper AABB collision)

import csv
import os
import pickle
from datetime import datetime

# must set before pygame import so SDL initializes headless
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import matplotlib
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = 200
learning_rate = 3e-4
action_repeat = 4
resume = True
render = False

# game renders at 160x160, downsampled to 80x80 for input
GAME_W, GAME_H = 160, 160
D = 80 * 80
n_actions = 3  # 0=noop, 1=left, 2=right
VERSION = 'v8'

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, 'runs', VERSION)
os.makedirs(RUNS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Custom Breakout environment — pure pygame rects, proper AABB collision
# ---------------------------------------------------------------------------

PADDLE_W, PADDLE_H = 32, 5
BALL_SIZE = 5
BRICK_ROWS, BRICK_COLS = 6, 8
BRICK_H = 8
BRICK_GAP = 2
BRICK_TOP = 24
BRICK_W = GAME_W // BRICK_COLS
PADDLE_SPEED = 6
BALL_SPEED_INIT = 3.0
LIVES = 3

BRICK_COLORS = [
    (200, 50,  50),
    (200, 120, 50),
    (200, 200, 50),
    (50,  200, 50),
    (50,  120, 200),
    (120, 50,  200),
]


class BreakoutEnv:
    def __init__(self, render_mode=None):
        pygame.init()
        if render_mode == 'human':
            self.screen = pygame.display.set_mode((GAME_W, GAME_H))
            pygame.display.set_caption('Breakout v8')
        else:
            self.screen = pygame.Surface((GAME_W, GAME_H))
        self.render_mode = render_mode
        self.clock = pygame.time.Clock()

    def _ball_rect(self):
        return pygame.Rect(int(self.bx), int(self.by), BALL_SIZE, BALL_SIZE)

    def reset(self):
        self.paddle = pygame.Rect(GAME_W // 2 - PADDLE_W // 2,
                                  GAME_H - 18, PADDLE_W, PADDLE_H)
        self.bx = float(GAME_W // 2 - BALL_SIZE // 2)
        self.by = float(self.paddle.top - BALL_SIZE - 1)
        angle = np.random.uniform(-0.6, 0.6)
        self.ball_vx = BALL_SPEED_INIT * np.sin(angle)
        self.ball_vy = -BALL_SPEED_INIT * np.cos(angle)
        self.bricks = []
        for row in range(BRICK_ROWS):
            for col in range(BRICK_COLS):
                x = col * BRICK_W
                y = BRICK_TOP + row * (BRICK_H + BRICK_GAP)
                self.bricks.append(pygame.Rect(x, y, BRICK_W - BRICK_GAP, BRICK_H))
        self.lives = LIVES
        self.score = 0
        return self._obs(), {}

    def step(self, action):
        if action == 1:
            self.paddle.x = max(0, self.paddle.x - PADDLE_SPEED)
        elif action == 2:
            self.paddle.x = min(GAME_W - PADDLE_W, self.paddle.x + PADDLE_SPEED)

        reward = 0.0
        self.bx += self.ball_vx
        self.by += self.ball_vy

        # wall collisions (left/right/top)
        if self.bx <= 0:
            self.bx = 0
            self.ball_vx = abs(self.ball_vx)
        elif self.bx + BALL_SIZE >= GAME_W:
            self.bx = GAME_W - BALL_SIZE
            self.ball_vx = -abs(self.ball_vx)
        if self.by <= 0:
            self.by = 0
            self.ball_vy = abs(self.ball_vy)

        ball = self._ball_rect()

        # paddle collision
        if self.ball_vy > 0 and ball.colliderect(self.paddle):
            self.by = self.paddle.top - BALL_SIZE
            hit_frac = (ball.centerx - self.paddle.left) / PADDLE_W
            self.ball_vx = (hit_frac - 0.5) * BALL_SPEED_INIT * 2.5
            speed = max(BALL_SPEED_INIT, (self.ball_vx ** 2 + self.ball_vy ** 2) ** 0.5)
            self.ball_vy = -abs(speed - abs(self.ball_vx) * 0.3)
            ball = self._ball_rect()

        # brick collisions — AABB with side detection to avoid phasing
        for brick in list(self.bricks):
            if not ball.colliderect(brick):
                continue
            self.bricks.remove(brick)
            reward += 1.0
            self.score += 1
            ol = ball.right - brick.left
            or_ = brick.right - ball.left
            ot = ball.bottom - brick.top
            ob = brick.bottom - ball.top
            if min(ot, ob) <= min(ol, or_):
                self.ball_vy = -self.ball_vy
                if ot < ob:
                    self.by = brick.top - BALL_SIZE
                else:
                    self.by = brick.bottom
            else:
                self.ball_vx = -self.ball_vx
                if ol < or_:
                    self.bx = brick.left - BALL_SIZE
                else:
                    self.bx = brick.right
            ball = self._ball_rect()

        # ball out of bounds
        terminated = False
        if self.by > GAME_H:
            self.lives -= 1
            if self.lives <= 0:
                terminated = True
            else:
                self.bx = float(GAME_W // 2 - BALL_SIZE // 2)
                self.by = float(self.paddle.top - BALL_SIZE - 1)
                angle = np.random.uniform(-0.6, 0.6)
                self.ball_vx = BALL_SPEED_INIT * np.sin(angle)
                self.ball_vy = -BALL_SPEED_INIT * np.cos(angle)

        if not self.bricks:
            terminated = True

        if self.render_mode == 'human':
            self._render()

        return self._obs(), reward, terminated, False, {}

    def _render(self):
        self.screen.fill((0, 0, 0))
        for brick in self.bricks:
            row = next((r for r in range(BRICK_ROWS)
                        if brick.y == BRICK_TOP + r * (BRICK_H + BRICK_GAP)), 0)
            pygame.draw.rect(self.screen, BRICK_COLORS[row % len(BRICK_COLORS)], brick)
        pygame.draw.rect(self.screen, (200, 200, 200), self.paddle)
        pygame.draw.rect(self.screen, (255, 255, 255), self._ball_rect())
        pygame.display.flip()
        self.clock.tick(60)

    def _obs(self):
        self.screen.fill((0, 0, 0))
        for brick in self.bricks:
            pygame.draw.rect(self.screen, (255, 255, 255), brick)
        pygame.draw.rect(self.screen, (255, 255, 255), self.paddle)
        pygame.draw.rect(self.screen, (255, 255, 255), self._ball_rect())
        raw = pygame.surfarray.array2d(self.screen)  # (W, H) uint32
        return raw.T.astype(np.float32)  # (H, W)

    def close(self):
        pygame.quit()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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


def prepro(I):
    # I is (160, 160) float, downsample to 80x80 and binarize
    I = I[::2, ::2]
    I = (I > 0).astype(np.float32)
    return I.ravel()


def step_repeat(env, action):
    total_reward = 0.0
    frames = []
    terminated = truncated = False

    for _ in range(action_repeat):
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frames.append(observation)
        if terminated or truncated:
            break

    observation = np.maximum(frames[-1], frames[-2]) if len(frames) >= 2 else frames[-1]
    return observation, total_reward, terminated, truncated, {}


def save_plot():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(log_episodes, log_rewards, alpha=0.3, color='teal', label='reward')
    ax.plot(log_episodes, log_running, color='teal', linewidth=2, label='running reward')
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    ax.set_title('v8 - custom pygame Breakout, Adam, raw return baseline')
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
        'env_id': 'custom-breakout-pygame',
    }
    pickle.dump(data, open(checkpoint_file, 'wb'))
    if is_best:
        pickle.dump(data, open(best_checkpoint_file, 'wb'))


env = BreakoutEnv(render_mode='human' if render else None)
observation, info = env.reset()
prev_x = None
log_probs = []
reward_sum = 0.0

print(f"H={H}, lr={learning_rate}, Adam, raw return baseline, repeat={action_repeat}, custom pygame Breakout")

try:
    while True:
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D, dtype=np.float32)
        prev_x = cur_x

        logits = model(torch.from_numpy(x))
        dist = torch.distributions.Categorical(F.softmax(logits, dim=0))
        action = dist.sample()
        log_probs.append(dist.log_prob(action))

        observation, reward, terminated, truncated, info = step_repeat(env, action.item())
        reward_sum += reward

        if terminated or truncated:
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
            observation, info = env.reset()
            prev_x = None

except KeyboardInterrupt:
    print(f'\nInterrupted at episode {episode_number}')
    save_checkpoint()
    save_plot()
    csv_f.close()
    env.close()
