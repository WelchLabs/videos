import os
import pickle
import sys

import numpy as np
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

H = 200
D = 80 * 80
n_actions = 4

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, 'runs')


def find_best_checkpoint():
    for variant in ('karpathy', 'breakout'):
        vdir = os.path.join(RUNS_DIR, variant)
        if not os.path.isdir(vdir):
            continue
        for run in sorted(os.listdir(vdir), reverse=True):
            best = os.path.join(vdir, run, 'best_checkpoint.p')
            if os.path.exists(best):
                return best
            ckpt = os.path.join(vdir, run, 'checkpoint.p')
            if os.path.exists(ckpt):
                return ckpt
    return None


def load_checkpoint(path):
    data = pickle.load(open(path, 'rb'))
    m = data['model']
    if isinstance(m, dict) and 'W1' in m:
        return 'karpathy', m, data
    else:
        import torch
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, n_actions))
        model.load_state_dict(m)
        model.eval()
        return 'pytorch', model, data


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def act_karpathy(model, x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0
    logits = np.dot(model['W2'], h)
    return int(np.argmax(softmax(logits)))


def act_pytorch(model, x):
    import torch
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        return int(logits.argmax().item())


def prepro(I):
    I = I[32:192]
    I = I[::2, ::2, 0]
    I[I != 0] = 1
    return I.astype(np.float32).ravel()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find_best_checkpoint()
    if path is None:
        print("No checkpoint found. Train first.")
        return

    print(f"Loading: {path}")
    kind, model, data = load_checkpoint(path)
    ep = data.get('episode_number', '?')
    rr = data.get('running_reward', None)
    print(f"Type: {kind} | Episode: {ep} | Running reward: {rr:.2f if rr else '?'}")

    act = act_karpathy if kind == 'karpathy' else act_pytorch

    env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
    observation, _ = env.reset()
    prev_x = None
    episode = 0
    score = 0

    try:
        while True:
            cur_x = prepro(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(D, dtype=np.float32)
            prev_x = cur_x

            action = act(model, x)
            observation, reward, terminated, truncated, _ = env.step(action)
            score += reward

            if terminated or truncated:
                episode += 1
                print(f'Episode {episode} | Score: {score:.0f}')
                score = 0
                observation, _ = env.reset()
                prev_x = None

    except KeyboardInterrupt:
        print(f'\nStopped after {episode} episodes.')
        env.close()


if __name__ == '__main__':
    main()
