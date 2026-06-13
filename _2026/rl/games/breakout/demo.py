import argparse

import numpy as np

from breakout import Breakout, RecordWrapper, LEFT, RIGHT, NOOP


def random_agent(rng):
    def pick(obs, info):
        return int(rng.integers(0, 3))
    return pick


def tracker_agent():
    def pick(obs, info):
        ball_x, _, _, _, paddle_x = info["state"]
        if ball_x < paddle_x - 0.02:
            return LEFT
        if ball_x > paddle_x + 0.02:
            return RIGHT
        return NOOP
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["random", "tracker"], default="random")
    ap.add_argument("--out", default="demo.mp4")
    ap.add_argument("-n", "--episodes", type=int, default=1)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pick = random_agent(rng) if args.agent == "random" else tracker_agent()

    env = RecordWrapper(Breakout(seed=args.seed), args.out, fps=args.fps)
    for ep in range(args.episodes):
        obs, info = env.reset()
        total = 0.0
        while True:
            obs, r, term, trunc, info = env.step(pick(obs, info))
            total += r
            if term or trunc:
                break
        print(f"episode {ep + 1}  reward={total:.0f}  lives_left={info['lives']}")
    env.close()


if __name__ == "__main__":
    main()
