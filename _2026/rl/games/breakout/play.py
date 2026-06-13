import argparse

import numpy as np
import pygame

from breakout import Breakout, RecordWrapper, LEFT, RIGHT, NOOP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", metavar="PATH")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--scale", type=int, default=3)
    args = ap.parse_args()

    base = Breakout(render_scale=args.scale)
    env = RecordWrapper(base, args.record, fps=args.fps) if args.record else base

    obs, info = env.reset()

    pygame.init()
    frame = base.render()
    h, w = frame.shape[:2]
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Breakout")
    clock = pygame.time.Clock()

    running = True
    try:
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action = LEFT
            elif keys[pygame.K_RIGHT]:
                action = RIGHT
            else:
                action = NOOP

            obs, reward, terminated, truncated, info = env.step(action)

            frame = base.render()
            pygame.surfarray.blit_array(screen, np.transpose(frame, (1, 0, 2)))
            pygame.display.set_caption(f"Breakout  score={info['score']:.0f}  lives={info['lives']}")
            pygame.display.flip()
            clock.tick(args.fps)

            if terminated or truncated:
                print(f"game over  score={info['score']:.0f}")
                obs, info = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
