import numpy as np

NOOP, LEFT, RIGHT = 0, 1, 2

RED = [200, 72, 72]
ORANGE = [198, 108, 58]
GREEN = [72, 160, 72]
YELLOW = [184, 172, 48]
ROW_COLORS = np.array([RED, RED, ORANGE, ORANGE, GREEN, GREEN, YELLOW, YELLOW], dtype=np.uint8)
ROW_POINTS = [7, 7, 5, 5, 3, 3, 1, 1]

BG = np.array([0, 0, 0], dtype=np.uint8)
PADDLE_COLOR = np.array([200, 72, 72], dtype=np.uint8)
BALL_COLOR = np.array([236, 236, 236], dtype=np.uint8)
WALL_COLOR = np.array([142, 142, 142], dtype=np.uint8)
SCORE_COLOR = np.array([236, 236, 236], dtype=np.uint8)

DIGITS = {
    0: ["111", "101", "101", "101", "111"],
    1: ["110", "010", "010", "010", "111"],
    2: ["111", "001", "111", "100", "111"],
    3: ["111", "001", "111", "001", "111"],
    4: ["101", "101", "111", "001", "001"],
    5: ["111", "100", "111", "001", "111"],
    6: ["111", "100", "111", "101", "111"],
    7: ["111", "001", "001", "001", "001"],
    8: ["111", "101", "111", "101", "111"],
    9: ["111", "101", "111", "001", "111"],
}


class Breakout:
    n_actions = 3

    def __init__(
        self,
        width=192,
        height=144,
        brick_rows=8,
        brick_cols=14,
        brick_top=32,
        brick_height=5,
        lives=5,
        ball_speed=2.0,
        paddle_speed=4.0,
        paddle_width=26,
        paddle_height=3,
        ball_size=2,
        wall=5,
        obs_size=84,
        render_scale=4,
        max_steps=10000,
        seed=None,
    ):
        self.W, self.H = width, height
        self.brick_rows, self.brick_cols = brick_rows, brick_cols
        self.brick_top, self.brick_h = brick_top, brick_height
        self.start_lives = lives
        self.ball_speed = ball_speed
        self.paddle_speed = paddle_speed
        self.full_pw, self.ph = paddle_width, paddle_height
        self.bs = ball_size
        self.wall = wall
        self.obs_size = obs_size
        self.render_scale = render_scale
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.brick_w = (self.W - 2 * wall) / brick_cols
        self.paddle_y = self.H - 14

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.lives = self.start_lives
        self.score = 0.0
        self.steps = 0
        self.bricks = np.ones((self.brick_rows, self.brick_cols), dtype=bool)
        self.pw = self.full_pw
        self.shrunk = False
        self.paddle_x = (self.W - self.pw) / 2.0
        self._launch_ball()
        return self._observe(), self._info()

    def _launch_ball(self):
        self.ball_x = self.paddle_x + self.pw / 2.0
        self.ball_y = self.paddle_y - 12.0
        angle = self.rng.uniform(-0.6, 0.6)
        self.ball_vx = self.ball_speed * np.sin(angle)
        self.ball_vy = -self.ball_speed * np.cos(angle)

    def step(self, action):
        self.steps += 1
        reward = 0.0

        if action == LEFT:
            self.paddle_x -= self.paddle_speed
        elif action == RIGHT:
            self.paddle_x += self.paddle_speed
        self.paddle_x = float(np.clip(self.paddle_x, self.wall, self.W - self.wall - self.pw))

        speed = np.hypot(self.ball_vx, self.ball_vy)
        n_sub = max(1, int(np.ceil(speed)))
        dx, dy = self.ball_vx / n_sub, self.ball_vy / n_sub
        lost = False
        for _ in range(n_sub):
            self.ball_x += dx
            self.ball_y += dy
            reward += self._collide()
            if self.ball_y - self.bs > self.H:
                lost = True
                break

        terminated = False
        if lost:
            self.lives -= 1
            if self.lives <= 0:
                terminated = True
            else:
                self._launch_ball()
        elif not self.bricks.any():
            terminated = True

        truncated = self.steps >= self.max_steps
        self.score += reward
        return self._observe(), reward, terminated, truncated, self._info()

    def _collide(self):
        reward = 0.0
        r = self.bs

        if self.ball_x - r < self.wall:
            self.ball_x = self.wall + r
            self.ball_vx = abs(self.ball_vx)
        elif self.ball_x + r > self.W - self.wall:
            self.ball_x = self.W - self.wall - r
            self.ball_vx = -abs(self.ball_vx)
        if self.ball_y - r < self.wall:
            self.ball_y = self.wall + r
            self.ball_vy = abs(self.ball_vy)
            if not self.shrunk:
                self.shrunk = True
                self.pw = self.full_pw // 2

        if (
            self.ball_vy > 0
            and self.paddle_y - r < self.ball_y < self.paddle_y + self.ph + r
            and self.paddle_x - r < self.ball_x < self.paddle_x + self.pw + r
        ):
            self.ball_y = self.paddle_y - r
            offset = (self.ball_x - (self.paddle_x + self.pw / 2.0)) / (self.pw / 2.0)
            offset = float(np.clip(offset, -1, 1))
            angle = offset * 1.0
            self.ball_vx = self.ball_speed * np.sin(angle)
            self.ball_vy = -self.ball_speed * np.cos(angle)

        col = int((self.ball_x - self.wall) // self.brick_w)
        row = int((self.ball_y - self.brick_top) // self.brick_h)
        if (
            0 <= row < self.brick_rows
            and 0 <= col < self.brick_cols
            and self.bricks[row, col]
        ):
            self.bricks[row, col] = False
            reward += ROW_POINTS[row]
            self.ball_vy = -self.ball_vy
        return reward

    def _observe(self):
        frame = self._draw(hud=False)
        if self.obs_size and (self.W != self.obs_size or self.H != self.obs_size):
            frame = _resize(frame, self.obs_size, self.obs_size)
        return frame

    def _info(self):
        state = np.array([
            self.ball_x / self.W * 2 - 1,
            self.ball_y / self.H * 2 - 1,
            self.ball_vx / self.ball_speed,
            self.ball_vy / self.ball_speed,
            (self.paddle_x + self.pw / 2) / self.W * 2 - 1,
        ], dtype=np.float32)
        return {
            "state": state,
            "bricks": self.bricks.copy(),
            "lives": self.lives,
            "score": self.score,
        }

    def render(self, scale=None):
        scale = self.render_scale if scale is None else scale
        frame = self._draw(hud=True)
        if scale != 1:
            frame = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
        return frame

    def _draw(self, hud):
        c = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        c[:] = BG
        c[: self.wall, :] = WALL_COLOR
        c[:, : self.wall] = WALL_COLOR
        c[:, self.W - self.wall :] = WALL_COLOR
        for row in range(self.brick_rows):
            y0 = self.brick_top + row * self.brick_h
            for col in range(self.brick_cols):
                if not self.bricks[row, col]:
                    continue
                x0 = int(self.wall + col * self.brick_w)
                x1 = int(self.wall + (col + 1) * self.brick_w) - 1
                c[y0 : y0 + self.brick_h - 1, x0:x1] = ROW_COLORS[row]
        px0, px1 = int(self.paddle_x), int(self.paddle_x + self.pw)
        c[self.paddle_y : self.paddle_y + self.ph, px0:px1] = PADDLE_COLOR
        bx, by, r = int(self.ball_x), int(self.ball_y), self.bs
        c[max(0, by - r) : by + r, max(0, bx - r) : bx + r] = BALL_COLOR
        if hud:
            _draw_score(c, int(self.score), self.W)
        return c

    def close(self):
        pass


def _draw_score(c, value, width, scale=3, y=10):
    s = f"{value:03d}"
    dw = 3 * scale
    gap = scale
    total = len(s) * dw + (len(s) - 1) * gap
    x = (width - total) // 2
    for ch in s:
        bitmap = DIGITS[int(ch)]
        for ry, rowbits in enumerate(bitmap):
            for rx, bit in enumerate(rowbits):
                if bit == "1":
                    yy = y + ry * scale
                    xx = x + rx * scale
                    c[yy : yy + scale, xx : xx + scale] = SCORE_COLOR
        x += dw + gap


def _resize(img, out_w, out_h):
    h, w = img.shape[:2]
    ys = (np.arange(out_h) * h / out_h).astype(int)
    xs = (np.arange(out_w) * w / out_w).astype(int)
    return img[ys][:, xs]


class RecordWrapper:
    def __init__(self, env, path, fps=60, scale=None):
        import imageio.v2 as imageio
        self.env = env
        self.path = path
        self.scale = scale
        self._writer = imageio.get_writer(path, fps=fps, macro_block_size=1)
        self._closed = False

    def _capture(self):
        self._writer.append_data(self.env.render(scale=self.scale))

    def reset(self, **kw):
        out = self.env.reset(**kw)
        self._capture()
        return out

    def step(self, action):
        out = self.env.step(action)
        self._capture()
        return out

    def render(self, *a, **kw):
        return self.env.render(*a, **kw)

    def close(self):
        if not self._closed:
            self._writer.close()
            self._closed = True
            print(f"recording saved -> {self.path}")
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


if __name__ == "__main__":
    env = RecordWrapper(Breakout(seed=0), "smoke.mp4")
    obs, info = env.reset(seed=0)
    rng = np.random.default_rng(0)
    total = 0.0
    while True:
        a = rng.integers(0, 3)
        obs, r, term, trunc, info = env.step(int(a))
        total += r
        if term or trunc:
            break
    env.close()
    print(f"obs {obs.shape} {obs.dtype} | episode reward {total} | lives {info['lives']}")
