"""Shared cartpole policy sweep animations and readouts."""
from welch_paths import welch_path

import json
import os
from pathlib import Path

import manimgx as gx
import numpy as np

from welch_scene import WelchScene
from welch_style import FRESH_TAN, MAGENTA

HACKIN_DIR = Path(os.environ.get("WELCH_RL_HACKIN", welch_path('rl_1/hackin')))
SWEEP_T1_RANGE = (-2.0, 2.0)
SWEEP_T2_RANGE = (-1.0, 1.0)
SWEEP_TIME = 4          # seconds, for each of the two straight sweeps
TRANSITION_TIME = 1.5   # seconds, for the two "get into position" moves
ELLIPSE_TIME = 6        # seconds, for the full loop around parameter space
SWEEP_T1_RANGE_V2 = (-0.2, 0.6)
SWEEP_T2_RANGE_V2 = (-0.15, 0.15)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def arrow_polygon(direction=1, shaft_w=0.28, head_w=0.9, head_len=0.9, total_len=2.0, **kwargs):
    """Same silhouette as the matplotlib arrow_path() in the notebook, but
    built as a manim Polygon that lies flat in the local xy-plane (z=0)."""
    hl, sw, hw, L = head_len, shaft_w / 2, head_w / 2, total_len / 2
    pts = np.array([
        (-L,  sw), (L - hl,  sw), (L - hl,  hw), (L, 0),
        (L - hl, -hw), (L - hl, -sw), (-L, -sw),
    ], dtype=float)
    pts[:, 0] *= direction
    verts = [np.array([x, y, 0.0]) for x, y in pts]
    return gx.Polygon(*verts, **kwargs)


def play_theta_sweep(scene, t1_tracker, t2_tracker):
    """Shared choreography, called identically from both the 3d scene and
    the text-overlay scene so their timing lines up exactly:
      1. sweep theta1 across its full range, theta2 held at 0
      2. move into position for the theta2 sweep
      3. sweep theta2 across its full range, theta1 held at 0
      4. move into position for the ellipse
      5. loop once around an ellipse covering the same theta1/theta2 extents

    Assumes t1_tracker starts at SWEEP_T1_RANGE[0] and t2_tracker starts at 0.
    """
    t1_lo, t1_hi = SWEEP_T1_RANGE
    t2_lo, t2_hi = SWEEP_T2_RANGE

    # 1. sweep theta1, -2 -> 2, theta2 fixed at 0
    scene.play(
        t1_tracker.animate.set_value(t1_hi),
        run_time=SWEEP_TIME, rate_func=gx.linear,
    )

    # 2. transition to the theta2 sweep's start point, (0, -0.1)
    scene.play(
        t1_tracker.animate.set_value(0),
        t2_tracker.animate.set_value(t2_lo),
        run_time=TRANSITION_TIME, rate_func=gx.smooth,
    )

    # 3. sweep theta2, -0.1 -> 0.1, theta1 fixed at 0
    scene.play(
        t2_tracker.animate.set_value(t2_hi),
        run_time=SWEEP_TIME, rate_func=gx.linear,
    )

    # 4. transition to the ellipse's start point, (2, 0)
    scene.play(
        t1_tracker.animate.set_value(t1_hi),
        t2_tracker.animate.set_value(0),
        run_time=TRANSITION_TIME, rate_func=gx.smooth,
    )

    # 5. one full loop around an ellipse with semi-axes (t1_hi, t2_hi) --
    # covers the same min/max range as the two straight sweeps above.
    # animate.set_value only does linear interpolation of a single value,
    # so the coupled (t1, t2) path is driven manually off an alpha-func.
    a, b = t1_hi, t2_hi

    def ellipse_step(mob, alpha):
        theta = alpha * gx.TAU
        t1_tracker.set_value(a * np.cos(theta))
        t2_tracker.set_value(b * np.sin(theta))

    scene.play(
        gx.UpdateFromAlphaFunc(gx.Mobject(), ellipse_step),
        run_time=ELLIPSE_TIME, rate_func=gx.linear,
    )


def play_theta_sweep_v2(scene, t1_tracker, t2_tracker,
                        t1_range=SWEEP_T1_RANGE_V2,
                        t2_range=SWEEP_T2_RANGE_V2):
    """Same choreography as play_theta_sweep (and generate_theta_sweep in
    the notebook), with the ellipse-center correction:
      1. sweep theta1 across its full range, theta2 held at 0
      2. transition to the theta2 sweep's start, (0, t2_lo)
      3. sweep theta2 across its full range, theta1 held at 0
      4. transition to the ellipse's start, (t1_hi, c2)
      5. one loop around the ellipse *inscribed in the sweep ranges*:
         center (c1, c2) = range midpoints, semi-axes = half-spans

    Note segment 4's target is (t1_hi, c2), not (t1_hi, 0) -- they only
    coincide while the t2 range is symmetric.

    Assumes t1_tracker starts at t1_range[0] and t2_tracker starts at 0.
    """
    t1_lo, t1_hi = t1_range
    t2_lo, t2_hi = t2_range
    c1, r1 = (t1_lo + t1_hi) / 2, (t1_hi - t1_lo) / 2
    c2, r2 = (t2_lo + t2_hi) / 2, (t2_hi - t2_lo) / 2

    # 1. sweep theta1, t1_lo -> t1_hi, theta2 fixed at 0
    scene.play(
        t1_tracker.animate.set_value(t1_hi),
        run_time=SWEEP_TIME, rate_func=gx.linear,
    )

    # 2. transition to the theta2 sweep's start point, (0, t2_lo)
    scene.play(
        t1_tracker.animate.set_value(0),
        t2_tracker.animate.set_value(t2_lo),
        run_time=TRANSITION_TIME, rate_func=gx.smooth,
    )

    # 3. sweep theta2, t2_lo -> t2_hi, theta1 fixed at 0
    scene.play(
        t2_tracker.animate.set_value(t2_hi),
        run_time=SWEEP_TIME, rate_func=gx.linear,
    )

    # 4. transition to the ellipse's start point, (t1_hi, c2)
    scene.play(
        t1_tracker.animate.set_value(t1_hi),
        t2_tracker.animate.set_value(c2),
        run_time=TRANSITION_TIME, rate_func=gx.smooth,
    )

    # 5. one full loop around the inscribed ellipse. As in play_theta_sweep,
    # the coupled (t1, t2) path is driven manually off an alpha-func since
    # animate.set_value only interpolates single values linearly.
    def ellipse_step(mob, alpha):
        theta = alpha * gx.TAU
        t1_tracker.set_value(c1 + r1 * np.cos(theta))
        t2_tracker.set_value(c2 + r2 * np.sin(theta))

    scene.play(
        gx.UpdateFromAlphaFunc(gx.Mobject(), ellipse_step),
        run_time=ELLIPSE_TIME, rate_func=gx.linear,
    )


class PolicySurfaceTextScene(WelchScene):
    hackin_dir = HACKIN_DIR
    text_color = FRESH_TAN

    def construct(self):
        # ---- 1. same episode as the 3d scenes (ep 13) -- needed here
        # because the gradient is a data-dependent quantity ----
        with open(self.hackin_dir / "cartpole_human_play/cartpole_human_demos_sw_3.json") as f:
            data = json.load(f)

        all_obs, all_actions = [], []
        for i in [13]:
            ep = data["episodes"][i]
            all_obs.append(ep["observations"])
            all_actions.append(ep["actions"])
            print(i, len(ep['observations']))

        obs = np.concatenate(all_obs)
        act = np.concatenate(all_actions)

        ang = np.degrees(obs[:, 2])      # degree-valued features, matching
        angvel = np.degrees(obs[:, 3])   # the surface's sigmoid(t1*A + t2*V)

        def grad_deg(t1, t2):
            """d(L1 loss)/d(theta) in the degree parametrization.
            L = 2 * mean|p - y|, same loss as the notebook's l1_loss."""
            logit = t1 * ang + t2 * angvel
            p = sigmoid(logit)
            diff = p - act
            dp = p * (1.0 - p)
            return np.array([
                -2.0 * np.mean(np.sign(diff) * dp * ang),  #SW negative grad actually, a little easier setup for loss function
                -2.0 * np.mean(np.sign(diff) * dp * angvel),
            ])

        t1_tracker = gx.ValueTracker(SWEEP_T1_RANGE_V2[0])
        t2_tracker = gx.ValueTracker(0)

        # ---- 2. build elements ----
        SCALE = 1.3
        BUFF = 0.15
        N_DEC = 2
        COMMA_DROP = 0.04   # commas hang a touch below the digit baseline;
                            # eyeball in embed() and tune

        def make_num(value):
            return gx.DecimalNumber(
                value, num_decimal_places=N_DEC, include_sign=True
            ).scale(SCALE)

        theta_sym = gx.MathTex(r"\boldsymbol{\theta} = [").scale(SCALE)
        theta_v1 = make_num(t1_tracker.get_value())
        theta_comma = gx.MathTex(",").scale(SCALE)
        theta_v2 = make_num(t2_tracker.get_value())
        theta_rb = gx.MathTex("]").scale(SCALE)

        g0 = grad_deg(t1_tracker.get_value(), t2_tracker.get_value())

        # g0*=-1 # Ok i want to do negative gradient actually.
        grad_sym = gx.MathTex(r"-\nabla_{\boldsymbol{\theta}} \mathcal{L} = [").scale(SCALE)
        grad_v1 = make_num(g0[0])
        grad_comma = gx.MathTex(",").scale(SCALE)
        grad_v2 = make_num(g0[1])
        grad_rb = gx.MathTex("]").scale(SCALE)

        everything = gx.VGroup(theta_sym, theta_v1, theta_comma, theta_v2, theta_rb,
                            grad_sym, grad_v1, grad_comma, grad_v2, grad_rb)
        everything.set_color(self.text_color)
        grad_sym.set_color(MAGENTA)
        grad_v1.set_color(MAGENTA)
        grad_comma.set_color(MAGENTA)
        grad_v2.set_color(MAGENTA)
        grad_rb.set_color(MAGENTA)

        # ---- 3. worst-case slot widths ----
        # thetas: bounded by the sweep ranges. gradients: sample a grid over
        # the sweep's bounding box (the whole path -- both straight sweeps,
        # both transitions, and the inscribed ellipse -- lives inside it),
        # so the slots are sized from the actual data, not a guess.
        tg1 = np.linspace(*SWEEP_T1_RANGE_V2, 41)
        tg2 = np.linspace(*SWEEP_T2_RANGE_V2, 41)
        gmax = np.zeros(2)
        for t1v in tg1:
            for t2v in tg2:
                gmax = np.maximum(gmax, np.abs(grad_deg(t1v, t2v)))

        def slot_width(max_abs):
            """Width of the widest rendering of a value in [-max_abs, max_abs]
            at N_DEC places -- checks both signs since + and - render at
            different widths (the source of the left/right jitter)."""
            return max(make_num(s * max_abs).width for s in (+1.0, -1.0))

        w_theta_1 = slot_width(max(abs(v) for v in SWEEP_T1_RANGE_V2))
        w_theta_2 = slot_width(max(abs(v) for v in SWEEP_T2_RANGE_V2))
        w_grad_1 = slot_width(gmax[0])
        w_grad_2 = slot_width(gmax[1])

        # ---- 4. explicit layout: place the sym, then hand-place the back
        # half of the row into fixed slots. Returns the two anchor points the
        # numbers re-pin their LEFT edges to each frame. ----
        def layout_row(sym, v1, comma, v2, rb, w1, w2, x_left, y_sym):
            sym.move_to(np.array([x_left, y_sym, 0.0]), aligned_edge=gx.LEFT)

            # one next_to at construction only, to inherit the vertical
            # placement that already read correctly for theta1 -- its y
            # becomes the fixed y for BOTH value slots
            v1.next_to(sym, gx.RIGHT, buff=BUFF)
            y_num = v1.get_center()[1]

            x = sym.get_right()[0] + BUFF
            anchor1 = np.array([x, y_num, 0.0])
            x += w1 + BUFF

            comma.move_to(np.array([x, y_num, 0.0]), aligned_edge=gx.LEFT)
            comma.align_to(v1, gx.DOWN).shift(COMMA_DROP * gx.DOWN)
            x = comma.get_right()[0] + BUFF
            anchor2 = np.array([x, y_num, 0.0])
            x += w2 + BUFF

            rb.move_to(np.array([x, y_num, 0.0]), aligned_edge=gx.LEFT)
            rb.align_to(sym, gx.DOWN)   # bracket bottoms match: "[" is sym's
                                     # lowest glyph
            return anchor1, anchor2

        def row_width(sym, w1, comma, w2, rb):
            return (sym.width + BUFF + w1 + BUFF + comma.width
                    + BUFF + w2 + BUFF + rb.width)

        # left-align both rows, centered horizontally on the wider (grad) row
        total_w = max(row_width(theta_sym, w_theta_1, theta_comma, w_theta_2, theta_rb),
                      row_width(grad_sym, w_grad_1, grad_comma, w_grad_2, grad_rb))
        x_left = -total_w / 2
        Y_THETA = -2.55   # theta row  } bottom-edge placement; tune in embed
        Y_GRAD = -3.5    # grad row   }

        theta_a1, theta_a2 = layout_row(
            theta_sym, theta_v1, theta_comma, theta_v2, theta_rb,
            w_theta_1, w_theta_2, x_left, Y_THETA)
        grad_a1, grad_a2 = layout_row(
            grad_sym, grad_v1, grad_comma, grad_v2, grad_rb,
            w_grad_1, w_grad_2, x_left, Y_GRAD)

        # ---- 5. updaters: numbers only. set_value, then re-pin the left
        # edge to the fixed slot anchor -- the static pieces never move. ----
        def pin(num, getter, anchor):
            def upd(d):
                d.set_value(getter())
                d.move_to(anchor, aligned_edge=gx.LEFT)
            num.add_updater(upd)

        pin(theta_v1, t1_tracker.get_value, theta_a1)
        pin(theta_v2, t2_tracker.get_value, theta_a2)
        pin(grad_v1,
            lambda: grad_deg(t1_tracker.get_value(), t2_tracker.get_value())[0],
            grad_a1)
        pin(grad_v2,
            lambda: grad_deg(t1_tracker.get_value(), t2_tracker.get_value())[1],
            grad_a2)

        self.add(everything)

        self.wait()

        play_theta_sweep_v2(self, t1_tracker, t2_tracker)

        self.wait()
