import manimgx as gx
from welch_scene import WelchScene
from _2026.rl.policy_storyboard import (
    HACKIN_DIR,
    PolicySurfaceTextScene,
    SWEEP_T1_RANGE,
    SWEEP_T1_RANGE_V2,
    sigmoid,
    arrow_polygon,
    play_theta_sweep,
    play_theta_sweep_v2,
)
from _2026.rl.policy_surface_gx import (
    color_surface_vertices, height_field, surface_wireframe, texture_surface,
)
from tqdm import tqdm
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

CHILL_BROWN = '#948979'
YELLOW = '#ffd35a'
YELLOW_FADE = '#7f6a2d'
BLUE = '#2ca3dd'  # '#65c8d0'
GREEN = '#00a14b'
CHILL_GREEN = '#6c946f'
CHILL_BLUE = '#3d5c6f'
FRESH_TAN = '#dfd0b9'
RED = '#ec2027'
MAGENTA = '#FF00FF'

ASSET_DIR = HACKIN_DIR / 'manim_assets'  # where the exported texture goes

ANG_LIM = (-13, 13)     # pole angle, degrees -- same as the 2d version
VEL_LIM = (-170, 170)   # pole angular velocity, degrees/s

# theta values used for the height field / colormap. These match the
# manually-overridden t1, t2 = 0.01, 0.01 in the notebook's final plot
# cell (fae658d2...). Swap in pi.theta[0/1].item() from the trained
# policy once you want the "real" surface instead of the flat test one.
# T1 = 0.00
# T2 = 0.05
# T1 = 1.0
# T2 = 0.0
# T1 = 1.0
# T2 = 0.035
T1 = 0.2
T2 = 0.05






class policy_surfaces_3d_1(WelchScene):
    def construct(self):
        # ---- 1. load the same episode as the 2d version (ep 13) ----
        with open(HACKIN_DIR / "cartpole_human_play/cartpole_human_demos_sw_3.json") as f:
            data = json.load(f)

        all_obs, all_actions = [], []
        for i in [13]:
            ep = data["episodes"][i]
            all_obs.append(ep["observations"])
            all_actions.append(ep["actions"])
            print(i, len(ep['observations']))

        obs = np.concatenate(all_obs)
        act = np.concatenate(all_actions)

        ang = np.degrees(obs[:, 2])
        angvel = np.degrees(obs[:, 3])

        # ---- 2. axes: x = pole angle, y = pole angular velocity, z = P(right) ----
        axes = gx.ThreeDAxes(
            x_range=(*ANG_LIM, 5),
            y_range=(*VEL_LIM, 50),
            z_range=(0, 1, 0.25),
            x_length=10, y_length=10, z_length=4,
        )
        self.add(axes)

        # ---- 3. build + export the blue -> yellow colormap as a texture ----
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        tex_path = str(ASSET_DIR / "policy_prob_texture.png")
        cmap = mcolors.LinearSegmentedColormap.from_list("blue_gold", [BLUE, YELLOW], N=256)

        TEX_RES = 200
        A, V = np.meshgrid(np.linspace(*ANG_LIM, TEX_RES), np.linspace(*VEL_LIM, TEX_RES))
        P_grid = sigmoid(T1 * A + T2 * V)

        fig = plt.figure(figsize=(4, 4), dpi=200)
        tex_ax = fig.add_axes([0, 0, 1, 1])
        tex_ax.imshow(P_grid, origin='lower', cmap=cmap, vmin=0, vmax=1, aspect='auto')
        tex_ax.axis('off')
        fig.savefig(tex_path)
        plt.close(fig)
        # NB: if the color ends up flipped top-to-bottom relative to the
        # height field once it's on the surface, swap to origin='upper'
        # above (or np.flipud(P_grid)) -- the image-row <-> v-coordinate
        # direction is the one fiddly bit of the texture pattern.

        # ---- 4. the surface itself: height = P(right) ----
        geom_surface = height_field(axes,
            lambda u, v: sigmoid(T1 * u + T2 * v),
            u_range=ANG_LIM,
            v_range=VEL_LIM,
            resolution=(51, 51),
            opacity=1.0,
        )
        surface = texture_surface(geom_surface, tex_path)
        self.add(surface)
        surface.set_opacity(0.5)

        # Texture-free fallback, in case the exported-image pattern misbehaves --
        # color each vertex straight from the colormap instead. Uses
        # set_rgba_array_by_color (base Mobject) rather than
        # Surface.color_by_uv_function, since the latter isn't in every
        # manimgl build (it's newer than v1.7.2) -- see policy_surfaces_3d_2
        # for the resolution-matched u/v grid this needs.
        # geom_surface.set_rgba_array_by_color([
        #     mcolors.to_hex(cmap(sigmoid(T1 * u + T2 * v)))
        #     for u, v in np.stack(np.meshgrid(
        #         np.linspace(*ANG_LIM, 51), np.linspace(*VEL_LIM, 51), indexing='ij'
        #     ), axis=-1).reshape(-1, 2)
        # ])
        # self.add(geom_surface)

        # thin gridlines so the geometry (not just the color) reads -- built
        # off geom_surface since that's what carries the real point/normal
        # data; nudged slightly off the surface along its normal to avoid
        # z-fighting
        mesh = surface_wireframe(
            geom_surface,
            resolution=(23, 23),
            stroke_width=0.8,
            stroke_color=CHILL_BROWN,  # CHILL_BROWN,
            stroke_opacity=0.5,
        )
        self.add(mesh)
        mesh.set_stroke(opacity=0.25, color=FRESH_TAN)

        # ---- 5. arrows: left actions flat on z=0, right actions flat on z=1 ----
        ARROW_WIDTH = 0.5
        arrows = gx.VGroup()
        for a, v, action in zip(ang, angvel, act):
            went_right = (action == 1)
            arrow = arrow_polygon(
                direction=(1 if went_right else -1),
                fill_color=(YELLOW if went_right else BLUE),
                fill_opacity=0.85,
                stroke_width=0,
            )
            arrow.scale_to_fit_width(ARROW_WIDTH)
            arrow.move_to(axes.c2p(a, v, 1 if went_right else 0))
            arrows.add(arrow)
        self.add(arrows)

        self.remove(surface); self.add(surface)

        # ---- 6. camera + hand off for interactive tuning ----
        # self.set_view(44, 51, 0, (np.float32(1.17), np.float32(-0.47), np.float32(1.38)), 16.18)
        # self.set_view(46, 61, 0, (np.float32(1.13), np.float32(-0.66), np.float32(1.23)), 16.40)
        self.set_view(-42, 59, 0, (np.float32(-0.64), np.float32(0.43), np.float32(0.46)), 16.40)
        self.wait(20)


# ----------------------------------------------------------------------
# policy_surfaces_3d_2: sweep theta1/theta2 to show how the surface morphs.
#
# Rendered as *two* scenes that share the exact same choreography
# (play_theta_sweep, called with the same starting tracker values in both)
# so they stay frame-synced when stitched together in Premiere:
#   - policy_surfaces_3d_2       the 3d surface + mesh
#   - policy_surfaces_3d_2_text  a plain 2d scene with just the
#                                 "(theta_1=X.XX, theta_2=Y.YY)" readout
# 3d text in manimgl tends to fight the camera (billboarding, legibility
# from odd angles, etc.), so it's easier to keep it as a separate flat
# overlay than to fight with fixed_in_frame text inside the 3d scene.
# ----------------------------------------------------------------------





class policy_surfaces_3d_2(WelchScene):
    def construct(self):
        axes = gx.ThreeDAxes(
            x_range=(*ANG_LIM, 5),
            y_range=(*VEL_LIM, 50),
            z_range=(0, 1, 0.25),
            x_length=10, y_length=10, z_length=4,
        )
        self.add(axes)

        cmap = mcolors.LinearSegmentedColormap.from_list("blue_gold", [BLUE, YELLOW], N=256)
        SURFACE_RES = (41, 41)   # a bit coarser than scene 1 -- rebuilt every frame
        MESH_RES = (23, 23)
        SURFACE_OPACITY = 0.5

        t1_tracker = gx.ValueTracker(SWEEP_T1_RANGE[0])
        t2_tracker = gx.ValueTracker(0)

        # Texture-free coloring (color_by_uv_function) on purpose here: it's
        # just per-vertex color data, so always_redraw can cheaply rebuild it
        # every frame. TexturedSurface's image is a GPU texture set once at
        # construction, which doesn't lend itself to being recolored live.
        #
        # Surface.color_by_uv_function isn't in every manimgl build (it's
        # newer than v1.7.2), so this colors vertices "by hand" via the more
        # fundamental Mobject.set_rgba_array_by_color, computing the u/v grid
        # ourselves to match axes.get_graph's resolution/ordering exactly.
        def vertex_colors(t1, t2, resolution):
            nu, nv = resolution
            U, V = np.meshgrid(
                np.linspace(*ANG_LIM, nu), np.linspace(*VEL_LIM, nv), indexing='ij'
            )
            heights = sigmoid(t1 * U + t2 * V)
            return [mcolors.to_hex(cmap(h)) for h in heights.reshape(-1)]

        def build_surface():
            t1, t2 = t1_tracker.get_value(), t2_tracker.get_value()
            surf = height_field(axes,
                lambda u, v: sigmoid(t1 * u + t2 * v),
                u_range=ANG_LIM, v_range=VEL_LIM,
                resolution=SURFACE_RES, opacity=SURFACE_OPACITY,
            )
            color_surface_vertices(surf, vertex_colors(t1, t2, SURFACE_RES))
            surf.set_opacity(SURFACE_OPACITY)
            return surf

        surface = gx.always_redraw(build_surface)
        mesh = gx.always_redraw(lambda: surface_wireframe(
            build_surface(),
            resolution=MESH_RES,
            stroke_width=1,
            stroke_color=FRESH_TAN,
            stroke_opacity=0.35,
        ))

        # mesh.set_stroke(opacity=0.25, color=FRESH_TAN)
        # surface.set_opacity(0.5)

        self.add(surface, mesh)

        self.set_view(-42, 59, 0, (np.float32(-0.64), np.float32(0.43), np.float32(0.46)), 16.40)
        self.wait()

        play_theta_sweep(self, t1_tracker, t2_tracker)

        self.wait()


class policy_surfaces_3d_2_text_c(WelchScene):
    def construct(self):
        t1_tracker = gx.ValueTracker(SWEEP_T1_RANGE[0])
        t2_tracker = gx.ValueTracker(0)
 
        # Static Tex for the symbols/punctuation ("\theta_1 =", etc) plus a
        # DecimalNumber for each live value. DecimalNumber just swaps digit
        # glyphs each frame instead of re-invoking latex, so this is much
        # cheaper than rebuilding a full Tex string every frame -- worth it
        # here since this scene is nothing but this one label for its whole
        # runtime.
        # No placeholder mobjects for the parens/comma/space -- Tex("") or
        # Tex(r"\quad") on its own compiles to zero glyphs (spacing commands
        # produce no ink), so you get an empty, zero-width mobject that
        # breaks the next_to() chain. Use a bigger buff between the two
        # theta groups instead of a spacer mobject.
        theta1_sym = gx.MathTex(r"\theta_1 =")
        theta1_val = gx.DecimalNumber(t1_tracker.get_value(), num_decimal_places=2, include_sign=True)
        theta2_sym = gx.MathTex(r"\theta_2 =")
        theta2_val = gx.DecimalNumber(t2_tracker.get_value(), num_decimal_places=3, include_sign=True)
 
        label = gx.VGroup(theta1_sym, theta1_val, theta2_sym, theta2_val)
        label.set_color(FRESH_TAN)
        label.scale(1.3)
        label.arrange(gx.RIGHT, buff=0.15)
        label.to_edge(gx.DOWN)
 
        # arrange() only sets the initial layout -- fixed decimal places keep
        # each value's width ~constant, but chain next_to() updaters off of
        # theta1_sym's fixed position anyway so nothing drifts as the digits
        # (and +/- sign) change frame to frame. The wider buff below
        # (theta1_val -> theta2_sym) is what reads as the gap between the
        # two theta groups.
        theta1_val.add_updater(lambda d: d.set_value(t1_tracker.get_value()))
        theta1_val.add_updater(lambda d: d.next_to(theta1_sym, gx.RIGHT, buff=0.15))
        theta2_sym.add_updater(lambda d: d.next_to(theta1_val, gx.RIGHT, buff=0.6))
        theta2_val.add_updater(lambda d: d.set_value(t2_tracker.get_value()))
        theta2_val.add_updater(lambda d: d.next_to(theta2_sym, gx.RIGHT, buff=0.15))
 
        self.add(label)
 
        # Simpler (but slower -- recompiles latex every frame) alternative,
        # if the DecimalNumber wiring above is more than you want:
        # label = always_redraw(lambda: Tex(
        #     rf"(\theta_1 = {t1_tracker.get_value():+.2f}, \ "
        #     rf"\theta_2 = {t2_tracker.get_value():+.3f})",
        #     color=FRESH_TAN,
        # ).scale(1.3).to_edge(DOWN))
        # self.add(label)
 
        self.wait()
 
        play_theta_sweep(self, t1_tracker, t2_tracker)
 
        self.wait()


class policy_surfaces_3d_3(WelchScene):
    def construct(self):
        # ---- 1. load the same episode as scene 1 (ep 13) ----
        with open(HACKIN_DIR / "cartpole_human_play/cartpole_human_demos_sw_3.json") as f:
            data = json.load(f)
 
        all_obs, all_actions = [], []
        for i in [13]:
            ep = data["episodes"][i]
            all_obs.append(ep["observations"])
            all_actions.append(ep["actions"])
            print(i, len(ep['observations']))
 
        obs = np.concatenate(all_obs)
        act = np.concatenate(all_actions)
 
        ang = np.degrees(obs[:, 2])
        angvel = np.degrees(obs[:, 3])
 
        # ---- 2. axes: x = pole angle, y = pole angular velocity, z = P(right) ----
        axes = gx.ThreeDAxes(
            x_range=(*ANG_LIM, 5),
            y_range=(*VEL_LIM, 50),
            z_range=(0, 1, 0.25),
            x_length=10, y_length=10, z_length=4,
        )
        self.add(axes)
 
        cmap = mcolors.LinearSegmentedColormap.from_list("blue_gold", [BLUE, YELLOW], N=256)
        SURFACE_RES = (41, 41)   # a bit coarser than scene 1 -- rebuilt every frame
        MESH_RES = (23, 23)
        SURFACE_OPACITY = 0.5
 
        t1_tracker = gx.ValueTracker(SWEEP_T1_RANGE[0])
        t2_tracker = gx.ValueTracker(0)
 
        # ---- 3. the sweeping surface + mesh (identical to policy_surfaces_3d_2) ----
        def vertex_colors(t1, t2, resolution):
            nu, nv = resolution
            U, V = np.meshgrid(
                np.linspace(*ANG_LIM, nu), np.linspace(*VEL_LIM, nv), indexing='ij'
            )
            heights = sigmoid(t1 * U + t2 * V)
            return [mcolors.to_hex(cmap(h)) for h in heights.reshape(-1)]
 
        def build_surface():
            t1, t2 = t1_tracker.get_value(), t2_tracker.get_value()
            surf = height_field(axes,
                lambda u, v: sigmoid(t1 * u + t2 * v),
                u_range=ANG_LIM, v_range=VEL_LIM,
                resolution=SURFACE_RES, opacity=SURFACE_OPACITY,
            )
            color_surface_vertices(surf, vertex_colors(t1, t2, SURFACE_RES))
            surf.set_opacity(SURFACE_OPACITY)
            return surf
 
        surface = gx.always_redraw(build_surface)
        mesh = gx.always_redraw(lambda: surface_wireframe(
            build_surface(),
            resolution=MESH_RES,
            stroke_width=1,
            stroke_color=FRESH_TAN,
            stroke_opacity=0.35,
        ))
        self.add(surface, mesh)
 
        # ---- 4. arrows: left actions flat on z=0, right actions flat on z=1 --
        # static throughout the sweep, same as policy_surfaces_3d_1
        ARROW_WIDTH = 0.5
        arrows = gx.VGroup()
        for a, v, action in zip(ang, angvel, act):
            went_right = (action == 1)
            arrow = arrow_polygon(
                direction=(1 if went_right else -1),
                fill_color=(YELLOW if went_right else BLUE),
                fill_opacity=0.85,
                stroke_width=0,
            )
            arrow.scale_to_fit_width(ARROW_WIDTH)
            arrow.move_to(axes.c2p(a, v, 1 if went_right else 0))
            arrows.add(arrow)
        self.add(arrows)
 
        # ---- 5. camera (same view as scenes 1 and 2) + sweep ----
        # self.set_view(-42, 59, 0, (np.float32(-0.64), np.float32(0.43), np.float32(0.46)), 16.40)
        self.set_view(-26, 64, 0, (np.float32(0.71), np.float32(-0.04), np.float32(0.87)), 16.40)
        self.wait()
 
        play_theta_sweep(self, t1_tracker, t2_tracker)
 
        self.wait()



class loss_to_surface_viz_1(WelchScene):
    def construct(self):
        # ---- 1. load the same episode as the 2d version (ep 13) ----
        with open(HACKIN_DIR / "cartpole_human_play/cartpole_human_demos_sw_3.json") as f:
            data = json.load(f)

        all_obs, all_actions = [], []
        for i in [13]:
            ep = data["episodes"][i]
            all_obs.append(ep["observations"])
            all_actions.append(ep["actions"])
            print(i, len(ep['observations']))

        obs = np.concatenate(all_obs)
        act = np.concatenate(all_actions)

        ang = np.degrees(obs[:, 2])
        angvel = np.degrees(obs[:, 3])

        # ---- 2. axes: x = pole angle, y = pole angular velocity, z = P(right) ----
        axes = gx.ThreeDAxes(
            x_range=(*ANG_LIM, 5),
            y_range=(*VEL_LIM, 50),
            z_range=(0, 1, 0.25),
            x_length=10, y_length=10, z_length=4,
        )
        self.add(axes)

        # ---- 3. build + export the blue -> yellow colormap as a texture ----
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        tex_path = str(ASSET_DIR / "policy_prob_texture.png")
        cmap = mcolors.LinearSegmentedColormap.from_list("blue_gold", [BLUE, YELLOW], N=256)

        TEX_RES = 200
        A, V = np.meshgrid(np.linspace(*ANG_LIM, TEX_RES), np.linspace(*VEL_LIM, TEX_RES))
        P_grid = sigmoid(T1 * A + T2 * V)

        fig = plt.figure(figsize=(4, 4), dpi=200)
        tex_ax = fig.add_axes([0, 0, 1, 1])
        tex_ax.imshow(P_grid, origin='lower', cmap=cmap, vmin=0, vmax=1, aspect='auto')
        tex_ax.axis('off')
        fig.savefig(tex_path)
        plt.close(fig)

        # ---- 4. the surface itself: height = P(right) ----
        geom_surface = height_field(axes,
            lambda u, v: sigmoid(T1 * u + T2 * v),
            u_range=ANG_LIM,
            v_range=VEL_LIM,
            resolution=(51, 51),
            opacity=1.0,
        )
        surface = texture_surface(geom_surface, tex_path)
        self.add(surface)
        surface.set_opacity(0.5)


        mesh = surface_wireframe(
            geom_surface,
            resolution=(23, 23),
            stroke_width=0.5,
            stroke_color=gx.WHITE,  # CHILL_BROWN,
            stroke_opacity=0.4,
        )
        self.add(mesh)
        mesh.set_stroke(opacity=0.25, color=FRESH_TAN)

        # ---- 5. arrows: left actions flat on z=0, right actions flat on z=1 ----
        ARROW_WIDTH = 0.5
        arrows = gx.VGroup()
        for a, v, action in zip(ang, angvel, act):
            went_right = (action == 1)
            arrow = arrow_polygon(
                direction=(1 if went_right else -1),
                fill_color=(YELLOW if went_right else BLUE),
                fill_opacity=0.85,
                stroke_width=0,
            )
            arrow.scale_to_fit_width(ARROW_WIDTH)
            arrow.move_to(axes.c2p(a, v, 1 if went_right else 0))
            arrows.add(arrow)
        self.add(arrows)

        # ---- 6. camera + hand off for interactive tuning ----
        # self.set_view(44, 51, 0, (np.float32(1.17), np.float32(-0.47), np.float32(1.38)), 16.18)
        # self.set_view(46, 61, 0, (np.float32(1.13), np.float32(-0.66), np.float32(1.23)), 16.40)
        self.set_view(-42, 59, 0, (np.float32(-0.64), np.float32(0.43), np.float32(0.46)), 16.40)
        self.wait(20)


# ----------------------------------------------------------------------
# loss_to_surface_viz_1: L1 loss as green cylinders connecting each ground-
# -truth arrow (z=0 or z=1) straight up/down to the surface's predicted
# P(right) at that same (angle, angvel) -- the line's length *is* the L1
# loss for that step, with a small sphere marking where it lands on the
# surface.
# ----------------------------------------------------------------------
 
class loss_to_surface_viz_1(WelchScene):
    def construct(self):
        # ---- 1. load the same episode as the 2d version (ep 13) ----
        with open(HACKIN_DIR / "cartpole_human_play/cartpole_human_demos_sw_3.json") as f:
            data = json.load(f)
 
        all_obs, all_actions = [], []
        for i in [13]:
            ep = data["episodes"][i]
            all_obs.append(ep["observations"])
            all_actions.append(ep["actions"])
            print(i, len(ep['observations']))
 
        obs = np.concatenate(all_obs)
        act = np.concatenate(all_actions)
 
        ang = np.degrees(obs[:, 2])
        angvel = np.degrees(obs[:, 3])
 
        # ---- 2. axes: x = pole angle, y = pole angular velocity, z = P(right) ----
        axes = gx.ThreeDAxes(
            x_range=(*ANG_LIM, 5),
            y_range=(*VEL_LIM, 50),
            z_range=(0, 1, 0.25),
            x_length=10, y_length=10, z_length=4,
        )
        self.add(axes)
 
        # ---- 3. build + export the blue -> yellow colormap as a texture ----
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        tex_path = str(ASSET_DIR / "policy_prob_texture.png")
        cmap = mcolors.LinearSegmentedColormap.from_list("blue_gold", [BLUE, YELLOW], N=256)
 
        TEX_RES = 200
        A, V = np.meshgrid(np.linspace(*ANG_LIM, TEX_RES), np.linspace(*VEL_LIM, TEX_RES))
        P_grid = sigmoid(T1 * A + T2 * V)
 
        fig = plt.figure(figsize=(4, 4), dpi=200)
        tex_ax = fig.add_axes([0, 0, 1, 1])
        tex_ax.imshow(P_grid, origin='lower', cmap=cmap, vmin=0, vmax=1, aspect='auto')
        tex_ax.axis('off')
        fig.savefig(tex_path)
        plt.close(fig)
 
        # ---- 4. the surface itself: height = P(right) ----
        geom_surface = height_field(axes,
            lambda u, v: sigmoid(T1 * u + T2 * v),
            u_range=ANG_LIM,
            v_range=VEL_LIM,
            resolution=(51, 51),
            opacity=1.0,
        )
        surface = texture_surface(geom_surface, tex_path)
        self.add(surface)
        surface.set_opacity(0.5)
 
        mesh = surface_wireframe(
            geom_surface,
            resolution=(23, 23),
            stroke_width=0.5,
            stroke_color=gx.WHITE,  # CHILL_BROWN,
            stroke_opacity=0.4,
        )
        self.add(mesh)
        mesh.set_stroke(opacity=0.25, color=FRESH_TAN)
 
        # ---- 5. arrows: left actions flat on z=0, right actions flat on z=1 ----
        ARROW_WIDTH = 0.5
        arrows = gx.VGroup()
        for a, v, action in zip(ang, angvel, act):
            went_right = (action == 1)
            arrow = arrow_polygon(
                direction=(1 if went_right else -1),
                fill_color=(YELLOW if went_right else BLUE),
                fill_opacity=0.85,
                stroke_width=0,
            )
            arrow.scale_to_fit_width(ARROW_WIDTH)
            arrow.move_to(axes.c2p(a, v, 1 if went_right else 0))
            arrows.add(arrow)
        self.add(arrows)
 
        # ---- 6. L1 loss: a green cylinder from each arrow straight to the
        # surface's predicted P(right) at that (angle, angvel), plus a small
        # sphere where it touches down. Line length == |prediction - target|,
        # i.e. exactly the per-step L1 loss.
        LOSS_LINE_WIDTH = 0.04
        LOSS_SPHERE_RADIUS = 0.06
        loss_lines = gx.Group()
        loss_spheres = gx.Group()
        for a, v, action in zip(ang, angvel, act):
            z_target = 1.0 if action == 1 else 0.0
            z_pred = sigmoid(T1 * a + T2 * v)
 
            surface_point = axes.c2p(a, v, z_pred)
            arrow_point = axes.c2p(a, v, z_target)
 
            # skip the (rare) zero-length case -- Line3D can't handle
            # coincident start/end points
            if np.allclose(surface_point, arrow_point):
                continue
 
            line = gx.Line3D(
                surface_point, arrow_point,
                thickness=LOSS_LINE_WIDTH, resolution=(9, 9), color=GREEN,
            )
            loss_lines.add(line)
 
            sphere = gx.Sphere(radius=LOSS_SPHERE_RADIUS, resolution=(8, 8), color=GREEN)
            sphere.move_to(surface_point)
            loss_spheres.add(sphere)
 
        self.add(loss_lines, loss_spheres)

        self.remove(loss_lines)
        self.add(loss_lines)
        self.remove(axes)
        self.remove(loss_spheres)
        self.remove(arrows)
        self.add(arrows)
 
        # ---- 7. camera + hand off for interactive tuning ----
        # self.set_view(-56, 54, 0, (np.float32(-0.48), np.float32(0.26), np.float32(0.33)), 13.15)
        self.set_view(-55, 57, 0, (np.float32(-0.52), np.float32(0.23), np.float32(0.26)), 13.15)
        self.wait(20)




# ----------------------------------------------------------------------
# loss_to_surface_sweep_1: the L1-loss-cylinder viz from
# loss_to_surface_viz_1, but animated through the theta sweep -- the
# surface morphs and every green loss line stretches/shrinks live as
# (theta1, theta2) move.
#
# Uses the notebook's *updated* sweep ranges and the corrected ellipse
# (inscribed in the ranges, center at the range midpoints) rather than
# the origin-centered ellipse in play_theta_sweep above, which overshoots
# to theta1 = -0.6 with the new asymmetric t1 range. Kept as separate
# _V2 constants + function so the already-rendered policy_surfaces_3d_2*
# scenes stay frame-synced with their existing choreography.
# ----------------------------------------------------------------------





class loss_to_surface_sweep_1(WelchScene):
    def construct(self):
        # ---- 1. load the same episode as the other scenes (ep 13) ----
        with open(HACKIN_DIR / "cartpole_human_play/cartpole_human_demos_sw_3.json") as f:
            data = json.load(f)

        all_obs, all_actions = [], []
        for i in [13]:
            ep = data["episodes"][i]
            all_obs.append(ep["observations"])
            all_actions.append(ep["actions"])
            print(i, len(ep['observations']))

        obs = np.concatenate(all_obs)
        act = np.concatenate(all_actions)

        ang = np.degrees(obs[:, 2])
        angvel = np.degrees(obs[:, 3])

        # ---- 2. axes: x = pole angle, y = pole angular velocity, z = P(right) ----
        # Built for c2p() geometry but NOT added to the scene -- matches the
        # net end state of loss_to_surface_viz_1, which removes the axes
        # (and the touchdown spheres) during its add/remove shuffle.
        axes = gx.ThreeDAxes(
            x_range=(*ANG_LIM, 5),
            y_range=(*VEL_LIM, 50),
            z_range=(0, 1, 0.25),
            x_length=10, y_length=10, z_length=4,
        )

        cmap = mcolors.LinearSegmentedColormap.from_list("blue_gold", [BLUE, YELLOW], N=256)
        SURFACE_RES = (41, 41)   # rebuilt every frame, same as policy_surfaces_3d_2
        MESH_RES = (23, 23)
        SURFACE_OPACITY = 0.5

        t1_tracker = gx.ValueTracker(SWEEP_T1_RANGE_V2[0])
        t2_tracker = gx.ValueTracker(0)

        # ---- 3. sweeping surface + mesh (same pattern as policy_surfaces_3d_2:
        # per-vertex colors instead of a texture, since TexturedSurface's image
        # is set once at construction and can't be recolored live) ----
        def vertex_colors(t1, t2, resolution):
            nu, nv = resolution
            U, V = np.meshgrid(
                np.linspace(*ANG_LIM, nu), np.linspace(*VEL_LIM, nv), indexing='ij'
            )
            heights = sigmoid(t1 * U + t2 * V)
            return [mcolors.to_hex(cmap(h)) for h in heights.reshape(-1)]

        def build_surface():
            t1, t2 = t1_tracker.get_value(), t2_tracker.get_value()
            surf = height_field(axes,
                lambda u, v: sigmoid(t1 * u + t2 * v),
                u_range=ANG_LIM, v_range=VEL_LIM,
                resolution=SURFACE_RES, opacity=SURFACE_OPACITY,
            )
            color_surface_vertices(surf, vertex_colors(t1, t2, SURFACE_RES))
            surf.set_opacity(SURFACE_OPACITY)
            return surf

        surface = gx.always_redraw(build_surface)
        mesh = gx.always_redraw(lambda: surface_wireframe(
            build_surface(),
            resolution=MESH_RES,
            stroke_width=1,
            stroke_color=FRESH_TAN,
            stroke_opacity=0.35,
        ))
        axes.set_color(CHILL_BROWN)
        self.add(axes)
        self.add(surface, mesh)

        # ---- 4. L1 loss lines: rebuilt every frame off the trackers, so each
        # green cylinder stretches/shrinks as the surface morphs. Line length
        # == |P(right) - target| == the per-step L1 loss, exactly as in the
        # static scene. Coarser cylinder resolution than the static scene's
        # (9, 9) since ~150 of these get rebuilt per frame.
        LOSS_LINE_WIDTH = 0.04
        LOSS_LINE_RES = (5, 5)
        SHOW_LOSS_SPHERES = False   # static scene's final state drops the
        LOSS_SPHERE_RADIUS = 0.06   # touchdown spheres; flip this to bring
        LOSS_SPHERE_RES = (6, 6)    # them back

        def build_loss_group():
            t1, t2 = t1_tracker.get_value(), t2_tracker.get_value()
            group = gx.Group()
            for a, v, action in zip(ang, angvel, act):
                z_target = 1.0 if action == 1 else 0.0
                z_pred = sigmoid(t1 * a + t2 * v)

                surface_point = axes.c2p(a, v, z_pred)
                arrow_point = axes.c2p(a, v, z_target)

                # skip the (rare) zero-length case -- Line3D can't handle
                # coincident start/end points
                if np.allclose(surface_point, arrow_point):
                    continue

                group.add(gx.Line3D(
                    surface_point, arrow_point,
                    thickness=LOSS_LINE_WIDTH, resolution=LOSS_LINE_RES, color=GREEN,
                ))
                if SHOW_LOSS_SPHERES:
                    sphere = gx.Sphere(
                        radius=LOSS_SPHERE_RADIUS,
                        resolution=LOSS_SPHERE_RES, color=GREEN,
                    )
                    sphere.move_to(surface_point)
                    group.add(sphere)
            return group

        loss_group = gx.always_redraw(build_loss_group)
        self.add(loss_group)

        # ---- 5. arrows: static targets on z=0 / z=1, added last so they draw
        # on top of the loss lines (same order the static scene's add/remove
        # shuffle lands on) ----
        ARROW_WIDTH = 0.5
        arrows = gx.VGroup()
        for a, v, action in zip(ang, angvel, act):
            went_right = (action == 1)
            arrow = arrow_polygon(
                direction=(1 if went_right else -1),
                fill_color=(YELLOW if went_right else BLUE),
                fill_opacity=0.85,
                stroke_width=0,
            )
            arrow.scale_to_fit_width(ARROW_WIDTH)
            arrow.move_to(axes.c2p(a, v, 1 if went_right else 0))
            arrows.add(arrow)
        self.add(arrows)

        # ---- 6. camera (same view as loss_to_surface_viz_1) + sweep ----
        # self.set_view(-55, 57, 0, (np.float32(-0.52), np.float32(0.23), np.float32(0.26)), 13.15)
        self.set_view(-45, 61, 0, (np.float32(-0.38), np.float32(0.17), np.float32(0.34)), 13.15)
        self.wait()

        play_theta_sweep_v2(self, t1_tracker, t2_tracker)

        self.wait()


# ----------------------------------------------------------------------
# policy_surfaces_3d_text_v2: flat overlay for the v2 sweep, frame-synced
# with loss_to_surface_sweep_1.
#
#     \boldsymbol{\theta} = [X.YZ, X.YZ]
#     \nabla_{\boldsymbol{\theta}} = [X.YZ, X.YZ]
#
# Layout is fully explicit this time -- no next_to chains at runtime.
# Every element after the sym gets a fixed slot computed once at
# construction, sized from the worst-case number widths over the whole
# sweep (thetas from the sweep ranges, gradients from a grid over the
# sweep's bounding box, which contains the entire path). The commas,
# brackets, and slot positions never move; each DecimalNumber just
# re-pins its LEFT edge to its slot every frame, so nothing downstream
# jitters as digit/sign widths change, and both value slots share one
# fixed y taken from the first number's placement.
# ----------------------------------------------------------------------







class policy_surfaces_3d_text_v6(PolicySurfaceTextScene):
    text_color = FRESH_TAN
