from manimlib import *
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

HACKIN_DIR = Path('/Volumes/hot_1/Stephencwelch Dropbox/welch_labs/rl_1/hackin')
ASSET_DIR = HACKIN_DIR / 'manim_assets'  # where the exported texture goes

ANG_LIM = (-13, 13)     # pole angle, degrees -- same as the 2d version
VEL_LIM = (-170, 170)   # pole angular velocity, degrees/s

# theta values used for the height field / colormap. These match the
# manually-overridden t1, t2 = 0.01, 0.01 in the notebook's final plot
# cell (fae658d2...). Swap in pi.theta[0/1].item() from the trained
# policy once you want the "real" surface instead of the flat test one.

# T1 = 0.00
# T2 = 0.05

T1 = 1.0
T2 = 0.0

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
    return Polygon(*verts, **kwargs)


class policy_surfaces_3d_1(InteractiveScene):
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
        axes = ThreeDAxes(
            x_range=(*ANG_LIM, 5),
            y_range=(*VEL_LIM, 50),
            z_range=(0, 1, 0.25),
            width=10, height=10, depth=4,
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
        geom_surface = axes.get_graph(
            lambda u, v: sigmoid(T1 * u + T2 * v),
            u_range=ANG_LIM,
            v_range=VEL_LIM,
            resolution=(51, 51),
            opacity=1.0,
        )
        surface = TexturedSurface(geom_surface, tex_path)
        self.add(surface)
        surface.set_opacity(0.5)

        # Texture-free fallback, in case the exported-image pattern misbehaves --
        # color each vertex straight from the colormap instead:
        # geom_surface.color_by_uv_function(
        #     lambda u, v: mcolors.to_hex(cmap(sigmoid(T1 * u + T2 * v)))
        # )
        # self.add(geom_surface)

        # thin gridlines so the geometry (not just the color) reads -- built
        # off geom_surface since that's what carries the real point/normal
        # data; nudged slightly off the surface along its normal to avoid
        # z-fighting
        mesh = SurfaceMesh(
            geom_surface,
            resolution=(23, 23),
            stroke_width=0.5,
            stroke_color=WHITE, #CHILL_BROWN,
            stroke_opacity=0.4,
        )
        self.add(mesh)
        mesh.set_stroke(opacity=0.25, color=FRESH_TAN)

        # ---- 5. arrows: left actions flat on z=0, right actions flat on z=1 ----
        ARROW_WIDTH = 0.5
        arrows = VGroup()
        for a, v, action in zip(ang, angvel, act):
            went_right = (action == 1)
            arrow = arrow_polygon(
                direction=(1 if went_right else -1),
                fill_color=(YELLOW if went_right else BLUE),
                fill_opacity=0.85,
                stroke_width=0,
            )
            arrow.set_width(ARROW_WIDTH)
            arrow.move_to(axes.c2p(a, v, 1 if went_right else 0))
            arrows.add(arrow)
        self.add(arrows)

        # ---- 6. camera + hand off for interactive tuning ----
        # self.frame.reorient(44, 51, 0, (np.float32(1.17), np.float32(-0.47), np.float32(1.38)), 16.18)
        # self.frame.reorient(46, 61, 0, (np.float32(1.13), np.float32(-0.66), np.float32(1.23)), 16.40)
        self.frame.reorient(-42, 59, 0, (np.float32(-0.64), np.float32(0.43), np.float32(0.46)), 16.40)


        self.wait(20)
        self.embed()






# from manimlib import *
# from tqdm import tqdm
# import json
# from pathlib import Path

# CHILL_BROWN='#948979'
# YELLOW='#ffd35a'
# YELLOW_FADE='#7f6a2d'
# BLUE='#2ca3dd' #'#65c8d0'
# GREEN='#00a14b' 
# CHILL_GREEN='#6c946f'
# CHILL_BLUE='#3d5c6f'
# FRESH_TAN='#dfd0b9'
# RED='#ec2027'


# HACKIN_DIR=Path('/Volumes/hot_1/Stephencwelch Dropbox/welch_labs/rl_1/hackin')


# class policy_surfaces_1(InteractiveScene):
#     def construct(self):


#         with open(HACKIN_DIR/"cartpole_human_play/cartpole_human_demos_sw_3.json") as f: 
#             data = json.load(f)

#         all_obs=[]
#         all_actions=[]
#         episode_ids=[]

#         num_training_episodes=1
#         min_episode_len=30
#         episode_id=0


#         for i in [13]: 
#             ep=data["episodes"][i]
#             all_obs.append(ep["observations"])
#             all_actions.append(ep["actions"])
#             episode_ids.append(np.ones(len(ep["actions"]))*episode_id)
#             episode_id+=1
#             print(i, len(ep['observations']))

#         obs=np.concatenate(all_obs)
#         act=np.concatenate(all_actions)
#         ep_ids=np.concatenate(episode_ids)




#         self.wait()


#         self.wait(20)
#         self.embed()