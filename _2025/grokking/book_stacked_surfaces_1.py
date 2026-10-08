from manimlib import *
from pathlib import Path

# Static book version of P45_3Db: neuron surface on top, with its three
# Fourier components stacked underneath.

data_dir = Path('/Volumes/hot_1/Stephencwelch Dropbox/welch_labs/grokking/from_linux/grok_1764706121')

CHILL_BROWN = '#948979'
ORANGE = '#eb8423'
LT_BLUE = '#5badb6'
PINK = '#ed5e78'
RED = '#d73b2f'
BLUE = '#236c94'
FRESH_TAN = '#f4ebd9'
YELLOW = '#fcc947'

# ---------------------------------- Config ---------------------------------- #
NEURON_IDX = 106
P = 113

# Vertical gap above each component surface, top to bottom:
# [top -> product, product -> y-only, y-only -> x-only]
# SURFACE_GAPS = [1.7, 1.7, 1.7]
SURFACE_GAPS = [1.9]*3

# Component surface colors, top to bottom: [product, y-only, x-only]
SURFACE_COLORS = [ORANGE, LT_BLUE, PINK]

# Edge curves on the top surface
X_LINE_COLOR = RED    # neuron output vs x, with y=0
Y_LINE_COLOR = YELLOW   # neuron output vs y, with x=0
LINE_WIDTH = 7

# theta, phi, gamma, center, height. Center z is roughly the middle of the
# stack, so nudge it if you change SURFACE_GAPS much.
CAMERA = (137, 59, 0, (0.28, 0.08, -2.34), 8.54)

# Fourier components of the neuron, top to bottom
FOURIER_FUNCS = [
    lambda x, y: 0.354 * np.cos(2*np.pi*4*x/P - 0.516) * np.cos(2*np.pi*4*y/P - 0.516),
    lambda x, y: 0.173 * np.cos(2*np.pi*8*y/P + 2.653),
    lambda x, y: 0.173 * np.cos(2*np.pi*8*x/P + 2.653),
]
# ---------------------------------------------------------------------------- #


def make_axes(z_range, z_shift=0):
    axes = ThreeDAxes(
        x_range=[0, P, 10],
        y_range=[0, P, 10],
        z_range=z_range,
        width=4,
        height=4,
        depth=1.4,
        axis_config={
            "color": CHILL_BROWN,
            "include_ticks": False,
            "include_numbers": False,
            "include_tip": True,
            "stroke_width": 2,
            "tip_config": {"width": 0.05, "length": 0.05},
        },
    )
    x_label = Tex("x", font_size=32).next_to(axes.x_axis.get_end(), RIGHT, buff=0.1)
    y_label = Tex("y", font_size=32).next_to(axes.y_axis.get_end(), UP, buff=0.1)
    x_label.rotate(90*DEGREES, RIGHT).rotate(180*DEGREES, OUT)
    y_label.rotate(90*DEGREES, RIGHT).rotate(90*DEGREES, OUT)
    labels = VGroup(x_label, y_label).set_color(CHILL_BROWN)

    axes.x_axis.rotate(90*DEGREES, RIGHT)
    axes.y_axis.rotate(90*DEGREES, UP)

    Group(axes, labels).shift(z_shift * OUT)
    return axes, labels


def make_surface(uv_func):
    return ParametricSurface(uv_func, u_range=[0, 1], v_range=[0, 1], resolution=(P, P))


def make_curve(points, color):
    curve = VMobject(stroke_color=color, stroke_width=LINE_WIDTH)
    curve.set_points_smoothly(points)
    return curve



WIRE_WIDTH = 0.5
WIRE_OPACITY = 0.5
WIRE_CAMERA = (132, 36, 0, (-0.4, -0.43, -0.72), 8.05)  # Top-surface-only view from the video




class P45_3Db_book(InteractiveScene):
    def construct(self):
        act = np.load(data_dir/'mlp_hook_pre.npy')[:, :, 2, NEURON_IDX]
        act = act - (act.mean() - 0.20)  # Same vertical offset as the video

        # Top: actual neuron output
        axes_1, labels_1 = make_axes([-1, 1, 1])

        def neuron_point(u, v):
            i, j = int(round(u*(P - 1))), int(round(v*(P - 1)))
            return axes_1.c2p(u*P, v*P, act[i, j])

        ts = TexturedSurface(
            make_surface(neuron_point),
            str(data_dir/f'activations_{NEURON_IDX:03d}.png'),
        )
        ts.set_shading(0.0, 0.1, 0)
        ts.set_opacity(1.0) #0.8

        # Pure x / pure y slices, drawn along the y=0 and x=0 edges of the top surface.
        # These draw over the surface; add .apply_depth_test() to let it occlude them.
        us = np.linspace(0, 1, P)
        x_line = make_curve([neuron_point(u, 0) for u in us], X_LINE_COLOR)
        y_line = make_curve([neuron_point(0, v) for v in us], Y_LINE_COLOR)

        # Fourier components, stacked below
        component_axes = Group()
        component_labels = VGroup()
        component_surfaces = Group()
        for func, color, z in zip(FOURIER_FUNCS, SURFACE_COLORS, -np.cumsum(SURFACE_GAPS)):
            ax, labels = make_axes([-0.5, 0.5, 0.5], z_shift=z)
            surf = make_surface(lambda u, v, ax=ax, func=func: ax.c2p(u*P, v*P, func(u*P, v*P)))
            surf.set_color(color).set_shading(0.1, 0.5, 0.5)
            component_axes.add(ax)
            component_labels.add(labels)
            component_surfaces.add(surf)

        self.frame.reorient(*CAMERA)

        self.add(axes_1)
        self.add(x_line, y_line)
        self.add(ts)
        self.frame.reorient(135, 37, 0, (np.float32(-0.39), np.float32(-0.44), np.float32(-1.26)), 7.05)
        self.wait()

        # self.add(component_axes) #, component_labels, labels_1)
        # self.add(component_surfaces)

        self.wait()
        self.embed()

class P45_3Db_book_2(InteractiveScene):
    def construct(self):
        act = np.load(data_dir/'mlp_hook_pre.npy')[:, :, 2, NEURON_IDX]
        act = act - (act.mean() - 0.20)  # Same vertical offset as the video

        # Top: actual neuron output
        axes_1, labels_1 = make_axes([-1, 1, 1])

        def neuron_point(u, v):
            i, j = int(round(u*(P - 1))), int(round(v*(P - 1)))
            return axes_1.c2p(u*P, v*P, act[i, j])

        ts = TexturedSurface(
            make_surface(neuron_point),
            str(data_dir/f'activations_{NEURON_IDX:03d}.png'),
        )
        ts.set_shading(0.0, 0.1, 0)
        ts.set_opacity(1.0) #0.8

        # Pure x / pure y slices, drawn along the y=0 and x=0 edges of the top surface.
        # These draw over the surface; add .apply_depth_test() to let it occlude them.
        us = np.linspace(0, 1, P)
        x_line = make_curve([neuron_point(u, 0) for u in us], X_LINE_COLOR)
        y_line = make_curve([neuron_point(0, v) for v in us], Y_LINE_COLOR)

        # Fourier components, stacked below
        component_axes = Group()
        component_labels = VGroup()
        component_surfaces = Group()
        for func, color, z in zip(FOURIER_FUNCS, SURFACE_COLORS, -np.cumsum(SURFACE_GAPS)):
            ax, labels = make_axes([-0.5, 0.5, 0.5], z_shift=z)
            surf = make_surface(lambda u, v, ax=ax, func=func: ax.c2p(u*P, v*P, func(u*P, v*P)))
            surf.set_color(color).set_shading(0.1, 0.5, 0.5)
            component_axes.add(ax)
            component_labels.add(labels)
            component_surfaces.add(surf)

        self.frame.reorient(*CAMERA)

        self.add(axes_1)
        # self.add(x_line, y_line)
        # self.add(ts)
        # self.frame.reorient(135, 37, 0, (np.float32(-0.39), np.float32(-0.44), np.float32(-1.26)), 7.05)
        self.wait()

        self.add(component_axes) #, component_labels, labels_1)
        self.add(component_surfaces)
        self.add(ts)
        self.frame.reorient(133, 51, 0, (np.float32(0.27), np.float32(0.28), np.float32(-2.58)), 9.29)

        self.wait()
        self.embed()



class P45_3Db_wireframe(InteractiveScene):
    def construct(self):
        act = np.load(data_dir/'mlp_hook_pre.npy')[:, :, 2, NEURON_IDX]
        act = act - (act.mean() - 0.20)  # Same vertical offset as the video
        # act=act*0.3

        axes, labels = make_axes([-1, 1, 1])

        # pts[i, j] = neuron output at x index i, y index j
        coords = np.arange(P) * P / (P - 1)
        pts = np.array([
            [axes.c2p(coords[i], coords[j], act[i, j]) for j in range(P)]
            for i in range(P)
        ])

        # Wireframe: one curve per sweep step, in both directions
        wires = VGroup(
            *[make_curve(pts[:, j], FRESH_TAN) for j in range(P)],  # x sweeps, one per y
            *[make_curve(pts[i, :], FRESH_TAN) for i in range(P)],  # y sweeps, one per x
        )
        wires.set_stroke(width=WIRE_WIDTH, opacity=WIRE_OPACITY)

        # Pure x / pure y slices along the y=0 and x=0 edges
        x_line = make_curve(pts[:, 0], X_LINE_COLOR)
        y_line = make_curve(pts[0, :], Y_LINE_COLOR)

        self.frame.reorient(*WIRE_CAMERA)
        self.add(axes, labels, wires, x_line, y_line)

        self.wait()
        self.embed()
