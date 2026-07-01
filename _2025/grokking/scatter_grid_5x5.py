from pathlib import Path
import pickle
import os

import matplotlib.cm as cm
import matplotlib.colors as colors
from manimlib import *


CHILL_BROWN = "#948979"
DATA_DIR = Path(
    "/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/grokking/"
    "from_linux/grok_1764706121"
)
NEURON_INDICES = [0, 3, 4, 5, 9]
SCATTER_DOT_RADIUS = float(os.environ.get("GROKKING_DOT_RADIUS", "0.055"))
BACKGROUND_COLOR = os.environ.get("GROKKING_BACKGROUND_COLOR", WHITE)


def viridis_hex(value, vmin, vmax):
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return colors.to_hex(cm.viridis(norm(value)))


class GrokkingScatterGrid5x5(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR}

    def construct(self):
        p = 113

        with open(DATA_DIR / "final_model_activations_sample.p", "rb") as file:
            activations = pickle.load(file)

        values = activations["blocks.0.mlp.hook_pre"][:p, 2]

        axes = VGroup()
        for _ in range(25):
            axes.add(
                Axes(
                    x_range=[-1.0, 1.0, 1],
                    y_range=[-1.0, 1.0, 1],
                    width=1.15,
                    height=1.15,
                    axis_config={
                        "color": CHILL_BROWN,
                        "include_ticks": False,
                        "include_numbers": False,
                        "include_tip": True,
                        "stroke_width": 1.5,
                        "tip_config": {"width": 0.025, "length": 0.025},
                    },
                )
            )

        axes.arrange_in_grid(n_rows=5, n_cols=5, buff=0.25)

        dots = VGroup()
        for row, neuron_y in enumerate(NEURON_INDICES):
            for column, neuron_x in enumerate(NEURON_INDICES):
                plot_dots = VGroup()
                x_values = values[:, neuron_x]
                y_values = values[:, neuron_y]
                x_center = x_values.mean()
                y_center = y_values.mean()
                x_scale = np.max(np.abs(x_values - x_center))
                y_scale = np.max(np.abs(y_values - y_center))

                for index in range(p):
                    x = (x_values[index] - x_center) / x_scale
                    y = (y_values[index] - y_center) / y_scale
                    dot = Dot(
                        axes[5 * row + column].c2p(x, y),
                        radius=SCATTER_DOT_RADIUS,
                        stroke_width=0,
                    )
                    dot.set_color(viridis_hex(index, 0, p))
                    plot_dots.add(dot)

                dots.add(plot_dots)

        self.add(dots)
        self.wait()
