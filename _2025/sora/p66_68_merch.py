from welch_paths import welch_path
import manimgx as gx
from sora_motion import CustomTracedPath, get_color_wheel_colors
import numpy as np
from sora_fields import TrackerControlledVectorField
from welch_geometry import diagram_axes
from welch_scene import WelchScene

import glob

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'

from torch.utils.data import DataLoader
from smalldiffusion import (
    ScheduleLogLinear, samples, Swissroll, ModelMixin, ScheduleDDPM
)

from typing import Callable
from tqdm import tqdm
import torch
from itertools import pairwise


class p66v4(WelchScene):
    def construct(self):

        '''
        DDIM sampling on spiral dataset
        I'm tempted here to run both "noiseless" DDPM and DDIM from here
        I can make sure same starting points/scaling, and I think I actually want to be a 
        bit more zoomed out for nice side-by-side

        '''

        batch_size=2130
        dataset = Swissroll(np.pi/2, 5*np.pi, 100)
        loader = DataLoader(dataset, batch_size=batch_size)
        batch=next(iter(loader)).numpy()

        axes = diagram_axes(
            x_range=[-1.2, 1.2, 0.5],
            y_range=[-1.2, 1.2, 0.5],
            height=7,
            width=7,
            axis_config={
                "color": CHILL_BROWN, 
                "stroke_width": 2,
                "include_tip": True,
                "include_ticks": False,
                "tick_size": 0.06,
                "tip_config": {"color": CHILL_BROWN, "length": 0.15, "width": 0.15}
            }
        )

        axes.set_opacity(0.5)

        # Create extended axes with SAME center point and proportional scaling
        extended_axes = diagram_axes(
            x_range=[-2.0, 2.0, 0.5],    # Extended range
            y_range=[-2.0, 2.0, 0.5],    # Extended range
            height=7 * (4.0/2.4),        # Scale height proportionally: original_height * (new_range/old_range)
            width=7 * (4.0/2.4),         # Scale width proportionally: original_width * (new_range/old_range)
            axis_config={"stroke_width": 0}  # Make invisible
        )

        # Move extended axes to same position as original axes
        extended_axes.move_to(axes.get_center())


        dots = gx.VGroup()
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = gx.Dot(screen_point, radius=0.05)
            # dot.set_color(YELLOW)
            dots.add(dot)
        dots.set_color(CHILL_BROWN)
        dots.set_opacity(0.5)


        #Leaning towards exporting tracjectories from jupyter instead of running live here. 
        xt_history=np.load(welch_path('sora/hackin/ddim_history_2.npy'))
        heatmaps=np.load(welch_path('sora/hackin/ddim_heatmaps_2.npy'))
        model=torch.load(welch_path('sora/hackin/jun_26_2.pt'), weights_only=False)
        # model=torch.load('/Volumes/hot_1/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_24_1.pt', weights_only=False)

        time_tracker = gx.ValueTracker(0.0)  # Start at time 0


        schedule = ScheduleLogLinear(N=256, sigma_min=0.01, sigma_max=1) #N=200
        sigmas=schedule.sample_sigmas(256)

        def vector_function_with_tracker(coords_array):
            """Vector function that uses the ValueTracker for time"""
            current_time = time_tracker.get_value()
            max_time = 8.0  # Map time 0-8 to sigma indices 0-255
            sigma_idx = int(np.clip(current_time * 255 / max_time, 0, 255)) #Needs to be N-1
            
            try:
                res = model.forward(torch.tensor(coords_array).float(), sigmas[sigma_idx], cond=None)
                return -res.detach().numpy()
            except:
                return np.zeros((len(coords_array), 2))


        # Create the tracker-controlled vector field
        vector_field = TrackerControlledVectorField(
            time_tracker=time_tracker,
            func=vector_function_with_tracker,
            coordinate_system=extended_axes,
            density=5.0,
            stroke_width=2.0,
            max_radius=6.0,      # Vectors fade to min_opacity at this distance
            min_opacity=0.5,     # Minimum opacity at max_radius
            max_opacity=0.5,     # Maximum opacity at origin
            tip_width_ratio=4,
            tip_len_to_width=0.01,
            max_vect_len_to_step_size=0.7,
            color=CHILL_BROWN
        )

        # self.wait()
        # self.set_view(0, 0, 0, (0, 0, 0.0), 8)
        self.set_view(0, 0, 0, (0.0, 0.0, 0.0), 10)
        self.add(axes, dots)

        # self.add(vector_field)
        # vector_field.set_opacity(1.0)


        num_dots=256 #256
        colors=get_color_wheel_colors(num_dots)
        all_traced_paths=gx.VGroup()
        all_dots_to_move=gx.VGroup()
        for path_index in range(num_dots): 
            dot_to_move=gx.Dot(axes.c2p(*np.concatenate((xt_history[0, path_index, :], [0]))), radius=0.06)
            dot_to_move.set_color(colors[path_index])
            all_dots_to_move.add(dot_to_move)

            traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=colors[path_index], stroke_width=2.0, 
                                          opacity_range=(0.4, 0.4), fade_length=64) #Trying a full fade to zero
            # traced_path.set_opacity(0.5)
            # traced_path.set_fill(opacity=0)
            all_traced_paths.add(traced_path)
        self.add(all_traced_paths)

        # self.wait()
        # self.play(FadeIn(all_dots_to_move), FadeIn(vector_field))
        self.add(all_dots_to_move)

        self.wait()

        for k in range(xt_history.shape[0]):
            self.play(time_tracker.animate.set_value(8.0*(k/256.0)), 
                      *[all_dots_to_move[path_index].animate.move_to(axes.c2p(*[xt_history[k, path_index, 0], 
                                                                                xt_history[k, path_index, 1]])) for path_index in range(len(all_dots_to_move))],
                     rate_func=gx.linear, run_time=0.1)

        self.wait()

        self.wait()
        self.remove(all_traced_paths)
        self.wait()
        self.add(all_traced_paths)
        #BOook?

        self.set_view(0, 0, 0, (-0.23, 0.03, 0.0), 7.34)
        self.wait()

        self.remove(axes)
        self.set_view(0, 0, 0, (-0.39, -0.12, 0.0), 7.81)
        self.wait()

        self.set_view(0, 0, 0, (-0.26, -0.33, 0.0), 6.40)
        self.wait()


        self.remove(all_traced_paths)
        self.wait()



        self.wait(20)
        # Scene authoring stops here in the interactive editor.
