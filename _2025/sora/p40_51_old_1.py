import manimgx as gx
from sora_motion import manual_camera_interpolation
import numpy as np
from welch_geometry import diagram_axes
from welch_scene import WelchScene

import glob

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'
GREEN='#00a14b'
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'

from torch.utils.data import DataLoader
from smalldiffusion import (
    ScheduleLogLinear, samples, Swissroll, ModelMixin
)

class p40_51_sketch(WelchScene):
    def construct(self):

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
                "include_ticks": True,
                "tick_size": 0.06,
                "tip_config": {"color": CHILL_BROWN, "length": 0.15, "width": 0.15}
            }
        )

        self.add(axes)
        # self.wait()

        dots = gx.VGroup()
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = gx.Dot(screen_point, radius=0.04)
            # dot.set_color(YELLOW)
            dots.add(dot)
        dots.set_color(YELLOW)

        self.wait()

        # Animate the points appearing
        self.play(gx.FadeIn(dots, lag_ratio=0.1), run_time=2)

        self.wait()

        # Ok, let's zoom in on one point, lower opacity on all other points, 
        # and send it on a random walk
        # I think we overlay the image stuff for p42 in illustrator/premier

        #Example_point_index
        i=75
        dot_to_move=dots[i].copy()
        

        self.wait()
        self.play(dots.animate.set_opacity(0.1), 
                 dot_to_move.animate.scale(1.25), #Make main dot a little bigger!
                 self.animate_view(0, 0, 0, (2.92, 1.65, 0.0), 4.19),
                 run_time=2.0)


        self.wait()

        traced_path = gx.TracedPath(dot_to_move.get_center, stroke_color=YELLOW, stroke_width=2)
        traced_path.set_opacity(0.5)
        traced_path.set_fill(opacity=0)
        self.add(traced_path)
        self.add(dot_to_move)

        # Ok let's try generating this random walk with a seed in manim - I may want to 
        # compute elsewhere and cache, we'll see. Also, I think proably ignore noise schedule
        # right now? Not sure if/when it will make sense to add. 

        # for random_seed in range(2000):
        #     np.random.seed(random_seed) #4 is maybe best so far, #52 is ok
        #     random_walk=0.07*np.random.randn(100,2) #I might want to manually make the first step larger/more obvious.
        #     random_walk[0]=np.array([0.3, 0.18]) #make first step go up and to the right
        #     random_walk=np.cumsum(random_walk,axis=0)
        #     random_walk=np.hstack((random_walk, np.zeros((len(random_walk), 1))))
        #     random_walk_shifted=random_walk+np.array([batch[i][0], batch[i][1], 0])
        #     if random_walk_shifted[-1][0]>1.7 and random_walk_shifted[-1][0]<1.9 and random_walk_shifted[-1][1]>1.1 and random_walk_shifted[-1][1]<1.3:
        #         print(random_seed, random_walk_shifted[-1])

        # 485 [1.82222791 1.19839322 0.        ]                                                                                                                         
        # 509 [1.711807   1.28528277 0.        ]                                                                                                                         
        # 1164 [1.75093007 1.23755551 0.        ]                                                                                                                        
        # 1297 [1.88499977 1.28501196 0.        ]                                                                                                                        
        # 1547 [1.86467848 1.20829298 0.        ]                                                                                                                        


        np.random.seed(485) #485 is nice, 4 is maybe best so far, #52 is ok
        random_walk=0.07*np.random.randn(100,2) #I might want to manually make the first step larger/more obvious.
        random_walk[0]=np.array([0.2, 0.12]) #make first step go up and to the right
        random_walk[-1]=np.array([0.08, -0.02])
        random_walk=np.cumsum(random_walk,axis=0) 
        print(random_walk[0])

        random_walk=np.hstack((random_walk, np.zeros((len(random_walk), 1))))
        random_walk_shifted=random_walk+np.array([batch[i][0], batch[i][1], 0])
        print(random_walk_shifted[-1])

        #Hmm do I want an arrow or does the point just move with a tail?
        self.wait()
        
        dot_history=gx.VGroup()
        dot_history.add(dot_to_move.copy().scale(0.25))
        self.add(dot_history[-1])
        self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[0])), run_time=1.0)


        self.wait()

        for i in range(100):
            dot_history.add(dot_to_move.copy().scale(0.25))
            self.add(dot_history[-1])
            self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[i])), run_time=0.2, rate_func=gx.linear)

        self.wait()

        # Now zoom out and run all paths -> maybe while zoooming out. 
        # I think looking at variable path opacity (highest closes to end point) will be worth looking at for 
        # the batch option. 

        #Ok so I want to fadd in rest of point and center my plot. Then diffuse everybody!!
        self.play(self.animate_view(0, 0, 0, (-0.07, 0.01, 0.0), 7.59),
                        dots.animate.set_opacity(1.0), 
                        run_time=3.0)
        self.wait()

        random_walks=[]
        np.random.seed(2)
        for i in range(100):
            rw=0.07*np.random.randn(100,2) #I might want to manually make the first step larger/more obvious.
            # rw[0]=np.array([0.2, 0.12]) #make first step go up and to the right
            # rw[-1]=np.array([0.08, -0.02])
            rw=np.cumsum(rw,axis=0) 
            rw=np.hstack((rw, np.zeros((len(rw), 1))))
            rw_shifted=rw+np.array([batch[i][0], batch[i][1], 0])
            random_walks.append(rw_shifted)

        # maybe we actually totally lost the spiral for this animation? That might make for a more dramatic
        # bringing stuff back together phase

        traced_paths=gx.VGroup()
        for d in dots: 
            tp = gx.TracedPath(d.get_center, stroke_color=YELLOW, stroke_width=2)
            tp.set_opacity(0.2)
            tp.set_fill(opacity=0)
            traced_path.add(tp)

        remaining_indices=np.concatenate((np.arange(75), np.arange(76,len(batch))))    


        start_orientation=[0, 0, 0, (-0.07, 0.01, 0.0), 7.59]
        end_orientation=[0, 0, 0, (0.23, -0.24, 0.0), 14.98]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=100)

        self.wait()
        remaining_indices=np.concatenate((np.arange(75), np.arange(76,len(batch))))
        for step in range(100):
            self.play(*[dots[i].animate.move_to(axes.c2p(*random_walks[i][step])) for i in remaining_indices], 
                     self.animate_view(*interp_orientations[step]),
                     run_time=0.1, rate_func=gx.linear)


        #So there are 3 nice to haves here that I think I'll skip for now, can do if i have time
        # 1. little dots between steps
        # 2. Variable opacity traces
        # 
        # Eh actually it kinda looks like i need to replace tracedpaths, probably a good job for claude. 
        # Let me try to figure out the camear zoom first. 


        self.wait()

        #Now play random walk backwards and zoom back in! Don't forget to remove traced paths as we go backwards





        self.wait(20)
        # Scene authoring stops here in the interactive editor.
