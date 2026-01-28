from manimlib import *
from tqdm import tqdm
import re
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import wavfile


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b' #6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
MAGENTA='#FF00FF'

SCALE_FACTOR=0.4


audio_fn='/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/exports/tell_me_about_china.wav'


class P5b(InteractiveScene):
    def construct(self): 

        '''
        Ok let me hack a bit on how to do the waveform here. I guess first I need to 
        get the take from the final a-roll, glad that's done!


        '''
        sample_rate, data = wavfile.read(audio_fn)


        data = data / np.max(np.abs(data))
        
        # Downsample for plotting (adjust factor as needed)
        # downsample_factor = max(1, len(data) // 5000)
        downsample_factor=12
        data_ds = data[::downsample_factor]
        print(len(data_ds))
        
        # Time array
        duration = len(data) / sample_rate
        t = np.linspace(0, duration, len(data_ds))
        
        # Create axes
        axes = Axes(
            x_range=[0, duration, duration/4],
            y_range=[-1, 1, 0.5],
            width=12,
            height=2.5,
            axis_config={
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_ticks": False,
                "tick_size": 0.05,
                "include_tip": True,
                "tip_config": {"width":0.02, "length":0.02}
            },
        )
        axes.move_to([0, 2.5, 0])
        # self.add(axes)
        
        waveform = VMobject()
        waveform.set_stroke(BLUE, width=3)

        self.wait()
        self.add(waveform)

        N = 20
        #Uncomment for final render
        # for i in range(N, len(data_ds), N):
        #     points = [axes.c2p(t[j], data_ds[j]) for j in range(i)]
        #     waveform.set_points_as_corners(points)
        #     self.wait(1/30) 
        
        # Final frame with all samples
        points = [axes.c2p(t[j], data_ds[j]) for j in range(len(data_ds))]
        waveform.set_points_as_corners(points)

        self.wait()




        




        self.wait(20)
        self.embed()



class P5a(InteractiveScene):
    def construct(self): 

        #I think I want to use the typewriter font!
        
        #Phone, node index, x coord, y coord
        nodes=[['start', 0, 0, 0],
               ['T',     1, 4, 4],
               ['AH',    2, 6, 7],
               ['EL',    3, 8, 4]]

        #Connections between node ids
        edges=[[0, 1], 
                  [1, 2],
                  [2, 3],
                  [1, 3],]

        node_mobjects = {}
        buff = 0.2
        for label, idx, x, y in nodes:
            text = Text(label, font="American Typewriter", color=CHILL_BROWN)
            text.set_color(CHILL_BROWN)
            box = RoundedRectangle(
                width=text.get_width() + 2 * buff,
                height=text.get_height() + 2 * buff,
                corner_radius=0.15,
                color=CHILL_BROWN,
            )
            node = VGroup(box, text)
            node.move_to([x*SCALE_FACTOR, y*SCALE_FACTOR, 0])
            node_mobjects[idx] = node
        
        # Create arrows
        arrows = VGroup()
        for start_idx, end_idx in edges:
            start_node = node_mobjects[start_idx]
            end_node = node_mobjects[end_idx]
            
            arrow = Arrow(
                start_node.get_center(),
                end_node.get_center(),
                buff=0.5,  # keeps arrow from overlapping rounded rects
            )
            arrow.set_color(CHILL_BROWN)
            arrows.add(arrow)
        
        # Group everything
        all_nodes = VGroup(*node_mobjects.values())
        graph = VGroup(arrows, all_nodes)
        graph.center()
        
        self.wait()
        self.add(graph)



        self.wait(20)
        self.embed()