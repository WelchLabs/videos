from manimlib import *
from tqdm import tqdm
import re
from pathlib import Path
import matplotlib.pyplot as plt


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

       # Create node mobjects
        node_mobjects = {}
        for label, idx, x, y in nodes:
            text = Text(label, font="American Typewriter")
            text.set_color(CHILL_BROWN)
            box = SurroundingRectangle(
                text,
                color=CHILL_BROWN,
                buff=0.2,
                # corner_radius=0.15,
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