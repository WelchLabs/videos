from manimlib import *
import json
from collections import defaultdict
from pathlib import Path
import copy

CHILL_BROWN = '#948979'

SCALE_FACTOR=0.4
JSON_SCALE_FACTOR_X = 0.012  # Scale down the coordinates from phone_dag.json
JSON_SCALE_FACTOR_Y = 0.03

json_dir=Path('/Users/stephen/manim/videos/_2026/the_bitter_lesson')

small_graph_nodes=[['start', 0, 0, 0],
       ['T',     1, 4, 4], #Tell
       ['AH',    2, 6, 7],
       ['EL',    3, 8, 4], 
       ['M',     4, 11, 7], #Me
       ['IY',    5, 15, 7],
       ['IH',    6, 11, 1], #Us
       ['S',     7, 15, 1],
       ['OW',    8, 18, 4], #all
       ['EL',    9, 22, 4],
       ['A',     10, 26, 4], #About
       ['B',     11, 30, 4],
       ['AW',    12, 34, 4],
       ['T',     13, 38, 4],
       ['SH',    14, 41, 7], #CHINA
       ['AY',    15, 45, 7],
       ['N',     16, 49, 7],
       ['UH',    17, 53, 7],
       ['N',     18, 41, 1], #NIXON
       ['IH',    19, 45, 1],
       ['X',     20, 49, 1],
       ['EN',    21, 53, 1],
       ['G',     22, 4, -6], #GIVE
       ['IH',    23, 8, -6],
       ['V',     24, 12, -6],
       ['M',     25, 16, -6], #ME
       ['IY',    26, 20, -6],
       ['TH',    27, 24, -6], #THE
       ['UH',    28, 27, -3],
       ['EE',    29, 27, -9],
       ['H',     30, 31, -3], #HEADLINES
       ['AA',    31, 34.5, -3],
       ['D',     32, 38, -3],
       ['L',     33, 41, -3],
       ['AY',    34, 45, -3],
       ['N',     35, 49, -3],
       ['S',     36, 53, -3],
       ['N',     37, 31, -9], #NEWS
       ['OO',    38, 42, -9],
       ['S',     39, 53, -9],
       ['end',   40, 58, 0]
       ]
small_graph_edges=[[0, 1], 
          [1, 2],
          [2, 3],
          [1, 3],
          [3, 4],
          [3, 6],
          [4, 5],
          [6, 7],
          [5, 8],
          [5, 10],
          [7, 10],
          [7, 8],
          [8, 9],
          [9, 10],
          [10, 11],
          [11, 12],
          [12, 13],
          [13, 14],
          [14, 15],
          [15, 16],
          [16, 17],      
          [13, 18], 
          [18, 19], 
          [19, 20],  
          [20, 21], 
          [17, 40], 
          [21, 40],
          [0, 22],
          [22, 23],   
          [23, 24],  
          [24, 25],  
          [25, 26],  
          [26, 27],  
          [27, 28], 
          [27, 29],
          [28, 30],
          [28, 37],
          [29, 30],
          [29, 37],
          [30, 31],
          [31, 32],
          [32, 33],
          [33, 34],
          [34, 35],
          [35, 36],
          [36, 40],
          [37, 38], 
          [38, 39], 
          [39, 40]

        ]

def get_rect_edge_point(rect, direction):
    """
    Get the point on a rectangle's edge in a given direction from its center.
    direction should be a unit vector.
    """
    center = rect.get_center()
    w = rect.get_width() / 2
    h = rect.get_height() / 2
    
    dx, dy = direction[0], direction[1]
    
    # Avoid division by zero
    if abs(dx) < 1e-8:
        # Vertical line
        t = h / abs(dy) if abs(dy) > 1e-8 else 0
    elif abs(dy) < 1e-8:
        # Horizontal line
        t = w / abs(dx)
    else:
        # Find intersection with both edges and take the closer one
        t_x = w / abs(dx)  # time to hit vertical edge
        t_y = h / abs(dy)  # time to hit horizontal edge
        t = min(t_x, t_y)
    
    return center + t * direction

class RenderNetworkV2(Scene):
    def construct(self):
        with open(json_dir/"phone_dag_v2.json", "r") as f:
            data = json.load(f)

        nodes_to_render=-1

        #Maybe a little preprocessing here?
        all_nodes_list=copy.deepcopy(small_graph_nodes[:-1]) #Leave off ending node for now
        for n in all_nodes_list:
            n[1]=n[1]+100000 #Avoid collisions
            n[2]=n[2]*SCALE_FACTOR
            n[3]=n[3]*SCALE_FACTOR
        
        for node_data in data["nodes"][:nodes_to_render]:
            if node_data['highlighted']: continue #Might make weird gaps, we'll see 
            label=node_data["phoneme"]
            idx=node_data["id"]
            x = node_data["x"] * JSON_SCALE_FACTOR_X
            y = node_data["y"] * JSON_SCALE_FACTOR_Y
            all_nodes_list.append([label, idx, x, y])

        self.wait()

        node_mobjects = {}
        buff = 0.2
        for label, idx, x, y in all_nodes_list:
            text = Text(label, font="American Typewriter", color=CHILL_BROWN)
            text.set_color(CHILL_BROWN)
            box = RoundedRectangle(
                width=text.get_width() + 2 * buff,
                height=text.get_height() + 2 * buff,
                corner_radius=0.15,
                color=CHILL_BROWN,
            )
            node = VGroup(box, text)
            node.move_to([x, y, 0])
            node_mobjects[idx] = node
        

        #Reindex to avoid collisions
        small_graph_edges_r=list(np.array(small_graph_edges)+100000)


        arrows = VGroup()
        arrow_dict={}
        # for start_idx, end_idx in edges:
        # for edge_data in data["edges"]:
        #     start_idx = edge_data["source"]
        #     end_idx = edge_data["target"]

        #     start_node = node_mobjects[start_idx]
        #     end_node = node_mobjects[end_idx]
            
        #     start_box = start_node[0]  # The RoundedRectangle
        #     end_box = end_node[0]
            
        #     # Direction from start to end
        #     direction = end_node.get_center() - start_node.get_center()
        #     direction = direction / np.linalg.norm(direction)  # normalize
            
        #     # Get edge points
        #     start_point = get_rect_edge_point(start_box, direction)
        #     end_point = get_rect_edge_point(end_box, -direction)
            
        #     # Small additional buffer for visual breathing room
        #     gap = 0.05
        #     start_point = start_point + gap * direction
        #     end_point = end_point - gap * direction
            
        #     arrow = Arrow(start_point, end_point, buff=0)
        #     arrow.set_color(CHILL_BROWN)
        #     arrows.add(arrow)
        #     arrow_dict[(start_idx, end_idx)]=arrow

        # Group everything
        all_nodes = VGroup(*node_mobjects.values())
        graph = VGroup(arrows, all_nodes)


        self.add(graph)
        self.wait()




        self.wait(20)
        self.embed()













