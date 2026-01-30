from manimlib import *
import json
from collections import defaultdict
from pathlib import Path
import copy
from tqdm import tqdm

CHILL_BROWN = '#948979'

SCALE_FACTOR=0.4
JSON_SCALE_FACTOR_X = 0.012  # Scale down the coordinates from phone_dag.json
JSON_SCALE_FACTOR_Y = 0.03
GLOBAL_SHIFT=[11.52887616, -0.4       ,  0.        ] #To match p5

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

def purge_priority_neighbors(all_nodes_list, neighbor_buffer=1.0):
    """
    Remove non-priority nodes that are within neighbor_buffer distance 
    of any priority node (id > 10000).
    
    Args:
        all_nodes_list: List of [name, id, x, y] nodes
        neighbor_buffer: Distance threshold for removal
    
    Returns:
        Filtered list with nearby non-priority nodes removed
    """
    # Separate priority and non-priority nodes
    priority_nodes = [n for n in all_nodes_list if n[1] > 100000]
    other_nodes = [n for n in all_nodes_list if n[1] <= 100000]
    
    def distance(n1, n2):
        return ((n1[2] - n2[2])**2 + (n1[3] - n2[3])**2)**0.5
    
    # Filter out non-priority nodes that are too close to any priority node
    kept_nodes = []
    for node in other_nodes:
        too_close = False
        for p_node in priority_nodes:
            if distance(node, p_node) < neighbor_buffer:
                too_close = True
                break
        if not too_close:
            kept_nodes.append(node)
    
    return priority_nodes + kept_nodes

def nudge_neighbors(all_nodes_list, num_graph_walks=2, neighbor_buffer=1.0, nudge_step_size=0.1):
    """
    Iteratively nudge non-priority nodes away from their nearest neighbor if too close.
    Priority nodes (id > 10000) remain fixed.
    
    Args:
        all_nodes_list: List of [name, id, x, y] nodes
        num_graph_walks: Number of iterations through the graph
        neighbor_buffer: Distance threshold that triggers nudging
        nudge_step_size: How far to nudge each step
    
    Returns:
        Modified list with adjusted node positions
    """
    nodes = [n.copy() for n in all_nodes_list]
    
    def distance(n1, n2):
        return ((n1[2] - n2[2])**2 + (n1[3] - n2[3])**2)**0.5
    
    def find_nearest_neighbor(node, all_nodes):
        nearest = None
        min_dist = float('inf')
        for other in all_nodes:
            if other[1] == node[1]:
                continue
            d = distance(node, other)
            if d < min_dist:
                min_dist = d
                nearest = other
        return nearest, min_dist
    
    for walk in range(num_graph_walks):
        for node in tqdm(nodes):
            # Skip priority nodes - they stay fixed
            if node[1] > 100000:
                continue
                
            nearest, dist = find_nearest_neighbor(node, nodes)
            
            if nearest is not None and dist < neighbor_buffer:
                dx = node[2] - nearest[2]
                dy = node[3] - nearest[3]
                
                if dist > 0:
                    dx = (dx / dist) * nudge_step_size
                    dy = (dy / dist) * nudge_step_size
                else:
                    dx = nudge_step_size
                    dy = 0
                
                node[2] += dx
                node[3] += dy
    
    return nodes


class RenderNetworkV2(Scene):
    def construct(self):
        with open(json_dir/"phone_dag_v2.json", "r") as f:
            data = json.load(f)

        nodes_to_render=-1



        #Maybe a little preprocessing here?
        all_nodes_list=copy.deepcopy(small_graph_nodes[:-1]) #Leave off ending node for now
        for n in all_nodes_list:
            n[1]=n[1]+100000 #Avoid collisions
            n[2]=n[2]*SCALE_FACTOR-GLOBAL_SHIFT[0]
            n[3]=n[3]*SCALE_FACTOR-GLOBAL_SHIFT[1]
        
        for node_data in data["nodes"][:nodes_to_render]:
            if node_data['highlighted']: continue #Might make weird gaps, we'll see 
            label=node_data["phoneme"]
            idx=node_data["id"]
            x = node_data["x"] * JSON_SCALE_FACTOR_X-3-GLOBAL_SHIFT[0] #Shift everything left a bit
            y = node_data["y"] * JSON_SCALE_FACTOR_Y-GLOBAL_SHIFT[1]
            all_nodes_list.append([label, idx, x, y])

        #Ok so I probably want to crank up the buffer, but will come back to this
        print(len(all_nodes_list))
        all_nodes_list=purge_priority_neighbors(all_nodes_list, neighbor_buffer=1.5)
        print(len(all_nodes_list))

        #Ok now I need some kind nudge method, that doesn't nudge priority nodes

        all_nodes_list=nudge_neighbors(all_nodes_list, num_graph_walks=4, neighbor_buffer=2.0, nudge_step_size=0.5)
        print(len(all_nodes_list))

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
            box.set_fill(opacity=1.0, color=BLACK) 
            node = VGroup(box, text)
            node.move_to([x, y, 0])
            node_mobjects[idx] = node
        

        #Reindex to avoid collisions
        all_edges=list(np.array(small_graph_edges)+100000)

        for edge_data in data["edges"]:
            if edge_data["source"]==0:
                edge_data["source"]=100000
            all_edges.append([edge_data["source"], edge_data["target"]])

        # Next need to connect my priority start node to everyone else 
        # How did this get disconnected in the first lace exactly?


        arrows = VGroup()
        arrow_dict={}
        for start_idx, end_idx in all_edges:
            if start_idx not in node_mobjects: continue
            if end_idx not in node_mobjects: continue

            start_node = node_mobjects[start_idx]
            end_node = node_mobjects[end_idx]
            
            start_box = start_node[0]  # The RoundedRectangle
            end_box = end_node[0]
            
            # Direction from start to end
            direction = end_node.get_center() - start_node.get_center()
            direction = direction / np.linalg.norm(direction)  # normalize
            
            # Get edge points
            start_point = get_rect_edge_point(start_box, direction)
            end_point = get_rect_edge_point(end_box, -direction)
            
            # Small additional buffer for visual breathing room
            gap = 0.05
            start_point = start_point + gap * direction
            end_point = end_point - gap * direction
            
            arrow = Arrow(start_point, end_point, buff=0)
            arrow.set_color(CHILL_BROWN)
            arrows.add(arrow)
            arrow_dict[(start_idx, end_idx)]=arrow

        # Group everything
        all_nodes = VGroup(*node_mobjects.values())
        graph = VGroup(arrows, all_nodes)


        self.add(arrows)
        self.wait()


        self.add(all_nodes)
        self.wait()


        node_objects_non_priority=[node_mobjects[i] for i in node_mobjects.keys() if i<100000]
        arrows_non_priority={}
        for k, v in arrow_dict.items():
            if k[0]<100000 or k[1]<100000:
                arrows_non_priority[k]=v

        non_priority_nodes_group=VGroup(*node_objects_non_priority)
        non_priority_arrows_group=VGroup(*arrows_non_priority.values())


        self.wait()





        self.frame.reorient(0, 0, 0, (-0.11, 0.17, 0.0), 14.76)


        self.wait()
        self.remove(non_priority_nodes_group)
        self.remove(non_priority_arrows_group)


        self.wait(20)
        self.embed()













