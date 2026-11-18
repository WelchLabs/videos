import numpy as np
import manimgx as gx
from welch_scene import WelchScene
import json
from collections import defaultdict
import copy
from tqdm import tqdm

CHILL_BROWN = '#948979'

from _2026.the_bitter_lesson.phoneme_graph import (
    GLOBAL_SHIFT,
    JSON_SCALE_FACTOR_X,
    JSON_SCALE_FACTOR_Y,
    SCALE_FACTOR,
    get_rect_edge_point,
    json_dir,
    nudge_neighbors,
    purge_priority_neighbors,
    small_graph_edges,
    small_graph_nodes,
)


class P4a(WelchScene):
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

        # self.wait()

        node_mobjects = {}
        buff = 0.2
        for label, idx, x, y in all_nodes_list:
            text = gx.Text(label, font="American Typewriter", color=CHILL_BROWN)
            text.set_color(CHILL_BROWN)
            box = gx.RoundedRectangle(
                width=text.width + 2 * buff,
                height=text.height + 2 * buff,
                corner_radius=0.15,
                color=CHILL_BROWN,
            )
            box.set_fill(opacity=1.0, color=gx.BLACK)
            node = gx.VGroup(box, text)
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


        arrows = gx.VGroup()
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
            
            arrow = gx.Arrow(start_point, end_point, buff=0)
            arrow.set_color(CHILL_BROWN)
            arrows.add(arrow)
            arrow_dict[(start_idx, end_idx)]=arrow

        # Group everything
        all_nodes = gx.VGroup(*node_mobjects.values())
        graph = gx.VGroup(arrows, all_nodes)


        node_objects_non_priority=[node_mobjects[i] for i in node_mobjects.keys() if i<100000]
        arrows_non_priority={}
        for k, v in arrow_dict.items():
            if k[0]<100000 or k[1]<100000:
                arrows_non_priority[k]=v

        non_priority_nodes_group=gx.VGroup(*node_objects_non_priority)
        non_priority_arrows_group=gx.VGroup(*arrows_non_priority.values())


        # self.wait()

        #Option 1 -> wide angle side view.
        # self.set_view(54.702143403436715, 52.815862596421724, 0.0, (-38.418354, -57.3952, -45.180237), 291.11)
        # self.set_view(37.72561614030112, 61.77569642974304, 0.0, (36.621754, 7.254342, -5.611898), 69.38)
        self.set_view(46.21387977186886, 61.775696429743014, -3.1805546814635168e-15, (35.291615, 6.35425, -4.9829783), 69.38)
        self.add(arrows)
        self.add(all_nodes)


        self.wait()
        self.play(self.animate_view(0, 0, 0, (-0.11, 0.17, 0.0), 14.76),
                 run_time=30)     

        # self.set_view(0, 0, 0, (-0.11, 0.17, 0.0), 14.76)

        self.wait()
        self.remove(non_priority_arrows_group)
        self.play(
                  # FadeOut(non_priority_arrows_group),
                  gx.FadeOut(non_priority_nodes_group),
                  run_time=7)
        self.wait()
        # self.remove(non_priority_nodes_group)
        # self.remove(non_priority_arrows_group)


        self.wait(20)













