from welch_paths import welch_path
import manimgx as gx
from _2026.the_bitter_lesson.go_storyboards import P29_36 as _P29_36, P37_42b as _P37_42b
from _2026.the_bitter_lesson.go_gx import create_cnn_layer, create_stone, find_captures, get_board as make_go_board, parse_sgf
import numpy as np
import copy
from welch_scene import WelchScene, image_at_height
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

self_games_dir=Path(welch_path('bitter_lesson/games/alpha_go_self_play'))
human_games_dir=Path(welch_path('bitter_lesson/games/human_human_kgs-19-2015'))
# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/games_with_videos')
games_dir=Path(welch_path('bitter_lesson/games/less_wrong_reverse_engineer'))

svg_dir=Path(welch_path('bitter_lesson/graphics/to_manim/'))

size = 19  # 19x19 board
padding = 0.5
board_width = 8 # Total width in Manim units
step = board_width / size







def get_board():
    return make_go_board(2, grid_stroke_width=1.5)





def get_value_network():
        
    spacing = 0.8
    cell_size = 0.15
    layer_size = 19 * cell_size  # Width/height of the 19x19 grid
    
    layers = gx.Group()
    
    # Create 3 convolutional layers (green)
    for i in range(3):
        layer = create_cnn_layer(
            width=19, 
            height=19, 
            cell_size=cell_size, 
            depth=0.15,
            fill_color=CHILL_GREEN,
        )
        layer.rotate(90 * gx.DEGREES, [1, 0, 0])
        layer[0].set_color(CHILL_GREEN) #, opacity=0.6)
        layer[0].set_opacity(0.6)
        layer[1].set_opacity(0.6)
        layer.move_to([0, -spacing * i, 0])
        layers.add(layer)
    
    # Position for pyramid base (bottom of last conv layer)
    last_layer_y = -spacing * 2
    pyramid_base_y = last_layer_y - 0.15 / 2 - 0.01  # Just below last layer
    
    # Position for output cube
    output_y = last_layer_y - spacing * 1.5
    
    # Output cube size
    output_size = 0.15
    pyramid_top_y = output_y + output_size / 2 + 0.01
    
    # Create pyramid connecting last conv layer to output cube
    # After rotation, the grid is in the x-z plane
    half_size = layer_size / 2
    half_output = output_size / 2
    
    # Pyramid vertices - base is the 19x19 layer, top is the small output
    # Base corners (at pyramid_base_y)
    base_corners = [
        np.array([-half_size, pyramid_base_y, -half_size]),  # back-left
        np.array([half_size, pyramid_base_y, -half_size]),   # back-right
        np.array([half_size, pyramid_base_y, half_size]),    # front-right
        np.array([-half_size, pyramid_base_y, half_size]),   # front-left
    ]
    
    # Top corners (at pyramid_top_y) - small square for output cube
    top_corners = [
        np.array([-half_output, pyramid_top_y, -half_output]),  # back-left
        np.array([half_output, pyramid_top_y, -half_output]),   # back-right
        np.array([half_output, pyramid_top_y, half_output]),    # front-right
        np.array([-half_output, pyramid_top_y, half_output]),   # front-left
    ]
    
    # Create the 4 trapezoidal faces of the frustum/pyramid
    pyramid_faces = gx.Group()
    
    for i in range(4):
        next_i = (i + 1) % 4
        # Each face is a quadrilateral: base[i], base[next_i], top[next_i], top[i]
        face = gx.Polygon(
            base_corners[i],
            base_corners[next_i],
            top_corners[next_i],
            top_corners[i],
        )
        face.set_fill(CHILL_GREEN, opacity=0.7)
        face.set_stroke(width=0)  # No borders on pyramid
        pyramid_faces.add(face)
    
    # Add Line3D borders to pyramid edges
    pyramid_edges = gx.Group()
    line_width = 0.02
    
    # 4 angled edges connecting base to top corners
    for i in range(4):
        edge = gx.Line3D(
            start=base_corners[i],
            end=top_corners[i],
            thickness=line_width,
            color=gx.WHITE,
        )
        pyramid_edges.add(edge)
    
    # 4 edges around the top square
    for i in range(4):
        next_i = (i + 1) % 4
        edge = gx.Line3D(
            start=top_corners[i],
            end=top_corners[next_i],
            thickness=line_width,
            color=gx.WHITE,
        )
        pyramid_edges.add(edge)
    
    # Create single output cube with white borders
    output_cube = gx.Cube(side_length=output_size)
    output_cube.set_color(CHILL_GREEN)
    output_cube.move_to([0, output_y, 0])
    
    # Add Line3D borders to output cube
    cube_edges = gx.Group()
    s = output_size / 2
    
    # Define the 8 corners of the cube
    cube_corners = [
        np.array([-s, output_y - s, -s]),  # 0: bottom-back-left
        np.array([s, output_y - s, -s]),   # 1: bottom-back-right
        np.array([s, output_y - s, s]),    # 2: bottom-front-right
        np.array([-s, output_y - s, s]),   # 3: bottom-front-left
        np.array([-s, output_y + s, -s]),  # 4: top-back-left
        np.array([s, output_y + s, -s]),   # 5: top-back-right
        np.array([s, output_y + s, s]),    # 6: top-front-right
        np.array([-s, output_y + s, s]),   # 7: top-front-left
    ]
    
    # Define the 12 edges of the cube (pairs of corner indices)
    cube_edge_pairs = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    
    for start_idx, end_idx in cube_edge_pairs:
        edge = gx.Line3D(
            start=cube_corners[start_idx],
            end=cube_corners[end_idx],
            thickness=line_width,
            color=gx.WHITE,
        )
        cube_edges.add(edge)
    
    # Add everything to scene
    # self.add(layers, pyramid_faces, pyramid_edges, output_cube, cube_edges)
    # self.remove(pyramid_faces); self.add(pyramid_faces)
    # self.remove(cube_edges[6]); self.add(cube_edges[6])
    # self.remove(cube_edges[7]); self.add(cube_edges[7])
    # self.remove(cube_edges[11]); self.add(cube_edges[11])

    value_network=gx.Group()
    value_network.add(layers)
    value_network.add(pyramid_edges)
    value_network.add(output_cube)
    value_network.add(pyramid_faces[1:]) #Super hacky but leave off one side, cuasing occlusion issues 
    value_network.add(cube_edges)

    return value_network



class P29_36(_P29_36):
    self_games_dir = self_games_dir
    human_games_dir = human_games_dir
    svg_dir = svg_dir
    get_board = staticmethod(get_board)
    get_value_network = staticmethod(get_value_network)



class P37_42b(_P37_42b):
    self_games_dir = self_games_dir
    human_games_dir = human_games_dir
    svg_dir = svg_dir
    get_board = staticmethod(get_board)
    get_value_network = staticmethod(get_value_network)
















        # board_4=get_board()


        # self.wait()
        # self.remove(board_1, board_2, next_move_1)
        # self.add(board_3, board_4)
        # self.wait()
        # self.wait()


        #Ok that looks nice -> now how do we want to bring in the LLM action?

        # border_2 = RoundedRectangle(
        #     width=4.8,
        #     height=4.8,
        #     corner_radius=0.2,
        #     stroke_color=CHILL_BROWN,
        #     stroke_width=5,
        #     fill_opacity=0,
        # )
        # border_2.move_to([-0.1, -6.2, 0])

        # self.add(all_svgs[0])
        # self.add(all_svgs[3:])
        # self.add(all_svgs[1])
        # self.add(all_svgs[2]) #[1:])
        # self.add(border_2)














