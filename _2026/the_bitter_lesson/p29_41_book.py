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
# games_dir=Path('/Volumes/hot_1/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/games_with_videos')
games_dir=Path(welch_path('bitter_lesson/games/less_wrong_reverse_engineer'))

svg_dir=Path(welch_path('bitter_lesson/graphics/to_manim/'))

size = 19  # 19x19 board
padding = 0.5
board_width = 8 # Total width in Manim units
step = board_width / size







def get_board(stroke_width=0.5):
    return make_go_board(stroke_width)





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
            color=CHILL_BROWN,
        )
        pyramid_edges.add(edge)
    
    # 4 edges around the top square
    for i in range(4):
        next_i = (i + 1) % 4
        edge = gx.Line3D(
            start=top_corners[i],
            end=top_corners[next_i],
            thickness=line_width,
            color=CHILL_BROWN,
        )
        pyramid_edges.add(edge)
    
    # Create single output cube with CHILL_BROWN borders
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
            color=CHILL_BROWN,
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



class RenderPolicyNetwork(WelchScene):
    def construct(self): 

        spacing=0.8
        cnn=gx.Group()
        for i in range(4):

            layer = create_cnn_layer(
                width=19, 
                height=19, 
                cell_size=0.15, 
                depth=0.15,
                fill_color=CHILL_BLUE,
            )

            layer.rotate(90*gx.DEGREES, [1, 0, 0])
            layer[0].set_opacity(0.6)
            layer[1].set_opacity(0.6)
            layer.move_to([0, -spacing*i, 0])
            cnn.add(layer)


        #Ok this looks decent. 
        # self.add(cnn)
        cnn.move_to([0, 0, 0])
        cnn.rotate(90*gx.DEGREES, axis=gx.OUT)
        cnn.rotate(30*gx.DEGREES, axis=gx.RIGHT)
        cnn.rotate(-30*gx.DEGREES, axis=gx.UP)
        cnn.rotate(-15*gx.DEGREES, axis=gx.OUT)

        self.add(cnn)


        self.wait(20)

class RenderValueNetwork(WelchScene):
    def construct(self): 


        value_network=get_value_network()
        value_network.scale(0.72)
        value_network.rotate(90*gx.DEGREES, axis=gx.OUT)
        value_network.rotate(-10*gx.DEGREES, axis=gx.OUT)
        value_network.rotate(15*gx.DEGREES, axis=gx.RIGHT)
        value_network.rotate(-30*gx.DEGREES, axis=gx.UP)
        value_network.move_to([-7.1, -4.4, 0])

        self.set_view(-2, 0, 0, (np.float32(-6.99), np.float32(-4.37), np.float32(-0.03)), 5.28)
        self.add(value_network)

        self.wait()


        self.wait(20)


def get_board_state_after(moves, n):
    """Replay the first n moves (with captures) and return the board state dict."""
    state = {}
    for i in range(n):
        x, y, color = moves[i]
        state[(x, y)] = color
        captured = find_captures(x, y, state)
        for c in captured:
            del state[c]
    return state
 
 
def create_flat_stone(x, y, color=gx.BLACK, z_offset=0.02):
    """2D stone — much faster to render than spheres at grid scale."""
    stone_radius = step * 0.45
    pos = [(-(size - 1) / 2 + x) * step,
           (-(size - 1) / 2 + y) * step,
           z_offset]
    stone = gx.Circle(radius=stone_radius)
    if color == gx.BLACK or color == '#000000':
        stone.set_fill('#000000', opacity=1)
        stone.set_stroke('#000000', width=0.5)
    else:
        stone.set_fill('#FFFFFF', opacity=1)
        stone.set_opacity(1.0)
        # stone.set_stroke('#888888', width=0.5)
        stone.set_stroke('#FFFFFF', width=0.5)
    stone.move_to(pos)
    return stone
 


class RenderRolloutsBookGrid3(WelchScene):
    def construct(self):

        #Game files from: https://homepages.cwi.nl/~aeb/go/games/games/AlphaGo/

        game_files = ['G01.sgf',  # White wins
                      'G02.sgf',  # White wins
                      'G05.sgf',  # Black wins
                      # 'G06.sgf',  # White wins
                      # 'G07.sgf',  # White wins
                      # 'G08.sgf',  # White wins
                      'G09.sgf',  # Black wins
                      'G10.sgf',  # White wins
                      'G11.sgf']  # Black wins

        flat_stones = False

        # Layout
        cell_size = 0.85
        h_spacing = cell_size * 1.12
        v_spacing = cell_size * 1.12
        board_full_width = board_width + padding

        # Rows = snapshots, columns = games
        num_rows = 8  # len of snapshot_counts below
        num_cols = len(game_files)

        # Extra vertical gaps: 0-indexed row after which to insert a gap + ellipsis.
        # Rows come in pairs (0,1), (20,21), (100,101), (n-1,n) ->
        # gaps after rows 1, 3, 5 separate the pairs.
        gap_after_rows = [1, 3, 5]
        extra_gap = 0.15 * cell_size

        # Uniform lighting: put the light directly overhead, far away,
        # so every board sees the same angle.
        self.camera.light_source.move_to([0, 0, 300])

        # Precompute row y-positions (top row first), then center vertically
        row_y = []
        y = 0.0
        for r in range(num_rows):
            row_y.append(y)
            y -= v_spacing
            if r in gap_after_rows:
                y -= extra_gap
        y_offset = (row_y[0] + row_y[-1]) / 2
        row_y = [ry - y_offset for ry in row_y]

        col_x = [(c - (num_cols - 1) / 2) * h_spacing for c in range(num_cols)]

        grid = gx.Group()
        black_stones = gx.Group()
        white_stones = gx.Group()
        all_boards = gx.Group()

        for col, fname in enumerate(tqdm(game_files)):
            moves = parse_sgf(self_games_dir / fname)
            n = len(moves)

            snapshot_counts = [0, 1, 20, 21, 100, 101, n - 1, n]

            for row, count in enumerate(snapshot_counts):
                state = get_board_state_after(moves, count)

                cell = gx.Group()
                board = get_board(stroke_width=0.5)
                all_boards.add(board)
                cell.add(board)
                for (x, y), color in state.items():
                    if flat_stones:
                        stone = create_flat_stone(x, y, color)
                    else:
                        stone = create_stone(x, y, color, resolution=(17, 9))
                    cell.add(stone)
                    # print(color)
                    if color == '#000000':
                        black_stones.add(stone)
                    elif color == '#FFFFFF':
                        white_stones.add(stone)

                cell.scale(cell_size / board_full_width)
                cell.move_to([col_x[col], row_y[row], 0])
                grid.add(cell)

        # Ellipses in each gap, one per column
        ellipses = gx.VGroup()
        for r in gap_after_rows:
            y_mid = (row_y[r] + row_y[r + 1]) / 2
            for cx in col_x:
                dots = gx.MathTex(R"\vdots")
                dots.set_color(CHILL_BROWN)
                dots.scale_to_fit_height(extra_gap * 0.7)
                dots.move_to([cx, y_mid, 0])
                ellipses.add(dots)

        self.add(all_boards)
        self.add(black_stones, white_stones)


        self.add(ellipses)

        total_height = row_y[0] - row_y[-1] + v_spacing
        self.camera.frame.scale_to_fit_height(total_height * 1.08)
        self.set_view(0, 0, 0, (0, 0, 0), total_height * 1.08)

        self.wait()


        self.set_view(-90, 0, 0, (np.float32(-0.01), np.float32(-0.16), np.float32(-0.0)), 5.92)

        self.wait()






# class RenderRolloutsBookGrid(InteractiveScene):
#     def construct(self):
 
#         game_files = ['G01.sgf',  # White wins
#                       'G02.sgf',  # White wins
#                       'G05.sgf',  # Black wins
#                       # 'G06.sgf',  # White wins
#                       # 'G07.sgf',  # White wins
#                       # 'G08.sgf',  # White wins
#                       'G09.sgf',  # Black wins
#                       'G10.sgf',  # White wins
#                       'G11.sgf']  # Black wins
 
#         num_snapshots = 5
#         flat_stones = False  # Set False for the 3D sphere stones (slower — ~4-5k spheres)
 
#         # Layout
#         cell_size = 0.85                    # rendered width of each mini board
#         h_spacing = cell_size * 1.12
#         v_spacing = cell_size * 1.12
#         board_full_width = board_width + padding  # native board size before scaling
 
#         num_rows = len(game_files)
#         grid = Group()
 
#         all_stones=Group()
#         all_boards=Group()
#         for col, fname in enumerate(tqdm(game_files)):
#             moves = parse_sgf(self_games_dir / fname)
#             n = len(moves)
 
#             # Start, end, and 3 evenly spaced snapshots in between.
#             # First column is move 0 (empty board) — swap the 0 for e.g. 1 or n//10
#             # if you'd rather open with some stones on the board.
#             # snapshot_counts = np.round(np.linspace(0, n, num_snapshots)).astype(int)
#             snapshot_counts=[0, 1, 20, 21, 100, 101, n-1, n]
 
#             for row, count in enumerate(snapshot_counts):
#                 state = get_board_state_after(moves, count)
 
#                 cell = Group()
#                 board=get_board(stroke_width=0.5)
#                 all_boards.add(board)
#                 cell.add(board)
#                 for (x, y), color in state.items():
#                     if flat_stones:
#                         stone=create_flat_stone(x, y, color)
#                         cell.add(stone)
#                     else:
#                         stone=create_stone(x, y, color, resolution=(17, 9)) #(101, 51)
#                         cell.add(stone)
#                     all_stones.add(stone)
 
#                 cell.scale(cell_size / board_full_width)
#                 x_pos = (col - (num_snapshots - 1) / 2) * h_spacing
#                 y_pos = ((num_rows - 1) / 2 - row) * v_spacing
#                 cell.move_to([x_pos, y_pos, 0])
#                 grid.add(cell)
 
#         # self.add(grid)
#         self.camera.frame.scale_to_fit_height(num_rows * v_spacing * 1.05)

#         self.add(all_boards)

#         # self.remove(all_stones)
#         self.add(all_stones)

#         # all_stones.set_color(WHITE)
#         # all_stones.set_opacity(1.0)

#         # all_stones.shift([0, 0, 0])
#         self.set_view(0, 0, 0, (np.float32(0.43), np.float32(-0.99), np.float32(0.0)), 7.97)

 
#         self.wait()
#         pass




class RenderRolloutsBook(WelchScene):
    def construct(self): 


        board=get_board()
        self.add(board)

        p=self_games_dir/'G01.sgf' #White wins
        # p=self_games_dir/'G02.sgf' #White Wins
        # p=self_games_dir/'G05.sgf'   #Black wins
        # p=self_games_dir/'G06.sgf' #White Wins
        # p=self_games_dir/'G07.sgf'   #White wins
        # p=self_games_dir/'G08.sgf' #White Wins
        # p=self_games_dir/'G09.sgf'   #Black wins
        # p=self_games_dir/'G10.sgf' #White Wins
        # p=self_games_dir/'G11.sgf'   #Black wins



        moves = parse_sgf(p)
        stones=gx.Group()
        board_state = {}  # (x, y) -> color
        stone_objects = {}  # (x, y) -> Mobject
        for i, (x, y, color) in enumerate(moves):
            stone = create_stone(x, y, color)
            stones.add(stone)
            self.add(stone)
            self.wait(0.1) #Uncomment in FINAL RENDER
            
            board_state[(x, y)] = color
            stone_objects[(x, y)] = stone
            
            # Check for captures
            captured = find_captures(x, y, board_state)
            for cx, cy in captured:
                # Remove from scene and state
                stones.remove(stone_objects[(cx, cy)])
                self.remove(stone_objects[(cx, cy)])
                del board_state[(cx, cy)]
                del stone_objects[(cx, cy)]

        self.wait()







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














