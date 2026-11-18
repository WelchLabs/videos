import manimgx as gx
from _2026.the_bitter_lesson.go_gx import create_stone, find_captures, get_board as make_go_board, parse_sgf
from welch_scene import WelchScene
from tqdm import tqdm
import re
from itertools import cycle, islice
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

self_games_dir = Path(__file__).resolve().parent / 'alpha_go_self_play'
# human_games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/human_human_kgs-19-2015')
# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/games_with_videos')
# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/less_wrong_reverse_engineer')


size = 19  # 19x19 board
padding = 0.5
board_width = 8 # Total width in Manim units
step = board_width / size




def get_board():
    return make_go_board(2, grid_stroke_width=1.5)






class GamesGridThree(WelchScene):
    def construct(self):
        # Grid configuration
        N = 5  # NxN grid of games
        spacing = board_width + 1.0  # Space between board centers
        
        # Camera start and end positions
        # Format: theta, phi, gamma, (center_x, center_y, center_z), height
        # cam_start = (0, 0, 0, (0, 0, 0), 30)  # Start: top-down view, zoomed out
        
        # cam_end = (0, 0, 0, (-0.11, -2.82, 0.0), 13.77)  # End position

        #For 8x8
        # cam_start =(42, 53, 0, (-1.75, 0.64, 2.08), 5.56)
        # cam_end = (0, 0, 0, (-0.93, -0.67, 0.0), 75.13)

        # 5x5
        cam_start=(44, 49, 0, (0.49, -0.77, -1.31), 8.91)
        cam_end=(0, 0, 0, (-0.56, -0.14, 0.0), 46.47)

        # Load game files
        self_games_files = sorted(self_games_dir.glob('*.sgf'))
        if not self_games_files:
            raise FileNotFoundError(f'No SGF files in {self_games_dir}')
        
        # Make sure we have enough games
        num_games = N * N
        if len(self_games_files) < num_games:
            print(f"Warning: Only {len(self_games_files)} games available, need {num_games}")
        
        # Parse all games
        all_moves = [parse_sgf(path) for path in islice(cycle(self_games_files), num_games)]
        
        # Find the maximum number of moves across all games
        max_moves = max(len(moves) for moves in all_moves)
        
        # Create boards and position them in a grid
        board_groups = []
        board_states = []
        stone_objects_list = []
        
        for row in range(N):
            for col in range(N):
                board = get_board()
                board_group = gx.Group()
                board_group.add(board)
                
                # Position the board in the grid
                # Center the grid around origin
                offset_x = (col - (N - 1) / 2) * spacing
                offset_y = (row - (N - 1) / 2) * spacing
                board_group.move_to([offset_x, offset_y, 0])
                
                self.add(board_group)
                board_groups.append(board_group)
                board_states.append({})  # (x, y) -> color
                stone_objects_list.append({})  # (x, y) -> Mobject
        
        # Set initial camera position
        self.set_view(cam_start[0], cam_start[1], cam_start[2], cam_start[3], cam_start[4])
        
        # Play all games simultaneously with camera movement
        # Camera updates more frequently than board moves for smooth motion
        camera_steps_per_move = 10
        total_camera_steps = max_moves * camera_steps_per_move
        
        self.wait()
        for step_idx in range(total_camera_steps):
            # Calculate interpolation factor for camera
            t = step_idx / max(total_camera_steps - 1, 1)
            
            # Interpolate camera parameters
            theta = cam_start[0] + t * (cam_end[0] - cam_start[0])
            phi = cam_start[1] + t * (cam_end[1] - cam_start[1])
            gamma = cam_start[2] + t * (cam_end[2] - cam_start[2])
            center_x = cam_start[3][0] + t * (cam_end[3][0] - cam_start[3][0])
            center_y = cam_start[3][1] + t * (cam_end[3][1] - cam_start[3][1])
            center_z = cam_start[3][2] + t * (cam_end[3][2] - cam_start[3][2])
            height = cam_start[4] + t * (cam_end[4] - cam_start[4])
            
            # Update camera
            self.set_view(theta, phi, gamma, (center_x, center_y, center_z), height)
            
            # Only update boards every camera_steps_per_move steps
            if step_idx % camera_steps_per_move == 0:
                move_idx = step_idx // camera_steps_per_move
                
                # Update each game
                for game_idx in range(num_games):
                    moves = all_moves[game_idx]
                    
                    # Skip if this game has no more moves
                    if move_idx >= len(moves):
                        continue
                    
                    x, y, color = moves[move_idx]
                    board_group = board_groups[game_idx]
                    board_state = board_states[game_idx]
                    stone_objects = stone_objects_list[game_idx]
                    
                    # Create stone at local position
                    stone = create_stone(x, y, color)
                    
                    # Get board offset to position stone correctly
                    board_center = board_group.get_center()
                    stone.shift(board_center)
                    
                    board_group.add(stone)
                    
                    board_state[(x, y)] = color
                    stone_objects[(x, y)] = stone
                    
                    # Check for captures
                    captured = find_captures(x, y, board_state)
                    for cx, cy in captured:
                        # Remove from scene and state
                        board_group.remove(stone_objects[(cx, cy)])
                        del board_state[(cx, cy)]
                        del stone_objects[(cx, cy)]
            
            self.wait(0.01)

        self.wait(20)


class GamesGridTwo(WelchScene):
    def construct(self):
        # Grid configuration
        N = 8  # NxN grid of games
        spacing = board_width + 1.0  # Space between board centers
        
        # Camera start and end positions
        # Format: theta, phi, gamma, (center_x, center_y, center_z), height
        # cam_start = (0, 0, 0, (0, 0, 0), 30)  # Start: top-down view, zoomed out
        cam_start =(42, 53, 0, (-1.75, 0.64, 2.08), 5.56)
        # cam_end = (0, 0, 0, (-0.11, -2.82, 0.0), 13.77)  # End position
        cam_end = (0, 0, 0, (-0.93, -0.67, 0.0), 75.13)
        
        # Load game files
        self_games_files = sorted(list(self_games_dir.glob('*.sgf')))+sorted(list(self_games_dir.glob('*.sgf')))
        
        # Make sure we have enough games
        num_games = N * N
        if len(self_games_files) < num_games:
            print(f"Warning: Only {len(self_games_files)} games available, need {num_games}")
            # Cycle through available games if not enough
            while len(self_games_files) < num_games:
                self_games_files = self_games_files + self_games_files
        
        # Parse all games
        all_moves = []
        for i in range(num_games):
            moves = parse_sgf(self_games_files[i])
            all_moves.append(moves)
        
        # Find the maximum number of moves across all games
        max_moves = max(len(moves) for moves in all_moves)
        
        # Create boards and position them in a grid
        board_groups = []
        board_states = []
        stone_objects_list = []
        
        for row in range(N):
            for col in range(N):
                board = get_board()
                board_group = gx.Group()
                board_group.add(board)
                
                # Position the board in the grid
                # Center the grid around origin
                offset_x = (col - (N - 1) / 2) * spacing
                offset_y = (row - (N - 1) / 2) * spacing
                board_group.move_to([offset_x, offset_y, 0])
                
                self.add(board_group)
                board_groups.append(board_group)
                board_states.append({})  # (x, y) -> color
                stone_objects_list.append({})  # (x, y) -> Mobject
        
        # Set initial camera position
        self.set_view(cam_start[0], cam_start[1], cam_start[2], cam_start[3], cam_start[4])

        # self.set_view(cam_end[0], cam_end[1], cam_end[2], cam_end[3], cam_end[4])
        
        # Play all games simultaneously with camera movement
        self.wait()
        for move_idx in range(max_moves):
            # Calculate interpolation factor for camera
            t = move_idx / max(max_moves - 1, 1)
            
            # Interpolate camera parameters
            theta = cam_start[0] + t * (cam_end[0] - cam_start[0])
            phi = cam_start[1] + t * (cam_end[1] - cam_start[1])
            gamma = cam_start[2] + t * (cam_end[2] - cam_start[2])
            center_x = cam_start[3][0] + t * (cam_end[3][0] - cam_start[3][0])
            center_y = cam_start[3][1] + t * (cam_end[3][1] - cam_start[3][1])
            center_z = cam_start[3][2] + t * (cam_end[3][2] - cam_start[3][2])
            height = cam_start[4] + t * (cam_end[4] - cam_start[4])
            
            # Update camera
            self.set_view(theta, phi, gamma, (center_x, center_y, center_z), height)
            
            # Update each game
            for game_idx in range(num_games):
                moves = all_moves[game_idx]
                
                # Skip if this game has no more moves
                if move_idx >= len(moves):
                    continue
                
                x, y, color = moves[move_idx]
                board_group = board_groups[game_idx]
                board_state = board_states[game_idx]
                stone_objects = stone_objects_list[game_idx]
                
                # Create stone at local position
                stone = create_stone(x, y, color)
                
                # Get board offset to position stone correctly
                board_center = board_group.get_center()
                stone.shift(board_center)
                
                board_group.add(stone)
                
                board_state[(x, y)] = color
                stone_objects[(x, y)] = stone
                
                # Check for captures
                captured = find_captures(x, y, board_state)
                for cx, cy in captured:
                    # Remove from scene and state
                    board_group.remove(stone_objects[(cx, cy)])
                    del board_state[(cx, cy)]
                    del stone_objects[(cx, cy)]
            
            self.wait(0.1)

        self.wait(20)




class GamesGridOne(WelchScene):
    def construct(self):

        self_games_files=sorted(list(self_games_dir.glob('*.sgf')))

        game_index=0
        p=self_games_files[game_index]
        moves = parse_sgf(p)

        board=get_board()
        board_group=gx.Group()
        board_group.add(board)
        self.add(board_group)

        board_state = {}  # (x, y) -> color
        stone_objects = {}  # (x, y) -> Mobject
        for i, (x, y, color) in enumerate(moves):
            stone = create_stone(x, y, color)
            board_group.add(stone)
            
            board_state[(x, y)] = color
            stone_objects[(x, y)] = stone
            
            # Check for captures
            captured = find_captures(x, y, board_state)
            for cx, cy in captured:
                # Remove from scene and state
                board_group.remove(stone_objects[(cx, cy)])
                del board_state[(cx, cy)]
                del stone_objects[(cx, cy)]
            self.wait(0.1)



        self.wait()



        self.wait(20)
