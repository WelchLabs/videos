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

self_games_dir='_2026/the_bitter_lesson/alpha_go_self_play'
# human_games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/human_human_kgs-19-2015')
# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/games_with_videos')
# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/less_wrong_reverse_engineer')


size = 19  # 19x19 board
padding = 0.5
board_width = 8 # Total width in Manim units
step = board_width / size

def parse_sgf(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Regex to find B[xx] or W[xx]
    moves = re.findall(r"([BW])\[([a-s]{2})\]", content)
    
    parsed_moves = []
    for color_char, coords in moves:
        # Convert letters to 0-18 integers
        x = ord(coords[0]) - ord('a')
        y = ord(coords[1]) - ord('a')
        # Flip Y because SGF is top-down, Manim is bottom-up
        y = 18 - y 
        
        color = BLACK if color_char == 'B' else WHITE
        parsed_moves.append((x, y, color))
    return parsed_moves

def create_stone(x, y, color=BLACK, squash = 0.3):
    """Create a 3D stone at the given grid position."""
    stone_radius=step*0.45
    pos = [(-(size-1)/2 + x) * step, (-(size-1)/2 + y) * step, 0]
    
    if color == BLACK:
        stone = Sphere(radius=stone_radius)
        # stone.set_color("#1a1a1a")
        # stone.set_color("#222222")
        stone.set_color('#000000')
        # stone.set_shading(0.3, 0.8, 0.2)  # ambient, diffuse, specular
        stone.set_shading(0.1, 0.4, 0.1)
    else:  # white
        stone = Sphere(radius=stone_radius)
        stone.set_color("#B0B0B0")
        # stone.set_shading(0.5, 0.7, 0.3)
        stone.set_shading(0.7, 0.9, 0.9)

    # Squash in z-direction to make it stone-shaped
    stone.scale([1, 1, squash])
    
    # Lift it slightly so it sits on the board
    pos[2] = stone_radius * squash
    stone.move_to(pos)
    
    return stone


def get_board():
    board_rect = Square(side_length=board_width + padding)
    board_rect.set_fill(FRESH_TAN, opacity=1)
    board_rect.set_stroke(CHILL_BROWN, width=2)

    # We center the grid so the middle intersection is at (0,0,0)
    lines = VGroup()
    start_point = -(size - 1) / 2 * step
    
    for i in range(size):
        # Vertical lines
        v_line = Line(
            [start_point + i * step, start_point, 0],
            [start_point + i * step, -start_point, 0]
        )
        # Horizontal lines
        h_line = Line(
            [start_point, start_point + i * step, 0],
            [-start_point, start_point + i * step, 0]
        )
        lines.add(v_line, h_line)
        
    lines.set_stroke(BLACK, width=1.5)

    # For a 19x19, these are usually at 4, 10, and 16 (1-indexed)
    hoshi_indices = [3, 9, 15] # 0-indexed
    hoshi_dots = VGroup()
    for x in hoshi_indices:
        for y in hoshi_indices:
            dot = Circle(radius=0.05, fill_color=BLACK, fill_opacity=1, stroke_width=0)
            # Position the dot based on grid coordinates
            dot.move_to([start_point + x * step, start_point + y * step, 0])
            hoshi_dots.add(dot)

    board=Group()
    board.add(board_rect)
    board.add(lines)
    board.add(hoshi_dots)

    return board

def get_neighbors(x, y, size=19):
    """Return orthogonally adjacent positions within bounds."""
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            neighbors.append((nx, ny))
    return neighbors

def get_group(x, y, board_state):
    """Find all connected stones of the same color using flood fill."""
    color = board_state.get((x, y))
    if color is None:
        return set()
    
    group = set()
    stack = [(x, y)]
    
    while stack:
        cx, cy = stack.pop()
        if (cx, cy) in group:
            continue
        if board_state.get((cx, cy)) == color:
            group.add((cx, cy))
            for nx, ny in get_neighbors(cx, cy):
                if (nx, ny) not in group:
                    stack.append((nx, ny))
    return group

def get_liberties(group, board_state):
    """Count empty intersections adjacent to the group."""
    liberties = set()
    for x, y in group:
        for nx, ny in get_neighbors(x, y):
            if (nx, ny) not in board_state:
                liberties.add((nx, ny))
    return liberties

def find_captures(x, y, board_state):
    """After placing a stone at (x,y), find any captured opponent stones."""
    color = board_state.get((x, y))
    opponent = WHITE if color == BLACK else BLACK
    
    captured = set()
    for nx, ny in get_neighbors(x, y):
        if board_state.get((nx, ny)) == opponent:
            group = get_group(nx, ny, board_state)
            if len(get_liberties(group, board_state)) == 0:
                captured.update(group)
    
    return captured


class GamesGridThree(InteractiveScene):
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
        self_games_files = sorted(list(Path.cwd().glob(self_games_dir+'/*.sgf')))+sorted(list(Path.cwd().glob(self_games_dir+'/*.sgf')))
        
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
                board_group = Group()
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
        self.frame.reorient(cam_start[0], cam_start[1], cam_start[2], cam_start[3], cam_start[4])
        
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
            self.frame.reorient(theta, phi, gamma, (center_x, center_y, center_z), height)
            
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
                    if color == '#000000':
                        stone.set_shading(0.15, 0.05, 0)
                    
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
        self.embed()


class GamesGridTwo(InteractiveScene):
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
                board_group = Group()
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
        self.frame.reorient(cam_start[0], cam_start[1], cam_start[2], cam_start[3], cam_start[4])

        # self.frame.reorient(cam_end[0], cam_end[1], cam_end[2], cam_end[3], cam_end[4])
        
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
            self.frame.reorient(theta, phi, gamma, (center_x, center_y, center_z), height)
            
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
                if color == '#000000':
                    stone.set_shading(0.15, 0.05, 0)
                
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
        self.embed()




class GamesGridOne(InteractiveScene):
    def construct(self):

        self_games_files=sorted(list(self_games_dir.glob('*.sgf')))

        game_index=0
        p=self_games_files[game_index]
        moves = parse_sgf(p)

        board=get_board()
        board_group=Group()
        board_group.add(board)
        self.add(board_group)

        board_state = {}  # (x, y) -> color
        stone_objects = {}  # (x, y) -> Mobject
        for i, (x, y, color) in enumerate(moves):
            stone = create_stone(x, y, color)
            if color=='#000000':
                stone.set_shading(0.15, 0.05, 0) #Noodling here to make things consistent-ish
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
        self.embed()