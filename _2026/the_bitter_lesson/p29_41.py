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

self_games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/alpha_go_self_play')
human_games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/human_human_kgs-19-2015')
# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/games_with_videos')
games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/less_wrong_reverse_engineer')

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/graphics/to_manim/')

size = 19  # 19x19 board
padding = 0.5
board_width = 8 # Total width in Manim units
step = board_width / size

heatmap_1 = np.array([
    [0.17, 0.16, 0.16, 0.16, 0.16, 0.15, 0.16, 0.15, 0.16, 0.15, 0.15, 0.15, 0.15, 0.15, 0.16, 0.17, 0.16, 0.16, 0.17],
    [0.17, 0.17, 0.17, 0.18, 0.19, 0.19, 0.17, 0.17, 0.17, 0.16, 0.16, 0.15, 0.17, 0.17, 0.20, 0.20, 0.19, 0.19, 0.16],
    [0.17, 0.17, 0.30, 0.30, 0.22, 0.28, 0.21, 0.19, 0.18, 0.17, 0.17, 0.19, 0.20, 0.22, 0.27, 0.96, 0.71, 0.20, 0.17],
    [0.16, 0.18, 0.26, 0.01, 0.24, 0.21, 0.19, 0.18, 0.17, 0.17, 0.17, 0.17, 0.29, 0.21, 0.24, 0.78, 0.97, 0.22, 0.17],
    [0.16, 0.17, 0.20, 0.22, 0.19, 0.17, 0.17, 0.17, 0.18, 0.20, 0.67, 0.02, 0.20, 0.19, 0.20, 0.29, 0.35, 0.22, 0.16],
    [0.16, 0.17, 0.20, 0.19, 0.17, 0.16, 0.16, 0.17, 0.16, 0.48, 0.03, 0.01, 0.00, 0.17, 0.19, 0.20, 0.25, 0.19, 0.15],
    [0.16, 0.17, 0.19, 0.17, 0.17, 0.16, 0.17, 0.15, 0.15, 0.03, 0.02, 0.01, 0.01, 0.17, 0.19, 0.20, 0.21, 0.17, 0.15],
    [0.15, 0.16, 0.17, 0.18, 0.17, 0.16, 0.16, 0.16, 0.48, 0.51, 0.07, 0.04, 0.28, 0.20, 0.19, 0.19, 0.20, 0.17, 0.15],
    [0.15, 0.15, 0.17, 0.17, 0.17, 0.16, 0.16, 0.17, 0.51, 0.52, 0.78, 0.98, 0.23, 0.20, 0.19, 0.20, 0.19, 0.17, 0.15],
    [0.15, 0.16, 0.17, 0.17, 0.17, 0.15, 0.16, 0.18, 0.20, 0.27, 0.22, 0.20, 0.20, 0.20, 0.18, 0.19, 0.19, 0.17, 0.15],
    [0.15, 0.15, 0.17, 0.17, 0.17, 0.17, 0.18, 0.19, 0.20, 0.20, 0.20, 0.19, 0.19, 0.20, 0.18, 0.20, 0.20, 0.17, 0.15],
    [0.15, 0.15, 0.17, 0.19, 0.18, 0.17, 0.19, 0.20, 0.20, 0.18, 0.19, 0.19, 0.19, 0.19, 0.18, 0.20, 0.20, 0.18, 0.15],
    [0.15, 0.15, 0.17, 0.17, 0.17, 0.18, 0.20, 0.19, 0.18, 0.18, 0.18, 0.19, 0.19, 0.18, 0.17, 0.19, 0.20, 0.17, 0.15],
    [0.15, 0.14, 0.15, 0.25, 0.17, 0.18, 0.19, 0.18, 0.18, 0.18, 0.19, 0.19, 0.19, 0.18, 0.18, 0.20, 0.20, 0.18, 0.16],
    [0.15, 0.15, 0.00, 0.35, 0.17, 0.16, 0.17, 0.16, 0.17, 0.17, 0.17, 0.19, 0.18, 0.18, 0.19, 0.23, 0.22, 0.19, 0.16],
    [0.15, 0.17, 0.24, 0.44, 0.20, 0.17, 0.16, 0.16, 0.16, 0.17, 0.17, 0.17, 0.18, 0.19, 0.23, 0.44, 0.40, 0.19, 0.17],
    [0.15, 0.17, 0.30, 0.29, 0.21, 0.16, 0.17, 0.16, 0.17, 0.17, 0.17, 0.17, 0.18, 0.18, 0.22, 0.38, 0.25, 0.19, 0.17],
    [0.15, 0.15, 0.17, 0.18, 0.17, 0.16, 0.16, 0.16, 0.16, 0.15, 0.15, 0.15, 0.16, 0.17, 0.19, 0.19, 0.18, 0.17, 0.16],
    [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.14, 0.15, 0.15, 0.15, 0.16, 0.16, 0.17, 0.16, 0.17]
])

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



def create_stone(x, y, color=BLACK):
    """Create a 3D stone at the given grid position."""
    stone_radius=step*0.45
    pos = [(-(size-1)/2 + x) * step, (-(size-1)/2 + y) * step, 0]
    squash = 0.3  # How flat the stone is
    
    if color == BLACK:
        stone = Sphere(radius=stone_radius)
        # stone.set_color("#1a1a1a")
        # stone.set_color("#222222")
        stone.set_color(BLACK)
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

def create_cnn_layer(width=19, height=19, cell_size=0.15, depth=0.1, 
                     fill_color=BLUE, fill_opacity=0.8, line_width=0.02):
    """Create a single CNN layer as a flat prism with grid lines."""
    
    layer_w = width * cell_size
    layer_h = height * cell_size
    
    # Main box
    box = Cube(side_length=1)
    box.set_width(layer_h, stretch=True)
    box.set_depth(depth, stretch=True)
    box.set_height(layer_w, stretch=True)
    
    grid_lines = Group()
    
    front_z = depth / 2 + 0.001
    back_z = -depth / 2 - 0.001
    
    # Front face grid - vertical lines
    for i in range(width + 1):
        x = -layer_w / 2 + i * cell_size
        line = Line3D(
            start=np.array([x, -layer_h / 2, front_z]),
            end=np.array([x, layer_h / 2, front_z]),
            width=line_width, color=WHITE,
        )
        grid_lines.add(line)
    
    # Front face grid - horizontal lines
    for j in range(height + 1):
        y = -layer_h / 2 + j * cell_size
        line = Line3D(
            start=np.array([-layer_w / 2, y, front_z]),
            end=np.array([layer_w / 2, y, front_z]),
            width=line_width, color=WHITE,
        )
        grid_lines.add(line)
    
    # Back face grid - vertical lines
    for i in range(width + 1):
        x = -layer_w / 2 + i * cell_size
        line = Line3D(
            start=np.array([x, -layer_h / 2, back_z]),
            end=np.array([x, layer_h / 2, back_z]),
            width=line_width, color=WHITE,
        )
        grid_lines.add(line)
    
    # Back face grid - horizontal lines
    for j in range(height + 1):
        y = -layer_h / 2 + j * cell_size
        line = Line3D(
            start=np.array([-layer_w / 2, y, back_z]),
            end=np.array([layer_w / 2, y, back_z]),
            width=line_width, color=WHITE,
        )
        grid_lines.add(line)
    
    # Edge lines connecting front to back (top and bottom edges)
    for i in range(width + 1):
        x = -layer_w / 2 + i * cell_size
        # Top edge
        line = Line3D(
            start=np.array([x, layer_h / 2, front_z]),
            end=np.array([x, layer_h / 2, back_z]),
            width=line_width, color=WHITE,
        )
        grid_lines.add(line)
        # Bottom edge
        line = Line3D(
            start=np.array([x, -layer_h / 2, front_z]),
            end=np.array([x, -layer_h / 2, back_z]),
            width=line_width, color=WHITE,
        )
        grid_lines.add(line)
    
    # Edge lines connecting front to back (left and right edges)
    for j in range(height + 1):
        y = -layer_h / 2 + j * cell_size
        # Right edge
        line = Line3D(
            start=np.array([layer_w / 2, y, front_z]),
            end=np.array([layer_w / 2, y, back_z]),
            width=line_width, color=WHITE,
        )
        grid_lines.add(line)
        # Left edge
        line = Line3D(
            start=np.array([-layer_w / 2, y, front_z]),
            end=np.array([-layer_w / 2, y, back_z]),
            width=line_width, color=WHITE,
        )
        grid_lines.add(line)
    
    layer = Group(box, grid_lines)
    return layer



def render_example_go_game_1():
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

    moves=[
        # White Stones (#FFFFFF)
        (3, 15, '#FFFFFF'),  # Top left stone
        (2, 4, '#FFFFFF'),   # Bottom left stone
        # (10, 13, '#FFFFFF'), # Center cluster
        (11, 13, '#FFFFFF'), # Center cluster
        (10, 12, '#FFFFFF'), # Center cluster
        (11, 12, '#FFFFFF'), # Center cluster
        (10, 11, '#FFFFFF'), # Center cluster
        
        # Black Stones (#000000)
        (11, 14, '#000000'), # Center cluster
        (10, 13, '#000000'), # Center cluster (Overlap/Contact)
        (12, 13, '#000000'), # Center cluster
        (9, 12, '#000000'),  # Center cluster
        (12, 12, '#000000'), # Center cluster
        (11, 11, '#000000')  # Center cluster
    ]

    board=Group()
    board.add(board_rect)
    board.add(lines)
    board.add(hoshi_dots)

    for i, (x, y, color) in enumerate(moves):
        stone = create_stone(x, y, color)
        board.add(stone)
    
    return board


def create_heatmap_overlay(heatmap_data): #, opacity=0.6):
    """Create a heatmap overlay for a Go board using viridis colormap."""
    cmap = plt.cm.viridis
    
    heatmap_group = VGroup()
    start_point = -(size - 1) / 2 * step
    
    for i in range(size):
        for j in range(size):
            # heatmap array is top-down, manim is bottom-up, so flip j
            value = heatmap_data[18 - j, i]
            
            # Get RGB from viridis colormap
            rgb = cmap(value)[:3]
            hex_color = rgb_to_hex(rgb)
            
            square = Square(side_length=step)
            square.set_fill(hex_color) #, opacity=opacity * value)  # opacity scales with value
            square.set_stroke(width=0)
            
            # Position at grid intersection
            x = start_point + i * step
            y = start_point + j * step
            square.move_to([x, y, 0])
            
            heatmap_group.add(square)
    
    return heatmap_group

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

class P29_41(InteractiveScene):
    def construct(self): 
        '''
        Ok leaning towards doing a bit more in manim than I first thought, 
        mostly becuase I think there's a path that's not so bad
        Let me see how this goes/feels, and I can fall back to less Manim 
        if I need to. 
        '''

        # alphago_logo=SVGMobject(str(svg_dir/'alpha_go_logo.svg'))
        alphago_logo=ImageMobject(str(svg_dir/'alpha_go_logo.png'))
        alphago_logo.scale(0.25)
        alphago_logo.move_to([-0.15, 3, 0])


        spacing=0.8
        cnn=Group()
        for i in range(4):

            layer = create_cnn_layer(
                width=19, 
                height=19, 
                cell_size=0.15, 
                depth=0.15,
                fill_color=CHILL_BLUE,
            )

            layer.rotate(90*DEGREES, [1, 0, 0])
            layer[0].set_opacity(0.6)
            layer[1].set_opacity(0.6)
            layer.move_to([0, -spacing*i, 0])
            cnn.add(layer)


        self.wait()


        #Ok this looks decent. 
        # self.add(cnn)
        cnn.move_to([0, 0, 0])
        cnn.rotate(90*DEGREES, axis=OUT) 
        cnn.rotate(30*DEGREES, axis=RIGHT) 
        cnn.rotate(-30*DEGREES, axis=UP)
        cnn.rotate(-15*DEGREES, axis=OUT)

        border = RoundedRectangle(
            width=4.8,
            height=4.8,
            corner_radius=0.2,
            stroke_color=CHILL_BROWN,
            stroke_width=5,
            fill_opacity=0,
        )
        border.move_to([-0.1, -0.2, 0])

        # Add label text below
        label = Text(
            "SUPERVISED POLICY NETWORK (CNN)",
            font="Myriad Pro",
            font_size=32,
        )
        label.set_color(CHILL_BROWN)
        label.next_to(border, DOWN, buff=0.3)

        self.wait()
        self.play(FadeIn(alphago_logo))
        self.play(ShowCreation(cnn), Write(border), Write(label), run_time=5)

        # self.add(border)
        # self.add(label)

        self.wait()

        arrow_in = Arrow(
            border.get_left() + LEFT * 0.85,
            border.get_left(),
            stroke_width=5,
            stroke_color=CHILL_BROWN,
            fill_color=CHILL_BROWN,
            buff=0.2,
        )
        

        arrow_out = Arrow(
            border.get_right(),
            border.get_right() + RIGHT * 0.85,
            stroke_width=5,
            stroke_color=CHILL_BROWN,
            fill_color=CHILL_BROWN,
            buff=0.2,
        )

        #Ok now we bring in the actualy go board renderig?
        board_1=render_example_go_game_1()
        board_1.scale(0.4)
        board_1.move_to([-5, -0.3, 0])

        board_2=render_example_go_game_1()
        board_2.scale(0.4)
        board_2.move_to([5, -0.3, 0])
        board_2.set_opacity(0.5)


        policy_label=Text("POLICY", font="Myriad Pro", weight='bold', font_size=44)
        policy_label.set_color(YELLOW)
        policy_label.move_to([0, -3.5, 0])

        state_label=Text("STATE", font="Myriad Pro", weight='bold', font_size=44)
        state_label.set_color(YELLOW)
        state_label.move_to([-4, -0.2, 0])

        action_label=Text("ACTION", font="Myriad Pro", weight='bold', font_size=44)
        action_label.set_color(YELLOW)
        action_label.move_to([4, -0.2, 0])

        self.wait()
        self.play(FadeIn(policy_label), 
                  FadeIn(state_label), 
                  FadeIn(action_label), 
                  FadeIn(arrow_in), 
                  FadeIn(arrow_out))

        self.wait()

        self.play(state_label.animate.move_to([-5, -2.5, 0]), run_time=2)
        self.add(board_1)
        self.wait()


        


        # Yellow square indicating next move at 9, 12 on 
        # board 2
        next_move_1=Rectangle(0.2, 0.2)
        next_move_1.set_stroke(color=YELLOW, width=5)
        next_move_1.move_to([5.325, -0.13, 0])

        arrow_next_move_1 = Arrow(
            next_move_1.get_bottom() + DOWN * 0.85,
            next_move_1.get_bottom(),
            stroke_width=5,
            stroke_color=YELLOW,
            fill_color=YELLOW,
            buff=0.2,
        )
        
        self.wait()
        self.play(action_label.animate.move_to([5, -2.5, 0]), run_time=2)
        self.add(board_2)
        self.play(ShowCreation(next_move_1), 
                  ShowCreation(arrow_next_move_1))

        policy_network_label = Text(
            "POLICY NETWORK",
            font="Myriad Pro",
            font_size=42,
        )
        policy_network_label.set_color(CHILL_BROWN)
        policy_network_label.next_to(border, DOWN, buff=0.3)


    
        # Ok, kind of an annoying large amount of stuff to figure out here
        # One step at a time. 
        svg_files=list(sorted(svg_dir.glob('*p29_42*')))

        all_svgs=Group()
        for svg_file in svg_files: 
            svg_image=SVGMobject(str(svg_file))
            all_svgs.add(svg_image[1:]) #Thowout background

        
        all_svgs.move_to([-2.1, -6, 0])
        all_svgs.scale(7)

        for o in all_svgs[4]: o.set_opacity(0.4+0.6*np.random.rand())
        for o in all_svgs[5]: o.set_opacity(0.4+0.6*np.random.rand())
 

        self.wait()
        self.remove(state_label, action_label, policy_label, label, arrow_next_move_1)
        self.play(self.frame.animate.reorient(0, 0, 0, (-0.11, -2.82, 0.0), 13.77),
                  # FadeOut(state_label),
                  # FadeOut(action_label),
                  # FadeOut(policy_label), 
                  # FadeOut(label), 
                  board_1.animate.scale(1.2).move_to([-5.4 , -0.3,  0. ]),
                  board_2.animate.scale(1.2).move_to([5.4 , -0.3,  0. ]),
                  next_move_1.animate.move_to([ 5.792, -0.10,  0. ]),
                  Write(all_svgs[1]),
                  Write(all_svgs[2][:-5]),
                  Write(all_svgs[2][-1]),
                  Write(all_svgs[4]),
                  Write(all_svgs[5]),
                  # Write(all_svgs[3:5]),
                  run_time=3
                 )
        self.add(policy_network_label)

        self.wait()
        self.play(FadeIn(all_svgs[0]), 
                  FadeIn(all_svgs[2][-5:-3]))
        self.wait()
        self.play(FadeIn(all_svgs[3]), 
                  FadeIn(all_svgs[2][-3:-1]))

        
        #Ok now add heatmap to the go board. 
        heatmap=create_heatmap_overlay(heatmap_1)
        heatmap.set_opacity(0.8) 
        heatmap.scale(0.4 * 1.2)  # Match board_2's final scale
        heatmap.move_to([5.4, -0.3, 0])  # Match board_2's final position

        self.wait()
        self.play(ShowCreation(heatmap), FadeOut(next_move_1), run_time=4)
        self.wait()
        
        ## Alrighty, now the end of p31 here, flipping through boards and next moves
        ## from different games. 

        self.play(self.frame.animate.reorient(0, 0, 0, (0.03, -0.06, 0.0), 8.94), 
                  FadeOut(all_svgs[:6]), 
                  FadeOut(heatmap), 
                  FadeIn(next_move_1),
                  run_time=4)
        self.wait()

        human_game_files=list(human_games_dir.glob('*.sgf'))

        # Ok so now I want to iterate throught these game files
        # However I have yet to implement capturing -> and it will just
        # be easier if I do that first
        # Will slow down immediate progress, but I'm tempted to 
        # just go ahead and do the big grid of games animations then come back. 
        # Ok I think I got it. 

        self.remove(board_1, board_2, next_move_1)
        num_games_to_play=2 ## CRANK UP IN FINAL RENDER

        for game_index in range(num_games_to_play):

            p=human_game_files[game_index]
            moves = parse_sgf(p)
            moves_to_show=np.random.randint(0, len(moves)-2)

            board_3=get_board()

            board_and_stones=Group()
            board_and_stones.add(board_3)
            board_state = {}  # (x, y) -> color
            stone_objects = {}  # (x, y) -> Mobject
            for i, (x, y, color) in enumerate(moves[:moves_to_show]):
                stone = create_stone(x, y, color)
                board_and_stones.add(stone)
                
                board_state[(x, y)] = color
                stone_objects[(x, y)] = stone
                
                # Check for captures
                captured = find_captures(x, y, board_state)
                for cx, cy in captured:
                    # Remove from scene and state
                    board_and_stones.remove(stone_objects[(cx, cy)])
                    del board_state[(cx, cy)]
                    del stone_objects[(cx, cy)]

            #Hmm ok now need yellow box around next move...
            # self.wait()
            # self.add(board_and_stones)

            x, y, c = moves[moves_to_show]
            pos = [(-(size-1)/2 + x) * step, (-(size-1)/2 + y) * step, 0]
            next_move=Rectangle(0.5, 0.5)
            next_move.set_stroke(color=YELLOW, width=7)
            next_move.move_to(pos)

            # self.add(next_move)
            
            board_and_stones_copy=copy.deepcopy(board_and_stones)
            board_and_stones_copy.set_opacity(0.5)
            board_and_stones_copy.add(next_move)
            board_and_stones_copy.scale(0.4*1.2)
            board_and_stones_copy.move_to([5.4 , -0.3,  0. ])
            

            board_and_stones.scale(0.4*1.2)
            board_and_stones.move_to([-5.4 , -0.3,  0. ])

            
            self.add(board_and_stones, board_and_stones_copy)
            self.wait()
            self.remove(board_and_stones, board_and_stones_copy)

        

        # Start p32
        # Move to side and do bar graph
        # Kinda want to do in manim, but I think premiere is probably the 
        # move - I think the only animation I really need is the bar graph coming 
        # up. Ok yeah just made pngs for premiere. Now i just need to 
        # lost the arrows and move to the side.  Camera move for sure. 
        # Eh the 3D-ness here is becoming a challenge. 
        # I think it's time to replace the 3d conv with just and image of it. 
        # Let me try a dumb screen shot first and see how that goes. 
        # Alright time for the old switcheroo


        conv_raster=ImageMobject(str(svg_dir/'conv_raster.png'))
        conv_raster.scale(1.135)
        conv_raster.move_to([-0.1, -0.25, 0])
        # conv_raster.set_opacity(0.5)

        self.wait()
        self.remove(cnn); self.add(conv_raster)
        self.play(self.frame.animate.reorient(0, 0, 0, (3.77, 0.08, 0.0), 9.49), 
                  FadeOut(arrow_in), 
                  FadeOut(arrow_out),
                  run_time=4)
        self.wait() #Ok now premiere!

        # Nice, kinda hacky going to the screenshot, but seems to work fine 
        # P33 let's go

        ag_group_1=Group(conv_raster, border, policy_network_label, alphago_logo)
        ag_group_2=copy.deepcopy(ag_group_1)

        # self.remove(ag_group_1)
        # self.add(ag_group_1)


        board_5=get_board()

        # ag_group_1.scale(0.7)
        # ag_group_1.move_to([-7.2, 0.2, 0])

        ag_1_label=Text('V1', font="Myriad Pro", weight='bold', font_size=32)
        ag_1_label.set_color(CHILL_BROWN)
        ag_1_label.move_to([-5.9, -1.4, 0])

        #I can have a little stone now?
        ag_1_stone=create_stone(0, 0, WHITE)
        ag_1_stone.scale(0.7)
        ag_1_stone.move_to([-8.5, -1.4, 0])

        # self.remove(ag_1_stone)
        # ag_group_2.scale(0.7)
        # ag_group_2.move_to([7.2, 0.2, 0])


        ag_2_label=Text('V2', font="Myriad Pro", weight='bold', font_size=32)
        ag_2_label.set_color(CHILL_BROWN)
        ag_2_label.move_to([8.5, -1.4, 0])

        #I can have a little stone now?
        ag_2_stone=create_stone(0, 0, BLACK)
        ag_2_stone.scale(0.7)
        ag_2_stone.move_to([5.85, -1.4, 0])  
        ag_2_stone.set_shading(0.5, 0.4, 0.2)

        self.wait()
        self.add(ag_group_2)
        self.play(self.frame.animate.reorient(0, 0, 0, (0.1, 0.15, 0.0), 11.51), 
                 ag_group_1.animate.scale(0.7).move_to([-7.2, 0.2, 0]), 
                 ag_group_2.animate.scale(0.7).move_to([7.2, 0.2, 0]),
                 run_time=4)

        arrow_ag_1 = Arrow(
            ag_group_1.get_right() ,
            ag_group_1.get_right() + RIGHT * 1.2,
            stroke_width=5,
            stroke_color=CHILL_BROWN,
            fill_color=CHILL_BROWN,
            buff=0.3,
        )
        arrow_ag_2 = Arrow(
            ag_group_2.get_left(),
            ag_group_2.get_left() + LEFT * 1.2,
            stroke_width=5,
            stroke_color=CHILL_BROWN,
            fill_color=CHILL_BROWN,
            buff=0.3,
        )

        self.play(FadeIn(board_5), 
                  FadeIn(ag_1_label), 
                  FadeIn(ag_1_stone), 
                  FadeIn(ag_2_label), 
                  FadeIn(ag_2_stone), 
                  FadeIn(arrow_ag_1),
                  FadeIn(arrow_ag_2))  
        self.wait()            

        # P34
        # Ok now we need to play a game, show the winner (how do I figure this out?)
        # Then highlight all winning moves in green, and all losing moves 
        # in red. And want to be able to cycle through a few of these. 
        

        # self_game_files=sorted(list(self_games_dir.glob('*.sgf')))
        # game_index=0
        # p=self_game_files[game_index]

        #White wins
        p=self_games_dir/'G01.sgf'
        winning_color='#FFFFFF'

        #Black wins
        # p=self_games_dir/'G05.sgf' 
        # winning_color='#000000'

        #White wins
        # p=self_games_dir/'G02.sgf' 
        # winning_color='#FFFFFF'

        moves = parse_sgf(p)
        stones=Group()
        board_state = {}  # (x, y) -> color
        stone_objects = {}  # (x, y) -> Mobject
        for i, (x, y, color) in enumerate(moves):
            stone = create_stone(x, y, color)
            stones.add(stone)
            self.add(stone)
            # self.wait(0.1) #Uncomment in FINAL RENDER
            
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

        
        #Ok, now I need a winner icon, and to add green/red squares. 

        winner_graphic=ImageMobject(str(svg_dir/'winner_icon.png'))
        winner_graphic.scale(0.25)

        if winning_color=='#FFFFFF':
            winner_graphic.move_to([-7.2, 0, 0])
        else:
            winner_graphic.move_to([7.2, 0, 0])

        self.wait()
        self.play(FadeIn(winner_graphic))
        self.wait()

        ##So i would love to have the quares behind the stones here
        ## but ti's being a bit finnicky and clock is ticking
        ## so probably keep rolling

        squares_1=VGroup()
        for i, (x, y, color) in enumerate(moves):
            if winning_color==color:
                pos = [(-(size-1)/2 + x) * step, (-(size-1)/2 + y) * step, 0]
                square=Rectangle(0.35, 0.35)
                square.set_stroke(color=GREEN, width=7)
                square.move_to(pos) 
                # square.shift([0.1, -0.2, 0])
                self.add(square)
                squares_1.add(square)
                # self.wait(0.1)   #Uncomment for FINAL RENDER

        self.wait()   

        squares_2=VGroup()
        for i, (x, y, color) in enumerate(moves):
            if winning_color!=color:
                pos = [(-(size-1)/2 + x) * step, (-(size-1)/2 + y) * step, 0]
                square=Rectangle(0.35, 0.35)
                square.set_stroke(color=RED, width=7)
                square.move_to(pos) 
                # square.shift([0.1, -0.2, 0])
                self.add(square)
                squares_2.add(square)
                # self.wait(0.1)   #Uncomment for FINAL RENDER

        self.wait()           
        self.play(FadeOut(squares_1),
                  FadeOut(squares_2),
                  FadeOut(winner_graphic),
                  run_time=3)
        self.play(FadeOut(stones))
        self.wait()



        #Ok let's do a second game for the back half of p34. 

        #Black wins
        p=self_games_dir/'G05.sgf' 
        winning_color='#000000'

        #White wins
        # p=self_games_dir/'G02.sgf' 
        # winning_color='#FFFFFF'

        moves = parse_sgf(p)
        stones=Group()
        board_state = {}  # (x, y) -> color
        stone_objects = {}  # (x, y) -> Mobject
        for i, (x, y, color) in enumerate(moves):
            stone = create_stone(x, y, color)
            stones.add(stone)
            self.add(stone)
            # self.wait(0.1) #Uncomment in FINAL RENDER
            
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

        
        #Ok, now I need a winner icon, and to add green/red squares. 

        winner_graphic=ImageMobject(str(svg_dir/'winner_icon.png'))
        winner_graphic.scale(0.25)

        if winning_color=='#FFFFFF':
            winner_graphic.move_to([-7.2, 0, 0])
        else:
            winner_graphic.move_to([7.2, 0, 0])

        self.wait()
        self.play(FadeIn(winner_graphic))
        self.wait()

        ##So i would love to have the quares behind the stones here
        ## but ti's being a bit finnicky and clock is ticking
        ## so probably keep rolling

        squares_1=VGroup()
        for i, (x, y, color) in enumerate(moves):
            if winning_color==color:
                pos = [(-(size-1)/2 + x) * step, (-(size-1)/2 + y) * step, 0]
                square=Rectangle(0.35, 0.35)
                square.set_stroke(color=GREEN, width=7)
                square.move_to(pos) 
                # square.shift([0.1, -0.2, 0])
                self.add(square)
                squares_1.add(square)
                # self.wait(0.1)   #Uncomment for FINAL RENDER

        self.wait()   

        squares_2=VGroup()
        for i, (x, y, color) in enumerate(moves):
            if winning_color!=color:
                pos = [(-(size-1)/2 + x) * step, (-(size-1)/2 + y) * step, 0]
                square=Rectangle(0.35, 0.35)
                square.set_stroke(color=RED, width=7)
                square.move_to(pos) 
                # square.shift([0.1, -0.2, 0])
                self.add(square)
                squares_2.add(square)
                # self.wait(0.1)   #Uncomment for FINAL RENDER

        self.wait()           
        self.play(FadeOut(squares_1),
                  FadeOut(squares_2),
                  FadeOut(winner_graphic),
                  # FadeOut(stones),
                  run_time=3)
        self.play(FadeOut(stones))
        self.wait()








        self.wait(20)
        self.embed()








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














