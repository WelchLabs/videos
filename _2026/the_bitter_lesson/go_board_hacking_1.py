from manimlib import *
from tqdm import tqdm
import re
from pathlib import Path

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

# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/alpha_go_self_play')
games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/human_human_kgs-19-2015')

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


class GoHackingOne(InteractiveScene):
    def construct(self): 


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


        self.add(board_rect)
        self.add(lines)
        self.add(hoshi_dots)

        # moves = parse_sgf(games_dir/'alpha_go_self_play/1c.sgf')
        p=list(games_dir.glob('*.sgf'))[1]
        moves = parse_sgf(p)

        self.wait()
        for i, (x, y, color) in enumerate(moves[:20]):
            stone = create_stone(x, y, color)
            self.add(stone)
            # self.wait(0.1)

        self.wait()

        # To do -> make sure I've got the right orientation
        # by comparing to game comentary, especially considering
        # which play is on which side. 
        

        # i=10
        # x, y, color=moves[i]


        # stone = create_stone(x, y, color)
            
            # Animating the stone "dropping" from a hand
        # We start it slightly higher (Z-axis) and fade it in
        # Woudl be maybe cool in some scense to animate stones being placed down
        # stone.shift(0.5 * OUT) 
        # self.play(
        #     stone.animate.shift(0.5 * IN),
        #     FadeIn(stone),
        #     run_time=1.0, # if i < 10 else 0.1 # Speed up as game progresses
        # )
    

        # black_1 = create_stone(14, 16, BLACK)
        # white_1 = create_stone(4, 4, WHITE)
        # black_2 = create_stone(15, 3, BLACK)
        # white_2 = create_stone(15, 4, WHITE)

        # self.add(black_1, white_1, black_2, white_2)

        #Ok, not bad visually! Next I can work on importing games. 
        #A transition from a ncie overhead shot to this could be pretty dope. 







        self.wait()
        self.embed()






        # black_1.set_color(BLACK)
        # white_1.set_color('#F5F5F5')
        # # white_1.set_color(WHITE)
        # white_1.set_shading(0.5, 0.8, 1.0)

        # white_2.set_shading(1.0,0.5,1.0)

        # white_2.set_color("#B0B0B0")  # Medium gray instead of near-white
        # white_2.set_shading(0.7, 0.9, 0.9)  # Low ambient, high diffuse, moderate specular
        # # Try these additional methods if available:
        # white_2.set_gloss(0.8)
        # white_2.set_shadow(0.3)



