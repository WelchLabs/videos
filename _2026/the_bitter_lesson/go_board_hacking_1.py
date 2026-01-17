from manimlib import *
from tqdm import tqdm

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

size = 19  # 19x19 board
padding = 0.5
board_width = 8 # Total width in Manim units
step = board_width / size

def create_stone(x, y, color=BLACK):
    stone_radius=step*0.45
    pos = [(-(size-1)/2 + x) * step, (-(size-1)/2 + y) * step, 0]
    stone = Circle(radius=stone_radius, fill_color=color, fill_opacity=1)
    stone.set_stroke(color=BLACK, width=0.5)
    
    # Add a "shine" for the white stones or a "matte" highlight for black
    shine_color = WHITE if color == BLACK else GREY_B
    shine = Dot(radius=stone_radius*0.3, fill_color=shine_color, fill_opacity=0.3)
    shine.move_to(stone.get_center() + stone_radius*0.3*(UP+LEFT))
    
    return Group(stone, shine).move_to(pos)

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

        black_1 = create_stone(15, 15, BLACK)
        white_1 = create_stone(3, 3, WHITE)
        black_2 = create_stone(15, 3, BLACK)

        self.add(black_1, white_1, black_2)


        self.wait()
        self.embed()





