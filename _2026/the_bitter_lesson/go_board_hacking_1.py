from welch_paths import welch_path
import manimgx as gx
from _2026.the_bitter_lesson.go_gx import create_stone, parse_sgf
from welch_scene import WelchScene
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
# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/human_human_kgs-19-2015')
# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/games_with_videos')
games_dir=Path(welch_path('bitter_lesson/games/less_wrong_reverse_engineer'))



size = 19  # 19x19 board
padding = 0.5
board_width = 8 # Total width in Manim units
step = board_width / size








class GoHackingOne(WelchScene):
    def construct(self): 


        board_rect = gx.Square(side_length=board_width + padding)
        board_rect.set_fill(FRESH_TAN, opacity=1)
        board_rect.set_stroke(CHILL_BROWN, width=2)

        # We center the grid so the middle intersection is at (0,0,0)
        lines = gx.VGroup()
        start_point = -(size - 1) / 2 * step
        
        for i in range(size):
            # Vertical lines
            v_line = gx.Line(
                [start_point + i * step, start_point, 0],
                [start_point + i * step, -start_point, 0]
            )
            # Horizontal lines
            h_line = gx.Line(
                [start_point, start_point + i * step, 0],
                [-start_point, start_point + i * step, 0]
            )
            lines.add(v_line, h_line)
            
        lines.set_stroke(gx.BLACK, width=1.5)

        # For a 19x19, these are usually at 4, 10, and 16 (1-indexed)
        hoshi_indices = [3, 9, 15] # 0-indexed
        hoshi_dots = gx.VGroup()
        for x in hoshi_indices:
            for y in hoshi_indices:
                dot = gx.Circle(radius=0.05, fill_color=gx.BLACK, fill_opacity=1, stroke_width=0)
                # Position the dot based on grid coordinates
                dot.move_to([start_point + x * step, start_point + y * step, 0])
                hoshi_dots.add(dot)


        self.add(board_rect)
        self.add(lines)
        self.add(hoshi_dots)

        # moves = parse_sgf(games_dir/'alpha_go_self_play/1c.sgf')
        p=sorted(list(games_dir.glob('*.sgf')))[0]
        moves = parse_sgf(p)

        self.wait()
        for i, (x, y, color) in enumerate(moves[:20]):
            stone = create_stone(x, y, color)
            self.add(stone)
            # self.wait(0.1)

        self.wait()

        # x, y, color = moves[20]
        # stone = create_stone(x, y, color)
        # self.add(stone)

        # self.remove(stone)

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






        # black_1.set_color(BLACK)
        # white_1.set_color('#F5F5F5')
        # # white_1.set_color(WHITE)
        # white_1.set_shade_in_3d(True)

        # white_2.set_shade_in_3d(True)

        # white_2.set_color("#B0B0B0")  # Medium gray instead of near-white
        # white_2.set_shade_in_3d(True)  # Low ambient, high diffuse, moderate specular
        # # Try these additional methods if available:
        # white_2.set_gloss(0.8)
        # white_2.set_shadow(0.3)


class GoHackingTwo(WelchScene):
    def construct(self): 


        board_rect = gx.Square(side_length=board_width + padding)
        board_rect.set_fill(FRESH_TAN, opacity=1)
        board_rect.set_stroke(CHILL_BROWN, width=2)

        # We center the grid so the middle intersection is at (0,0,0)
        lines = gx.VGroup()
        start_point = -(size - 1) / 2 * step
        
        for i in range(size):
            # Vertical lines
            v_line = gx.Line(
                [start_point + i * step, start_point, 0],
                [start_point + i * step, -start_point, 0]
            )
            # Horizontal lines
            h_line = gx.Line(
                [start_point, start_point + i * step, 0],
                [-start_point, start_point + i * step, 0]
            )
            lines.add(v_line, h_line)
            
        lines.set_stroke(gx.BLACK, width=1.5)

        # For a 19x19, these are usually at 4, 10, and 16 (1-indexed)
        hoshi_indices = [3, 9, 15] # 0-indexed
        hoshi_dots = gx.VGroup()
        for x in hoshi_indices:
            for y in hoshi_indices:
                dot = gx.Circle(radius=0.05, fill_color=gx.BLACK, fill_opacity=1, stroke_width=0)
                # Position the dot based on grid coordinates
                dot.move_to([start_point + x * step, start_point + y * step, 0])
                hoshi_dots.add(dot)


        self.add(board_rect)
        self.add(lines)
        self.add(hoshi_dots)

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

        self.wait()
        for i, (x, y, color) in enumerate(moves):
            stone = create_stone(x, y, color)
            self.add(stone)
            # self.wait(0.1)

        self.wait()


        self.remove(board_rect, lines,  hoshi_dots)
        

        # self.add(stone)
        # stone.set_color('#222222')


        # self.set_view(0, 0, 0, (0.87, 0.84, 0.0), 0.66)
        # self.wait()




        self.wait(20)



