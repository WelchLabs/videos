from welch_paths import welch_path
import manimgx as gx
from _2026.the_bitter_lesson.go_gx import create_stone, find_captures, get_board as make_go_board, parse_sgf
import numpy as np
from welch_scene import WelchScene
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

# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/alpha_go_self_play')
# human_games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/human_human_kgs-19-2015')
# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/games_with_videos')
# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/less_wrong_reverse_engineer')

# svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/graphics/to_manim/')


games_dir=Path(welch_path('bitter_lesson/games/games_with_videos'))
svg_dir=Path(welch_path('bitter_lesson/graphics/to_manim'))


size = 19  # 19x19 board
padding = 0.5
board_width = 8 # Total width in Manim units
step = board_width / size








def get_board():
    return make_go_board(2, grid_stroke_width=1.5)









class RL_Render_2(WelchScene):
    def construct(self): 

        '''
        Ok so here I want to implement captures, and compare the replace against a couple 
        games with video commentary to make sure everything looks good!
        '''
        board=get_board()
        
        games=sorted(list(games_dir.glob('*sgf')))

        p=games[0]
        print(p)
        moves = parse_sgf(p)

        #Game 1 video: https://www.youtube.com/watch?v=E3g-kBtqtMo
        #Game 2 video: https://www.youtube.com/watch?v=0uTlyJ4ITnQ
        #Ok, game 2 actually has captures!

        self.set_view(0, 0, 0, (np.float32(0.09), np.float32(-0.02), np.float32(0.0)), 9.25)

        # Track board state and stone objects
        board_state = {}  # (x, y) -> color
        stone_objects = {}  # (x, y) -> Mobject

        self.add(board)
        for i, (x, y, color) in enumerate(moves):
            stone = create_stone(x, y, color)
            self.add(stone)
            
            board_state[(x, y)] = color
            stone_objects[(x, y)] = stone
            
            # Check for captures
            captured = find_captures(x, y, board_state)
            for cx, cy in captured:
                # Remove from scene and state
                self.remove(stone_objects[(cx, cy)])
                del board_state[(cx, cy)]
                del stone_objects[(cx, cy)]
            self.wait(0.1)

        self.wait()


        self.wait(20)














