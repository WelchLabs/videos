"""Common Go network storyboards for the video and book scene entry points."""

import copy

import manimgx as gx
import matplotlib.pyplot as plt
import numpy as np

from _2026.the_bitter_lesson.go_gx import (
    board_width, create_cnn_layer, create_stone, find_captures, padding, parse_sgf, size, step,
)
from welch_scene import WelchScene, image_at_height
from welch_style import BLUE, CHILL_BLUE, CHILL_BROWN, FRESH_TAN, GREEN, YELLOW


class GoNetworkScene(WelchScene):
    """Uses the game paths and board styles supplied by its entry point."""

    self_games_dir = None
    human_games_dir = None
    svg_dir = None


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


def render_example_go_game_1():
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

    board=gx.Group()
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

    heatmap_group = gx.VGroup()
    start_point = -(size - 1) / 2 * step

    for i in range(size):
        for j in range(size):
            # heatmap array is top-down, manim is bottom-up, so flip j
            value = heatmap_data[18 - j, i]

            # Get RGB from viridis colormap
            rgb = cmap(value)[:3]
            hex_color = gx.rgb_to_color(rgb).to_hex()

            square = gx.Square(side_length=step)
            square.set_fill(hex_color) #, opacity=opacity * value)  # opacity scales with value
            square.set_stroke(width=0)

            # Position at grid intersection
            x = start_point + i * step
            y = start_point + j * step
            square.move_to([x, y, 0])

            heatmap_group.add(square)

    return heatmap_group


class P29_36(GoNetworkScene):
    def construct(self):
        '''
        Ok leaning towards doing a bit more in manim than I first thought,
        mostly becuase I think there's a path that's not so bad
        Let me see how this goes/feels, and I can fall back to less Manim
        if I need to.
        '''

        # alphago_logo=SVGMobject(str(self.svg_dir/'alpha_go_logo.svg'))
        alphago_logo=image_at_height(str(self.svg_dir/'alpha_go_logo.png'))
        alphago_logo.scale(0.25)
        alphago_logo.move_to([-0.15, 3, 0])


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


        self.wait()


        #Ok this looks decent.
        # self.add(cnn)
        cnn.move_to([0, 0, 0])
        cnn.rotate(90*gx.DEGREES, axis=gx.OUT)
        cnn.rotate(30*gx.DEGREES, axis=gx.RIGHT)
        cnn.rotate(-30*gx.DEGREES, axis=gx.UP)
        cnn.rotate(-15*gx.DEGREES, axis=gx.OUT)

        border = gx.RoundedRectangle(
            width=4.8,
            height=4.8,
            corner_radius=0.2,
            stroke_color=CHILL_BROWN,
            stroke_width=5,
            fill_opacity=0,
        )
        border.move_to([-0.1, -0.2, 0])

        # Add label text below
        label = gx.Text(
            "SUPERVISED POLICY NETWORK (CNN)",
            font="Myriad Pro",
            font_size=32,
        )
        label.set_color(CHILL_BROWN)
        label.next_to(border, gx.DOWN, buff=0.3)

        self.wait()
        self.play(gx.FadeIn(alphago_logo))
        self.play(gx.Create(cnn), gx.Write(border), gx.Write(label), run_time=5)

        # self.add(border)
        # self.add(label)

        self.wait()

        arrow_in = gx.Arrow(
            border.get_left() + gx.LEFT * 0.85,
            border.get_left(),
            stroke_width=10,
            stroke_color=CHILL_BROWN,
            fill_color=CHILL_BROWN,
            buff=0.2,
        )


        arrow_out = gx.Arrow(
            border.get_right(),
            border.get_right() + gx.RIGHT * 0.85,
            stroke_width=7,
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


        policy_label=gx.Text("POLICY", font="Myriad Pro", weight='bold', font_size=44)
        policy_label.set_color(YELLOW)
        policy_label.move_to([0, -3.5, 0])

        state_label=gx.Text("STATE", font="Myriad Pro", weight='bold', font_size=44)
        state_label.set_color(YELLOW)
        state_label.move_to([-4, -0.2, 0])

        action_label=gx.Text("ACTION", font="Myriad Pro", weight='bold', font_size=44)
        action_label.set_color(YELLOW)
        action_label.move_to([4, -0.2, 0])

        self.wait()
        self.play(gx.FadeIn(policy_label),
                  gx.FadeIn(state_label),
                  gx.FadeIn(action_label),
                  gx.FadeIn(arrow_in),
                  gx.FadeIn(arrow_out))

        self.wait()

        self.play(state_label.animate.move_to([-5, -2.5, 0]), run_time=2)
        self.add(board_1)
        self.wait()





        # Yellow square indicating next move at 9, 12 on
        # board 2
        next_move_1=gx.Rectangle(0.2, 0.2)
        next_move_1.set_stroke(color=YELLOW, width=5)
        next_move_1.move_to([5.325, -0.13, 0])

        arrow_next_move_1 = gx.Arrow(
            next_move_1.get_bottom() + gx.DOWN * 0.85,
            next_move_1.get_bottom(),
            stroke_width=5,
            stroke_color=YELLOW,
            fill_color=YELLOW,
            buff=0.2,
        )

        self.wait()
        self.play(action_label.animate.move_to([5, -2.5, 0]), run_time=2)
        self.add(board_2)
        self.play(gx.Create(next_move_1),
                  gx.Create(arrow_next_move_1))

        policy_network_label = gx.Text(
            "POLICY NETWORK",
            font="Myriad Pro",
            font_size=42,
        )
        policy_network_label.set_color(CHILL_BROWN)
        policy_network_label.next_to(border, gx.DOWN, buff=0.3)



        # Ok, kind of an annoying large amount of stuff to figure out here
        # One step at a time.
        svg_files=list(sorted(self.svg_dir.glob('*p29_42*')))

        all_svgs=gx.Group()
        for svg_file in svg_files:
            svg_image=gx.SVGMobject(str(svg_file))
            all_svgs.add(svg_image[1:]) #Thowout background


        all_svgs.move_to([-2.1, -6, 0])
        all_svgs.scale(7)

        for o in all_svgs[4]: o.set_opacity(0.4+0.6*np.random.rand())
        for o in all_svgs[5]: o.set_opacity(0.4+0.6*np.random.rand())


        self.wait()
        self.remove(state_label, action_label, policy_label, label, arrow_next_move_1)
        self.play(self.animate_view(0, 0, 0, (-0.11, -2.82, 0.0), 13.77),
                  # FadeOut(state_label),
                  # FadeOut(action_label),
                  # FadeOut(policy_label),
                  # FadeOut(label),
                  board_1.animate.scale(1.2).move_to([-5.4 , -0.3,  0. ]),
                  board_2.animate.scale(1.2).move_to([5.4 , -0.3,  0. ]),
                  next_move_1.animate.move_to([ 5.792, -0.10,  0. ]),
                  gx.Write(all_svgs[1]),
                  gx.Write(all_svgs[2][:-5]),
                  gx.Write(all_svgs[2][-1]),
                  gx.Write(all_svgs[4]),
                  gx.Write(all_svgs[5]),
                  # Write(all_svgs[3:5]),
                  run_time=3
                 )
        self.add(policy_network_label)

        self.wait()
        self.play(gx.FadeIn(all_svgs[0]),
                  gx.FadeIn(all_svgs[2][-5:-3]))
        self.wait()
        self.play(gx.FadeIn(all_svgs[3]),
                  gx.FadeIn(all_svgs[2][-3:-1]))


        #Ok now add heatmap to the go board.
        heatmap=create_heatmap_overlay(heatmap_1)
        heatmap.set_opacity(0.8)
        heatmap.scale(0.4 * 1.2)  # Match board_2's final scale
        heatmap.move_to([5.4, -0.3, 0])  # Match board_2's final position

        self.wait()
        self.play(gx.Create(heatmap), gx.FadeOut(next_move_1), run_time=4)
        self.wait()

        ## Alrighty, now the end of p31 here, flipping through boards and next moves
        ## from different games.

        self.play(self.animate_view(0, 0, 0, (0.03, -0.06, 0.0), 8.94),
                  gx.FadeOut(all_svgs[:6]),
                  gx.FadeOut(heatmap),
                  gx.FadeIn(next_move_1),
                  run_time=4)
        self.wait()

        human_game_files=list(self.human_games_dir.glob('*.sgf'))

        # Ok so now I want to iterate throught these game files
        # However I have yet to implement capturing -> and it will just
        # be easier if I do that first
        # Will slow down immediate progress, but I'm tempted to
        # just go ahead and do the big grid of games animations then come back.
        # Ok I think I got it.

        self.remove(board_1, board_2, next_move_1)
        num_games_to_play=12 ## CRANK UP IN FINAL RENDER

        for game_index in range(num_games_to_play):

            p=human_game_files[game_index]
            moves = parse_sgf(p)
            moves_to_show=np.random.randint(0, len(moves)-2)

            board_3=self.get_board()

            board_and_stones=gx.Group()
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
            next_move=gx.Rectangle(0.5, 0.5)
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


        conv_raster=image_at_height(str(self.svg_dir/'conv_raster.png'))
        conv_raster.scale(1.135)
        conv_raster.move_to([-0.1, -0.25, 0])
        # conv_raster.set_opacity(0.5)

        self.wait()
        self.remove(cnn); self.add(conv_raster)
        self.play(self.animate_view(0, 0, 0, (3.77, 0.08, 0.0), 9.49),
                  gx.FadeOut(arrow_in),
                  gx.FadeOut(arrow_out),
                  run_time=4)
        self.wait() #Ok now premiere!

        # Nice, kinda hacky going to the screenshot, but seems to work fine
        # P33 let's go

        ag_group_1=gx.Group(conv_raster, border, policy_network_label, alphago_logo)
        ag_group_2=copy.deepcopy(ag_group_1)

        # self.remove(ag_group_1)
        # self.add(ag_group_1)


        board_5=self.get_board()

        # ag_group_1.scale(0.7)
        # ag_group_1.move_to([-7.2, 0.2, 0])

        ag_1_label=gx.Text('V1', font="Myriad Pro", weight='bold', font_size=32)
        ag_1_label.set_color(CHILL_BROWN)
        ag_1_label.move_to([-5.9, -1.4, 0])

        #I can have a little stone now?
        ag_1_stone=create_stone(0, 0, gx.WHITE)
        ag_1_stone.scale(0.7)
        ag_1_stone.move_to([-8.5, -1.4, 0])

        # self.remove(ag_1_stone)
        # ag_group_2.scale(0.7)
        # ag_group_2.move_to([7.2, 0.2, 0])


        ag_2_label=gx.Text('V2', font="Myriad Pro", weight='bold', font_size=32)
        ag_2_label.set_color(CHILL_BROWN)
        ag_2_label.move_to([8.5, -1.4, 0])

        #I can have a little stone now?
        ag_2_stone=create_stone(0, 0, gx.BLACK)
        ag_2_stone.scale(0.7)
        ag_2_stone.move_to([5.85, -1.4, 0])

        self.wait()
        self.add(ag_group_2)
        self.play(self.animate_view(0, 0, 0, (0.1, 0.15, 0.0), 11.51),
                 ag_group_1.animate.scale(0.7).move_to([-7.2, 0.2, 0]),
                 ag_group_2.animate.scale(0.7).move_to([7.2, 0.2, 0]),
                 run_time=4)

        arrow_ag_1 = gx.Arrow(
            ag_group_1.get_right() ,
            ag_group_1.get_right() + gx.RIGHT * 1.2,
            stroke_width=5,
            stroke_color=CHILL_BROWN,
            fill_color=CHILL_BROWN,
            buff=0.3,
        )
        arrow_ag_2 = gx.Arrow(
            ag_group_2.get_left(),
            ag_group_2.get_left() + gx.LEFT * 1.2,
            stroke_width=5,
            stroke_color=CHILL_BROWN,
            fill_color=CHILL_BROWN,
            buff=0.3,
        )

        self.play(gx.FadeIn(board_5),
                  gx.FadeIn(ag_1_label),
                  gx.FadeIn(ag_1_stone),
                  gx.FadeIn(ag_2_label),
                  gx.FadeIn(ag_2_stone),
                  gx.FadeIn(arrow_ag_1),
                  gx.FadeIn(arrow_ag_2))
        self.wait()

        # P34
        # Ok now we need to play a game, show the winner (how do I figure this out?)
        # Then highlight all winning moves in green, and all losing moves
        # in red. And want to be able to cycle through a few of these.


        # self_game_files=sorted(list(self.self_games_dir.glob('*.sgf')))
        # game_index=0
        # p=self_game_files[game_index]

        #White wins
        p=self.self_games_dir/'G01.sgf'
        winning_color='#FFFFFF'

        #Black wins
        # p=self.self_games_dir/'G05.sgf'
        # winning_color='#000000'

        #White wins
        # p=self.self_games_dir/'G02.sgf'
        # winning_color='#FFFFFF'

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


        #Ok, now I need a winner icon, and to add green/red squares.

        winner_graphic=image_at_height(str(self.svg_dir/'winner_icon.png'))
        winner_graphic.scale(0.25)

        if winning_color=='#FFFFFF':
            winner_graphic.move_to([-7.2, 0, 0])
        else:
            winner_graphic.move_to([7.2, 0, 0])

        self.wait()
        self.play(gx.FadeIn(winner_graphic))
        self.wait()

        ##So i would love to have the quares behind the stones here
        ## but ti's being a bit finnicky and clock is ticking
        ## so probably keep rolling

        squares_1=gx.VGroup()
        for i, (x, y, color) in enumerate(moves):
            if winning_color==color:
                pos = [(-(size-1)/2 + x) * step, (-(size-1)/2 + y) * step, 0]
                square=gx.Rectangle(0.35, 0.35)
                square.set_stroke(color=GREEN, width=7)
                square.move_to(pos)
                # square.shift([0.1, -0.2, 0])
                self.add(square)
                squares_1.add(square)
                self.wait(0.1)   #Uncomment for FINAL RENDER

        self.wait()

        squares_2=gx.VGroup()
        for i, (x, y, color) in enumerate(moves):
            if winning_color!=color:
                pos = [(-(size-1)/2 + x) * step, (-(size-1)/2 + y) * step, 0]
                square=gx.Rectangle(0.35, 0.35)
                square.set_stroke(color=gx.RED, width=7)
                square.move_to(pos)
                # square.shift([0.1, -0.2, 0])
                self.add(square)
                squares_2.add(square)
                self.wait(0.1)   #Uncomment for FINAL RENDER

        self.wait()
        self.play(gx.FadeOut(squares_1),
                  gx.FadeOut(squares_2),
                  gx.FadeOut(winner_graphic),
                  run_time=3)
        self.play(gx.FadeOut(stones))
        self.wait()



        #Ok let's do a second game for the back half of p34.

        #Black wins
        p=self.self_games_dir/'G05.sgf'
        winning_color='#000000'

        #White wins
        # p=self.self_games_dir/'G02.sgf'
        # winning_color='#FFFFFF'

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


        #Ok, now I need a winner icon, and to add green/red squares.

        winner_graphic=image_at_height(str(self.svg_dir/'winner_icon.png'))
        winner_graphic.scale(0.25)

        if winning_color=='#FFFFFF':
            winner_graphic.move_to([-7.2, 0, 0])
        else:
            winner_graphic.move_to([7.2, 0, 0])

        self.wait()
        self.play(gx.FadeIn(winner_graphic))
        self.wait()

        ##So i would love to have the quares behind the stones here
        ## but ti's being a bit finnicky and clock is ticking
        ## so probably keep rolling

        squares_1=gx.VGroup()
        for i, (x, y, color) in enumerate(moves):
            if winning_color==color:
                pos = [(-(size-1)/2 + x) * step, (-(size-1)/2 + y) * step, 0]
                square=gx.Rectangle(0.35, 0.35)
                square.set_stroke(color=GREEN, width=7)
                square.move_to(pos)
                # square.shift([0.1, -0.2, 0])
                self.add(square)
                squares_1.add(square)
                self.wait(0.1)   #Uncomment for FINAL RENDER

        self.wait()

        squares_2=gx.VGroup()
        for i, (x, y, color) in enumerate(moves):
            if winning_color!=color:
                pos = [(-(size-1)/2 + x) * step, (-(size-1)/2 + y) * step, 0]
                square=gx.Rectangle(0.35, 0.35)
                square.set_stroke(color=gx.RED, width=7)
                square.move_to(pos)
                # square.shift([0.1, -0.2, 0])
                self.add(square)
                squares_2.add(square)
                self.wait(0.1)   #Uncomment for FINAL RENDER


        #P35 Add titling
        pg_title = gx.Text(
            "POLICY GRADIENT METHOD",
            font="Myriad Pro",
            weight='bold',
            font_size=46,
        )
        pg_title.set_color(gx.WHITE)
        pg_title.move_to([0, 5.15, 0])

        pg_subtitle = gx.Text(
            "REINFORCE , WIlliams 1992",
            font="Myriad Pro",
            # weight='bold',
            font_size=32,
        )
        pg_subtitle.set_color(CHILL_BROWN)
        pg_subtitle.next_to(pg_title, gx.DOWN, buff=0.15)


        # self.add(pg_title)
        # self.add(pg_subtitle)
        self.wait()
        self.play(gx.Write(pg_title), run_time=3)
        self.add(pg_subtitle)
        self.wait()

        self.play(gx.FadeOut(squares_1),
                  gx.FadeOut(squares_2),
                  gx.FadeOut(winner_graphic),
                  # FadeOut(stones),
                  run_time=3)
        self.play(gx.FadeOut(stones))
        self.wait()

        # Ok I was thinking about another game here, but
        # I don't think that really fits with the vibe of the paragraph
        # I think lets just transition to a 2 row format to introduce
        # value network
        # I could also cut in the lots of games 3d action
        # Anyway, can't go wrong wiht a nice transition here
        # Then will start a new class!

        self.wait()
        self.play(gx.FadeOut(pg_title),
                  gx.FadeOut(pg_subtitle),
                  gx.FadeOut(board_5),
                  gx.FadeOut(ag_group_2),
                  gx.FadeOut(ag_2_label),
                  gx.FadeOut(ag_2_stone),
                  gx.FadeOut(arrow_ag_2),
                  gx.FadeOut(arrow_ag_1),
                  gx.FadeOut(ag_1_label),
                  gx.FadeOut(ag_1_stone),
                  policy_network_label.animate.set_color(BLUE),
                  self.animate_view(0, 0, 0, (-7.2, -2.21, 0.0), 10.53),
                  run_time=4)


        self.wait(20)


class P37_42b(GoNetworkScene):
    def construct(self):
        '''
        Ok leaning towards doing a bit more in manim than I first thought,
        mostly becuase I think there's a path that's not so bad
        Let me see how this goes/feels, and I can fall back to less Manim
        if I need to.
        '''

        # alphago_logo=SVGMobject(str(self.svg_dir/'alpha_go_logo.svg'))
        alphago_logo=image_at_height(str(self.svg_dir/'alpha_go_logo.png'))
        alphago_logo.scale(0.25)
        alphago_logo.move_to([-0.15, 3, 0])

        conv_raster=image_at_height(str(self.svg_dir/'conv_raster.png'))
        conv_raster.scale(1.135)
        conv_raster.move_to([-0.1, -0.25, 0])
        # conv_raster.set_opacity(0.5)

        border = gx.RoundedRectangle(
            width=4.8,
            height=4.8,
            corner_radius=0.2,
            stroke_color=CHILL_BROWN,
            stroke_width=5,
            fill_opacity=0,
        )
        border.move_to([-0.1, -0.2, 0])


        policy_network_label = gx.Text(
            "POLICY NETWORK",
            font="Myriad Pro",
            font_size=42,
        )
        policy_network_label.set_color(BLUE)
        policy_network_label.next_to(border, gx.DOWN, buff=0.3)

        ag_group_1=gx.Group(conv_raster, border, policy_network_label, alphago_logo)
        ag_group_1.scale(0.7).move_to([-7.2, 0.2, 0])


        arrow_in = gx.Arrow(
            border.get_left() + gx.LEFT * 1.4,
            border.get_left(),
            stroke_width=10,
            stroke_color=CHILL_BROWN,
            fill_color=CHILL_BROWN,
            buff=0.4,
        )


        arrow_out = gx.Arrow(
            border.get_right(),
            border.get_right() + gx.RIGHT * 1.4,
            stroke_width=10,
            stroke_color=CHILL_BROWN,
            fill_color=CHILL_BROWN,
            buff=0.4,
        )

        #Ok now we bring in the actualy go board renderig?
        board_1=render_example_go_game_1()
        board_1.scale(0.4*1.0)
        board_1.move_to([-12 , -0.05,  0. ])

        board_2=render_example_go_game_1()
        board_2.scale(0.4*1.0)
        board_2.set_opacity(0.5)
        board_2.move_to([-2.2 , -0.05,  0. ])

        state_label=gx.Text("STATE", font="Myriad Pro", weight='bold', font_size=44)
        state_label.set_color(CHILL_BROWN)
        state_label.move_to([-12 , -2.1,  0. ])

        action_label=gx.Text("ACTION", font="Myriad Pro", weight='bold', font_size=44)
        action_label.set_color(CHILL_BROWN)
        action_label.move_to([-2.2, -2.1, 0])

        heatmap=create_heatmap_overlay(heatmap_1)
        heatmap.set_opacity(0.8)
        heatmap.scale(0.4 * 1.0)  # Match board_2's final scale
        heatmap.move_to([-2.2 , -0.05,  0. ])



        self.set_view(0, 0, 0, (-7.2, -2.21, 0.0), 10.53)
        self.add(ag_group_1)

        # self.add(board_1, arrow_in, arrow_out, board_2)
        # self.play(ShowCreation(heatmap))
        # self.add(state_label, action_label)

        self.wait()
        self.play(gx.FadeIn(board_1),
                  gx.FadeIn(arrow_in),
                  gx.FadeIn(arrow_out),
                  gx.FadeIn(board_2),
                  gx.FadeIn(state_label),
                  gx.FadeIn(action_label),
                  gx.Create(heatmap),
                  run_time=3)
        self.wait()


        #Now start building out bottom row.
        border_2 = gx.RoundedRectangle(
            width=4.8,
            height=4.8,
            corner_radius=0.2,
            stroke_color=CHILL_BROWN,
            stroke_width=5,
            fill_opacity=0,
        )
        border_2.scale(0.7)
        border_2.move_to([-7.2, -4.5, 0])

        board_3=render_example_go_game_1()
        board_3.scale(0.4*1.0)
        board_3.move_to([-12 , -0.05-4.5,  0. ])

        state_label_2=gx.Text("STATE", font="Myriad Pro", weight='bold', font_size=44)
        state_label_2.set_color(CHILL_BROWN)
        state_label_2.move_to([-12 , -2.1-4.5,  0. ])

        arrow_in_2 = gx.Arrow(
            border_2.get_left() + gx.LEFT * 1.4,
            border_2.get_left(),
            stroke_width=10,
            stroke_color=CHILL_BROWN,
            fill_color=CHILL_BROWN,
            buff=0.4,
        )

        arrow_out_2 = gx.Arrow(
            border_2.get_right(),
            border_2.get_right() + gx.RIGHT * 1.4,
            stroke_width=10,
            stroke_color=CHILL_BROWN,
            fill_color=CHILL_BROWN,
            buff=0.4,
        )


        outputs_label_1=gx.Text('Quality of board position', font='Georgia', slant='ITALIC', font_size=32)
        outputs_label_1.set_color(FRESH_TAN)
        outputs_label_1.move_to([-2, -4.5, 0])

        value_network=self.get_value_network()
        value_network.scale(0.72)
        value_network.rotate(90*gx.DEGREES, axis=gx.OUT)
        value_network.rotate(-10*gx.DEGREES, axis=gx.OUT)
        value_network.rotate(15*gx.DEGREES, axis=gx.RIGHT)
        value_network.rotate(-30*gx.DEGREES, axis=gx.UP)
        value_network.move_to([-7.1, -4.4, 0])


        self.wait()
        self.play(gx.FadeIn(border_2),
                  gx.FadeIn(board_3),
                  gx.FadeIn(state_label_2),
                  gx.FadeIn(arrow_in_2),
                  gx.FadeIn(arrow_out_2),
                  # FadeIn(outputs_label_1),
                  gx.Create(value_network),
                  run_time=4)
        self.wait()

        # self.remove(value_network[3])
        # self.add(value_network[3][-])
        self.play(gx.Write(outputs_label_1), run_time=3)
        self.wait()

        #Ok now move this down and Write in P(Win)
        pwin=gx.MathTex(r'P(\text{win})', font_size=60)
        pwin.move_to([[-2, -4.5, 0]])


        self.wait()
        self.play(outputs_label_1.animate.move_to([-2, -5.4, 0]), run_time=3)
        self.play(gx.Write(pwin),
                  run_time=3)
        self.wait()


        value_function_label = gx.Text(
            "VALUE FUNCTION",
            font="Myriad Pro",
            font_size=42,
        )
        value_function_label.set_color(GREEN)
        value_function_label.next_to(border_2, gx.DOWN, buff=0.3)

        self.play(gx.Write(value_function_label), run_time=3)
        self.wait()

        #P41 let's go!
        value_network_label = gx.Text(
            "VALUE NETWORK",
            font="Myriad Pro",
            font_size=36,
        )
        value_network_label.set_color(GREEN)
        value_network_label.next_to(border_2, gx.DOWN, buff=0.2)


        self.remove(value_function_label, outputs_label_1)
        self.wait()

        self.play(gx.Write(value_network_label))
        self.wait()

        # Ok so now I need to cycle through some games.
        # Should be able to borrow something from earlier, cycling through games

        self.remove(heatmap)
        self.remove(board_1, board_2, board_3)

        self_games_files=sorted(list(self.self_games_dir.glob('*.sgf')))

        num_games_to_play=30 ## CRANK UP IN FINAL RENDER
        for game_index in range(num_games_to_play):
        # game_index=0

            p=self_games_files[game_index]
            moves = parse_sgf(p)
            moves_to_show=np.random.randint(0, len(moves)-2)

            board_4=self.get_board()
            board_4_group=gx.Group()
            board_4_group.add(board_4)

            board_state = {}  # (x, y) -> color
            stone_objects = {}  # (x, y) -> Mobject
            for i, (x, y, color) in enumerate(moves[:moves_to_show]):
                stone = create_stone(x, y, color)
                board_4_group.add(stone)

                board_state[(x, y)] = color
                stone_objects[(x, y)] = stone

                # Check for captures
                captured = find_captures(x, y, board_state)
                for cx, cy in captured:
                    # Remove from scene and state
                    board_4_group.remove(stone_objects[(cx, cy)])
                    del board_state[(cx, cy)]
                    del stone_objects[(cx, cy)]

            x, y, c = moves[moves_to_show]
            pos = [(-(size-1)/2 + x) * step, (-(size-1)/2 + y) * step, 0]
            next_move=gx.Rectangle(0.5, 0.5)
            next_move.set_stroke(color=YELLOW, width=7)
            next_move.move_to(pos)

            board_5_group=copy.deepcopy(board_4_group)
            board_5_group.set_opacity(0.5)
            board_5_group.add(next_move)
            next_move.set_opacity(1.0)

            board_6_group=copy.deepcopy(board_4_group)

            board_4_group.scale(0.4*1.0)
            board_4_group.move_to([-12 , -0.05,  0. ])
            board_5_group.scale(0.4*1.0)
            board_5_group.move_to([-2.2 , -0.05,  0. ])
            board_6_group.scale(0.4*1.0)
            board_6_group.move_to([-12 , -0.05-4.5,  0. ])



            self.add(board_4_group, board_5_group, board_6_group)
            if game_index==0:
                self.play(pwin.animate.shift([-0.8, 0, 0]))

            if game_index%2==1: #Kinda ambiguous, I thnk ok to fake here.
                pwin_number = gx.MathTex('=0.0', font_size=60)
            else:
                pwin_number = gx.MathTex('=1.0', font_size=60)
            pwin_number.next_to(pwin, gx.RIGHT, buff=0.2)

            self.add(pwin_number)
            self.wait()

            if game_index<num_games_to_play-1:
                self.remove(board_4_group, board_5_group, board_6_group, pwin_number)

        self.wait()

        ## Hmm yeah when we do get to P42, I am going to want continuity with the
        ## little networks sliding to their spots in the MTCS tree.
        ## That's going to be kinda annoying, but I think makes sense.
        ## ok so i'm going to go ahead and hack on P42 in a separate class
        ## then when I know where the images and labels should be
        ## I'll come back here and do the transition
        ## Also gun to my head I can do the transition in Premiere




        self.wait(20)
