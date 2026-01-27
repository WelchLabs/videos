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
# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/human_human_kgs-19-2015')
# games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/games_with_videos')
games_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/games/less_wrong_reverse_engineer')

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/graphics/to_manim/')

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
                  next_move_1.animate.move_to([ 5.78, -0.11,  0.        ]),
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

        self.wait()


        self.wait()

        self.wait(20)
        self.embed()




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














