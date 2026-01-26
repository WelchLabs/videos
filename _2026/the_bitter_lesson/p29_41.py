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
        alphago_logo.move_to([0, 3, 0])


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

        
        # cnn.rotate(-10*DEGREES, axis=UP)



        # cnn.rotate(20*DEGREES, axis=UP)

        # cnn.rotate(62*DEGREES, axis=RIGHT) 
        # cnn.rotate, axis=UP)  
        # cnn.rotate(-59*DEGREES, axis=OUT)    


        self.add(alphago_logo)




        self.wait(20)
        self.embed()

















