from manimlib import *
from tqdm import tqdm
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


# def create_cnn_layer(width=19, height=19, cell_size=0.15, depth=0.3, 
#                      fill_color=BLUE, fill_opacity=0.8, line_width=1.0):
#     """Create a single CNN layer as a flat prism with grid lines."""
    
#     layer_w = width * cell_size
#     layer_h = height * cell_size
    
#     # Main box - using Prism for a rectangular shape
#     box = Cube(side_length=1)
#     box.set_width(layer_h, stretch=True)
#     box.set_depth(depth, stretch=True)
#     box.set_height(layer_w, stretch=True)
#     # box.set_opacity(0.5)
#     # box.set_fill(fill_color, opacity=fill_opacity)
#     # box.set_stroke(WHITE, width=line_width)
    
#     # Grid lines on the front face (facing camera)
#     grid_lines = Group()
    
#     front_z = depth / 2 + 0.001  # Slightly in front to avoid z-fighting
    
#     # Vertical lines
#     for i in range(width + 1):
#         x = -layer_w / 2 + i * cell_size
#         line = Line3D(
#             start=np.array([x, -layer_h / 2, front_z]),
#             end=np.array([x, layer_h / 2, front_z]),
#             width=0.02,
#             color=WHITE,
#         )
#         grid_lines.add(line)
    
#     # Horizontal lines
#     for j in range(height + 1):
#         y = -layer_h / 2 + j * cell_size
#         line = Line3D(
#             start=np.array([-layer_w / 2, y, front_z]),
#             end=np.array([layer_w / 2, y, front_z]),
#             width=0.02,
#             color=WHITE,
#         )
#         grid_lines.add(line)
    
#     layer = Group(box, grid_lines)
#     return layer

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

class AlphaGoCNN(InteractiveScene):
    def construct(self):
        spacing=0.8
        layers=Group()
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
            layers.add(layer)


        self.add(layers)
        # self.frame.reorient(-79, 82, 0, (0.91, -1.52, -0.24), 5.52)
        self.frame.reorient(-59, 62, 0, (0.7, -0.79, -0.76), 6.99)

        self.wait()


        # layer.rotate(90*DEGREES, [0, 1, 0])


        # self.frame.reorient(-74, 83, 0, (0.96, -1.75, -0.16), 8.47)
        

        self.wait()




        
        self.wait()
        self.embed()

