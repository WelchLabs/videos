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

class AlphaGoValueNetwork(InteractiveScene):
    def construct(self):
        '''
        This network should have 3 convolutional layers, and then a "pyramid" projecting
        the 19x19 grid down to a single pixel. The single pixel should be a Cube with white 
        borders on all sides. The pyramid should be fairly opaque, and I don't think that we
        need borders on it. Let's make all volumes in this network green. 
        '''
        
        spacing = 0.8
        cell_size = 0.15
        layer_size = 19 * cell_size  # Width/height of the 19x19 grid
        
        layers = Group()
        
        # Create 3 convolutional layers (green)
        for i in range(3):
            layer = create_cnn_layer(
                width=19, 
                height=19, 
                cell_size=cell_size, 
                depth=0.15,
                fill_color=CHILL_GREEN,
            )
            layer.rotate(90 * DEGREES, [1, 0, 0])
            layer[0].set_color(CHILL_GREEN) #, opacity=0.6)
            layer[0].set_opacity(0.6)
            layer[1].set_opacity(0.6)
            layer.move_to([0, -spacing * i, 0])
            layers.add(layer)
        
        # Position for pyramid base (bottom of last conv layer)
        last_layer_y = -spacing * 2
        pyramid_base_y = last_layer_y - 0.15 / 2 - 0.01  # Just below last layer
        
        # Position for output cube
        output_y = last_layer_y - spacing * 1.5
        
        # Output cube size
        output_size = 0.15
        pyramid_top_y = output_y + output_size / 2 + 0.01
        
        # Create pyramid connecting last conv layer to output cube
        # After rotation, the grid is in the x-z plane
        half_size = layer_size / 2
        half_output = output_size / 2
        
        # Pyramid vertices - base is the 19x19 layer, top is the small output
        # Base corners (at pyramid_base_y)
        base_corners = [
            np.array([-half_size, pyramid_base_y, -half_size]),  # back-left
            np.array([half_size, pyramid_base_y, -half_size]),   # back-right
            np.array([half_size, pyramid_base_y, half_size]),    # front-right
            np.array([-half_size, pyramid_base_y, half_size]),   # front-left
        ]
        
        # Top corners (at pyramid_top_y) - small square for output cube
        top_corners = [
            np.array([-half_output, pyramid_top_y, -half_output]),  # back-left
            np.array([half_output, pyramid_top_y, -half_output]),   # back-right
            np.array([half_output, pyramid_top_y, half_output]),    # front-right
            np.array([-half_output, pyramid_top_y, half_output]),   # front-left
        ]
        
        # Create the 4 trapezoidal faces of the frustum/pyramid
        pyramid_faces = Group()
        
        for i in range(4):
            next_i = (i + 1) % 4
            # Each face is a quadrilateral: base[i], base[next_i], top[next_i], top[i]
            face = Polygon(
                base_corners[i],
                base_corners[next_i],
                top_corners[next_i],
                top_corners[i],
            )
            face.set_fill(CHILL_GREEN, opacity=0.7)
            face.set_stroke(width=0)  # No borders on pyramid
            pyramid_faces.add(face)
        
        # Add Line3D borders to pyramid edges
        pyramid_edges = Group()
        line_width = 0.02
        
        # 4 angled edges connecting base to top corners
        for i in range(4):
            edge = Line3D(
                start=base_corners[i],
                end=top_corners[i],
                width=line_width,
                color=WHITE,
            )
            pyramid_edges.add(edge)
        
        # 4 edges around the top square
        for i in range(4):
            next_i = (i + 1) % 4
            edge = Line3D(
                start=top_corners[i],
                end=top_corners[next_i],
                width=line_width,
                color=WHITE,
            )
            pyramid_edges.add(edge)
        
        # Create single output cube with white borders
        output_cube = Cube(side_length=output_size)
        output_cube.set_color(CHILL_GREEN)
        output_cube.move_to([0, output_y, 0])
        
        # Add Line3D borders to output cube
        cube_edges = Group()
        s = output_size / 2
        
        # Define the 8 corners of the cube
        cube_corners = [
            np.array([-s, output_y - s, -s]),  # 0: bottom-back-left
            np.array([s, output_y - s, -s]),   # 1: bottom-back-right
            np.array([s, output_y - s, s]),    # 2: bottom-front-right
            np.array([-s, output_y - s, s]),   # 3: bottom-front-left
            np.array([-s, output_y + s, -s]),  # 4: top-back-left
            np.array([s, output_y + s, -s]),   # 5: top-back-right
            np.array([s, output_y + s, s]),    # 6: top-front-right
            np.array([-s, output_y + s, s]),   # 7: top-front-left
        ]
        
        # Define the 12 edges of the cube (pairs of corner indices)
        cube_edge_pairs = [
            # Bottom face
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top face
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        
        for start_idx, end_idx in cube_edge_pairs:
            edge = Line3D(
                start=cube_corners[start_idx],
                end=cube_corners[end_idx],
                width=line_width,
                color=WHITE,
            )
            cube_edges.add(edge)
        
        # Add everything to scene
        self.add(layers, pyramid_faces, pyramid_edges, output_cube, cube_edges)
        self.remove(pyramid_faces); self.add(pyramid_faces)
        self.remove(cube_edges[6]); self.add(cube_edges[6])
        self.remove(cube_edges[7]); self.add(cube_edges[7])
        self.remove(cube_edges[11]); self.add(cube_edges[11])

        
        # Set camera angle similar to AlphaGoCNN
        # self.frame.reorient(-59, 62, 0, (0.7, -1.5, -0.76), 8.0)

        self.frame.reorient(-58, 62, 0, (0.4, -0.59, -0.36), 6.51)
        
        self.wait()
        self.embed()




# class AlphaGoValueNetwork(InteractiveScene):
#     def construct(self):
#         '''
#         This network should have 3 convolutional layers, and then a "pyramid" projecting
#         the 19x19 grid down to a single pixel. The single pixel should be a Cube with white 
#         borders on all sides. The pyramid should be fairly opaque, and I don't think that we
#         need borders on it. Let's make all volumes in this network green. 
#         '''
        
#         spacing = 0.8
#         cell_size = 0.15
#         layer_size = 19 * cell_size  # Width/height of the 19x19 grid
        
#         layers = Group()
        
#         # Create 3 convolutional layers (green)
#         for i in range(3):
#             layer = create_cnn_layer(
#                 width=19, 
#                 height=19, 
#                 cell_size=cell_size, 
#                 depth=0.15,
#                 fill_color=CHILL_GREEN,
#             )
#             layer.rotate(90 * DEGREES, [1, 0, 0])
#             layer[0].set_color(CHILL_GREEN) #, opacity=0.6)
#             layer[0].set_opacity(0.6)
#             layer[1].set_opacity(0.6)
#             layer.move_to([0, -spacing * i, 0])
#             layers.add(layer)
        
#         # Position for pyramid base (bottom of last conv layer)
#         last_layer_y = -spacing * 2
#         pyramid_base_y = last_layer_y - 0.15 / 2 - 0.01  # Just below last layer
        
#         # Position for output cube
#         output_y = last_layer_y - spacing * 1.5
        
#         # Output cube size
#         output_size = 0.15
#         pyramid_top_y = output_y + output_size / 2 + 0.01
        
#         # Create pyramid connecting last conv layer to output cube
#         # After rotation, the grid is in the x-z plane
#         half_size = layer_size / 2
#         half_output = output_size / 2
        
#         # Pyramid vertices - base is the 19x19 layer, top is the small output
#         # Base corners (at pyramid_base_y)
#         base_corners = [
#             np.array([-half_size, pyramid_base_y, -half_size]),  # back-left
#             np.array([half_size, pyramid_base_y, -half_size]),   # back-right
#             np.array([half_size, pyramid_base_y, half_size]),    # front-right
#             np.array([-half_size, pyramid_base_y, half_size]),   # front-left
#         ]
        
#         # Top corners (at pyramid_top_y) - small square for output cube
#         top_corners = [
#             np.array([-half_output, pyramid_top_y, -half_output]),  # back-left
#             np.array([half_output, pyramid_top_y, -half_output]),   # back-right
#             np.array([half_output, pyramid_top_y, half_output]),    # front-right
#             np.array([-half_output, pyramid_top_y, half_output]),   # front-left
#         ]
        
#         # Create the 4 trapezoidal faces of the frustum/pyramid
#         pyramid_faces = Group()
        
#         for i in range(4):
#             next_i = (i + 1) % 4
#             # Each face is a quadrilateral: base[i], base[next_i], top[next_i], top[i]
#             face = Polygon(
#                 base_corners[i],
#                 base_corners[next_i],
#                 top_corners[next_i],
#                 top_corners[i],
#             )
#             face.set_fill(CHILL_GREEN, opacity=0.7)
#             face.set_stroke(width=0)  # No borders on pyramid
#             pyramid_faces.add(face)
        
#         # Create single output cube with white borders
#         output_cube = Cube(side_length=output_size)
#         output_cube.set_color(CHILL_GREEN) #, opacity=0.8)
#         # output_cube.set_stroke(WHITE, width=2)
#         output_cube.move_to([0, output_y, 0])
        
#         # Add everything to scene
#         self.add(layers, pyramid_faces, output_cube)
        
#         # Set camera angle similar to AlphaGoCNN
#         self.frame.reorient(-59, 62, 0, (0.7, -1.5, -0.76), 8.0)
        
#         self.wait()
#         self.embed()


