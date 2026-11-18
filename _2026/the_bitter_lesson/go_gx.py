"""Reusable Go board graphics, SGF move parsing, and capture logic."""

import re
from pathlib import Path

import manimgx as gx
import numpy as np

from welch_style import BLUE, CHILL_BROWN, FRESH_TAN

size = 19
padding = 0.5
board_width = 8
step = board_width / size


def parse_sgf(file_path):
    """Read 19×19 SGF moves, with y increasing up the board."""
    text = Path(file_path).read_text()
    return [
        (ord(coordinates[0]) - ord('a'), ord('s') - ord(coordinates[1]),
         '#000000' if player == 'B' else '#FFFFFF')
        for player, coordinates in re.findall(r"([BW])\[([a-s]{2})\]", text)
    ]


def create_stone(x, y, color=gx.BLACK, squash=0.3, resolution=(17, 9)):
    """Place a flattened sphere on one board intersection."""
    radius = step * 0.45
    first = -(size - 1) / 2 * step
    color = gx.BLACK if color == gx.BLACK or color == '#000000' else '#B0B0B0'
    stone = gx.Sphere(radius=radius, resolution=resolution, color=color, checkerboard_colors=False)
    stone.stretch(squash, dim=2)
    return stone.move_to([first + x * step, first + y * step, radius * squash])


def get_board(stroke_width=2, *, grid_stroke_width=None):
    """Return the wood panel, intersection lines, and nine star points."""
    board = gx.Square(side_length=board_width + padding, fill_color=FRESH_TAN,
                      fill_opacity=1, stroke_color=CHILL_BROWN, stroke_width=stroke_width)
    first = -(size - 1) / 2 * step
    lines = gx.VGroup()
    for index in range(size):
        pos = first + index * step
        lines.add(gx.Line([pos, first, 0], [pos, -first, 0]),
                  gx.Line([first, pos, 0], [-first, pos, 0]))
    lines.set_stroke(gx.BLACK, width=stroke_width if grid_stroke_width is None else grid_stroke_width)
    stars = gx.VGroup(*(
        gx.Dot([first + x * step, first + y * step, 0], radius=0.05, color=gx.BLACK, stroke_width=0)
        for x in (3, 9, 15) for y in (3, 9, 15)
    ))
    return gx.Group(board, lines, stars)


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
    opponent = '#FFFFFF' if color == gx.BLACK or color == '#000000' else '#000000'

    captured = set()
    for nx, ny in get_neighbors(x, y):
        if board_state.get((nx, ny)) == opponent:
            group = get_group(nx, ny, board_state)
            if len(get_liberties(group, board_state)) == 0:
                captured.update(group)

    return captured


def create_cnn_layer(width=19, height=19, cell_size=0.15, depth=0.1, fill_color=BLUE, fill_opacity=0.8, line_width=0.02):
    """Create a single CNN layer as a flat prism with grid lines."""

    layer_w = width * cell_size
    layer_h = height * cell_size

    box = gx.Prism(dimensions=(layer_w, layer_h, depth), fill_color=fill_color,
                   fill_opacity=fill_opacity)

    grid_lines = gx.Group()

    front_z = depth / 2 + 0.001
    back_z = -depth / 2 - 0.001

    # Front face grid - vertical lines
    for i in range(width + 1):
        x = -layer_w / 2 + i * cell_size
        line = gx.Line3D(
            start=np.array([x, -layer_h / 2, front_z]),
            end=np.array([x, layer_h / 2, front_z]),
            thickness=line_width, color=gx.WHITE,
        )
        grid_lines.add(line)

    # Front face grid - horizontal lines
    for j in range(height + 1):
        y = -layer_h / 2 + j * cell_size
        line = gx.Line3D(
            start=np.array([-layer_w / 2, y, front_z]),
            end=np.array([layer_w / 2, y, front_z]),
            thickness=line_width, color=gx.WHITE,
        )
        grid_lines.add(line)

    # Back face grid - vertical lines
    for i in range(width + 1):
        x = -layer_w / 2 + i * cell_size
        line = gx.Line3D(
            start=np.array([x, -layer_h / 2, back_z]),
            end=np.array([x, layer_h / 2, back_z]),
            thickness=line_width, color=gx.WHITE,
        )
        grid_lines.add(line)

    # Back face grid - horizontal lines
    for j in range(height + 1):
        y = -layer_h / 2 + j * cell_size
        line = gx.Line3D(
            start=np.array([-layer_w / 2, y, back_z]),
            end=np.array([layer_w / 2, y, back_z]),
            thickness=line_width, color=gx.WHITE,
        )
        grid_lines.add(line)

    # Edge lines connecting front to back (top and bottom edges)
    for i in range(width + 1):
        x = -layer_w / 2 + i * cell_size
        # Top edge
        line = gx.Line3D(
            start=np.array([x, layer_h / 2, front_z]),
            end=np.array([x, layer_h / 2, back_z]),
            thickness=line_width, color=gx.WHITE,
        )
        grid_lines.add(line)
        # Bottom edge
        line = gx.Line3D(
            start=np.array([x, -layer_h / 2, front_z]),
            end=np.array([x, -layer_h / 2, back_z]),
            thickness=line_width, color=gx.WHITE,
        )
        grid_lines.add(line)

    # Edge lines connecting front to back (left and right edges)
    for j in range(height + 1):
        y = -layer_h / 2 + j * cell_size
        # Right edge
        line = gx.Line3D(
            start=np.array([layer_w / 2, y, front_z]),
            end=np.array([layer_w / 2, y, back_z]),
            thickness=line_width, color=gx.WHITE,
        )
        grid_lines.add(line)
        # Left edge
        line = gx.Line3D(
            start=np.array([-layer_w / 2, y, front_z]),
            end=np.array([-layer_w / 2, y, back_z]),
            thickness=line_width, color=gx.WHITE,
        )
        grid_lines.add(line)

    layer = gx.Group(box, grid_lines)
    return layer
