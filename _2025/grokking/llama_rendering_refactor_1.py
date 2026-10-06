"""
llama_rendering_refactor_1.py

Static render of a forward + backward pass through Llama, for print.
Refactored from llama_learning_animation_11.py: no draw-in animation, no camera
moves, just the finished network. Everything you'd want to tweak lives in the
CONFIG block below.

Save a still:
    manimgl llama_rendering_refactor_1.py LlamaRender -ws --config_file custom_config.yml
    (add e.g. `-r 3840x2160` for more pixels)

Variants: subclass LlamaRender and set `overrides` -- see the examples at the
bottom of the file.
"""

from manimlib import *
import glob
import os
import pickle
from types import SimpleNamespace

import numpy as np
import matplotlib.colors as mcolors

CHILL_BROWN='#928779'
FRESH_TAN='#f2ecdb'


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = dict(

    # --- Data ----------------------------------------------------------------
    # A single .p file, or a directory of them (sorted, then SNAPSHOT_INDEX picks one).
    SNAPSHOT_PATH='/Users/stephen/Library/CloudStorage/Dropbox-Stephencwelch/welch_labs/ai_book_vol_2/2_grokking/hackin/jun_3_1',
    SNAPSHOT_INDEX=9,

    # --- What to draw --------------------------------------------------------
    SHOW_WEIGHTS=True,    # forward-pass connection lines
    SHOW_FORWARD=True,    # activations: neuron fills + attention patterns (False = empty network)
    SHOW_BACKWARD=True,   # gradient lines
    SHOW_TEXT=True,       # prompt tokens + predicted tokens

    # --- Colors --------------------------------------------------------------
    BACKGROUND_COLOR='#153235',  # page / scene background
    EMPTY_COLOR=None,            # 1. empty attention squares + neuron interiors. None = BACKGROUND_COLOR
    LINE_COLOR='#cec3b0', #948979',        # 2. outlines: attention grids + borders, neuron circles, ellipses
    FORWARD_COLOR=FRESH_TAN, #'#dfd0b9',     # 3. forward pass: connection lines, neuron fills, attention fills
    TEXT_COLOR=FRESH_TAN, #dfd0b9',        # 4. token text
    GRADIENT_COLOR='#70abb4', ##70abb4', ##00FFFF',    # 5. backward pass lines

    # Optional per-element overrides. None = use the palette entry noted.
    NEURON_STROKE_COLOR=None,    # None = LINE_COLOR     (original video: '#dfd0b9')
    ATTN_STROKE_COLOR=None,      # None = LINE_COLOR
    ELLIPSIS_COLOR=None,         # None = LINE_COLOR
    WEIGHT_COLOR=None,           # None = FORWARD_COLOR  (original video: '#948979')

    # --- 6. Depth ------------------------------------------------------------
    # An int N draws N layers spread evenly from first to last (16 = full depth for this model),
    # or give an explicit list of layer indices.
    # DEPTH=[0, 1, 6, 7, 8, 9, 14, 15],
    DEPTH=16,

    # --- 7. Width ------------------------------------------------------------
    # Counts are neurons / heads actually drawn (each column also gets one "..." slot).
    # Limits come from what's stored in the snapshots.
    MLP_WIDTH=31,            # residual-stream neurons per MLP in/out column   (max 32)
    MLP_HIDDEN_WIDTH=None,   # MLP hidden column. None = MLP_WIDTH + 2         (max 34)
    ATTN_WIDTH=10,           # attention heads per layer                        (max 32)
    ATTN_HEADS=None,         # which heads: None = evenly strided, or an explicit list of head indices
    IO_WIDTH=None,           # token neurons in input/output columns. None = MLP_WIDTH + 4 (max 40)

    # --- 8. Line rendering ---------------------------------------------------
    # One row per family of connection lines. For each matrix: s = |value| / percentile(|values|, pct)
    #   pct     : normalisation percentile (100 = max). Lower -> more lines saturate.
    #   min     : draw only lines with s >= min. Raise for a sparser picture.
    #   window  : draw only "local" lines -- max vertical offset, in neuron (mlp_*, unembed*)
    #             or head (attn_*, embed*) units. Raise for more long diagonal lines.
    #   gain    : stroke width = clip(gain * s, *width)
    #   width   : (lo, hi) clip for stroke width; hi=None means unclipped
    #   opacity : (lo, hi) clip for stroke opacity (= s)
    #   alpha   : optional, overall opacity multiplier for the whole family (default 1.0)
    RENDER=dict(
        # forward weights
        embed=dict(pct=95, min=0.0, window=3, gain=1.0, width=(0.5, 3), opacity=(0.4, 1)),          # input tokens -> first attention
        attn_in=dict(pct=100, min=0.5, window=3, gain=1.0, width=(0, 4), opacity=(0, 1)),             # prev MLP -> attention (W_Q)
        attn_out=dict(pct=99, min=0.6, window=3, gain=1.0, width=(0, 4), opacity=(0, 1)),             # attention -> MLP (W_O)
        mlp_in=dict(pct=99, min=0.75, window=6, gain=1.0, width=(0, 4), opacity=(0, 1)),           # MLP in -> hidden
        mlp_out=dict(pct=99, min=0.45, window=6, gain=1.0, width=(0, 4), opacity=(0, 1)),          # MLP hidden -> out
        unembed=dict(pct=98, min=0.5, window=8, gain=1.0, width=(0.5, 3), opacity=(0.4, 1)),        # last MLP -> output tokens
        # gradients
        # (alpha=0.5: in the video the embedding gradients only ever reached half opacity,
        #  because the fade-in timer stopped 0.5s early. Set to 1.0 for full strength.)

        # embed_grad=dict(pct=95, min=0.0, window=4, gain=1.0, width=(0, 3), opacity=(0, 1), alpha=1.0),
        # attn_in_grad=dict(pct=98, min=0.5, window=3, gain=1.0, width=(0, 2), opacity=(0, 1)),
        # attn_out_grad=dict(pct=98, min=0.5, window=3, gain=1.0, width=(0, 3), opacity=(0, 1)),
        # mlp_in_grad=dict(pct=95, min=0.5, window=6, gain=2.0, width=(0, 3), opacity=(0, 1)),
        # mlp_out_grad=dict(pct=97, min=0.5, window=6, gain=1.0, width=(0, 3), opacity=(0, 1)),
        # unembed_grad=dict(pct=99, min=0.5, window=8, gain=0.7, width=(0, 3), opacity=(0, 1)),

        embed_grad=dict(pct=95, min=0.0, window=4, gain=1.0, width=(0, 4), opacity=(0, 1), alpha=1.0),
        attn_in_grad=dict(pct=98, min=0.5, window=3, gain=1.0, width=(0, 4), opacity=(0, 1)),
        attn_out_grad=dict(pct=98, min=0.5, window=3, gain=1.0, width=(0, 4), opacity=(0, 1)),
        mlp_in_grad=dict(pct=95, min=0.5, window=6, gain=2.0, width=(0, 4), opacity=(0, 1)),
        mlp_out_grad=dict(pct=97, min=0.5, window=6, gain=1.0, width=(0, 4), opacity=(0, 1)),
        unembed_grad=dict(pct=99, min=0.5, window=8, gain=0.7, width=(0, 4), opacity=(0, 1)),
    ),
    # Gradient lines are always tinted BACKGROUND_COLOR -> GRADIENT_COLOR by magnitude.
    # 1.0 = fully opaque (how the finished video frame looked). None = also fade opacity by
    # magnitude using the `opacity` clip above, which tends to work better on a light page.
    GRAD_OPACITY=1.0,
    ATTN_PATTERN_GAIN=2.5,   # attention square brightness = clip(gain * value / max, 0, 1)
    NEURON_FILL_GAIN=1.0,    # neuron brightness           = clip(gain * |value| / max, 0, 1)

    # Stroke widths. ManimGL strokes don't scale with camera zoom, so a deeper (wider) network
    # would otherwise get relatively heavier lines. AUTO keeps line weight proportional to the
    # drawing, matched to the original 8-layer framing.
    STROKE_SCALE=1.0, #1.0,        # global multiplier on every stroke
    AUTO_STROKE_SCALE=True,
    OUTLINE_PASSES=2,        # times neuron + attention-grid outlines are drawn. The video drew them twice
                             # (empty network under the filled one), which reads slightly bolder. 1 = finer.
    NEURON_STROKE_WIDTH=1.0,
    ATTN_GRID_STROKE_WIDTH=0.5,
    ATTN_BORDER_STROKE_WIDTH=1.0,

    # --- Text ----------------------------------------------------------------
    FONT='myriad-pro',
    INPUT_FONT_SIZE=24,
    OUTPUT_FONT_SIZES=(22, 16, 12),       # top-1, top-2..4, everything else
    OUTPUT_TEXT_FADE=True,                # fade predicted tokens toward the background by probability
    OUTPUT_TEXT_FLOOR=0.1,                # ...treating probabilities below this as this (keeps the tail legible)
    TEXT_NUDGES={' capital': -0.02},      # per-token vertical nudges for descenders etc.
    PROMPT_NEURONS=None,                  # input neurons the prompt tokens sit on. None = seeded random draw
    PROMPT_SEED=25,                       #   (seed 25 -> [4, 26, 15, 23, 8] at the default width)

    # --- Camera --------------------------------------------------------------
    FRAME=None,              # None = auto-fit. Or (center_x, center_y, height); original was (2.06, -0.06, 9.36)
    FRAME_MARGIN=0.35,       # padding around the drawing when auto-fitting

    # --- Geometry (rarely needs changing) --------------------------------------
    NEURON_RADIUS=0.06,
    NEURON_SPACING=0.18,          # vertical pitch of neurons
    MLP_COLUMN_SPACING=0.23,      # horizontal gap between MLP in / hidden / out columns
    LAYER_PITCH=1.6,              # horizontal distance between consecutive layers
    MLP_OFFSET=0.8,               # attention block -> its MLP
    INPUT_GAP=0.7,                # input column -> first attention block
    OUTPUT_GAP=0.36,              # last MLP -> output column
    TEXT_GAP=0.2,                 # neuron column -> token text
    ATTN_SQUARE_SIZE=0.08,
    ATTN_PATTERN_SPACING=0.51,    # vertical pitch of attention patterns
    ATTN_ELLIPSIS_SQUEEZE=0.15,   # how far the two halves close in on the "..."
    ATTN_BORDER_PAD=0.095,        # padding between patterns and the rounded border
    START_X=-4.0,                 # x of the first attention block
)

# The locality windows for attention/embedding lines were tuned at these widths; they get
# rescaled proportionally when the widths change.
_REF = dict(mlp=31, attn=10, io=35, frame_height=9.36)


# =============================================================================
# Helpers
# =============================================================================
def resolve_config(overrides=None):
    """Merge overrides into CONFIG, fill in the None defaults, return a namespace."""
    merged = dict(CONFIG)
    overrides = dict(overrides or {})
    if 'RENDER' in overrides:  # allow partial overrides, e.g. RENDER=dict(mlp_in=dict(min=0.9))
        render = {k: dict(v) for k, v in CONFIG['RENDER'].items()}
        for name, row in overrides.pop('RENDER').items():
            render[name].update(row)
        merged['RENDER'] = render
    unknown = set(overrides) - set(CONFIG)
    if unknown:
        raise KeyError(f"Unknown config keys: {sorted(unknown)}")
    merged.update(overrides)
    cfg = SimpleNamespace(**merged)

    def default(name, fallback):
        if getattr(cfg, name) is None:
            setattr(cfg, name, fallback)

    default('EMPTY_COLOR', cfg.BACKGROUND_COLOR)
    default('NEURON_STROKE_COLOR', cfg.LINE_COLOR)
    default('ATTN_STROKE_COLOR', cfg.LINE_COLOR)
    default('ELLIPSIS_COLOR', cfg.LINE_COLOR)
    default('WEIGHT_COLOR', cfg.FORWARD_COLOR)
    default('MLP_HIDDEN_WIDTH', cfg.MLP_WIDTH + 2)
    default('IO_WIDTH', cfg.MLP_WIDTH + 4)

    two_color = lambda a, b: mcolors.LinearSegmentedColormap.from_list('c', [a, b], N=256)
    cfg.fill_cmap = two_color(cfg.EMPTY_COLOR, cfg.FORWARD_COLOR)
    cfg.grad_cmap = two_color(cfg.BACKGROUND_COLOR, cfg.GRADIENT_COLOR)
    cfg.text_cmap = two_color(cfg.BACKGROUND_COLOR, cfg.TEXT_COLOR)
    return cfg


def load_snapshot(path, index=0):
    """Load one snapshot and convert every array to numpy (handles torch tensors too)."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.p')))
        if not files:
            raise FileNotFoundError(f"No .p files in {path}")
        path = files[index]
    with open(path, 'rb') as f:
        raw = pickle.load(f)

    def to_numpy(v):
        return v.detach().cpu().numpy() if hasattr(v, 'detach') else v  # torch tensor -> numpy

    return {k: to_numpy(v) for k, v in raw.items()}


def check_limit(name, value, limit):
    if not 1 <= value <= limit:
        raise ValueError(f"{name}={value} is out of range: the snapshot supports 1..{limit}")


def pick_layers(depth, n_layers):
    if isinstance(depth, (int, np.integer)):
        check_limit('DEPTH', depth, n_layers)
        if depth == 1:
            return [0]
        return [int(round(x)) for x in np.linspace(0, n_layers - 1, depth)]
    layers = list(depth)
    for l in layers:
        if not 0 <= l < n_layers:
            raise ValueError(f"DEPTH contains layer {l}, but the snapshot has layers 0..{n_layers - 1}")
    return layers


def pick_heads(cfg, n_heads):
    if cfg.ATTN_HEADS is not None:
        heads = list(cfg.ATTN_HEADS)
        for h in heads:
            if not 0 <= h < n_heads:
                raise ValueError(f"ATTN_HEADS contains head {h}, but the snapshot has heads 0..{n_heads - 1}")
        return heads
    n = cfg.ATTN_WIDTH
    check_limit('ATTN_WIDTH', n, n_heads)
    step = max(n_heads // n, 1)
    start = 1 if 1 + step * (n - 1) < n_heads else 0  # n=10 of 32 -> 1, 4, ..., 28 as in the original
    return [start + step * k for k in range(n)]


def cmap_hex(cmap, t):
    t = float(t)
    if not np.isfinite(t):
        t = 0.0
    return mcolors.to_hex(cmap(float(np.clip(t, 0, 1)))[:3])


def make_ellipsis(center, scale, color):
    dots = Tex("...").rotate(PI / 2, OUT).scale(scale)
    dots.move_to(center)
    dots.set_color(color)
    return dots


def centers_of(mobjects):
    return [m.get_center() for m in mobjects]


# =============================================================================
# Building blocks
# =============================================================================
def neuron_column(n, x, cfg, fill_colors=None, ellipsis_scale=0.5):
    """
    A vertical column of n neurons with a "..." in the middle slot.
    fill_colors: list of n colors, or None for empty neurons.
    Returns (neurons VGroup, ellipsis).
    """
    slots = n + 1
    neurons = VGroup()
    ellipsis = None
    k = 0
    for slot in range(slots):
        pos = np.array([x, (slots // 2 - slot) * cfg.NEURON_SPACING, 0.0])
        if slot == slots // 2:
            ellipsis = make_ellipsis(pos, ellipsis_scale, cfg.ELLIPSIS_COLOR)
            continue
        neuron = Circle(radius=cfg.NEURON_RADIUS, stroke_color=cfg.NEURON_STROKE_COLOR)
        neuron.set_stroke(width=cfg.NEURON_STROKE_WIDTH)
        neuron.set_fill(color=cfg.EMPTY_COLOR if fill_colors is None else fill_colors[k], opacity=1.0)
        neuron.move_to(pos)
        neurons.add(neuron)
        k += 1
    return neurons, ellipsis


def activation_colors(values, n, cfg):
    """Fill colors for the first n entries of an activation vector (normalised by its max |value|)."""
    values = np.asarray(values)
    vmax = np.abs(values).max()
    return [cmap_hex(cfg.fill_cmap, cfg.NEURON_FILL_GAIN * abs(values[k]) / vmax) for k in range(n)]


def attention_pattern(matrix, cfg, filled=True):
    """Grid of squares for one head's attention pattern."""
    matrix = np.asarray(matrix)
    vmax = np.max(matrix)
    size = cfg.ATTN_SQUARE_SIZE
    squares = VGroup()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if filled and vmax > 0:
                color = cmap_hex(cfg.fill_cmap, cfg.ATTN_PATTERN_GAIN * matrix[i, j] / vmax)
            else:
                color = cfg.EMPTY_COLOR
            square = Square(side_length=size)
            square.set_fill(color, opacity=1.0)
            square.set_stroke(cfg.ATTN_STROKE_COLOR, width=cfg.ATTN_GRID_STROKE_WIDTH)
            square.move_to(RIGHT * j * size + DOWN * i * size)
            squares.add(square)
    squares.move_to(ORIGIN)
    return squares


def attention_block(patterns, x, cfg):
    """
    One layer's attention block: stacked patterns with a "..." in the middle, inside a rounded border.
    Returns (patterns VGroup, border, left anchor points, right anchor points).
    """
    slots = len(patterns) + 1
    pitch = cfg.ATTN_PATTERN_SPACING
    rows, cols = np.asarray(patterns[0]).shape
    pattern_w, pattern_h = cols * cfg.ATTN_SQUARE_SIZE, rows * cfg.ATTN_SQUARE_SIZE

    border_w = pattern_w + 2 * cfg.ATTN_BORDER_PAD
    top_center = (slots - 1) * pitch / 2 - cfg.ATTN_ELLIPSIS_SQUEEZE
    border_h = 2 * (top_center + pattern_h / 2 + 0.1)
    border = RoundedRectangle(width=border_w, height=border_h, corner_radius=0.1)
    border.set_stroke(width=cfg.ATTN_BORDER_STROKE_WIDTH, color=cfg.ATTN_STROKE_COLOR)
    border.move_to([x, 0, 0])

    group = VGroup()
    left_pts, right_pts = [], []
    k = 0
    for slot in range(slots):
        y = slots * pitch / 2 - pitch * (slot + 0.5)
        if slot == slots // 2:
            group.add(make_ellipsis([x, y, 0], 0.5, cfg.ELLIPSIS_COLOR))
            continue
        y += cfg.ATTN_ELLIPSIS_SQUEEZE if slot > slots // 2 else -cfg.ATTN_ELLIPSIS_SQUEEZE
        pattern = attention_pattern(patterns[k], cfg, filled=cfg.SHOW_FORWARD)
        pattern.move_to([x, y, 0])
        group.add(pattern)
        left_pts.append(np.array([x - border_w / 2, y, 0.0]))
        right_pts.append(np.array([x + border_w / 2, y, 0.0]))
        k += 1
    return group, border, left_pts, right_pts


def connection_lines(values, src, dst, style, cfg, offset, src_radius=0.0, dst_radius=0.0, grad=False):
    """
    Lines between two sets of points, styled by a weight (or gradient) matrix.

    values[i, j] : weight between src[i] and dst[j]; may be larger than len(src) x len(dst),
                   normalisation uses the whole matrix.
    src, dst     : lists of 3D points
    style        : one row of cfg.RENDER
    offset(i, j) : vertical "distance" between the two ends, compared against style['window']
    *_radius     : lines stop this far short of the point (i.e. at the edge of a neuron)
    """
    mag = np.abs(np.asarray(values))
    ref = np.percentile(mag, style['pct'])
    scaled = mag / ref if ref > 0 else np.zeros_like(mag)
    w_lo, w_hi = style['width']
    o_lo, o_hi = style['opacity']
    alpha = style.get('alpha', 1.0)

    lines = VGroup()
    for i, a in enumerate(src):
        for j, b in enumerate(dst):
            s = float(scaled[i, j])
            if s < style['min']:
                continue
            if offset(i, j) > style['window']:
                continue
            unit = (b - a) / np.linalg.norm(b - a)
            line = Line(a + unit * src_radius, b - unit * dst_radius)
            if grad:
                color = cmap_hex(cfg.grad_cmap, s)
                opacity = cfg.GRAD_OPACITY if cfg.GRAD_OPACITY is not None else float(np.clip(s, o_lo, o_hi))
            else:
                color = cfg.WEIGHT_COLOR
                opacity = float(np.clip(s, o_lo, o_hi))
            line.set_stroke(color=color, width=float(np.clip(style['gain'] * s, w_lo, w_hi)), opacity=float(opacity) * alpha)
            lines.add(line)
    return lines


def io_text(token, font_size, color, x_side, slot_y, cfg):
    """Token label beside an input (x_side=-1) or output (x_side=+1) neuron at height slot_y."""
    t = Text(token, font_size=font_size, font=cfg.FONT)
    t.set_color(color)
    half_width = t.get_right()[0]
    # Small upward nudge proportional to the text's height, carried over from the original layout.
    y = slot_y - t.get_bottom()[1] * cfg.NEURON_SPACING
    t.move_to([x_side * (cfg.TEXT_GAP + half_width), y + cfg.TEXT_NUDGES.get(token, 0.0), 0])
    return t


# =============================================================================
# Whole network
# =============================================================================
def build_network(snapshot, cfg):
    """
    Returns a dict of VGroups, in back-to-front drawing order:
        background, weights, grads, activations, text
    """
    R = cfg.RENDER
    r = cfg.NEURON_RADIUS

    # ---- What does the snapshot contain? ----
    n_layers = 1 + max(int(k.split('.')[1]) for k in snapshot if k.startswith('blocks.'))
    d_resid, d_hidden = snapshot['blocks.0.mlp.W_in'].shape
    n_heads = snapshot['blocks.0.attn.hook_pattern'].shape[1]
    n_tokens = len(snapshot['topk.probs'])

    M, H, V = cfg.MLP_WIDTH, cfg.MLP_HIDDEN_WIDTH, cfg.IO_WIDTH
    check_limit('MLP_WIDTH', M, d_resid)
    check_limit('MLP_HIDDEN_WIDTH', H, d_hidden)
    check_limit('IO_WIDTH', V, n_tokens)
    layers = pick_layers(cfg.DEPTH, n_layers)
    heads = pick_heads(cfg, n_heads)
    A = len(heads)

    # Locality: map a neuron index onto the head axis so "nearby" means the same thing at any width.
    mlp_to_head = 0.25 * (A / _REF['attn']) * (_REF['mlp'] / M)
    io_to_head = 0.25 * (A / _REF['attn']) * (_REF['io'] / V)

    background, weights, grads, activations, text = VGroup(), VGroup(), VGroup(), VGroup(), VGroup()

    # ---- Layers ----
    first_attn_left = None
    prev_mlp_out = None
    last_mlp_out = None
    last_mlp_right = None
    layer_weights, layer_grads = [], []

    for count, layer in enumerate(layers):
        key = f'blocks.{layer}.'
        x_attn = cfg.START_X + count * cfg.LAYER_PITCH
        x_mlp = x_attn + cfg.MLP_OFFSET

        # Attention block (drop BOS and the final "answer" token from each pattern)
        all_patterns = snapshot[key + 'attn.hook_pattern'][0]
        patterns = [all_patterns[h][1:-1, 1:-1] for h in heads]
        attn_group, border, left_pts, right_pts = attention_block(patterns, x_attn, cfg)
        background.add(border)
        activations.add(attn_group)
        if first_attn_left is None:
            first_attn_left = left_pts

        # MLP: in / hidden / out columns
        if cfg.SHOW_FORWARD:
            fills = [activation_colors(snapshot[key + 'hook_resid_mid'], M, cfg),
                     activation_colors(snapshot[key + 'mlp.hook_post'], H, cfg),
                     activation_colors(snapshot[key + 'hook_mlp_out'], M, cfg)]
        else:
            fills = [None, None, None]
        dx = cfg.MLP_COLUMN_SPACING
        col_in, dots_in = neuron_column(M, -dx, cfg, fills[0])
        col_hid, dots_hid = neuron_column(H, 0, cfg, fills[1])
        col_out, dots_out = neuron_column(M, dx, cfg, fills[2])
        mlp = VGroup(col_in, col_hid, col_out, dots_in, dots_hid, dots_out)
        mlp.move_to([x_mlp, 0, 0])
        background.add(dots_in, dots_hid, dots_out)
        activations.add(VGroup(col_in, col_hid, col_out))

        p_in, p_hid, p_out = centers_of(col_in), centers_of(col_hid), centers_of(col_out)

        # Attention <-> MLP wiring. Head h's query weights come in from the previous MLP,
        # its output weights go to this layer's MLP.
        W_Q = snapshot[key + 'attn.W_Q'][heads][:, :, 0].T       # [resid, head]
        W_O = snapshot[key + 'attn.W_O'][heads][:, 0, :]         # [head, resid]
        W_Q_grad = snapshot[key + 'attn.W_Q.grad'][heads][:, :, 0].T
        W_O_grad = snapshot[key + 'attn.W_O.grad'][heads][:, 0, :]

        this_w, this_g = [], []
        if prev_mlp_out is not None:
            off = lambda i, j: abs(i * mlp_to_head - j)
            this_w.append(connection_lines(W_Q, prev_mlp_out, left_pts, R['attn_in'], cfg, off, src_radius=r))
            this_g.append(connection_lines(W_Q_grad, prev_mlp_out, left_pts, R['attn_in_grad'], cfg, off, src_radius=r, grad=True))

        off = lambda i, j: abs(j * mlp_to_head - i)
        this_w.append(connection_lines(W_O, right_pts, p_in, R['attn_out'], cfg, off, dst_radius=r))
        this_g.append(connection_lines(W_O_grad, right_pts, p_in, R['attn_out_grad'], cfg, off, dst_radius=r, grad=True))

        # MLP wiring
        off = lambda i, j: abs(i - j)
        mlp_w = VGroup(
            connection_lines(snapshot[key + 'mlp.W_in'], p_in, p_hid, R['mlp_in'], cfg, off, r, r),
            connection_lines(snapshot[key + 'mlp.W_out'], p_hid, p_out, R['mlp_out'], cfg, off, r, r),
        )
        mlp_g = VGroup(
            connection_lines(snapshot[key + 'mlp.W_in.grad'], p_in, p_hid, R['mlp_in_grad'], cfg, off, r, r, grad=True),
            connection_lines(snapshot[key + 'mlp.W_out.grad'], p_hid, p_out, R['mlp_out_grad'], cfg, off, r, r, grad=True),
        )
        layer_weights.extend(this_w + [mlp_w])
        layer_grads.extend(this_g + [mlp_g])

        prev_mlp_out = p_out
        last_mlp_out = p_out
        last_mlp_right = mlp.get_right()[0]

    # ---- Input column + prompt text ----
    io_slots = V + 1
    prompt_tokens = snapshot['prompt.tokens'][:-1]  # last token is the answer
    if cfg.PROMPT_NEURONS is None:
        rng = np.random.RandomState(cfg.PROMPT_SEED)
        prompt_idx = rng.choice(np.arange(io_slots), len(prompt_tokens))  # same draw as the original scene
        if len(set(prompt_idx)) < len(prompt_tokens) or prompt_idx.max() >= V:
            # Collision (more likely at small widths) would drop a token -- draw distinct neurons instead.
            prompt_idx = rng.choice(np.arange(V), len(prompt_tokens), replace=False)
    else:
        prompt_idx = np.array(cfg.PROMPT_NEURONS)
    prompt_set = set(int(i) for i in prompt_idx)

    in_fills = [cfg.FORWARD_COLOR if (cfg.SHOW_FORWARD and k in prompt_set) else cfg.EMPTY_COLOR for k in range(V)]
    in_neurons, in_dots = neuron_column(V, 0, cfg, in_fills, ellipsis_scale=0.4)
    in_text = VGroup()
    if cfg.SHOW_TEXT:
        lit = sorted(k for k in prompt_set if k < V)  # tokens read top-to-bottom
        for token, k in zip(prompt_tokens, lit):
            y = in_neurons[k].get_center()[1]
            in_text.add(io_text(token, cfg.INPUT_FONT_SIZE, cfg.TEXT_COLOR, -1, y, cfg))
    input_layer = VGroup(in_neurons, in_dots, in_text)
    input_layer.shift(-in_neurons.get_center())
    input_layer.shift(RIGHT * (cfg.START_X - cfg.INPUT_GAP - in_neurons.get_right()[0]))

    # Embedding lines: rows are token neurons, columns are the residual dims that the shown
    # heads stand in for. Rows for the prompt's neurons use the prompt tokens' embeddings.
    rows = min(io_slots, snapshot['embed.W_E'].shape[1])
    W_E = snapshot['embed.W_E'][0, :rows][:, heads].copy()
    W_E_grad = snapshot['embed.W_E.grad'][0, :rows][:, heads].copy()
    prompt_E = snapshot['prompt.embed.W_E'][:, 0][:, heads]
    prompt_E_grad = snapshot['prompt.embed.W_E.grad'][:, 0][:, heads]
    for count, i in enumerate(prompt_idx):
        if i < rows:
            W_E[i] = prompt_E[count]
            W_E_grad[i] = prompt_E_grad[count]
    p_input = centers_of(in_neurons)
    off = lambda i, j: abs(j - i * io_to_head)
    embed_w = connection_lines(W_E, p_input, first_attn_left, R['embed'], cfg, off, src_radius=r)
    embed_g = connection_lines(W_E_grad, p_input, first_attn_left, R['embed_grad'], cfg, off, src_radius=r, grad=True)

    # ---- Output column + predicted tokens ----
    probs = np.asarray(snapshot['topk.probs'], dtype=float)
    p_max = probs.max()
    out_fills = [cmap_hex(cfg.fill_cmap, probs[k] / p_max) for k in range(V)] if cfg.SHOW_FORWARD else None
    out_neurons, out_dots = neuron_column(V, 0, cfg, out_fills, ellipsis_scale=0.4)
    out_text = VGroup()
    if cfg.SHOW_TEXT:
        for k in range(V):
            font_size = cfg.OUTPUT_FONT_SIZES[0 if k == 0 else (1 if k < 4 else 2)]
            if cfg.OUTPUT_TEXT_FADE:
                color = cmap_hex(cfg.text_cmap, np.clip(probs[k], cfg.OUTPUT_TEXT_FLOOR, 1.0) / p_max)
            else:
                color = cfg.TEXT_COLOR
            y = out_neurons[k].get_center()[1]
            out_text.add(io_text(snapshot['topk.tokens'][k], font_size, color, +1, y, cfg))
    output_layer = VGroup(out_neurons, out_dots, out_text)
    output_layer.shift(-out_neurons.get_center())
    output_layer.shift(RIGHT * (last_mlp_right + cfg.OUTPUT_GAP - out_neurons.get_left()[0]))

    W_U = snapshot['topk.unembed.W_U'][:, 0, :].T        # [resid, token]
    W_U_grad = snapshot['topk.unembed.W_U.grad'][:, 0, :].T
    p_output = centers_of(out_neurons)
    off = lambda i, j: abs(j - i)
    unembed_w = connection_lines(W_U, last_mlp_out, p_output, R['unembed'], cfg, off, r, r)
    unembed_g = connection_lines(W_U_grad, last_mlp_out, p_output, R['unembed_grad'], cfg, off, r, r, grad=True)

    # ---- Assemble, back to front ----
    background.add(in_dots, out_dots)
    if cfg.SHOW_WEIGHTS:
        weights.add(embed_w, *layer_weights, unembed_w)
    if cfg.SHOW_BACKWARD:
        grads.add(embed_g, *layer_grads, unembed_g)
    activations.add(in_neurons, out_neurons)
    text.add(in_text, out_text)

    return dict(background=background, weights=weights, grads=grads, activations=activations, text=text)


def scale_strokes(mobject, factor):
    if factor == 1.0:
        return
    for m in mobject.get_family():
        if isinstance(m, VMobject) and m.get_num_points() > 0:
            m.set_stroke(width=m.get_stroke_widths() * factor, recurse=False)


# =============================================================================
# Scenes
# =============================================================================
class LlamaRender(InteractiveScene):
    overrides = {}  # subclasses: any CONFIG keys to change

    def construct(self):
        cfg = resolve_config(self.overrides)
        self.camera.background_rgba = list(color_to_rgba(cfg.BACKGROUND_COLOR))

        snapshot = load_snapshot(cfg.SNAPSHOT_PATH, cfg.SNAPSHOT_INDEX)
        parts = build_network(snapshot, cfg)
        network = VGroup(*parts.values())  # dict order = drawing order

        # Camera
        if cfg.FRAME is None:
            aspect = self.frame.get_width() / self.frame.get_height()
            width = network.get_width() + 2 * cfg.FRAME_MARGIN
            height = network.get_height() + 2 * cfg.FRAME_MARGIN
            frame_height = max(height, width / aspect)
            center = network.get_center()
        else:
            frame_height = cfg.FRAME[2]
            center = np.array([cfg.FRAME[0], cfg.FRAME[1], 0.0])
        self.frame.reorient(0, 0, 0, tuple(center), frame_height)

        #SW quick hacky reorient for bleed spacing
        # self.frame.reorient(0, 0, 0, tuple(center), 1)

        stroke_scale = cfg.STROKE_SCALE
        if cfg.AUTO_STROKE_SCALE:
            stroke_scale *= _REF['frame_height'] / frame_height
        scale_strokes(network, stroke_scale)

        # Add back-to-front, each layer on its own z_index. ManimGL batches adjacent VGroups and
        # draws every fill in a batch before every stroke, which would let the connection lines
        # show through the neurons; distinct z_indices keep the layers in separate batches.
        layers = [parts['background'], parts['weights'], parts['grads']]
        layers += [parts['activations'].copy() for _ in range(cfg.OUTLINE_PASSES - 1)]
        layers += [parts['activations'], parts['text']]
        for z, part in enumerate(layers):
            part.set_z_index(z)
            self.add(part)
        self.parts = parts  # handy when poking around interactively
        self.wait()


# ---- Example variants -------------------------------------------------------

class LlamaRenderOriginalLook(LlamaRender):
    """Matches the finished frame of the original video scene (colors + framing)."""
    overrides = dict(
        NEURON_STROKE_COLOR='#dfd0b9',
        WEIGHT_COLOR='#948979',
        ELLIPSIS_COLOR='#dfd0b9',
        FRAME=(2.06, -0.06, 9.36),
    )


class LlamaRenderEmpty(LlamaRender):
    overrides = dict(SHOW_FORWARD=False, SHOW_BACKWARD=False, SHOW_TEXT=False)


class LlamaRenderForwardOnly(LlamaRender):
    overrides = dict(SHOW_BACKWARD=False)


class LlamaRenderFullDepth(LlamaRender):
    overrides = dict(DEPTH=16)


class LlamaRenderLight(LlamaRender):
    """Starting point for a light page."""
    overrides = dict(
        BACKGROUND_COLOR='#ffffff',
        LINE_COLOR='#948979',
        FORWARD_COLOR='#5c5346',
        TEXT_COLOR='#2b2620',
        GRADIENT_COLOR='#008b8b',
        GRAD_OPACITY=None,
    )
