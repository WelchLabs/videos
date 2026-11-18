import manimgx as gx
from kt_colors import KT_AQUA, KT_BLUE, KT_GREEN, KT_LT_BLUE, KT_ORANGE, KT_PINK, KT_PURPLE, KT_RED, KT_YELLOW
from welch_style import BLUE, CHILL_BLUE, CHILL_BROWN, CHILL_GREEN, CYAN, FRESH_TAN, GREEN, MAGENTA, YELLOW, YELLOW_FADE
from resnet_assets import ACTIVATION_ROOT
from resnet_network import build_fc, collect_tensors, fc_kernel_viz, fc_slot_centers, fc_stats, image_as_layer, image_plane, kernel_viz, kernels_for, layer_layout, load_act, relu_viz_block, square_outline
from welch_scene import WelchScene
from resnet_geometry import VoxelBlock, conv_data_block, layer_stats, orient, polyline, prism, viridis
import numpy as np
import matplotlib.pyplot as plt
import os


data_dir = str(ACTIVATION_ROOT) + '/'
act_dir=data_dir+'general_activations/'         #activations_{model_id}.npy from general_activation_saving_1
image_path=data_dir+'p25/screwdriver.jpg'       #every cache in act_dir is the screwdriver (idx 39209)

## ---- Geometry shared by every model (carried over from p25_35 / p13_30) ----
line_radius=0.18
cell_depth=0.1
pixel_dim=0.5
block_cell=0.48
base_depth=20.0                                  #world-unit depth of a 64-channel layer at depth_scale=1
fc_cell=1.35
fc_factor=1.2                                    #fc column height / depth of the last conv block
image_n=224
image_bounds=(-32.0, 32.0, -32.0, 32.0, 0.0, 3*pixel_dim)

still_hold=1.0

#Block counts per stage, from general_activation_saving_1 (for reference; the layout is read off the cache)
LAYER_CFG={8: [1,1,1,1], 14: [2,1,1,2], 20: [2,2,3,2], 26: [3,3,3,3],
           34: [4,4,4,4], 56: [3,4,17,3], 74: [3,4,26,3]}


## ---- Reading the activation cache into a list of drawable tensors ----


## ---- Kernel / filter visualization ----


## ---- Skip connections (resnets) ----

def rounded_skip(src, dst, height, corner_radius, color, radius,
                 head_length=1.5, head_width=1.5, n_arc_pts=12):
    """Skip as a rounded-corner path: up from the center of src's top face, along the network,
    down into the center of dst's top face, ending in a flat triangular arrowhead (un-oriented:
    +y is image-up, +z is network depth; orient() sends those to world +z, +x). Returns (line, head)."""
    src_top, dst_top=src['bounds'][3], dst['bounds'][3]
    src_zc=0.5*(src['z0']+src['z1'])
    dst_zc=0.5*(dst['z0']+dst['z1'])
    y_run=max(src_top, dst_top)+height
    y_end=dst_top+head_length                     #line stops where the arrowhead starts

    #A corner can't be longer than half of any leg it touches
    r=max(0.0, min(corner_radius, 0.5*(dst_zc-src_zc), 0.5*(y_run-src_top), 0.5*(y_run-y_end)))
    t=np.linspace(0.0, 0.5*np.pi, n_arc_pts)

    pts=[[0, src_top, src_zc]]                                                     #start, then up
    pts+=[[0, y_run-r+r*np.sin(a), src_zc+r-r*np.cos(a)] for a in t]             #corner: up -> along
    pts+=[[0, y_run-r+r*np.cos(a), dst_zc-r+r*np.sin(a)] for a in t]             #corner: along -> down
    pts+=[[0, y_end, dst_zc]]                                                      #down
    line=gx.VMobject()
    line.set_points_as_corners(np.array(pts, dtype=np.float64))
    line.set_stroke(color, width=200*radius, opacity=1.0)
    line.set_fill(opacity=0.0)

    head=gx.Polygon([0, dst_top, dst_zc],                                             #tip on the top face
                 [0, y_end, dst_zc+0.5*head_width],
                 [0, y_end, dst_zc-0.5*head_width])
    head.set_fill(color, opacity=1.0)
    head.set_stroke(width=0)
    return line, head


## ---- fc column (from p25_35, minus the probability axes) ----


## ---- The scene ----

class P47_Net_74(WelchScene):
    """Base class: draws one activation cache. Subclasses only override the attributes below."""

    model_id='plain74'

    #Per-model layout knobs
    depth_scale=0.14                   #multiplies base_depth; 1.0 reproduces p25_35's plain8 proportions
    layer_spacing=1.2                #world units between consecutive blocks, default = 5.0
    tensors_per_block=2               #2 -> post-conv1 relu + block output; 1 -> block outputs only
    depth_mults={64: 1.0, 128: 1.35, 256: 1.8, 512: 2.3}
    max_layers=None                   #debug: only draw the first N blocks

    #Activation thresholding
    act_pct=97
    act_alpha=0.4 #0.25

    #fc column
    show_fc=True
    fc_pct=70
    fc_vmax_div=2.0


    #Kernel viz: {destination layer index: dict(i, j, prism, ksize, color)} -- indices are printed at
    #startup by describe(). Index 0 is the stem, whose source is the input image (default ksize 7).
    kernels={}
             # 0: dict(i=18, j=80, prism=True, color=KT_ORANGE),           #image (7x7, stride 2) -> stem
             # 1: dict(i=12, j=40, prism=True, color=KT_ORANGE), #, color=CYAN),
             # 2: dict(i=8, j=40, prism=True, color=KT_ORANGE),
             # 3: dict(i=8, j=40, prism=True, color=KT_ORANGE),
             # 70: dict(i=5, j=7, prism=True, color=KT_AQUA),
             # 71: dict(i=5, j=7, prism=True, color=KT_AQUA),
             # 72: dict(i=3, j=3, prism=True, color=KT_AQUA),
             # # 7: dict(i=3, j=3, prism=True, color=KT_AQUA),

             # }

    kernel_color=KT_ORANGE
    kernel_ksize=3
    kernel_stem_ksize=7
    kernel_prism=False                #default for kernels that don't specify prism=
    kernel_radius=0.25

    #fc "kernel": patch on the last conv block's output face (prism optional) fanned out to the fc
    #column. dict(i, j, ksize, prism, color, radius); ksize='full' uses the whole face. None -> off.
    fc_kernel=dict(i=3, j=3, ksize=3, prism=True, color=KT_AQUA)


    #Skip connections (resnets): rounded path over the top of the net with an arrowhead into dst
    show_skips=True
    skip_color=gx.WHITE
    skip_downsample_color=gx.WHITE      #first block of layer2/3/4: 1x1 stride-2 conv on the skip path
    skip_height=6.0                  #horizontal run sits this far above the taller of the two blocks
    skip_radius=0.2                 #line width, same units as kernel_radius
    skip_corner_radius=1.5           #world units; clamped so it never exceeds half a leg
    skip_head_length=2.5
    skip_head_width=2.5
    skip_src_offset=-1               #endpoints relative to each block-output index k:
    skip_dst_offset=1                #  src=k-1 (this block's .relu1), dst=k+1 (next block's .relu1)
    skip_anim_time=6.0
    skip_lag_ratio=0.15              #0 -> all arrows draw at once, 1 -> strictly one after another

    #Presentation
    image_opacity=0.6
    fade_in=True
    fade_in_time=8.0
    default_view=(0, 66, 0, (np.float32(207.78), np.float32(44.54), np.float32(57.79)), 288.92)

    # ------------------------------------------------------------------

    def load(self):
        self.act=load_act(self.model_id)
        tensors=collect_tensors(self.act, self.tensors_per_block)
        if self.max_layers is not None:
            tensors=tensors[:self.max_layers]
        self.layers, self.fc_z=layer_layout(tensors, self.depth_scale, self.layer_spacing,
                                            self.depth_mults)
        self.total_z=self.fc_z if not self.show_fc else self.fc_z+fc_cell
        return self.layers

    def describe(self):
        print(f'\n{self.model_id}: {len(self.layers)} blocks, network spans z=0 -> {self.total_z:.1f}')
        print(f'{"idx":>4}  {"name":18s} {"shape":16s} {"z0":>8} {"z1":>8}  kernel')
        for L in self.layers:
            k=self.kernels.get(L['idx'])
            tag='' if k is None else f"(i={k.get('i')}, j={k.get('j')}, prism={k.get('prism', self.kernel_prism)})"
            print(f"{L['idx']:>4}  {L['name']:18s} {str((L['n_c'], L['n'], L['n'])):16s} "
                  f"{L['z0']:8.1f} {L['z1']:8.1f}  {tag}")
        print()

    def build_blocks(self):
        blocks, borders=[], []
        for L in self.layers:
            cz=min(cell_depth, 0.8*L['z_step'])      #keep voxels from overlapping when depth_scale is small
            blk, _=relu_viz_block(L['data'], L['z0'], L['z_step'], L['cell'],
                                  pct=self.act_pct, alpha=self.act_alpha, cell_z=cz)
            blocks.append(orient(blk))
            borders.append(orient(prism(*L['bounds'], CHILL_BROWN, line_radius)))
        return blocks, borders

    def build_fc_column(self):
        fc=self.act['fc'][0]
        n_fc=len(fc)
        fc_height=fc_factor*self.layers[-1]['depth']
        fc_step=fc_height/n_fc
        hp, hh=0.5*fc_cell, 0.5*fc_height
        self.fc_geom=dict(z=self.fc_z, hp=hp, hh=hh)
        col=orient(build_fc(fc, fc_step, self.fc_z, pct=self.fc_pct, vmax_div=self.fc_vmax_div,
                            alpha=self.act_alpha))
        border=orient(prism(-hp, hp, -hh, hh, self.fc_z-hp, self.fc_z+hp, CHILL_BROWN, line_radius))
        return col, border

    def build_fc_kernel(self):
        if self.fc_kernel is None or not self.show_fc:
            return []
        spec=self.fc_kernel
        src=self.layers[-1]
        g=fc_kernel_viz(src, self.fc_geom,
                        spec.get('i', src['n']//2), spec.get('j', src['n']//2),
                        spec.get('ksize', self.kernel_ksize),
                        spec.get('color', self.kernel_color), spec.get('radius', self.kernel_radius),
                        show_prism=spec.get('prism', True))
        return [orient(g)]

    def build_kernels(self):
        by_idx={L['idx']: L for L in self.layers}
        image=image_as_layer()
        out=[]
        for dst_idx, spec in sorted(self.kernels.items()):
            if dst_idx not in by_idx:
                continue
            dst=by_idx[dst_idx]
            src=image if dst_idx==0 else by_idx[dst_idx-1]
            ksize=spec.get('ksize', self.kernel_stem_ksize if dst_idx==0 else self.kernel_ksize)
            i=spec.get('i', dst['n']//2)
            j=spec.get('j', dst['n']//2)
            g=kernel_viz(dst, src, i, j, ksize,
                         spec.get('color', self.kernel_color),
                         spec.get('radius', self.kernel_radius),
                         show_prism=spec.get('prism', self.kernel_prism))
            out.append(orient(g))
        return out

    def build_skips(self):
        """One arrow per BasicBlock, endpoints offset from the block-output index (skip_*_offset)."""
        by_idx={L['idx']: L for L in self.layers}
        self.skip_lines, self.skip_heads=[], []
        for L in self.layers:
            if not L['is_block_out']:
                continue
            src=by_idx.get(L['idx']+self.skip_src_offset)
            dst=by_idx.get(L['idx']+self.skip_dst_offset)
            if src is None or dst is None:         #runs off either end of the net
                continue
            downsample=(L['block']==0 and L['stage']>1)
            color=self.skip_downsample_color if downsample else self.skip_color
            line, head=rounded_skip(src, dst, self.skip_height, self.skip_corner_radius, color,
                                    self.skip_radius, self.skip_head_length, self.skip_head_width)
            self.skip_lines.append(orient(line))
            self.skip_heads.append(orient(head))
        return [gx.VGroup(l, h) for l, h in zip(self.skip_lines, self.skip_heads)]

    def skip_animation(self):
        """Each arrow draws source -> destination, staggered left to right."""
        arrows=[gx.Succession(gx.Create(l, run_time=1.0), gx.FadeIn(h, run_time=0.25))
                for l, h in zip(self.skip_lines, self.skip_heads)]
        return gx.LaggedStart(*arrows, lag_ratio=self.skip_lag_ratio, run_time=self.skip_anim_time)
    def build(self):
        self.load()
        self.describe()
        self.img=orient(image_plane(image_path, opacity=self.image_opacity))
        self.image_border=orient(prism(*image_bounds, CHILL_BROWN, line_radius))
        self.blocks, self.borders=self.build_blocks()
        self.fc_block, self.fc_border=self.build_fc_column() if self.show_fc else (None, None)
        self.kernel_mobs=self.build_kernels()+self.build_fc_kernel()
        self.skip_mobs=self.build_skips() if self.show_skips else []
        self.net=gx.Group(*self.blocks)
        return self

    def view(self):
        if self.default_view is not None:
            return self.default_view
        L=self.total_z
        return (0, 57, 0, (0.5*L, 0.0, 0.0), max(60.0, 0.95*L))

    def construct(self):
        self.build()
        self.set_view(*self.view())
        self.add(self.img, self.image_border)

        pairs=list(zip(self.blocks, self.borders))
        if self.fc_block is not None:
            pairs.append((self.fc_block, self.fc_border))

        extras=self.kernel_mobs

        if self.fade_in:
            self.wait(1)
            fades=[gx.AnimationGroup(gx.FadeIn(b), gx.FadeIn(p)) for b, p in pairs]
            self.play(gx.LaggedStart(*fades, lag_ratio=1.0), run_time=self.fade_in_time)
            if extras:
                self.play(*[gx.FadeIn(m) for m in extras], run_time=1.5)
            if self.skip_mobs:
                self.play(self.skip_animation())
        else:
            for b, p in pairs:
                self.add(b, p)
            self.add(*extras, *self.skip_mobs)


        self.wait()


        self.wait(20)


