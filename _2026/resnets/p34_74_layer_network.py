import manimgx as gx
from kt_colors import KT_AQUA, KT_BLUE, KT_GREEN, KT_LT_BLUE, KT_ORANGE, KT_PINK, KT_PURPLE, KT_RED, KT_YELLOW
from welch_style import BLUE, CHILL_BLUE, CHILL_BROWN, CHILL_GREEN, CYAN, FRESH_TAN, GREEN, MAGENTA, YELLOW, YELLOW_FADE
from resnet_assets import ACTIVATION_ROOT
from resnet_network import build_fc, collect_tensors, fc_kernel_viz, fc_slot_centers, fc_stats, image_as_layer, image_plane, kernel_viz, kernels_for, layer_layout, load_act, relu_viz_block, skip_arc, square_outline
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


## ---- fc column (from p25_35, minus the probability axes) ----


## ---- The scene ----

class P34_Net_74(WelchScene):
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
    kernels={0: dict(i=18, j=80, prism=True, color=KT_ORANGE),           #image (7x7, stride 2) -> stem
             1: dict(i=12, j=40, prism=True, color=KT_ORANGE), #, color=CYAN),
             2: dict(i=8, j=40, prism=True, color=KT_ORANGE),
             3: dict(i=8, j=40, prism=True, color=KT_ORANGE),
             7: dict(i=5, j=7, prism=True, color=KT_AQUA),
             71: dict(i=5, j=7, prism=True, color=KT_AQUA),
             72: dict(i=3, j=3, prism=True, color=KT_AQUA),
             # 7: dict(i=3, j=3, prism=True, color=KT_AQUA),

             }

    kernel_color=KT_ORANGE
    kernel_ksize=3
    kernel_stem_ksize=7
    kernel_prism=False                #default for kernels that don't specify prism=
    kernel_radius=0.25

    #fc "kernel": patch on the last conv block's output face (prism optional) fanned out to the fc
    #column. dict(i, j, ksize, prism, color, radius); ksize='full' uses the whole face. None -> off.
    fc_kernel=dict(i=3, j=3, ksize=3, prism=True, color=KT_AQUA)

    #Skip connections (resnets): arc from block input to block output over the top of the net
    show_skips=False
    skip_color=BLUE
    skip_downsample_color=YELLOW      #first block of layer2/3/4: 1x1 stride-2 conv on the skip path
    skip_height=6.0
    skip_radius=0.12

    #Presentation
    image_opacity=0.6
    fade_in=True
    fade_in_time=8.0
    default_view=(0, 59, 0, (np.float32(208.58), np.float32(44.2), np.float32(57.06)), 288.92)

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
        """One arc per BasicBlock, from the block's input tensor to its output tensor."""
        out=[]
        step=self.tensors_per_block
        for L in self.layers:
            if not L['is_block_out']:
                continue
            src_idx=L['idx']-step
            if src_idx<0:
                continue
            src=self.layers[src_idx]
            downsample=(L['block']==0 and L['stage']>1)
            color=self.skip_downsample_color if downsample else self.skip_color
            out.append(orient(skip_arc(src, L, self.skip_height, color, self.skip_radius)))
        return out

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
        extras=self.kernel_mobs+self.skip_mobs

        if self.fade_in:
            self.wait(1)
            fades=[gx.AnimationGroup(gx.FadeIn(b), gx.FadeIn(p)) for b, p in pairs]
            self.play(gx.LaggedStart(*fades, lag_ratio=1.0), run_time=self.fade_in_time)
            if extras:
                self.play(*[gx.FadeIn(m) for m in extras], run_time=1.5)
        else:
            for b, p in pairs:
                self.add(b, p)
            self.add(*extras)

        self.wait(still_hold)

# class Plain8(GeneralNet):
#     model_id='plain8'
#     depth_scale=1.0
#     layer_spacing=5.0
#     kernels={0: dict(i=40, j=70, prism=False),           #image (7x7, stride 2) -> stem
#              1: dict(i=22, j=30, prism=True),
#              3: dict(i=9, j=16, prism=False),
#              5: dict(i=5, j=7, prism=True)}
#     fc_kernel=dict(i=5, j=7, ksize=3, prism=True)      #or ksize='full' for the whole 14x14 face
#     default_view=(2, 57, 0, (105.69, 16.14, -11.07), 179.39)   #from p25_35


# class Plain14(GeneralNet):
#     model_id='plain14'
#     depth_scale=0.9
#     layer_spacing=4.0
#     kernels=kernels_for([1, 5, 9], i=10, j=14, prism=True)


# class Plain20(GeneralNet):
#     model_id='plain20'
#     depth_scale=0.7
#     layer_spacing=3.5
#     kernels=kernels_for([1, 7, 13], i=10, j=14, prism=True)


# class Plain26(GeneralNet):
#     model_id='plain26'
#     depth_scale=0.55
#     layer_spacing=3.0
#     kernels=kernels_for([1, 7, 13, 19], i=10, j=14, prism=True)


# class Plain34(GeneralNet):
#     model_id='plain34'
#     depth_scale=0.45
#     layer_spacing=2.5
#     kernels=kernels_for([1, 9, 17, 25], i=10, j=14, prism=True)


# class Plain56(GeneralNet):
#     model_id='plain56'
#     depth_scale=0.3
#     layer_spacing=2.0
#     kernels=kernels_for([1, 7, 15, 49], i=10, j=14, prism=False)


# class Plain74(GeneralNet):
#     model_id='plain74'
#     depth_scale=0.25
#     layer_spacing=1.5
#     kernels=kernels_for([1, 7, 15, 67], i=10, j=14, prism=False)


# class ResNet74(Plain74):
#     model_id='resnet74'
#     show_skips=True
#     skip_height=6.0