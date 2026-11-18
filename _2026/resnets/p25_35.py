import manimgx as gx
from welch_style import BLUE, CHILL_BLUE, CHILL_BROWN, CHILL_GREEN, CYAN, FRESH_TAN, GREEN, MAGENTA, YELLOW, YELLOW_FADE
from resnet_assets import ACTIVATION_ROOT
from resnet_network import fc_slot_centers, fc_stats, image_plane
from welch_scene import WelchScene
from resnet_geometry import VoxelBlock, cap_with_triangle, clear_scene, conv_data_block, layer_stats, orient, polyline, prism, swap_out, viridis
import numpy as np
import matplotlib.pyplot as plt


data_dir = str(ACTIVATION_ROOT) + '/'
sweep_dir=data_dir+'p27_sweep_1/cache/'
sweep_dir_2=data_dir+'p27_sweep_4/cache'
sweep_dir_3=data_dir+'p27_sweep_5/cache'

spacing_between_layers=5
line_radius=0.18
depth_step=0.125
cell_depth=0.1
pixel_dim=0.5
block_cell=0.48
still_hold=1.0

image_bounds=(-32.0, 32.0, -32.0, 32.0, 0.0, 3*pixel_dim)

#Layer depths, carried over from p13_30: layer 1 compressed to 64*squish_step=20 world units
z0=spacing_between_layers+1
base_depth=64*(10*depth_step/4)              #20
depth_mults={64: 1.0, 128: 1.35, 256: 1.8}
deep_keys=['relu', 'layer1.0.relu', 'layer1.0', 'layer2.0.relu', 'layer2.0',
           'layer3.0.relu', 'layer3.0']       #the 7 post-ReLU tensors of plain8

#fc column and probability plot
fc_cell=1.35                                  #was pooled_cell=14*0.48/5 in p13_30
fc_factor=1.2                                 #fc height / layer3.0 depth
fc_gap=4.0                                    #fc column -> index axis; one-sided plot, so this can be small
plot_w=11.0                                   #world units for prob_axis_max
prob_axis_max=1.0                             #fixed so sweep frames are comparable; drop toward max prob to zoom


def softmax(x):
    e=np.exp(x-x.max())
    return e/e.sum()


def relu_viz_block(r, z0, z_step, cell, pct=97, alpha=0.7, stats=None):
    """Per-channel normalized, per-channel percentile-thresholded, one merged block.
    Pass `stats` from a reference frame to hold the normalization fixed across a sweep;
    otherwise every frame renormalizes and always shows exactly (100-pct)% of each channel."""
    r=np.asarray(r, dtype=np.float64)
    vmax, thresh=layer_stats(r, pct) if stats is None else stats
    rn=r/vmax
    return conv_data_block(rn, z0, vmin=0.0, vmax=1.0, keep=(rn>thresh),
                           cell_size=cell, alpha=alpha, z_step=z_step)


## ---- Sweep helpers: geometry is fixed by the shapes, only the values change per file ----

import os, glob

def sweep_paths(d):
    return sorted(glob.glob(os.path.join(d, '*.npy')))   #zero-padded names sort in sweep order

def load_act(path):
    return np.load(path, allow_pickle=True).item()

# def load_act(sweep_id):
#     return np.load(f'{sweep_dir}{sweep_id:03d}.npy', allow_pickle=True).item()


def layer_layout(act):
    """(key, z_start, z_step) for each conv layer, plus the un-oriented depth of the fc column."""
    layout=[]
    z=z0
    for key in deep_keys:
        n_c=act[key].shape[1]
        depth=base_depth*depth_mults[n_c]
        layout.append((key, z, depth/n_c))
        z+=depth+spacing_between_layers
    return layout, z


def fc_geometry(act, fc_z):
    n_fc=act['fc'].shape[-1]
    fc_height=fc_factor*base_depth*depth_mults[256]     #relative to layer3.0 depth, as in p13_30
    fc_step=fc_height/n_fc
    return n_fc, fc_height, fc_step


def build_blocks(act, layout, stats=None):
    """The 7 thresholded conv blocks for one activation file, oriented."""
    blocks=[]
    for li, (key, z, z_step) in enumerate(layout):
        s=None if stats is None else stats[li]
        blk, _=relu_viz_block(act[key][0], z, z_step, block_cell, stats=s)
        blocks.append(orient(blk))
    return blocks


def build_fc(fc, fc_step, fc_z, pct=97, alpha=0.7, stats=None):
    """Column of wafers, max-normalized and percentile-thresholded exactly like relu_viz_block:
    only the top (100-pct)% of logits get a wafer, colored by viridis(logit/max)."""
    fc=np.asarray(fc, dtype=np.float64)
    vmax, thresh=fc_stats(fc, pct) if stats is None else stats

    vmax=vmax/2.0 #SW hack

    fcn=fc/float(vmax.ravel()[0])
    keep=fcn>float(thresh.ravel()[0])
    rgba=viridis(fcn[keep])
    rgba[:,3]=alpha
    wafer=np.array([fc_cell, cell_depth, fc_cell])   #thin slice per unit, stacked in y
    return orient(VoxelBlock(fc_slot_centers(len(fc), fc_step, fc_z)[keep], wafer, rgba))


def build_prob_axes(n_fc, fc_step, fc_z):
    """Axes with index pointing down and a one-sided probability axis pointing right.
    Returns (axes, tips, prob_unit); c2p(0, 0) sits at the top of the index axis."""
    prob_unit=plot_w/prob_axis_max
    y_max=1.05*n_fc

    axes=gx.Axes(x_range=(0, prob_axis_max, prob_axis_max), y_range=(0, y_max, y_max),
              x_length=prob_axis_max*prob_unit, y_length=y_max*fc_step,
              axis_config=dict(stroke_width=8, include_ticks=False, include_tip=False))
    axes.set_color(CHILL_BROWN)

    axes.rotate(-90*gx.DEGREES, gx.RIGHT, about_point=gx.ORIGIN)   #+y -> -z: index 0 at the top
    axis_x=fc_z+fc_gap
    top_z=0.5*(n_fc-1)*fc_step
    axes.shift(np.array([axis_x, 0, top_z])-axes.c2p(0, 0))

    o=axes.c2p(0, 0)
    axes.stretch(1.02, 2, about_point=o)   #dim 2 = world z after the rotate; lines the plot up with the wafers

    tips=gx.VGroup(cap_with_triangle(axes.x_axis, length=1.0, width=0.85),   #positive end of the prob axis
                cap_with_triangle(axes.y_axis, length=1.0, width=0.85))   #bottom of the index axis
    return axes, tips, prob_unit


def build_prob_curve(axes, probs, prob_unit, color_max):
    n_fc=len(probs)
    curve=gx.VMobject()
    curve.set_points_as_corners(axes.c2p(probs, np.arange(n_fc)))
    curve.set_stroke(width=6, opacity=1.0)
    o=axes.c2p(0, 0)
    pts=curve.get_points()
    v=(pts[:,0]-o[0])/prob_unit
    rgba=viridis(v/color_max)
    rgba[:,3]=1.0
    curve.paint = curve.paint.but(stroke=rgba)
    return curve


def run_sweep(scene, paths, net, layout, fc_step, fc_z, stats, fc_stat, hold, quick_mode,
              fc_vmax_per_frame=False):
    if quick_mode: paths=paths[-1:]
    for p in paths:
        act=load_act(p)
        fc=act['fc'][0]
        fs=fc_stat
        if fc_vmax_per_frame and fc_stat is not None:
            fs=(np.array([[[fc.max()]]]), fc_stat[1])   #current max, frame-63 threshold
        new=gx.Group(*build_blocks(act, layout, stats),
                  build_fc(fc, fc_step, fc_z, stats=fs))
        swap_out(scene, net)
        net=new
        scene.add(net)
        scene.wait(hold)
    scene.wait(still_hold)
    return net

class P25(WelchScene):
    def construct(self):

        quick_mode=False

        start_id=63
        sweep_ids=list(range(0, 128, 1))     #0 -> 128; thin this out or reorder as you like
        sweep_hold=0.25                       #seconds per sweep frame
        fixed_norm=True                       #normalize/threshold every frame with start_id's stats
        side_view=(0, 90, 0, (115.0, 0.0, 0.0), 150.0)   #tune in embed

        # act=load_act(start_id)
        act=load_act(os.path.join(sweep_dir, f'{start_id:03d}.npy'))
        layout, fc_z=layer_layout(act)
        n_fc, fc_height, fc_step=fc_geometry(act, fc_z)

        #Stats from the starting frame, reused for every sweep frame when fixed_norm
        stats=[layer_stats(act[key][0]) for key, _, _ in layout] if fixed_norm else None
        fc_stat=fc_stats(act['fc'][0]) if fixed_norm else None

        def color_max(probs):
            return prob_axis_max if fixed_norm else float(probs.max())

        ## ---- Static geometry: image, one border per layer, fc border, axes ----
        img=orient(image_plane(data_dir+'/p25/screwdriver.jpg', opacity=0.6))   #edge-on from a pure side view
        image_border=orient(prism(*image_bounds, CHILL_BROWN, line_radius))

        borders=[]
        for key, z, z_step in layout:
            _, bnds=conv_data_block(act[key][0], z, cell_size=block_cell, z_step=z_step)
            borders.append(orient(prism(*bnds, CHILL_BROWN, line_radius)))

        hp=0.5*fc_cell
        hh=0.5*fc_height
        fc_border=orient(prism(-hp, hp, -hh, hh, fc_z-hp, fc_z+hp, CHILL_BROWN, line_radius))

        axes, tips, prob_unit=build_prob_axes(n_fc, fc_step, fc_z)

        ## ---- First frame ----
        probs=softmax(act['fc'][0])
        blocks=build_blocks(act, layout, stats)
        fc_block=build_fc(act['fc'][0], fc_step, fc_z, stats=fc_stat)
        curve=build_prob_curve(axes, probs, prob_unit, color_max(probs))

        # self.set_view(*side_view)
        # self.set_view(38, 60, 0, (np.float32(151.06), np.float32(16.83), np.float32(-21.6)), 162.19)
        self.set_view(2, 57, 0, (np.float32(105.69), np.float32(16.14), np.float32(-11.07)), 179.39)
        self.wait(1)
        self.add(img, image_border)
        

        fades=[gx.AnimationGroup(gx.FadeIn(b), gx.FadeIn(p)) for b, p in zip(blocks, borders)]
        fades.append(gx.AnimationGroup(gx.FadeIn(fc_block), gx.FadeIn(fc_border)))

        self.wait(1)
        self.play(gx.LaggedStart(*fades, lag_ratio=1.0), run_time=6.0)
        self.wait(still_hold)

        # self.play(ShowCreation(axes), FadeIn(tips), run_time=2.0)
        # self.play(ShowCreation(curve), run_time=3.0)
        # self.wait(still_hold)

        ## ---- Sweep: rebuild only what depends on the file ----

        # Ok, so I'm not sure yet about what view angle will be nice for blending everything together. 
        # Maybe try a couple here. 
        # self.set_view(42, 79, 0, (np.float32(130.97), np.float32(9.46), np.float32(-7.34)), 175.86) #Option 1.
        # self.set_view(32, 65, 0, (np.float32(135.4), np.float32(11.45), np.float32(-9.87)), 175.86) #Option 2
        # self.set_view(38, 60, 0, (np.float32(151.06), np.float32(16.83), np.float32(-21.6)), 162.19) #Ok getting some traction with this one in p26 in illustrator!
        # self.set_view(33, 90, 0, (np.float32(137.31), np.float32(67.91), np.float32(-0.33)), 201.08)

        net=gx.Group(*blocks, fc_block)
        
        self.wait(1)
        # self.play(self.animate_view(38, 60, 0, (np.float32(151.06), np.float32(16.83), np.float32(-21.6)), 162.19), run_time=6.0)
        self.play(self.animate_view(35, 61, 0, (np.float32(172.47), np.float32(32.64), np.float32(-36.48)), 203.29), run_time=6.0)

        self.wait(1)
        self.remove(blocks[0], blocks[1], blocks[2]) #Remove blocks to see picture for a moment
        self.wait(1)
        self.add(blocks[0], blocks[1], blocks[2])

        self.wait(1)

        # Quick still grab here for print outs for P41ish
        # FRESH_TAN=#EFE5D1
        # self.remove(blocks[0], blocks[1], blocks[2], blocks[4], blocks[5])
        # self.remove(borders[0], borders[1], borders[2], borders[4], borders[5])
        # # self.set_view(0, 50, 0, (np.float32(94.77), np.float32(33.95), np.float32(-29.24)), 77.79)
        # self.set_view(0, 65, 0, (np.float32(94.53), np.float32(36.94), np.float32(-22.72)), 77.79)

        # self.remove(blocks[0], blocks[1], blocks[2], blocks[3], blocks[5])
        # self.remove(borders[0], borders[1], borders[2], borders[3], borders[5])
        # # reorient(0, 48, 0, (np.float32(127.61), np.float32(33.57), np.float32(-29.77)), 77.79)
        # self.set_view(0, 64, 0, (np.float32(126.36), np.float32(36.57), np.float32(-23.5)), 77.79)


        self.wait(1)
        net=run_sweep(self, sweep_paths(sweep_dir), net, layout, fc_step, fc_z, stats, fc_stat,
                      sweep_hold, quick_mode, fc_vmax_per_frame=True)
        self.play(self.animate_view(0, 50, 0, (np.float32(156.33), np.float32(16.44), np.float32(-30.65)), 247.62), run_time=8)
        self.wait(1)

        net=run_sweep(self, sweep_paths(sweep_dir_2), net, layout, fc_step, fc_z, stats, fc_stat, sweep_hold, quick_mode)   #P27
        self.wait(1)
        net=run_sweep(self, sweep_paths(sweep_dir_3), net, layout, fc_step, fc_z, stats, fc_stat, sweep_hold, quick_mode)   #P28

        self.wait(20)
        # self.set_view(1, 53, 0, (np.float32(144.27), np.float32(22.04), np.float32(-23.48)), 234.65)
        # self.set_view(0, 50, 0, (np.float32(150.35), np.float32(19.08), np.float32(-27.47)), 238.02)
        # self.set_view(0, 50, 0, (np.float32(156.33), np.float32(16.44), np.float32(-30.65)), 247.62)

        # Ok, to finish this scene, we need to do two more sweeps from this new FOV
        # Same sweep approach as before
        # P27 - sweep_dir_2


        # P28 - sweep_dir_3


        self.wait(20)
        # conv1 (1, 64, 112, 112)
        # bn1 (1, 64, 112, 112)
        # relu (1, 64, 112, 112)
        # maxpool (1, 64, 56, 56)
        # layer1.0.conv1 (1, 64, 56, 56)
        # layer1.0.bn1 (1, 64, 56, 56)
        # layer1.0.relu (1, 64, 56, 56)
        # layer1.0.conv2 (1, 64, 56, 56)
        # layer1.0.bn2 (1, 64, 56, 56)
        # layer1.0 (1, 64, 56, 56)
        # layer2.0.conv1 (1, 128, 28, 28)
        # layer2.0.bn1 (1, 128, 28, 28)
        # layer2.0.relu (1, 128, 28, 28)
        # layer2.0.conv2 (1, 128, 28, 28)
        # layer2.0.bn2 (1, 128, 28, 28)
        # layer2.0 (1, 128, 28, 28)
        # layer3.0.conv1 (1, 256, 14, 14)
        # layer3.0.bn1 (1, 256, 14, 14)
        # layer3.0.relu (1, 256, 14, 14)
        # layer3.0.conv2 (1, 256, 14, 14)
        # layer3.0.bn2 (1, 256, 14, 14)
        # layer3.0 (1, 256, 14, 14)
        # avgpool (1, 256, 1, 1)
        # fc (1, 1000)
        # image (224, 224, 3)