"""Common activation layout, connectivity, and classifier geometry."""

import manimgx as gx
import numpy as np

from resnet_assets import ACTIVATION_ROOT
from resnet_geometry import VoxelBlock, box_image, conv_data_block, layer_stats, polyline, prism, viridis

base_depth = 20.0
block_cell = 0.48
cell_depth = 0.1
fc_cell = 1.35
image_n = 224
image_bounds = (-32.0, 32.0, -32.0, 32.0, 0.0, 1.5)


def load_act(model_id):
    path = ACTIVATION_ROOT / "general_activations" / f"activations_{model_id}.npy"
    return np.load(path, allow_pickle=True).item()


def image_plane(im_path, opacity):
    return box_image(im_path, opacity, image_bounds)


def square_outline(x0, x1, y0, y1, z, color, radius):
    """Axis-aligned square on the plane z=const."""
    corners=[(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z), (x0, y0, z)]
    return polyline(corners, color, radius)


def relu_viz_block(r, z0, z_step, cell, pct=97, alpha=0.7, cell_z=cell_depth, stats=None):
    """Per-channel normalized, per-channel percentile-thresholded, one merged block."""
    r=np.asarray(r, dtype=np.float64)
    vmax, thresh=layer_stats(r, pct) if stats is None else stats
    rn=r/vmax
    return conv_data_block(rn, z0, vmin=0.0, vmax=1.0, keep=(rn>thresh),
                           cell_size=cell, alpha=alpha, z_step=z_step, cell_z=cell_z)


def collect_tensors(act, tensors_per_block=2):
    """(name, (C, H, W) array, stage, block) for every activation block we draw, in forward order.

    The stem 'relu' is drawn directly. Inside each BasicBlock the shared self.relu module is
    hooked twice and the cache only keeps the *second* call, so the post-conv1 activation is
    reconstructed as max(bn1, 0). The block output ('layerS.B') is the post-conv2 activation
    (post-skip-add for resnets). tensors_per_block=1 keeps only the block outputs."""
    tensors=[('relu', act['relu'][0], 0, 0)]
    for s in range(1, 5):
        b=0
        while f'layer{s}.{b}' in act:                    #plain8 has no layer4 (nn.Identity, never hooked)
            blk=f'layer{s}.{b}'
            if tensors_per_block==2:
                tensors.append((blk+'.relu1', np.maximum(act[blk+'.bn1'][0], 0.0), s, b))
            tensors.append((blk, act[blk][0], s, b))
            b+=1
    return tensors


def layer_layout(tensors, depth_scale, spacing, depth_mults, cell=block_cell, z_start=None):
    """One dict per drawn tensor with everything the geometry needs, plus the z of the fc column."""
    layers=[]
    z=image_bounds[5]+spacing if z_start is None else z_start
    for idx, (name, a, s, b) in enumerate(tensors):
        n_c, n, _=a.shape
        depth=base_depth*depth_scale*depth_mults[n_c]
        half=np.floor(n/2)
        he=(half+0.5)*cell
        layers.append(dict(idx=idx, name=name, data=a, stage=s, block=b,
                           is_block_out=(name!='relu' and not name.endswith('.relu1')),   #'layerS.B'
                           n_c=n_c, n=n, cell=cell, depth=depth, z_step=depth/n_c,
                           z0=z, z1=z+depth,
                           bounds=(-he, he, -he, he, z, z+depth)))
        z+=depth+spacing
    return layers, z


def image_as_layer():
    """The input image as a pseudo source layer, so a kernel can be drawn from image -> stem."""
    return dict(idx=-1, name='image', n=image_n, cell=(image_bounds[1]-image_bounds[0])/image_n,
                z0=image_bounds[4], z1=image_bounds[5], bounds=image_bounds)


def kernel_viz(dst, src, i, j, ksize, color, radius, show_prism=False):
    """One conv kernel: output cell (i, j) on the *input face* of `dst` wired to its
    ksize x ksize patch on the *output face* of `src`. Optionally a prism extends that patch
    back through the full channel depth of `src`, showing what the conv is reading."""
    cell=dst['cell']
    hc=0.5*cell
    half_dst=np.floor(dst['n']/2)
    dst_x=(j-half_dst)*cell
    dst_y=(-i+half_dst)*cell
    dst_z=dst['z0']

    #Stride maps output (i, j) to source cell (stride*i, stride*j): 1 within a stage, 2 across
    stride=src['n']/dst['n']
    half_src=np.floor(src['n']/2)
    sc=src['cell']
    src_cx=(stride*j-half_src)*sc
    src_cy=(-stride*i+half_src)*sc
    r=0.5*ksize*sc                                  #patch is ksize *source* cells wide
    x0, x1, y0, y1=src_cx-r, src_cx+r, src_cy-r, src_cy+r
    z_face=src['z1']

    dst_corners=[(dst_x-hc, dst_y-hc, dst_z), (dst_x+hc, dst_y-hc, dst_z),
                 (dst_x+hc, dst_y+hc, dst_z), (dst_x-hc, dst_y+hc, dst_z)]
    src_corners=[(x0, y0, z_face), (x1, y0, z_face), (x1, y1, z_face), (x0, y1, z_face)]

    group=gx.VGroup()
    for p, q in zip(dst_corners, src_corners):
        group.add(polyline([p, q], color, radius))                       #4 connectors
    for k in range(4):
        group.add(polyline([dst_corners[k], dst_corners[(k+1)%4]], color, radius))   #output cell
    if show_prism:
        group.add(prism(x0, x1, y0, y1, src['z0'], src['z1'], color, radius))  #includes the patch face
    else:
        group.add(square_outline(x0, x1, y0, y1, z_face, color, radius))
    return group


def kernels_for(indices, **kw):
    """Same kernel spec on several destination layers: kernels_for([1, 3, 5], i=10, j=12, prism=True)."""
    return {int(k): dict(kw) for k in indices}


def fc_kernel_viz(src, fc_geom, i, j, ksize, color, radius, show_prism=True):
    """avgpool + fc: a patch on the output face of the last conv block (optionally a prism through
    its full channel depth), with its 4 corners fanned out to the top/bottom of the fc column's
    near face. ksize='full' spans the whole output face, which is what avgpool actually reads."""
    sc=src['cell']
    if ksize=='full':
        x0, x1, y0, y1=src['bounds'][:4]
    else:
        half=np.floor(src['n']/2)
        cx=(j-half)*sc
        cy=(-i+half)*sc
        r=0.5*ksize*sc
        x0, x1, y0, y1=cx-r, cx+r, cy-r, cy+r
    z_face=src['z1']
    fz, hp, hh=fc_geom['z']-fc_geom['hp'], fc_geom['hp'], fc_geom['hh']   #near face of the column

    group=gx.VGroup()
    if show_prism:
        group.add(prism(x0, x1, y0, y1, src['z0'], src['z1'], color, radius))
    else:
        group.add(square_outline(x0, x1, y0, y1, z_face, color, radius))
    connectors=[[(x0, y0, z_face), (-hp, -hh, fz)],       #bottom corners -> bottom of the column
                [(x1, y0, z_face), ( hp, -hh, fz)],
                [(x0, y1, z_face), (-hp,  hh, fz)],       #top corners -> top of the column
                [(x1, y1, z_face), ( hp,  hh, fz)]]
    for cc in connectors:
        group.add(polyline(cc, color, radius))
    return group


def skip_arc(src, dst, height, color, radius, n_pts=32):
    """Arc over the top of the network from src's output face to dst's input face (un-oriented
    +y is image-up, which orient() sends to world +z)."""
    p0=np.array([0.0, src['bounds'][3], src['z1']])
    p1=np.array([0.0, dst['bounds'][3], dst['z0']])
    t=np.linspace(0.0, 1.0, n_pts)
    pts=p0[None,:]+(p1-p0)[None,:]*t[:,None]
    pts[:,1]+=height*np.sin(np.pi*t)
    curve=gx.VMobject()
    curve.set_points_as_corners(pts)
    curve.set_stroke(color, width=200*radius, opacity=1.0)
    curve.set_fill(opacity=0.0)
    return curve


def fc_stats(fc, pct=70):
    """The fc vector as a single (1, n_fc, 1) channel, so layer_stats treats it like a conv layer."""
    return layer_stats(np.asarray(fc, dtype=np.float64).reshape(1, -1, 1), pct)


def fc_slot_centers(n_fc, fc_step, fc_z):
    idx=np.arange(n_fc)
    y=(0.5*(n_fc-1)-idx)*fc_step       #index 0 at top
    return np.stack([np.zeros(n_fc), y, np.full(n_fc, fc_z)], axis=-1)


def build_fc(fc, fc_step, fc_z, pct=70, vmax_div=2.0, alpha=0.7):
    """Column of wafers, max-normalized and percentile-thresholded like relu_viz_block:
    only the top (100-pct)% of logits get a wafer, colored by viridis(logit/(max/vmax_div))."""
    fc=np.asarray(fc, dtype=np.float64)
    vmax, thresh=fc_stats(fc, pct)
    vmax=vmax/vmax_div                 #SW hack from p25_35: pushes the top logits into the bright end
    fcn=fc/float(vmax.ravel()[0])
    keep=fcn>float(thresh.ravel()[0])
    rgba=viridis(fcn[keep])
    rgba[:,3]=alpha
    wafer=np.array([fc_cell, cell_depth, fc_cell])   #thin slice per unit, stacked in y
    return VoxelBlock(fc_slot_centers(len(fc), fc_step, fc_z)[keep], wafer, rgba)
