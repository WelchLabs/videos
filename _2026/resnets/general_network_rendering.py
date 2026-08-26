from manimlib import *
import numpy as np
import matplotlib.pyplot as plt
import os

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b'
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
MAGENTA='#FF00FF'

KT_PINK='#ED5E78'
KT_RED='#D73B2F'
KT_ORANGE='#EB8423'
KT_YELLOW='#FCC947'
KT_GREEN='#419C52'
KT_LT_BLUE='#B2C2E4'
KT_AQUA='#5BADB6'
KT_BLUE='#236C94'
KT_PURPLE='#7E5B76'

data_dir='/Volumes/hot_1/Stephencwelch Dropbox/welch_labs/resnet/hackin/'
# data_dir='/Users/stephen/Library/CloudStorage/Dropbox-Stephencwelch/welch_labs/resnet/hackin/'
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


viridis_lut=plt.get_cmap('viridis')(np.linspace(0, 1, 256))


def viridis(values):
    x=np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    idx=np.clip((x*256).astype(np.int32), 0, 255) #matplotlib bins with int(x*N), not round(x*(N-1))
    return viridis_lut[idx]


# 6 faces x 4 corners of a unit cube, wound counter-clockwise seen from outside
face_corners=np.array([
    [[-.5, -.5,  .5], [.5, -.5,  .5], [.5,  .5,  .5], [-.5,  .5,  .5]],
    [[ .5, -.5, -.5], [-.5, -.5, -.5], [-.5, .5, -.5], [ .5,  .5, -.5]],
    [[ .5, -.5,  .5], [.5, -.5, -.5], [.5,  .5, -.5], [ .5,  .5,  .5]],
    [[-.5, -.5, -.5], [-.5, -.5, .5], [-.5, .5,  .5], [-.5,  .5, -.5]],
    [[-.5,  .5,  .5], [.5,  .5,  .5], [.5,  .5, -.5], [-.5,  .5, -.5]],
    [[-.5, -.5, -.5], [.5, -.5, -.5], [.5, -.5,  .5], [-.5, -.5,  .5]],
], dtype=np.float64)

quad_tris=np.array([0, 1, 2, 0, 2, 3], dtype=int)
box_edges=[(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
           (0,4), (1,5), (2,6), (3,7)]


class VoxelBlock(Surface):
    """Every box in a layer batched into one mesh."""
    shader_folder='surface'
    render_primitive=moderngl.TRIANGLES

    def __init__(self, centers, sizes, rgba, view_forward=None, depth_test=None, **kwargs):
        centers=np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        rgba=np.asarray(rgba, dtype=np.float64).reshape(-1, 4)
        sizes=np.asarray(sizes, dtype=np.float64)
        if sizes.ndim==1:
            sizes=np.tile(sizes.reshape(1, 3), (len(centers), 1))

        # Painter's algorithm: with the depth test on, the nearest translucent cell writes
        # depth and hides the rest, flattening a 64 layer block to one face
        if view_forward is not None and len(centers):
            f=np.asarray(view_forward, dtype=np.float64)
            order=np.argsort(-(centers@(f/np.linalg.norm(f)))) #farthest first
            centers, sizes, rgba=centers[order], sizes[order], rgba[order]
        if depth_test is None:
            depth_test=view_forward is None

        self.centers=centers
        self.sizes=sizes
        self.rgba=rgba
        super().__init__(shading=(0, 0, 0), depth_test=depth_test, **kwargs) #Faces read flat

    @Surface.affects_data
    def init_points(self):
        n=len(self.centers)
        if n==0:
            self.set_points(np.zeros((0, 3)))
            self.tri_indices=np.zeros(0, dtype=int)
            return

        corners=face_corners[None,:,:,:]*self.sizes[:,None,None,:]+self.centers[:,None,None,:]
        verts=corners.reshape(-1, 3)
        self.set_points(verts)

        # The shader takes normals from (du_point - point) x (dv_point - point)
        c0=corners[:,:,0,:]
        du=np.repeat((corners[:,:,1,:]-c0)[:,:,None,:], 4, axis=2)
        dv=np.repeat((corners[:,:,3,:]-c0)[:,:,None,:], 4, axis=2)
        self.data['du_point'][:]=verts+du.reshape(-1, 3)
        self.data['dv_point'][:]=verts+dv.reshape(-1, 3)

        self.tri_indices=(np.arange(n*6)*4).repeat(6)+np.tile(quad_tris, n*6)

    def compute_triangle_indices(self):
        self.triangle_indices=self.tri_indices
        return self.triangle_indices

    def init_colors(self):
        if len(self.centers):
            self.data['rgba'][:]=np.repeat(self.rgba, 24, axis=0)


def conv_data_block(a, start_depth, vmin=None, vmax=None, keep=None, cell_size=1.0,
                    alpha=0.5, z_step=0.125, cell_z=cell_depth, view_forward=None):
    a=np.asarray(a, dtype=np.float64)
    n_c, n_i, n_j=a.shape
    kk, ii, jj=np.meshgrid(np.arange(n_c), np.arange(n_i), np.arange(n_j), indexing='ij')

    vmin=a.min() if vmin is None else vmin
    vmax=a.max() if vmax is None else vmax
    vals=(a-vmin)/(vmax-vmin)          #imshow's default min-max stretch
    keep=np.ones(a.shape, dtype=bool) if keep is None else keep

    half=np.floor(n_j/2)
    centers=np.stack([(jj[keep]-half)*cell_size, (-ii[keep]+half)*cell_size,
                      z_step*kk[keep]+start_depth], axis=-1)
    rgba=viridis(vals[keep])
    rgba[:,3]=alpha

    block=VoxelBlock(centers, np.array([cell_size, cell_size, cell_z]), rgba,
                     view_forward=view_forward)
    half_extent=(half+0.5)*cell_size
    bounds=(-half_extent, half_extent, -half_extent, half_extent,
            start_depth, n_c*z_step+start_depth)
    return block, bounds


def polyline(points, color, radius):
    """One independent segment per step, since manim miters the joins of a continuous path."""
    pts=np.asarray(points, dtype=np.float64)
    group=VGroup()
    for start, end in zip(pts[:-1], pts[1:]):
        if np.allclose(start, end):
            continue
        seg=Line(start, end)
        seg.set_stroke(color, width=200*radius, opacity=1.0) #Width in world units
        seg.set_fill(opacity=0.0)
        seg.set_scale_stroke_with_zoom(True)
        seg.apply_depth_test()
        group.add(seg)
    return group


def prism(min_x, max_x, min_y, max_y, min_z, max_z, color, radius):
    vertices=np.array([
        [min_x, min_y, min_z], [max_x, min_y, min_z],
        [max_x, max_y, min_z], [min_x, max_y, min_z],
        [min_x, min_y, max_z], [max_x, min_y, max_z],
        [max_x, max_y, max_z], [min_x, max_y, max_z],
    ], dtype=np.float64)
    group=VGroup()
    for p, q in box_edges:
        group.add(*polyline([vertices[p], vertices[q]], color, radius))
    return group


def square_outline(x0, x1, y0, y1, z, color, radius):
    """Axis-aligned square on the plane z=const."""
    corners=[(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z), (x0, y0, z)]
    return polyline(corners, color, radius)


def image_plane(im_path, opacity):
    img=ImageMobject(im_path)
    img.set_width(image_bounds[1]-image_bounds[0], stretch=True)
    img.set_height(image_bounds[3]-image_bounds[2], stretch=True)
    img.set_opacity(opacity)
    img.move_to([0, 0, 0.5*(image_bounds[4]+image_bounds[5])])
    return img


def orient(mob):
    """Network depth z -> world x, image x -> world y, image y -> world z."""
    mob.rotate(90*DEGREES, [0, 1, 0], about_point=ORIGIN)
    mob.rotate(90*DEGREES, [1, 0, 0], about_point=ORIGIN)
    return mob


def layer_stats(r, pct=97):
    """Per-channel max and per-channel percentile threshold (of the normalized map)."""
    r=np.asarray(r, dtype=np.float64)
    vmax=r.max(axis=(1, 2), keepdims=True)
    vmax[vmax==0]=1.0
    thresh=np.percentile(r/vmax, pct, axis=(1, 2), keepdims=True)
    return vmax, thresh


def relu_viz_block(r, z0, z_step, cell, pct=97, alpha=0.7, cell_z=cell_depth, stats=None):
    """Per-channel normalized, per-channel percentile-thresholded, one merged block."""
    r=np.asarray(r, dtype=np.float64)
    vmax, thresh=layer_stats(r, pct) if stats is None else stats
    rn=r/vmax
    return conv_data_block(rn, z0, vmin=0.0, vmax=1.0, keep=(rn>thresh),
                           cell_size=cell, alpha=alpha, z_step=z_step, cell_z=cell_z)


## ---- Reading the activation cache into a list of drawable tensors ----

def load_act(model_id):
    return np.load(os.path.join(act_dir, f'activations_{model_id}.npy'), allow_pickle=True).item()


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


## ---- Kernel / filter visualization ----

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

    group=VGroup()
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

    group=VGroup()
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


## ---- Skip connections (resnets) ----

def skip_arc(src, dst, height, color, radius, n_pts=32):
    """Arc over the top of the network from src's output face to dst's input face (un-oriented
    +y is image-up, which orient() sends to world +z)."""
    p0=np.array([0.0, src['bounds'][3], src['z1']])
    p1=np.array([0.0, dst['bounds'][3], dst['z0']])
    t=np.linspace(0.0, 1.0, n_pts)
    pts=p0[None,:]+(p1-p0)[None,:]*t[:,None]
    pts[:,1]+=height*np.sin(np.pi*t)
    curve=VMobject()
    curve.set_points_as_corners(pts)
    curve.set_stroke(color, width=200*radius, opacity=1.0)
    curve.set_fill(opacity=0.0)
    curve.set_scale_stroke_with_zoom(True)
    curve.apply_depth_test()
    curve.set_joint_type('bevel')
    return curve


## ---- fc column (from p25_35, minus the probability axes) ----

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


## ---- The scene ----

class GeneralNet(InteractiveScene):
    """Base class: draws one activation cache. Subclasses only override the attributes below."""

    model_id='plain14'

    #Per-model layout knobs
    depth_scale=0.8                   #multiplies base_depth; 1.0 reproduces p25_35's plain8 proportions
    layer_spacing=7.0                 #world units between consecutive blocks, default = 5.0
    tensors_per_block=2               #2 -> post-conv1 relu + block output; 1 -> block outputs only
    depth_mults={64: 1.0, 128: 1.35, 256: 1.8, 512: 2.3}
    max_layers=None                   #debug: only draw the first N blocks

    #Activation thresholding
    act_pct=97
    act_alpha=0.25

    #fc column
    show_fc=True
    fc_pct=70
    fc_vmax_div=2.0

    #Kernel viz: {destination layer index: dict(i, j, prism, ksize, color)} -- indices are printed at
    #startup by describe(). Index 0 is the stem, whose source is the input image (default ksize 7).
    kernels={0: dict(i=10, j=10, prism=True),           #image (7x7, stride 2) -> stem
             1: dict(i=20, j=40, prism=True), #, color=CYAN),
             2: dict(i=20, j=40, prism=True),
             3: dict(i=9, j=16, prism=True),
             4: dict(i=5, j=7, prism=True),
             5: dict(i=5, j=7, prism=True),
             6: dict(i=3, j=3, prism=True),
             7: dict(i=3, j=3, prism=True),
             8: dict(i=3, j=3, prism=True),
             9: dict(i=3, j=3, prism=True),
             10: dict(i=3, j=3, prism=True),
             11: dict(i=3, j=3, prism=True),
             12: dict(i=3, j=3, prism=True)
             }

    kernel_color=MAGENTA
    kernel_ksize=3
    kernel_stem_ksize=7
    kernel_prism=False                #default for kernels that don't specify prism=
    kernel_radius=line_radius

    #fc "kernel": patch on the last conv block's output face (prism optional) fanned out to the fc
    #column. dict(i, j, ksize, prism, color, radius); ksize='full' uses the whole face. None -> off.
    fc_kernel=dict(i=3, j=3, ksize=3, prism=True)

    #Skip connections (resnets): arc from block input to block output over the top of the net
    show_skips=False
    skip_color=BLUE
    skip_downsample_color=YELLOW      #first block of layer2/3/4: 1x1 stride-2 conv on the skip path
    skip_height=6.0
    skip_radius=0.12

    #Presentation
    image_opacity=0.6
    fade_in=False
    fade_in_time=6.0
    default_view=None                 #(theta, phi, gamma, center, height); None -> auto from the layout

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
        self.net=Group(*self.blocks)
        return self

    def view(self):
        if self.default_view is not None:
            return self.default_view
        L=self.total_z
        return (0, 57, 0, (0.5*L, 0.0, 0.0), max(60.0, 0.95*L))

    def construct(self):
        self.build()
        self.frame.reorient(*self.view())
        self.add(self.img, self.image_border)

        pairs=list(zip(self.blocks, self.borders))
        if self.fc_block is not None:
            pairs.append((self.fc_block, self.fc_border))
        extras=self.kernel_mobs+self.skip_mobs

        if self.fade_in:
            self.wait(1)
            fades=[AnimationGroup(FadeIn(b), FadeIn(p)) for b, p in pairs]
            self.play(LaggedStart(*fades, lag_ratio=1.0), run_time=self.fade_in_time)
            if extras:
                self.play(*[FadeIn(m) for m in extras], run_time=1.5)
        else:
            for b, p in pairs:
                self.add(b, p)
            self.add(*extras)

        self.wait(still_hold)
        self.embed()


## ---- One class per model; depth_scale/spacing below are starting guesses to tune ----

class Plain8(GeneralNet):
    model_id='plain8'
    depth_scale=1.0
    layer_spacing=5.0
    kernels={0: dict(i=40, j=70, prism=False),           #image (7x7, stride 2) -> stem
             1: dict(i=22, j=30, prism=True),
             3: dict(i=9, j=16, prism=False),
             5: dict(i=5, j=7, prism=True)}
    fc_kernel=dict(i=5, j=7, ksize=3, prism=True)      #or ksize='full' for the whole 14x14 face
    default_view=(2, 57, 0, (105.69, 16.14, -11.07), 179.39)   #from p25_35


class Plain14(GeneralNet):
    model_id='plain14'
    depth_scale=0.9
    layer_spacing=4.0
    kernels=kernels_for([1, 5, 9], i=10, j=14, prism=True)


class Plain20(GeneralNet):
    model_id='plain20'
    depth_scale=0.7
    layer_spacing=3.5
    kernels=kernels_for([1, 7, 13], i=10, j=14, prism=True)


class Plain26(GeneralNet):
    model_id='plain26'
    depth_scale=0.55
    layer_spacing=3.0
    kernels=kernels_for([1, 7, 13, 19], i=10, j=14, prism=True)


class Plain34(GeneralNet):
    model_id='plain34'
    depth_scale=0.45
    layer_spacing=2.5
    kernels=kernels_for([1, 9, 17, 25], i=10, j=14, prism=True)


class Plain56(GeneralNet):
    model_id='plain56'
    depth_scale=0.3
    layer_spacing=2.0
    kernels=kernels_for([1, 7, 15, 49], i=10, j=14, prism=False)


class Plain74(GeneralNet):
    model_id='plain74'
    depth_scale=0.25
    layer_spacing=1.5
    kernels=kernels_for([1, 7, 15, 67], i=10, j=14, prism=False)


class ResNet74(Plain74):
    model_id='resnet74'
    show_skips=True
    skip_height=6.0