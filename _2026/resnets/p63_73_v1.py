from manimlib import *
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from PIL import Image

#ulimit -n 16384

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

# data_dir='/Volumes/hot_1/Stephencwelch Dropbox/welch_labs/resnet/hackin/'
data_dir='/Users/stephen/Library/CloudStorage/Dropbox-Stephencwelch/welch_labs/resnet/hackin/'
cache_dir=data_dir+'p63_cache/'                      #from p63_cache_activations.ipynb
key_image_name='0111.png'                            #the single image behind streams/ and max_grids/
key_image_fallback=data_dir+'p63_vit_input_images/stephen/0111.png'   #un-cropped original, if frames.json is missing

## ---- Geometry shared with general_network_rendering (p25_35 / p13_30 lineage) ----
line_radius=0.18
cell_depth=0.1
pixel_dim=0.5
block_cell=0.48
base_depth=20.0                                  #world-unit depth of one block at depth_scale=1
G=37                                             #patch grid side (518 / 14)

still_hold=1.0


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
    vals=(a-vmin)/(vmax-vmin)
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


def front_square(cx, cz, w, y, color, radius):
    """Axis-aligned square outline of width w in the vertical plane y=const (the grid plane)."""
    h=0.5*w
    corners=[(cx-h, y, cz-h), (cx+h, y, cz-h), (cx+h, y, cz+h), (cx-h, y, cz+h), (cx-h, y, cz-h)]
    return polyline(corners, color, radius)


def orient(mob):
    """Network depth z -> world x, image x -> world y, image y -> world z."""
    mob.rotate(90*DEGREES, [0, 1, 0], about_point=ORIGIN)
    mob.rotate(90*DEGREES, [1, 0, 0], about_point=ORIGIN)
    return mob


def to_front_plane(mob, y=0.0):
    """Lay a flat xy-plane mobject (ImageMobject, SVGMobject, ...) upright in the vertical
    plane y=const, facing the camera at theta=0/phi=90. Rotates about the mobject's own center."""
    c=mob.get_center()
    mob.rotate(90*DEGREES, RIGHT, about_point=c)
    mob.move_to([c[0], y, c[1]])
    return mob


def layer_stats(r, pct=97):
    """Per-channel max and per-channel percentile threshold (of the normalized map)."""
    r=np.asarray(r, dtype=np.float64)
    vmax=r.max(axis=(1, 2), keepdims=True)
    vmax[vmax<=0]=1.0
    thresh=np.percentile(r/vmax, pct, axis=(1, 2), keepdims=True)
    return vmax, thresh


def vit_viz_block(a, z0, z_step, cell, pct=99, alpha=0.25, cell_z=cell_depth):
    """Positive part only, per-channel max-normalized, per-channel percentile-thresholded --
    same recipe as relu_viz_block in general_network_rendering, applied to a signed stream."""
    r=np.maximum(np.asarray(a, dtype=np.float64), 0.0)
    vmax, thresh=layer_stats(r, pct)
    rn=r/vmax
    return conv_data_block(rn, z0, vmin=0.0, vmax=1.0, keep=(rn>thresh),
                           cell_size=cell, alpha=alpha, z_step=z_step, cell_z=cell_z)


## ---- Reading the p63 cache ----

def load_meta():
    p=os.path.join(cache_dir, 'meta.json')
    if os.path.exists(p):
        return json.load(open(p))
    return dict(n_skip={'plain': 1, 'reg': 5}, grid=G)


def load_stream(model):
    """(41, N, 1536) float16, memory-mapped. stream[i] = input to block i; stream[40] = last output."""
    return np.load(os.path.join(cache_dir, 'streams', f'stream_{model}.npy'), mmap_mode='r')


def channel_pool(t, factor, mode='max'):
    """Position-independent channel pooling: (C, G, G) -> (C/factor, G, G), each output channel
    the max (or mean) over `factor` consecutive input channels."""
    if factor<=1:
        return t
    c=t.shape[0]//factor
    t=t[:c*factor].reshape(c, factor, *t.shape[1:])
    return t.max(axis=1) if mode=='max' else t.mean(axis=1)


def stream_tensors(stream, n_skip, first_idx=1, depth_pool=4, pool='max', max_blocks=None):
    """(name, (C', G, G)) for stream[first_idx:], patch tokens only, channel-first, pooled."""
    idx=range(first_idx, stream.shape[0])
    if max_blocks is not None:
        idx=list(idx)[:max_blocks]
    out=[]
    for i in idx:
        t=np.asarray(stream[i, n_skip:], dtype=np.float32)          #(G*G, D)
        t=t.reshape(G, G, -1).transpose(2, 0, 1)                    #(D, G, G): row-major patches
        out.append((f'block_{i}', channel_pool(t, depth_pool, pool)))
    return out


def max_grid_png(model, stream_idx):
    return os.path.join(cache_dir, 'max_grids', model, f'layer_{stream_idx:02d}.png')


def key_input_png():
    """The center-cropped 518px input frame that matches the 37x37 grid (from the streaming
    export), found through stephen/frames.json. Falls back to the original file."""
    fj=os.path.join(cache_dir, 'stephen', 'frames.json')
    if os.path.exists(fj):
        for k, v in json.load(open(fj)).items():
            if v==key_image_name:
                return os.path.join(cache_dir, 'stephen', 'input', f'frame_{int(k):04d}.png')
    return key_image_fallback


def upsampled_png(path, factor):
    """ImageMobject textures are sampled bilinearly, which turns a 37x37 png into a blur.
    Nearest-neighbor upsample once into a sibling _upN/ folder and use that file instead."""
    if factor is None or factor<=1:
        return path
    out=os.path.join(os.path.dirname(path), f'_up{factor}', os.path.basename(path))
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        im=Image.open(path)
        im.resize((im.width*factor, im.height*factor), Image.NEAREST).save(out)
    return out


## ---- Layout ----

def block_layout(tensors, depth_scale, spacing, cell, z_start):
    layers=[]
    z=z_start
    for idx, (name, a) in enumerate(tensors):
        n_c, n, _=a.shape
        depth=base_depth*depth_scale
        he=(np.floor(n/2)+0.5)*cell
        layers.append(dict(idx=idx, name=name, data=a, n_c=n_c, n=n, cell=cell,
                           depth=depth, z_step=depth/n_c, z0=z, z1=z+depth,
                           bounds=(-he, he, -he, he, z, z+depth)))
        z+=depth+spacing
    return layers, z


## ---- Cheap in-place animations for big meshes ----
# Rotate/Transform copy the whole mobject per frame; a 40-block chain of VoxelBlocks is millions of
# vertices, so these apply incremental rotations / stretches to the live points instead.

class SpinTo(Animation):
    def __init__(self, mobject, angle, axis=OUT, about_point=None, **kwargs):
        self.angle=angle
        self.axis=axis
        self.about_point=about_point
        self.prev=0.0
        super().__init__(mobject, **kwargs)

    def create_starting_mobject(self):
        return self.mobject                 #skip the copy

    def interpolate_mobject(self, alpha):
        a=self.rate_func(alpha)
        if a!=self.prev:
            self.mobject.rotate((a-self.prev)*self.angle, axis=self.axis, about_point=self.about_point)
            self.prev=a


class SquashTo(Animation):
    """Stretch along `dim` from factor 1 to `factor`, about a fixed point."""
    def __init__(self, mobject, factor, dim=1, about_point=None, **kwargs):
        self.factor=factor
        self.dim=dim
        self.about_point=about_point
        self.prev=1.0
        super().__init__(mobject, **kwargs)

    def create_starting_mobject(self):
        return self.mobject

    def interpolate_mobject(self, alpha):
        a=self.rate_func(alpha)
        f=1.0+(self.factor-1.0)*a
        if f!=self.prev:
            self.mobject.stretch(f/self.prev, self.dim, about_point=self.about_point)
            self.prev=f


## ---- The scene ----

class ViTStream(InteractiveScene):
    """40 residual-stream tensors of DINOv2 ViT-g/14 drawn as CNN-style activation blocks, then
    (optionally) collapsed into a 5x8 grid of the cached channel-max pngs."""

    model='plain'                     #'plain' or 'reg'
    first_idx=1                       #stream[1] = output of block 0. 0 would include the patch-embed input
    depth_pool=64                      #1536 -> 384 channels
    pool='max'                        #'max' or 'mean' over each group of depth_pool channels
    max_blocks=None                   #debug: only the first N blocks

    #Layout (same meaning as the CNN scenes)
    depth_scale=0.5                   #horizontal scale: block depth = base_depth*depth_scale world units
    layer_spacing=3.0                 #world units between consecutive blocks
    cell=block_cell                   #face cell size; face width = 37*cell

    #Voxel selection
    act_pct=99                        #keep the top (100-pct)% of cells per channel. 40x384x37x37 cells:
    act_alpha=0.25                    # pct=99 -> ~210k voxels, pct=97 -> ~630k. Start high.

    #Input image block at the front of the chain
    show_image=True
    image_scale=1.5                   #image face width / block face width
    image_opacity=0.6
    image_depth=3*pixel_dim

    #Presentation
    fade_in=False
    fade_in_time=6.0
    default_view=None                 #(theta, phi, gamma, center, height); None -> auto side view
    do_collapse=False

    #Collapse-to-grid
    grid_rows=5
    grid_cols=8
    grid_cell=None                    #png width in the grid; None -> same as the block face (no rescale)
    grid_gap=None                     #None -> 0.15*grid_cell
    grid_image_scale=1.0              #input image width in the grid / grid_cell
    grid_image_gap=None               #space between the image and the first column; None -> 2*grid_gap
    grid_center_x=None                #None -> center of the network chain
    grid_y=0.0                        #the vertical plane the grid lives in
    grid_border=True
    grid_image_opacity=1.0
    png_upsample=8                    #nearest-neighbor upsample of the 37x37 pngs (None/1 -> off)
    collapse_factor=0.02              #how flat the block gets before the png takes over
    rotate_time=3.0
    collapse_time=3.0
    move_time=4.0
    lag_ratio=0.03                    #stagger between blocks within each stage
    arrows_svg=None                   #path to an Illustrator svg drawn over export_layout_svg's output
    arrows_color=None                 #None -> keep the svg's own colors
    arrows_stroke=3.0
    final_view=None                   #None -> auto front view framing the grid

    # ------------------------------------------------------------------

    def load(self):
        meta=load_meta()
        self.n_skip=meta['n_skip'][self.model]
        stream=load_stream(self.model)
        tensors=stream_tensors(stream, self.n_skip, self.first_idx, self.depth_pool, self.pool,
                               self.max_blocks)
        self.face_w=G*self.cell
        if self.show_image:
            self.image_w=self.image_scale*self.face_w
            hw=0.5*self.image_w
            self.image_bounds=(-hw, hw, -hw, hw, 0.0, self.image_depth)
            z_start=self.image_depth+self.layer_spacing
        else:
            z_start=0.0
        self.layers, self.total_z=block_layout(tensors, self.depth_scale, self.layer_spacing,
                                               self.cell, z_start)
        for L in self.layers:
            L['stream_idx']=self.first_idx+L['idx']
            L['png']=max_grid_png(self.model, L['stream_idx'])
        return self.layers

    def describe(self):
        n_vox=0
        print(f'\n{self.model}: {len(self.layers)} blocks, chain spans x=0 -> {self.total_z:.1f}, '
              f'face {self.face_w:.1f} wide')
        print(f'{"idx":>4}  {"name":10s} {"shape":16s} {"x0":>8} {"x1":>8}  voxels  png')
        for L, b in zip(self.layers, self.blocks):
            n_vox+=len(b.centers)
            print(f"{L['idx']:>4}  {L['name']:10s} {str(L['data'].shape):16s} "
                  f"{L['z0']:8.1f} {L['z1']:8.1f}  {len(b.centers):6d}  {os.path.basename(L['png'])}")
        print(f'total voxels: {n_vox}\n')

    def build_blocks(self):
        blocks, borders=[], []
        for L in self.layers:
            cz=min(cell_depth, 0.8*L['z_step'])
            blk, _=vit_viz_block(L['data'], L['z0'], L['z_step'], L['cell'],
                                 pct=self.act_pct, alpha=self.act_alpha, cell_z=cz)
            blocks.append(orient(blk))
            borders.append(orient(prism(*L['bounds'], CHILL_BROWN, line_radius)))
        return blocks, borders

    def build_image(self):
        b=self.image_bounds
        img=ImageMobject(key_input_png())
        img.set_width(b[1]-b[0], stretch=True)
        img.set_height(b[3]-b[2], stretch=True)
        img.set_opacity(self.image_opacity)
        img.move_to([0, 0, 0.5*(b[4]+b[5])])
        return orient(img), orient(prism(*b, CHILL_BROWN, line_radius))

    def build(self):
        self.load()
        self.blocks, self.borders=self.build_blocks()
        self.describe()
        if self.show_image:
            self.img, self.image_border=self.build_image()
        else:
            self.img=self.image_border=None
        self.net=Group(*self.blocks)
        return self

    def view(self):
        if self.default_view is not None:
            return self.default_view
        L=self.total_z
        return (2, 57, 0, (0.5*L, 0.0, 0.0), max(60.0, 0.95*L))

    # ---- grid geometry (world coords; grid plane is y=grid_y, x right, z up) ----

    def grid_geometry(self):
        cell=self.face_w if self.grid_cell is None else self.grid_cell
        gap=0.15*cell if self.grid_gap is None else self.grid_gap
        img_w=self.grid_image_scale*cell
        img_gap=2*gap if self.grid_image_gap is None else self.grid_image_gap
        grid_w=self.grid_cols*cell+(self.grid_cols-1)*gap
        grid_h=self.grid_rows*cell+(self.grid_rows-1)*gap
        comp_w=grid_w+(img_w+img_gap if self.show_image else 0.0)
        comp_h=max(grid_h, img_w if self.show_image else 0.0)
        cx=0.5*self.total_z if self.grid_center_x is None else self.grid_center_x
        x_left=cx-0.5*comp_w
        z_top=0.5*comp_h
        grid_left=x_left+(img_w+img_gap if self.show_image else 0.0)
        slots=[]
        for k in range(self.grid_rows*self.grid_cols):
            r, c=divmod(k, self.grid_cols)
            slots.append(np.array([grid_left+c*(cell+gap)+0.5*cell, self.grid_y,
                                   z_top-r*(cell+gap)-0.5*cell]))
        image_slot=np.array([x_left+0.5*img_w, self.grid_y, z_top-0.5*img_w])
        return dict(cell=cell, gap=gap, img_w=img_w, slots=slots, image_slot=image_slot,
                    bbox=(x_left, x_left+comp_w, -z_top, z_top), center=np.array([cx, self.grid_y, 0.0]),
                    width=comp_w, height=comp_h)

    def front_view(self):
        if self.final_view is not None:
            return self.final_view
        g=self.grid_geometry()
        h=max(1.2*g['height'], 1.1*g['width']*9/16)
        return (0, 90, 0, tuple(g['center']), h)

    def export_layout_svg(self, path=None, scale=10.0):
        """Write an svg of the final layout (frame rect + image slot + 40 numbered slots) to draw
        arrows over in Illustrator. Keep the outer 'frame' rect in the file you save back out (make
        it invisible if you like) so the import lands exactly; delete the slot rects or not."""
        g=self.grid_geometry()
        x0, x1, z0, z1=g['bbox']
        W, H=(x1-x0)*scale, (z1-z0)*scale
        rects=[]
        def rect(cx, cz, w, rid):
            x=(cx-0.5*w-x0)*scale
            y=(z1-(cz+0.5*w))*scale
            rects.append(f'<rect id="{rid}" x="{x:.2f}" y="{y:.2f}" width="{w*scale:.2f}" '
                         f'height="{w*scale:.2f}" fill="none" stroke="#948979" stroke-width="1"/>')
        if self.show_image:
            s=g['image_slot']; rect(s[0], s[2], g['img_w'], 'input')
        for k, s in enumerate(g['slots'][:len(self.layers)]):
            rect(s[0], s[2], g['cell'], f'layer_{self.layers[k]["stream_idx"]:02d}')
        svg=(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.2f}" height="{H:.2f}" '
             f'viewBox="0 0 {W:.2f} {H:.2f}">\n'
             f'<rect id="frame" x="0" y="0" width="{W:.2f}" height="{H:.2f}" fill="none" '
             f'stroke="#888888" stroke-width="0.5"/>\n'+'\n'.join(rects)+'\n</svg>\n')
        path=path or os.path.join(cache_dir, f'p63_grid_layout_{self.model}.svg')
        open(path, 'w').write(svg)
        print(f'layout svg -> {path}  ({W:.0f}x{H:.0f}, scale {scale} px per world unit)')
        return path

    def load_arrows_svg(self):
        """Illustrator svg (drawn over export_layout_svg's file, frame rect kept) into the grid plane."""
        if self.arrows_svg is None:
            return None
        g=self.grid_geometry()
        svg=SVGMobject(self.arrows_svg)
        if self.arrows_color is not None:
            svg.set_stroke(self.arrows_color, width=self.arrows_stroke)
            svg.set_fill(self.arrows_color, opacity=1.0)
        else:
            svg.set_stroke(width=self.arrows_stroke)
        svg.set_width(g['width'])
        svg.move_to(ORIGIN)
        to_front_plane(svg, y=self.grid_y-0.01)               #a hair in front of the pngs
        svg.move_to([g['center'][0], self.grid_y-0.01, 0.0])
        return svg

    # ---- collapse animation ----

    def build_grid_pngs(self):
        """One ImageMobject per block, sized like the block face, upright in the grid plane,
        parked at each block's center (where the collapsed block will be)."""
        pngs=[]
        for L in self.layers:
            im=ImageMobject(upsampled_png(L['png'], self.png_upsample))
            im.set_width(self.face_w)
            im.set_height(self.face_w, stretch=True)
            to_front_plane(im, y=0.0)
            im.move_to([0.5*(L['z0']+L['z1']), 0.0, 0.0])
            pngs.append(im)
        return pngs

    def collapse_to_grid(self):
        g=self.grid_geometry()
        pngs=self.build_grid_pngs()
        centers=[np.array([0.5*(L['z0']+L['z1']), 0.0, 0.0]) for L in self.layers]

        # Stage A: every block spins -90 deg about the vertical axis so its output face points at
        # the camera (upright, image x right, image y up)
        spins=[]
        if self.show_image:
            ic=self.img.get_center()
            spins.append(AnimationGroup(SpinTo(self.img, -90*DEGREES, OUT, ic),
                                        SpinTo(self.image_border, -90*DEGREES, OUT, ic)))
        for blk, bdr, c in zip(self.blocks, self.borders, centers):
            spins.append(AnimationGroup(SpinTo(blk, -90*DEGREES, OUT, c),
                                        SpinTo(bdr, -90*DEGREES, OUT, c)))
        self.play(LaggedStart(*spins, lag_ratio=self.lag_ratio), run_time=self.rotate_time)

        # Stage B: squash each block flat along its (now y) depth while the png fades in at its center
        squashes=[]
        for blk, bdr, png, c in zip(self.blocks, self.borders, pngs, centers):
            squashes.append(AnimationGroup(SquashTo(blk, self.collapse_factor, 1, c),
                                           SquashTo(bdr, self.collapse_factor, 1, c),
                                           FadeIn(png)))
        self.play(LaggedStart(*squashes, lag_ratio=self.lag_ratio), run_time=self.collapse_time)

        self.remove(*self.blocks, *self.borders)
        outlines=[]
        if self.grid_border:
            for c in centers:
                o=front_square(c[0], c[2], self.face_w, 0.0, CHILL_BROWN, line_radius)
                outlines.append(o)
                self.add(o)
        if self.show_image:
            self.remove(self.image_border)
            ic=self.img.get_center()
            img_outline=front_square(ic[0], ic[2], self.image_w, 0.0, CHILL_BROWN, line_radius)
            if self.grid_border:
                outlines.append(img_outline)
                self.add(img_outline)

        # Stage C: everything glides to its grid slot while the camera tilts up to the front view
        moves=[]
        if self.show_image:
            t=g['image_slot']
            anims=[self.img.animate.move_to(t).set_width(g['img_w']).set_opacity(self.grid_image_opacity)]
            if self.grid_border:
                anims.append(img_outline.animate.move_to(t).set_width(g['img_w']))
            moves.append(AnimationGroup(*anims))
        for k, (png, c) in enumerate(zip(pngs, centers)):
            t=g['slots'][k]
            anims=[png.animate.move_to(t).set_width(g['cell'])]
            if self.grid_border:
                anims.append(outlines[k].animate.move_to(t).set_width(g['cell']))
            moves.append(AnimationGroup(*anims))
        self.play(self.frame.animate.reorient(*self.front_view()),
                  LaggedStart(*moves, lag_ratio=self.lag_ratio), run_time=self.move_time)

        self.grid_pngs=pngs
        self.grid_outlines=outlines
        arrows=self.load_arrows_svg()
        if arrows is not None:
            self.play(Write(arrows), run_time=2.0)
            self.arrows=arrows

    def construct(self):
        self.wait()


        self.build()
        
        self.frame.reorient(*self.view())
        if self.show_image:
            self.add(self.img, self.image_border)

        pairs=list(zip(self.blocks, self.borders))
        if self.fade_in:
            self.wait(1)
            fades=[AnimationGroup(FadeIn(b), FadeIn(p)) for b, p in pairs]
            self.play(LaggedStart(*fades, lag_ratio=1.0), run_time=self.fade_in_time)
        else:
            for b, p in pairs:
                self.add(b, p)

        self.wait(still_hold)
        if self.do_collapse:
            self.collapse_to_grid()
            self.wait(still_hold)
        self.embed()





## ---- Scenes ----

class P63_Plain(ViTStream):
    """Step 1: tune depth_scale / layer_spacing / act_pct here."""
    model='plain'
    depth_scale=0.5
    layer_spacing=3.0





class P63_PlainCollapse(P63_Plain):
    """Step 2: blocks -> 5x8 grid of max_grids/plain/layer_01..40.png."""
    do_collapse=True
    # arrows_svg=cache_dir+'p63_arrows_plain.svg'   #after drawing over export_layout_svg()'s file


class P63_Reg(ViTStream):
    model='reg'
    depth_scale=0.5
    layer_spacing=3.0