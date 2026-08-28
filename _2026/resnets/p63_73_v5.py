from manimlib import *
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from PIL import Image

CHILL_BROWN='#948979'

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
    """Incremental rotation about a fixed point, optionally drifting by `shift` over the course of
    the spin (the pivot travels with the drift)."""
    def __init__(self, mobject, angle, axis=OUT, about_point=None, shift=None, **kwargs):
        self.angle=angle
        self.axis=axis
        self.about_point=np.asarray(about_point, dtype=np.float64)
        self.shift_vec=None if shift is None else np.asarray(shift, dtype=np.float64)
        self.prev=0.0
        super().__init__(mobject, **kwargs)

    def create_starting_mobject(self):
        return self.mobject                 #skip the copy

    def interpolate_mobject(self, alpha):
        a=self.rate_func(alpha)
        if a==self.prev:
            return
        pivot=self.about_point
        if self.shift_vec is not None:
            pivot=pivot+self.prev*self.shift_vec
        self.mobject.rotate((a-self.prev)*self.angle, axis=self.axis, about_point=pivot)
        if self.shift_vec is not None:
            self.mobject.shift((a-self.prev)*self.shift_vec)
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

class P63_Plain(InteractiveScene):
    """40 residual-stream tensors of DINOv2 ViT-g/14 drawn as CNN-style activation blocks:
    spin in place -> squash flat into the max-grid pngs -> fly into a 5x8 grid."""

    def construct(self):
        # ---- config ----
        model='plain'                 #'plain' or 'reg'
        first_idx=1                   #stream[1] = output of block 0. 0 would include the patch-embed input
        depth_pool=32                 #1536 -> 48 channels
        pool='max'                    #'max' or 'mean' over each group of depth_pool channels
        max_blocks=None               #debug: only the first N blocks

        depth_scale=1.0               #horizontal scale: block depth = base_depth*depth_scale world units
        layer_spacing=3.0            #world units between consecutive blocks
        cell=block_cell               #face cell size; face width = 37*cell

        act_pct=90 #99.9 to run faster                  #keep the top (100-pct)% of cells per channel. Put at like 90 before final render
        act_alpha=0.8                 #pct=99 -> ~210k voxels, pct=97 -> ~630k. Start high.

        image_scale=1.5               #image face width / block face width
        image_opacity=0.6
        image_depth=3*pixel_dim
        img_shift=np.array([-11.0, 0, 0])   #where the input image drifts to during its spin - tune me

        grid_rows, grid_cols=5, 8
        grid_y=0.0                    #the vertical plane the grid lives in
        grid_image_opacity=1.0
        png_upsample=8                #nearest-neighbor upsample of the 37x37 pngs (None/1 -> off)
        collapse_factor=0.02          #how flat the blocks get before the pngs take over
        lag_ratio=0.03                #stagger between blocks within each stage

        # ---- load the cached residual stream ----
        n_skip=load_meta()['n_skip'][model]
        tensors=stream_tensors(load_stream(model), n_skip, first_idx, depth_pool, pool, max_blocks)

        face_w=G*cell
        image_w=image_scale*face_w
        hw=0.5*image_w
        image_bounds=(-hw, hw, -hw, hw, 0.0, image_depth)
        layers, total_z=block_layout(tensors, depth_scale, layer_spacing, cell,
                                     image_depth+layer_spacing)
        for L in layers:
            L['stream_idx']=first_idx+L['idx']
            L['png']=max_grid_png(model, L['stream_idx'])

        # ---- voxel blocks + borders ----
        blocks, borders=[], []
        for L in layers:
            cz=min(cell_depth, 0.8*L['z_step'])
            blk, _=vit_viz_block(L['data'], L['z0'], L['z_step'], L['cell'],
                                 pct=act_pct, alpha=act_alpha, cell_z=cz)
            blocks.append(orient(blk))
            borders.append(orient(prism(*L['bounds'], CHILL_BROWN, line_radius)))

        n_vox=sum(len(b.centers) for b in blocks)
        print(f'\n{model}: {len(layers)} blocks, chain spans x=0 -> {total_z:.1f}, '
              f'face {face_w:.1f} wide')
        print(f'{"idx":>4}  {"name":10s} {"shape":16s} {"x0":>8} {"x1":>8}  voxels  png')
        for L, b in zip(layers, blocks):
            print(f"{L['idx']:>4}  {L['name']:10s} {str(L['data'].shape):16s} "
                  f"{L['z0']:8.1f} {L['z1']:8.1f}  {len(b.centers):6d}  {os.path.basename(L['png'])}")
        print(f'total voxels: {n_vox}\n')

        # ---- input image block at the front of the chain ----
        img_path=key_input_png()
        print(f'input image: {img_path}  (exists={os.path.exists(img_path)})', flush=True)
        img=ImageMobject(img_path)
        img.set_width(image_bounds[1]-image_bounds[0], stretch=True)
        img.set_height(image_bounds[3]-image_bounds[2], stretch=True)
        img.set_opacity(image_opacity)
        img.move_to([0, 0, 0.5*(image_bounds[4]+image_bounds[5])])
        orient(img)
        image_border=orient(prism(*image_bounds, CHILL_BROWN, line_radius))

        # ---- grid geometry (plane y=grid_y; x right, z up) ----
        gap=0.15*face_w
        img_w=face_w                  #input image width in the grid
        img_gap=2*gap                 #space between the image and the first column
        grid_w=grid_cols*face_w+(grid_cols-1)*gap
        grid_h=grid_rows*face_w+(grid_rows-1)*gap
        comp_w=grid_w+img_w+img_gap
        comp_h=max(grid_h, img_w)
        # Anchored so stage C reads as "the rest fly in from the right": slot 0 sits exactly on
        # block 1's current center and row-0 slot centers sit at z=0 (the chain's height), so the
        # first 8 pngs barely move (spacing just compresses 25 -> ~20.4) while blocks 9-40 stream
        # in from the right and the camera zooms down onto the assembling grid.
        first_cx=0.5*(layers[0]['z0']+layers[0]['z1'])
        grid_left=first_cx-0.5*face_w
        x_left=grid_left-img_gap-img_w
        z_top=0.5*face_w
        slots=[]
        for k in range(grid_rows*grid_cols):
            r, c=divmod(k, grid_cols)
            slots.append(np.array([grid_left+c*(face_w+gap)+0.5*face_w, grid_y,
                                   z_top-r*(face_w+gap)-0.5*face_w]))
        image_slot=np.array([x_left+0.5*img_w, grid_y, z_top-0.5*img_w])
        front_view=(0, 90, 0, (x_left+0.5*comp_w, grid_y, z_top-0.5*comp_h),
                    max(1.2*comp_h, 1.1*comp_w*9/16))

        # ---- draw the chain ----
        self.frame.reorient(0, 62, 0, (np.float32(453.62), np.float32(314.9), np.float32(-181.54)), 867.08)

        self.add(img, image_border)
        for b, p in zip(blocks, borders):
            self.add(b, p)
            self.wait(0.1)

        self.wait(1)
        self.play(self.frame.animate.reorient(0, 70, 0, (np.float32(64.0), np.float32(340.99), np.float32(-119.7)), 409.31),
                  run_time=9.0)
        self.wait(1)

        # ---- max-grid pngs, born at the block centers ----
        centers=[np.array([0.5*(L['z0']+L['z1']), 0.0, 0.0]) for L in layers]
        pngs=[]
        for L, pos in zip(layers, centers):
            im=ImageMobject(upsampled_png(L['png'], png_upsample))
            im.set_width(face_w)
            im.set_height(face_w, stretch=True)
            to_front_plane(im, y=grid_y)
            im.move_to(pos)
            pngs.append(im)

        # ---- Stage A: spin in place; the input image drifts left as it spins ----
        ic=np.array([0.5*image_depth, 0.0, 0.0])
        ic_final=ic+img_shift
        spins=[AnimationGroup(SpinTo(img, -90*DEGREES, OUT, ic, shift=img_shift),
                              SpinTo(image_border, -90*DEGREES, OUT, ic, shift=img_shift))]
        for blk, bdr, c in zip(blocks, borders, centers):
            spins.append(AnimationGroup(SpinTo(blk, -90*DEGREES, OUT, c),
                                        SpinTo(bdr, -90*DEGREES, OUT, c)))

        self.wait()
        self.play(LaggedStart(*spins, lag_ratio=lag_ratio), run_time=2.0) #Increase for final animation

        # ---- Stage B: squash flat about each center while the png fades in there ----
        squashes=[AnimationGroup(SquashTo(img, collapse_factor, 1, ic_final),
                                 SquashTo(image_border, collapse_factor, 1, ic_final))]
        for blk, bdr, png, c in zip(blocks, borders, pngs, centers):
            squashes.append(AnimationGroup(SquashTo(blk, collapse_factor, 1, c),
                                           SquashTo(bdr, collapse_factor, 1, c),
                                           FadeIn(png)))

        self.wait()
        self.play(LaggedStart(*squashes, lag_ratio=lag_ratio), run_time=2.0)

        # swap the squashed prisms for clean squares on the grid plane; drop the voxel blocks
        self.remove(*blocks, *borders)
        outlines=[]
        for c in centers:
            o=front_square(c[0], c[2], face_w, grid_y, CHILL_BROWN, line_radius)
            outlines.append(o)
            self.add(o)

        self.remove(image_border)
        img_outline=front_square(ic_final[0], ic_final[2], image_w, grid_y, CHILL_BROWN, line_radius)
        self.add(img_outline)

        self.wait(1.0)

        # ---- Stage C: straighten the camera, then fly the rest in from the right ----
        # The old single move tilted phi 70->90, swung the frame center's y from ~341 to 0, panned,
        # and zoomed all at once -- the tilt+swing is what shoved row 0 off the top mid-move. So:
        # first straighten to a true front view over the still scene. Its center is chosen so row 0
        # sits at the SAME screen height as in the final view; the fly-in after it is then a pure
        # in-plane zoom, and since gap-to-row-0 and frame height both interpolate linearly with
        # equal start/end ratios, that ratio holds for the whole move -- row 0 stays pinned on
        # screen while rows 2-5 stream in below it.
        row0_z=z_top-0.5*face_w
        row0_ratio=(row0_z-front_view[3][2])/(0.5*front_view[4])   #row-0 screen height in the final view
        pre_h=250.0                   #tune: how wide the straight-on view is before the fly-in
        # pre_view=(0, 90, 0, (front_view[3][0], 0.0, row0_z-row0_ratio*0.5*pre_h), pre_h)
        pre_view=(0, 90, 0, (np.float32(436.06), np.float32(0.0), np.float32(-70.7)), 572.59)
        self.wait()

        self.play(self.frame.animate.reorient(*pre_view), run_time=5.0)
        self.wait(0.5)

        moves=[AnimationGroup(img.animate.move_to(image_slot).set_width(img_w)
                                 .set_opacity(grid_image_opacity),
                              img_outline.animate.move_to(image_slot).set_width(img_w))]
        for png, outline, slot in zip(pngs, outlines, slots):
            moves.append(AnimationGroup(png.animate.move_to(slot).set_width(face_w),
                                        outline.animate.move_to(slot).set_width(face_w)))


        self.play(self.frame.animate.reorient(*front_view),
                  LaggedStart(*moves, lag_ratio=lag_ratio), run_time=3.0) #tune; longer reads calmer

        self.wait(1.0)






