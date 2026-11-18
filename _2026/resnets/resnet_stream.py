"""Cached ViT activation streams and the geometry used to animate them."""

import json
import os

import manimgx as gx
import numpy as np
from PIL import Image

from resnet_assets import ACTIVATION_ROOT
from resnet_geometry import VoxelBlock, polyline, viridis

G = 37
base_depth = 20.0
cell_depth = 0.1
cache_dir = str(ACTIVATION_ROOT / 'p63_cache') + '/'
key_image_name = '0111.png'
key_image_fallback = str(ACTIVATION_ROOT / 'p63_vit_input_images' / 'stephen' / key_image_name)


def front_square(cx, cz, w, y, color, radius):
    """Axis-aligned square outline of width w in the vertical plane y=const (the grid plane)."""
    h=0.5*w
    corners=[(cx-h, y, cz-h), (cx+h, y, cz-h), (cx+h, y, cz+h), (cx-h, y, cz+h), (cx-h, y, cz-h)]
    return polyline(corners, color, radius)


def to_front_plane(mob, y=0.0):
    """Lay a flat xy-plane mobject (ImageMobject, SVGMobject, ...) upright in the vertical
    plane y=const, facing the camera at theta=0/phi=90. Rotates about the mobject's own center."""
    c=mob.get_center()
    mob.rotate(90*gx.DEGREES, gx.RIGHT, about_point=c)
    mob.move_to([c[0], y, c[1]])
    return mob


def register_viz_block(reg, vmax, z0, z_step, reg_cell, face_he, gap, z_offset=0.0,
                       alpha=0.8, cell_z=cell_depth):
    """The register tokens as an n_reg-wide x 1-cell-tall strip whose channel slabs march
    along the same z-range as the paired patch block. Colors reuse the paired block's
    per-channel vmax, so registers are directly comparable to the patches next to them --
    values past the patch max clip to the top of viridis. No percentile threshold: all
    n_reg*C cells are kept (it's tiny).

    In network coords the strip hangs in front of the x=-face_he face (the camera-facing
    side after orient) with its bottom edge at y=-face_he+z_offset; token 0 sits nearest
    the block, the last token nearest the camera. Returns (VoxelBlock, bounds)."""
    reg=np.asarray(reg, dtype=np.float64)                        #(C, 1, R)
    n_c, _, n_r=reg.shape
    rn=np.clip(np.maximum(reg, 0.0)/vmax, 0.0, 1.0)              #vmax broadcasts from (C,1,1)

    kk, tt=np.meshgrid(np.arange(n_c), np.arange(n_r), indexing='ij')
    x0=-face_he-gap                                              #strip edge nearest the block face
    y0=-face_he+z_offset                                         #bottom edge of the strip
    centers=np.stack([x0-(tt+0.5)*reg_cell,
                      np.full(kk.shape, y0+0.5*reg_cell),
                      z_step*kk+z0], axis=-1).reshape(-1, 3)
    rgba=viridis(rn[:, 0, :].reshape(-1))                        #k-major, matches centers
    rgba[:, 3]=alpha
    block=VoxelBlock(centers, np.array([reg_cell, reg_cell, cell_z]), rgba)
    bounds=(x0-n_r*reg_cell, x0, y0, y0+reg_cell, z0, n_c*z_step+z0)
    return block, bounds


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


def register_tensors(stream, n_skip, first_idx=1, depth_pool=4, pool='max', max_blocks=None):
    """(C', 1, R) per stream location: the R=n_skip-1 register tokens (stream[:, 1:n_skip]),
    channel-first, pooled with the SAME grouping as stream_tensors so pooled channels line
    up 1:1 with the patch blocks. Returns None if the model has no registers."""
    n_reg=n_skip-1
    if n_reg<1:
        return None
    idx=range(first_idx, stream.shape[0])
    if max_blocks is not None:
        idx=list(idx)[:max_blocks]
    out=[]
    for i in idx:
        t=np.asarray(stream[i, 1:n_skip], dtype=np.float32)      #(R, D)
        t=t.T.reshape(-1, 1, n_reg)                              #(D, 1, R)
        out.append(channel_pool(t, depth_pool, pool))
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


class _InPlaceAnimation(gx.Animation):
    """Animate large meshes without storing ManimGX's default copied keyframe."""

    keys = ()

    def eased_progress(self, alpha):
        return self.rate_func(1 - alpha if self.reverse_rate_function else alpha)


class SpinTo(_InPlaceAnimation):
    """Incremental rotation about a fixed point, optionally drifting by `shift` over the course of
    the spin (the pivot travels with the drift)."""
    def __init__(self, mobject, angle, axis=gx.OUT, about_point=None, shift=None, **kwargs):
        self.angle=angle
        self.axis=axis
        self.about_point=None if about_point is None else np.asarray(about_point, dtype=np.float64)
        self.shift_vec=None if shift is None else np.asarray(shift, dtype=np.float64)
        super().__init__(mobject, **kwargs)

    def begin(self):
        self.prev = 0.0
        self.pivot = self.mobject.get_center() if self.about_point is None else self.about_point
        super().begin()

    def interpolate_mobject(self, alpha):
        a=self.eased_progress(alpha)
        if a==self.prev:
            return
        pivot=self.pivot
        if self.shift_vec is not None:
            pivot=pivot+self.prev*self.shift_vec
        self.mobject.rotate((a-self.prev)*self.angle, axis=self.axis, about_point=pivot)
        if self.shift_vec is not None:
            self.mobject.shift((a-self.prev)*self.shift_vec)
        self.prev=a


class SquashTo(_InPlaceAnimation):
    """Stretch along `dim` from factor 1 to `factor`, about a fixed point."""
    def __init__(self, mobject, factor, dim=1, about_point=None, **kwargs):
        self.factor=factor
        self.dim=dim
        self.about_point=about_point
        super().__init__(mobject, **kwargs)

    def begin(self):
        self.prev = 1.0
        super().begin()

    def interpolate_mobject(self, alpha):
        a=self.eased_progress(alpha)
        f=1.0+(self.factor-1.0)*a
        if f!=self.prev:
            self.mobject.stretch(f/self.prev, self.dim, about_point=self.about_point)
            self.prev=f


class GlideTo(_InPlaceAnimation):
    """Move a mobject to `target` while uniformly scaling it by `scale`, incrementally.
    `start` is the mobject's logical center (a VoxelBlock's bounding box only spans the kept
    voxels, so get_center() is not the block center)."""
    def __init__(self, mobject, target, start=None, scale=1.0, **kwargs):
        self.target=np.asarray(target, dtype=np.float64)
        self.logical_center=None if start is None else np.asarray(start, dtype=np.float64)
        self.scale_factor=scale
        super().__init__(mobject, **kwargs)

    def begin(self):
        self.start = self.mobject.get_center() if self.logical_center is None else self.logical_center
        self.prev = 0.0
        self.prev_scale = 1.0
        super().begin()

    def interpolate_mobject(self, alpha):
        a=self.eased_progress(alpha)
        if a==self.prev:
            return
        s=1.0+(self.scale_factor-1.0)*a
        if s!=self.prev_scale:
            here=self.start+self.prev*(self.target-self.start)
            self.mobject.scale(s/self.prev_scale, about_point=here)
            self.prev_scale=s
        self.mobject.shift((a-self.prev)*(self.target-self.start))
        self.prev=a
