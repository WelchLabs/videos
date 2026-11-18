"""Shared ManimGX geometry for activation volumes and training landscapes."""

import manimgx as m
import matplotlib as mpl
import numpy as np

from welch_style import CHILL_BROWN

cell_depth = 0.1


_FACE_CORNERS = np.array([
    [[-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]],
    [[.5, -.5, -.5], [-.5, -.5, -.5], [-.5, .5, -.5], [.5, .5, -.5]],
    [[.5, -.5, .5], [.5, -.5, -.5], [.5, .5, -.5], [.5, .5, .5]],
    [[-.5, -.5, -.5], [-.5, -.5, .5], [-.5, .5, .5], [-.5, .5, -.5]],
    [[-.5, .5, .5], [.5, .5, .5], [.5, .5, -.5], [-.5, .5, -.5]],
    [[-.5, -.5, -.5], [.5, -.5, -.5], [.5, -.5, .5], [-.5, -.5, .5]],
])
_QUAD_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])
_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]
_VIRIDIS = mpl.colormaps["viridis"](np.linspace(0, 1, 256))


def viridis(values):
    x = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return _VIRIDIS[np.minimum((x * 256).astype(np.int32), 255)]


class VoxelBlock(m.MeshMobject):
    """Draw every voxel as one indexed, unlit mesh with a color per vertex."""

    def __init__(self, centers, sizes, rgba, view_forward=None):
        centers = np.asarray(centers, dtype=float).reshape(-1, 3)
        sizes = np.broadcast_to(np.asarray(sizes, dtype=float), centers.shape)
        rgba = np.asarray(rgba, dtype=float).reshape(-1, 4)
        if view_forward is not None and len(centers):
            order = np.argsort(-(centers @ np.asarray(view_forward, dtype=float)))
            centers, sizes, rgba = centers[order], sizes[order], rgba[order]
        corners = _FACE_CORNERS[None] * sizes[:, None, None] + centers[:, None, None]
        faces = len(centers) * 6
        triangles = 4 * np.arange(faces)[:, None, None] + _QUAD_TRIANGLES
        super().__init__(corners.reshape(-1, 3), triangles.reshape(-1, 3),
                         vertex_colors=np.repeat(rgba, 24, axis=0))


def polyline(points, color, radius):
    """Use separate rounded ends for the original non-mitered line joins."""
    points = np.asarray(points, dtype=float)
    return m.VGroup(*(m.Line(a, b, stroke_color=color, stroke_width=200 * radius)
                      for a, b in zip(points[:-1], points[1:]) if not np.allclose(a, b)))


def prism(min_x, max_x, min_y, max_y, min_z, max_z, color, radius):
    corners = [(min_x, min_y, min_z), (max_x, min_y, min_z),
               (max_x, max_y, min_z), (min_x, max_y, min_z),
               (min_x, min_y, max_z), (max_x, min_y, max_z),
               (max_x, max_y, max_z), (min_x, max_y, max_z)]
    return m.VGroup(*(line for a, b in _BOX_EDGES for line in polyline([corners[a], corners[b]], color, radius)))


def orient(mob):
    """Map network depth to x, image x to y and image y to z."""
    return (mob.rotate(m.PI / 2, m.UP, about_point=m.ORIGIN)
               .rotate(m.PI / 2, m.RIGHT, about_point=m.ORIGIN))


def swap_out(scene, mobject):
    if mobject is not None:
        scene.remove(mobject)


def clear_scene(scene):
    scene.clear()


def blend_views(start, end, progress):
    angles = m.interpolate(np.asarray(start[:3]), np.asarray(end[:3]), progress)
    center = m.interpolate(np.asarray(start[3]), np.asarray(end[3]), progress)
    height = m.interpolate(start[4], end[4], progress)
    return (*angles, center, height)


def conv_data_block(a, start_depth, vmin=None, vmax=None, keep=None, cell_size=1.0,
                    alpha=0.5, z_step=0.125, cell_z=0.1, view_forward=None):
    """Color and place a channel-first activation tensor in an indexed mesh."""
    a = np.asarray(a, dtype=float)
    channels, _, width = a.shape
    vmin = a.min() if vmin is None else vmin
    vmax = a.max() if vmax is None else vmax
    values = (a - vmin) / (vmax - vmin) if vmax != vmin else np.zeros_like(a)
    if keep is None:
        keep = np.ones(a.shape, dtype=bool)
    k, i, j = np.nonzero(keep)
    half = width // 2
    centers = np.column_stack(((j - half) * cell_size, (half - i) * cell_size,
                               start_depth + k * z_step))
    rgba = viridis(values[keep])
    rgba[:, 3] = alpha
    block = VoxelBlock(centers, [cell_size, cell_size, cell_z], rgba, view_forward)
    extent = (half + 0.5) * cell_size
    return block, (-extent, extent, -extent, extent, start_depth, start_depth + channels * z_step)


def threshold_volume(a, start_depth, threshold, *, opacity=0.5, view_forward=None, vmax=None, dense=False):
    """Activation volume for older shots that threshold against a global maximum."""
    a = np.asarray(a, dtype=float)
    maximum = a.max() if vmax is None else vmax
    keep = None if dense else a >= threshold * maximum
    return conv_data_block(a, start_depth, vmin=0.0, vmax=maximum, keep=keep,
                           alpha=opacity, view_forward=view_forward)


def layer_stats(values, pct=97):
    """Return per-channel maxima and normalized percentile thresholds."""
    values = np.asarray(values, dtype=float)
    maxima = values.max(axis=(1, 2), keepdims=True)
    maxima[maxima <= 0] = 1.0
    return maxima, np.percentile(values / maxima, pct, axis=(1, 2), keepdims=True)


def box_image(source, opacity, bounds):
    """Fit a photo to the image plane described by a 3D box's bounds."""
    x0, x1, y0, y1, z0, z1 = bounds
    image = m.ImageMobject(source)
    return (image.stretch_to_fit_width(x1 - x0).stretch_to_fit_height(y1 - y0)
                 .set_opacity(opacity).move_to([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]))


def fc_data_block(a, start_depth, viz_len, thresh, view_forward, cell_spacing=0.25):
    a=np.asarray(a, dtype=np.float64)
    sI=np.argsort(a)[::-1]
    vector_to_viz=a[np.sort(sI[:viz_len])] #Top activations, back in original order
    vals=vector_to_viz/a.max()
    keep=vals>=thresh

    n=len(vector_to_viz)
    i=np.arange(n)
    centers=np.stack([np.zeros(int(keep.sum())),
                      -i[keep]*cell_spacing+cell_spacing*np.floor(n/2),
                      np.full(int(keep.sum()), start_depth)], axis=-1)
    rgba=viridis(vals[keep])
    rgba[:,3]=0.5

    block=VoxelBlock(centers, np.array([1.0, cell_spacing, 1.0]), rgba, view_forward)
    bounds=(-0.5, 0.5, -cell_spacing*np.floor(n/2), cell_spacing*np.floor(n/2),
            start_depth-0.5, start_depth+0.5)
    return block, bounds


def cap_with_triangle(axis, at_start=False, length=2.5, width=2.0, normal=m.UP, color=CHILL_BROWN):
    """Filled triangle on one end of a NumberLine, in the plane perpendicular to `normal`.
    Shortens the line to the triangle's base so the two don't z-fight."""
    p0, p1=axis.get_start(), axis.get_end()
    tip, d=(p0, p0-p1) if at_start else (p1, p1-p0)
    d=d/np.linalg.norm(d)
    perp=np.cross(normal, d)
    base=tip-length*d
    tri=m.Polygon(tip, base+0.5*width*perp, base-0.5*width*perp)
    tri.set_fill(color, 1.0)
    tri.set_stroke(width=0)
    if at_start:
        axis.put_start_and_end_on(base, p1)
    else:
        axis.put_start_and_end_on(p0, base)
    return tri


def masked_conv1(a, i, j, k):
    """Conv-1 revealed in raster order up to cell (i, j) of channel <= k."""
    out=np.zeros_like(a)
    n_j=a.shape[2]
    revealed=(np.arange(a.shape[1])[:,None]*n_j+np.arange(n_j)[None,:])<=(i*n_j+j)
    #The floor keeps near-zero revealed cells above the render threshold
    out[:k+1]=np.where(revealed, np.maximum(a[:k+1], 0.01*a[0].max()), 0.0)
    return out


def forward_from(view):
    """The direction the camera looks, for the painter's-algorithm sort."""
    from scipy.spatial.transform import Rotation
    theta, phi, gamma=np.radians(view[:3])
    return -Rotation.from_euler('zxz', [gamma, phi, theta]).as_matrix()[:,2]


def swing_path(pivot, angle, axis=m.OUT):
    """Rigid rotation about pivot, plus a lerp of whatever the rotation doesn't explain."""
    pivot=np.asarray(pivot, dtype=np.float64)
    full=m.rotation_matrix(angle, axis)
    def path(start_points, end_points, alpha):
        part=m.rotation_matrix(alpha*angle, axis)
        swung=pivot+np.dot(start_points-pivot, part.T)
        landed=pivot+np.dot(start_points-pivot, full.T)
        return swung+alpha*(end_points-landed)
    return path


def axis_arrow(start, end, color, radius, head_len=2.0, head_w=0.9, normal=m.UP):
    """Straight axis with a chevron head at `end`, lying in the plane perpendicular to `normal`."""
    start=np.asarray(start, dtype=np.float64)
    end=np.asarray(end, dtype=np.float64)
    d=end-start
    d/=np.linalg.norm(d)
    perp=np.cross(normal, d)
    group=m.VGroup(*polyline([start, end], color, radius))
    for s in (1, -1):
        group.add(*polyline([end, end-head_len*d+s*head_w*perp], color, radius))
    return group


def make_channel_images(im_path, out_dir):
    """Split the input into R/G/B-only PNGs, once."""
    from PIL import Image
    import os
    paths=[]
    src=None
    for ch, name in enumerate('rgb'):
        p=f'{out_dir}/lemon_{name}.png'
        paths.append(p)
        if not os.path.exists(p):
            if src is None:
                src=np.array(Image.open(im_path).convert('RGB'))
            solo=np.zeros_like(src)
            solo[...,ch]=src[...,ch]
            Image.fromarray(solo).save(p)
    return paths


def reveal_mask(shape, i, j, k):
    """Raster-order reveal up to cell (i, j), channels <= k."""
    n_c, n_i, n_j=shape
    revealed=(np.arange(n_i)[:,None]*n_j+np.arange(n_j)[None,:])<=(i*n_j+j)
    mask=np.zeros(shape, dtype=bool)
    mask[:k+1]=revealed
    return mask


def activation_image_stack(image_dir, n_c, z0, width, z_step, skip_channel=None):
    group=m.Group()
    for c in range(n_c):
        if c==skip_channel:
            continue
        img=m.ImageMobject(f'{image_dir}/act_{c:02d}.png')
        img.stretch_to_fit_width(width)
        img.stretch_to_fit_height(width)
        img.move_to([-0.5*width/56, 0.5*width/56, z0+c*z_step])  #match the voxel grid's half-cell offset
        group.add(img)
    return group


def kernel_weights_stack(weights, extent, z0, z_step):
    """One (C, 3, 3) filter painted through the depth of the source stack."""
    w=np.asarray(weights, dtype=np.float64)
    w=w-w.min()
    min_x, max_x, min_y, max_y=extent
    step=(max_x-min_x)/w.shape[-1]

    kk, ii, jj=np.meshgrid(np.arange(w.shape[0]), np.arange(w.shape[1]),
                           np.arange(w.shape[2]), indexing='ij')
    vals=(w/w.max()).ravel()
    centers=np.stack([(jj.ravel()+0.5)*step+min_x, -(ii.ravel()+0.5)*step+max_y,
                      kk.ravel()*z_step+z0], axis=-1)
    rgba=np.zeros((len(vals), 4))
    rgba[:,0]=vals
    rgba[:,2]=vals
    rgba[:,3]=0.5
    return VoxelBlock(centers, np.array([step, step, cell_depth]), rgba)
