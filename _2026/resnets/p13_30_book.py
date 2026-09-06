from manimlib import *
import numpy as np
import matplotlib.pyplot as plt

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b'
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
MAGENTA='#EB8423' #Sike! '#FF00FF'

# data_dir='/Users/stephen/Library/CloudStorage/Dropbox-Stephencwelch/welch_labs/resnet/hackin'
data_dir='/Volumes/hot_1/Stephencwelch Dropbox/welch_labs/resnet/hackin/'

spacing_between_layers=5
line_radius=0.18
depth_step=0.125
cell_depth=0.1
pixel_dim=0.5

image_opacity=0.936  
still_hold=1.0
steps_per_viz=5     #was 11
fov=PI/3
kernel_k=0
block_cell=0.48


channel_pitch=1.0   #world units between color planes; tune against conv1's depth_step*64
image_bounds=(-32.0, 32.0, -32.0, 32.0, 0.0, 3*channel_pitch)

# image_bounds=(-32.0, 32.0, -32.0, 32.0, 0.0, 3*pixel_dim)
# image_bounds=(-224.0, 224.0, -224.0, 224.0, 0.0, 3*pixel_dim)

#Camera keyframes: theta, phi, gamma, center, height
# p24_end=(-135.4168, 63.3448, 114.4748, (0.0, 0.0, 0.0), 96.224)
# p24d_end=(-139.5221, 31.6032, 134.9423, (0.0, 0.0, 0.0), 96.224)


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
                    alpha=0.5, z_step=depth_step, view_forward=None):

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

    # block=VoxelBlock(centers, np.array([cell_size, cell_size, cell_depth]), rgba)
    block=VoxelBlock(centers, np.array([cell_size, cell_size, cell_depth]), rgba,
                     view_forward=view_forward)
    half_extent=(half+0.5)*cell_size
    bounds=(-half_extent, half_extent, -half_extent, half_extent,
            start_depth, n_c*z_step+start_depth)
    return block, bounds


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


def conv2_kernel(i, j, act_shape, n_src, src_cell, src_z0, src_z1, src_z_step, weights,
                 cell_size, dst_z, ksize=3, show_weights=True):
    """A layer-2 kernel: activation (i, j) wired to its 3x3xC patch in the source stack."""
    group=Group()
    half_dst=np.floor(act_shape[-1]/2)
    hc=0.5*cell_size
    dst_x=(j-half_dst)*cell_size
    dst_y=(-i+half_dst)*cell_size

    #Stride maps output (i, j) to source cell (stride*i, stride*j)
    stride=n_src/act_shape[-1]              #112/56 = 2
    half_src=np.floor(n_src/2)
    src_cx=(stride*j-half_src)*src_cell
    src_cy=(-stride*i+half_src)*src_cell
    r=0.5*ksize*src_cell                    #patch is still 3 *source* cells wide
    extent=(src_cx-r, src_cx+r, src_cy-r, src_cy+r)

    connectors=[
        [(dst_x-hc, dst_y-hc, dst_z), (extent[0], extent[2], src_z1)],
        [(dst_x+hc, dst_y-hc, dst_z), (extent[1], extent[2], src_z1)],
        [(dst_x-hc, dst_y+hc, dst_z), (extent[0], extent[3], src_z1)],
        [(dst_x+hc, dst_y+hc, dst_z), (extent[1], extent[3], src_z1)],
    ]
    for cc in connectors:
        group.add(polyline(cc, MAGENTA, line_radius))
    for p, q in [(0, 1), (1, 3), (3, 2), (2, 0)]:
        group.add(polyline([connectors[p][0], connectors[q][0]], MAGENTA, line_radius))

    group.add(prism(extent[0], extent[1], extent[2], extent[3], src_z0, src_z1,
                    MAGENTA, line_radius))
    if show_weights: group.add(kernel_weights_stack(weights, extent, src_z0, src_z_step))
    return group


def activation_image_stack(image_dir, n_c, z0, width, z_step, skip_channel=None):
    group=Group()
    for c in range(n_c):
        if c==skip_channel:
            continue
        img=ImageMobject(f'{image_dir}/act_{c:02d}.png')
        img.set_width(width, stretch=True)
        img.set_height(width, stretch=True)
        img.move_to([-0.5*width/56, 0.5*width/56, z0+c*z_step])  #match the voxel grid's half-cell offset
        group.add(img)
    return group


def reveal_mask(shape, i, j, k):
    """Raster-order reveal up to cell (i, j), channels <= k."""
    n_c, n_i, n_j=shape
    revealed=(np.arange(n_i)[:,None]*n_j+np.arange(n_j)[None,:])<=(i*n_j+j)
    mask=np.zeros(shape, dtype=bool)
    mask[:k+1]=revealed
    return mask


def image_plane(im_path, opacity):
    img=ImageMobject(im_path)
    img.set_width(image_bounds[1]-image_bounds[0], stretch=True)
    img.set_height(image_bounds[3]-image_bounds[2], stretch=True)
    img.set_opacity(opacity)
    img.move_to([0, 0, 0.5*(image_bounds[4]+image_bounds[5])])
    return img


def kernel_weights_block(weights, extent, pitch=channel_pitch):
    """A conv-1 filter painted into its patch, black through to magenta."""
    w=np.asarray(weights, dtype=np.float64)
    w=w-w.min()
    min_x, max_x, _, max_y=extent
    step=(max_x-min_x)/w.shape[1]

    kk, ii, jj=np.meshgrid(np.arange(w.shape[0]), np.arange(w.shape[1]), np.arange(w.shape[2]),
                           indexing='ij')
    vals=(w/w.max()).ravel()
    centers=np.stack([jj.ravel()*step+min_x, -ii.ravel()*step+max_y,
                      (kk.ravel()+0.5)*pitch+image_bounds[4]], axis=-1)

    #Make that this orange bruh
    EB8423 = np.array([0xEB, 0x84, 0x23]) / 255.0   # (0.9216, 0.5176, 0.1373)
    rgba = np.zeros((len(vals), 4))
    rgba[:, :3] = vals[:, None] * EB8423
    rgba[:, 3] = 0.5

    # rgba=np.zeros((len(vals), 4))
    # rgba[:,0]=vals
    # rgba[:,2]=vals
    # rgba[:,3]=0.5
    return VoxelBlock(centers, np.array([1.0, 1.0, depth_step]), rgba)


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


def conv1_kernel(i, j, k, act_shape, weights, cell_size=1.0, stride=1):
    """The conv-1 kernel wired to its activation, with the filter painted into the patch."""
    min_x, max_x, min_y, _, min_z, max_z=image_bounds
    group=Group()

    #Activation cell (i, j) -> its footprint on the image plane
    step=(max_x-min_x)/act_shape[-1]      #image units per activation cell
    px=step/stride                        #image units per input pixel
    size=weights.shape[-1]*px             #receptive field of the kernel

    half=np.floor(act_shape[-1]/2)
    hc=0.5*cell_size
    dst_x=(j-half)*cell_size
    dst_y=(-i+half)*cell_size
    dst_z=depth_step*k+spacing_between_layers+1

    src_x=min_x+step*j
    src_y=-min_y-step*i  #-min_y spells max_y; the bounds are symmetric

    connectors=[
        [(dst_x-hc, dst_y-hc, dst_z), (src_x,      src_y-size, max_z)],
        [(dst_x+hc, dst_y-hc, dst_z), (src_x+size, src_y-size, max_z)],
        [(dst_x-hc, dst_y+hc, dst_z), (src_x,      src_y,      max_z)],
        [(dst_x+hc, dst_y+hc, dst_z), (src_x+size, src_y,      max_z)],
    ]
    for cc in connectors:
        group.add(polyline(cc, MAGENTA, line_radius))
    for p, q in [(0, 1), (1, 3), (3, 2), (2, 0)]:
        group.add(polyline([connectors[p][0], connectors[q][0]], MAGENTA, line_radius))

    extent=(src_x, src_x+size, src_y-size, src_y)
    group.add(prism(extent[0], extent[1], extent[2], extent[3], min_z, max_z, MAGENTA,
                    line_radius))
    group.add(kernel_weights_block(weights, extent, pitch=channel_pitch))
    return group


def masked_conv1(a, i, j, k):
    """Conv-1 revealed in raster order up to cell (i, j) of channel <= k."""
    out=np.zeros_like(a)
    n_j=a.shape[2]
    revealed=(np.arange(a.shape[1])[:,None]*n_j+np.arange(n_j)[None,:])<=(i*n_j+j)
    #The floor keeps near-zero revealed cells above the render threshold
    out[:k+1]=np.where(revealed, np.maximum(a[:k+1], 0.01*a[0].max()), 0.0)
    return out


def swap_out(scene, mobject):
    """Scene.remove leaves id_to_mobject_map, reference cycles and moderngl buffers behind,
    so without this a per-frame shot leaks until the machine runs out of memory."""
    if mobject is None:
        return
    scene.remove(mobject)
    for sm in mobject.get_family():
        scene.id_to_mobject_map.pop(id(sm), None)
        if sm.shader_wrapper is not None:
            sm.shader_wrapper.release() #Buffers only: the textures are shared, ctx-wide caches
        sm.family=None
        sm.parents.clear()
        sm.submobjects.clear()
        sm.data=None
        sm.shader_wrapper=None
        for attr in ('centers', 'sizes', 'rgba', 'tri_indices', 'triangle_indices'):
            if hasattr(sm, attr):
                setattr(sm, attr, None)


def clear_scene(scene):
    for mob in list(scene.mobjects):
        if mob is not scene.camera.frame:
            swap_out(scene, mob)
    scene.clear()
    scene.id_to_mobject_map.clear()
    scene.add(scene.camera.frame)


def blend_views(a, b, t):
    return (a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t,
            tuple(p+(q-p)*t for p, q in zip(a[3], b[3])), a[4]+(b[4]-a[4])*t)


def orient(mob):
    mob.rotate(90*DEGREES, [0, 1, 0], about_point=ORIGIN)
    mob.rotate(90*DEGREES, [1, 0, 0], about_point=ORIGIN)
    return mob

def activation_image_grid(image_dir, n_c, start_depth, map_width, pitch,
                          skip_channel=None, n_cols=8):
    group=Group()
    for c in range(n_c):
        if c==skip_channel:
            continue
        r, col=divmod(c, n_cols)
        img=ImageMobject(f'{image_dir}/act_{c:02d}.png')
        img.set_width(map_width, stretch=True)
        img.set_height(map_width, stretch=True)
        img.move_to([col*pitch-0.5*block_cell, -r*pitch+0.5*block_cell, start_depth])
        group.add(img)
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

def relu_viz_block(r, z0, z_step, cell, pct=97, alpha=0.7):
    """Per-channel normalized, per-channel percentile-thresholded, one merged block."""
    r=np.asarray(r, dtype=np.float64)
    vmax=r.max(axis=(1, 2), keepdims=True)
    vmax[vmax==0]=1.0
    rn=r/vmax
    thresh=np.percentile(rn, pct, axis=(1, 2), keepdims=True)
    return conv_data_block(rn, z0, vmin=0.0, vmax=1.0, keep=(rn>thresh),
                           cell_size=cell, alpha=alpha, z_step=z_step)

def swing_path(pivot, angle, axis=OUT):
    """Rigid rotation about pivot, plus a lerp of whatever the rotation doesn't explain."""
    pivot=np.asarray(pivot, dtype=np.float64)
    full=rotation_matrix(angle, axis)
    def path(start_points, end_points, alpha):
        part=rotation_matrix(alpha*angle, axis)
        swung=pivot+np.dot(start_points-pivot, part.T)
        landed=pivot+np.dot(start_points-pivot, full.T)
        return swung+alpha*(end_points-landed)
    return path

def axis_arrow(start, end, color, radius, head_len=2.0, head_w=0.9, normal=UP):
    """Straight axis with a chevron head at `end`, lying in the plane perpendicular to `normal`."""
    start=np.asarray(start, dtype=np.float64)
    end=np.asarray(end, dtype=np.float64)
    d=end-start
    d/=np.linalg.norm(d)
    perp=np.cross(normal, d)
    group=VGroup(*polyline([start, end], color, radius))
    for s in (1, -1):
        group.add(*polyline([end, end-head_len*d+s*head_w*perp], color, radius))
    return group

def cap_with_triangle(axis, at_start=False, length=2.5, width=2.0, normal=UP, color=CHILL_BROWN):
    """Filled triangle on one end of a NumberLine, in the plane perpendicular to `normal`.
    Shortens the line to the triangle's base so the two don't z-fight."""
    p0, p1=axis.get_start(), axis.get_end()
    tip, d=(p0, p0-p1) if at_start else (p1, p1-p0)
    d=d/np.linalg.norm(d)
    perp=np.cross(normal, d)
    base=tip-length*d
    tri=Polygon(tip, base+0.5*width*perp, base-0.5*width*perp)
    tri.set_fill(color, 1.0)
    tri.set_stroke(width=0)
    tri.apply_depth_test()
    if at_start:
        axis.put_start_and_end_on(base, p1)
    else:
        axis.put_start_and_end_on(p0, base)
    return tri

def rgb_image_planes(im_path, opacity=0.4, pitch=channel_pitch):
    """Three copies of the full-color image, one per depth slot inside image_bounds."""
    planes=[]
    w=image_bounds[1]-image_bounds[0]
    h=image_bounds[3]-image_bounds[2]
    for c in range(3):
        img=ImageMobject(im_path)
        img.set_width(w, stretch=True)
        img.set_height(h, stretch=True)
        img.set_opacity(opacity)
        img.move_to([0, 0, image_bounds[4]+(c+0.5)*pitch])
        planes.append(img)
    return planes

class P13_22_book_2(InteractiveScene):
    def construct(self):
        quick_mode=True   #flip to False for the real render

        act=np.load(data_dir+'/p13/lemon_activations_47587.npy', allow_pickle=True).item()
        layer_1_weights=np.load(data_dir+'/p13/plain_8_conv_1.npy')

        # start_position=(15, 52, 0, (np.float32(4.48), np.float32(4.88), np.float32(-6.58)), 106.23)
        # end_position=(61, 73, 0, (np.float32(3.79), np.float32(7.16), np.float32(-1.82)), 94.39)

        start_position=(34, 66, 0, (np.float32(-1.05), np.float32(4.07), np.float32(-5.15)), 111.08)
        end_position=(34, 66, 0, (np.float32(-1.05), np.float32(4.07), np.float32(-5.15)), 111.08)


        thresh=-100
        a=act['conv1'][0]
        temp=a[0].copy()
        a[0]=a[22] #Start with nice vertical edges
        a[22]=temp

        #Static geometry
        image_border=orient(prism(*image_bounds, CHILL_BROWN, line_radius))

        rgb=rgb_image_planes(data_dir+'/p13/lemon.jpg', opacity=1.0)
        img=orient(Group(*reversed(rgb)))
        self.add(img)
        self.remove(img[0]); self.add(img[0])

        # rgb=rgb_image_planes(data_dir+'/p13/lemon.jpg', data_dir+'/p13', opacity=0.75)
        # img=orient(Group(*reversed(rgb)))   #camera sits on the -x side after orient, so blue first
        # self.add(img)

        # img=orient(image_plane(data_dir+'/p13/lemon.jpg', opacity=0.6))

        n_i, n_j=a.shape[1], a.shape[2]

        vmin=float(a[kernel_k].min())
        vmax=float(a[kernel_k].max())

        #bounds (static border)
        _, bounds=conv_data_block(a, spacing_between_layers+1, vmin=vmin, vmax=vmax,
                                  cell_size=block_cell, alpha=0.75)

        # _.set_opacity(0.0)


        conv_1_border=orient(prism(*bounds, CHILL_BROWN, line_radius))

        # self.frame.reorient(32, 66, 0, (np.float32(6.76), np.float32(11.09), np.float32(-0.32)), 106.23)
        # self.add(img)
        self.frame.reorient(90, 90, 0, (np.float32(11.27), np.float32(-0.41), np.float32(0.2)), 65.81)


        self.wait(1)
        self.play(FadeIn(image_border),
                  FadeIn(conv_1_border), 
                  img.animate.set_opacity(0.5),
                  self.frame.animate.reorient(*start_position),
                  run_time=8)
        self.remove(image_border); self.add(image_border)
        self.remove(conv_1_border); self.add(conv_1_border)


        self.wait(1)

        if quick_mode:
            block, _=conv_data_block(a, spacing_between_layers+1, vmin=vmin, vmax=vmax,
                                     keep=reveal_mask(a.shape, n_i-1, n_j-1, kernel_k),
                                     cell_size=block_cell, alpha=0.75)
            orient(block)
            self.add(block)
            self.frame.reorient(*end_position)
            self.wait(0.1)
        else:
            block=None
            kernel=None
            positions=list(np.ndindex(n_i, n_j))
            n_steps=len(positions)
            for step, (i, j) in enumerate(positions):
                last=(step==n_steps-1)
                if step%steps_per_viz!=0 and not last:
                    continue

                swap_out(self, kernel)
                swap_out(self, block)

                kernel=orient(conv1_kernel(i, j, kernel_k, a.shape, layer_1_weights[0],
                                           cell_size=block_cell, stride=1))
                block, _=conv_data_block(a, spacing_between_layers+1, vmin=vmin, vmax=vmax,
                                         keep=reveal_mask(a.shape, i, j, kernel_k),
                                         cell_size=block_cell, alpha=0.75)
                orient(block)
                self.add(block, kernel)

                t=smooth(step/(n_steps-1))   #ease in/out over the whole sweep
                self.frame.reorient(*blend_views(start_position, end_position, t))
                self.wait(1/30)
            swap_out(self, kernel)
        self.wait(still_hold)


        flat_view=(90, 90, 0, (np.float32(32.08), np.float32(202.27), np.float32(-200.76)), 456.79)
        # self.frame.reorient(90, 90, 0, (np.float32(32.15), np.float32(192.87), np.float32(-191.94)), 479.34)
        pitch=n_j*block_cell+4.0     #map width ~26.9 + gap

        #Fade everything but the map; bring the map to full opacity
        self.play(FadeOut(img), FadeOut(image_border), FadeOut(conv_1_border),
                  block.animate.set_opacity(1.0), run_time=2.0)
        # swap_out(self, img); 
        # swap_out(self, image_border); 
        # swap_out(self, conv_1_border)
        # img=image_border=conv_1_border=None

        #The other 63 maps; channel 0 (your swapped-in vertical edge map) stays put as upper-left
        grid=activation_image_grid(data_dir+'/p13/conv_1_activations', a.shape[0],
                                   spacing_between_layers+1,
                                   n_j*block_cell, pitch, skip_channel=kernel_k)
        orient(grid)
        grid.set_opacity(0.0)
        self.add(grid)
        self.play(grid.animate.set_opacity(1.0),
                  self.frame.animate.reorient(*flat_view), run_time=12.0)

        self.wait()

        # SW 9/5, hacking on the book
        # Made it to here, might need to add an extra step here to show a nice "expanded" block. 
        # Rending to here first thoough to start noodling. 


        ## P15 ---- Stack the maps back into a tensor ----
        n_c=a.shape[0]
        z0=spacing_between_layers+1

        #Boook, cranking this up, was 10: 
        spread_step=20*depth_step   #spread stack fills a prism 6x the depth of conv_1_border
        wide_bounds=(bounds[0], bounds[1], bounds[2], bounds[3],
                     z0, z0+n_c*spread_step)

        def orient_point(p):
            x, y, z=p
            return np.array([z, x, y])  #the same permutation orient() applies about ORIGIN

        stack_x, stack_y=-0.5*block_cell, 0.5*block_cell  #channel-0's slot in the grid

        wide_border=orient(prism(*wide_bounds, CHILL_BROWN, line_radius))
        spread_view=(38, 63, 0, (np.float32(37.84), np.float32(7.36), np.float32(-11.49)), 138.34)

        channels=[c for c in range(n_c) if c!=kernel_k]
        gather=[m.animate.move_to(orient_point([stack_x, stack_y, z0+c*spread_step]))
                for m, c in zip(grid, channels)]

        self.wait()
        self.play(LaggedStart(*gather, lag_ratio=0.01),
                  FadeIn(wide_border),
                  FadeIn(img), 
                  FadeIn(image_border),
                  self.frame.animate.reorient(*spread_view),
                  run_time=8.0)
        self.wait(still_hold)

        # Ok, book hacking here.
        # I want do do a wide "record collection" view wiht some middles missing so you can cleary see an etnry or two 
        self.remove(img, image_border)
        self.remove(grid[7:40])
        self.remove(grid[40:])
        for ii in range(len(grid)-1, 40, -1): 
            # print(ii)
            self.add(grid[ii])

        # self.frame.reorient(0, 64, 0, (np.float32(84.89), np.float32(12.3), np.float32(-10.11)), 154.76)
        self.wait(2)

        self.remove(img, image_border)
        self.remove(grid[7:40])
        self.remove(grid[40:])
        for ii in range(len(grid)-1, 40, -1): 
            # print(ii)
            self.add(grid[ii])

        # self.frame.reorient(0, 64, 0, (np.float32(84.89), np.float32(12.3), np.float32(-10.11)), 154.76)
        self.frame.reorient(0, 58, 0, (np.float32(83.37), np.float32(11.93), np.float32(-10.72)), 145.97)
        self.wait(2)




        img=orient(image_plane(data_dir+'/p13/lemon.jpg', opacity=0.4))
        image_border=orient(prism(*image_bounds, CHILL_BROWN, line_radius))
        interlude_view=(38, 68, 0, (10.0, 8.0, -2.0), 120)  #placeholder; tune in embed

        self.wait(1)
        self.remove(grid)
        self.remove(block)
        self.wait(1)
        grid.set_opacity(0.04)
        block.set_opacity(0.04)
        self.play(grid.animate.set_opacity(0.01), block.animate.set_opacity(0.01), run_time=2.0)  #wide_border stays


        # self.play(FadeIn(img), FadeIn(image_border),
        #           self.frame.animate.reorient(*interlude_view), run_time=3.0)
        # self.wait(still_hold)

        ## ---- Flip through R, G, B, then back to color ----
        channel_paths=make_channel_images(data_dir+'/p13/lemon.jpg', data_dir+'/p13')
        current=img

        self.wait()
        for p in channel_paths:
            nxt=orient(image_plane(p, opacity=1.0))
            self.remove(current)
            self.add(nxt)
            # self.play(FadeIn(nxt), FadeOut(current), run_time=1.0)
            swap_out(self, current)
            current=nxt
            self.wait(0.5)
        img=orient(image_plane(data_dir+'/p13/lemon.jpg', opacity=0.4))
        # self.play(FadeIn(img), FadeOut(current), run_time=1.0)
        self.wait()
        self.remove(current)
        self.add(img)
        swap_out(self, current)
        

        ## ---- Bring the activation maps back one at a time ----
        block.set_opacity(1.0)  
        grid.set_opacity(1.0)

        cascade_start=interlude_view  #or self.frame.get_... wherever you are at this point
        cascade_end=(58, 69, 0, (np.float32(40.75), np.float32(13.26), np.float32(-9.89)), 138.34)

        click_hold=0.15   #seconds per map; 65 maps ≈ 10s total
        members=[block, *grid]
        n=len(members)

        self.wait(still_hold)
        for idx, m in enumerate(members):
            self.add(m)
            t=smooth(idx/(n-1))
            # self.frame.reorient(*blend_views(cascade_start, cascade_end, t))
            self.wait(click_hold)
        self.wait(still_hold)

        #Optionally send the image away again before the compress:
        # self.play(FadeOut(img), FadeOut(image_border), run_time=1.5)
        # swap_out(self, img); swap_out(self, image_border)
        # img=image_border=None


        # ## --- P16 --- ##
        # Ok, now we have batch norm and ReLU
        # Kinda feel like we just show these in one step?
        # This is where I want to shift to just showing values above a certain threshold
        # Ok right so act['relu'] is what we want to show here, but 
        # only activations above some threshold, let's start with 0.2
        # Everything below that we just won't plot. 


         ## --- P16: batch norm + ReLU ---
        r=act['relu'][0].copy()
        r[[0, 22]]=r[[22, 0]]   #match the conv1 channel swap

        # relu_thresh=1.0
        # r_vmax=float(r.max())   #global normalization; see note below

        slabs=[]
        for c in range(n_c):
            relu_thresh=np.percentile(r[c:c+1], 95) #SW drop down for whide book view
            # relu_thresh=0
            # print(relu_thresh)
            slab, _=conv_data_block(r[c:c+1], z0+c*spread_step,
                                    vmin=0.0, 
                                    vmax=float(r[c:c+1].max()),
                                    keep=(r[c:c+1]>relu_thresh),
                                    cell_size=block_cell, alpha=0.7)
            orient(slab)
            slabs.append(slab)
        relu_stack=Group(*slabs)

        # ## ---- Crossfade conv1 maps -> thresholded relu slabs ----
        self.play(FadeIn(relu_stack), FadeOut(grid), FadeOut(block),
                  run_time=3.0)
        swap_out(self, grid); swap_out(self, block)
        grid=block=None
        self.wait(still_hold)

        self.remove(img)
        self.frame.reorient(0, 58, 0, (np.float32(83.37), np.float32(11.93), np.float32(-10.72)), 145.97)

        self.wait()


        ## ---- Compress the stack, morphing the prism with it ----
        # end_position_2=(37, 64, 0, (np.float32(16.68), np.float32(18.15), np.float32(-7.34)), 106.72)
        # compress_factor=4                       #try 4x; spread_step/4 = 2.5*depth_step
        # squish_step=spread_step/compress_factor
        # squish_bounds=(bounds[0], bounds[1], bounds[2], bounds[3],
        #                z0, z0+n_c*squish_step)
        # squish_border=orient(prism(*squish_bounds, CHILL_BROWN, line_radius))

        # squeeze=[slab.animate.shift([(squish_step-spread_step)*c, 0, 0])  #orient: z -> world x
        #          for c, slab in enumerate(slabs)]

        # self.wait(1)
        # self.play(*squeeze,
        #           #LaggedStart(*squeeze, lag_ratio=0.005),
        #           Transform(wide_border, squish_border),
        #           self.frame.animate.reorient(*end_position_2),
        #           run_time=6.0)
        # self.wait(still_hold)







        self.embed()

class P13_22_book_3(InteractiveScene):
    def construct(self):
        quick_mode=False   #flip to False for the real render

        act=np.load(data_dir+'/p13/lemon_activations_47587.npy', allow_pickle=True).item()
        layer_1_weights=np.load(data_dir+'/p13/plain_8_conv_1.npy')

        # start_position=(15, 52, 0, (np.float32(4.48), np.float32(4.88), np.float32(-6.58)), 106.23)
        # end_position=(61, 73, 0, (np.float32(3.79), np.float32(7.16), np.float32(-1.82)), 94.39)

        start_position=(34, 66, 0, (np.float32(-1.05), np.float32(4.07), np.float32(-5.15)), 111.08)
        end_position=(34, 66, 0, (np.float32(-1.05), np.float32(4.07), np.float32(-5.15)), 111.08)


        thresh=-100
        a=act['conv1'][0]
        temp=a[0].copy()
        a[0]=a[22] #Start with nice vertical edges
        a[22]=temp

        #Static geometry
        image_border=orient(prism(*image_bounds, CHILL_BROWN, line_radius))

        rgb=rgb_image_planes(data_dir+'/p13/lemon.jpg', opacity=1.0)
        img=orient(Group(*reversed(rgb)))
        self.add(img)
        self.remove(img[0]); self.add(img[0])

        # rgb=rgb_image_planes(data_dir+'/p13/lemon.jpg', data_dir+'/p13', opacity=0.75)
        # img=orient(Group(*reversed(rgb)))   #camera sits on the -x side after orient, so blue first
        # self.add(img)

        # img=orient(image_plane(data_dir+'/p13/lemon.jpg', opacity=0.6))

        n_i, n_j=a.shape[1], a.shape[2]

        vmin=float(a[kernel_k].min())
        vmax=float(a[kernel_k].max())

        #bounds (static border)
        _, bounds=conv_data_block(a, spacing_between_layers+1, vmin=vmin, vmax=vmax,
                                  cell_size=block_cell, alpha=0.75)

        # _.set_opacity(0.0)


        conv_1_border=orient(prism(*bounds, CHILL_BROWN, line_radius))

        # self.frame.reorient(32, 66, 0, (np.float32(6.76), np.float32(11.09), np.float32(-0.32)), 106.23)
        # self.add(img)
        self.frame.reorient(90, 90, 0, (np.float32(11.27), np.float32(-0.41), np.float32(0.2)), 65.81)


        self.wait(1)
        self.play(FadeIn(image_border),
                  FadeIn(conv_1_border), 
                  img.animate.set_opacity(0.5),
                  self.frame.animate.reorient(*start_position),
                  run_time=8)
        self.remove(image_border); self.add(image_border)
        self.remove(conv_1_border); self.add(conv_1_border)


        self.wait(1)

        if quick_mode:
            block, _=conv_data_block(a, spacing_between_layers+1, vmin=vmin, vmax=vmax,
                                     keep=reveal_mask(a.shape, n_i-1, n_j-1, kernel_k),
                                     cell_size=block_cell, alpha=0.75)
            orient(block)
            self.add(block)
            self.frame.reorient(*end_position)
            self.wait(0.1)
        else:
            block=None
            kernel=None
            positions=list(np.ndindex(n_i, n_j))
            n_steps=len(positions)
            for step, (i, j) in enumerate(positions):
                last=(step==n_steps-1)
                if step%steps_per_viz!=0 and not last:
                    continue

                swap_out(self, kernel)
                swap_out(self, block)

                kernel=orient(conv1_kernel(i, j, kernel_k, a.shape, layer_1_weights[0],
                                           cell_size=block_cell, stride=1))
                block, _=conv_data_block(a, spacing_between_layers+1, vmin=vmin, vmax=vmax,
                                         keep=reveal_mask(a.shape, i, j, kernel_k),
                                         cell_size=block_cell, alpha=0.75)
                orient(block)
                self.add(block, kernel)

                t=smooth(step/(n_steps-1))   #ease in/out over the whole sweep
                self.frame.reorient(*blend_views(start_position, end_position, t))
                self.wait(1/30)
            swap_out(self, kernel)
        self.wait(still_hold)


        flat_view=(90, 90, 0, (np.float32(32.08), np.float32(202.27), np.float32(-200.76)), 456.79)
        # self.frame.reorient(90, 90, 0, (np.float32(32.15), np.float32(192.87), np.float32(-191.94)), 479.34)
        pitch=n_j*block_cell+4.0     #map width ~26.9 + gap

        #Fade everything but the map; bring the map to full opacity
        self.play(FadeOut(img), FadeOut(image_border), FadeOut(conv_1_border),
                  block.animate.set_opacity(1.0), run_time=2.0)
        # swap_out(self, img); 
        # swap_out(self, image_border); 
        # swap_out(self, conv_1_border)
        # img=image_border=conv_1_border=None

        #The other 63 maps; channel 0 (your swapped-in vertical edge map) stays put as upper-left
        grid=activation_image_grid(data_dir+'/p13/conv_1_activations', a.shape[0],
                                   spacing_between_layers+1,
                                   n_j*block_cell, pitch, skip_channel=kernel_k)
        orient(grid)
        grid.set_opacity(0.0)
        self.add(grid)
        self.play(grid.animate.set_opacity(1.0),
                  self.frame.animate.reorient(*flat_view), run_time=12.0)

        self.wait()

        # SW 9/5, hacking on the book
        # Made it to here, might need to add an extra step here to show a nice "expanded" block. 
        # Rending to here first thoough to start noodling. 


        ## P15 ---- Stack the maps back into a tensor ----
        n_c=a.shape[0]
        z0=spacing_between_layers+1
        spread_step=10*depth_step   #spread stack fills a prism 6x the depth of conv_1_border
        wide_bounds=(bounds[0], bounds[1], bounds[2], bounds[3],
                     z0, z0+n_c*spread_step)

        def orient_point(p):
            x, y, z=p
            return np.array([z, x, y])  #the same permutation orient() applies about ORIGIN

        stack_x, stack_y=-0.5*block_cell, 0.5*block_cell  #channel-0's slot in the grid

        wide_border=orient(prism(*wide_bounds, CHILL_BROWN, line_radius))
        spread_view=(38, 63, 0, (np.float32(37.84), np.float32(7.36), np.float32(-11.49)), 138.34)

        channels=[c for c in range(n_c) if c!=kernel_k]
        gather=[m.animate.move_to(orient_point([stack_x, stack_y, z0+c*spread_step]))
                for m, c in zip(grid, channels)]

        self.wait()
        self.play(LaggedStart(*gather, lag_ratio=0.01),
                  FadeIn(wide_border),
                  FadeIn(img), 
                  FadeIn(image_border),
                  self.frame.animate.reorient(*spread_view),
                  run_time=8.0)
        self.wait(still_hold)

        # Ok, book hacking here.
        # I want do do a wide "record collection" view wiht some middles missing so you can cleary see an etnry or two 
        self.remove(img, image_border)
        self.remove(grid[7:40])
        self.remove(grid[40:])
        for ii in range(len(grid)-1, 40, -1): 
            # print(ii)
            self.add(grid[ii])

        self.frame.reorient(0, 64, 0, (np.float32(84.89), np.float32(12.3), np.float32(-10.11)), 154.76)
        self.wait(2)




        img=orient(image_plane(data_dir+'/p13/lemon.jpg', opacity=0.4))
        image_border=orient(prism(*image_bounds, CHILL_BROWN, line_radius))
        interlude_view=(38, 68, 0, (10.0, 8.0, -2.0), 120)  #placeholder; tune in embed

        self.wait(1)
        self.remove(grid)
        self.remove(block)
        self.wait(1)
        grid.set_opacity(0.04)
        block.set_opacity(0.04)
        self.play(grid.animate.set_opacity(0.01), block.animate.set_opacity(0.01), run_time=2.0)  #wide_border stays


        # self.play(FadeIn(img), FadeIn(image_border),
        #           self.frame.animate.reorient(*interlude_view), run_time=3.0)
        # self.wait(still_hold)

        ## ---- Flip through R, G, B, then back to color ----
        channel_paths=make_channel_images(data_dir+'/p13/lemon.jpg', data_dir+'/p13')
        current=img

        self.wait()
        for p in channel_paths:
            nxt=orient(image_plane(p, opacity=1.0))
            self.remove(current)
            self.add(nxt)
            # self.play(FadeIn(nxt), FadeOut(current), run_time=1.0)
            swap_out(self, current)
            current=nxt
            self.wait(0.5)
        img=orient(image_plane(data_dir+'/p13/lemon.jpg', opacity=0.4))
        # self.play(FadeIn(img), FadeOut(current), run_time=1.0)
        self.wait()
        self.remove(current)
        self.add(img)
        swap_out(self, current)
        

        ## ---- Bring the activation maps back one at a time ----
        block.set_opacity(1.0)  
        grid.set_opacity(1.0)

        cascade_start=interlude_view  #or self.frame.get_... wherever you are at this point
        cascade_end=(58, 69, 0, (np.float32(40.75), np.float32(13.26), np.float32(-9.89)), 138.34)

        click_hold=0.15   #seconds per map; 65 maps ≈ 10s total
        members=[block, *grid]
        n=len(members)

        self.wait(still_hold)
        for idx, m in enumerate(members):
            self.add(m)
            t=smooth(idx/(n-1))
            # self.frame.reorient(*blend_views(cascade_start, cascade_end, t))
            self.wait(click_hold)
        self.wait(still_hold)

        #Optionally send the image away again before the compress:
        # self.play(FadeOut(img), FadeOut(image_border), run_time=1.5)
        # swap_out(self, img); swap_out(self, image_border)
        # img=image_border=None


        # ## --- P16 --- ##
        # Ok, now we have batch norm and ReLU
        # Kinda feel like we just show these in one step?
        # This is where I want to shift to just showing values above a certain threshold
        # Ok right so act['relu'] is what we want to show here, but 
        # only activations above some threshold, let's start with 0.2
        # Everything below that we just won't plot. 


         ## --- P16: batch norm + ReLU ---
        r=act['relu'][0].copy()
        r[[0, 22]]=r[[22, 0]]   #match the conv1 channel swap

        # relu_thresh=1.0
        # r_vmax=float(r.max())   #global normalization; see note below

        slabs=[]
        for c in range(n_c):
            relu_thresh=np.percentile(r[c:c+1], 97)
            # print(relu_thresh)
            slab, _=conv_data_block(r[c:c+1], z0+c*spread_step,
                                    vmin=0.0, 
                                    vmax=float(r[c:c+1].max()),
                                    keep=(r[c:c+1]>relu_thresh),
                                    cell_size=block_cell, alpha=0.7)
            orient(slab)
            slabs.append(slab)
        relu_stack=Group(*slabs)

        # ## ---- Crossfade conv1 maps -> thresholded relu slabs ----
        self.play(FadeIn(relu_stack), FadeOut(grid), FadeOut(block),
                  run_time=3.0)
        swap_out(self, grid); swap_out(self, block)
        grid=block=None
        self.wait(still_hold)


        ## ---- Compress the stack, morphing the prism with it ----
        end_position_2=(34, 66, 0, (np.float32(-1.05), np.float32(4.07), np.float32(-5.15)), 111.08)
        compress_factor=4                       #try 4x; spread_step/4 = 2.5*depth_step
        squish_step=spread_step/compress_factor
        squish_bounds=(bounds[0], bounds[1], bounds[2], bounds[3],
                       z0, z0+n_c*squish_step)
        squish_border=orient(prism(*squish_bounds, CHILL_BROWN, line_radius))

        squeeze=[slab.animate.shift([(squish_step-spread_step)*c, 0, 0])  #orient: z -> world x
                 for c, slab in enumerate(slabs)]

        self.wait(1)
        self.play(*squeeze,
                  #LaggedStart(*squeeze, lag_ratio=0.005),
                  Transform(wide_border, squish_border),
                  self.frame.animate.reorient(*end_position_2),
                  run_time=6.0)
        self.wait(still_hold)



        #P17, conv 2 let's go. 
        ## --- P17: second layer (layer1.0.conv1) ---
        a2=act['layer1.0.conv1'][0]                    #(64, 56, 56)
        layer_2_weights=np.load(data_dir+'/p13/plain_8_layer1_0_conv1.npy')  #(64, 64, 3, 3)

        cell2=block_cell #0.5*block_cell                           #half the width & height
        step2=squish_step                              #same depth as compressed layer 1
        n_c2, n_i2, n_j2=a2.shape
        z1_end=z0+n_c*squish_step                      #far face of the layer-1 stack
        z2_0=z1_end+spacing_between_layers

        vmin2=float(a2[0].min())
        vmax2=float(a2[0].max())

        _, bounds2=conv_data_block(a2, z2_0, vmin=vmin2, vmax=vmax2,
                                   cell_size=cell2, z_step=step2)
        conv_2_border=orient(prism(*bounds2, CHILL_BROWN, line_radius))
        self.add(conv_2_border)

        # l2_start=(61, 73, 0, (np.float32(30.0), np.float32(5.0), np.float32(-4.0)), 100)  #tune in embed
        # l2_end=(56, 74, 0, (np.float32(11.0), np.float32(10.64), np.float32(-3.53)), 94.75)
        self.frame.reorient(34, 66, 0, (np.float32(-1.05), np.float32(4.07), np.float32(-5.15)), 111.08)
        l2_end=(33, 63, 0, (np.float32(2.45), np.float32(5.69), np.float32(-6.56)), 111.08)


        ## ---- Sliding kernel sweep, channel 0 ----
        self.wait()
        # quick_mode=True
        img.set_opacity(0.5)
        self.add(image_border)
        self.add(img)

        if quick_mode:
            block2, _=conv_data_block(a2, z2_0, vmin=vmin2, vmax=vmax2,
                                      keep=reveal_mask(a2.shape, n_i2-1, n_j2-1, 0),
                                      alpha=0.9,
                                      cell_size=cell2, z_step=step2)
            orient(block2)
            self.add(block2)
            self.frame.reorient(*l2_end)
            self.wait(0.1)
        else:
            block2=None
            kernel2=None
            positions=list(np.ndindex(n_i2, n_j2))
            n_steps=len(positions)
            for step, (i, j) in enumerate(positions):
                last=(step==n_steps-1)
                if step%steps_per_viz!=0 and not last:
                    continue

                swap_out(self, kernel2)
                swap_out(self, block2)

                kernel2=orient(conv2_kernel(i, j, a2.shape, n_j, block_cell, z0, z1_end,
                                            squish_step, layer_2_weights[0],
                                            cell_size=cell2, dst_z=z2_0, show_weights=False))
                block2, _=conv_data_block(a2, z2_0, vmin=vmin2, vmax=vmax2,
                                          keep=reveal_mask(a2.shape, i, j, 0),
                                          alpha=0.9,
                                          cell_size=cell2, z_step=step2)
                orient(block2)
                self.add(block2, kernel2)

                t=smooth(step/(n_steps-1))
                self.frame.reorient(*blend_views(end_position_2, l2_end, t))
                self.wait(1/30)
            swap_out(self, kernel2)
        self.wait(still_hold)


        # ## ---- Stack the remaining 63 maps (imshow images, clicked in) ----
        # stack2=activation_image_stack(data_dir+'/p13/conv_2_activations', n_c2,
        #                               z2_0, n_j2*cell2, step2, skip_channel=0)
        # orient(stack2)
        # for m in stack2:
        #     self.add(m)
        #     self.wait(0.08)
        # self.wait(still_hold)


        # ## ---- ReLU: crossfade to thresholded slabs ----
        # r2=act['layer1.0.relu'][0]     #check this key against your hook names
        # slabs2=[]
        # self.wait()
        # for c in range(n_c2):
        #     relu_thresh=np.percentile(r2[c:c+1], 97)
        #     slab, _=conv_data_block(r2[c:c+1], z2_0+c*step2,
        #                             vmin=0.0,
        #                             vmax=float(r2[c:c+1].max()),
        #                             keep=(r2[c:c+1]>relu_thresh),
        #                             cell_size=cell2, alpha=0.7)
        #     orient(slab)
        #     slabs2.append(slab)
        # relu_stack2=Group(*slabs2)

        # self.play(FadeIn(relu_stack2), FadeOut(stack2), FadeOut(block2), run_time=3.0)
        # swap_out(self, stack2); swap_out(self, block2)
        # stack2=block2=None
        # self.wait(still_hold)


        # self.play(self.frame.animate.reorient(31, 62, 0, (np.float32(17.28), np.float32(12.32), np.float32(-7.73)), 107.89), 
        #              run_time=6)
        # self.wait()



        # ## --- P18: layer1.0 through layer3.0, one camera move ---
        # base_depth=n_c*squish_step               #layer-1 compressed depth, ≈20 world units
        # depth_mults={64: 1.0, 128: 1.35, 256: 1.8}
        # # depth_mults={64: 1.0, 128: 1.5, 256: 3.0}   #the calmer version

        # deep_keys=['layer1.0', 'layer2.0.relu', 'layer2.0', 'layer3.0.relu', 'layer3.0']

        # z_cursor=z2_0+n_c2*step2+spacing_between_layers
        # deep_blocks, deep_borders, deep_bounds=[], [], []

        # for key in deep_keys:
        #     rl=act[key][0]
        #     n_cl=rl.shape[0]
        #     z_step_l=base_depth*depth_mults[n_cl]/n_cl
        #     blk, bnds=relu_viz_block(rl, z_cursor, z_step_l, cell2)
        #     orient(blk)
        #     border=orient(prism(*bnds, CHILL_BROWN, line_radius))
        #     deep_blocks.append(blk)
        #     deep_borders.append(border)
        #     deep_bounds.append(bnds)
        #     z_cursor=bnds[5]+spacing_between_layers

        # fades=[AnimationGroup(FadeIn(b), FadeIn(p))
        #        for b, p in zip(deep_blocks, deep_borders)]
        # deep_view=(28, 64, 0, (np.float32(130.97), np.float32(-14.4), np.float32(7.38)), 133.84)
        
        # self.wait(1)
        # self.play(LaggedStart(*fades, lag_ratio=0.25),
        #           self.frame.animate.reorient(*deep_view),
        #           run_time=10.0)
        # self.wait(still_hold)


        # #Zoom in on final block to show dimensions and pooling
        # self.play(self.frame.animate.reorient(54, 71, 0, (np.float32(196.61), np.float32(-12.14), np.float32(7.1)), 46.97), run_time=5.0)
        # self.wait()

        # #Ok now I think we show the downsamplign process in place. 
        # pool_factor=5                        #true value is 14; adjustable
        # r3=act['layer3.0'][0]                #(256, 14, 14)
        # p=act['avgpool'][0,:,0,0]            #(256,)

        # #Recreate exactly the normalization/threshold the layer3.0 block used
        # vmax3=r3.max(axis=(1,2), keepdims=True); vmax3[vmax3==0]=1.0
        # rn3=r3/vmax3
        # keep3=rn3>np.percentile(rn3, 97, axis=(1,2), keepdims=True)

        # n_c3, n_i3, n_j3=r3.shape
        # z3_step=base_depth*depth_mults[n_c3]/n_c3
        # z3_0=deep_bounds[-1][4]
        # pooled_cell=n_j3*cell2/pool_factor

        # #Target: every surviving voxel of channel c converges on channel c's pooled slot,
        # #shrinking to the pooled cell size and tinting toward the pooled color
        # kk3=np.meshgrid(np.arange(n_c3), np.arange(n_i3), np.arange(n_j3), indexing='ij')[0]
        # ch=kk3[keep3]
        # pn=p/p.max()
        # tgt_centers=np.stack([np.zeros(len(ch)), np.zeros(len(ch)), z3_0+ch*z3_step], axis=-1)
        # tgt_rgba=viridis(pn[ch]); tgt_rgba[:,3]=0.7
        # pool_target=orient(VoxelBlock(tgt_centers,
        #                               np.array([pooled_cell, pooled_cell, cell_depth]),
        #                               tgt_rgba))

        # hp=0.5*pooled_cell
        # pool_bounds=(-hp, hp, -hp, hp, z3_0, z3_0+n_c3*z3_step)
        # pool_border=orient(prism(*pool_bounds, CHILL_BROWN, line_radius))

        # self.wait(1)
        # self.play(Transform(deep_blocks[-1], pool_target),
        #           Transform(deep_borders[-1], pool_border),
        #           run_time=3.0)

        # #Swap the pile of coincident voxels for one clean voxel per channel
        # pool_block, _=conv_data_block(pn.reshape(n_c3, 1, 1), z3_0, vmin=0.0, vmax=1.0,
        #                               cell_size=pooled_cell, alpha=0.7, z_step=z3_step)
        # orient(pool_block)
        # self.add(pool_block)
        # swap_out(self, deep_blocks[-1])
        # deep_blocks[-1]=pool_block
        # self.add(pool_border)                                     #Transform mutated the old border in place; keep refs tidy
        # self.wait(still_hold)


        # #Now FC layer!
        # ## --- : fc, vertical ---
        # fc_factor=1.2                          #fc height / avgpool length; adjustable
        # fc=act['fc'][0]                        #(1000,) logits
        # n_fc=len(fc)
        # fcn=(fc-fc.min())/np.ptp(fc)           #logits go negative; min-max for viridis

        # pool_len=n_c3*z3_step
        # fc_height=fc_factor*pool_len           #≈96 world units at 1.2
        # fc_step=fc_height/n_fc
        # fc_z=z3_0+pool_len+spacing_between_layers   #un-oriented depth of the fc column

        # def fc_slot_centers(idx):
        #     y=(0.5*(n_fc-1)-idx)*fc_step       #index 0 at top
        #     return np.stack([np.zeros(len(idx)), y, np.full(len(idx), fc_z)], axis=-1)

        # wafer=np.array([pooled_cell, cell_depth, pooled_cell])  #thin slice per unit, now stacked in y

        # #Morph target: the 256 pooled voxels spread to 256 sampled slots along the column
        # # seed_idx=np.round(np.linspace(0, n_fc-1, n_c3)).astype(int)
        # seed_idx=np.round(np.linspace(n_fc-1, 0, n_c3)).astype(int)   #channel 0 -> bottom slot
        # seed_rgba=viridis(fcn[seed_idx]); seed_rgba[:,3]=0.7
        # fc_seed=orient(VoxelBlock(fc_slot_centers(seed_idx), wafer, seed_rgba))

        # hp=0.5*pooled_cell
        # hh=0.5*fc_height

        # fc_view=(2, 80, 0, (np.float32(218.33), np.float32(8.92), np.float32(-0.35)), 60.67) #tune in embed


        # pool_ghost=deep_blocks[-1].copy()
        # border_ghost=deep_borders[-1].copy()
        # self.add(pool_ghost, border_ghost)

        # pivot=np.array([z3_0+pool_len+0.5*spacing_between_layers, 0, 0])  #world x between the columns

        # # fc_border=orient(prism(-hp, hp, -hh, hh, fc_z-hp, fc_z+hp, CHILL_BROWN, line_radius))
        # fc_border=deep_borders[-1].copy()
        # fc_border.rotate(-90*DEGREES, UP, about_point=pivot)
        # fc_border.stretch(fc_factor, 2)              #lengthen along world z, about its center
        # fc_border.move_to([fc_z, 0, 0])              #fc column center in world coords

        
        # # arc=swing_path(pivot, 90*DEGREES, axis=UP)   #flip sign if it dives instead of lifts
        # # arc=swing_path(pivot, -90*DEGREES, axis=UP)
        # arc=swing_path(pivot, -90*DEGREES, axis=UP)

        # self.wait(1.0)
        # self.play(Transform(pool_ghost, fc_seed, path_func=arc),
        #           Transform(border_ghost, fc_border, path_func=arc),
        #           self.frame.animate.reorient(*fc_view), run_time=3.0)

        # #Densify: fade in the full 1000, retire the 256 seeds -> Sw, yeah that makes sense
        # full_rgba=viridis(fcn); full_rgba[:,3]=0.7
        # fc_block=orient(VoxelBlock(fc_slot_centers(np.arange(n_fc)), wafer, full_rgba))
        # self.play(FadeIn(fc_block), FadeOut(pool_ghost), run_time=1.5)
        # swap_out(self, pool_ghost)
        # fc_border_ref=border_ghost   #this mobject IS the fc border now
        # self.wait(still_hold)




        # #Logits plot
        # ## --- Logit plot: index axis down, value axis right ---
        # fc_gap=12.0                    #fc column -> index axis; must exceed |fc.min()|*logit_scale or the negatives cross it
        # ## --- Logit plot: Axes with index pointing down, value axis spanning -/+ ---
        # plot_w=11.0                                   #world units for fc.max()
        # logit_unit=plot_w/fc.max()
        # x_pos=0.7*fc.max()                            #value axis extent to the right of zero
        # x_neg=x_pos                                   #and to the left; shrink this if it runs into the fc column
        # y_max=1.05*n_fc

        # axes=Axes(x_range=(-x_neg, x_pos, x_pos), y_range=(0, y_max, y_max),
        #           width=(x_neg+x_pos)*logit_unit, height=y_max*fc_step,
        #           axis_config=dict(stroke_width=8, include_ticks=False, include_tip=False))
        # axes.set_color(CHILL_BROWN)
        # for axis in axes.get_axes():
        #     axis.set_scale_stroke_with_zoom(True)
        #     axis.apply_depth_test()

        # axes.rotate(-90*DEGREES, RIGHT, about_point=ORIGIN)   #+y -> -z: index 0 at the top
        # axis_x=fc_z+fc_gap
        # top_z=0.5*(n_fc-1)*fc_step
        # axes.shift(np.array([axis_x, 0, top_z])-axes.c2p(0, 0))


        # o=axes.c2p(0, 0)
        # plot_step=(o-axes.c2p(0, 1))[2]        #world z per index on the plot; should equal fc_step
        # # print(fc_step, plot_step)
        # # print(fc_step/plot_step)
        # axes.stretch(1.02, 2, about_point=o)   #dim 2 = world z after the rotate




        # #Curve points and the origin, before the caps shorten the lines
        # curve_pts=axes.c2p(fc, np.arange(n_fc))
        # o=axes.c2p(0, 0)

        # tips=VGroup(cap_with_triangle(axes.x_axis, at_start=True, length=1.0, width=0.85),   #negative end
        #             cap_with_triangle(axes.x_axis, length=1.0, width=0.85),                  #positive end
        #             cap_with_triangle(axes.y_axis, length=1.0, width=0.85))                  #bottom of the index axis


        # curve=VMobject()
        # curve.set_points_as_corners(axes.c2p(fc, np.arange(n_fc)))
        # curve.set_stroke(width=6, opacity=1.0)
        # curve.set_scale_stroke_with_zoom(True)
        # curve.apply_depth_test()
        # curve.set_joint_type('bevel')

        # #Same min-max -> viridis map as fc_block, read back off the point positions
        # o=axes.c2p(0, 0)
        # pts=curve.get_points()
        # v=((pts[:,0]-o[0])/logit_unit-fc.min())/np.ptp(fc)
        # rgba=viridis(v)
        # rgba[:,3]=1.0
        # curve.data['stroke_rgba'][:]=rgba


        # # self.add(axes, tips, curve)
        # # self.remove(axes, tips, curve)

        # # self.frame.reorient(0, 90, 0, (np.float32(214.91), np.float32(8.72), np.float32(-0.68)), 60.67)
        # plot_view=(0, 90, 0, (np.float32(222.89), np.float32(8.72), np.float32(-0.51)), 60.67)


        # self.wait()
        # self.play(ShowCreation(axes), FadeIn(tips), 
        #     self.frame.animate.reorient(*plot_view), 
        #     run_time=2.0)
        # self.play(ShowCreation(curve), run_time=5.0)
        # self.wait()


        # self.play(FadeOut(axes), 
        #           FadeOut(curve),
        #           FadeOut(tips),
        #           self.frame.animate.reorient(0, 59, 0, (np.float32(105.79), np.float32(12.11), np.float32(-34.57)), 186.07),
        #           run_time=8)


        # #P20
        # #Ok ok ok ok now add 4 example images, accuracy on the bottom
        # #I thin doing this in illustrator is probably the move, just some chill fade-ins should be ok
        # self.wait(1)

        # # P22
        # # Alrighy now we're going to add 6 layers to get to Plain 18
        # # I think we can animated this nicely. 
        # act14=np.load(data_dir+'/p22/lemon_activations_47587_plain14.npy', allow_pickle=True).item()
        # # print(act14.keys())

        # layer4_depth_mult=1.2                         #layer4 (512 ch) depth relative to layer3 (256 ch); true ratio is 2
        # depth_mults[512]=depth_mults[256]*layer4_depth_mult

        # def deep_layer(rl, z_start, depth):
        #     """Thresholded block + border for one layer, `depth` world units deep."""
        #     blk, bnds=relu_viz_block(rl, z_start, depth/rl.shape[0], cell2)
        #     orient(blk)
        #     border=orient(prism(*bnds, CHILL_BROWN, line_radius))
        #     return blk, border, bnds

        # swap_out(self, pool_border)                   #coincident duplicate of deep_borders[-1] from the pooling step
        # pool_border=None

        # #1. Zoom out
        # # wide_view=(1, 60, 0, (np.float32(132.95), np.float32(-18.99), np.float32(-7.27)), 251.41)

        # # self.wait()
        # # self.play(self.frame.animate.reorient(*wide_view), run_time=4.0)
        # # self.wait()

        # #2. Slide open two slots after layer1.0, fade in layer1.1.relu and layer1.1
        # gap1=2*(base_depth+spacing_between_layers)    #64-channel layers are base_depth deep
        # downstream=[*deep_blocks[1:], *deep_borders[1:], fc_block, fc_border_ref]

        # self.wait()
        # self.play(*[m.animate.shift([gap1, 0, 0]) for m in downstream], 
        #             self.frame.animate.reorient(0, 59, 0, (np.float32(136.24), np.float32(-16.45), np.float32(-2.75)), 230.32),
        #             run_time=3.0)   #orient: depth z -> world x
        # z3_0+=gap1
        # fc_z+=gap1

        # z_cursor=deep_bounds[0][5]+spacing_between_layers
        # new1=[]
        # for key in ['layer1.1.relu', 'layer1.1']:
        #     blk, border, bnds=deep_layer(act14[key][0], z_cursor, base_depth)
        #     new1.append((blk, border))
        #     z_cursor=bnds[5]+spacing_between_layers
        # self.play(LaggedStart(*[AnimationGroup(FadeIn(b), FadeIn(p)) for b, p in new1], lag_ratio=0.3),
        #           run_time=3.0)

        # #3.
        # self.wait(still_hold)

        # #4. Un-pool layer3.0 back to 14x14 while opening four slots for layer4
        # depth4=base_depth*depth_mults[512]
        # gap4=4*(depth4+spacing_between_layers)

        # blk3, border3, bounds3=deep_layer(act['layer3.0'][0], z3_0, base_depth*depth_mults[256])
        # collapsed=orient(VoxelBlock(tgt_centers+np.array([0, 0, gap1]),          #same voxels as blk3, piled in the pooled slots
        #                             np.array([pooled_cell, pooled_cell, cell_depth]), tgt_rgba))
        # self.add(collapsed)
        # swap_out(self, deep_blocks[-1])               #the clean one-voxel-per-channel column
        # deep_blocks[-1]=collapsed

        # self.play(Transform(collapsed, blk3),
        #           Transform(deep_borders[-1], border3),
        #           fc_block.animate.shift([gap4, 0, 0]),
        #           fc_border_ref.animate.shift([gap4, 0, 0]),
        #           self.frame.animate.reorient(0, 60, 0, (np.float32(227.84), np.float32(-12.27), np.float32(1.5)), 293.88),
        #           run_time=4.0)
        # fc_z+=gap4

        # #Retire the morphed stand-ins for the real block and border
        # self.add(blk3, border3)
        # swap_out(self, collapsed)
        # swap_out(self, deep_borders[-1])
        # deep_blocks[-1]=blk3
        # deep_borders[-1]=border3
        # self.wait(still_hold)

        # #5. layer4.0.relu through layer4.1
        # z_cursor=bounds3[5]+spacing_between_layers
        # new4=[]
        # for key in ['layer4.0.relu', 'layer4.0', 'layer4.1.relu', 'layer4.1']:
        #     blk, border, bnds=deep_layer(act14[key][0], z_cursor, depth4)
        #     new4.append((blk, border))
        #     z_cursor=bnds[5]+spacing_between_layers
        # self.play(LaggedStart(*[AnimationGroup(FadeIn(b), FadeIn(p)) for b, p in new4], lag_ratio=0.25),
        #           run_time=5.0)
        # self.wait(still_hold)

        # #Ok fun little pan around, probably won't use it, but might be a fun b-roll. 
        # self.wait()
        # self.play(self.frame.animate.reorient(47, 64, 0, (np.float32(321.23), np.float32(36.1), np.float32(-33.87)), 215.92), 
        #           run_time=10.0)



        # self.wait(20)
        # self.embed()



        # Plain8
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


        # Plain14
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
        # layer1.1.conv1 (1, 64, 56, 56)
        # layer1.1.bn1 (1, 64, 56, 56)
        # layer1.1.relu (1, 64, 56, 56)
        # layer1.1.conv2 (1, 64, 56, 56)
        # layer1.1.bn2 (1, 64, 56, 56)
        # layer1.1 (1, 64, 56, 56)
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
        # layer4.0.conv1 (1, 512, 7, 7)
        # layer4.0.bn1 (1, 512, 7, 7)
        # layer4.0.relu (1, 512, 7, 7)
        # layer4.0.conv2 (1, 512, 7, 7)
        # layer4.0.bn2 (1, 512, 7, 7)
        # layer4.0 (1, 512, 7, 7)
        # layer4.1.conv1 (1, 512, 7, 7)
        # layer4.1.bn1 (1, 512, 7, 7)
        # layer4.1.relu (1, 512, 7, 7)
        # layer4.1.conv2 (1, 512, 7, 7)
        # layer4.1.bn2 (1, 512, 7, 7)
        # layer4.1 (1, 512, 7, 7)
        # avgpool (1, 512, 1, 1)
        # fc (1, 1000)
        # image (224, 224, 3)



 






        # #Sweep
        # block=None
        # kernel=None
        # positions=list(np.ndindex(n_i, n_j))
        # n_steps=len(positions)
        # for step, (i, j) in enumerate(positions):
        #     last=(step==n_steps-1)
        #     if step%steps_per_viz!=0 and not last:
        #         continue

        #     swap_out(self, kernel)
        #     swap_out(self, block)

        #     kernel=orient(conv1_kernel(i, j, kernel_k, a.shape, layer_1_weights[0],
        #                                cell_size=block_cell, stride=1))
        #     block, _=conv_data_block(masked_conv1(a, i, j, kernel_k),
        #                              spacing_between_layers+1, 0.005, vmax=vmax,
        #                              cell_size=block_cell)
        #     orient(block)
        #     self.add(block, kernel)

        #     t=smooth(step/(n_steps-1))   #ease in/out over the whole sweep
        #     self.frame.reorient(*blend_views(start_position, end_position, t))
        #     self.wait(1/30)

        # swap_out(self, kernel)       #drop the kernel at the end, leave the filled map
        # self.wait(still_hold)
        



# class P13(InteractiveScene):
#     def construct(self):

        
#         act=np.load(data_dir+'/p13/lemon_activations_47587.npy', allow_pickle=True).item()
#         layer_1_weights=np.load(data_dir+'/p13/plain_8_conv_1.npy')


#         ##Ok lets start simple with just the image here.
#         image_border=prism(*image_bounds, CHILL_BROWN, line_radius)
#         img=image_plane(data_dir+'/p13/lemon.jpg', opacity=0.4)


#         kernel_k=0 #20 is nice vertical edges I think, maybe just transpose 1 and 20

#         a=act['conv1'][0]

#         block, bounds=conv_data_block(masked_conv1(a, 0, 10, kernel_k),
#                                       spacing_between_layers+1, 0.005, cell_size=0.48)
#         conv_1_border=prism(*bounds, CHILL_BROWN, line_radius)

#         cell=(image_bounds[1]-image_bounds[0])/a.shape[-1]
#         k=conv1_kernel(0, 10, kernel_k, a.shape, layer_1_weights[0], cell_size=0.48, stride=1)


#         # self.add(conv1_kernel(0, 10, kernel_k, a.shape, weights, forward))
#         net_group=Group(img, image_border, conv_1_border, k)
#         net_group.rotate(90*DEGREES, [0, 1, 0])
#         net_group.rotate(90*DEGREES, [1, 0, 0])
#         self.add(net_group)

#         # image_border.move_to([0, 10, 0])

#         self.frame.reorient(32, 66, 0, (np.float32(6.76), np.float32(11.09), np.float32(-0.32)), 106.23)


#         self.wait()





        # self.camera.background_rgba=[0, 0, 0, 1]
        # self.frame.set_field_of_view(fov)
        # a=load_activation('hot_dog', 'features_2')
        # weights=load_activation('weights', 'features0')[0]
        # vmax=float(a[0].max())

        # self.frame.reorient(*p24_end)
        # forward=forward_from(p24_end)
        # self.add(prism(*image_bounds, WHITE, line_radius))
        # self.add(conv1_kernel(0, 10, kernel_k, a.shape, weights, forward))
        # block, bounds=conv_data_block(masked_conv1(a, 0, 10, kernel_k),
        #                               spacing_between_layers+1, 0.005, forward)
        # self.add(block)
        # self.add(prism(*bounds, WHITE, line_radius))
        # self.wait(still_hold)


        self.wait(20)
        self.embed()





