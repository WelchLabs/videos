# One conv-1 kernel: wired to its activation, then walked across all 55x55 positions.
# The whole-network wireframe shots of p21-p24 are in p21_24.py.

from manimlib import *
import numpy as np
import matplotlib.pyplot as plt
import moderngl

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

data_dir='/Volumes/PG Work/Stephencwelch Dropbox/Pranav Gundu/Welch Labs/videos/_2026/resnets/data'

spacing_between_layers=5
line_radius=0.18
depth_step=0.125
cell_depth=0.1
pixel_dim=0.5

image_opacity=0.936  #Measured: what the old three-deep voxel slab composited to
still_hold=1.0
steps_per_viz=11     #p24c/p24d step every 11th of the 55*55 kernel positions
kernel_k=0
fov=PI/3

image_bounds=(-32.0, 32.0, -32.0, 32.0, 0.0, 3*pixel_dim)

#Camera keyframes: theta, phi, gamma, center, height
p24_end=(-135.4168, 63.3448, 114.4748, (0.0, 0.0, 0.0), 96.224)
p24d_end=(-139.5221, 31.6032, 134.9423, (0.0, 0.0, 0.0), 96.224)


viridis_lut=plt.get_cmap('viridis')(np.linspace(0, 1, 256))


def viridis(values):
    x=np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    idx=np.clip((x*256).astype(np.int32), 0, 255) #matplotlib bins with int(x*N), not round(x*(N-1))
    return viridis_lut[idx]


def load_activation(tag, name):
    return np.load(data_dir+'/activations/'+tag+'/'+name+'.npy')


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


def conv_data_block(a, start_depth, thresh, view_forward, vmax=None):
    a=np.asarray(a, dtype=np.float64)
    n_c, n_i, n_j=a.shape
    kk, ii, jj=np.meshgrid(np.arange(n_c), np.arange(n_i), np.arange(n_j), indexing='ij')
    vals=a/(a.max() if vmax is None else vmax) #vmax keeps p24d's scale stable
    keep=vals>=thresh

    half=np.floor(n_j/2)
    centers=np.stack([jj[keep]-half, -ii[keep]+half, depth_step*kk[keep]+start_depth], axis=-1)
    rgba=viridis(vals[keep])
    rgba[:,3]=0.5

    block=VoxelBlock(centers, np.array([1.0, 1.0, cell_depth]), rgba, view_forward)
    bounds=(-half-0.5, half+0.5, -half-0.5, half+0.5, start_depth, n_c*depth_step+start_depth)
    return block, bounds


def image_plane(tag, opacity):
    img=ImageMobject(data_dir+'/activations/'+tag+'/im.png')
    img.set_width(image_bounds[1]-image_bounds[0], stretch=True)
    img.set_height(image_bounds[3]-image_bounds[2], stretch=True)
    img.set_opacity(opacity)
    img.move_to([0, 0, 0.5*(image_bounds[4]+image_bounds[5])])
    return img


def kernel_weights_block(weights, extent, view_forward):
    """A conv-1 filter painted into its patch, black through to magenta."""
    w=np.asarray(weights, dtype=np.float64)
    w=w-w.min()
    min_x, max_x, _, max_y=extent
    step=(max_x-min_x)/w.shape[1]

    kk, ii, jj=np.meshgrid(np.arange(w.shape[0]), np.arange(w.shape[1]), np.arange(w.shape[2]),
                           indexing='ij')
    vals=(w/w.max()).ravel()
    centers=np.stack([jj.ravel()*step+min_x, -ii.ravel()*step+max_y,
                      kk.ravel()*pixel_dim], axis=-1)
    rgba=np.zeros((len(vals), 4))
    rgba[:,0]=vals
    rgba[:,2]=vals
    rgba[:,3]=0.5
    return VoxelBlock(centers, np.array([1.0, 1.0, depth_step]), rgba, view_forward)


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


def conv1_kernel(i, j, k, a_shape, weights, view_forward):
    """The conv-1 kernel wired to its activation, with the filter painted into the patch."""
    min_x, max_x, min_y, _, min_z, max_z=image_bounds
    group=Group()

    step=(max_x-min_x)/a_shape[1]
    dst_x=j-np.floor(a_shape[-1]/2)
    dst_y=-i+np.floor(a_shape[-1]/2)
    dst_z=depth_step*k+spacing_between_layers+1
    size=8.0

    #-min_y spells max_y; the bounds are symmetric
    connectors=[
        [(dst_x-0.5, dst_y-0.5, dst_z), (min_x+step*j, -min_y-step*i-size, max_z)],
        [(dst_x+0.5, dst_y-0.5, dst_z), (min_x+step*j+size, -min_y-step*i-size, max_z)],
        [(dst_x-0.5, dst_y+0.5, dst_z), (min_x+step*j, -min_y-step*i, max_z)],
        [(dst_x+0.5, dst_y+0.5, dst_z), (min_x+step*j+size, -min_y-step*i, max_z)],
    ]
    for cc in connectors:
        group.add(polyline(cc, MAGENTA, line_radius))
    for p, q in [(0, 1), (1, 3), (3, 2), (2, 0)]:
        group.add(polyline([connectors[p][0], connectors[q][0]], MAGENTA, line_radius))

    extent=(min_x+step*j, min_x+step*j+size, -min_y-step*i-size, -min_y-step*i)
    group.add(prism(extent[0], extent[1], extent[2], extent[3], min_z, max_z, MAGENTA,
                    line_radius))
    group.add(kernel_weights_block(weights, extent, view_forward))
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


def forward_from(view):
    """The direction the camera looks, for the painter's-algorithm sort."""
    from scipy.spatial.transform import Rotation
    theta, phi, gamma=np.radians(view[:3])
    return -Rotation.from_euler('zxz', [gamma, phi, theta]).as_matrix()[:,2]


class P24B_D(InteractiveScene):
    def construct(self):
        self.camera.background_rgba=[0, 0, 0, 1]
        self.frame.set_field_of_view(fov)
        a=load_activation('hot_dog', 'features_2')
        weights=load_activation('weights', 'features0')[0]
        vmax=float(a[0].max())

        self.frame.reorient(*p24_end)
        forward=forward_from(p24_end)
        self.add(prism(*image_bounds, WHITE, line_radius))
        self.add(conv1_kernel(0, 10, kernel_k, a.shape, weights, forward))
        block, bounds=conv_data_block(masked_conv1(a, 0, 10, kernel_k),
                                      spacing_between_layers+1, 0.005, forward)
        self.add(block)
        self.add(prism(*bounds, WHITE, line_radius))
        self.wait(still_hold)

        num_steps=55**2+1
        for move_camera in [False, True]:
            clear_scene(self)
            step_num=0
            current=None
            for i in range(55):
                for j in range(55):
                    step_num+=1
                    if step_num!=1 and step_num%steps_per_viz!=0:
                        continue
                    view=(blend_views(p24_end, p24d_end, step_num/(num_steps-1))
                          if move_camera else p24_end)
                    fwd=forward_from(view)

                    shot=Group()
                    shot.add(image_plane('hot_dog', image_opacity))
                    shot.add(prism(*image_bounds, WHITE, line_radius))
                    shot.add(conv1_kernel(i, j, kernel_k, a.shape, weights, fwd))
                    block, bounds=conv_data_block(masked_conv1(a, i, j, kernel_k),
                                                  spacing_between_layers+1, 0.005, fwd, vmax)
                    shot.add(block)
                    shot.add(prism(*bounds, WHITE, line_radius))

                    swap_out(self, current)
                    self.add(shot)
                    current=shot
                    self.frame.reorient(*view)
                    self.wait(1/30)
