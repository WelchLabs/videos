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

data_dir='/Volumes/PG Work/Stephencwelch Dropbox/Pranav Gundu/Welch Labs/videos/_2026/resnet/data'

cell_depth=0.1

grid_sparse_thresh=0.1  #p33 grid/line animations
grid_tile=55
grid_spacing=1
fov=PI/3


#Camera keyframes: theta, phi, gamma, center, height
side_view=(-108.4402, 83.9753, 92.0043, (5.3775, -5.9592, 46.6544), 113.816)
p33_start=(0.0, 0.0, 0.0, (0.0, 0.0, 0.0), 591.468)

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


def activation_layer(a_scaled, center_point, grid_dim, thresh, opacity, view_forward):
    n_i, n_j=a_scaled.shape
    ii, jj=np.meshgrid(np.arange(n_i), np.arange(n_j), indexing='ij')
    keep=a_scaled>=thresh

    half=np.floor(grid_dim/2) #Offset comes from the enclosing stack, not this map
    cx, cy, cz=center_point
    centers=np.stack([jj[keep]-half+cx, -ii[keep]+half+cy,
                      np.full(int(keep.sum()), cz)], axis=-1)
    rgba=viridis(a_scaled[keep])
    rgba[:,3]=opacity
    return VoxelBlock(centers, np.array([1.0, 1.0, cell_depth]), rgba, view_forward)


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


def blend_views(a, b, t):
    return (a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t,
            tuple(p+(q-p)*t for p, q in zip(a[3], b[3])), a[4]+(b[4]-a[4])*t)


def forward_from(view):
    """The direction the camera looks, for the painter's-algorithm sort."""
    from scipy.spatial.transform import Rotation
    theta, phi, gamma=np.radians(view[:3])
    return -Rotation.from_euler('zxz', [gamma, phi, theta]).as_matrix()[:,2]


def grid_center_points():
    """The 8x8 layout the 64 maps start from, at z=30."""
    pts=[]
    for i in range(8):
        for j in range(8):
            pts.append([j*(grid_tile+grid_spacing)-4*(grid_tile+grid_spacing),
                        -i*(grid_tile+grid_spacing)+4*(grid_tile+grid_spacing), 30.0])
    return np.array(pts, dtype=np.float64)


def line_center_points(min_z, max_z):
    return np.vstack((np.zeros(64), np.zeros(64), np.linspace(min_z, max_z, 64))).T


def activation_maps(a, center_points, with_outline, view_forward):
    """The 64 conv-1 maps, each on a flat tile in the colormap's zero colour."""
    group=Group()
    a_max=a.max()
    background=viridis(np.array([0.0]))[0].copy()
    background[3]=0.95

    for idx in range(a.shape[0]):
        cp=center_points[idx]
        group.add(VoxelBlock(np.array([cp]), np.array([grid_tile, grid_tile, cell_depth]),
                             background[None,:], view_forward))
        group.add(activation_layer(a[idx]/a_max, cp, a.shape[-1], grid_sparse_thresh, 0.95,
                                   view_forward))
        if with_outline:
            half=grid_tile/2 #Closed, but as four separate segments so the corners do not miter
            group.add(polyline([(cp[0]-half, cp[1]-half, cp[2]),
                                (cp[0]-half, cp[1]+half, cp[2]),
                                (cp[0]+half, cp[1]+half, cp[2]),
                                (cp[0]+half, cp[1]-half, cp[2]),
                                (cp[0]-half, cp[1]-half, cp[2])], WHITE, 0.15))
    return group


class P33(InteractiveScene):
    def construct(self):
        self.camera.background_rgba=[0, 0, 0, 1]
        self.frame.set_field_of_view(fov)
        a=load_activation('hot_dog', 'features_2')

        # Not the final depth: part 2 carries max_z the rest of the way to 14
        num_steps=120
        tracks=np.stack([np.linspace(s, e, num_steps) for s, e in
                         zip(grid_center_points(), line_center_points(6.0, 80.0))])
        current=None
        for step in range(num_steps):
            view=blend_views(p33_start, side_view, step/(num_steps-1))
            maps=activation_maps(a, tracks[:,step], True, forward_from(view))
            swap_out(self, current)
            self.add(maps)
            current=maps
            self.frame.reorient(*view)
            self.wait(1/30)

        clear_scene(self)
        num_steps=60
        tracks=np.stack([np.linspace(s, e, num_steps) for s, e in
                         zip(line_center_points(6.0, 80.0), line_center_points(6.0, 14.0))])
        self.frame.reorient(*side_view)
        forward=forward_from(side_view)
        current=None
        for step in range(num_steps):
            maps=activation_maps(a, tracks[:,step], False, forward)
            swap_out(self, current)
            self.add(maps)
            current=maps
            self.wait(1/30)
