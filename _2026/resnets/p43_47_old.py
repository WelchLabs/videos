from manimlib import *
import numpy as np
import matplotlib.pyplot as plt
import moderngl
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

data_dir='/Volumes/PG Work/Stephencwelch Dropbox/Pranav Gundu/Welch Labs/videos/_2026/resnets/data'

spacing_between_layers=5
line_radius=0.18
depth_step=0.125
cell_depth=0.1
pixel_dim=0.5

borders_only=100     #A threshold no normalised activation can reach
p45_thresh=0.35
image_opacity=0.995  #Measured: what the old three-deep voxel slab composited to
still_hold=1.0
seconds_per_image=0.5  #These passes get cut in by hand, so this is a hold, not a frame rate
fov=PI/3

image_bounds=(-32.0, 32.0, -32.0, 32.0, 0.0, 3*pixel_dim)
#Camera keyframe: theta, phi, gamma, center, height
view=(-105.4699, 89.956, 90.0122, (-13.7275, 2.9934, 77.4039), 122.093)

conv_layers=['features_2', 'features_5', 'features_8', 'features_10', 'features_12']
fc_layers=[('classifier_3', 256), ('classifier_6', 256), ('classifier_7', 128)]

#How many conv blocks and which fc blocks each p45 pass draws; p_45_2 was skipped
p45_passes=[(0, ()), (1, ()), (2, ()), (3, ()), (4, ()), (5, ()),
            (5, fc_layers[:1]), (5, fc_layers)]

viridis_lut=plt.get_cmap('viridis')(np.linspace(0, 1, 256))


def viridis(values):
    x=np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    idx=np.clip((x*256).astype(np.int32), 0, 255) #matplotlib bins with int(x*N), not round(x*(N-1))
    return viridis_lut[idx]


def load_activation(tag, name):
    return np.load(data_dir+'/activations/'+tag+'/'+name+'.npy')


def imagenet_tags():
    return sorted(os.listdir(data_dir+'/activations/imagenet'))


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


def conv_data_block(a, start_depth, thresh, view_forward):
    a=np.asarray(a, dtype=np.float64)
    n_c, n_i, n_j=a.shape
    kk, ii, jj=np.meshgrid(np.arange(n_c), np.arange(n_i), np.arange(n_j), indexing='ij')
    vals=a/a.max()
    keep=vals>=thresh

    half=np.floor(n_j/2)
    centers=np.stack([jj[keep]-half, -ii[keep]+half, depth_step*kk[keep]+start_depth], axis=-1)
    rgba=viridis(vals[keep])
    rgba[:,3]=0.5

    block=VoxelBlock(centers, np.array([1.0, 1.0, cell_depth]), rgba, view_forward)
    bounds=(-half-0.5, half+0.5, -half-0.5, half+0.5, start_depth, n_c*depth_step+start_depth)
    return block, bounds


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


def image_plane(tag, opacity):
    img=ImageMobject(data_dir+'/activations/'+tag+'/im.png')
    img.set_width(image_bounds[1]-image_bounds[0], stretch=True)
    img.set_height(image_bounds[3]-image_bounds[2], stretch=True)
    img.set_opacity(opacity)
    img.move_to([0, 0, 0.5*(image_bounds[4]+image_bounds[5])])
    return img


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


def network_stack(tag, thresh, view_forward, conv_cuts=None, fc_specs=(), color=WHITE,
                  radius=line_radius, fc_radius=0.03, image_border=True, image=False):
    conv_cuts=conv_layers if conv_cuts is None else conv_cuts
    group=Group()

    if image:
        group.add(image_plane(tag, image_opacity))
    if image_border:
        group.add(prism(*image_bounds, color, radius))

    start_depth=spacing_between_layers+1
    for cut in conv_cuts:
        block, bounds=conv_data_block(load_activation(tag, cut), start_depth, thresh,
                                      view_forward)
        group.add(block)
        group.add(prism(*bounds, color, radius))
        start_depth=bounds[5]+spacing_between_layers

    for name, viz_len in fc_specs:
        block, bounds=fc_data_block(load_activation(tag, name), start_depth, viz_len, thresh,
                                    view_forward)
        group.add(block)
        group.add(prism(*bounds, color, fc_radius))
        start_depth=bounds[5]+spacing_between_layers

    return group


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


def forward_from(view):
    """The direction the camera looks, for the painter's-algorithm sort."""
    from scipy.spatial.transform import Rotation
    theta, phi, gamma=np.radians(view[:3])
    return -Rotation.from_euler('zxz', [gamma, phi, theta]).as_matrix()[:,2]


class P43_47(InteractiveScene):
    def construct(self):
        self.frame.set_field_of_view(fov)
        self.frame.reorient(*view)
        forward=forward_from(view)

        for fc_specs in [fc_layers, fc_layers[:1], ()]:
            clear_scene(self)
            self.camera.background_rgba=list(color_to_rgba(FRESH_TAN))
            self.frame.reorient(*view)
            self.add(network_stack('hot_dog', borders_only, forward, fc_specs=fc_specs,
                                   color=CHILL_BLUE, fc_radius=0.1))
            self.wait(still_hold)

        # The last pass drops back to borders so the layers can be cut in by hand
        passes=p45_passes+[(5, (), borders_only)]
        for entry in passes:
            n_conv, fc_specs=entry[0], entry[1]
            thresh=entry[2] if len(entry)>2 else p45_thresh
            clear_scene(self)
            self.camera.background_rgba=list(color_to_rgba(FRESH_TAN))
            self.frame.reorient(*view)
            current=None
            for tag in imagenet_tags():
                stack=network_stack('imagenet/'+tag, thresh, forward,
                                    conv_cuts=conv_layers[:n_conv], fc_specs=fc_specs,
                                    color=CHILL_BLUE, fc_radius=0.1, image=True)
                swap_out(self, current)
                self.add(stack)
                current=stack
                self.wait(seconds_per_image)

        clear_scene(self)
        self.camera.background_rgba=list(color_to_rgba(CHILL_BLUE))
        self.frame.reorient(*view)
        self.add(network_stack('hot_dog', borders_only, forward, conv_cuts=conv_layers[:3],
                               color=FRESH_TAN, radius=0.6, image_border=False))
        self.wait(still_hold)
