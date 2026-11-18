# The input image with conv-1 and conv-2: four stills and three camera moves.
# The 64 conv-1 maps flying from a grid into a line are in p33_old.py.

import manimgx as gx
from welch_scene import WelchScene
from resnet_geometry import VoxelBlock, clear_scene, polyline, prism, swap_out, viridis
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
MAGENTA='#FF00FF'

# data_dir='/Volumes/PG Work/Stephencwelch Dropbox/Pranav Gundu/Welch Labs/videos/_2026/resnets/data'
data_dir='/Users/stephen/videos/_2026/resnets/data'

spacing_between_layers=5
line_radius=0.18
depth_step=0.125
cell_depth=0.1
pixel_dim=0.5

sparse_thresh=0.15
block_opacity=0.9       #These blocks are drawn denser than p21_24's and p43_47's
image_opacity=0.936     #Measured: what the old three-deep voxel slab composited to
still_hold=1.0
fov=gx.PI/3

image_bounds=(-32.0, 32.0, -32.0, 32.0, 0.0, 3*pixel_dim)

#Camera keyframes: theta, phi, gamma, center, height
side_view=(-108.4402, 83.9753, 92.0043, (5.3775, -5.9592, 46.6544), 113.816)
kernel_view=(-127.2543, 91.3322, 88.987, (4.1924, -6.853, 32.5337), 113.816)
overhead_view=(-116.9959, 42.2256, 110.6684, (0.6655, 0.2306, 19.1857), 113.816)


def load_activation(tag, name):
    return np.load(data_dir+'/activations/'+tag+'/'+name+'.npy')


# 6 faces x 4 corners of a unit cube, wound counter-clockwise seen from outside


def conv_data_block(a, start_depth, render_dense, thresh, opacity, view_forward):
    a=np.asarray(a, dtype=np.float64)
    n_c, n_i, n_j=a.shape
    kk, ii, jj=np.meshgrid(np.arange(n_c), np.arange(n_i), np.arange(n_j), indexing='ij')
    vals=a/a.max()
    keep=np.ones(vals.shape, dtype=bool) if render_dense else vals>=thresh

    half=np.floor(n_j/2)
    centers=np.stack([jj[keep]-half, -ii[keep]+half, depth_step*kk[keep]+start_depth], axis=-1)
    rgba=viridis(vals[keep])
    rgba[:,3]=opacity

    block=VoxelBlock(centers, np.array([1.0, 1.0, cell_depth]), rgba, view_forward)
    bounds=(-half-0.5, half+0.5, -half-0.5, half+0.5, start_depth, n_c*depth_step+start_depth)
    return block, bounds


def image_plane(tag, opacity):
    img=gx.ImageMobject(data_dir+'/activations/'+tag+'/im.png')
    img.stretch_to_fit_width(image_bounds[1]-image_bounds[0])
    img.stretch_to_fit_height(image_bounds[3]-image_bounds[2])
    img.set_opacity(opacity)
    img.move_to([0, 0, 0.5*(image_bounds[4]+image_bounds[5])])
    return img


def kernel_overlay(i, j, a_shape, src_bounds, dst_z, kernel_size, single_curve=False):
    """The magenta patch plus the four lines back to the activation it feeds."""
    src_min_x, src_max_x, src_min_y, _, src_min_z, src_max_z=src_bounds
    group=gx.Group()

    step=(src_max_x-src_min_x)/a_shape[1]
    dst_x=j-np.floor(a_shape[-1]/2)
    dst_y=-i+np.floor(a_shape[-1]/2)

    #-src_min_y spells src_max_y; the bounds are symmetric
    connectors=[
        [(dst_x-0.5, dst_y-0.5, dst_z),
         (src_min_x+step*j, -src_min_y-step*i-kernel_size, src_max_z)],
        [(dst_x+0.5, dst_y-0.5, dst_z),
         (src_min_x+step*j+kernel_size, -src_min_y-step*i-kernel_size, src_max_z)],
        [(dst_x-0.5, dst_y+0.5, dst_z),
         (src_min_x+step*j, -src_min_y-step*i, src_max_z)],
        [(dst_x+0.5, dst_y+0.5, dst_z),
         (src_min_x+step*j+kernel_size, -src_min_y-step*i, src_max_z)],
    ]

    if single_curve: #p35a's second kernel runs all eight endpoints through one polyline
        group.add(polyline([pt for cc in connectors for pt in cc], MAGENTA, line_radius))
    else:
        for cc in connectors:
            group.add(polyline(cc, MAGENTA, line_radius))

    for p, q in [(0, 1), (1, 3), (3, 2), (2, 0)]:
        group.add(polyline([connectors[p][0], connectors[q][0]], MAGENTA, line_radius))

    group.add(prism(src_min_x+step*j, src_min_x+step*j+kernel_size,
                    -src_min_y-step*i-kernel_size, -src_min_y-step*i,
                    src_min_z, src_max_z, MAGENTA, line_radius))
    return group


def build_stack(view_forward, image=True, conv2=True, conv1_dense=False,
                conv2_dense=False, kernels=False):
    group=gx.Group()
    if image:
        group.add(image_plane('hot_dog', image_opacity))
    group.add(prism(*image_bounds, gx.WHITE, line_radius))

    a1=load_activation('hot_dog', 'features_2')
    start_depth=spacing_between_layers+1
    conv1, b1=conv_data_block(a1, start_depth, conv1_dense, sparse_thresh, block_opacity,
                              view_forward)
    group.add(conv1)
    group.add(prism(*b1, gx.WHITE, line_radius))

    if kernels: #An 8 unit patch of the image feeding conv-1 cell (0, 10)
        group.add(kernel_overlay(0, 10, a1.shape, image_bounds, start_depth, 8.0))

    if not conv2:
        return group

    a2=load_activation('hot_dog', 'features_5')
    start_depth_2=b1[5]+spacing_between_layers
    block, b2=conv_data_block(a2, start_depth_2, conv2_dense, sparse_thresh, block_opacity,
                              view_forward)
    group.add(block)
    group.add(prism(*b2, gx.WHITE, line_radius))

    if kernels: #Kernel two sits on the conv-1 block rather than the image
        group.add(kernel_overlay(3, 1, a2.shape, b1, start_depth_2, 5.0, True))

    return group


def forward_from(view):
    """The direction the camera looks, for the painter's-algorithm sort."""
    from scipy.spatial.transform import Rotation
    theta, phi, gamma=np.radians(view[:3])
    return -Rotation.from_euler('zxz', [gamma, phi, theta]).as_matrix()[:,2]


class P34_36(WelchScene):
    def construct(self):
        self.camera.background_color = gx.BLACK
        self.view_fov = fov
        forward=forward_from(side_view)

        self.set_view(*side_view)
        self.add(build_stack(forward, conv1_dense=True, conv2=False))
        self.wait(still_hold)

        clear_scene(self)
        self.set_view(*side_view)
        self.add(build_stack(forward, conv1_dense=True, conv2_dense=True))
        self.wait(still_hold)

        clear_scene(self)
        self.set_view(*side_view)
        self.add(build_stack(forward))
        self.wait(still_hold)

        clear_scene(self)
        self.set_view(*side_view)
        self.add(build_stack(forward, image=False, kernels=True))
        self.wait(still_hold)

        for start, end, kernels in [(side_view, kernel_view, True),
                                    (kernel_view, overhead_view, True),
                                    (overhead_view, side_view, False)]:
            clear_scene(self)
            self.set_view(*start)
            self.add(build_stack(forward_from(start), kernels=kernels))
            self.play(self.animate_view(*end), run_time=2, rate_func=gx.linear)


        self.wait()
