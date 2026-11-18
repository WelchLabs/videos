# One conv-1 kernel: wired to its activation, then walked across all 55x55 positions.
# The whole-network wireframe shots of p21-p24 are in p21_24_old.py.

import manimgx as gx
from welch_scene import WelchScene
from resnet_geometry import VoxelBlock, blend_views, clear_scene, polyline, prism, swap_out, viridis
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

data_dir='/Users/stephen/videos/_2026/resnets/data'

spacing_between_layers=5
line_radius=0.18
depth_step=0.125
cell_depth=0.1
pixel_dim=0.5

image_opacity=0.936  #Measured: what the old three-deep voxel slab composited to
still_hold=1.0
steps_per_viz=11     #p24c/p24d step every 11th of the 55*55 kernel positions
kernel_k=0
fov=gx.PI/3

image_bounds=(-32.0, 32.0, -32.0, 32.0, 0.0, 3*pixel_dim)

#Camera keyframes: theta, phi, gamma, center, height
p24_end=(-135.4168, 63.3448, 114.4748, (0.0, 0.0, 0.0), 96.224)
p24d_end=(-139.5221, 31.6032, 134.9423, (0.0, 0.0, 0.0), 96.224)


def load_activation(tag, name):
    return np.load(data_dir+'/activations/'+tag+'/'+name+'.npy')


# 6 faces x 4 corners of a unit cube, wound counter-clockwise seen from outside


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
    img=gx.ImageMobject(data_dir+'/activations/'+tag+'/im.png')
    img.stretch_to_fit_width(image_bounds[1]-image_bounds[0])
    img.stretch_to_fit_height(image_bounds[3]-image_bounds[2])
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


def conv1_kernel(i, j, k, a_shape, weights, view_forward):
    """The conv-1 kernel wired to its activation, with the filter painted into the patch."""
    min_x, max_x, min_y, _, min_z, max_z=image_bounds
    group=gx.Group()

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


def forward_from(view):
    """The direction the camera looks, for the painter's-algorithm sort."""
    from scipy.spatial.transform import Rotation
    theta, phi, gamma=np.radians(view[:3])
    return -Rotation.from_euler('zxz', [gamma, phi, theta]).as_matrix()[:,2]


class P24B_D(WelchScene):
    def construct(self):
        self.camera.background_color = gx.BLACK
        self.view_fov = fov
        a=load_activation('hot_dog', 'features_2')
        weights=load_activation('weights', 'features0')[0]
        vmax=float(a[0].max())

        self.set_view(*p24_end)
        forward=forward_from(p24_end)
        self.add(prism(*image_bounds, gx.WHITE, line_radius))
        self.add(conv1_kernel(0, 10, kernel_k, a.shape, weights, forward))
        block, bounds=conv_data_block(masked_conv1(a, 0, 10, kernel_k),
                                      spacing_between_layers+1, 0.005, forward)
        self.add(block)
        self.add(prism(*bounds, gx.WHITE, line_radius))
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

                    shot=gx.Group()
                    shot.add(image_plane('hot_dog', image_opacity))
                    shot.add(prism(*image_bounds, gx.WHITE, line_radius))
                    shot.add(conv1_kernel(i, j, kernel_k, a.shape, weights, fwd))
                    block, bounds=conv_data_block(masked_conv1(a, i, j, kernel_k),
                                                  spacing_between_layers+1, 0.005, fwd, vmax)
                    shot.add(block)
                    shot.add(prism(*bounds, gx.WHITE, line_radius))

                    swap_out(self, current)
                    self.add(shot)
                    current=shot
                    self.set_view(*view)
                    self.wait(1/30)
