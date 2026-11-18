# The whole network as wireframe: two stills on tan, then a push in onto the input.
# The single-kernel shots of p24b-p24d are in p24b_d_old.py.

import manimgx as gx
from welch_scene import WelchScene
from resnet_assets import LOCAL_DATA
from resnet_geometry import VoxelBlock, clear_scene, prism, viridis
import numpy as np

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

data_dir = str(LOCAL_DATA)

spacing_between_layers=5
line_radius=0.18
depth_step=0.125
cell_depth=0.1
pixel_dim=0.5

borders_only=100     #A threshold no normalised activation can reach
still_hold=1.0
fov=gx.PI/3

image_bounds=(-32.0, 32.0, -32.0, 32.0, 0.0, 3*pixel_dim)

#Camera keyframes: theta, phi, gamma, center, height
p21_camera=(-124.9504, 89.9626, 90.0261, (-13.9602, 2.7904, 80.5698), 141.852)
p24_camera=(-105.4699, 89.956, 90.0122, (-13.7275, 2.9934, 77.4039), 141.852)
p24_end=(-135.4168, 63.3448, 114.4748, (0.0, 0.0, 0.0), 96.224)

conv_layers=['features_2', 'features_5', 'features_8', 'features_10', 'features_12']
fc_layers=[('classifier_3', 256), ('classifier_6', 256), ('classifier_7', 128)]


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


def network_stack(thresh, view_forward, fc_specs=(), color=gx.WHITE, image_border=True):
    group=gx.Group()
    if image_border:
        group.add(prism(*image_bounds, color, line_radius))

    start_depth=spacing_between_layers+1
    for cut in conv_layers:
        block, bounds=conv_data_block(load_activation('hot_dog', cut), start_depth, thresh,
                                      view_forward)
        group.add(block)
        group.add(prism(*bounds, color, line_radius))
        start_depth=bounds[5]+spacing_between_layers

    for name, viz_len in fc_specs:
        block, bounds=fc_data_block(load_activation('hot_dog', name), start_depth, viz_len,
                                    thresh, view_forward)
        group.add(block)
        group.add(prism(*bounds, color, 0.1))
        start_depth=bounds[5]+spacing_between_layers

    return group


def forward_from(view):
    """The direction the camera looks, for the painter's-algorithm sort."""
    from scipy.spatial.transform import Rotation
    theta, phi, gamma=np.radians(view[:3])
    return -Rotation.from_euler('zxz', [gamma, phi, theta]).as_matrix()[:,2]


class P21_24(WelchScene):
    def construct(self):
        self.view_fov = fov

        self.camera.background_color = FRESH_TAN
        self.set_view(*p21_camera)
        self.add(network_stack(borders_only, forward_from(p21_camera), fc_specs=fc_layers[:2],
                               color=CHILL_BLUE, image_border=False))
        self.wait(still_hold)

        clear_scene(self)
        self.set_view(*p24_camera)
        self.add(network_stack(borders_only, forward_from(p24_camera), fc_specs=fc_layers,
                               color=CHILL_BLUE))
        self.wait(still_hold)

        clear_scene(self)
        self.camera.background_color = gx.BLACK
        self.set_view(*p24_camera)
        self.add(network_stack(borders_only, forward_from(p24_camera)))
        self.play(self.animate_view(*p24_end), run_time=2, rate_func=gx.linear)
