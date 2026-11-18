import manimgx as gx
from welch_style import BLUE, CHILL_BLUE, CHILL_BROWN, CHILL_GREEN, CYAN, FRESH_TAN, GREEN, MAGENTA, YELLOW, YELLOW_FADE
from resnet_assets import LOCAL_DATA
from welch_scene import WelchScene
from resnet_geometry import VoxelBlock, blend_views, clear_scene, forward_from, polyline, swap_out, viridis
import numpy as np
import matplotlib.pyplot as plt


data_dir = str(LOCAL_DATA)


cell_depth=0.1

grid_sparse_thresh=0.1  #p33 grid/line animations
grid_tile=55
grid_spacing=1
fov=gx.PI/3


#Camera keyframes: theta, phi, gamma, center, height
side_view=(-108.4402, 83.9753, 92.0043, (5.3775, -5.9592, 46.6544), 113.816)
p33_start=(0.0, 0.0, 0.0, (0.0, 0.0, 0.0), 591.468)


def load_activation(tag, name):
    return np.load(data_dir+'/activations/'+tag+'/'+name+'.npy')


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
    group=gx.Group()
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
                                (cp[0]-half, cp[1]-half, cp[2])], gx.WHITE, 0.15))
    return group


class P33(WelchScene):
    def construct(self):
        self.camera.background_color = gx.BLACK
        self.view_fov = fov
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
            self.set_view(*view)
            self.wait(1/30)

        clear_scene(self)
        num_steps=60
        tracks=np.stack([np.linspace(s, e, num_steps) for s, e in
                         zip(line_center_points(6.0, 80.0), line_center_points(6.0, 14.0))])
        self.set_view(*side_view)
        forward=forward_from(side_view)
        current=None
        for step in range(num_steps):
            maps=activation_maps(a, tracks[:,step], False, forward)
            swap_out(self, current)
            self.add(maps)
            current=maps
            self.wait(1/30)
