import manimgx as gx
from welch_style import BLUE, CHILL_BLUE, CHILL_BROWN, CHILL_GREEN, CYAN, FRESH_TAN, GREEN, MAGENTA, YELLOW, YELLOW_FADE
from resnet_assets import LOCAL_DATA
from welch_scene import WelchScene
from resnet_geometry import VoxelBlock, clear_scene, fc_data_block, forward_from, polyline, prism, swap_out, threshold_volume, viridis
import numpy as np
import matplotlib.pyplot as plt
import os


data_dir = str(LOCAL_DATA)

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
fov=gx.PI/3

image_bounds=(-32.0, 32.0, -32.0, 32.0, 0.0, 3*pixel_dim)
#Camera keyframe: theta, phi, gamma, center, height
view=(-105.4699, 89.956, 90.0122, (-13.7275, 2.9934, 77.4039), 122.093)

conv_layers=['features_2', 'features_5', 'features_8', 'features_10', 'features_12']
fc_layers=[('classifier_3', 256), ('classifier_6', 256), ('classifier_7', 128)]

#How many conv blocks and which fc blocks each p45 pass draws; p_45_2 was skipped
p45_passes=[(0, ()), (1, ()), (2, ()), (3, ()), (4, ()), (5, ()),
            (5, fc_layers[:1]), (5, fc_layers)]


def load_activation(tag, name):
    return np.load(data_dir+'/activations/'+tag+'/'+name+'.npy')


def imagenet_tags():
    return sorted(os.listdir(data_dir+'/activations/imagenet'))


def conv_data_block(a, start_depth, thresh, view_forward):
    return threshold_volume(a, start_depth, thresh, view_forward=view_forward)


def image_plane(tag, opacity):
    img=gx.ImageMobject(data_dir+'/activations/'+tag+'/im.png')
    img.stretch_to_fit_width(image_bounds[1]-image_bounds[0])
    img.stretch_to_fit_height(image_bounds[3]-image_bounds[2])
    img.set_opacity(opacity)
    img.move_to([0, 0, 0.5*(image_bounds[4]+image_bounds[5])])
    return img


def network_stack(tag, thresh, view_forward, conv_cuts=None, fc_specs=(), color=gx.WHITE,
                  radius=line_radius, fc_radius=0.03, image_border=True, image=False):
    conv_cuts=conv_layers if conv_cuts is None else conv_cuts
    group=gx.Group()

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


class P43_47(WelchScene):
    def construct(self):
        self.view_fov = fov
        self.set_view(*view)
        forward=forward_from(view)

        for fc_specs in [fc_layers, fc_layers[:1], ()]:
            clear_scene(self)
            self.camera.background_color = FRESH_TAN
            self.set_view(*view)
            self.add(network_stack('hot_dog', borders_only, forward, fc_specs=fc_specs,
                                   color=CHILL_BLUE, fc_radius=0.1))
            self.wait(still_hold)

        # The last pass drops back to borders so the layers can be cut in by hand
        passes=p45_passes+[(5, (), borders_only)]
        for entry in passes:
            n_conv, fc_specs=entry[0], entry[1]
            thresh=entry[2] if len(entry)>2 else p45_thresh
            clear_scene(self)
            self.camera.background_color = FRESH_TAN
            self.set_view(*view)
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
        self.camera.background_color = CHILL_BLUE
        self.set_view(*view)
        self.add(network_stack('hot_dog', borders_only, forward, conv_cuts=conv_layers[:3],
                               color=FRESH_TAN, radius=0.6, image_border=False))
        self.wait(still_hold)
