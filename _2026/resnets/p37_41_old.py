import manimgx as gx
from welch_style import BLUE, CHILL_BLUE, CHILL_BROWN, CHILL_GREEN, CYAN, FRESH_TAN, GREEN, MAGENTA, YELLOW, YELLOW_FADE
from resnet_assets import LOCAL_DATA
from welch_scene import WelchScene
from resnet_geometry import VoxelBlock, clear_scene, forward_from, polyline, prism, swap_out, threshold_volume, viridis
import numpy as np
import matplotlib.pyplot as plt
import os

# The five-layer stack, one still at a time. Paragraphs 37-39 are absent: every shot
# there ran a-roll footage through the network, and those clips are not in this repo.
# The flat conv-5 grids and captioned stills of p41 are in p41b_c_old.py.


data_dir = str(LOCAL_DATA)
image_dir=data_dir+'/high_activation_imagenet_images'
#Scratch, and deliberately outside Dropbox: these are written then read back immediately,
#and Dropbox grabs files to hash them in between, which intermittently breaks the read
frame_dir='/tmp/resnet_scratch/p41a'

spacing_between_layers=5
line_radius=0.18
depth_step=0.125
cell_depth=0.1
pixel_dim=0.5

sparse_thresh=0.25
block_opacity=0.95   #These blocks are drawn denser than p21_24's and p43_47's
image_opacity=0.936  #Measured: what the old three-deep voxel slab composited to
seconds_per_image=1/30
fov=gx.PI/3

image_bounds=(-32.0, 32.0, -32.0, 32.0, 0.0, 3*pixel_dim)
#Camera keyframe: theta, phi, gamma, center, height
five_layer_view=(-111.1996, 89.9576, 90.0164, (8.5339, -5.7512, 70.6408), 146.1426)
conv_cuts=[2, 5, 8, 10, 12]


def image_paths():
    return sorted(image_dir+'/'+f for f in os.listdir(image_dir) if f.lower().endswith('.jpg'))


alexnet=None
tfms=None


def activations(im, cuts):
    global alexnet, tfms
    import torch
    if alexnet is None:
        import torchvision.models as models
        from torchvision import transforms
        alexnet=models.alexnet(weights='IMAGENET1K_V1')
        alexnet.eval()
        tfms=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    batch=tfms(im)[None]
    with torch.no_grad():
        return {cut: alexnet.features[:cut](batch).cpu().numpy()[0] for cut in cuts}


def conv_data_block(a, start_depth, thresh, opacity, view_forward):
    return threshold_volume(a, start_depth, thresh, opacity=opacity, view_forward=view_forward)


def image_plane(im_pil, path, opacity):
    from PIL import Image
    a=np.array(im_pil.resize((128, 128)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(a[:,:,[0,1,1]], 'RGB').save(path) #Green stands in for blue, as in the original

    img=gx.ImageMobject(path)
    img.stretch_to_fit_width(image_bounds[1]-image_bounds[0])
    img.stretch_to_fit_height(image_bounds[3]-image_bounds[2])
    img.set_opacity(opacity)
    img.move_to([0, 0, 0.5*(image_bounds[4]+image_bounds[5])])
    return img


def network_stack(acts, im_pil, path, view_forward):
    group=gx.Group()
    group.add(image_plane(im_pil, path, image_opacity))
    group.add(prism(*image_bounds, gx.WHITE, line_radius))

    start_depth=spacing_between_layers+1
    for cut in conv_cuts:
        block, bounds=conv_data_block(acts[cut], start_depth, sparse_thresh, block_opacity,
                                      view_forward)
        group.add(block)
        group.add(prism(*bounds, gx.WHITE, line_radius))
        start_depth=bounds[5]+spacing_between_layers

    return group


class P37_41(WelchScene):
    def construct(self):
        from PIL import Image
        self.camera.background_color = gx.BLACK
        self.view_fov = fov

        self.set_view(*five_layer_view)
        forward=forward_from(five_layer_view)
        current=None
        for idx, path in enumerate(image_paths()):
            im=Image.open(path).convert('RGB')
            stack=network_stack(activations(im, conv_cuts), im,
                                frame_dir+f'/{idx:04d}.png', forward)
            swap_out(self, current)
            self.add(stack)
            current=stack
            self.wait(seconds_per_image)
