import manimgx as gx
from welch_style import BLUE, CHILL_BLUE, CHILL_BROWN, CHILL_GREEN, CYAN, FRESH_TAN, GREEN, MAGENTA, YELLOW, YELLOW_FADE
from resnet_assets import LOCAL_DATA
from welch_scene import WelchScene
from resnet_geometry import swap_out, viridis
import numpy as np
import matplotlib.pyplot as plt
import os

# The flat companions to p37_41: no 3D stack, just the conv-5 grids and the captioned stills.


data_dir = str(LOCAL_DATA)
image_dir=data_dir+'/high_activation_imagenet_images'
#Scratch, and deliberately outside Dropbox: these are written then read back immediately,
#and Dropbox grabs files to hash them in between, which intermittently breaks the read
grid_dir='/tmp/resnet_scratch/p41b'

seconds_per_image=1/30
grid_size=(16, 16)
grid_gap=2
conv_cut=12  #conv-5


def image_paths():
    return sorted(image_dir+'/'+f for f in os.listdir(image_dir) if f.lower().endswith('.jpg'))


alexnet=None
tfms=None


def activations(im, cut):
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
    with torch.no_grad():
        return alexnet.features[:cut](tfms(im)[None]).cpu().numpy()[0]


def activation_grid(a, path):
    """Tile a (C, H, W) stack into one image and write it out for ImageMobject."""
    from PIL import Image
    rows, cols=grid_size
    _, img_h, img_w=a.shape
    grid=np.zeros((rows*img_h+(rows-1)*grid_gap, cols*img_w+(cols-1)*grid_gap, 3))
    for idx in range(min(len(a), rows*cols)):
        row, col=idx//cols, idx%cols
        y0, x0=row*(img_h+grid_gap), col*(img_w+grid_gap)
        grid[y0:y0+img_h, x0:x0+img_w,:]=viridis(a[idx])[:,:,:3]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((np.clip(grid, 0, 1)*255).astype(np.uint8), 'RGB').save(path)
    return path


def square_image(path, side):
    """Both shots stretch their image square, matching the original 512x512 resize."""
    img=gx.ImageMobject(path)
    img.stretch_to_fit_height(side)
    img.stretch_to_fit_width(side)
    return img


class P41B_C(WelchScene):
    def construct(self):
        from PIL import Image
        self.camera.background_color = gx.BLACK
        paths=image_paths()

        current=None
        for idx, path in enumerate(paths):
            a=activations(Image.open(path).convert('RGB'), conv_cut)
            img=square_image(activation_grid(a/a.max(), grid_dir+f'/{idx:04d}.png'),
                             gx.config.frame_height)
            img.move_to(gx.ORIGIN)
            swap_out(self, current)
            self.add(img)
            current=img
            self.wait(seconds_per_image)

        current=None
        for path in paths:
            #The label is the last underscore field, so bow_tie gives "tie"
            caption=os.path.splitext(os.path.basename(path))[0].split('_')[-1]
            img=square_image(path, gx.config.frame_height-1.5)
            img.move_to(0.4*gx.DOWN)
            title=gx.Text(caption, font_size=48).set_color(gx.WHITE)
            title.next_to(img, gx.UP, buff=0.3)

            group=gx.Group(img, title)
            swap_out(self, current)
            self.add(group)
            current=group
            self.wait(seconds_per_image)
