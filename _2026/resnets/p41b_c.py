from manimlib import *
import numpy as np
import matplotlib.pyplot as plt
import os

# The flat companions to p37_41: no 3D stack, just the conv-5 grids and the captioned stills.

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
image_dir=data_dir+'/high_activation_imagenet_images'
#Scratch, and deliberately outside Dropbox: these are written then read back immediately,
#and Dropbox grabs files to hash them in between, which intermittently breaks the read
grid_dir='/tmp/resnet_scratch/p41b'

seconds_per_image=1/30
grid_size=(16, 16)
grid_gap=2
conv_cut=12  #conv-5

viridis_lut=plt.get_cmap('viridis')(np.linspace(0, 1, 256))


def viridis(values):
    x=np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    idx=np.clip((x*256).astype(np.int32), 0, 255) #matplotlib bins with int(x*N), not round(x*(N-1))
    return viridis_lut[idx]


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


def square_image(path, side):
    """Both shots stretch their image square, matching the original 512x512 resize."""
    img=ImageMobject(path)
    img.set_height(side, stretch=True)
    img.set_width(side, stretch=True)
    return img


class P41B_C(InteractiveScene):
    def construct(self):
        from PIL import Image
        self.camera.background_rgba=[0, 0, 0, 1]
        paths=image_paths()

        current=None
        for idx, path in enumerate(paths):
            a=activations(Image.open(path).convert('RGB'), conv_cut)
            img=square_image(activation_grid(a/a.max(), grid_dir+f'/{idx:04d}.png'),
                             self.camera.get_frame_height())
            img.move_to(ORIGIN)
            swap_out(self, current)
            self.add(img)
            current=img
            self.wait(seconds_per_image)

        current=None
        for path in paths:
            #The label is the last underscore field, so bow_tie gives "tie"
            caption=os.path.splitext(os.path.basename(path))[0].split('_')[-1]
            img=square_image(path, self.camera.get_frame_height()-1.5)
            img.move_to(0.4*DOWN)
            title=Text(caption, font_size=48).set_color(WHITE)
            title.next_to(img, UP, buff=0.3)

            group=Group(img, title)
            swap_out(self, current)
            self.add(group)
            current=group
            self.wait(seconds_per_image)
