import manimgx as gx
from resnet_assets import ACTIVATION_ROOT
from welch_scene import WelchScene
import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = str(ACTIVATION_ROOT) + '/'


class ColorBarForBookOne(WelchScene):

    def construct(self):
        img=gx.ImageMobject(data_dir+'colorbar_for_book_1.png')
        self.add(img)

        self.set_view(0, 0, 0, (np.float32(-0.04), np.float32(-0.08), np.float32(0.0)), 3.59)
        self.wait(2)
