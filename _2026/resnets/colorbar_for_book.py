from manimlib import *
import numpy as np
import matplotlib.pyplot as plt
import os

# data_dir='/Volumes/hot_1/Stephencwelch Dropbox/welch_labs/resnet/hackin/'
data_dir='/Users/stephen/Library/CloudStorage/Dropbox-Stephencwelch/welch_labs/resnet/hackin/'



class ColorBarForBookOne(InteractiveScene):

    def construct(self):
        img=ImageMobject(data_dir+'colorbar_for_book_1.png')
        self.add(img)

        self.frame.reorient(0, 0, 0, (np.float32(-0.04), np.float32(-0.08), np.float32(0.0)), 3.59)
        self.wait(2)
        self.embed()