from torch.utils.data import Dataset
import torch
import os
import numpy as np
from PIL import Image
from pathlib import Path
dir_mask = '../data/crack_segmentation_dataset/train/masks/'
mask_name='CFD_002.jpg'
img=Image.open(dir_mask+mask_name)
img=np.asarray(img)
# print(img[300])
a=254
b=253
c=5
d=3
print(round(a/255))
print(round(b/255))
print(round(c/255))
print(round(d/255))

