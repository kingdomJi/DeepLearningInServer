import os

import cv2
from PIL import Image
import numpy as np

def B2W(source_dir,save_dir):
    mask_list = os.listdir(source_dir)
    for i in range(len(mask_list)):  # 读取文件夹下每一个图片
        print(source_dir +'/'+ mask_list[i])
        # mask = Image.open(source_dir +'/'+ mask_list[i])
        mask =cv2.imread(source_dir +'/'+ mask_list[i],0)
        # mask = np.asarray(mask)
        mask[mask > 124] = 255
        mask[mask < 125] = 0
        mask[mask == 255] = 1
        mask[mask == 0] = 255
        mask[mask == 1]=0
        save_path=save_dir+'/'+mask_list[i]
        t=cv2.imwrite(save_path, mask)


if __name__=='__main__':
    source_dir=r'E:\jiangshan\crack_segmentation_dataset\masks_256'
    save_dir=r'E:\jiangshan\crack_segmentation_dataset\masksWhite_256'
    B2W(source_dir, save_dir)