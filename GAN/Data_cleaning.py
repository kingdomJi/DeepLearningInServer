
'''
输入待筛查文件夹路径，垃圾桶路径，指定的图像边长
函数自动筛除文件夹下不符合要求的图片
'''
import os
import shutil
from random import random

from cv2 import cv2


def sift_image(path, trashcan):
    """
    path:待处理目录
    trashcan:垃圾桶
    """
    image_list = os.listdir(path)  # 罗列其下所有文件的名称

    total = len(image_list)  # 算一下文件里的图像个数


    num_read_error = 0  # 因为无法读入被删除的图片数

    for i in range(total):

        real_path = os.path.join(path, image_list[i])  # 合成原始相对路径
        # real_path = os.path.join(path, str(i + random.uniform(0, 1)) + '.jpg')  # 生成不带中文名的新路径，否则imread读不出来
        # os.rename(old_path, real_path)  # 将可能带有中文的原始路径重命名成数字编号，如 162.jpg

        img = cv2.imread(real_path)  # 读图
        try:
            s = img.shape  # 如果读图失败会在这一步报错，所以干脆顺水推舟，来个except语句删掉读不了的图，反正读不了的也是损坏的
        except:
            print('图像: %s 读取失败,已移入垃圾桶' % image_list[i])  # 读不了，自觉进入垃圾桶
            shutil.move(real_path, trashcan)
            # os.remove(real_path)
            num_read_error += 1
            continue

    print('共读取 %d 张图像' % total)
    print('读取失败 %d 张，已移入垃圾桶' % num_read_error)

if __name__=='__main__':
    path=r'E:\jiangshan\U-net\Pytorch-UNet\data\Anime-Face-GAN-Keras\dataset'
    trashcan=r'E:\jiangshan\U-net\Pytorch-UNet\data\Anime-Face-GAN-Keras\dataset_error'
    sift_image(path, trashcan)