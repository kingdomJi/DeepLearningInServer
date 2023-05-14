import cv2.cv2
import albumentations as A
import unittest
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import albumentations.pytorch
from torch.utils import data
import torch
import glob
import os
from PIL import Image
import numpy as np

"""
将风格迁移后的图像对应的mask部分保留原图特征
"""
def img_NoChangeInMask(img_path_T,img_path,mask_white_path,mask_black_path,save_path):
    list_t=os.listdir(img_path_T)#风格迁移后的图像列表
    list_i=os.listdir(img_path)
    list_mw=os.listdir(mask_white_path)
    list_mb=os.listdir(mask_black_path)
    for i in range(len(list_t)):
        print(list_t[i])
        assert list_t[i] ==list_i[i]#名称对应
        img_t=torch.tensor(cv2.imread(img_path_T+'/'+list_t[i]))
        img = torch.tensor(cv2.imread(img_path+'/'+list_i[i]))
        mask_w = cv2.imread(mask_white_path+'/'+list_mw[i],0)
        mask_b= cv2.imread(mask_black_path + '/' + list_mb[i],0)
        # print(mask_w.shape)#256,256
        # mask_w=mask_w/255#特征处为0
        mask_w=torch.tensor(mask_w, dtype=torch.float)/255#转0，1
        # print(np.unique(mask_w))#不全是整0，1
        # mask_b=mask_b/255
        mask_b=torch.tensor(mask_b, dtype=torch.float)/255#转0，1
        # print(mask_w.shape)#[256, 256]
        # tensors_w =torch.tensor([])
        # tensors_b= torch.tensor([])
        tensors_w=torch.unsqueeze(mask_w,dim=0)
        tensors_b=torch.unsqueeze(mask_b,dim=0)
        tensors_w=torch.cat(( tensors_w, tensors_w, tensors_w),0).permute(1, 2, 0)
        tensors_b=torch.cat((tensors_b,tensors_b,tensors_b),0).permute(1, 2, 0)
        print("tensors_w.shape:",tensors_w.shape,"tensors_w.unque:")
        print('img_t.shape:',img_t.shape,)
        # mask_w=torch.stack(tensors_w, dim=2)
        # mask_w = torch.stack(mask_w, dim=0)  # 第1维的数据进行拼接
        # mask_w = torch.unsqueeze(mask_w, 0)
        # mask_b = torch.stack(tensors_b, dim=2)
        # mask_b = torch.stack(mask_b, dim=0)  # 第1维的数据进行拼接
        # mask_b = torch.unsqueeze(mask_b, 0)
        # print('mask_w.shape:',mask_w.shape)
        img_t=torch.mul(img_t,tensors_w)#点乘,对应元素相乘
        img= torch.mul(img,tensors_b)
        img_t=img_t+img
        img_t=img_t.numpy()
        print(np.unique(img_t))
        cv2.imwrite(save_path+'/'+list_t[i],img_t)
"""
将图片对应的mask部分变为全黑
"""
def img_MaskInblack(img_path_T,mask_white_path, save_path):
    list_t = os.listdir(img_path_T)  # 风格迁移后的图像列表
    list_m = os.listdir(mask_white_path)
    for i in range(len(list_t)):
        print(list_t[i])
        assert os.path.splitext(list_t[i])[0] ==os.path.splitext(list_m[i])[0]   # 名称对应
        img = torch.tensor(cv2.imread(img_path_T + '/' + list_t[i]))
        mask_b = cv2.imread(mask_white_path + '/' + list_m[i], 0)
        # print(mask_w.shape)#256,256
        # mask_w=mask_w/255#特征处为0
        # print(np.unique(mask_w))#不全是整0，1
        # mask_b=mask_b/255
        mask_b = torch.tensor(mask_b, dtype=torch.float) / 255  # 转0，1,特征部分为0
        # print(mask_w.shape)#[256, 256]
        # tensors_w =torch.tensor([])
        # tensors_b= torch.tensor([])
        tensors_b = torch.unsqueeze(mask_b, dim=0)
        tensors_b = torch.cat((tensors_b, tensors_b, tensors_b), 0).permute(1, 2, 0)
        # mask_w=torch.stack(tensors_w, dim=2)
        # mask_w = torch.stack(mask_w, dim=0)  # 第1维的数据进行拼接
        # mask_w = torch.unsqueeze(mask_w, 0)
        # mask_b = torch.stack(tensors_b, dim=2)
        # mask_b = torch.stack(mask_b, dim=0)  # 第1维的数据进行拼接
        # mask_b = torch.unsqueeze(mask_b, 0)
        # print('mask_w.shape:',mask_w.shape)
        img = torch.mul(img, tensors_b)
        img = img.numpy()
        print(np.unique(img))
        cv2.imwrite(save_path + '/' + list_t[i], img)

if __name__ == '__main__':
    input_path=r'E:\jiangshan\crack_segmentation_dataset\images_256'
    input_mask_w=r'E:\jiangshan\crack_segmentation_dataset\masksWhite_256'
    save_path=r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\public_Masktoblack_256'
    img_MaskInblack(input_path,input_mask_w,save_path)


