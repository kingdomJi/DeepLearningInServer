"""
清洗部分迁移后不符合需求的数据
"""
import os
import shutil

import numpy as np
from cv2 import cv2


def washing_publicSource(read_img_P,read_mask_P,trashcan_img,trashcan_mask):#根据色差筛选不符合要求的裂缝数据
    img_list = os.listdir(read_img_P)
    mask_list = os.listdir(read_mask_P)
    distance_color=0#色差
    feature_color_sum_R = 0
    feature_color_sum_G = 0
    feature_color_sum_B = 0
    color_sum_R=0
    color_sum_G = 0
    color_sum_B = 0
    count_feature=0
    for k in range(len(img_list)):
        img=cv2.imread(read_img_P+'/'+img_list[k])#BGR
        mask=cv2.imread(read_mask_P+'/'+mask_list[k],0)
        for i in range(mask.shape[0]):
            for j in range (mask.shape[1]):
                # print(mask[i][j])
                color_sum_B+=img[i][j][0]
                color_sum_G += img[i][j][1]
                color_sum_R += img[i][j][2]
                if mask[i][j]> 200 :
                    count_feature+=1#RGB
                    feature_color_sum_B+=img[i][j][0]
                    feature_color_sum_G += img[i][j][1]
                    feature_color_sum_R += img[i][j][2]
        avg_color_img_B = color_sum_B /(img.shape[0]*img.shape[1])
        avg_color_img_G = color_sum_G / (img.shape[0] * img.shape[1])
        avg_color_img_R = color_sum_R / (img.shape[0] * img.shape[1])
        if(count_feature!=0):
            avg_color_feature_B=feature_color_sum_B/count_feature
            avg_color_feature_G = feature_color_sum_G / count_feature
            avg_color_feature_R = feature_color_sum_R / count_feature

            Lightness_img=avg_color_img_R*0.299 + avg_color_img_G*0.587 + avg_color_img_B*0.114
            Lightness_feature=avg_color_feature_R*0.299 + avg_color_feature_G*0.587 + avg_color_feature_B*0.114
            print(Lightness_img)
            print(Lightness_feature)
            if abs(Lightness_img-Lightness_feature)<50:#
                shutil.move(read_img_P+'/'+img_list[k], trashcan_img)
                shutil.move(read_mask_P + '/' + mask_list[k], trashcan_mask)
        feature_color_sum_R = 0
        feature_color_sum_G = 0
        feature_color_sum_B = 0
        color_sum_R = 0
        color_sum_G = 0
        color_sum_B = 0
        count_feature = 0



def washing_package(read_img,read_mask,trans_imgPath,save_path):
    img_list=os.listdir(read_img)
    # print(img_list)
    mask_list=os.listdir(read_mask)
    trans_list=os.listdir(trans_imgPath)
    for k in range(len(trans_list)):
        img=cv2.imread(read_img+'/'+img_list[k])
        mask=cv2.imread(read_mask+'/'+mask_list[k],0)
        trans_img=cv2.imread(trans_imgPath+'/'+trans_list[k])
        for i in range(mask.shape[0]):
            for j in range (mask.shape[1]):
                # print(mask[i][j])
                if mask[i][j]> 200 and max(img[i][j])>50:
                    # img[i][j]=img[i][j]/3
                    img[i][j] = trans_img[i][j]
        # print(img_list[k])
        cv2.imwrite(save_path+'/'+trans_list[k],img)

def img_mix(path1,path2,save_path):
    list1=os.listdir(path1)
    list2=os.listdir(path2)
    for i in range(len(list1)):
        img1 = cv2.imread(path1+'/'+list1[i])
        img2 = cv2.imread(path2+'/'+list2[i])
        #addWeighted函数，有5个参数，可以列出为：图像源1，src1权重，图像源2，src2权重，伽玛。每个图像的权重值必须小于1。
        result=cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)

        cv2.imwrite(save_path+'/'+list1[i],result)

def washing_singel(read_img,read_mask,save_path):
    img = cv2.imread(read_img )
    mask = cv2.imread(read_mask , 0)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # print(mask[i][j])
            if mask[i][j] > 200 and max(img[i][j]) > 50:
                # print(img[i][j])
                img[i][j] = img[i][j] / 3
                # img[i][j] = np.random.randint(0,10,3)

    cv2.imwrite(save_path , img)


if __name__=='__main__':
    save_path = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\NewTransfer_img'
    read_img = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\Transfer_img'
    read_mask = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\Transfer_mask'
    trans_path=r'E:\jiangshan\U-net\UGATIT\results\UGATIT_1'
    trashcan_img=r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\Transfer_trashcan'
    trashcan_mask = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\mask_trashcan'
    # save_path = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\test_img\CRACK500_20160328_153020_1_721_new.png'
    # read_img = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\test_img\CRACK500_20160328_153020_1_721.png'
    # read_mask = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\test_img\CRACK500_20160328_153020_1_721_mask.jpg'
    # washing_singel(read_img, read_mask, save_path)
    # washing_package(read_img, read_mask,trans_path, save_path)
    washing_publicSource(read_img,read_mask,trashcan_img,trashcan_mask)

############################
    # T1_P=r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\test_img\transfer\T1'
    # T2_P=r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\test_img\transfer\T2'
    # Save_P=r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\test_img\transfer\Tmix'
    # img_mix(T1_P,T2_P,Save_P)

