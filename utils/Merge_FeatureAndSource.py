import PIL.Image
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from CropAndConnect import CropAndConnect
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


#分割结果调色
def colorful(img):
    '''
    img:需要上色的图片
    save_path:存储路径
    '''
    img=Image.fromarray(img)
    palette = []
    for i in range(256):
        palette.extend((i, i, i))#加元素
    palette[:3 * 2] = np.array([[0,0,0],
                                [255,0,0],
                                 ], dtype='uint8').flatten()
    img.putpalette(palette)#调色板
    # img = img.convert('RGBA')
    # img.save(save_path)
    return img


def Crop_sourceMap_256(img,mask):#这里把原图裁成和被输出的预测图一样大小的格式以进行merge
    img_np=np.asarray(img)
    mask_np=np.asarray(mask)
    # print(img_np.shape[:,:,0],mask_np.shape[:,:])
    # if(img_np.shape[:,:,0]==mask_np.shape[:,:]):#当本身已经相等，不需要切割
    #     return img
    new_source= img_np[64:mask_np.shape[0]+64,64:mask_np.shape[1]+64]#高，宽
    new_img=Image.fromarray(new_source)
    return new_img#返回图像IMG格式

def Crop_sourceMap_448(img,mask):#这里把原图裁成和被输出的预测图一样大小的格式以进行merge
    img_np=np.asarray(img)
    mask_np=np.asarray(mask)
    # print(img_np.shape[:,:,0],mask_np.shape[:,:])
    # if(img_np.shape[:,:,0]==mask_np.shape[:,:]):#当本身已经相等，不需要切割
    #     return img
    new_source= img_np[112:mask_np.shape[0]+112,112:mask_np.shape[1]+112]#高，宽
    new_img=Image.fromarray(new_source)
    return new_img#返回图像IMG格式

def Crop_sourceMap_LHS(img):#这里把原图裁成和被输出的预测图一样大小的格式以进行merge
    img_np=np.asarray(img)
    new_source= img_np[64:15744+64,64:25728+64]#高，宽
    new_img=Image.fromarray(new_source)
    return new_img#返回图像IMG格式
##test切割原图
# img_source=Image.open('.././data/imgs_WJScracks/WJS.png')
# new_source=Crop_sourceMap(img_source)
# path='.././data/imgs_WJScracks/Cut_WJS.png'
# new_source.save(path)
######################Jiang

def new_merge(pred_dir,raw_dir,save_dir):#这个加载时间更长,但是merge效果更好
    pre_mask = cv2.imread(pred_dir, cv2.IMREAD_GRAYSCALE)  # opencv读取图像，直接返回numpy.ndarray 对象(高,宽),BGR
    # print(np.unique(pre_mask))#[0,255]
    pre_mask[pre_mask == 255] = 1
    pre_mask[pre_mask == 127] = 1
    # print(np.unique(pre_mask))#黑色部分值为0
    mask_color = colorful(pre_mask)  # 输入图像(array类型),上色特征图的储藏路径,返回img类型
    image1 = cv2.imread(raw_dir)  # 传入原始图（剪切）,
    # image1 = image1.convert('RGBA')#转四通道,不能存.jpg
    # mask_color=mask_color.convert('RGBA')
    # print(mask_color.size)
    width, height = mask_color.size
    mask_color = np.asarray(mask_color)
    img_t = image1.copy()  # 这样copy

    for j in range(width):
        for i in range(height):
            # print(mask_t[i][j][0:3])
            # 把黑色像素改成透明色
            print(mask_color[i][j])
            if mask_color[i][j] != 0:  #
                img_t[i][j] = (img_t[i][j] + [0, 0, 255]) / 2
                print(img_t[i][j])

    # image = Image.blend(image1, mask_t, 0.4)#合并两图
    cv2.imwrite(save_dir, img_t)  # 存储合并图

def old_merge(pred_dir,raw_dir,save_dir):
    pre_mask = cv2.imread(pred_dir, cv2.IMREAD_GRAYSCALE)  # opencv读取图像，直接返回numpy.ndarray 对象(高,宽),BGR
    # print(np.unique(pre_mask))#[0,255]
    pre_mask[pre_mask == 255] = 1
    pre_mask[pre_mask == 127] = 1
    # print(np.unique(pre_mask))#黑色部分值为0
    mask_color = colorful(pre_mask)  # 输入图像(array类型),上色特征图的储藏路径,返回img类型
    image1 = Image.open(raw_dir)# 传入原始图（剪切）
    image1 = image1.convert('RGBA')#转四通道,不能存.jpg
    mask_color=mask_color.convert('RGBA')
    # print(mask_color.size)
    image = Image.blend(image1, mask_color, 0.4)#合并两图

    image.save(save_dir)# 存储合并图


if __name__=='__main__':
    ####王家山path
    # pred_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\trainbyChen_Resnet34L2=1e-4bias=0_aug4ByJiang\WJS_DoubleResolution_e88.png"
    # save_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\trainbyChen_Resnet34L2=1e-4bias=0_aug4ByJiang\WJS_DoubleResolution_e88.png_merge.png"
    # pred_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\trainbyChen_Resnet34_Aug4_withASPPIncreaseL2=1e-4bias=0\WJS_0.25_e42.png"
    # save_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\trainbyChen_Resnet34_Aug4_withASPPIncreaseL2=1e-4bias=0\WJS_0.25_e42_merge.png"
    # pred_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\trainbyChen_Resnet34_aug4L2=1e-8ByJiang_e43\WJS_half_e43.png"
    # save_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\trainbyChen_Resnet34_aug4L2=1e-8ByJiang_e43\WJS_half_e43_merge.png"
    epoch=25
    name_LHS='LHS_e{}'.format(epoch)
    name1='WJS_e{}'.format(epoch)
    name2='WJS_half_e{}'.format(epoch)
    name3='WJS_25%_e{}'.format(epoch)
    name_kq6send='kq6send_e{}'.format(epoch)
    pred_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\trainby_wholeNTPTokq6Style_UGAITe92_TransUnet\\"+name1+'.png'
    # pred_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\trainby_NTPToWJSStyle_UGAITnewe33_TransUnet\\" + name_LHS + '.png'
    # save_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\trainbyNewNeuralTransferAndAug4_resnet34L2=1e-6\\"+name2+'_merge.png'
    save_dir=pred_dir.split('.')[0]+'_merge.'+pred_dir.split('.')[1]
    print(save_dir)
    # pred_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\trainbyChen_Resnet34L2=1e-4_aug4ByJiang\WJS_0.25_e62.png"
    # save_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\trainbyChen_Resnet34L2=1e-4_aug4ByJiang\WJS_0.25_e62_merge.png"
    # raw_dir = ".././data/Cut_LHS.png"
    #
    # raw_dir = ".././data/Cut_WJS_0.25.png"
    #
    # raw_dir = ".././data/Cut_WJS_half.png"
    #
    raw_dir = ".././data/Cut_WJS.png"
    # raw_dir = ".././data/Cut_kq6_send_256.png"
    # raw_dir = ".././data/Cut_WJS_448.png"
    # raw_dir = ".././data/WJS_DoubleResolution_Cut.png"
    # raw_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\Cut_LHS.png"

    # pred_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data/YanZheng/shenyuan_dom_clip2_e65.png"
    # save_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data/YanZheng/shenyuan_dom_clip2_e65_merge.tif"
    # raw_dir = ".././data/YanZheng/Cut_shenyuan_dom_clip2.tif"
    ####################合并单张图片(特征图和原图)
    old_merge(pred_dir, raw_dir, save_dir)
    # new_merge(pred_dir, raw_dir, save_dir)
    ############################
    # pred_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\DJ\DJI_20220731124015_0299_resizeResult_Open.png"
    # save_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\DJ\DJI_20220731124015_0299_resizeResult_Open_merge.png"
    # # raw_dir = ".././data/Cut_WJS_0.25.png"
    # raw_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\DJ\Cut_DJI_20220731124015_0299_resizeResult.JPG"
    ##### 莲花石path
    # pred_dir = ".././data/trainbyChen__Resnet34_L2=1e-6_aug7withASPP_e9/LHS.png"#25728*15744，原版25863*15948
    # save_dir = ".././data/trainbyChen__Resnet34_L2=1e-6_aug7withASPP_e9/LHS_merge.png"
    # raw_dir = ".././data/Cut_LHS.png"
    #######################################合并一组图片
    # list_raw=os.listdir(raw_dir)#返回元组
    # list_pred=os.listdir(pred_dir)
    # for filename in list_raw:
    #     print(filename)
    #     split=os.path.splitext(filename)
    #     print(split[0])#输出图片名
    #     pre_mask=np.asarray(Image.open(pred_dir+'/'+split[0]+'_OUT'+split[1]))#Jiang
    #     # pre_mask = cv2.imread(pred_dir+'/WJS.jpg',cv2.IMREAD_GRAYSCALE)#opencv读取图像，直接返回numpy.ndarray 对象(高,宽)
    #     pre_mask[pre_mask==255] = 1
    #     # print(np.unique(pre_mask))#黑色部分值为0
    #     mask_color = colorful(pre_mask)#输入图像(array类型),上色特征图的储藏路径,返回img类型
    #     image1=Image.open(raw_dir+'/'+filename)#传入原始图（剪切）
    #     image1 = image1.convert('RGBA')
    #     mask_color=mask_color.convert('RGBA')
    #     image = Image.blend(image1, mask_color, 0.4)#合并两图
    #     image.save(save_dir+'/'+split[0]+'_Merge'+split[1])#存储合并图

    ###切割图片
    # souce_img_path=r'E:\jiangshan\U-net\Pytorch-UNet\data\kq6_send.png'
    # souce_mask_path=r'E:\jiangshan\U-net\Pytorch-UNet\data\trainby_wholeNTPTokq13Style_UGAITe100_TransUnet\kq6_send_e16.png'
    # save_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\Cut_kq6_send_256.png"
    # souce_img=Image.open(souce_img_path).convert('RGB')
    # souce_mask=Image.open(souce_mask_path)
    # new_img=Crop_sourceMap_256(souce_img,souce_mask)
    # new_img.save(save_dir)
    ##################################Chen
    # for i,f in tqdm(enumerate(pred_paths)):
    #     pred_path = os.path.join(pred_dir,f)
    #     print(pred_path)
    #     pred = cv2.imread(pred_path,cv2.IMREAD_GRAYSCALE)
    #     print(np.unique(pred))
    #     pred[pred==255] = 1
    #     print(np.unique(pred))
    #     blend_name = f.split(".")[0] + "_colorful.png"
    #     print(blend_name)
    #     img_color = colorful(pred, os.path.join(save_dir,blend_name))
    #
    #     image1_pth = os.path.join(raw_dir, raw_paths[i])
    #     print(image1_pth)
    #     image1 = Image.open(image1_pth)
    #     image2 = img_color
    #     image1 = image1.convert('RGBA')
    #     image2 = image2.convert('RGBA')
    #     # 两幅图像进行合并时，按公式：blended_img = img1 * (1 – alpha) + img2* alpha 进行
    #     image = Image.blend(image1, image2, 0.4)
    #     blend_save_pth = f.split(".")[0] + "_colorful_blend.png"
    #     blend_save_pth = os.path.join(save_dir,blend_save_pth)
    #     print("xxx",blend_save_pth)
    #     image.save(blend_save_pth)