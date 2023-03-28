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

def Crop_sourceMap_WJS(img,mask):#这里把原图裁成和被输出的预测图一样大小的格式以进行merge
    img_np=np.asarray(img)
    mask_np=np.asarray(mask)
    # print(img_np.shape[:,:,0],mask_np.shape[:,:])
    # if(img_np.shape[:,:,0]==mask_np.shape[:,:]):#当本身已经相等，不需要切割
    #     return img
    new_source= img_np[64:mask_np.shape[0]+64,64:mask_np.shape[1]+64]#高，宽
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
if __name__=='__main__':
    ####王家山path
    pred_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\trainbyPublic-transfer_resnet34L2=1e-6\WJS_e44.png"
    save_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\trainbyPublic-transfer_resnet34L2=1e-6\WJS_e44_merge.png"
    # pred_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\trainbyChen_Resnet34L2=1e-4bias=0_aug4ByJiang\LHS_e65.png"
    # save_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\trainbyChen_Resnet34L2=1e-4bias=0_aug4ByJiang\LHS_e65_merge.png"
    # raw_dir = ".././data/Cut_LHS.png"
    raw_dir = ".././data/Cut_WJS.png"
    # raw_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\Cut_LHS.png"

    # pred_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data/YanZheng/shenyuan_dom_clip2_e65.png"
    # save_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data/YanZheng/shenyuan_dom_clip2_e65_merge.tif"
    # raw_dir = ".././data/YanZheng/Cut_shenyuan_dom_clip2.tif"

    #######
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
    ####################合并单张图片(特征图和原图)
    ## pre_mask=np.asarray(Image.open(pred_dir))#这个用了会报错
    pre_mask = cv2.imread(pred_dir,cv2.IMREAD_GRAYSCALE)#opencv读取图像，直接返回numpy.ndarray 对象(高,宽)
    # print(np.unique(pre_mask))#[0,255]
    pre_mask[pre_mask==255] = 1
    pre_mask[pre_mask==127] = 1
    # print(np.unique(pre_mask))#黑色部分值为0
    mask_color = colorful(pre_mask)#输入图像(array类型),上色特征图的储藏路径,返回img类型
    image1=Image.open(raw_dir)#传入原始图（剪切）
    image1 = image1.convert('RGBA')#转四通道,不能存.jpg
    mask_color=mask_color.convert('RGBA')
    image = Image.blend(image1, mask_color, 0.4)#合并两图
    image.save(save_dir)#存储合并图
    ###切割图片
    # souce_img_path=r'E:\jiangshan\U-net\Pytorch-UNet\data\YanZheng/shenyuan_dom_clip2.tif'
    # souce_mask_path=r'E:\jiangshan\U-net\Pytorch-UNet\data\YanZheng/shenyuan_dom_clip2_e65.png'
    # save_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\YanZheng/Cut_shenyuan_dom_clip2.tif"
    # souce_img=Image.open(souce_img_path).convert('RGB')
    # souce_mask=Image.open(souce_mask_path)
    # new_img=Crop_sourceMap_WJS(souce_img,souce_mask)
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