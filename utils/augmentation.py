import copy
import os

from PIL import Image
from torchvision import transforms
import cv2.cv2
import albumentations as A
import numpy as np
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import albumentations.pytorch
class Aug():
    def __init__(self,open_path_souce, open_path_label_seg, save_path_source, save_path_label_seg, suffix):
        self.open_path_souce=open_path_souce
        self.open_path_label_seg=open_path_label_seg
        self.save_path_source=save_path_source
        self.save_path_label_seg=save_path_label_seg
        self.suffix=suffix
    def augmentation(self):
        num = np.random.randint(100, 500)
        max_height =np.random.randint(1, 4)
        max_width = np.random.randint(1, 4)
        fill_value=np.random.randint(0, 255,size=(3))#随机RGB
        trans = A.Compose(  # 创建流水线
            [
                #翻转
                # A.HorizontalFlip(p=1),  # 水平翻转，p这个参数代表进行这个操作的概率
                # 形变
                # A.ElasticTransform(alpha=155, sigma=210, alpha_affine=157, p=1), #弹性变形
                A.ElasticTransform(alpha=np.random.randint(1, 150), sigma=np.random.randint(50, 100),
                                   alpha_affine=np.random.randint(50, 100), p=1),
                # A.Perspective(p=1),#透视变换,执行图像输入的随机四点透视变换
                # A.GridDistortion(p=1),# 网格畸变 感觉能变瘦和变胖是随机的
                # A.RandomSizedCrop(min_max_height=(64, 128), height=256, width=256, p=1),#随机大小裁切放大
                # 裁切后的 height=256 width=256， min_max_height：在(64, 128)这个范围中随机裁切一个尺寸，裁切下来后再放大到height*width

                # 随机网格洗牌
                # A.RandomGridShuffle(grid=(3, 3), always_apply=False, p=1.0),#参数：将图像以网格方式生成几块，并随机打乱

                # 在图像上生成矩形区域。mask不随img变
                # A.CoarseDropout(max_holes=np.random.randint(100, 500), max_height=np.random.randint(1, 4), max_width=np.random.randint(1, 4), min_holes=None, min_height=None,min_width=None,
                #                 fill_value=np.random.randint(0, 255,size=(3)), always_apply=False, p=1),
                # 在图像上生成矩形区域。mask不随img变
                # A.OneOf(
                #     [
                #         A.CoarseDropout(max_holes=num, max_height=2, max_width=2, min_holes=None, min_height=None,
                #                         min_width=None,
                #                         fill_value=[40, 83, 77], always_apply=False, p=1),
                #         A.CoarseDropout(max_holes=num, max_height=2, max_width=2, min_holes=None, min_height=None,
                #                         min_width=None,
                #                         fill_value=[128, 178, 194], always_apply=False, p=1),
                #     ]
                #     , p=1),
                #输入BGR
                #40,83,77(林地绿)\22,102,76（某绿）#128,178,194(砂色)

                #像素值变换
                # A.HueSaturationValue(p=1),#随机改变图片的 HUE(色相)、饱和度和值
                # # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=.2, brightness_by_max=True,
                # #                                         p=1),#默认参数亮度和对比度都是0.2，将亮度brightness_limit调到0.8后发现图明显变亮
                # A.CLAHE(clip_limit=2, p=1),# 对输入图像应用对比度受限自适应直方图均衡化,有一种更清晰，对比更强烈的视觉效果
                # A.IAAEmboss(p=1),#浮雕
                # A.IAASharpen(p=1),#锐化
                # A.RandomGamma(gamma_limit=148, p=0.5)#伽马变换
                #清晰度变换，模糊处理
                # A.GaussianBlur(blur_limit=7, p=1),#高斯模糊
                A.Blur(blur_limit=7, p=1)#box模糊：

                # A.OneOf([#模糊处理，随机
                #     A.MotionBlur(p=0.2),
                #     A.MedianBlur(blur_limit=3, p=0.1),
                #     A.Blur(blur_limit=3, p=0.1),
                # ], p=1),
                # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                # A.OneOf([
                #     A.OpticalDistortion(p=0.3),
                #     A.GridDistortion(p=0.1),
                #     A.IAAPiecewiseAffine(p=0.3),
                # ], p=0.5),
                # A.HueSaturationValue(p=0.3),
            ])
        return trans

    def aug_seg(self):  # 传入source地址，保存地址，数据增强流水线实例,该次的增强标记
        '''
            这个函数对img和mask进行对应增强并保存
        '''
        img_list = os.listdir(self.open_path_souce)
        mask_seg_list = os.listdir(self.open_path_label_seg)  #
        for i in range(len(mask_seg_list)):  # 读取文件夹下每一个图片
            trans = self.augmentation()  # 每个图片有单独的增强器
            image_souce = cv2.imread(self.open_path_souce + img_list[i])
            mask = cv2.imread(self.open_path_label_seg + mask_seg_list[i], cv2.IMREAD_GRAYSCALE)  # 读灰度图
            # transformed = transform(image=image)
            # tranformed_image = transformed['image']#将图片数据传递给transform（很明显这是个可调用的对象）的image参数，它会返回一个处理完的对象，
            transformed = trans(image=image_souce, mask=mask)
            transformed_image = transformed['image']
            transformed_mask_seg = transformed['mask']
            # transformed_mask_seg = cv2.cvtColor(transformed_mask_seg, cv2.COLOR_BGR2GRAY)#转灰度图
            retval = cv2.imwrite(
                self.save_path_source + os.path.splitext(img_list[i])[0] + self.suffix + os.path.splitext(img_list[i])[1],
                transformed_image)
            retval2 = cv2.imwrite(self.save_path_label_seg + os.path.splitext(mask_seg_list[i])[0] + self.suffix +
                                  os.path.splitext(mask_seg_list[i])[1], transformed_mask_seg)


def augmentation():#针对单张图片的增强方法
    num = np.random.randint(100, 500)
    max_height =np.random.randint(1, 4)
    max_width = np.random.randint(1, 4)
    fill_value=np.random.randint(0, 255,size=(3))#随机RGB
    trans = A.Compose(  # 创建流水线
        [
            #翻转
            # A.HorizontalFlip(p=1),  # 水平翻转，p这个参数代表进行这个操作的概率
            # 形变
            # A.ElasticTransform(alpha=155, sigma=210, alpha_affine=157, p=1), #弹性变形
            A.ElasticTransform(alpha=np.random.randint(1,150), sigma=np.random.randint(50,100), alpha_affine=np.random.randint(50,100),p=1),  # 弹性变形
            # A.Perspective(p=1),#透视变换,执行图像输入的随机四点透视变换
            # A.GridDistortion(p=1),# 网格畸变 感觉能变瘦和变胖是随机的
            # A.RandomSizedCrop(min_max_height=(64, 128), height=256, width=256, p=1),#随机大小裁切放大
            # 裁切后的 height=256 width=256， min_max_height：在(64, 128)这个范围中随机裁切一个尺寸，裁切下来后再放大到height*width

            # 随机网格洗牌
            # A.RandomGridShuffle(grid=(3, 3), always_apply=False, p=1.0),#参数：将图像以网格方式生成几块，并随机打乱

            # 在图像上生成矩形区域。mask不随img变
            # A.CoarseDropout(max_holes=np.random.randint(100, 500), max_height=np.random.randint(1, 4), max_width=np.random.randint(1, 4), min_holes=None, min_height=None,min_width=None,
            #                 fill_value=np.random.randint(0, 255,size=(3)), always_apply=False, p=1),
            # 在图像上生成矩形区域。mask不随img变
            # A.OneOf(
            #     [
            #         A.CoarseDropout(max_holes=num, max_height=2, max_width=2, min_holes=None, min_height=None,
            #                         min_width=None,
            #                         fill_value=[40, 83, 77], always_apply=False, p=1),
            #         A.CoarseDropout(max_holes=num, max_height=2, max_width=2, min_holes=None, min_height=None,
            #                         min_width=None,
            #                         fill_value=[128, 178, 194], always_apply=False, p=1),
            #     ]
            #     , p=1),
            #输入BGR
            #40,83,77(林地绿)\22,102,76（某绿）#128,178,194(砂色)

            #像素值变换
            # A.HueSaturationValue(p=1),#随机改变图片的 HUE(色相)、饱和度和值
            # # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=.2, brightness_by_max=True,
            # #                                         p=1),#默认参数亮度和对比度都是0.2，将亮度brightness_limit调到0.8后发现图明显变亮
            # A.CLAHE(clip_limit=2, p=1),# 对输入图像应用对比度受限自适应直方图均衡化,有一种更清晰，对比更强烈的视觉效果
            # A.IAAEmboss(p=1),#浮雕
            # A.IAASharpen(p=1),#锐化
            # A.RandomGamma(gamma_limit=148, p=0.5)#伽马变换
            #清晰度变换，模糊处理
            # A.GaussianBlur(blur_limit=7, p=1),#高斯模糊
            # A.Blur(blur_limit=7, p=1)#box模糊：

            # A.OneOf([#模糊处理，随机
            #     A.MotionBlur(p=0.2),
            #     A.MedianBlur(blur_limit=3, p=0.1),
            #     A.Blur(blur_limit=3, p=0.1),
            # ], p=1),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            # A.OneOf([
            #     A.OpticalDistortion(p=0.3),
            #     A.GridDistortion(p=0.1),
            #     A.IAAPiecewiseAffine(p=0.3),
            # ], p=0.5),
            # A.HueSaturationValue(p=0.3),
        ])
    return trans

def delete_aug_seg(souceImg_path,souceMask_path,save_path_source,save_path_SegLabel,suffix):#aug_num是增强集合的标记

    img_list=os.listdir(souceImg_path)
    mask_list=os.listdir(souceMask_path)
    for i in range(len(mask_list)):#读取文件夹下每一个图片
        os.remove(save_path_source+os.path.splitext(img_list[i])[0]+suffix+os.path.splitext(img_list[i])[1])
        os.remove(save_path_SegLabel+os.path.splitext(mask_list[i])[0]+suffix+os.path.splitext(mask_list[i])[1])


def delete_aug_line(img_list,mask_list,save_path_source,save_path_LineLabel,suffix):#aug_num是增强集合的标记
    for i in range(len(mask_list)):#读取文件夹下每一个图片
        os.remove(save_path_source+os.path.splitext(img_list[i])[0]+suffix+os.path.splitext(img_list[i])[1])
        os.remove(save_path_LineLabel + os.path.splitext(mask_list[i])[0]+suffix+os.path.splitext(mask_list[i])[1])


def aug_test(img_path,save_path,trans):
    image=cv2.imread(img_path)
    img_aug=trans(image=image)['image']
    retval = cv2.imwrite(save_path,img_aug)
    return retval

def aug_line(open_path_souce,open_path_label_line,save_path_source,save_path_label_line,trans,suffix):#传入source地址，保存地址，数据增强流水线实例,该次的增强标记
    img_list = os.listdir(open_path_souce)
    mask_line_list = os.listdir(open_path_label_line)  # 线任务
    for i in range(len(mask_line_list)):#读取文件夹下每一个图片
        image_souce=cv2.imread(open_path_souce+img_list[i])
        mask = cv2.imread(open_path_label_line + mask_line_list[i])
    # transformed = transform(image=image)
    # tranformed_image = transformed['image']#将图片数据传递给transform（很明显这是个可调用的对象）的image参数，它会返回一个处理完的对象，
        transformed = trans(image=image_souce, mask=mask)
        transformed_image = transformed['image']
        transformed_mask_line=transformed['mask']
        transformed_mask_line = cv2.cvtColor(transformed_mask_line, cv2.COLOR_BGR2GRAY)  # 24位转8位灰度图
        retval = cv2.imwrite(save_path_source + os.path.splitext(img_list[i])[0] + suffix + os.path.splitext(img_list[i])[1],
            transformed_image)
        retval2 = cv2.imwrite(save_path_label_line + os.path.splitext(mask_line_list[i])[0] + suffix +
                              os.path.splitext(mask_line_list[i])[1], transformed_mask_line)



def aug_seg_excludemask(open_path_souce,open_path_label_seg,save_path_source,save_path_label_line,trans,suffix):
    '''
        这个函数只对img进行增强并保存，mask保存原版
    '''
    img_list = os.listdir(open_path_souce)
    mask_seg_list = os.listdir(open_path_label_seg)  #
    for i in range(len(mask_seg_list)):  # 读取文件夹下每一个图片
        image_souce = cv2.imread(open_path_souce + img_list[i])
        mask = cv2.imread(open_path_label_seg + mask_seg_list[i])#默认读RGB
        # transformed = transform(image=image)
        # tranformed_image = transformed['image']#将图片数据传递给transform（很明显这是个可调用的对象）的image参数，它会返回一个处理完的对象，
        transformed = trans(image=image_souce)
        transformed_image = transformed['image']
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)#转灰度图
        retval = cv2.imwrite(
            save_path_source + os.path.splitext(img_list[i])[0] + suffix + os.path.splitext(img_list[i])[1],
            transformed_image)
        retval2 = cv2.imwrite(save_path_label_line + os.path.splitext(mask_seg_list[i])[0] + suffix +
                              os.path.splitext(mask_seg_list[i])[1], mask)#mask没有做增强

def aug_segAndLine():
    pass

# transformed_SegMask =transformed_masks[0]
# transformed_LineMask=transformed_masks[1]
# transformed_mask_seg = cv2.cvtColor(transformed_SegMask, cv2.COLOR_BGR2GRAY)#24位转8位灰度图
# transformed_mask_line=cv2.cvtColor(transformed_LineMask, cv2.COLOR_BGR2GRAY)#24位转8位灰度图



def resize(path_img,path_save,mask=False,size=(256,256)):
    img_namelist=os.listdir(path_img)
    if mask==False:
        for each in img_namelist:
            print('读取:',each)
            img=Image.open(path_img+'/'+each)
            img_resize=img.resize(size)
            img_savePath=path_save+'/'+each#os.path.splitext(each)[0]+'.png'
            img_resize.save(img_savePath)
    if mask==True:
        for each in img_namelist:
            print('读取:', each)
            img=Image.open(path_img+'/'+each)
            img_resize=img.resize(size)
            img_savePath=path_save+'/'+each#不改后缀
            img_resize.save(img_savePath)

def Copy_source(open_path_souce, open_path_label_seg, save_path_source, save_path_label_seg):
    img_list = os.listdir(open_path_souce)
    mask_seg_list = os.listdir(open_path_label_seg)  #
    for i in range(len(mask_seg_list)):  # 读取文件夹下每一个图片
        image_souce = cv2.imread(open_path_souce + img_list[i])
        mask = cv2.imread(open_path_label_seg + mask_seg_list[i])
        retval = cv2.imwrite(save_path_source +img_list[i],image_souce)
        retval2 = cv2.imwrite(save_path_label_seg + mask_seg_list[i], mask)

def mask_24Transto8bit(path):
    mask_list = os.listdir(path)
    for i in range(len(mask_list)):
        mask = cv2.imread(path +mask_list[i])
        transformed_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # 24位转8位灰度图
        retval = cv2.imwrite(path + mask_list[i],transformed_mask)#写回原文件覆盖


if __name__=='__main__':
    ##test查看单张增强效果
    # image_path = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\test_img\kq6_100700.png'
    # save_path=r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\test_img\kq6_100700_ElasticTransform.png'
    # # trans_tool = transforms.Resize(256, 256)  # 设定筛选器图像器大小
    # # img_resize = trans_tool(img)
    # trans=augmentation()
    # print(aug_test(image_path,save_path,trans))
    ########################resize一个目录
    path_img = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\publicAndkq6\kq6'
    path_save = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\publicAndkq6\kq6_448'
    resize(path_img, path_save, mask=False, size=(448, 448))
    #########单张resize
    # path_img = r'..\data\WJS.png'
    # path_save = r'..\data\WJS_DoubleResolution.png'
    # # path_img = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\向日葵.jpg'
    # # path_save = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\向日葵.jpg'
    # img = Image.open(path_img)
    # print(np.size(img, 0))#h
    # print(np.size(img, 1))#w
    # img_resize = img.resize(size=((int(np.size(img, 1)*2)),int(np.size(img, 0)*2)))#先w后h
    # # img_resize = img.resize(size=(2693, 3368))  # 先w后h
    # img_resize.save(path_save)
    # ######################################
    # 删除某一次的增强数据集
    # delete_aug(img_list,mask_seg_list,save_path_source,save_path_label_seg,save_path_label_line,3)

    #####对Mask_seg进行增强
    # open_path_souce=r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\patches\kq6_dom\\'
    # open_path_label_seg=r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\patches\kq6_label_seg\\'
    # # open_path_label_line=r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\patches\kq6_label_line\\'
    # save_path_source = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\aug7_seg\kq6_dom_aug\\'
    # save_path_label_seg = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\aug7_seg\kq6_label_seg_aug\\'
    # # # save_path_label_line = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\kq6_label_line_aug\\'
    # # suffix='aug_4'
    # # myAug=Aug(open_path_souce, open_path_label_seg, save_path_source, save_path_label_seg, suffix)
    # # myAug.aug_seg()
    # mask_24Transto8bit(save_path_label_seg)#RGB转灰度图

#####复制源文件到指定目录
    # Copy_source(open_path_souce, open_path_label_seg, save_path_source, save_path_label_seg)
    # mask_24Transto8bit(save_path_label_seg)#RGB转灰度图
###删除某次增强，输入原始图像文件路径，保存增强结果的文件路径,后缀
    # delete_aug_seg(open_path_souce,open_path_label_seg,save_path_source,save_path_label_seg, suffix)


#######对Mask_line增强
    # savePath_source = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\aug_line\kq6_dom_aug\\'
    # savePath_label_line = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\aug_line\kq6_label_line_aug\\'
    # suffix='aug_3'#suffix是文件后缀，用来标识某一次增强

    # #####
    # trans = augmentation()
    # aug_line(open_path_souce,open_path_label_line, savePath_source,savePath_label_line,trans,suffix)


