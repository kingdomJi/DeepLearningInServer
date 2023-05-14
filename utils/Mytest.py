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

class TestStringMethods(unittest.TestCase):#TestCase类：一个TestCase的实例就是一个测试用例什么是测试用例呢？就是一个完整的测试流程，
    # 包括测试前准备环境的搭建(setUp)，执行测试代码 (run)，以及测试后环境的还原(tearDown)。元测试(unit test)的本质也就在这里，
    # 一个测试用例是一个完整的测试单元，通过运行这个测试单元，可以对某一个问题进行验证。

    def test_upper(self):
        print('test1')

        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        print('test2')

        self.assertTrue('FOO'.isupper())

        self.assertFalse('Foo'.isupper())

    def test_split(self):

        s = 'hello world'
        print(s)

        self.assertEqual(s.split(), ['hello', 'world'])



class TestDataset(data.Dataset):#输入待预测图像和真实label的目录，且俩目录下的图片要一一对应
    """
    load test image and label
    """
    def __init__(self, image_dataroot, label_dataroot):
        self.image_root = image_dataroot
        self.label_root = label_dataroot
        self.image_list = glob.glob(os.path.join(self.image_root, "*.{}".format("png")))
        self.label_list = glob.glob(os.path.join(self.label_root, "*.{}".format("bmp")))
        assert len(self.image_list) == len(self.label_list)

        self.image_name = [image.split("\\")[-1].split(".")[0] for image in self.image_list]

    def __getitem__(self, item):
        img = Image.open(self.image_list[item]).convert('RGB')  # 原,打开待预测图像,保持以RGB模式打开，防止四通道图片传入
        img = np.asarray(img)
        # im = cv2.cvtColor(cv2.imread(self.image_list[item]), cv2.COLOR_BGR2RGB)
        # im = im.transpose((2,0,1)).astype(np.float32)
        lab = cv2.imread(self.label_list[item], cv2.IMREAD_GRAYSCALE)

        return img, lab/255.0, self.image_name[item]     # return (image, label, image_name)

    def __len__(self):
        return len(self.image_list)


def png2png(input_path,save_path):
    list_file=os.listdir(input_path)
    for i in list_file:
        mask = Image.open(input_path+'/'+i)
        mask = np.asarray(mask)
        cv2.imwrite(save_path+'/'+i, mask)

def save_path2txt(read_path_img,read_path_mask,read_path_style,save_path):#绝对路径
    img_list = os.listdir(read_path_img)
    mask_list=os.listdir(read_path_mask)
    style_list=os.listdir(read_path_style)
    str = '\n'
    f = open(save_path, "w")
    for i in range(len(img_list)):
        f.write('python neural_style.py --content_img_dir ./Transfer_img --content_img '+img_list[i]+str+\
                ' --style_imgs_dir ./kq6 --style_imgs '+style_list[i%len(style_list)]+'\t'+style_list[i+1%len(style_list)]+'\t'+style_list[i+2%len(style_list)]\
                +' --style_mask --style_mask_imgs '+mask_list[i]+' --img_output_dir ./output_test --img_name public')
    f.close()

def img_mix(img_path_T,img_path,mask_white_path,mask_black_path,save_path):#将迁移后的public，mask部分保留原图特征
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

if __name__ == '__main__':
    # input_t_path=r'E:\jiangshan\UGATIT_old\results\UGATIT_publicToWJS_e59'
    # input_s_path=r'E:\jiangshan\U-net\UGATIT\dataset\publicToWJS\testA'
    # input_mask_w=r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\Transfer_mask_white'
    # input_mask_b = r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\Transfer_mask'
    # save_path=r'E:\jiangshan\UGATIT_old\results\UGATIT_publicToWJS_e59_withMask'
    # img_mix(input_t_path,input_s_path,input_mask_w,input_mask_b,save_path)


    # true_mask_path=r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\patches\kq6_label_seg'
    # image_dtaroot = input_path
    # label_dataroot = true_mask_path
    # test_dataset = TestDataset(image_dtaroot, label_dataroot)  # 测试数据集
    #
    # for i, list in enumerate(test_dataset):
    #     # print(i, filename, lab, img_name)
    #     print(i, list[0].shape, list[1].shape, list[2])
    # save_path=r'E:\jiangshan\U-net\Pytorch-UNet\data\WJS_trainbyChen_UnetSeg_e100\000000292_448.png'
    # pathroot='./data/evaluate_demo/pred_results'
    # csv_name = "test.csv"  # 保存PR曲线的数据
    # csv_path = pathroot+ '/'+ csv_name
    # print(r'{}'.format(csv_path))
    # png2png(input_path,save_path)

    # print(np.unique(mask))

    # checkpoint_Path = 'checkpoints/Resnet34/NeuralTransfer_L2=1e-6bias=0/'
    # list=checkpoint_Path.split('/')
    # print(list[2])




#unittest.main()会自动查找所有继承unittest.TestCase的测试类，并运行其中的所有测试方法。
    unittest.main()

