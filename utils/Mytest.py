import cv2.cv2
import albumentations as A
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import albumentations.pytorch
from torch.utils import data
import glob
import os
from PIL import Image
import numpy as np


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




if __name__ == '__main__':
    # input_path=r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\patches\kq6_dom'
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
    checkpoint_Path = 'checkpoints/Resnet34/NeuralTransfer_L2=1e-6bias=0/'
    list=checkpoint_Path.split('/')
    print(list[2])







