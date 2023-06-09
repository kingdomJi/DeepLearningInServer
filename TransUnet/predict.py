"""
该文件仅针对TransUnet模型预测，包含读取单个/多个图片并预测。以及求F1/PR/IOU参数等功能，评价函数所需参数将通过cvs文件存储
"""
import argparse
import logging
import os
import sklearn.metrics
from SegNet import segNet_model
import numpy as np
import cv2
import csv
import torch
import torch.nn.functional as F
from PIL import Image
from PIL import ImageFile
from torch.utils import data
import glob
ImageFile.LOAD_TRUNCATED_IMAGES = True#
Image.MAX_IMAGE_PIXELS = None#导入以防止因为图片过大而报错
import ttach as tta
from torchvision import transforms
from DeepCrack.codes.model.deepcrack import DeepCrack
from utils.CropAndConnect import CropAndConnect
from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from MyResnet import ResNet_baseLine,ResnetWithASPP_model,ResNetWithASPP_FPN,ResUNet
from ResUNetPlusPlus import resunetPP_pytorch
from TransUnet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUnet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):#out_threshold是阈值，当classes=1时调用到该值
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        #####################Jiang
        # output=flip_pred(net,img)#旋转加原图，三合一
        ######################
        print(img.shape)#(1,3,256,256)
        output = net(img)#原
        # output=TTA(net,img)#Jiang,增强预测
        # print(output.shape)#(classes=2时,size=[1, 2, 224, 224]),值为[-1.9345, -2.8159, -2.9056,  ..., -3.5215, -2.9572, -2.0101],[ 1.8863,  2.9546,  3.1037,  ...,  4.0547,  3.3417,  2.5655]格式
        # output的格式是classes层的概率图，概率尚且未归一化
        # print(output)

            # output= torch.tensor([item.cpu().detach().numpy() for item in output]).cuda()#Jiang,转tensor格式
        probs = torch.sigmoid(output)[0]#归一,
        tf = transforms.Compose([#用Compose把多个步骤整合到一起
        transforms.ToPILImage(),
        transforms.Resize((full_img.size[1], full_img.size[0])),#指定图片（高h，宽w）,size方法返回的是（宽，高）
        transforms.ToTensor(),
        ])

        full_mask = tf(probs.cpu()).squeeze()#将预测mask转化成原始输入图像大小，a.squeeze() 就是去掉a中的维数为一的维度比如是一行或者一列这种
                                             #c维度为(1，2，3) c.squeeze()将返回维度变为(2，3)
        # print(full_mask.shape)#(2,448,448)

        return F.one_hot(full_mask.argmax(dim=0), net.num_classes).permute(2, 0, 1).numpy(),full_mask[1]

        #找出两类中最大值（最大可能类）对应的位置，再把该位置填1，其他填0，相当于2分类了，1值处做为特征，返回三维（classes,高，宽）0，1图
        #classes维度下，只有一个为1，其他全为0，对应多分类中每个像素只对应一个类别
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default=r'checkpoints/TransUnet/UGATIT_WholePTokq6_optim=RMSprop_L2=1e-6/checkpoint_epoch25.pth', metavar='FILE',
    #parser.add_argument('--model', '-m',default='./checkpoints_UNet_Chen_Unenhance_e40/checkpoint_epoch40.pth',metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1, #0.5改1了
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    # parser.add_argument('--pr_curve','-pr',metavar='INPUT',nargs='+',help='precision_recall_curve')
    parser.add_argument('--crop1024','-c1024', action='store_true',default=False, help='crop whole img to size1024 to prediction')
    parser.add_argument('--crop448', '-c448', action='store_true', default=False,
                        help='crop whole img to size448 to prediction')
    parser.add_argument('--pr', '-pr', action='store_true', default=False,help='save PR curve')
    # for TransUnet
    parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    #action='store_true',在调用脚本时不调用该参，则默认为False,调用时则默认为True
    """
    'input'and'output'都可以是文件路径列表，input列表可以从.txt获取，
    '--pr':计算PR曲线，PR曲线计算需要测试集的mask和image路径，阈值会自动取。PR曲线上的一个点代表的是某阈值下，所有测试集图像的综合PR值
    nargs='+':可以输入多个参数，将输入参数返回一个成一个列表
    --is_whole_img：A large image that can not be predict by one input,cutting the image to pieces
    """
    return parser.parse_args()


def get_output_filenames(args):#原版
    def _generate_name(fn):#输出图像格式基于输入图像格式的变化函数
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'
    return args.output or list(map(_generate_name, args.input))#这里应该是输出的图片名称与输入的文件夹中图片名称对应?
#return True or False # 返回True，如果两边都是True，则or返回左边的

def get_output_filenames_J(in_files):#@JiangShan,当输入的是.txt时，调用这个方法为输出文件命名
    def _generate_name(fn):
        fn = fn.replace('images', 'pred')
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'
    return args.output or list(map(_generate_name,in_files))#这里输入的output不为空则返回output,否则根据input情况返回

def get_output_filenames_dir(in_files):#输入为目录
    output_files=[]
    for i in in_files:#i是绝对路径
        output_files.append(args.output[0]+'/'+i)
    return output_files#返回列表（保存图片的地址列表）
# def get_PR_curve_filename(in_files):#@JiangShan,true输入格式为0，1，pred格式为概率【0-1】


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:#当维度为2，高*宽，不需要减少维度
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:#当维度为3，classes*高*宽，需要减少classes维度，选classes里最大可能类的那个位置下标作为返回的值（二分类则是0，1图、n分类则是0，1，2，3...n-1图）,所以要*255/(shape[0]-1)
        if mask.shape[0]==1:#Jiang，当classes=1时
            return Image.fromarray((np.squeeze(mask) * 255).astype(np.uint8))#三维降二维
        return Image.fromarray((np.argmax(mask, axis=0) * 255/(mask.shape[0]-1)).astype(np.uint8))#Jiang，对0，1矩阵乘255
        #return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))#原版

def TTA(model,img):#测试时增强Test Time Augmentation
    transforms = tta.Compose(#增强流水线
        [
            # tta.HorizontalFlip(),
            # tta.Rotate90(angles=[0, 180]),
            # tta.Scale(scales=[1, 2,4]),
            tta.Scale(scales=[1,0.5]),
            # tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )
    tta_model = tta.SegmentationTTAWrapper(model, transforms)
    mask_pre=tta_model(img)
    return  mask_pre
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

class Metrics():#输入单张的net的输出，真实label【0，1】,设定的阈值
    def __init__(self, preds, labels,thresh):
        """
        :param preds: Tensor, [n,c,h,w]
        :param labels: Tensor, [n,c,h,w]
        """
        self.preds = F.sigmoid(preds).clone()
        self.preds = self.preds.cpu().detach().numpy()#tensor转np
        self.preds = (self.preds>thresh).astype(np.uint8)#大于阈值判定为1

        self.labels = labels.astype(np.uint8)

    def get_statistics(self):
        tp = np.sum((self.preds == 1) & (self.labels == 1))#且，交集
        fp = np.sum((self.preds == 1) & (self.labels == 0))
        fn = np.sum((self.preds == 0) & (self.labels == 1))
        tn = np.sum((self.preds == 0) & (self.labels == 0))
        return [tp,fp,fn,tn]#返回单张图片下的三个指标

def save_evaluate(csv_path,final_accuracy_all):
    # csv_name = "mffdn_prf_test.csv"  # 保存PR曲线的数据
    # csv_path = os.path.join(result, csv_name)  # 保存数据的真实路径
    with open(csv_path, "w+") as f:  # 写入结果文件到指定目录下，w+是读写打开，a+是在原文后追加写
        csv_writer = csv.writer(f)
        for prf in final_accuracy_all:
            csv_writer.writerow(prf)

#################Jiang
def flip_pred(net,image):#切入predict_img中,数据增强用
    pred1 = net(image.float())
    pred2 = net(torch.flip(image, [0, 3]).float())#转
    pred2 = torch.flip(pred2, [3, 0])#转回来
    pred3 = net(torch.flip(image, [0, 2]).float())
    pred3 = torch.flip(pred3, [2, 0])
    pred = pred1 + pred2 + pred3
    pred = pred / 3#
    # pred[pred > 0.5] = 1
    # pred[pred < 0.5] = 0
    return pred
#################################

def crop_1024(img,net):
    imgs = CropAndConnect.Crop_1024(img)  # 返回一个切割完的五维图片集数组
    # print(imgs.shape)
    imgs_h = imgs.shape[0]
    imgs_w = imgs.shape[1]
    imgs_hsize = imgs.shape[2]
    imgs_wsize = imgs.shape[3]
    #######################这里可以用循环把mask存在一个列表里
    # 传入Connect函数的masks.size=(块行数,块列数,classes,高，宽)
    classes = net.n_classes  # 这个地方要跟网络的classes保持一致
    masks = np.zeros((imgs_h, imgs_w, classes, imgs_hsize, imgs_wsize))  # masks是五维数组
    for n in range(imgs.shape[0]):
        for m in range(imgs.shape[1]):
            mask, probability_mask = predict_img(net=net,
                                                 full_img=Image.fromarray(imgs[n][m]),  # 这里还原成img的输入格式
                                                 scale_factor=args.scale,
                                                 out_threshold=args.mask_threshold,
                                                 device=device)
            # print(mask.shape)#（2，256，256）
            # print(np.unique(mask))#0或1
            masks[n][m] = mask

    # print('masks.shape'+masks.shape)
    ###################这里执行将mask拼接起来的Connect
    new_mask = CropAndConnect.Connect_1024(masks=masks, classes=classes)
    return new_mask

def crop_448(img,net):
    imgs = CropAndConnect.Crop(img)  # 返回一个切割完的五维图片集数组
    # print(imgs.shape)
    imgs_h = imgs.shape[0]
    imgs_w = imgs.shape[1]
    imgs_hsize = imgs.shape[2]
    imgs_wsize = imgs.shape[3]
    #######################这里可以用循环把mask存在一个列表里
    # 传入Connect函数的masks.size=(块行数,块列数,classes,高，宽)
    classes = net.n_classes  # 这个地方要跟网络的classes保持一致
    masks = np.zeros((imgs_h, imgs_w, classes, imgs_hsize, imgs_wsize))  # masks是五维数组
    for n in range(imgs.shape[0]):
        for m in range(imgs.shape[1]):
            mask, probability_mask = predict_img(net=net,
                                                 full_img=Image.fromarray(imgs[n][m]),  # 这里还原成img的输入格式
                                                 scale_factor=args.scale,
                                                 out_threshold=args.mask_threshold,
                                                 device=device)
            # print(mask.shape)#（2，256，256）
            # print(np.unique(mask))#0或1
            masks[n][m] = mask

    # print('masks.shape'+masks.shape)
    ###################这里执行将mask拼接起来的Connect
    new_mask = CropAndConnect.Connect(masks=masks, classes=classes)
    return new_mask

if __name__ == '__main__':
    args = get_args()
    PR = args.pr
    if len(os.path.splitext(args.input[0]))<2 or os.path.splitext(args.input[0])[1]!='.png':#输入为目录
        in_files=[]
        infile_names=os.listdir(args.input[0])
        for i in infile_names:
            in_files.append(args.input[0]+'/'+i)

        out_files = get_output_filenames_dir(infile_names)  # Jiang,当输入是文件夹时，调用该行
    else:
        in_files = args.input#原版，返回一个字符串列表，列表元素个数取决于是输入的元素个数，原作者大概想当输入为多个图片时，
            # 命令行输入的input参数为一个图片路径列表。比如用.txt保存所有输入图片的绝对路径，再用命令行调用该txt作为输入路径参数
        out_files = get_output_filenames(args)  # 原版，这里返回列表或单个值，取决于输入情况

    crop_1024=args.crop1024
    crop_448 = args.crop448
    # print(out_files)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = 2  # 默认为2
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')  # 调试错误时调用以查找错误来源

    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):#用enumerate会返回一个元组，前面是序号（对应后面的out_filename），后面是list下的文件名
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename).convert('RGB')#原,打开待预测图像,保持以RGB模式打开，防止四通道图片传入
        img = np.asarray(img)#这里是后面加的，把CropAndConnect.Crop里的开头转np拿到这里了
        # print(img.size)#(高，宽)
        if img.shape[0]*img.shape[1]>1000000 and crop_1024==False and crop_448==False: #当输入很大一幅图时调用这个地方，切割预测
            #######################这里执行Crop处理img
            imgs = CropAndConnect.Crop(img)  # 返回一个切割完的五维图片集数组
            print(imgs.shape)#()
            imgs_h = imgs.shape[0]
            imgs_w = imgs.shape[1]
            imgs_hsize = imgs.shape[2]
            imgs_wsize = imgs.shape[3]
            #######################这里可以用循环把mask存在一个列表里
            # 传入Connect函数的masks.size=(块行数,块列数,classes,高，宽)
            classes = net.num_classes#这个地方要跟网络的classes保持一致
            masks = np.zeros((imgs_h, imgs_w, classes, imgs_hsize, imgs_wsize))  # masks是五维数组
            for n in range(imgs.shape[0]):
                for m in range(imgs.shape[1]):
                    mask,probability_mask = predict_img(net=net,
                                       full_img=Image.fromarray(imgs[n][m]),  # 这里还原成img的输入格式
                                       scale_factor=args.scale,
                                       out_threshold=args.mask_threshold,
                                       device=device)
                    # print(mask.shape)#（2，256，256）
                    # print(np.unique(mask))#0或1
                    masks[n][m] = mask
            #print('masks.shape'+masks.shape)
            ###################这里执行将mask拼接起来的Connect
            new_mask = CropAndConnect.Connect(masks=masks, classes=classes)
        elif(crop_1024==True):
            imgs = CropAndConnect.Crop_1024(img)  # 返回一个切割完的五维图片集数组
            # print(imgs.shape)#(,,1024,1024,3)
            imgs_h = imgs.shape[0]
            imgs_w = imgs.shape[1]
            imgs_hsize = imgs.shape[2]
            imgs_wsize = imgs.shape[3]
            #######################这里可以用循环把mask存在一个列表里
            # 传入Connect函数的masks.size=(块行数,块列数,classes,高，宽)
            classes = net.num_classes  # 这个地方要跟网络的classes保持一致
            masks = np.zeros((imgs_h, imgs_w, classes, imgs_hsize, imgs_wsize))  # masks是五维数组
            for n in range(imgs.shape[0]):
                for m in range(imgs.shape[1]):
                    mask, probability_mask = predict_img(net=net,
                                                         full_img=Image.fromarray(imgs[n][m]),  # 这里还原成img的输入格式
                                                         scale_factor=args.scale,
                                                         out_threshold=args.mask_threshold,
                                                         device=device)
                    # print(mask.shape)#（2，256，256）
                    # print(np.unique(mask))#0或1
                    masks[n][m] = mask

            # print('masks.shape'+masks.shape)
            ###################这里执行将mask拼接起来的Connect
            new_mask = CropAndConnect.Connect_1024(masks=masks, classes=classes)
        elif (crop_448 == True):
            imgs = CropAndConnect.Crop_448(img)  # 返回一个切割完的五维图片集数组
            # print(imgs.shape)#(,,448,448,3)
            imgs_h = imgs.shape[0]
            imgs_w = imgs.shape[1]
            imgs_hsize = imgs.shape[2]
            imgs_wsize = imgs.shape[3]
            #######################这里可以用循环把mask存在一个列表里
            # 传入Connect函数的masks.size=(块行数,块列数,classes,高，宽)
            classes = net.num_classes  # 这个地方要跟网络的classes保持一致
            masks = np.zeros((imgs_h, imgs_w, classes, imgs_hsize, imgs_wsize))  # masks是五维数组
            for n in range(imgs.shape[0]):
                for m in range(imgs.shape[1]):
                    mask, probability_mask = predict_img(net=net,
                                                         full_img=Image.fromarray(imgs[n][m]),  # 这里还原成img的输入格式
                                                         scale_factor=args.scale,
                                                         out_threshold=args.mask_threshold,
                                                         device=device)
                    # print(mask.shape)#（2，256，256）
                    # print(np.unique(mask))#0或1
                    masks[n][m] = mask

            # print('masks.shape'+masks.shape)
            ###################这里执行将mask拼接起来的Connect
            new_mask = CropAndConnect.Connect_448(masks=masks, classes=classes)
        else:#当输入的单张图片并不大时调用原版
            new_mask, probability_mask= predict_img(net=net,
                               full_img=Image.fromarray(img),  # 这里还原成img的输入格式
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

        #########################JiangShan############
        if not args.no_save:
            out_filename = out_files[i]
            # print(new_mask.shape)#(2, 4736, 7296)
            #print(np.unique(new_mask))#new_mask值为0或1
            result = mask_to_image(new_mask)#result为图片格式
            # print(np.unique(np.asarray(result)))#0或255
            ##############Jiang
            result.save(out_filename)#原

            #####################
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, new_mask)


"""
使用范例
输入多个图片时，将它们放在同一目录下，如使用命令：
python TransUnet/predict.py -i ./data/WJS_cracks/images_256 -o ./data/train_TransUnetTransfer_NTPublicAndKq6_256_L2=1e-6/WJS_paches
输入单张图片时：python TransUnet/predict.py -i ./data/WJS.png -o ./data/trainbyChen_TransUnet_Public_L2=1e-6/WJS_e65.png

"""

