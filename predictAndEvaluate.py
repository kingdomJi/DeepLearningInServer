import argparse
import logging
import os
import glob
import sklearn.metrics
import csv
from SegNet import segNet_model
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True#
Image.MAX_IMAGE_PIXELS = None#导入以防止因为图片过大而报错
import cv2
from torchvision import transforms
from DeepCrack.codes.model.deepcrack import DeepCrack
from utils.CropAndConnect import CropAndConnect
from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from ResUnet.ResUnet_model import resUnet34

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5#out_threshold是阈值，当classes=1时调用到该值
                ):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        #####################Jiang
        # output=flip_pred(net,img)#旋转加原图，三合一
        ######################
        output = net(img)#原
        # print(output.shape)#(classes=2时,size=[1, 2, 224, 224]),值为[-1.9345, -2.8159, -2.9056,  ..., -3.5215, -2.9572, -2.0101],[ 1.8863,  2.9546,  3.1037,  ...,  4.0547,  3.3417,  2.5655]格式
        # output的格式是classes层的概率图，概率尚且未归一化
        # print(output)
        if net.n_classes > 1:#多分类问题
            probs = F.softmax(output, dim=1)[0]#归一
            # print(probs[0])
            # b=probs[1]>0.5
            # print(len(probs[1][b]))#输入classes为2情况，size=[2, 224, 224],probs[1]是特征类的概率图
        else:
            # output= torch.tensor([item.cpu().detach().numpy() for item in output]).cuda()#Jiang,转tensor格式
            probs = torch.sigmoid(output)[0]#归一
        tf = transforms.Compose([#用Compose把多个步骤整合到一起
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),#指定图片（高h，宽w）,size方法返回的是（宽，高）
            transforms.ToTensor(),
        ])

        full_mask = tf(probs.cpu()).squeeze()#将预测mask转化成原始输入图像大小，a.squeeze() 就是去掉a中的维数为一的维度比如是一行或者一列这种
                                             #c维度为(1，2，3) c.squeeze()将返回维度变为(2，3)
        # print(full_mask.shape)#(2,448,448)
    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy(),full_mask[1]#full_mask[1]是特征类的概率图
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy(),full_mask[1]


        #找出两类中最大值（最大可能类）对应的位置，再把该位置填1，其他填0，相当于2分类了，1值处做为特征，返回三维（classes,高，宽）0，1图
        #classes维度下，只有一个为1，其他全为0，对应多分类中每个像素只对应一个类别
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./checkpoints/U-net/data_Chen_new_kq6_aug2ByJiang_e100/checkpoint_epoch100.pth', metavar='FILE',
    #parser.add_argument('--model', '-m',default='./checkpoints_UNet_Chen_Unenhance_e40/checkpoint_epoch40.pth',metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--true_mask', '-tm', metavar='INPUT', nargs='+', help='Filenames of true masks', required=True)
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
    parser.add_argument('--is_whole_img','-iwi', action='store_true',default=False, help='whether the image is a complete one')
    #action='store_true',在调用脚本时不调用该参，则默认为False,调用时则默认为True
    """
    'input'and'output'都可以是文件路径列表，input列表可以从.txt获取，
    '--pr':计算PR曲线，PR曲线计算需要测试集的mask和image路径，阈值会自动取。PR曲线上的一个点代表的是某阈值下，所有测试集图像的综合PR值
    nargs='+':可以输入多个参数，将输入参数返回一个成一个列表
    --is_whole_img：A large image that can not be predict by one input,cutting the image to pieces
    """
    return parser.parse_args()


def get_output_filenames(args):#输出图像格式基于输入图像格式的变化函数
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))#这里应该是输出的图片名称与输入的文件夹中图片名称对应?
#return True or False # 返回True，如果两边都是True，则or返回左边的

def get_output_filenames_J(in_files):#@JiangShan,当输入的是文件夹时，调用这个方法为输出文件命名
    def _generate_name(fn):
        fn = fn.replace('images', 'result')
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'
    return args.output or list(map(_generate_name,in_files))#这里不懂什么情况

# def get_PR_curve_filename(in_files):#@JiangShan,true输入格式为0，1，pred格式为概率【0-1】


def mask_to_image(mask: np.ndarray):

    if mask.ndim == 2:#当维度为2，高*宽，不需要减少维度
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:#当维度为3，classes*高*宽，需要减少classes维度，选classes里最大可能类的那个位置下标作为返回的值（二分类则是0，1图、n分类则是0，1，2，3...n-1图）,所以要*255/(shape[0]-1)
        if mask.shape[0]==1:#Jiang，当classes=1时
            return Image.fromarray((np.squeeze(mask) * 255).astype(np.uint8))#三维降二维
        return Image.fromarray((np.argmax(mask, axis=0) * 255/(mask.shape[0]-1)).astype(np.uint8))#Jiang，对0，1矩阵乘255
        #return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))#原版

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

if __name__ == '__main__':
    args = get_args()

    if os.path.splitext(args.input[0])[1]=='.txt':#
        f=open(args.input[0])
        line=f.readline()
        in_files=[]#in_files是图片路径组成的列表
        while line:
            in_files.append(line.splitlines()[0])
            line=f.readline()
    else:
        in_files = args.input#原版，返回一个字符串列表，列表元素个数取决于是输入的元素个数，原作者大概想当输入为多个图片时，
            # 命令行输入的input参数为一个图片路径列表。比如用.txt保存所有输入图片的绝对路径，再用命令行调用该txt作为输入路径参数

    image_dtaroot=args.input[0]
    label_dataroot=args.true_mask[0]
    test_dataset = TestDataset(image_dtaroot, label_dataroot)  # 测试数据集


    out_files = get_output_filenames(args)#原版，这里返回列表或单个值，取决于输入情况
    # out_files=get_output_filenames_J(in_files)#Jiang,当输入是文件夹时，调用该行
    # print(out_files)

    net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)
    # net=segNet_model.SegNet(n_channels=3, n_classes=2)#选择网络
    # net = resUnet34(n_channels=3, n_classes=2, pretrained=False)
    # net=DeepCrack(n_classes=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')  # 调试错误时调用以查找错误来源

    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info('Model loaded!')

    # for i, filename in enumerate(in_files):#用enumerate会返回一个元组，前面是序号（对应后面的out_filename），后面是list下的文件名
    final_accuracy_all = []  # 存储最终指标结果的列表
    thresh_step=0.01
    eps = 1e-6  # 防止除零的偏置值
    for thresh in np.arange(0.0, 1.0, thresh_step):
        statistics = []  # 存储当前阈值及其对应的结果，最终指标结果的子集
        for i,list in enumerate(test_dataset):#循环读取原图和真实label（0，1）,遍历一次测试集
            img=list[0]#待预测图的np格式
            true_label=list[1]#真label的【0，1】格式
            img_name=list[2]#不带后缀的img名

            # print(img.size)#(高，宽)

            new_mask, probability_mask= predict_img(net=net,
                               full_img=Image.fromarray(img),  # 这里还原成img的输入格式
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

            statistics.append(Metrics(probability_mask, true_label,thresh).get_statistics()) # 写入单张图某阈值下计算的[tp,fp,fn,tn]
            #########################JiangShan############
            if (not args.no_save )& (thresh==0.5) :#当阈值=0.5时保存结果，等于总共保存一次
                out_filename = args.output[0]+'/'+img_name+'.png'
                print(out_filename)
                # print(new_mask.shape)#(2, 4736, 7296)
                #print(np.unique(new_mask))#new_mask值为0或1
                result = mask_to_image(new_mask)#result为图片格式
                # print(np.unique(np.asarray(result)))#0或255
                ##############Jiang
                result.save(out_filename)#原
                #####################
                logging.info(f'Mask saved to {out_filename}')
            if args.viz:
                plot_img_and_mask(img, new_mask)

        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])
        tn = np.sum([v[3] for v in statistics])
        #计算ROC所需参数真阳率和假阳率
        FPR = fp / (tn + fp)
        TPR = tp / (fn + tp)
        # calculate precision
        precision = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp + eps)
        # calculate recall
        recall = tp / (tp + fn + eps)
        # calculate f-score（F1值）
        f = 2 * precision * recall / (precision + recall + eps)
        # calculate iou
        iou = tp / (tp + fp + fn + eps)
        final_accuracy_all.append([thresh, precision, recall, f, iou,FPR,TPR])
        print("precision: {}, recall: {}, f 得分： {}, IoU: {}".format(precision, recall, f, iou))
    #将各种评价指标输出成文件
    csv_name = "test.csv"  # 保存PR曲线的数据
    csv_path = args.output[0]+ '/'+ csv_name  # 保存数据的真实路径,args.output应该是一个目录，装预测masks的目录
    save_evaluate(csv_path,final_accuracy_all)#保存格式【阈值，precision, recall,f,Iou】


#使用范例：python predictAndEvaluate.py -i ./data/evaluate_demo/imgs -tm ./data/evaluate_demo/masks -o ./data/evaluate_demo/pred_results

