import argparse
import copy
from TransUnet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUnet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import torch
# from unet.unet_model import UNet
from MyResnet.ResnetWithASPP_model import resnet34
from MyResnet.ResNet_baseLine import resnet34
import re
# net = UNet(n_channels=3, n_classes=2, bilinear=False)
# net= resnet34(n_channels=3,n_classes=2,pretrained=False)

def get_args():
    parser = argparse.ArgumentParser(description='Train the Resnet on public_img and target masks')
    parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')  # 设置图像一共有几个种类需要分割
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    return parser.parse_args()
args = get_args()
config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.classes#默认为2
config_vit.n_skip = args.n_skip
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()



# pthfile = r'../checkpoints/U-net/data_crack_segmentation_dataset_e50_scale=1/checkpoint_epoch50.pth' #U-net模型地址
# pthfile = r'../checkpoints/Resnet34/crack_segmentation_dataset_e100_class=2/checkpoint_epoch100.pth'  #Resnet
pthfile = r'../checkpoints/TransUnet/UGATIT_ToWJSNewE33_optim=RMSprop_L2=1e-6/checkpoint_epoch40.pth' #TransUnet


net.load_state_dict(torch.load(pthfile,map_location=torch.device('cuda'))) #

for name, value in net.named_parameters():#针对TransUNet，前3层冻结
    # matchObj=re.match('.*encoder|.*embeddings', name)#设置冻结嵌入层和编码层参数
    if "encoder" or "embeddings" in name:
        value.requires_grad=False# requires_grad 为 true 则进行更新，为 False 时权重和偏置不进行更新。
    print(name + ' ', value.requires_grad)  #



# print(type(net)) # 类型是 dict

# net.state_dict()
# a='aaabbccc'
# b='ccc'
# d='aaa'
# print(re.match('ccc|aaa', a))
# print(re.match(d, a))
#


# for name, value in net.named_parameters():#针对U-net
#     matchObj=re.match('inc|down', name)#设置冻结编码层参数
#     print(name+' ',value.requires_grad)#
#     if matchObj:
#         value.requires_grad=False# requires_grad 为 true 则进行更新，为 False 时权重和偏置不进行更新。


# for name, value in net.named_parameters():#针对ResNet，前3层冻结
#     matchObj=re.match('layer[1,2,3]|bn|conv', name)#设置冻结编码层参数
#     if matchObj:
#         value.requires_grad=False# requires_grad 为 true 则进行更新，为 False 时权重和偏置不进行更新。
#     # print(name + ' ', value.requires_grad)  #
#
# for name, value in net.named_parameters():#针对ResNet，*bias
#     matchObj=re.match(r'.*bias', name)#设置
#     # print(name+' ',value.requires_grad)#
#     if matchObj:
#         value.requires_grad=False# requires_grad
#     print(name + ' ', value.requires_grad)

# params_conv = filter(lambda p: p.requires_grad==False, net.parameters())#筛选出没被冻结的层

# params_conv = filter(lambda p: p.requires_grad==False, net.parameters())#筛选bias
# params_conv_copy=[]
# for value in params_conv:
#     params_conv_copy.append(value)


# for name, value in net.named_parameters():#针对ResNet
#     matchObj=re.match(r'.*bias', name)#设置
#     # print(name+' ',value.requires_grad)#
#     if matchObj:
#         value.requires_grad=True# requires_grad
#     print(name + ' ', value.requires_grad)


# print(list(params_conv))
#####查看参数名称

# for name, value in net.named_parameters():
#     print(name+' ',value.requires_grad)



# 将模型中属性 requires_grad = True 的参数选出来（要更新的参数在 params_conv 当中）
# params_conv = filter(lambda p: p.requires_grad, net.parameters())
# print(list(params_conv.requires_grad))







# #这是在pth为仅保存参数的情况下
# for key,value in net.items():#查看参数的键，值的尺寸size
#     print(key,value.size(),sep=" ")





