import copy

import torch
# from unet.unet_model import UNet
from MyResnet.ResnetWithASPP_model import resnet34
from MyResnet.ResNet_baseLine import resnet34
import re
# net = UNet(n_channels=3, n_classes=2, bilinear=False)
net= resnet34(n_channels=3,n_classes=2,pretrained=False)
# pthfile = r'../checkpoints/U-net/data_crack_segmentation_dataset_e50_scale=1/checkpoint_epoch50.pth' #U-net模型地址
pthfile = r'../checkpoints/Resnet34/crack_segmentation_dataset_e100_class=2/checkpoint_epoch100.pth'  #Resnet
net.load_state_dict(torch.load(pthfile,map_location=torch.device('cuda'))) #
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


# for name, value in net.named_parameters():#针对ResNet
#     matchObj=re.match('layer[1,2,3]|bn|conv', name)#设置冻结编码层参数
#     if matchObj:
#         value.requires_grad=False# requires_grad 为 true 则进行更新，为 False 时权重和偏置不进行更新。
    # print(name + ' ', value.requires_grad)  #




for name, value in net.named_parameters():#针对ResNet
    matchObj=re.match(r'.*bias', name)#设置
    # print(name+' ',value.requires_grad)#
    if matchObj:
        value.requires_grad=False# requires_grad

# params_conv = filter(lambda p: p.requires_grad==False, net.parameters())#筛选出没被冻结的层

params_conv = filter(lambda p: p.requires_grad==False, net.parameters())#筛选bias

params_conv_copy=[]
for value in params_conv:
    params_conv_copy.append(value)


for name, value in net.named_parameters():#针对ResNet
    matchObj=re.match(r'.*bias', name)#设置
    # print(name+' ',value.requires_grad)#
    if matchObj:
        value.requires_grad=True# requires_grad
    print(name + ' ', value.requires_grad)


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





