import argparse
import logging
import sys
from pathlib import Path
import os
import re
import numpy as np
from SegNet import segNet_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm#tqdm是一个进度条可视化库，可以帮助我们监测程序运行的进度，估计运行的时长，甚至可以协助debug。
from DeepCrack.codes.model.deepcrack import DeepCrack
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from resnet.models import networks
from MyResnet.ResNet_baseLine import resnet34
# from MyResnet.ResnetWithASPP_model import resnet34
from evaluate import evaluate,evaluate_J
from unet import UNet

# dir_img = Path(r'.\data\data_Chen_new\augmentation_Jiang\patches\aug_seg\kq6_dom_aug\\')
dir_img = Path(r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\patches\kq6_dom\\')
# dir_img=Path(r'.\data\crack_segmentation_dataset\images\\')
# dir_img=Path(r'.\data\LHS\images\\')
# dir_mask = Path(r'.\data\data_Chen_new\augmentation_Jiang\patches\aug_seg\kq6_label_seg_aug\\')
dir_mask = Path(r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\patches\kq6_label_seg\\')
# dir_mask = Path(r'.\data\crack_segmentation_dataset\masks\\')
# dir_mask = Path(r'.\data\LHS\labels\\')
dir_checkpoint = Path('checkpoints/Resnet34/Transfer_freezeDecoder[1-3]/Public_e100_TransferToChen_increaseL2=1e-5/')#这里基于使用的网络
# dir_checkpoint = Path('checkpoints/U-net/data_LHS_e100_scale=1/')#这里基于使用的网络
# dir_checkpoint = Path('./checkpoints/test/')#这里基于使用的网络

def train_net(net,
              device,
              epochs: int = 5,#时期
              batch_size: int = 8,
              learning_rate: float = 1e-4,#学习效率
              val_percent: float = 0.1,#验证集占总图片的比例
              save_checkpoint: bool = True,#保存检查点
              img_scale: float = 1,#图像尺度
              amp: bool = False):
    # 1. Create dataset
    try:
        dataset = BasicDataset(dir_img, dir_mask, img_scale)#读取数据集
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)
        #dataloder会用Image.open打开图像，然后用np.asarray()转换成数组处理
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)#验证集的个数
    n_train = len(dataset) - n_val#训练集个数
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        #基于训练集和验证集的个数，随机分割数据集，manual：手动
        #torch.Generator().manual_seed：设置CPU生成随机数的种子。
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=False)
    #加载器参数是字典形式，包含栅格尺寸、工作者数（关系到内存）、是否设置锁页内存
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    #训练集加载器，传入训练集，打乱数据为真，**loader_args指将字典作为参数传入，没有传入的参数按照源代码构造器中的默认值算。
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    #验证集加载器，传入验证集，不打乱数据，。。。
    # (Initialize logging)初始化日志
    # experiment = wandb.init(project='Test', resume='allow', anonymous='must', name='test训练')
    # experiment = wandb.init(project='Resnet34', resume='allow', anonymous='must',name='crack_segmentation_dataset_e100_c=2训练')#每次训练更改
    experiment = wandb.init(project='Resnet34', resume='allow', anonymous='must', name='Public_e100_TransferToChen_Aug5_2_e100训练')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}#栅格尺寸
        Learning rate:   {learning_rate}#学习速度
        Training size:   {n_train}#训练集个数
        Validation size: {n_val}#验证集个数
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    #设定优化器、损失函数、学习速率调度器、AMP的损耗缩放
    #############Jiang
    for name, value in net.named_parameters():  # 冻结部分层
        # matchObj = re.match('inc|down', name)  # U-net设置冻结编码层参数，只更新解码层
        matchObj = re.match('layer[1,2,3]|bn|conv', name)  # ResNet,设置冻结编码层参数前3层参数，改变第四层和解码层参数
        # print(value.requires_grad)
        if matchObj:
            value.requires_grad = False  ## requires_grad 为 true 则进行更新，为 False 时权重和偏置不进行更新。
    params_conv = filter(lambda p: p.requires_grad, net.parameters())#筛选出没被冻结的层
    ######################
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.RMSprop(params_conv, lr=learning_rate, weight_decay=1e-5, momentum=0.9)#Jiang
    #######Jiang
    for name, value in net.named_parameters():  # 针对ResNet
        matchObj = re.match(r'.*bias', name)  # 设置
        if matchObj:
            value.requires_grad = False  # requires_grad

    params_bias = filter(lambda p: p.requires_grad == False, net.parameters())  # 筛选bias
    params_others = filter(lambda p: p.requires_grad, net.parameters())  # 筛选其他
    params_bias_copy = []  # 存bias
    params_others_copy = []
    for value in params_bias:
        params_bias_copy.append(value)
    for value in params_others:
        params_others_copy.append(value)
    for name, value in net.named_parameters():  # 针对ResNet
        matchObj = re.match(r'.*bias', name)  # 设置
        if matchObj:
            value.requires_grad = True  # 改回来
    ################################
    optimizer = optim.RMSprop([
        {'params': params_others_copy, 'weight_decay': 1e-6},
        {'params': params_bias_copy, 'weight_decay': 0}
    ], lr=learning_rate, momentum=0.9)
    # optimizer=optim.Adam(net.parameters(),lr=learning_rate)#这个不能乱用，可能会优化不了模型
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.1, patience=5)
    # 学习率调整策略，监视loss的，patience个epoch的loss没降，他就会降低学习率,ReduceLROnPlateau可能不适合diceloss这种容易震荡的loss函数收敛
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9) #学习速率调整策略,指数衰减策略，gamma是衰减因子，每个epoch的lr*0.5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=- 1,
                                                     verbose=False)  # 余弦退火
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    criterion = nn.CrossEntropyLoss()#应用于输入classes》=2的情况
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                # print(np.unique(true_masks))#应该是0和1
                #print(true_masks.shape)#torch.Size([1, 640, 959])1是指单通道
                # print(true_masks[0][64])
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)#核心处理,将图片输入到net中
                    # print(masks_pred.shape)#torch.Size([batch, classes, 高, 宽]),值有正有负，大概在[-5，5]左右
                    # print(np.unique(true_masks.cpu().data.numpy()))#torch.Size([1, 高, 宽]),值只有0，1
                    ###JiangShan
                    if net.n_classes==1:#当n_classes=1
                         # print(torch.sigmoid(masks_pred).shape)#[batch, classes, 高, 宽]
                         # print(true_masks.shape)#[batch, 高, 宽]
                         criterion=nn.BCELoss()#针对二分类的交叉熵
                         loss = criterion(np.squeeze(torch.sigmoid(masks_pred)).float(), true_masks.float()) + \
                               dice_loss(np.squeeze(torch.sigmoid(masks_pred)).float(),  #n_classes=1分类（二分类）归一,将值收敛到[0,1]
                                           true_masks.float(),
                                           multiclass=False)  # 把true_masks变成4维，增加n_classes维度和masks_pred对应
                    else:
                         # print(F.softmax(masks_pred, dim=1).shape)#[batch, classes, 高, 宽]
                         # print(F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).shape)
                         loss = criterion(masks_pred, true_masks) + \
                               dice_loss(F.softmax(masks_pred, dim=1).float(),#对向量进行归一化,多个类加起来概率为1
                                           F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True)#把true_masks变成4维，增加n_classes维度和masks_pred对应
                    ########################
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round，评估阶段
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        # for tag, value in net.named_parameters():#原
                        for tag, value in params_conv:#Jiang

                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        #原版
                        # val_score = evaluate(net, val_loader, device)
                        # scheduler.step(val_score)
                        ################Jiang
                        val_loss = evaluate_J(net, val_loader, device)
                        scheduler.step(val_loss)  # 监测(val_loss)，当val_loss几个epoch后不再降低，则降低学习率
                        ##########

                        logging.info('Validation loss: {}'.format(val_loss))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation loss': val_loss,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():#传入参数
    #1.创建解析器，ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    #description ：在参数帮助文档之前显示的文本
    #2，添加参数，给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成
    # 开头的：name or flags - 一个命名或者一个选项字符串的列表，例如foo或 - f, --foo。
    # action - 当参数在命令行中出现时使用的动作基本类型。
    # nargs - 命令行参数应当消耗的数目。
    # const - 被一些action和nargs选择所需求的常数。
    # default - 当参数未在命令行中出现时使用的值。
    # type - 命令行参数应当被转换成的类型。
    # choices - 可用的参数的容器。
    # required - 此命令行选项是否可省略 （仅选项可用）。
    # help - 一个此选项作用的简单描述。
    # metavar - 在使用方法消息中使用的参数值示例。
    # dest - 被添加到parse_args()所返回对象上的属性名。
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    # parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    ## 这里加载已经训练过的待迁移训练的模型
    parser.add_argument('--load', '-f', type=str, default='./checkpoints/Resnet34/crack_segmentation_dataset_e100_class=2/checkpoint_epoch100.pth')#Jiang
    #例如./checkpoints_SegNet_crack/checkpoint_epoch20.pth
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')#使用双线性上采样
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')#设置图像一共有几个种类需要分割
    parser.add_argument('--Transfer', '-tr', type=bool, default=False, help='The Transfer learning')
    #3，解析参数，ArgumentParser 通过 parse_args() 方法解析参数
    return parser.parse_args()#返回解析参数


if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'#指定GPU
    # torch.cuda.current_device()
    args = get_args()#获取参数
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')#日志基本设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#设置优先使用gpu
    # device= torch.device('cpu')#调试错误时调用以查找错误来源
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel

    # net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)#通道数：PGB、每个像素要获取的概率、双线性
    # net = segNet_model.SegNet(n_channels=3, n_classes=args.classes)
    # net= networks.resnet34(pretrained=False)#初始化网络
    net=resnet34(n_channels=3,n_classes=2,pretrained=False)
    # net=DeepCrack(num_classes=args.classes)##默认input_channnel为3

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')
                 # f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')##U-net独有

    if args.load:#加载预训练模型时
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
        #############Jiang
        # for name, value in net.named_parameters():#冻结部分层
        #     matchObj = re.match('inc|down', name)  # 设置冻结编码层参数
        #     # print(value.requires_grad)
        #     if matchObj:
        #         value.requires_grad = False  ## requires_grad 为 true 则进行更新，为 False 时权重和偏置不进行更新。
        ####################
    net.to(device=device)
    try:
        train_net(net=net,#正式训练网络，传入参数，net选用U-net或其他,部分参数从args中获取
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')#只保存参数，没保持整个模型
        logging.info('Saved interrupt')
        raise