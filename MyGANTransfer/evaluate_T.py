import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import numpy as np

from utils.dice_score import multiclass_dice_coeff, dice_coeff

def evaluate_J(net, dataloader, device):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    e_loss=0 #交叉熵
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        if net.num_classes > 1:  # classes大于1时对mask_true进行增维变换
            mask_true = F.one_hot(mask_true, net.num_classes).permute(0, 3, 1, 2).float()
        else:
            mask_true = mask_true.float()  # Jiang

        with torch.no_grad():  # 针对验证集，不需要计算梯度
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.num_classes == 1:
                # mask_pred = (F.sigmoid(mask_pred) > 0.5).float()#原
                mask_pred = (np.squeeze(F.sigmoid(mask_pred) > 0.5)).float()  # Jiang
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.num_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],#计算一整个验证集的dice coefficient和
                                                    reduce_batch_first=False)
                e_loss +=criterion(mask_pred, mask_true)
    net.train()

    # Fixes a potential division by zero error,修复潜在的除以零错误
    if num_val_batches == 0:
        return 1
    return (1-(dice_score/ num_val_batches))*0.5+(e_loss/ num_val_batches)*0.5  # 返回验证集平均loss


def evaluate(net, dataloader, device):#原版，输入验证集dataloader
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        if net.num_classes>1:#classes大于1时对mask_true进行增维变换
            mask_true = F.one_hot(mask_true, net.num_classes).permute(0, 3, 1, 2).float()
        else:
            mask_true=mask_true.float()#Jiang

        with torch.no_grad():#针对验证集，不需要计算梯度
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.num_classes == 1:
                # mask_pred = (F.sigmoid(mask_pred) > 0.5).float()#原
                mask_pred = (np.squeeze(F.sigmoid(mask_pred)>0.5)).float()#Jiang
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.num_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches #返回平均dice coefficient
