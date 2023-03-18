import cv2
import numpy as np
from PIL import Image
pred_dir = r"E:\jiangshan\U-net\Pytorch-UNet\data\trainbyChen+Public__Resnet34_IncreaseL2_aug6_e16\\"
mask=np.asarray(Image.open(pred_dir+'WJS.png'))
k=np.ones((2,2),np.uint8)#处理的核
mask_new=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)#闭运算(先膨胀、后腐蚀)，iterations是运行次数
mask_new2=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)#开运算,先腐蚀
mask_OandC=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)#
mask_OandC=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k,iterations=3)#闭运算
mask_dilate=cv2.dilate(mask,k)#膨胀
#先open消除噪点，再close消除内部空点
mask_img=Image.fromarray(mask_new)
mask_img2=Image.fromarray(mask_new2)
mask_dilate=Image.fromarray(mask_dilate)
mask_OandC=Image.fromarray(mask_OandC)
######存储
# mask_img.save(pred_dir+'WJS_Close.png')
mask_img2.save(pred_dir+'WJS_Open.png')
mask_OandC.save(pred_dir+'WJS_OandC.png')
# mask_dilate.save(pred_dir+'WJS_Dilate.png')