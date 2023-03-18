import numpy as np
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True#
Image.MAX_IMAGE_PIXELS = None#导入以防止因为图片过大而报错

from torchvision import transforms
#该文件用来将大张图像进行整张图像切割然后预测完之后再合并
class CropAndConnect:
    def Crop(img_np,size_h=256,size_w=256,sideLength=64):#裁成256*256大小,其中边界大小64

        batchArray=[]#存裁剪块的列表
        w=img_np.shape[1]#宽
        h=img_np.shape[0]#高
        size_h_ture=size_h-sideLength*2
        size_w_ture=size_w-sideLength*2
        #############计算行列数，有可能不被整除
        column=int((w-sideLength*2)/size_w_ture)#块列数,按照去除边界的标准裁出的数量，即128*128能裁的数量算
        row=int((h-sideLength*2)/size_h_ture)#块行数
        for i in range(row):#先遍历行
            batchArray_inner=[]#存每一行的结果
            for j in range(column):
                batch =img_np[i*size_h_ture:(i+1)*size_h_ture+2*sideLength,j*size_w_ture:(j+1)*size_w_ture+sideLength*2]
                #截取img中某一块图像的格式是img[高1：高2，宽1：宽2]而不是img[高1：高2][宽1：宽2]
                #img_np[i]是遍历height,img_np[i][j]是遍历高和宽
                batchArray_inner.append(batch)
            batchArray.append(batchArray_inner)#batchArray存最终裁剪结果
        batchArray=np.asarray(batchArray)
        return batchArray#返回的是一个数组size=(块行数,块列数,高，宽，RGB)
        ################裁剪是从图像正上和正左方向开始的，待处理边角料在下方和右方以及右下角。
        ###########(暂时不考虑边角料，如需要，则再写存入，拼接时再拼入)
        #####可以在原图外围padding一下再切割，这样拼接时就不必考虑边角部分的拼接问题了，拼接时直接舍弃被padding的部分，统一每一块的拼接操作。

    def Connect(masks,classes,size_h=256,size_w=256,sideLength=64):
        #传入矩阵masks中的图片前两维与batchArray中的块位置一一对应，但经过网络后三维变成(classes,高，宽了)注：（非最终输出mask格式）
        #传入的masks.size=(块行数,块列数,classes,高，宽)
        masks=torch.tensor(masks).permute(0,1,3,4,2)
        #(块行数,块列数,高，宽,classes)要用permute函数转化一下位置
        result=torch.tensor(np.zeros((masks.shape[0]*(size_h-2*sideLength),masks.shape[1]*(size_w-2*sideLength),classes)))
        #Tensor类型(高，宽,classes)
        #结果图大小基于masks图片集大小,若有padding时加上边界sideLength*2
        for i in range(masks.shape[0]):#遍历行
            for j in range(masks.shape[1]):#遍历列
                ######这里要用permute函数转化一下位置
                result[i*(size_h-2*sideLength):(i+1)*(size_h-2*sideLength),j*(size_w-2*sideLength):(j+1)*(size_w-2*sideLength)]\
                    =masks[i,j,sideLength:size_h-sideLength,sideLength:size_w-sideLength]
        result=result.permute(2,0,1)#(高，宽,classes)变（classes,高，宽）
        return np.asarray(result)#返回（classes,高，宽）的格式

if __name__=='__main__':#test
    path=r'E:\jiangshan\U-net\Pytorch-UNet\data\WJS.png'
    path_2=r'E:\jiangshan\U-net\Pytorch-UNet\data\LHS.png'
    # t=5945425920/(1024*1024)
    # print(t)
    img=Image.open(path).convert('RGB')
    img_np=np.asarray(img)
    img2 = Image.open(path_2).convert('RGB')
    # if img2.size()!=3:
    #     img2 = img2.convert('RGB')#四通道转三通道
    img2_np = np.asarray(img2)
    print(img_np.shape)
    print(img2_np.shape)


    # print(img_np[1000:1020,1000:1020])
    # imgs=CropAndConnect.Crop(img)#返回一个数组
    # print(imgs.shape)
    # imgs_test=np.random.randint(0,255,size=(90, 63, 2,256, 256))
    # result=CropAndConnect.Connect(imgs_test,classes=2)#拼接
    # print(result.shape)