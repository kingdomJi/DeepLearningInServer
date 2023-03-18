import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
# 断线特点：断点间隔较短但整体趋势较长，整体有整条线段的趋势但点不完全落在一条直线上
# 需求是连接似同一直线的线段们之间的边界点，要点是如何将不在完全同一直线下但是在同一直线趋势下的连续线段像素点考虑进去。
# 策略1：检测输出点的周围是否有与之相连的点，找到确定是断点位置的点保留（一对点都需要是断点才行）
# 问题1：如果预测的两点间已存在连接线怎么办？
#   保留该两点，沿着两点间的连续线段检索，若已经相连则排除该点对，若中间有中断点，则从两边的中断点开始连接
# 问题2：若得到的两断点间路径夹杂了第三条线在中间怎么办？
#   先根据预测点计算大概直线的斜率，根据斜率确定需要考虑的范围，大概在该预测线段的周围10（不大的一个范围内）个像素范围？
#   确定了范围后，找到范围内（预测两点间的空间内）的所有端点（断点），依据原始线段确定断点们的排列顺序。
# 问题3：若沿线找端点时找到多个端点怎么办？
#   将从两边得到的所有端点进行相互连接计算斜率，斜率与原始两点间线斜率最接近的两点是真正的断点
# 策略2：
##原始霍夫变换只能连接似直线的线段与线段中间的某个点（不是连接端点）
img=cv2.imread(r'E:\jiangshan\U-net\Pytorch-UNet\data\Line_Test.png',0)
print(img.shape[1])#【高，宽】，[y=shape[0],x=shape[1]]
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#彩图转灰度图
# edges=cv2.Canny(gray,50,150,apertureSize=3)#边缘检测
# edges=cv2.Canny(img,50,150,apertureSize=3)
img[img==255]=1#黑白背景转换,0 是黑,255 是白
img[img==0]=255
img[img==1]=0
edges=img
# orgb=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_demo=img[400:600,600:800]
print(img_demo)
retval=cv2.imwrite(r'E:\jiangshan\U-net\Pytorch-UNet\data\Line_Test_demo.png',img_demo)
orgb=img
oShow=orgb.copy()
# lines=cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength=50,maxLineGap=20)
#参数2：精度r，值越大考虑线越多；参数3：角度的精度，值越小考虑越多的线。
#这里的参数2和3我认为应该是设定多少角度偏差内的像素点们可以被视作在同一直线上的点
# print(lines)#返回一个np数组,每个元素都是一对像素位置（两个端点的坐标）

# for line in lines:
#     x1,y1,x2,y2=line[0]#读取端点坐标
#     #Jiang,找到连接点所在线段的端点
#
#     cv2.line(orgb,(x1,y1),(x2,y2),(255,0,0),1)#输入（图像，端点1坐标，端点2坐标，当前绘画的颜色，画笔的粗细）
# retval=cv2.imwrite(r'E:\jiangshan\U-net\Pytorch-UNet\data\Line_Test_Result.png',orgb)



#找线段端点,查找范围往两个点中间靠，两个点的分布只可能是左上右下或左下右上
def findPoint(img,x1,y1,x2,y2):#这里x与y先与边界保持一定距离的好
    width=5#取线段周围多少像素范围
    pointxy = list()
    img_size = img.shape  # height，width，passageway
    ##计算斜率
    if(x2!=x1 and y1!=y2):
        k=(y2-y1) / (x2-x1)  # 原始线段斜率
        k2=-1 / k  # 原始线段垂线的斜率
    ##当斜率特殊
    else:
        if(x2==x1 and x1-width>=1 and x1+width<=img.shape[1]):#当原线段是垂线,含前不含后，x1-5要排除边界点
            for i in range(x1-width,x1+width):
                PX_PRE = img[i - 1]
                PX_curr = img[i]
                PX_Next = img[i + 1]
                for j in range(min(y1,y2),max(y1,y2)):#y暂时没有考虑边界
                    p1 = PX_curr[j]
                    if p1 != 255:
                        continue
                    p2 = PX_PRE[j]
                    p3 = PX_PRE[j + 1]
                    p4 = PX_curr[j + 1]
                    p5 = PX_Next[j + 1]
                    p6 = PX_Next[j]
                    p7 = PX_Next[j - 1]
                    p8 = PX_curr[j - 1]
                    p9 = PX_PRE[j - 1]
                    if p1 == 255:
                        if (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) == 255:
                            pointxy.append((j, i))

        elif(y1==y2 and y1-width>=1 and y1+width<=img.shape[0]):#当是横线
            for i in range(min(x1,x2),max(x1,x2)):
                PX_PRE = img[i - 1]
                PX_curr = img[i]
                PX_Next = img[i + 1]
                for j in range(y1-width,y1+width):
                    p1 = PX_curr[j]
                    if p1 != 255:
                        continue
                    p2 = PX_PRE[j]
                    p3 = PX_PRE[j + 1]
                    p4 = PX_curr[j + 1]
                    p5 = PX_Next[j + 1]
                    p6 = PX_Next[j]
                    p7 = PX_Next[j - 1]
                    p8 = PX_curr[j - 1]
                    p9 = PX_PRE[j - 1]
                    if p1 == 255:
                        if (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) == 255:
                            pointxy.append((j, i))
    ##根据斜率，确定待检测断点的范围（两点间线段周围若干个像素）【求四边形范围点集】
        #斜率=tan(角度)=y/x
        #若要在垂线斜率方向上移动5个像素，x方向要移动cos(角度)*5的距离
        #那么则计算方式为x=根号下（1/（k方+1））*width
        #先求四边形四个点的坐标，求出四个边界线的公式
        #根据四个点确定要遍历的xy轴范围，判断每个点是否在四条边界内，（点带入边界线公式，判断是否大于0或小于0）
    area=list()
    dx=math.sqrt(1/(k2*k2+1))*width#dx肯定大于0，在原点x轴方向上的取值幅度（一半）,这里没有考虑dx<1的这种极端情况，这种属于特殊斜率情况
    dy=k2*dx#dy不确定正负
    #dx方+dy方=25
    #四个点分别是(x1+dx,y1+dy)(x1-dx,y1-dy)(x2+dx,y2+dy)(x2-dx,y2-dy)
        # (x1 + dx, y1 + dy)
        # (x1 - dx, y1 - dy)
        # (x2 + dx, y2 + dy)
        # (x2 - dx, y2 - dy)
    x_left=int(min(x1-dx,x2-dx))
    x_right=int(max(x1+dx,x2+dx))
    y_up=int(max(y1 + dy,y1 - dy,y2 + dy,y2 - dy))
    y_down=int(min(y1 + dy,y1 - dy,y2 + dy,y2 - dy))
    #两点式的表达公式是(y-y2)/(y1-y2) = (x-x2)/(x1-x2)，y=(x-x2)*(y1-y2)/(x1-x2)+y2
    #点斜式（用于已知斜率和一点坐标）:y-y1=k(x-x1)
        #则已知两条线为y-y1=k2(x-x1) ,y-y2=k2(x-x2) 为过原始两点的上下边界
        #另外两条线用两点式：(x1 + dx, y1 + dy)、(x2 + dx, y2 + dy)|(x1 - dx, y1 - dy)、(x2 - dx, y2 - dy)
            #公式： (y-y2-dy)/(y1-y2) = (x-x2-dx)/(x1-x2),(y-y2+dy)/(y1-y2) = (x-x2+dx)/(x1-x2)
    #取中点x0,y0带入四个公式，判断他们的位置关系

    ##找出该范围内所有断点（断点集合）

    while (img[x1][y1] != 0):
        PX_PRE = img[x1 - 1]
        PX_curr = img[x1]
        PX_Next = img[x1 + 1]
        for j in range(1, img_size[1] - 1):
            p1 = PX_curr[j]
            if p1 != 255:
                continue
            p2 = PX_PRE[y1]
            p3 = PX_PRE[y1 + 1]
            p4 = PX_curr[y1 + 1]
            p5 = PX_Next[y1 + 1]
            p6 = PX_Next[y1]
            p7 = PX_Next[y1 - 1]
            p8 = PX_curr[y1 - 1]
            p9 = PX_PRE[y1 - 1]
            if p1 == 255:
                if (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) == 255:
                    pointxy.append((j, i))

    # 断点集合可以从离原始端点（以原始两点中的任一点为原点）最近的断点开始考虑，其他断点也依照与原点的距离排序
    # 距离短的断点间可以直接连接（如间距2像素以内），距离长的断点间连线需要考虑斜率是否与原始斜率相近（具体合适误差范围再试）
    return pointxy

# plt.subplots(121)
# plt.imshow(oShow)
# plt.axis('off')
#############腐蚀操作
# kernel=np.ones((5,5),np.uint8)
# orgb=cv2.erode(orgb,kernel,iterations=1)
#####################二次霍夫变化
# lines=cv2.HoughLinesP(orgb,1,np.pi/180,20,minLineLength=50,maxLineGap=30)
# for line in lines:
#     x1,y1,x2,y2=line[0]
#     cv2.line(orgb,(x1,y1),(x2,y2),(255,0,0),1)
# #######################
# retval=cv2.imwrite(r'E:\jiangshan\U-net\Pytorch-UNet\data\LineDouble_Test_Result.png',orgb)
# #####################三次霍夫变化
# lines=cv2.HoughLinesP(orgb,1,np.pi/180,20,minLineLength=50,maxLineGap=30)
# for line in lines:
#     x1,y1,x2,y2=line[0]
#     cv2.line(orgb,(x1,y1),(x2,y2),(255,0,0),1)
# #######################
# retval=cv2.imwrite(r'E:\jiangshan\U-net\Pytorch-UNet\data\LineTrouble_Test_Result.png',orgb)
# #####################四次霍夫变化
# lines=cv2.HoughLinesP(orgb,1,np.pi/180,20,minLineLength=50,maxLineGap=30)
# for line in lines:
#     x1,y1,x2,y2=line[0]
#     cv2.line(orgb,(x1,y1),(x2,y2),(255,0,0),1)
# #######################
# retval=cv2.imwrite(r'E:\jiangshan\U-net\Pytorch-UNet\data\Line4_Test_Result.png',orgb)
# #####################五次霍夫变化
# lines=cv2.HoughLinesP(orgb,1,np.pi/180,20,minLineLength=50,maxLineGap=30)
# for line in lines:
#     x1,y1,x2,y2=line[0]
#     cv2.line(orgb,(x1,y1),(x2,y2),(255,0,0),1)
# #######################
# retval=cv2.imwrite(r'E:\jiangshan\U-net\Pytorch-UNet\data\Line5_Test_Result.png',orgb)
# #####################十次霍夫变化
# for i in range(5):
#     lines = cv2.HoughLinesP(orgb, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(orgb, (x1, y1), (x2, y2), (255, 0, 0), 1)


# retval=cv2.imwrite(r'E:\jiangshan\U-net\Pytorch-UNet\data\Line10_Test_Result.png',orgb)
# plt.subplots(122)
# plt.imshow(orgb)

