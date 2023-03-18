import glob

import exifread
from os.path import splitext
import os
# img_path = 'C:/Users/Administrator/Desktop/INsar和光学图像/无人机航拍图-巧家/微信图片_20220805234537.jpg'
# img_path='G:\\棉沙沟\\DJI_202207311036_009_AbsM300棉沙沟1厘米仿地001\\DJI_20220731111808_0001.JPG'#仿地
def GNSSdataIO(root_path,img_path):

    root_data_path=root_path+'GNSSdata\\'#存位置信息的文件夹
    #img_path='G:\\棉沙沟\\DJI_202207311036_013_AbsM300棉沙沟2厘米变坡右001\\DJI_20220731124419_0003.JPG'
    output_path=splitext(img_path)[0]+'.txt'#data的输出名称
    # print(output_path)
    f = open(root_path+img_path, 'rb')
    xmp=exifread._get_xmp(f)#f这个应该是读二进制(rb)数据，xmp里有相机角度信息
    xmp=xmp.decode('utf-8')#解码二进制
    list=xmp.split()#按空格分割
    contents = exifread.process_file(f)#经纬度，高程等,'rb'格式读取，字典形式存储
    # print(contents)
    # print(xmp)
    # print(list)
    data=list[26:36]
    # print(data)
    f.close()
    isExists = os.path.exists(root_data_path)#判断文件夹是否存在
    if not isExists:
        os.makedirs(root_data_path)#创建文件夹
    else:# 如果目录存在则不创建，并提示目录已存在
            print(root_data_path + ' 目录已存在')
    fw=open(root_data_path+output_path,'w')#
    for each in data:
        fw.write(each)
        fw.write('\n')
    fw.close()

def img_RelativeAltitude(img_path):
    # img_path='G:\\棉沙沟\\DJI_202207311036_013_AbsM300棉沙沟2厘米变坡右001\\DJI_20220731124419_0003.JPG'
    # print(output_path)
    f = open(img_path, 'rb')
    xmp = exifread._get_xmp(f)  # f这个应该是读二进制(rb)数据，xmp里有相机角度信息
    xmp = xmp.decode('utf-8')  # 解码二进制
    RelativeAltitude=
    return xmp



if __name__=='__main__':
    root_path=r'E:\jiangshan\U-net\Pytorch-UNet\data\DJI_20220731124015_0299.JPG'
    xmp=img_RelativeAltitude(root_path)
    print(xmp)
    # print(list_root_path)
    # img_path=glob.glob(root_path+'DJI_*.JPG')
    # print(img_path[0][len(root_path):])





# def convert_gps(coord_arr):
#     arr = str(coord_arr).replace('[', '').replace(']', '').split(', ')
#     d = float(arr[0])
#     m = float(arr[1])
#     s = float(arr[2].split('/')[0]) / float(arr[2].split('/')[1])
#     return float(d) + (float(m) / 60) + (float(s) / 3600)
#
#
# lon = contents['GPS GPSLongitude'].printable  # 经度
# lon = convert_gps(lon)
# lat = contents['GPS GPSLatitude'].printable  # 纬度
# lat = convert_gps(lat)
# altitude = contents['GPS GPSAltitude'].printable  # 相对高度
# altitude = float(altitude.split('/')[0]) / float(altitude.split('/')[1])
#
# print("GPSLongitude:", lon, "\nGPSLatitude:", lat, "\naltitude:", altitude)
