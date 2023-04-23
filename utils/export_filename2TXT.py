import os


def save_filename2txt(read_path,save_path):#以read_path目录为根
    name_list=os.listdir(read_path)
    str = '\n'
    f = open(save_path, "w")
    f.write(str.join(name_list))
    f.close()

def save_path2txt(read_path,save_path):#绝对路径
    name_list = os.listdir(read_path)
    str = '\n'
    f = open(save_path, "w")
    for i in name_list:
        f.write(read_path+'\\'+i+str)
    f.close()

if __name__ == '__main__':
    read_path=r'E:\jiangshan\U-net\Pytorch-UNet\data\WJS_cracks\images_256'
    save_path=r'E:\jiangshan\U-net\Pytorch-UNet\data\WJS_cracks\images_256.txt'
    # save_filename2txt(read_path, save_path)
    save_path2txt(read_path, save_path)