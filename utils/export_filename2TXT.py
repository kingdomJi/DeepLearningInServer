import os


def save_filename2txt(read_path,save_path):
    name_list=os.listdir(read_path)
    str = '\n'
    f = open(save_path, "w")
    f.write(str.join(name_list))
    f.close()



if __name__ == '__main__':
    read_path=r'E:\jiangshan\U-net\AdaptSegNet\data\Cityscapes\data\leftImg8bit\val'
    save_path=r'E:\jiangshan\U-net\AdaptSegNet\data\Cityscapes\data\leftImg8bit\val.txt'
    save_filename2txt(read_path, save_path)