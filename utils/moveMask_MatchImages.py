import os
import shutil
def searchMask(source_dir,mask_dir,save_mask_dir):#复制与指定目录下imgs相同名称的masks到指定文件夹
    img_list = os.listdir(source_dir)
    mask_list = os.listdir(mask_dir)
    for i in range(len(img_list)):
        filename=os.path.splitext(img_list[i])[0]
        mask_name=filename+'.jpg'
        print(mask_name)
        if mask_name in mask_list:
            mask_path=os.path.join(mask_dir,mask_name)
            save_mask_path=os.path.join(save_mask_dir,mask_name)
            shutil.copy(mask_path,save_mask_path)


if __name__=='__main__':
    source_dir=r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\UGATIT_publicToWJS_e33'
    mask_dir=r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\Transfer_mask'
    save_dir=r'E:\jiangshan\U-net\Pytorch-UNet\data\data_Chen_new\augmentation_Jiang\patches\UGATIT_publicToWJS_e33_mask'
    searchMask(source_dir, mask_dir, save_dir)










