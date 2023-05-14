import os
import shutil

def change_maskAndImgsNames(img_path, mask_path,suffix,img_savePath,mask_savePath):
    list_img = os.listdir(img_path)
    list_m = os.listdir(mask_path)
    for i in range(len(list_img)):
        print(list_img[i])
        assert os.path.splitext(list_img[i])[0] ==os.path.splitext(list_m[i])[0]   # 名称对应
        img_name = os.path.splitext( list_img[i])[0]+suffix+os.path.splitext( list_img[i])[1]
        mask_name = os.path.splitext(list_m[i])[0]+suffix+ os.path.splitext(list_m[i])[1]
        save_img_path=os.path.join(img_savePath, img_name)
        save_mask_path = os.path.join(mask_savePath, mask_name)
        source_img_path=os.path.join(img_path, list_img[i])
        source_mask_path=os.path.join(mask_path, list_m[i])
        shutil.copy(source_img_path,save_img_path)
        shutil.copy(source_mask_path, save_mask_path)



if __name__=='__main__':
    img_path=r'..\data\data_Chen_new\augmentation_Jiang\patches\UGATIT_publicTokq13_e100_whole'
    mask_path=r'..\data\data_Chen_new\augmentation_Jiang\patches\UGATIT_WholePublic_masks_256'
    suffix=r'_Tokq13'
    img_savePath=r'..\data\data_Chen_new\augmentation_Jiang\patches\UGATIT_imgs_stylesSum'
    mask_savePath=r'..\data\data_Chen_new\augmentation_Jiang\patches\UGATIT_masks_stylesSum'
    change_maskAndImgsNames(img_path, mask_path, suffix, img_savePath, mask_savePath)