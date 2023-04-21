import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

"""
这个dataset一次需要读取包括公共训练集的img，以及我所关注的地裂缝的img.
这两个训练集图片数量和名称都不一致,但是需要同时随机读取其中的一个图,同时还需要保证两个训练集中每个图像都被读取过
"""
class GANTransferDataset(Dataset):
    def __init__(self, public: str, public_mask:str,mydata: str, scale: float = 1.0):
        self.public_dir = Path(public)
        self.mydata_dir = Path(mydata)
        self.public_mask=Path(public_mask)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        self.ids_P = [splitext(file)[0] for file in listdir(public) if not file.startswith('.')]#ids是public的文件名列表
        self.ids_M = [splitext(file)[0] for file in listdir(mydata) if not file.startswith('.')]
        self.ids_PM=[splitext(file)[0] for file in listdir(public_mask) if not file.startswith('.')]
        if not self.ids_P:
            raise RuntimeError(f'No input file found in {public}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids_P)} examples')

    def __len__(self):

        return max(len(self.ids_P),len(self.ids_M))

    @staticmethod
    def preprocess(pil_img, scale, is_mask):#返回处理后的图像array
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)
        #----JiangShan-------
        # if is_mask:
        #     print(img_ndarray[64])#如果mask是.gif格式，到调用这里时已经是0和1了
        if is_mask:
            img_ndarray=numpy.round(img_ndarray/255)
        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255#原始图像归一化
        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):#idx大概是根据len的大小来循环?,也就是取两个目录下文件多的size

        name_P = self.ids_P[idx%len(self.ids_P)]
        name_M= self.ids_M[idx%len(self.ids_M)]
        name_PM =self.ids_PM[idx%len(self.ids_PM)]
        mydata_file = list(self.mydata_dir.glob(name_M + '.*'))#glob用来匹配符合条件的文件名,做成列表,这里是为了防止重名文件
        public_file = list(self.public_dir.glob(name_P + '.*'))
        publicMask_file=list(self.public_mask.glob(name_PM + '.*'))
        assert len(public_file) == 1, f'Either no image or multiple images found for the ID {name_P}: {public_file}'
        assert len(mydata_file) == 1, f'Either no mask or multiple masks found for the ID {name_M}: {mydata_file}'
        assert len(publicMask_file) == 1, f'Either no mask or multiple masks found for the ID {name_PM}: {publicMask_file}'
        mydata= self.load(mydata_file[0])#IMG
        public = self.load(public_file[0])
        publicMask=self.load(publicMask_file[0])
        # assert public.size == mydata.size, \
        #     f'Image_P and Image_M  should be the same size, but are {public.size} and {mydata.size}'

        public_img = self.preprocess(public, self.scale, is_mask=False)#array
        mydata_img = self.preprocess(mydata, self.scale, is_mask=False)
        publicMask_img = self.preprocess(publicMask, self.scale, is_mask=True)  # array
        return {
            'public_img': torch.as_tensor(public_img.copy()).float().contiguous(),
            'mydata_img': torch.as_tensor(mydata_img.copy()).long().contiguous(),
            'publicMask_img' :torch.as_tensor(publicMask_img.copy()).float().contiguous(),
        }


