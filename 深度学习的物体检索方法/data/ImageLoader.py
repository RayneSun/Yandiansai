import torch.utils.data as data
import torch
from PIL import Image
import numpy as np


def default_loader(path):
    return Image.open(path).convert('RGB')


###############################################
class myImageFloder(data.Dataset):

    def __init__(self, img, label, loader=default_loader):
        self.img = img
        self.label = label
        self.loader = loader

    def __getitem__(self, index):
        img = self.img[index]
        label = self.label[index]

        # 数据打开
        img_open = self.loader(img)
        data = np.ascontiguousarray(img_open, dtype=np.float32) / 256

        label = np.array([label[0]/540.0,label[1]/384.0],dtype=np.float32)

        data = torch.from_numpy(data).view(3, 540, 384).cuda()
        label = torch.from_numpy(label).cuda()

        return data, label

    def __len__(self):
        return len(self.img)
