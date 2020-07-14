import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
def save_data(tensor_batch):
    for i,tensor in enumerate(tensor_batch):
        slice=tensor.cpu().detach().numpy()
        img_save = np.clip(slice, 0,2**16)
        img_save = (img_save * 256.0).astype(np.uint16).squeeze()
        image=Image.fromarray(img_save)
        image.save('data/stage'+str(i)+'.png')

def imshow(tensor_batch):

    for i,tensor in enumerate(tensor_batch):
        img_cpu = tensor.cpu().detach().numpy()
        img_save = np.clip(img_cpu, 0, 2**16)
        img_save = (img_save ).astype(np.uint16)
        img_save=img_save.squeeze()
        plt.subplot(3, 3, i + 1)
        plt.imshow(img_save)
        plt.title('stage '+str(i), fontsize=12)
    plt.show()
