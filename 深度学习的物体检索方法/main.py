import model
import torch
import torch.nn as nn
from data import dir_xy, ImageLoader
import torch.optim as optim
import train_test
import torchvision


def train():
    torch.set_default_tensor_type(torch.FloatTensor)
    # net = model.Net().cuda()

    net = torchvision.models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 2)
    net=net.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.00001)

    dir, xy = dir_xy.get_all(img_path='data/train', label_path='data/label/train.json')
    loader = ImageLoader.myImageFloder(dir, xy)
    TrainImgLoader = torch.utils.data.DataLoader(loader,batch_size=10,shuffle = True)

    dir, xy = dir_xy.get_all(img_path='data/test', label_path='data/label/test.json')
    loader2 = ImageLoader.myImageFloder(dir, xy)
    TestImgLoader = torch.utils.data.DataLoader(loader2)

    train_test.train(net=net.cuda(), criterion=criterion, optimizer=optimizer, TrainImgLoader=TrainImgLoader,
                     TestImgLoader=TestImgLoader)


train()
