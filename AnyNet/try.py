import argparse
import torch
import torch.nn.parallel
import torch.utils.data
import models.anynet
from utils.show import save_data,imshow

from PIL import Image
from dataloader import preprocess
import random


parser = argparse.ArgumentParser(description='AnyNet with Flyingthings3d')

parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datapath', default='F:/data_scene_flow/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6,
                    help='batch size for training (default: 12)')
parser.add_argument('--test_bsize', type=int, default=4,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/pretrained_anynet',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels of the 3d network')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers of the 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')


args = parser.parse_args()
def main():
    global args
    model = models.anynet.AnyNet(args)

    # print(model)
    model = torch.nn.DataParallel(model).cuda()
    #载入预训练模型
    checkpoint = torch.load('results/finetune_anynet/checkpoint.tar')
    model.load_state_dict(checkpoint['state_dict'])

    # imgL=torch.rand(6,3,256,512)
    # imgR=torch.rand(6,3,256,512)
    imgL, imgR = data_load('data/left_1.png', 'data/right_1.png')
    imgL=torch.unsqueeze(imgL,0)
    imgR=torch.unsqueeze(imgR,0)
    imgL = imgL.float().cuda()
    imgR = imgR.float().cuda()
    outputs=model(imgL,imgR)
    save_data(outputs)
    imshow(outputs)


def data_load(left_img_dir,right_img_dir):
    left_img=Image.open(left_img_dir).convert('RGB')
    right_img=Image.open(right_img_dir).convert('RGB')

    w, h = left_img.size
    th, tw = 256, 512
    # 变为256，512
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
    right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

    # dataL = dataL.crop((w - 1232, h - 368, w, h))
    # dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
    processed = preprocess.get_transform(augment=False)
    left_img = processed(left_img)
    right_img = processed(right_img)
    return left_img, right_img


main()
# imgL,imgR=data_load('data/left.png','data/right.png')
