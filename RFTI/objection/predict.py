import torch
import argparse
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from makeDataset import *
import torchvision.transforms as transforms
import torch.utils.data
import torch.nn as nn
import time
from utils import accuracy, AverageMeter, save_checkpoint
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from resnest.torch import resnest50,resnest50_fast_2s1x64d,resnest101


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# root = 'F:/'

# imgs = sorted(glob.glob(os.path.join(rootPath) + "/*.*"))

def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        return Image.new('RGB', (224,224), 'white')
    return img



class ResNeSt(nn.Module):
    def __init__(self):
        super(ResNeSt, self).__init__()
        self.resnest50 = resnest50(pretrained=False)
        # model_state_dict = torch.load(
        #     'H:\SoftwareProgram\Data\model/resnest50-528c19ca.pth'
        # )
        # self.resnest50.load_state_dict(model_state_dict)
        self.ReLU1 = nn.ReLU(inplace=True)  # ReLU层必须在__init__中先定义，再在forward中使用
        self.fc1 = nn.Linear(in_features=1000,out_features=2)
        # self.fc2 = nn.Linear(in_features=1000,out_features=50)

        # self.ReLU2 = nn.ReLU(inplace=True)
        # self.softmax = nn.Softmax()#ReLU层必须在__init__中先定义，再在forward中使用

    def forward(self,input):
        x = self.resnest50(input)
        x = self.ReLU1(x)
        output = self.fc1(x)
        # output = self.ReLU2(x)
        # output = self.softmax(x)
        return output


def main():
    rootPath = 'F:/'
    fileList1 = ['001','004','007','018','053','089','095','100IMAGE','101','102','0124','144','1000']
    fileList2 = ['A005','A008','A014','A075-A093','A0113','A118','A119','A120','A121','TJ54']
    # 定义两个数组
    # net = ResNeSt().to(device)
    net = ResNeSt()
    test_model = "./models_pkl/model_best_3.pth.tar"
    ckpt = torch.load(test_model, map_location="cpu")
    net.load_state_dict(ckpt["state_dict"])
    net = net.cuda()
    net.eval()
    with open(rootPath + 'images.txt', 'r') as fid:
        imglist = fid.readlines()
    transform = transforms.Compose([
        transforms.Resize([512, 512]),
        # transforms.CenterCrop([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )])
    idx = 0
    f1 = open(rootPath + 'class.txt', 'w')
    for imagePath in imglist:
        img = default_loader(imagePath)
        img = transform(img).float().unsqueeze(0).cuda()
        logits = net(img)
        _, ind = logits.topk(1, 1, True, True)  # _, ind分别为最大值和对应索引
        idx += 1
        f1.write('{},,,{}\n'.format(imagePath.strip(),int(ind)))
        print(idx)


if __name__ == '__main__':
    main()