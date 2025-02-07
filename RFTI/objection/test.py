import torch
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from makeDataset import *
from build_model import *
import torchvision.transforms as transforms
import torch.utils.data
import torch.nn as nn
import time
from utils import accuracy, AverageMeter, save_checkpoint
from PIL import  Image
import numpy as np

from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# rootPath = 'F:\SoftwareProgram\Data\dataset\snow leopard'
rootPath = 'H:\SoftwareProgram\Data\dataset\snow leopard'


def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        return Image.new('RGB', (224,224), 'white')
    return img

class RandomDataset(Dataset): #产生测试集/验证集
    def __init__(self, transform=None, dataloader=default_loader):
        self.transform = transform
        self.dataloader = dataloader

        with open(rootPath+'\original/original/test1.txt', 'r') as fid:
            self.imglist = fid.readlines()

    def __getitem__(self, index):
        image_name, label = self.imglist[index].strip().split(',,,')
        image_path = image_name
        image_path = 'H:' + image_path.split(':')[-1]  #修改移动硬盘中的绝对路径
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)
        label = torch.LongTensor([label])

        return [img, label,image_path]


    def __len__(self):
        return len(self.imglist)


def main():
    batchsize = 2
    # 定义两个数组
    test_Loss_list = []
    test_Accuracy_list = []
    # net = ResNeSt().to(device)
    net = ResNeSt()
    test_model = "./models_pkl/checkpoint_relu_fc2.pth.tar"
    ckpt = torch.load(test_model, map_location="cpu")
    # net.load_state_dict(ckpt["state_dict"])
    net.load_state_dict(ckpt["state_dict"])
    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    test_dataset = RandomDataset(transform=transforms.Compose([
        transforms.Resize([512, 512]),
        # transforms.CenterCrop([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batchsize, shuffle=False,
        num_workers=0, pin_memory=True)
    Accuracy,Precision,Recall = tests11(test_loader,net,criterion)
    print('测试集准确率：'+str(Accuracy) + '| 精准率：'+str(Precision) + '| 召回率：'+str(Recall))



def tests11(test_loader, net, criterion):
    batch_time = AverageMeter()
    softmax_losses = AverageMeter()
    top1 = AverageMeter()

    # top5 = AverageMeter()
    #
    # # switch to evaluate mode
    # model.eval()
    end = time.time()
    net.eval()
    # net = ResNeSt().to(device)
    # f1 = open('F:\SoftwareProgram\Data\dataset\snow leopard\original\\test_negative.txt', 'w')
    f1 = open('./test_negative_hard_samples.txt', 'w')
    TP = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for i, (inputs, targets,image_paths) in enumerate(test_loader):

            inputs = inputs.cuda()
            targets = targets.long().cuda().squeeze()  # 对变量维度进行压缩

            # compute output
            logits = net(inputs)
            loss = criterion(logits, targets)

            #logits.topk(6, 1, True, True)

            prec1,correct= accuracy(logits, targets, 1)
            for index,prec_index in enumerate(correct):
                if prec_index == True and targets[index] == 1:
                    TP += 1
                if prec_index == False:
                    f1.write('{},,,{}\n'.format(image_paths[index],targets[index]))   #输出预测错误的图像到test_negative.txt
                    if targets[index] == 0:
                        FP += 1
                    if targets[index] == 1:
                        FN += 1
                # if prec_index == True and targets[index] == 0:
                #     TN += 1
            # prec5 = accuracy(logits, targets, 5)
            softmax_losses.update(loss.item(), logits.size(0))
            top1.update(prec1, logits.size(0))
            # top5.update(prec5, logits.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()



            if i % 1 == 0:
                print('Time: {time}\nTest: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'SoftmaxLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(test_loader), batch_time=batch_time, softmax_loss=softmax_losses,
                        top1=top1, time=time.asctime(time.localtime(time.time()))))

        # test_Loss_list.append(softmax_losses.avg)
        # test_Accuracy_list.append(100 * top1.avg)

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    f1.close()
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    return top1.avg,Precision*100,Recall*100

if __name__ == '__main__':
    main()