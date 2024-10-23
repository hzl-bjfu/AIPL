import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets

import os
import time
import torch.nn.functional as F

import torchvision
import numpy as np

import argparse
import sys

from torch.utils.data import Dataset, DataLoader
import pickle
import plda
from tqdm import tqdm

from TDNN.tdnn import TDNN

def readpkl(path):
    with open(path,'rb') as f:
        data,label = pickle.load(f)
        return data,label

class getdata(Dataset):
    # dirname 为训练/测试数据地址，使得训练/测试分开
    def __init__(self, dirname, train=True):
        super(getdata, self).__init__()
        self.images, self.labels = readpkl(dirname)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        image = image.astype(np.float32)
        # image = torch.from_numpy(image).div(255.0)
        label = self.labels[index]
        label = int(label)
        return image, label

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)


    BATCH_SIZE = 32
    plda_thr = -20

    pkl_path = 'D:\\work\\code\\TDNN\\Data'
    # train_path = pkl_path + '\\D3train.pkl'
    # val_path = pkl_path + '\\D1val.pkl'
    test_path = pkl_path + '\\D1test.pkl'

    # dataset_train = getdata(train_path)
    # dataset_val = getdata(val_path)
    dataset_test = getdata(test_path)

    # train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)


    # net=torch.load('best_network.pth')
    net=torch.load('D3.pth')

    net.to(DEVICE)
    test_plda(net,test_loader,plda_thr,DEVICE)

def test_plda(net, testloader,plda_thr, device):

    net.eval()
    testdata = np.array([[]])
    testlabel = np.array([])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _,outputs = net(inputs)
            for emb in outputs:
                emb = torch.unsqueeze(emb,0)
                l2_norms = np.linalg.norm(emb.cpu(), axis=1, keepdims=True)
                emb = emb.cpu() / l2_norms
                try:
                    testdata = np.append(testdata, emb, axis=0)
                except:
                    testdata = emb
            for label in targets:
                testlabel = np.append(testlabel, label.cpu())

    print('testdata:',testdata.shape)
    print('testlabel',testlabel.shape)

    open_num = len(testlabel)
    test_X, test_y = testdata, testlabel

    # print(test_y)

    g = open('D3.txt', 'rb')
    # g = open('./model/2/2023crossS2_PCEN2.txt', 'rb')
    bb = pickle.load(g)
    g.close()
    predictions, log_p_predictions = bb.predict(test_X)  # 预测，使用plda
    # print(predictions)
    # print(log_p_predictions)

    for i in range(open_num):
        score = log_p_predictions[i]
        # score2 = plda_w[i]
        # score = score/score2
        # print('score:',score)
        if max(score) < -plda_thr:   #阈值，与acc_known正比
            predictions[i] = 7.0  #Subscript of the unknown class

    # print(predictions)

    known_num = 500  #num of known samples from S1,1457####S2,2668
    print('known_Accuracy: {}'.format((test_y[0:known_num] == predictions[0:known_num]).mean()))
    print('unknown_Accuracy: {}'.format((test_y[known_num:] == predictions[known_num:]).mean()))
    acc_k = (test_y[0:known_num] == predictions[0:known_num]).mean()
    acc_u = (test_y[known_num:] == predictions[known_num:]).mean()
    ACC = (acc_u+acc_k)/2
    print('ACC:',ACC)


if __name__ == '__main__':
    main()
