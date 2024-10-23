import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.manifold import TSNE
import os
import torch.nn.functional as F

import numpy as np

import argparse
import sys

from torch.utils.data import Dataset, DataLoader
import pickle
import plda
from TDNN.tdnn import TDNN
from TDNN2 import tdnn_IFN,tdnn_BN,tdnn_both,tdnn_LSTM,tdnn_TN,tdnn_GW
from sklearn.metrics import f1_score,precision_score,recall_score

parser = argparse.ArgumentParser(description='传入参数')  #定义：解析参数
#type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('--model_name',default='D3' ,type=str, help='保存模型')
parser.add_argument('--tdnn_method', default='tdnn_both', type=str, help='网络结构')

args = parser.parse_args()  #实际来解析参数


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

def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(DEVICE)

    BATCH_SIZE = 32

    pkl_path = 'D:\\work\\code\\TDNN\\Data'
    train_path = pkl_path + '\\D3train.pkl'
    val_path = pkl_path + '\\D3val.pkl'
    # test_path = pkl_path + "/2023cross_S2test.pkl"

    # pcen_test_path = 'E:\\zly2023\\pkl\\pcen\\2023cross_S2test_pcen.pkl'
    dataset_train = getdata(train_path)
    dataset_val = getdata(val_path)
    # dataset_test = getdata(test_path)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    # net = torch.load("best_network.pth")
    file_name = args.model_name
    # path = './revise0719/new_plda_CSR/bn/crossS1_BN_2'
    path = args.model_name
    net = torch.load('D3.pth')

    # net = torch.load('./revise0719/cross/'+ file_name + '.pth')

    net.to(DEVICE)
    # plot_xvector(net,test_loader,DEVICE)
    # os._exit(0)

    train_plda(net,train_loader,val_loader,DEVICE,file_name)


def plot_xvector(net,dataloader,device):
    net.eval()

    testdata = np.array([[]])
    testlabel = np.array([])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            for emb in outputs:
                # print(emb)
                # print(emb.shape)
                emb = torch.unsqueeze(emb.cpu(),0)
                # l2_norms = np.linalg.norm(emb.cpu(), axis=1, keepdims=True)
                # emb = emb.cpu() / l2_norms
                try:
                    testdata = np.append(testdata, emb, axis=0)
                except:
                    testdata = emb
            for label in targets:
                testlabel = np.append(testlabel, label.cpu())

    # print('testdata:',testdata.shape)
    # print('testlabel',testlabel.shape)

    # embedding = umap.UMAP(
    #     n_neighbors=10,
    #     min_dist=0.3,
    #     n_components=2,
    #     random_state=42,
    # ).fit_transform(testdata)

    ts = TSNE(
        perplexity=100,
        n_components=2)
    embedding = ts.fit_transform(testdata)

    print('embeddingshape:',embedding.shape)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
    for label_idx in range(9):
        features = embedding[testlabel == label_idx, :]
        plt.scatter(
            features[:,0],
            features[:,1],
            c=colors[label_idx],
            s=1,
        )
    legends = ['0', '1', '2', '3', '4', '5', '6', '7', 'unknown']
    plt.legend(legends[0:9], loc='upper left')
    plt.savefig('tsne_S1_nonorm')
    plt.close


def train_plda(net,trainloader,  testloader, device,file_name):
    net.eval()
    traindata = np.array([[]])
    trainlabel = np.array([])
    testdata = np.array([[]])
    testlabel = np.array([])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _,outputs = net(inputs)
            for emb in outputs:
                emb = torch.unsqueeze(emb,0)
                # print(emb)
                # print(emb.shape)
                l2_norms = np.linalg.norm(emb.cpu(), axis=1, keepdims=True)
                emb = emb.cpu() / l2_norms
                try:
                    traindata = np.append(traindata, emb, axis=0)
                except:
                    traindata = emb
            for label in targets:
                trainlabel = np.append(trainlabel, label.cpu())

    # print('traindata:',traindata.shape)
    # print('trainlabel',trainlabel.shape)

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

    # print('testdata:',testdata.shape)
    # print('testlabel',testlabel.shape)
    # for key, value in testdata
    train_X, train_y = traindata, trainlabel
    test_X, test_y = testdata, testlabel

    overfit_classifier = plda.Classifier()
    overfit_classifier.fit_model(train_X, train_y)  # 训练plda

    plda_file = file_name + '.txt'
    f = open(plda_file, 'wb')
    pickle.dump(overfit_classifier, f)
    f.close()
    g = open(plda_file, 'rb')
    bb = pickle.load(g)
    g.close()
    predictions, log_p_predictions = bb.predict(test_X)  # 预测，使用plda
    # print(predictions)
    # print(log_p_predictions)
    # predictions, log_p_predictions = overfit_classifier.predict(test_X)
    acc = (test_y == predictions).mean()
    print('Accuracy: {:.4f}'.format(acc))

    f1score = f1_score(test_y,predictions,average='macro')
    print('f1score:{:.4f}'.format(f1score))
    # acc_str = 'Acc: ' + str(acc)
    # f1_str = 'F1: ' + str(f1score)
    # with open(file_name + '_result.txt', 'w') as f:
    #     f.write(acc_str + '\n')
    #     f.write(f1_str)



def test_tdnn(net,testloader,device):
    net.eval()

    testdata = np.array([])
    testlabel = np.array([])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs,_ = net(inputs)
            # print(outputs)
            for out in outputs:
                # print(out)
                pre = np.argmax(out.cpu())
                # print(pre)
                testdata = np.append(testdata, pre.cpu())
            for label in targets:
                testlabel = np.append(testlabel, label.cpu())

    print('testdata:',testdata.shape)
    print('testlabel',testlabel.shape)
    # print(testdata)
    # print(testlabel)
    # s1:1457
    # s2:2668
    # print('jiance')
    # known_num = 2342
    # test_X, test_y = testdata[0:known_num], testlabel[0:known_num]
    test_X, test_y = testdata, testlabel

    print('Accuracy: {}'.format((test_X == test_y).mean()))
    f1score = f1_score(test_X, test_y, average='macro')
    print('f1score=', f1score)

if __name__ == '__main__':
    main(args)
