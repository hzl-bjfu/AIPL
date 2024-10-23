import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import os
import time
import pickle
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import numpy as np
# from TDNN.tdnn import TDNN
from early_stopping import EarlyStopping
from Utils import adjust_learning_rate, progress_bar, Logger, mkdir_p, Evaluation
import random
import argparse
from TDNN2 import tdnn_IFN,tdnn_BN,tdnn_both,tdnn_LSTM,tdnn_TN,tdnn_GW

parser = argparse.ArgumentParser(description='传入参数')  #定义：解析参数
#type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('--model_name', default='tdnn',type=str, help='保存模型')    ##
parser.add_argument('--tdnn_method', default='tdnn_both',type=str, help='网络结构')  ##选择模型结构

args = parser.parse_args()  #实际来解析参数
from sklearn.metrics import f1_score
# os.system('python train_plda.py')
os.environ["CUDA_VISIBLE_DEVICES"]="0"

BATCH_SIZE = 32
class_num = 8  ##训练类别
EPOCHS = 50
lr = 1e-4


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def zenododata(path):
    with open(path,'rb') as f:
        data,label = pickle.load(f)
        return data,label

class getdata(Dataset):
    # dirname 为训练/测试数据地址，使得训练/测试分开
    def __init__(self, dirname, train=True):
        super(getdata, self).__init__()
        self.images, self.labels = zenododata(dirname)

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

    # setup_seed(55)

    pkl_path = 'D:\\work\\code\\TDNN\\Data'
    train_path = pkl_path + '\\D3train.pkl'
    val_path = pkl_path + '\\D3val.pkl'
    # test_path = pkl_path + "/2023cross_S2test.pkl"

    dataset_train = getdata(train_path)
    dataset_val = getdata(val_path)
    # dataset_test = getdata(test_path)        x = self.dropoutlayer2(x)

    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
    valloader = torch.utils.data.DataLoader(dataset_val, batch_size=32, shuffle=False)
    # testloader = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=False)
    print('------------------------')
    print(f"Start training...")
    start_epoch = 0

    # Model
    # print('==> Building model..')
    if args.tdnn_method == 'tdnn_BN':
        net = tdnn_BN.TDNN(feat_dim=128, embedding_size=512, num_classes=class_num)
    if args.tdnn_method == 'tdnn_IFN':
        net = tdnn_IFN.TDNN(feat_dim=128, embedding_size=512, num_classes=class_num)
    if args.tdnn_method == 'tdnn_LSTM':
        net = tdnn_LSTM.TDNN(feat_dim=128, embedding_size=512, num_classes=class_num)
    if args.tdnn_method == 'tdnn_both':
        net = tdnn_both.TDNN(feat_dim=128, embedding_size=512, num_classes=class_num)
    if args.tdnn_method == 'tdnn_GW':
        net = tdnn_GW.TDNN(feat_dim=128, embedding_size=512, num_classes=class_num)
    if args.tdnn_method == 'tdnn_TN':
        net = tdnn_TN.TDNN(feat_dim=128, embedding_size=512, num_classes=class_num)
    # net = TDNN(feat_dim=128,embedding_size=512,num_classes=class_num)   ###暂时注销
    net = net.to(DEVICE)

    if DEVICE == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr, weight_decay=5e-4)
    train_losses = []
    eval_losses = []
    eval_acces = []
    save_path = ".\\"
    early_stopping = EarlyStopping(save_path)
    for epoch in range(start_epoch, EPOCHS):
        print('\nEpoch: %d   Learning rate: %f' % (epoch+1, optimizer.param_groups[0]['lr']))   #暂时注销
        adjust_learning_rate(optimizer, epoch, lr,factor=0.8,step=3)
        train_loss, train_acc = train(net,trainloader,optimizer,criterion,DEVICE)
        train_losses.append(train_loss / len(trainloader))
        # ==========================eval==========================
        eval_loss = 0
        eval_acc = 0

        net.eval()
        testdata = np.array([])
        testlabel = np.array([])
        with torch.no_grad():
            for im, label in valloader:
                im, label = im.to(DEVICE), label.to(DEVICE)
                out,_ = net(im)
                loss = criterion(out, label)

                # 记录误差
                eval_loss += loss.item()

                # 记录准确率
                out_t = out.argmax(dim=1)  # 取出预测的最大值的索引
                num_correct = (out_t == label).sum().item()  # 判断是否预测正确
                acc = num_correct / im.shape[0]  # 计算准确率
                eval_acc += acc

                #----------------------------------------------------
                # for outputs in out:
                #     pre = np.argmax(outputs.cpu())
                #     testdata = np.append(testdata, pre.cpu())
                # for labels in label:
                #     testlabel = np.append(testlabel, labels.cpu())
                #----------------------------------------------------


        eval_losses.append(eval_loss / len(valloader))
        eval_acces.append(eval_acc / len(valloader))
        # scheduler.step()

        # print('epoch: {}, Eval Loss: {:.6f}, Eval Acc: {:.4f}'
        #       .format(epoch + 1, eval_loss / len(valloader), eval_acc / len(valloader)))   #暂时注销
        # f1score = f1_score(testdata, testlabel, average='macro')
        # print('f1score: {:.4f}' .format(f1score))

        # early stopping
        # early_stopping(eval_loss, net)
        early_stopping(eval_loss, net)
        if early_stopping.early_stop:
            # print("Early stopping")   #暂时注销
            break  # 跳出迭代，结束训练
    # with open('bothS2_train_losses.txt','w') as f:
    #     f.write(str(train_losses))
    # with open('bothS2_val_losses.txt','w') as F:
    #     F.write(str(eval_losses))

    print(f"Finish training...\n")
    return net

# Training
def train(net,trainloader,optimizer,criterion,device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs,_ = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1), correct/total

def test(net,valloader,optimizer,criterion,device):
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(valloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # loss.backward()
        optimizer.step()

        val_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(valloader), 'val_Loss: %.3f | val_Acc: %.3f%% (%d/%d)'
                     % (val_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return val_loss / (batch_idx + 1), correct / total

if __name__ == '__main__':
    main(args)

