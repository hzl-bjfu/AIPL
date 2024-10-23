import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets

import os
import time
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import numpy as np

import argparse
import sys

from Utils import adjust_learning_rate, progress_bar, Logger, mkdir_p, Evaluation
from openmax import compute_train_score_and_mavs_and_dists,fit_weibull,openmax
from Plotter import plot_feature
from torch.utils.data import Dataset, DataLoader
import pickle
from TDNN.tdnn import TDNN

parser = argparse.ArgumentParser(description='PyTorch Bird10 Testing')
parser.add_argument('--train_class_num', default=10, type=int, help='Classes used in training')
parser.add_argument('--test_class_num', default=11, type=int, help='Classes used in testing')
parser.add_argument('--includes_all_train_class', default=True,  action='store_true',
                    help='If required all known classes included in testing')
parser.add_argument('--evaluate', action='store_true',help='Evaluate without training')

#Parameters for weibull distribution fitting.
parser.add_argument('--weibull_tail', default=20, type=int, help='Classes used in testing')
parser.add_argument('--weibull_alpha', default=3, type=int, help='Classes used in testing')
parser.add_argument('--weibull_threshold', default=0.9, type=float, help='Classes used in testing')

# Parameters for stage plotting
parser.add_argument('--plot', default=True, action='store_true', help='Plotting the training set.')
parser.add_argument('--plot_max', default=0, type=int, help='max examples to plot in each class, 0 indicates all.')
parser.add_argument('--plot_quality', default=200, type=int, help='DPI of plot figure')
args = parser.parse_args()

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

    args.checkpoint = './checkpoint/zenodo'
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    args.plotfolder = './checkpoints/zenodo/' + '/plotter'
    if not os.path.isdir(args.plotfolder):
        mkdir_p(args.plotfolder)

    BATCH_SIZE = 32

    pkl_path = 'E:/data/pkl'
    train_path = pkl_path + "/cross_S2train.pkl"
    val_path = pkl_path + "/cross_S2val.pkl"
    test_path = pkl_path + "/cross_S2test.pkl"

    dataset_train = getdata(train_path)
    dataset_val = getdata(val_path)
    dataset_test = getdata(test_path)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    net = TDNN(feat_dim=128, embedding_size=512, num_classes=10)
    net.load_state_dict(torch.load('crossS2_BN.pth'))
    net.to(DEVICE)
    test(net,train_loader,test_loader,DEVICE)
    # plot_feature(net, test_loader, DEVICE, args.plotfolder, epoch="test",
    #               plot_class_num=args.train_class_num + 1, maximum=args.plot_max, plot_quality=args.plot_quality)


def test( net,trainloader,  testloader, device):
    net.eval()
    #net.train()

    test_loss = 0
    correct = 0
    total = 0

    scores, labels = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs,_ = net(inputs)

            scores.append(outputs)
            labels.append(targets)

            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader))

    # Get the prdict results.
    scores = torch.cat(scores,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    scores = np.array(scores)[:, np.newaxis, :]
    labels = np.array(labels)

    # Fit the weibull distribution from training data.
    print("Fittting Weibull distribution...")
    _, mavs, dists = compute_train_score_and_mavs_and_dists(args.train_class_num, trainloader, device, net)
    categories = list(range(0, args.train_class_num))
    weibull_model = fit_weibull(mavs, dists, categories, args.weibull_tail, "euclidean")

    pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
    score_softmax, score_openmax = [], []
    for score in scores:
        so, ss = openmax(weibull_model, categories, score,
                         0.5, args.weibull_alpha, "euclidean")  # openmax_prob, softmax_prob
        pred_softmax.append(np.argmax(ss))
        pred_softmax_threshold.append(np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
        pred_openmax.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
        score_softmax.append(ss)
        score_openmax.append(so)

    print("Evaluation...")
    eval_softmax = Evaluation(pred_softmax, labels, score_softmax)
    eval_softmax_threshold = Evaluation(pred_softmax_threshold, labels, score_softmax)
    eval_openmax = Evaluation(pred_openmax, labels, score_openmax)
    torch.save(eval_softmax, os.path.join(args.checkpoint, 'eval_softmax.pkl'))
    torch.save(eval_softmax_threshold, os.path.join(args.checkpoint, 'eval_softmax_threshold.pkl'))
    torch.save(eval_openmax, os.path.join(args.checkpoint, 'eval_openmax.pkl'))

    # confus_sofmax = eval_softmax.confusion_matrix
    # confus_sofmax_thr = eval_softmax_threshold.confusion_matrix
    # confus_openmax = eval_openmax.confusion_matrix
    #
    # print(confus_sofmax)
    # print(confus_sofmax_thr)
    # print(confus_openmax)

    print(f"Softmax accuracy is %.3f" % (eval_softmax.accuracy))
    #print(f"Softmax F1 is %.3f" % (eval_softmax.f1_measure))
    #print(f"Softmax f1_macro is %.3f" % (eval_softmax.f1_macro))
    print(f"Softmax f1_macro_weighted is %.3f" % (eval_softmax.f1_macro_weighted))
    #print(f"Softmax area_under_roc is %.3f" % (eval_softmax.area_under_roc))
    #print(f"Softmax recall_weighted is %.3f" % (eval_softmax.recall_weighted))
    print(f"_________________________________________")

    print(f"SoftmaxThreshold accuracy is %.3f" % (eval_softmax_threshold.accuracy))
    #print(f"SoftmaxThreshold F1 is %.3f" % (eval_softmax_threshold.f1_measure))
    #print(f"SoftmaxThreshold f1_macro is %.3f" % (eval_softmax_threshold.f1_macro))
    print(f"SoftmaxThreshold f1_macro_weighted is %.3f" % (eval_softmax_threshold.f1_macro_weighted))
    #print(f"SoftmaxThreshold area_under_roc is %.3f" % (eval_softmax_threshold.area_under_roc))
    #print(f"SoftmaxThreshold recall_weighted is %.3f" % (eval_softmax_threshold.recall_weighted))
    print(f"_________________________________________")

    print(f"OpenMax accuracy is %.3f" % (eval_openmax.accuracy))
    #print(f"OpenMax F1 is %.3f" % (eval_openmax.f1_measure))
    #print(f"OpenMax f1_macro is %.3f" % (eval_openmax.f1_macro))
    print(f"OpenMax f1_macro_weighted is %.3f" % (eval_openmax.f1_macro_weighted))
    #print(f"OpenMax area_under_roc is %.3f" % (eval_openmax.area_under_roc))
    #print(f"OpenMax recall_weighted is %.3f" % (eval_openmax.recall_weighted))
    print(f"_________________________________________")
    print(eval_openmax.confusion_matrix)

    label_names = ["0" ,"1", "2", "3", "4", "5","6","7","8","9","10","11","12", "unknown"]
    # label_names = ["0" ,"1", "2", "3", "4", "5","6","7","8","9", "unknown"]
    save_path3 = './checkpoints/zenodo/confusion/openmax.png'
    #
    #eval_openmax.plot_confusion_matrix(normalize="true", labels=label_names,savepath=save_path3)

if __name__ == '__main__':
    main()
