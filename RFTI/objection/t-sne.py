import os
import numpy as np
import cv2
from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets #手写数据集要用到
from sklearn.manifold import TSNE
import torchvision.transforms as transforms
import torch.utils.data
import torch.nn as nn
import torch
from PIL import  Image
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from makeDataset import *
from build_model import *
# import torchvision.transforms as transforms
import torch.utils.data
import torch.nn as nn
import time
# from utils import accuracy, AverageMeter, save_checkpoint
from PIL import  Image
import numpy as np

# from torch.utils.data import Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#该函数是关键，需要根据自己的数据加以修改，将图片存到一个np.array里面，并且制作标签
#因为是两类数据，所以我分别用0,1来表示
def get_data(Input_path): #Input_path为你自己原始数据存储路径，我的路径就是上面的'./Images'
    net = ResNeSt()
    test_model = "./models_pkl/checkpoint_relu_fc2.pth.tar"
    # test_model = "./models_pkl/model_best_np_hard_samples.pth.tar"
    ckpt = torch.load(test_model,map_location="cpu")
    # net.load_state_dict(ckpt["state_dict"])
    net.load_state_dict(ckpt["state_dict"])
    net = net.cuda()

    rootPath = 'H:\SoftwareProgram\Data\dataset\snow leopard\original'
    # imageListPath = rootPath + '/original/test1.txt'
    imageListPath = rootPath + '/test3.txt'
    imagePaths = open(imageListPath).read().strip().split('\n')  # 1424
    dataLen = 1000
    # Image_names=os.listdir(Input_path) #获取目录下所有图片名称列表
    data=np.zeros((len(imagePaths),dataLen)) #初始化一个np.array数组用于存数据
    label=np.zeros((len(imagePaths),)) #初始化一个np.array数组用于存数据
    #为前500个分配标签1，后500分配0
    # for k in range(500):
    #     label[k]=1

    #读取并存储图片数据，原图为rgb三通道，而且大小不一，先灰度化，再resize成200x200固定大小
    for i,imagePath in enumerate(imagePaths):
        # image_path=os.path.join(Input_path,Image_names[i])
        image_path = 'H:' + imagePath.strip().split(',,,')[0].split('F:')[-1]
        # if '/original/original/' in imagePath.strip().split(',,,')[0] and imagePath.strip().split(',,,')[-1] == '0':
        #     label[i] = 0
        # if '/original/original/' in imagePath.strip().split(',,,')[0] and imagePath.strip().split(',,,')[-1] == '1':
        #     label[i] = 1
        # if '\hard_samples\hard_samples' in imagePath.strip().split(',,,')[0] and imagePath.strip().split(',,,')[-1] == '0':
        #     label[i] = 2
        # if '\hard_samples\hard_samples' in imagePath.strip().split(',,,')[0] and imagePath.strip().split(',,,')[-1] == '1':
        #     label[i] = 3
        if imagePath.strip().split(',,,')[-1] == '0':
            label[i] = 0
        if imagePath.strip().split(',,,')[-1] == '1':
            label[i] = 1
        # else:
        #     label[i] = 3
        # img=cv2.imread(image_path)
        try:
            img = Image.open(image_path).convert('RGB')
        except:
            print(image_path)
        # try:
        #     img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # except:
        #     print(image_path)
        # img=cv2.resize(img,(512,512))
        # img = np.transpose(img,(2,0,1))
        img = transforms.Compose([
        transforms.Resize([512, 512]),
        # transforms.CenterCrop([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )])(img)
        feature = embedding(img,net) #1000维
        # img=img.reshape(1,40000)
        data[i]=feature
        print(i)
    n_samples,n_features = data.shape
    return data, label, n_samples, n_features

#下面的两个函数，
#一个定义了二维数据，一个定义了3维数据的可视化
#不作详解，也无需再修改感兴趣可以了解matplotlib的常见用法

def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    cls = ['n','p']
    # cls = ['0','1','2','3']
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], cls[int(label[i])],
                 color=plt.cm.Set1(int(label[i])), #plt.cm.Set1的参数必须是整数
                 fontdict={'weight': 'bold', 'size': 13})
        # print(label[i],str(label[i]),plt.cm.Set1(int(label[i])))
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    # plt.legend(['pOD','nOD','pHS','nHS'])
    return fig
# def plot_embedding_3D(data,label,title):
#     x_min, x_max = np.min(data,axis=0), np.max(data,axis=0)
#     data = (data- x_min) / (x_max - x_min)
#     ax = plt.figure().add_subplot(111,projection='3d')
#     # cls = ['pOD','nOD','pHS','nHS']
#     cls = ['0','1','2','3']
#     for i in range(data.shape[0]):
#         ax.text(data[i, 0], data[i, 1], data[i,2],cls[int(label[i])], color=plt.cm.Set1(int(label[i])),fontdict={'weight': 'bold', 'size': 9})
#     # return fig
#     plt.title(title)
#     return ax
#
# def default_loader(path):
#     try:
#         img = Image.open(path).convert('RGB')
#     except:
#         with open('read_error.txt', 'a') as fid:
#             fid.write(path+'\n')
#         return Image.new('RGB', (224,224), 'white')
#     return img

def embedding(img,net):

    net.eval()
    # img = torch.from_numpy(img)
    inputs = img.float().cuda().unsqueeze(0)
    feature = net.resnest50(inputs).squeeze(0) #1000维
    # feature = net.ReLU1(feature).squeeze(0) #1000维
    # feature = net.fc1(feature).squeeze(0) #1000维

    feature = feature.cpu().detach().numpy()

    return feature



#主函数
def main():
    data, label, n_samples, n_features = get_data('./Images') #根据自己的路径合理更改
    print('Begining......') #时间会较长，所有处理完毕后给出finished提示
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0) #调用TSNE
    result_2D = tsne_2D.fit_transform(data)
    # tsne_3D = TSNE(n_components=3, init='pca', random_state=0)
    # result_3D = tsne_3D.fit_transform(data)
    print('Finished......')
    #调用上面的两个函数进行可视化
    fig1 = plot_embedding_2D(result_2D, label,'t-SNE')
    plt.show(fig1)

    # fig2 = plot_embedding_3D(result_3D, label,'t-SNE')
    # plt.show(fig2)
if __name__ == '__main__':
    main()