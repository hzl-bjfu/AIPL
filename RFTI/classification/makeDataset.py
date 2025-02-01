import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from PIL import  Image
import numpy as np

# rootPath = 'F:\cfy_mistake_trigger'


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

        # with open('./data/test_allCamera - 副本.txt', 'r') as fid:
        with open('./data/allCamera/test_allCamera_2n.txt', 'r') as fid:
                self.imglist = fid.readlines()

    def __getitem__(self, index):
        image_name, label = self.imglist[index].strip().split(',,,')
        # if 'SoftwareProgram' in image_name:
        #     image_path = 'H:' + image_name.strip().split(':')[-1]
        # else:
        #     image_path = image_name
        image_path = image_name
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)
        label = torch.LongTensor([label])

        return [img, label,image_path]


    def __len__(self):
        return len(self.imglist)

class BatchDataset(Dataset): #产生训练集
    def __init__(self, transform=None, dataloader=default_loader):
        self.transform = transform
        self.dataloader = dataloader

        with open('./data/allCamera/train_allCamera_2n.txt', 'r') as fid:
                self.imglist = fid.readlines()

        self.labels = []
        for line in self.imglist:
            image_path, label = line.strip().split(',,,')
            self.labels.append(int(label))
        self.labels = np.array(self.labels)
        self.labels = torch.LongTensor(self.labels)


    def __getitem__(self, index):
        image_name, label = self.imglist[index].strip().split(',,,')
        # if 'SoftwareProgram' in image_name:
        #     image_path = 'H:' + image_name.strip().split(':')[-1]
        # else:
        #     image_path = image_name
        image_path = image_name
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)
        label = torch.LongTensor([label])

        return [img, label]


    def __len__(self):
        return len(self.imglist)
