import torch
import torch.nn as nn
import cv2
from torchvision import models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageFont
from torchsummary import summary
from resnest.torch import resnest50,resnest50_fast_2s1x64d,resnest101


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResNeSt(nn.Module):
    def __init__(self):
        super(ResNeSt, self).__init__()
        self.model = resnest50(pretrained=False)
        model_state_dict = torch.load(
            'D:\ScienceResearch\SoftwareProgram\Data\model/resnest50-528c19ca.pth'
        )
        self.model.load_state_dict(model_state_dict)
        self.model = nn.Sequential(*list(self.model.children())[:-2])  # 去掉模型的最后两层：最大全局平均池化与fc
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Bottleneck = nn.Linear(2048, 128, bias=False)

    def forward(self,input):
        x = self.model(input)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Bottleneck(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = models.resnet50(pretrained=False)
        model_state_dict = torch.load(
            'D:\ScienceResearch\SoftwareProgram\Data\model/resnet50-19c8e357.pth'
        )
        self.model.load_state_dict(model_state_dict)
        self.model = nn.Sequential(*list(self.model.children())[:-2])  # 去掉模型的最后两层：最大全局平均池化与fc


    def forward(self,input):
        x = self.model(input)
        # x = self.avg(x)
        # x = x.view(x.size(0), -1)
        # x = self.Bottleneck(x)
        return x

net = ResNet().to(device)
for i in net.children():
    print(i)
# net2 = models.resnet50(pretrained=False).to(device)
# # for i in net2.children():
# #     print(i)
#
#
# # img = cv2.imread("D:\ScienceResearch\SoftwareProgram\Data\dataset\CUB_200_2011\images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg")
# img = Image.open("D:\ScienceResearch\SoftwareProgram\Data\dataset\CUB_200_2011\images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg")
#
# x = transforms.Resize([512,512])(img)#只有PIL中Image.open的图像才能transforms.Resize
# x = transforms.ToTensor()(img)#用于将Image读取的img转换为tensor，只有tensor对象才有view成员函数
# # x = torch.tensor(img) #用于将cv2读取的img转换为tensor，只有tensor对象才有view成员函数
# x = transforms.RandomHorizontalFlip()(x)
#
# # x = x.view(1,x.size(2),x.size(0),x.size(1))
# # x = x.view(-1,x.size(0),x.size(1),x.size(2)).cuda()#只有tensor对象才有view成员函数
# x = x.view(2,x.size(0),x.size(1),-1).cuda()#只有tensor对象才有view成员函数
# # print(x.size())
# out = net(x)
#
# summary(net,(3,512,512)) #查看每一层输出的特征图大小


