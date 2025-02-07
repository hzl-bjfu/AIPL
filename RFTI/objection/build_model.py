"""
-*- coding:utf-8 -*-
resnest50 baseline
"""
import sys
# sys.path.append("..")  #将使本文件上一级目录加入路径
import os
import torch
import torch.nn as nn

from resnest.torch import resnest50,resnest50_fast_2s1x64d,resnest101



class ResNeSt(nn.Module):
    def __init__(self):
        super(ResNeSt, self).__init__()
        self.resnest50 = resnest50(pretrained=False)
        model_state_dict = torch.load(
            'H:\SoftwareProgram\Data\model/resnest50-528c19ca.pth'
        )
        self.resnest50.load_state_dict(model_state_dict)
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



