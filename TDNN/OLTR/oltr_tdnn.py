from collections import OrderedDict

from torch import nn
from layers import DenseLayer, StatsPool, TDNNLayer, get_nonlinear
from classifier import *
from oltr import ModulatedAttLayer
import torch.nn.functional as F

class TDNN(nn.Module):
    def __init__(self,
                 feat_dim=30,
                 embedding_size=512,
                 num_classes=8,
                 use_fc=False,
                 attmodule=True,
                 classifier='dotproduct',
                 config_str='batchnorm-relu',
                 config_str2='batchnorm-relu'):
        super(TDNN, self).__init__()

        self.use_fc = use_fc
        if attmodule:
            self.att = ModulatedAttLayer(feat_dim, height=4, width=4)

        # here for use fc
        if self.use_fc:
            self.fc_add = nn.Linear(feat_dim, embedding_size)
            feat_dim = embedding_size

        self.xvector = nn.Sequential(OrderedDict([
            ('tdnn1', TDNNLayer(128, 512, 5, dilation=1, padding=-1,
                                config_str=config_str)),
            ('tdnn2', TDNNLayer(512, 512, 3, dilation=2, padding=-1,
                                config_str=config_str)),
            # ('LSTM', LSTM()),
            ('tdnn3', TDNNLayer(512, 512, 3, dilation=3, padding=-1,
                                config_str=config_str)),
            ('tdnn4', DenseLayer(512, 512, config_str=config_str)),
            ('tdnn5', DenseLayer(512, 1500, config_str=config_str)),
            ('stats', StatsPool()),
            ('affine', nn.Linear(3000, embedding_size))
        ]))

        self.nonlinear = get_nonlinear(config_str2, embedding_size)
        self.dense = DenseLayer(embedding_size,
                                embedding_size,
                                config_str=config_str2)
        classifier_map = {
            "dotproduct": DotProduct_Classifier(feat_dim, num_classes),
            "cosnorm": CosNorm_Classifier(feat_dim, num_classes),
            "metaembedding": MetaEmbedding_Classifier(feat_dim, num_classes)
        }
        # self.classifier = nn.Linear(embedding_size, num_classes)
        self.classifier = classifier_map[classifier]

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)  #进行一种特定的初始化
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # print('inputsize:',x.shape)
        xvector = self.xvector(x)
        # if self.training:
            # print('x.shape:',x.shape)
            # x = SimAM(x)
        x = self.dense(self.nonlinear(xvector))
        # print('x.shape:', x.shape)
        #backnone
        feature_maps = None
        if hasattr(self, 'att'):
            x, feature_maps = self.att(x)
            # print('x.shape:', x.shape)
        if self.use_fc:
            x = F.relu(self.fc_add(x))
        y, fea = self.classifier(x)
        # x = self.classifier(x)
        return y,fea, feature_maps, x

# class SimAM(torch.nn.Module):
#     def __init__(self, channels=None, e_lambda=1e-4):
#         super(SimAM, self).__init__()
#
#         self.activaton = nn.Sigmoid()
#         self.e_lambda = e_lambda
#
#     def __repr__(self):
#         s = self.__class__.__name__ + '('
#         s += ('lambda=%f)' % self.e_lambda)
#         return s
#
#     @staticmethod
#     def get_module_name():
#         return "simam"
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         n = w * h - 1
#
#         x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
#         y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
#
#         return x * self.activaton(y)