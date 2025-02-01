import random

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import torch


# def rand(a=0, b=1):
#     return np.random.rand() * (b - a) + a
#
# def get_random_data(image, input_shape, jitter=.1, hue=.05, sat=1.3, val=1.3, flip_signal=True,channel = 3):
#     image = image.convert("RGB")
#
#     h, w = input_shape
#     rand_jit1 = rand(1 - jitter, 1 + jitter)
#     rand_jit2 = rand(1 - jitter, 1 + jitter)
#     new_ar = w / h * rand_jit1 / rand_jit2
#
#     scale = rand(0.9, 1.1)
#     if new_ar < 1:
#         nh = int(scale * h)
#         nw = int(nh * new_ar)
#     else:
#         nw = int(scale * w)
#         nh = int(nw / new_ar)
#     image = image.resize((nw, nh), Image.BICUBIC)
#
#     flip = rand() < .5
#     if flip and flip_signal:
#         image = image.transpose(Image.FLIP_LEFT_RIGHT)
#
#     dx = int(rand(0, w - nw))
#     dy = int(rand(0, h - nh))
#     new_image = Image.new('RGB', (w, h), (128, 128, 128))
#     new_image.paste(image, (dx, dy))
#     image = new_image
#
#     rotate = rand() < .5
#     if rotate:
#         angle = np.random.randint(-5, 5)
#         a, b = w / 2, h / 2
#         M = cv2.getRotationMatrix2D((a, b), angle, 1)
#         image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[128, 128, 128])
#
#     hue = rand(-hue, hue)
#     sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
#     val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
#     x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
#     x[..., 0] += hue * 360
#     x[..., 0][x[..., 0] > 1] -= 1
#     x[..., 0][x[..., 0] < 0] += 1
#     x[..., 1] *= sat
#     x[..., 2] *= val
#     x[x[:, :, 0] > 360, 0] = 360
#     x[:, :, 1:][x[:, :, 1:] > 1] = 1
#     x[x < 0] = 0
#     image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
#
#     if channel == 1:
#         image_data = Image.fromarray(np.uint8(image_data)).convert("L")
#     # cv2.imshow("TEST",np.uint8(cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)))
#     # cv2.waitKey(0)
#     return image_data
#
# def letterbox_image(image, input_shape):
#     image = image.convert("RGB")
#     iw, ih = image.size
#     w, h = [input_shape[1], input_shape[0]]
#     scale = min(w / iw, h / ih)
#     nw = int(iw * scale)
#     nh = int(ih * scale)
#
#     image = image.resize((nw, nh), Image.BICUBIC)
#     new_image = Image.new('RGB', [input_shape[1], input_shape[0]], (128, 128, 128))
#     new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
#     if input_shape[-1] == 1:
#         new_image = new_image.convert("L")
#     return new_image


# print(1)
#
# a = torch.tensor([[1.0,2.0,3],[4.0,5.0,6.0]]).type(torch.FloatTensor)
# b = torch.tensor([[10,20,30],[40,50,60]]).type(torch.FloatTensor)
# c = torch.tensor([[100,200,300],[400,500,600]]).type(torch.FloatTensor)
# d = torch.tensor([[100,200,300],[400,500,600]]).type(torch.FloatTensor)
#
# pos_dist = torch.sqrt(torch.sum(torch.pow(a - b, 2), axis=-1))
# neg_dist = torch.sqrt(torch.sum(torch.pow(a - c, 2), axis=-1))
# neg_dist2 = torch.sqrt(torch.sum(torch.pow(a - d, 2), axis=-1))
#
# alpha=1000
# alpha2=500
# # alpha_list =torch.tensor(alpha,a.shape).type(torch.FloatTensor)
#
# p_n_alp = pos_dist-neg_dist+alpha
# p_n_alp2 = pos_dist-neg_dist+alpha2
# p_n2_alp2 = pos_dist-neg_dist2+alpha2
# n_n2_alp = neg_dist-neg_dist2+alpha2
#
# p_n_alp = p_n_alp[torch.where(p_n_alp > 0)].cuda()
# p_n_alp2 = p_n_alp2[torch.where(p_n_alp2 < 0)].cuda()  #不一定有
# p_n2_alp2 = p_n2_alp2[torch.where(p_n2_alp2 > 0)].cuda()
# n_n2_alp = n_n2_alp[torch.where(n_n2_alp > 0)].cuda()  #不一定有
#
#
# basic_loss = torch.sum(p_n_alp)-torch.sum(p_n_alp2)+torch.sum(p_n2_alp2)+torch.sum(n_n2_alp)
# loss = basic_loss/torch.max(torch.tensor(1), torch.tensor(len(p_n_alp)+len(p_n_alp2)+len(p_n2_alp2)+len(n_n2_alp)))
# print(loss)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

# from nets.facenet import Facenet
from nets.my_facenet import Facenet
# from nets.facenet_training import triplet_loss, LossHistory, weights_init
from nets.my_facenet_training import triplet_loss,triplet_loss_N2, LossHistory, weights_init
# from utils.dataloader import FacenetDataset, dataset_collate
from utils.my_dataloader import FacenetDataset,dataset_collate,FacenetDataset_N2,dataset_collate_N2
from utils.eval_metrics import evaluate
from utils.LFWdataset import LFWDataset
from utils.myEarlyStopping import EarlyStopping

from mytest import face_distance,compare_faces,myFacenet
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Facenet(num_classes=2 * 25, backbone='mobilenet')  # cfy,num_classes*2为所有标签数量，计算CEloss时使用
Cuda = True
model_path = "models_pkl/facenet_mobilenet.pth"

model_dict = model.state_dict()
pretrained_dict = torch.load(model_path, map_location=device)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

net = model.train()
model.to(device)
if Cuda:
    net = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    net = net.cuda()

# net.eval()
net.module.eval()
print(1)