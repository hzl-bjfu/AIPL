"""cam for model attention visual"""
import torch
from time import strftime, localtime
import shutil
import cv2
import sys
import numpy
import numpy as np
import os
import os.path as osp
from torch.autograd import Function
import sys

import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from build_model import *
# from models.build_model import BuildModel
from torchvision.transforms import transforms


# extract features
class FeatureExtractor:

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):

        self.gradients = []
        outputs = []
        for name, module in self.model._modules.items():
            # if name == 'layer4':
            if name == 'resnest50':
                for name1, module1 in self.model._modules[name]._modules.items():
                    if name1 == 'layer4':
                        x = self.model._modules[name]._modules[name1][0](x)
                        x = self.model._modules[name]._modules[name1][1](x)
                        x = self.model._modules[name]._modules[name1][2].conv1(x)
                        x = self.model._modules[name]._modules[name1][2].bn1(x)
                        x = self.model._modules[name]._modules[name1][2].conv2(x)
                        # x = self.model._modules[name]._modules[name1][2].bn2(x)
                        output = self.model._modules[name]._modules[name1][2].conv3(x)
                        # add a register_hook for gradient
                        output.register_hook(self.save_gradient)
                        x = self.model._modules[name]._modules[name1][2].bn3(output)
                        x = self.model._modules[name]._modules[name1][2].relu(x)
                        outputs += [output]
                    elif name1 == 'avgpool':
                        continue
                    elif name1 == 'fc':
                        continue
                    else:
                        x = module1(x)
            elif name == 'fc1':
                continue
            elif name == 'ReLU1':
                continue
            else:
                x = module(x)

        return outputs, x


class ModelOutputs:

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        # output = self.model.avgpool(output)
        output = self.model.resnest50.avgpool(output)
        output = output.view(output.size(0), -1)
        # output = self.model.fc(output)
        output = self.model.resnest50.fc(output)
        output = self.model.fc1(output)
        output = self.model.ReLU(output)
        return target_activations, output


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class GradCam:

    def __init__(self, model, use_cuda, target_layer):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            input = input.to(self.device)
            features, output = self.extractor(input)
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot)

        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (448, 448))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, index


def paste_image(image, heat_map, num_pred, output_path):
    txt_file = "C:\Windows\Fonts/AdobeArabic-Regular.otf"
    ft = ImageFont.truetype(txt_file, 18)
    heat_image = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)
    if type(image) == np.ndarray:
        image_raw = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_map = Image.fromarray(heat_image)
    target_shape = (448 * 2, 448 + 20)
    target = Image.new("RGB", target_shape)
    target.paste(image_raw, (0, 0))
    target.paste(image_map, (448, 0))
    draw = ImageDraw.Draw(target)
    draw.text((10, 450), str(num_pred), fill=(255, 0, 0), font=ft)
    target.save(output_path)


def tensor2image(image_tensor):
    if len(image_tensor.size()) == 4:
        image_tensor = image_tensor.squeeze(0)
    else:
        image_tensor = image_tensor
    image_array = image_tensor.cpu().detach().numpy() * 255
    image_array.astype(np.uint8)
    img_bgr = np.transpose(image_array, (2, 1, 0))
    image_CV2 = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
    return image_CV2

def search(root, target):
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if path.split('/')[-1] == target:
            print('[+]',path)
        else:
            print('[!]',path)
        # if os.path.isdir(path):
        #     print('[-]', path)
        #     search(path, target)
        # elif path.split('/')[-1] == target:
        #     print('[+]', path)
        # else:
        #     print('[!]', path)

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_model = "./models_pkl/checkpoint_relu_fc2_relu.pth.tar"
    # net = BuildModel("resnet50", 5000, False)
    # net = ResNeSt().to(device)
    net = ResNeSt()
    # model = net()
    model = net
    ckpt = torch.load(test_model, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.cuda()
    print(model)

    # grad_cam = GradCam(model, True, 'layer4')
    grad_cam = GradCam(model, True, 'layer4')

    # test_1000_cfy_path = './data/test_1000_cfy'
    test_path = './data/test_original'

    images = os.listdir(test_path)
    i = 1
    for image_name in images:
        image_path = test_path + '/' + image_name

        image_raw = Image.open(image_path)
        if image_raw.mode is not "RGB":
            image_raw = image_raw.convert("RGB")

        imagenet_normalization_paramters = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        normalize = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            imagenet_normalization_paramters
        ])

        image = normalize(image_raw).unsqueeze(dim=0)
        # num_classes 用来区分响应类别
        num_classes = None
        # num_classes = 0
        mask, num_index = grad_cam(image, num_classes)
        # output_path = "./data/cam_csae_new/" + str(num_index) + "_" + str(i) + ".jpg"
        output_path = "./data/cam_csae_new_test/" + str(num_index) + "_" + str(i) +str(strftime("%m_%d_%H_%M_%S", localtime())) + ".jpg"
        i = i + 1
        image_array = cv2.cvtColor(np.array(image_raw), cv2.COLOR_RGB2BGR)
        image_array = cv2.resize(image_array, (448, 448))

        image_ = image_array / 255
        heat_image = show_cam_on_image(image_, mask)
        cv2.imwrite(output_path, heat_image)
        # print(num_index)

        paste_image(image_array, heat_image, num_index, output_path)