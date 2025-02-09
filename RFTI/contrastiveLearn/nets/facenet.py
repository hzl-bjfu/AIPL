import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import efficientnet_b0


# 假设的 nets.mobilenet 模块中的 MobileNetV1 实现
class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        # 这里只是简单示例结构，并非完整的 MobileNetV1 实现
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 假设的 nets.inception_resnetv1 模块中的 InceptionResnetV1 实现
class InceptionResnetV1(nn.Module):
    def __init__(self):
        super(InceptionResnetV1, self).__init__()
        # 以下只是简单示例结构，并非完整的 InceptionResnetV1 实现
        self.conv2d_1a = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_3b = nn.Conv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = nn.Conv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = nn.Conv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.mixed_6a = nn.Conv2d(256, 768, kernel_size=3, stride=2)
        self.repeat_2 = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.mixed_7a = nn.Conv2d(768, 1792, kernel_size=3, stride=2)
        self.repeat_3 = nn.Sequential(
            nn.Conv2d(1792, 1792, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1792),
            nn.ReLU(inplace=True)
        )
        self.block8 = nn.Conv2d(1792, 1792, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        return x


class mobilenet(nn.Module):
    def __init__(self):
        super(mobilenet, self).__init__()
        self.model = MobileNetV1()
        del self.model.fc
        del self.model.avg

    def forward(self, x):
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        return x


class inception_resnet(nn.Module):
    def __init__(self):
        super(inception_resnet, self).__init__()
        self.model = InceptionResnetV1()

    def forward(self, x):
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.conv2d_4b(x)
        x = self.model.repeat_1(x)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_2(x)
        x = self.model.mixed_7a(x)
        x = self.model.repeat_3(x)
        x = self.model.block8(x)
        return x


class efficientnet_b0_model(nn.Module):
    def __init__(self):
        super(efficientnet_b0_model, self).__init__()
        self.model = efficientnet_b0(pretrained=False)
        del self.model.classifier

    def forward(self, x):
        x = self.model.features(x)
        return x


class Facenet(nn.Module):
    def __init__(self, backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train"):
        super(Facenet, self).__init__()
        if backbone == "mobilenet":
            self.backbone = mobilenet()
            flat_shape = 1024
        elif backbone == "inception_resnetv1":
            self.backbone = inception_resnet()
            flat_shape = 1792
        elif backbone == "efficientnet_b0":
            self.backbone = efficientnet_b0_model()
            flat_shape = 1280
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1, efficientnet_b0.'.format(backbone))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        x = self.last_bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward_feature(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        return before_normalize, x

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x


# 简单测试代码
if __name__ == "__main__":
    model = Facenet(backbone="efficientnet_b0", num_classes=10, mode="train")
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print("Output shape:", output.shape)