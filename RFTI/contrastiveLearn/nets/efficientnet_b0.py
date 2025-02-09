import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

class efficientnet_b0_model(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(efficientnet_b0_model, self).__init__()
        # 加载预训练的 EfficientNet-B0 模型
        self.model = efficientnet_b0(pretrained=pretrained)
        # 获取模型的特征提取部分
        self.features = self.model.features
        # 获取自适应平均池化层
        self.avgpool = self.model.avgpool
        # 重新定义分类器部分，以适应自定义的类别数
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.model.classifier[0].p),
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )

    def forward(self, x):
        # 通过特征提取部分
        x = self.features(x)
        # 进行自适应平均池化
        x = self.avgpool(x)
        # 展平张量
        x = torch.flatten(x, 1)
        # 通过分类器部分
        x = self.classifier(x)
        return x

# 以下是使用示例
if __name__ == "__main__":
    # 创建一个 EfficientNet-B0 模型实例，假设分类类别数为 10
    model = efficientnet_b0_model(num_classes=10, pretrained=False)
    # 生成一个随机输入张量，模拟一个批量大小为 1，通道数为 3，高度和宽度为 224 的图像
    input_tensor = torch.randn(1, 3, 224, 224)
    # 进行前向传播
    output = model(input_tensor)
    print("Output shape:", output.shape)