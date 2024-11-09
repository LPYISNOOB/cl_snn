import torch
from torch import nn
import torchvision
from avalanche.models import MultiTaskModule, MultiHeadClassifier


# 定义一个使用VGG网络作为特征提取器，并采用多头分类器的多任务模块
class MultiHeadVGG(MultiTaskModule):
    def __init__(self, n_classes=20):
        super().__init__()
        # 加载预训练的VGG11模型作为特征提取器
        self.vgg = torchvision.models.vgg11()
        # 初始化多头分类器，输入特征维度为VGG模型的1000个输出特征
        self.classifier = MultiHeadClassifier(in_features=1000, initial_out_features=n_classes)

    def forward(self, x, task_labels):
        # 通过VGG网络提取特征
        x = self.vgg(x)
        # 展平特征图以输入到分类器
        x = torch.flatten(x, 1)
        # 通过多头分类器进行分类
        return self.classifier(x, task_labels)


# 以下是Small VGG网络配置，它根据提供的配置创建VGG特征提取器，并自定义分类器
cfg = [64, 'M', 64, 'M', 64, 64, 'M', 128, 128, 'M']  # VGG配置
conv_kernel_size = 3  # 卷积核大小
img_input_channels = 3  # 输入图像的通道数


# 自定义VGG模型类
class VGGSmall(torchvision.models.VGG):
    """
    根据提供的配置创建VGG特征提取器，并自定义分类器。
    """

    def __init__(self):
        in_channels = img_input_channels
        layers = []
        # 根据配置构建VGG网络层
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=conv_kernel_size, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        super(VGGSmall, self).__init__(nn.Sequential(*layers), init_weights=True)

        # 如果存在avgpool层，则将其替换为Identity层以兼容PyTorch新版本
        if hasattr(self, 'avgpool'):
            self.avgpool = torch.nn.Identity()

        # 删除预定义的分类器
        del self.classifier

    def forward(self, x):
        # 通过VGG网络提取特征
        x = self.features(x)
        return x


# 定义一个使用自定义VGGSmall网络作为特征提取器，并采用多头分类器的多任务模块
class MultiHeadVGGSmall(MultiTaskModule):
    def __init__(self, n_classes=200, hidden_size=128):
        super().__init__()
        # 初始化自定义的VGGSmall网络
        self.vgg = VGGSmall()
        # 定义一个前馈神经网络，用于在分类之前进一步处理特征
        self.feedforward = nn.Sequential(
            nn.Linear(128*4*4, hidden_size),  # 第一个全连接层
            nn.ReLU(True),  # 激活函数
            nn.Linear(hidden_size, hidden_size),  # 第二个全连接层
            nn.ReLU(True),  # 激活函数
        )
        # 初始化多头分类器
        self.classifier = MultiHeadClassifier(in_features=128, initial_out_features=n_classes)

    def forward(self, x, task_labels):
        # 通过VGGSmall网络提取特征
        x = self.vgg(x)
        # 展平特征图以输入到分类器
        x = torch.flatten(x, 1)
        # 通过前馈神经网络进一步处理特征
        x = self.feedforward(x)
        # 通过多头分类器进行分类
        return self.classifier(x, task_labels)


# 定义一个使用自定义VGGSmall网络作为特征提取器，并采用单头分类器的单任务模块
class SingleHeadVGGSmall(nn.Module):
    def __init__(self, n_classes=200, hidden_size=128):
        super().__init__()
        # 初始化自定义的VGGSmall网络
        self.vgg = VGGSmall()
        # 定义一个前馈神经网络，用于在分类之前进一步处理特征
        self.feedforward = nn.Sequential(
            nn.Linear(128 * 4 * 4, hidden_size),  # 第一个全连接层
            nn.ReLU(True),  # 激活函数
            nn.Linear(hidden_size, hidden_size),  # 第二个全连接层
            nn.ReLU(True),  # 激活函数
        )
        # 初始化单头分类器
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # 通过VGGSmall网络提取特征
        x = self.vgg(x)
        # 展平特征图以输入到分类器
        x = torch.flatten(x, 1)
        # 通过前馈神经网络进一步处理特征
        x = self.feedforward(x)
        # 通过单头分类器进行分类
        return self.classifier(x)