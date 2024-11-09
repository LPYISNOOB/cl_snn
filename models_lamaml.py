import torch.nn as nn
from avalanche.models.dynamic_modules import MultiTaskModule, MultiHeadClassifier


# 定义一个用于CIFAR-100数据集的卷积神经网络
class ConvCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvCIFAR, self).__init__()
        # 定义卷积层序列
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 160, kernel_size=(3, 3), stride=2, padding=1),  # 第一个卷积层，输入通道3，输出通道160
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),  # 第二个卷积层，输入通道160，输出通道160
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),  # 第三个卷积层，输入通道160，输出通道160
            nn.ReLU(inplace=True),  # 激活函数
        )
        # 定义线性层和激活函数
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(16*160, 320)  # 第一个全连接层
        self.linear2 = nn.Linear(320, 320)  # 第二个全连接层
        # 定义分类器
        self.classifier = nn.Linear(320, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)  # 通过卷积层
        x = x.view(-1, 2560)  # 展平特征图
        x = self.relu(self.linear1(x))  # 通过第一个全连接层和激活函数
        x = self.relu(self.linear2(x))  # 通过第二个全连接层和激活函数
        x = self.classifier(x)  # 通过分类器
        return x


# 定义一个用于CIFAR-100的多任务卷积神经网络
class MTConvCIFAR(ConvCIFAR, MultiTaskModule):
    def __init__(self):
        super(MTConvCIFAR, self).__init__()
        # 使用多任务分类器替换原有的分类器
        self.classifier = MultiHeadClassifier(320)

    def forward(self, x, task_labels):
        x = self.conv_layers(x)  # 通过卷积层
        x = x.view(-1, 16*160)  # 展平特征图
        x = self.relu(self.linear1(x))  # 通过第一个全连接层和激活函数
        x = self.relu(self.linear2(x))  # 通过第二个全连接层和激活函数
        x = self.classifier(x, task_labels)  # 通过多任务分类器
        return x


# 定义一个用于TinyImageNet数据集的卷积神经网络
class ConvTinyImageNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvTinyImageNet, self).__init__()
        # 定义卷积层序列
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 160, kernel_size=(3, 3), stride=2, padding=1),  # 第一个卷积层
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),  # 第二个卷积层
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),  # 第三个卷积层
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),  # 第四个卷积层
            nn.ReLU(inplace=True),  # 激活函数
        )
        # 定义线性层和激活函数
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(16*160, 640)  # 第一个全连接层
        self.linear2 = nn.Linear(640, 640)  # 第二个全连接层
        # 定义分类器
        self.classifier = nn.Linear(640, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)  # 通过卷积层
        x = x.view(-1, 16*160)  # 展平特征图
        x = self.relu(self.linear1(x))  # 通过第一个全连接层和激活函数
        x = self.relu(self.linear2(x))  # 通过第二个全连接层和激活函数
        x = self.classifier(x)  # 通过分类器
        return x


# 定义一个用于TinyImageNet的多任务卷积神经网络
class MTConvTinyImageNet(ConvTinyImageNet, MultiTaskModule):
    def __init__(self):
        super(MTConvTinyImageNet, self).__init__()
        # 使用多任务分类器替换原有的分类器
        self.classifier = MultiHeadClassifier(640)

    def forward(self, x, task_labels):
        x = self.conv_layers(x)  # 通过卷积层
        x = x.view(-1, 16*160)  # 展平特征图
        x = self.relu(self.linear1(x))  # 通过第一个全连接层和激活函数
        x = self.relu(self.linear2(x))  # 通过第二个全连接层和激活函数
        x = self.classifier(x, task_labels)  # 通过多任务分类器
        return x