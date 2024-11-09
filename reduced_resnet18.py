import torch
from avalanche.models import MultiHeadClassifier, MultiTaskModule
from torch import nn, relu
from torch.nn.functional import avg_pool2d

# 注释说明了代码的起始部分是从GEM（Gradient Episodic Memory）项目代码中来的，
# 并且将分类器替换成了Avalanche的多头分类器。
"""
START: FROM GEM CODE https://github.com/facebookresearch/GradientEpisodicMemory/
CLASSIFIER REMOVED AND SUBSTITUTED WITH AVALANCHE MULTI-HEAD CLASSIFIER
"""


# 定义一个3x3卷积层，步长为1，带偏置项
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


# 定义ResNet的基本块
class BasicBlock(nn.Module):
    expansion = 1  # 基本块的扩展系数

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # 创建卷积层和批归一化层
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # 创建快捷连接，用于匹配输入和输出的维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        # 应用卷积、批归一化和ReLU激活函数
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 添加快捷连接并应用ReLU激活函数
        out += self.shortcut(x)
        out = relu(out)
        return out


# 定义ResNet网络
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf  # 输入通道数

        # 创建初始卷积层和批归一化层
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        # 创建四个层，每个层由多个基本块组成
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)  # 获取批次大小
        # 应用初始卷积层、批归一化和四个层
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 应用平均池化层
        out = avg_pool2d(out, 4)
        return out


"""
END: FROM GEM CODE
"""

# 定义一个减少通道数的ResNet18模型，用于多任务学习
class MultiHeadReducedResNet18(MultiTaskModule):
    """
    根据GEM论文，这是一个更小的ResNet18版本，所有层中的特征图数量减少了三倍。
    它使用了多头输出层。
    """

    def __init__(self, size_before_classifier=160):
        super().__init__()
        # 初始化ResNet结构，使用BasicBlock，四个层，每个层有两个基本块
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], 20)
        # 初始化分类器为多头分类器
        self.classifier = MultiHeadClassifier(size_before_classifier)

    def forward(self, x, task_labels):
        # 通过ResNet结构，然后展平特征，最后通过多头分类器
        out = self.resnet(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out, task_labels)


# 定义一个单头版本的减少通道数的ResNet18模型
class SingleHeadReducedResNet18(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 初始化ResNet结构
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], 20)
        # 初始化分类器为线性层
        self.classifier = nn.Linear(160, num_classes)

    def feature_extractor(self, x):
        # 通过ResNet结构提取特征
        out = self.resnet(x)
        return out.view(out.size(0), -1)

    def forward(self, x):
        # 通过特征提取器，然后应用分类器
        out = self.feature_extractor(x)
        return self.classifier(out)


# 导出这两个类以便在其他地方使用
__all__ = ['MultiHeadReducedResNet18', 'SingleHeadReducedResNet18']