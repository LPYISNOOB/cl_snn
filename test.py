# 在 Avalanche 中导入数据集
# from avalanche.benchmarks.datasets import MNIST, FashionMNIST, KMNIST, EMNIST, \
#     QMNIST, FakeData, CocoCaptions, CocoDetection, LSUN, ImageNet, CIFAR10, \
#     CIFAR100, STL10, SVHN, PhotoTour, SBU, Flickr8k, Flickr30k, VOCDetection, \
#     VOCSegmentation, Cityscapes, SBDataset, USPS, HMDB51, UCF101, CelebA, \
#     CORe50Dataset, TinyImagenet, CUB200, OpenLORIS, MiniImageNetDataset, \
#     Stream51, CLEARDataset

#导入经典基准classic
from avalanche.benchmarks.classic import CORe50, SplitTinyImageNet, SplitCIFAR10, \
    SplitCIFAR100, SplitCIFAR110, SplitMNIST, RotatedMNIST, PermutedMNIST, SplitCUB200

# 创建基准测试（场景对象）
perm_mnist = PermutedMNIST(
    n_experiences=3,  # 定义任务数量
    seed=1234,  # 设置随机种子以确保结果可复现
)

# 获取训练和测试数据流
train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream

# 遍历训练数据流
for experience in train_stream:
    print("任务开始 ", experience.task_label)
    print('当前任务包含的类别:', experience.classes_in_this_experience)

    # 通过experience对象可以方便地获取当前任务的训练集
    current_training_set = experience.dataset
    # ...同样也可以获取任务标签
    print('任务', experience.task_label)
    print('该任务包含', len(current_training_set), '个训练样本')

    # 我们可以通过在测试数据流中找到与当前任务相对应的体验来恢复对应的测试集
    current_test_set = test_stream[experience.current_experience].dataset
    print('该任务包含', len(current_test_set), '个测试样本')
#
#










#如果我们想创建一个 “Classic” 中不存在的新基准测试怎么办？在这种情况下，Avalanche 提供了许多实用程序，您可以使用它们以最大的灵活性创建自己的基准测试：基准测试生成器！

#当从一个或多个 PyTorch 数据集开始时，并且您想要创建“新实例”或“新类”基准测试时，特定的场景生成器非常有用：即它支持轻松灵活地创建域增量、类增量或任务增量场景等。

# from avalanche.benchmarks import nc_benchmark, ni_benchmark
# from torchvision.datasets import MNIST
# # import ssl
# #
# # ssl._create_default_https_context = ssl._create_unverified_context
#
# mnist_train = MNIST('.', train=True, download=True)
# mnist_test = MNIST('.', train=False)
#
#
# benchmark = ni_benchmark(
#     mnist_train, mnist_test, n_experiences=10, shuffle=True, seed=1234,
#     balance_experiences=True
# )
# benchmark = nc_benchmark(
#     mnist_train, mnist_test, n_experiences=10, shuffle=True, seed=1234,
#     task_labels=False
# )






#如果您想将您的策略与其他经典的持续学习算法或基线进行比较，在 Avalanche 中，这就像创建一个对象一样简单：
# from avalanche.models import SimpleMLP
# from avalanche.training import Naive
# from torch.optim import SGD
# from torch.nn import CrossEntropyLoss
#
# model = SimpleMLP(num_classes=10)
# cl_strategy = Naive(
#     model=model,  # 指定模型
#     optimizer=SGD(model.parameters(), lr=0.001, momentum=0.9),  # 指定优化器
#     criterion=CrossEntropyLoss(),  # 指定损失函数
#     train_mb_size=100,  # 训练时的mini-batch大小
#     train_epochs=4,  # 训练时的epoch数
#     eval_mb_size=100  # 评估时的mini-batch大小
# )


# #创建自己的策略
# from torch.nn import CrossEntropyLoss
# from torch.optim import SGD
# from torch.utils.data import DataLoader
#
# class MyStrategy():
#     """My Basic Strategy"""
#
#     def __init__(self, model, optimizer, criterion):
#         self.model = model
#         self.optimizer = optimizer
#         self.criterion = criterion
#
#     def train(self, experience):
#         # here you can implement your own training loop for each experience (i.e.
#         # batch or task).
#
#         train_dataset = experience.dataset
#         t = experience.task_label
#         train_data_loader = DataLoader(
#             train_dataset, num_workers=4, batch_size=128
#         )
#
#         for epoch in range(1):
#             for mb in train_data_loader:
#                 # you magin here...
#                 pass
#
#     def eval(self, experience):
#         # here you can implement your own eval loop for each experience (i.e.
#         # batch or task).
#
#         eval_dataset = experience.dataset
#         t = experience.task_label
#         eval_data_loader = DataLoader(
#             eval_dataset, num_workers=4, batch_size=128
#         )
#
#         # eval here
#
# from avalanche.models import SimpleMLP
# from avalanche.benchmarks import SplitMNIST
# # from avalanche.training import MyStrategy  # 假设MyStrategy是您自定义的策略类
# from torch.optim import SGD
# from torch.nn import CrossEntropyLoss
#
# # Benchmark creation
# benchmark = SplitMNIST(n_experiences=5)
#
# # Model Creation
# model = SimpleMLP(num_classes=benchmark.n_classes)
#
# # Create the Strategy Instance (MyStrategy)
# cl_strategy = MyStrategy(
#     model=model,
#     optimizer=SGD(model.parameters(), lr=0.001, momentum=0.9),
#     criterion=CrossEntropyLoss())
#
# # Training Loop
# print('Starting experiment...')
#
# # 确保整个训练循环在主模块中执行
# if __name__ == '__main__':
#     for exp_id, experience in enumerate(benchmark.train_stream):
#         print("Start of experience ", experience.current_experience)
#
#         cl_strategy.train(experience)
#         print('Training completed')
#
#         print('Computing accuracy on the current test set')
#         cl_strategy.eval(benchmark.test_stream[exp_id])



#使用我们的策略
# from avalanche.models import SimpleMLP
# from avalanche.benchmarks import SplitMNIST
# from multiprocessing import freeze_support
# # Benchmark creation
# benchmark = SplitMNIST(n_experiences=5)
#
# # Model Creation
# model = SimpleMLP(num_classes=benchmark.n_classes)
#
# # Create the Strategy Instance (MyStrategy)
# cl_strategy = MyStrategy(
#     model=model, optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9),
#     criterion = CrossEntropyLoss())
#
# # Training Loop
# print('Starting experiment...')
#
# for exp_id, experience in enumerate(benchmark.train_stream):
#     print("Start of experience ", experience.current_experience)
#
#     cl_strategy.train(experience)
#     print('Training completed')
#
#     print('Computing accuracy on the current test set')
#     cl_strategy.eval(benchmark.test_stream[exp_id])

# 导入必要的库和模块
# from torch.nn import CrossEntropyLoss
# from torch.optim import SGD
# from torch.utils.data import DataLoader
#
# # 创建自定义策略类
# class MyStrategy():
#     """My Basic Strategy
#     一个简单的策略类，用于定义训练和评估过程。
#     """
#
#     def __init__(self, model, optimizer, criterion):
#         self.model = model  # 模型
#         self.optimizer = optimizer  # 优化器
#         self.criterion = criterion  # 损失函数
#
#     def train(self, experience):
#         # 实现每个经验（即每个批次或任务）的训练循环。
#         train_dataset = experience.dataset  # 获取训练数据集
#         t = experience.task_label  # 获取当前任务标签
#         train_data_loader = DataLoader(
#             train_dataset, num_workers=4, batch_size=128
#         )  # 创建数据加载器
#
#         for epoch in range(1):  # 遍历每个epoch
#             for mb in train_data_loader:  # 遍历每个mini-batch
#                 # 在这里实现训练逻辑
#                 pass
#
#     def eval(self, experience):
#         # 实现每个经验（即每个批次或任务）的评估循环。
#         eval_dataset = experience.dataset  # 获取评估数据集
#         t = experience.task_label  # 获取当前任务标签
#         eval_data_loader = DataLoader(
#             eval_dataset, num_workers=4, batch_size=128
#         )  # 创建数据加载器
#
#         # 在这里实现评估逻辑
#
# # 导入Avalanche库中的模型和基准测试
# from avalanche.models import SimpleMLP
# from avalanche.benchmarks import SplitMNIST
#
# # 创建基准测试
# benchmark = SplitMNIST(n_experiences=5)  # 创建SplitMNIST基准测试，分为5个经验
#
# # 创建模型
# model = SimpleMLP(num_classes=benchmark.n_classes)  # 创建SimpleMLP模型，类别数与基准测试中的类别数相同
#
# # 创建策略实例
# cl_strategy = MyStrategy(
#     model=model,
#     optimizer=SGD(model.parameters(), lr=0.001, momentum=0.9),
#     criterion=CrossEntropyLoss()
# )  # 使用自定义策略类创建策略实例
#
# # 训练循环
# print('Starting experiment...')  # 打印开始实验的信息
#
# # 确保整个训练循环在主模块中执行
# if __name__ == '__main__':
#     for exp_id, experience in enumerate(benchmark.train_stream):
#         print("Start of experience ", experience.current_experience)  # 打印当前经验的开始
#
#         cl_strategy.train(experience)  # 执行训练
#         print('Training completed')  # 打印训练完成的信息
#
#         print('Computing accuracy on the current test set')  # 打印正在计算当前测试集的准确性
#         cl_strategy.eval(benchmark.test_stream[exp_id])  # 执行评估
#
# utility functions to create plugin metrics
# from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
# from avalanche.logging import InteractiveLogger, TensorboardLogger
# from avalanche.training.plugins import EvaluationPlugin
#
# eval_plugin = EvaluationPlugin(
#     # accuracy after each training epoch
#     # and after each evaluation experience
#     accuracy_metrics(epoch=True, experience=True),
#     # loss after each training minibatch and each
#     # evaluation stream
#     loss_metrics(minibatch=True, stream=True),
#     # catastrophic forgetting after each evaluation
#     # experience
#     forgetting_metrics(experience=True, stream=True),
#     # add as many metrics as you like
#     loggers=[InteractiveLogger(), TensorboardLogger()])

# pass the evaluation plugin instance to the strategy
# strategy = EWC(..., evaluator=eval_plugin)

# THAT'S IT!!
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics, gpu_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive

from torch.optim import SGD
from torch.nn import CrossEntropyLoss

benchmark = SplitMNIST(n_experiences=5)

# MODEL CREATION
model = SimpleMLP(num_classes=benchmark.n_classes)

# DEFINE THE EVALUATION PLUGIN and LOGGERS
# The evaluation plugin manages the metrics computation.
# It takes as argument a list of metrics, collectes their results and returns
# them to the strategy it is attached to.

# log to Tensorboard
tb_logger = TensorboardLogger()

# log to text file
text_logger = TextLogger(open('log.txt', 'a'))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True),
    cpu_usage_metrics(experience=True),
    forgetting_metrics(experience=True, stream=True),
    StreamConfusionMatrix(num_classes=benchmark.n_classes, save_image=False),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tb_logger]
)

# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = Naive(
    model, SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=500, train_epochs=1, eval_mb_size=100,
    evaluator=eval_plugin)

# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in benchmark.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience, num_workers=4)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    # eval also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(benchmark.test_stream, num_workers=4))