import torch
from torch import nn
from torchvision import dataset

# 一些重要概念：
# 1. tensor      Pytorch中的「变量」，接口几乎和numpy的ndarray一样。如果变量的requires_grad选项被打开，
#                他在运算过程中会自动计算出梯度，这样在反向传播的时候就很容易被更新了，这样的变量一般作为模型的权值。
# 2. dataloader  用来加载数据的类，支持多进程并发加载数据，在神经网络训练中，数据的处理经常会成为瓶颈，而数据的处理过程一般是在
#                CPU上进行的（只有正向传播和反向传播在GPU上进行），所以并发处理是很有必要的。
# 3. 模型        被训练的对象，他的权重就是一大堆requires_grad=True的变量，他还有两个比较重要的接口：
#                `forward`正向传播和`backward`反向传播，所有的模型都是`nn.Module`的子类，模型可以被嵌套
# 4. 优化器       用来更新模型的权值，模型并不会自己去更新权值，而是要定义优化器，并且把想要更新的权值传给优化器，
#                SGD或者Adam都是常见的优化器

# 使用 Pytorch 的一般步骤

# 第一步：建立模型
class SimpleNet(nn.Module):
  """一个「模型」是`nn.Module`的一个实例 ，通过让某个类继承`nn.Module`来创建一个模型的类，
  不要直接使用`nn.Module`来创建实例，因为这个类中没有创建任何「Layer」，应该始终使用
  `nn.Module`的子类。
  """

  def __init__(self):
    """模型的所有「参数」或「Layer」都需要在构造函数中「注册」，一旦某个参数或Layer被注册到了
    模型中，模型就会在反向传播的时候，就会自动计算「已注册」的变量梯度，
    """
    super().__init__() # 在注册任何参数前必须调用父类的构造方法

    # 注册实际上就是往 self 对象上添加 `nn.Module` 或 `nn.Parameter` 类型的变量
    self.param = nn.Parameter(torch.Tensor([5.0])) # 注册一个scalar
    self.conv = nn.Conv2d(10, 10, kernel_size=3) # 注册一个2D卷积层，这个卷积层实际上内部的实现也是一个`nn.Module`，这里实际就是模型的「嵌套」
    self.fc = nn.Linear(10, 10) # 注册一个全连接层

    # 下面的代码没有任何意义
    param = nn.Parameter(torch.Tensor([5.0]))
    conv = nn.Conv2d(10, 10, kernel_size=3)
    fc = nn.Linear(10, 10)

  def forward(self, x):
    # 需要自己定义正向传播过程
    x = x + 10
    x = x * 10
    x = x * self.param
    x = self.conv(x)
    x = self.fc(x)
    x = x * 100
    return x

# 实例化一个模型
my_net = SimpleNet()

# 第二步：创建 dataloader，Pytorch有一些开箱即用的 Dataset，比如 `datasets.MNIST`，主要用于实验，大部分时候需要自己创建 dataset
class SimpleDataset(torch.utils.data.Dataset):
  """所有的 dataset 类需要继承自`torch.utils.data.dataset`，一个 dataset 需要实现两个接口：
  1. __len__  需要返回总共的数据数量，比如有1000张训练图片，就需要返回1000
  2. __getitem__ 需要返回「一条」训练数据，一般来说是 (data, label) 这样的一个二元组
  """

  def __init__(self, image_paths, image_labels):
    """实现一个用于图片分类的 dataset，一般来说，image_paths 和 image_labels 是通过
    解析每个 annotation 文件（一般是一个CSV文件）得到的。
    """
    super().__init__();
    self.image_names = image_paths
    self.image_labels = image_labels

  def __len__(self):
    return len(self.image_names)

  def __getitem__(self, index):
    image = self.load_image(self.image_names[index])
    label = self.image_labels[0]

    image = self.preprocessing(image)
    return (image, label)

  def load_image(self, image_path):
    # TODO: load image by image_path
    # return cv2.imread(image_path)
    return image_path

  def preprocessing(self, image):
    # TODO: do image preprocessing
    return image

# 把dataset包成dataloader
image_paths = [
  '/tmp/1.png',
  '/tmp/2.png',
  # ...
]

image_label = [
  0,
  3,
  # ...
]

dataloader = torch.utils.data.DataLoader(
  SimpleDataset(image_paths, image_label),
  batch_size=256,
  shuffle=True
)

# 第三步：选择 Optimizer
optimizer = torch.optim.SGD(my_net.parameters(), lr=0.01)

# 第四步：选择损失函数
loss_function = torch.nn.CrossEntropyLoss()

# 第五步：开始训练
for (images, labels) in dataloader:
  # 迭代 dataloader，每次迭代会yield一个batch的数据

  results = my_net(images) # 正向传播训练数据（图片）
  loss = loss_function(images, labels) # 用正向传播获得的结果和标签计算Error/Loss

  # 接下来的散步几乎是固定的
  optimizer.zero_grad() # 先将记录在所有权值上梯度信息清零
  loss.backward() # 在 Loss 上调用 backward 来调用回归算法，会让所有的权值计算其梯度
  optimizer.step() # 使用某种优化算法，来让权值的梯度来更新权值的值，达到训练效果
