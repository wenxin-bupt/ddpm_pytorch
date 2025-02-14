import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载CIFAR10训练数据集
train_set = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# 下载CIFAR10测试数据集
test_set = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)