# Dataset -> Stores sample and label
# DataLoader -> DataLoader 는 Dataset 을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 감쌉니다.
# Fashion - MNIST
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Preprocess -> torchvision.transforms
transform = torchvision.transforms.Compose([
    transforms.ToTensor()
]
)

# Load FashionMNIST dataset

# Train
train_dataset = torchvision.datasets.FashionMNIST(
    root = './data',
    download = True,
    train = True,
    transform = transform
)

# Test
test_dataset = torchvision.datasets.FashionMNIST(
    root = './data',
    train = False,
    download = True,
    transform = transform
)

# Dataloader
train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle = True)

# Extract first batch
images, labels = next(iter(train_dataloader))

print(f"first batch imgs shape: {images.shape}\n") # (N, C, H, W)
print(f"first batch labels shape: {labels.shape}\n") # (N, )