# FashionMNIST consists of PIL images, and integer type label
import torch
import torchvision
from torchvision import datasets, transforms

# Preprocess
x_transform = transforms.Compose([transforms.ToTensor()])
y_transform = transforms.Compose([transforms.Lambda(lambda y : torch.zeros((10,), dtype = torch.float).scatter_(dim=0, index = torch.tensor(y), value = 1))]) # One-hot encoding

# Load FashionMNIST
dataset = torchvision.datasets.FashionMNIST(
    root = './data',
    download = True,
    train = True,
    transform = x_transform,
    target_transform = y_transform
)

# Extract Sample 
img, label = dataset[0]
print(f"img shape : {img.shape}\n")
print(f"label shape : {label.shape}\n")

print(f"label : {label}\n")