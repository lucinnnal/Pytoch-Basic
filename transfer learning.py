# Training Flow?
"""
# Model & Optimizer
model = resnet18(weights =  ResNet18_Weights) # Load pretrained ResNet18 model & Weights
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9)

# Dummy data : x, label
data = torch.rand(1, 3, 64, 64)
labels = torch.rand((1,1000))

print(f"data shape : {data.shape}\n")
print(f"label shape : {labels.shape}\n")

# zero grad the gradient of the model params of loss
optimizer.zero_grad()

# Forward process
pred = model(data)

print(f"pred shape : {pred.shape}\n")

# Loss calculation
loss = (pred - labels).sum()

print(f"loss shape : {loss.shape}\n")

# Backprop -> calculate the model parameters' gradient of loss
loss.backward()

# Optimizer (Gradient descent update the model parameter through the gradients) P_new = P_new - learning_rate * P_old
optimizer.step()
"""

# frozen parameter? : 변화도를 계산하지 않는 매개변수(갱신시키지 않을 것임), 신경망의 일부를 freeze & fine-tuning
# freeze a model and only change a classifier layer

# Freeze RenNet model & Only change the classifier layer
import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
from torch import nn

# model
model = resnet18(weights = ResNet18_Weights)

# freeze all params -> grad 계산을 하지 않도록
for param in model.parameters():
    param.requires_grad = False

# re-define classifier layer
model.fc = nn.Linear(512, 10) # re define the fc layer by using nn.Linear -> make the fc layer to be learnable -> classify 10 images

# Optimizer -> fc layer 수정 시켜주고 optimizer에 넘겨 주기
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9)