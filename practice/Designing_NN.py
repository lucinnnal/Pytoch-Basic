import os
import torch
import torchvision
from torchvision import transforms, datasets
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Seed fix
torch.manual_seed(1)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device is {device}\n")

# Define NeuralNet Class
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x) # (N, 28 * 28)
        logits = self.layers(x) # (N, 10)

        return logits

# Create NeuralNet instance
model = NeuralNet()
# move model to device
model.to(device)
# Check structure
print(f"{model}\n")

# Dummy data
shape = (3, 28, 28)
x = torch.rand(shape, device = device) # (3,28,28)
logits = model(x) # (3, 10)
print(f"shape of x: {x.shape}\n")
print(f"shape of output: {logits.shape}\n")

# classification? -> softmax
softmax = nn.Softmax(dim = 1) # nn.softmax 함수 인스턴스 생성
prob = softmax(logits) # (3, 10)
predicted_classes = torch.argmax(prob, dim = 1)  # (3,) -> prob.argmax(dim = 1)
print(f"predicted class: {predicted_classes}\n")