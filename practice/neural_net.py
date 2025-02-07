# Neural Net can be made by using torch.nn Package
import torch
from torch import nn
import torch.nn.functional as F

# Define neural net
class Neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutions
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Maxpool
        self.subsampling = nn.MaxPool2d(2)

        # Activation function
        self.relu = nn.ReLU()

        # Fc layers
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.subsampling(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.subsampling(x)
        x = torch.flatten(x, 1) # torch.flatten(x, start_dim = 1, end_dim = -1) -> flatten from the 1st dimension (N, C*H*W)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)

        return out
    
# Model instance
model = Neuralnet()
print("Model Structure\n")
print(model) # You can see model structure
print()

# Model parameters
params = list(model.parameters())
print(len(params))
"""
for i in range(len(params)): print(f"{i}-th param shape: {params[i].shape}")
"""

# Input -> random dummy data
shape = (1, 1, 32, 32)
data = torch.randn(shape)
print(f"input shape : {data.shape}\n")
pred = model(data)
print(f"output shape : {pred.shape}\n")

# Loss function -> usually gets output - target pair : eg. F.mse_loss
"""
input = torch.randn((1, 1, 32, 32))
output = model(input) # (N, 10) -> (1, 10)
target = torch.randn(10)
target = target.view(1, -1) # change shape to (1, 10)
criterion = nn.MSELoss()

loss = criterion(output, target) # or loss = torch.nn.functional.mse_loss(output, target)
print(loss)
"""