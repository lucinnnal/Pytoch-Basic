import sys
import os

# Module 불러오기 위한 절대경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
dir = os.path.abspath(os.path.join(current_dir, ".."))

# 모듈 탐색 경로에 추가
sys.path.append(dir)

from utils.import_lib import *

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9)

    print(f"model's statedict:")
    # Check parameter type and size -> model.state_dict() contains all the parameter informations (so you can save the model params or load model params)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print()

    print("Optimizer's state_dict:")
    # Check optimizer state
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])