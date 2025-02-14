from torch_state_dict import *

net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# path for save model state
# ./ means current work directory
path = "./state_dict.pt"

# save model state dict
torch.save(net.state_dict(), path)

# Load state dict for "inference"
model = Net()
state_dict = torch.load(path)
model.load_state_dict(state_dict)
model.eval()