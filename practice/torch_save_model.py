from torch_state_dict import *

# path to save the entire model -> "NOT RECOMMENDED JUST USE STATE_DICT SAVE!"
path = "./model.pt"

# Save entire model
net = Net()
torch.save(net, path)

# Load entire model & Inference -> model.eval()
model = torch.load(path)
model.eval()