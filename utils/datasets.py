# Libraries import  (From utils folder)
from utils.import_lib import *

# Image Preprocess
img_preprocess = torchvision.transforms.Compose([transforms.ToTensor()])

# Dataset & Dataloader
def get_train_dataset():

    return torchvision.datasets.FashionMNIST(
        root = "/Users/kipyokim/Desktop/Pytorch Basic/data/train",
        train = True,
        transform = img_preprocess,
        download = True,
        )

def get_test_dataset():
    return torchvision.datasets.FashionMNIST(
        root = '/Users/kipyokim/Desktop/Pytorch Basic/data/test',
        download = True,
        transform = img_preprocess,
        train = False
        )

def get_train_dataloader(): return DataLoader(get_train_dataset(), batch_size = 64, shuffle = True)

def get_test_dataloader(): return DataLoader(get_test_dataset(), batch_size = 64, shuffle = False)