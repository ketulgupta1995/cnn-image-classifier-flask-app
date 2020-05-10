import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from AdaGradNetwork import AdaGradNetwork


def loadImages(is_download):
    test_set = torchvision.datasets.FashionMNIST(
        root='../data/FashionMNIST',
        train=False,
        download=is_download,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    return test_set


def create_nn():
    myNN = AdaGradNetwork()
    return  myNN