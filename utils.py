import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import AdaGradNetwork

myNN = AdaGradNetwork.AdaGradNetwork()
test_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
