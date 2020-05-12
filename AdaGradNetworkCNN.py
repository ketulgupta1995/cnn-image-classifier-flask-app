# import standard PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaGradNetwork(nn.Module):
    def __init__(self):
        super(AdaGradNetwork, self).__init__()

        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        # output = 10*(28-5+1)*(28-5+1) = 10*24*24
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        #     10*13*13
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        #     10*20*(12-3+1)*(12-3+1)=64*10*10
        #    64 *20*20
        self.bacthNorm32 = nn.BatchNorm2d(32)
        self.bacthNorm64 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(in_features=12544, out_features=128)

        self.out = nn.Linear(in_features=128, out_features=10)

    # define forward function
    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        #     t = self.bacthNorm32(t)
        #     t = F.dropout2d(t,p=0.25)
        #     t = self.maxpool(t)

        t = self.conv2(t)
        t = F.relu(t)
        #     t = self.bacthNorm64(t)
        #     t = F.dropout2d(t,p=0.40)
        t = self.maxpool(t)
        t = F.dropout2d(t, p=0.40)

        t = torch.flatten(t, 1)
        #     t = t.view(-1,20*10*10)
        t = self.fc1(t)
        t = F.relu(t)
        #     t = F.dropout2d(t,p=0.40)

        # output
        t = self.out(t)

        return t