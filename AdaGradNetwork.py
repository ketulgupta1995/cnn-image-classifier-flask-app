# import standard PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# import torchvision module to handle image manipulation

# calculate train time, writing train data to files etc.

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


class AdaGradNetwork(nn.Module):
    def __init__(self):
        super(AdaGradNetwork, self).__init__()

        # define layers
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=1000)
        self.fc3 = nn.Linear(in_features=1000, out_features=500)
        self.fc4 = nn.Linear(in_features=500, out_features=200)
        self.out = nn.Linear(in_features=200, out_features=10)

    # define forward function
    def forward(self, t):
        # fc1  make input 1 dimentional
        t = t.view(-1, 28 * 28)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # fc3
        t = self.fc3(t)
        t = F.relu(t)

        # fc4
        t = self.fc4(t)
        t = F.relu(t)

        # output
        t = self.out(t)

        return t
