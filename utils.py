import AdaGradNetwork
import torch
import torchvision
from torchvision import transforms

myNN = AdaGradNetwork.AdaGradNetwork()
myNN.load_state_dict(torch.load("./static/FMNIST.pt"))
test_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

images = dict()

categories =['T-shirt/top',
                                                            'Trouser',
                                                            'Pullover',
                                                            'Dress',
                                                            'Coat',
                                                            'Sandal',
                                                            'Shirt',
                                                            'Sneaker',
                                                            'Bag',
                                                            'Ankle boot']