import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, inp, out):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(

            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 1, 3, 1, 0, bias=False),
            nn.Flatten(),
            nn.Linear(25, 1),

            nn.Sigmoid()
        )

    def forward(self, input):
        #print("input size :",input.shape)
        OUT = self.main(input)
        return OUT


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def Disc():
    torch.cuda.is_available()

    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("Device", device)
    discriminator = Discriminator(1, 1).to(device)
    discriminator.apply(weights_init)
    summary(discriminator, (1, 49, 49), batch_size=1)
    return discriminator


if __name__ == '__main__':

    DIS = Disc()
























