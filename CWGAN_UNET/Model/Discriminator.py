import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import numpy as np


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(

            nn.Conv2d(3, 64, 3, 2, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 3, 1, 0, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(512*25, 1),
            #nn.Sigmoid()


        )
        self.adapt_h = nn.Sequential(
            nn.Linear(256, 1024),


            #nn.LeakyReLU(),
        )
        self.Condition = nn.Sequential(
            nn.Linear(1024, 1),
            #nn.Sigmoid()


        )



    def forward(self, input1):

        OUT = self.main(input1)
        #h = input2.repeat(1,256)
        #h = self.adapt_h(h)
        #Out_h = torch.cat((OUT,h), dim=1)
        #Condi_OUT = self.Condition(OUT)
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
    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)
    #summary(discriminator, [(1, 49, 49), (1,0)], batch_size=2)
    return discriminator


if __name__ == '__main__':

    DIS = Disc()




