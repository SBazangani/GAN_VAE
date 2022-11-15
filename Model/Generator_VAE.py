import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import numpy as np
import math
from scipy.linalg import circulant
m = 100 # number of block
l = 8 # size of each block
eigen_values = m * l
batch_size = 50


class Generator_AE(nn.Module):
    def __init__(self, inp, out):
        super(Generator_AE, self).__init__()

        ########### Encoder ############
        self.conv1 = nn.Conv2d(1, 64, 3, 2, 1)  # out: 64 x 50 x 50
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)  # out: 64x2 x # 25 x 25
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)  # out: 64x4 x 12 x 12
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)  # out: 512 x 6 x 6
        self.fc_eigen = nn.Linear(512 * 4 * 4, eigen_values)
        ######### DECODER ###########

        self.linear_de = nn.Linear(l*m*l*m, 512 * 3 * 3)
        self.convT1 = nn.ConvTranspose2d(512, 128, 3, 2, 1)
        self.convT2 = nn.ConvTranspose2d(128,  64, 3, 2)
        self.convT3 = nn.ConvTranspose2d(64,  32, 3, 2)
        self.convT4 = nn.ConvTranspose2d(32, 16, 3, 2)
        self.convT5 = nn.ConvTranspose2d(16, 1, 3, 1)

    def forward(self, input):
        #print("input size:", input.shape)
        x = nn.LeakyReLU()(self.conv1(input))
        x = torch.nn.BatchNorm2d(64)(x)
        #print("Encoder1:", x.shape)

        x = nn.LeakyReLU()(self.conv2(x))
        x = torch.nn.BatchNorm2d(128)(x)
        # print("Encoder2:", x.shape)

        x = nn.LeakyReLU()(self.conv3(x))
        x = torch.nn.BatchNorm2d(256)(x)
        # print("Encoder3:", x.shape)

        x4 = nn.LeakyReLU()(self.conv4(x))
        x4 = torch.nn.BatchNorm2d(512)(x4)
        # print("Encoder4:", x4.shape)

        x = x4.view(x.size(0), -1)  # flatten batch of multichannel feature maps to a batch of feature vectors

        x_eigen = self.fc_eigen(x)
        x_eigen = nn.ReLU()(x_eigen)


        ########## Circulant Emb ############
        Sigma_matrix = x_eigen.reshape((batch_size, m, l))
        row_circulant = math.sqrt(m * l) * torch.fft.ifft2(Sigma_matrix)
        C = torch.zeros((batch_size, m * l, m * l))
        for i in range(batch_size):
            one_image = torch.reshape(row_circulant[i, :, :], (m * l, 1))
            Circulant = circulant(one_image.detach().numpy())
            C[i, :, :] = torch.tensor(Circulant, requires_grad=True)
        T = C[:, 0:400, 0:400]  # Toeplitz matrix
        ################### DECODER #######################
        x = nn.Flatten()(C)
        x = self.linear_de(x)
        x0 = x.view((batch_size, 512, 3, 3))

        x1 = (self.convT1(x0))  # print("decoder1:", x1.shape)

        x2 = (self.convT2(x1))  # print("decoder2:", x2.shape)

        x3 = (self.convT3(x2))  # print("decoder3:", x3.shape)

        x4 = (self.convT4(x3))

        x5 = (self.convT5(x4))

        return x5, C


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def Gen_AE():
    torch.cuda.is_available()

    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("Device", device)
    Generator = Generator_AE(1, 1).to(device)
    Generator.apply(weights_init)
    # summary(Generator, (1, 49, 49), batch_size=1)
    return Generator


if __name__ == '__main__':
    GEN = Gen_AE()













class Generator_VAE(nn.Module):
    def __init__(self, inp, out):
        super(Generator_VAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.main = nn.Sequential(

            nn.Conv3d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv3d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 1, 3, 1, 0, bias=False),
            nn.Flatten(),
            nn.Linear(125, 1),

            # nn.Sigmoid()
        )

    def forward(self, input):
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
    summary(discriminator, (1, 49, 49, 49), batch_size=1)
    return discriminator


if __name__ == '__main__':
    batch_size = 1
    DIS = Disc()
























