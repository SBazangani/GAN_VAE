
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import numpy as np
import math
from scipy.linalg import circulant
m = 100 # number of block
l = 8 # size of each block
eigen_values = m*l
batch_size = 80


class Generator_AE(nn.Module):
    def __init__(self, inp, out):
        super(Generator_AE, self).__init__()

        ########### Encoder ############

        self.conv1 = nn.Conv2d(1, 128, 3, 2, 1)  # out: 64 x 50 x 50
        self.Batch128 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 254, 3, 2, 1)  # out: 64x2 x # 25 x 25
        self.Batch256 = nn.BatchNorm2d(254)
        self.conv3 = nn.Conv2d(254, 512, 3, 2, 1)  # out: 64x4 x 12 x 12
        self.Batch512 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 3, 2, 1)  # out: 512 x 6 x 6
        self.Batch1024 = nn.BatchNorm2d(1024)
        self.fc_eigen = nn.Linear(1024 * 4 * 4, eigen_values)
        ######### DECODER ###########

        self.linear_de = nn.Linear(m*l*m*l, 150 * 3 * 3)
        self.convT1 = nn.ConvTranspose2d(150, 128, 3, 2, 1)
        self.convT2 = nn.ConvTranspose2d(128,  64, 3, 2)
        self.convT3 = nn.ConvTranspose2d(64,  32, 3, 2)
        self.convT4 = nn.ConvTranspose2d(32, 16, 3, 2)
        self.convT5 = nn.ConvTranspose2d(16, 1, 3, 1)
        self.Batch_norm_128 = nn.BatchNorm2d(128)
        self.Batch_norm_64 = nn.BatchNorm2d(64)
        self.Batch_norm_32 = nn.BatchNorm2d(32)
        self.Batch_norm_16 = nn.BatchNorm2d(16)



    def forward(self, input):
        #print("input size:", input.shape)
        x = nn.LeakyReLU()(self.conv1(input))
        x = self.Batch128(x)
        #print("Encoder1:", x.shape)

        x = nn.LeakyReLU()(self.conv2(x))
        x = self.Batch256(x)
        # print("Encoder2:", x.shape)

        x = nn.LeakyReLU()(self.conv3(x))
        x = self.Batch512(x)
        # print("Encoder3:", x.shape)

        x4 = nn.LeakyReLU()(self.conv4(x))
        x4 = self.Batch1024(x4)
        # print("Encoder4:", x4.shape)

        x = x4.view(x.size(0), -1)  # flatten batch of multichannel feature maps to a batch of feature vectors

        x_eigen = self.fc_eigen(x)
        x_eigen = nn.ReLU()(x_eigen)





        ########## Circulant Emb ############
        Sigma_matrix = x_eigen.reshape((batch_size, m, l))
        row_circulant = math.sqrt(m * l) * torch.fft.ifft2(Sigma_matrix)
        C = torch.zeros((batch_size,1, m * l, m * l)).to('cuda:0')
        #C =torch.rand(((batch_size,1, m * l, m * l))).to('cuda:0')
        for i in range(batch_size):
            one_image = torch.reshape(row_circulant[i, :, :], (m * l, 1))
            one_image_cpu = one_image.detach().cpu().numpy()
            Circulant = circulant(one_image_cpu)
            C[i,:, :, :] = torch.tensor(Circulant, requires_grad=True).to('cuda:0')
        #T = C[:, 0:400, 0:400].to('cuda:0') # Toeplitz matrix

        ################### DECODER #######################

        #y1 = self.x00(C)
        #y1 = nn.LeakyReLU()(y1)
        #y1 = self.Batch16(y1)
        #y2 = self.x11(y1)
        #y2 = nn.LeakyReLU()(y2)
        #y2 = self.Batch32(y2)
        #y3 = self.x22(y2)
        #y3 = nn.LeakyReLU()(y3)
        #y3 = self.Batch64(y3)
        #y4 = self.x33(y3)



        x = nn.Flatten()(C)
        x = self.linear_de(x)
        x = nn.LeakyReLU()(x)

        x0 = x.view((batch_size, 150, 3, 3))


        x1 = (self.convT1(x0))
        x1 = self.Batch_norm_128(x1)
        x1 = nn.LeakyReLU()(x1)


        #x11 = nn.functional.interpolate(x0,(6,6))
        #x11 = (self.x11(x11))
        #x11 = nn.LeakyReLU()(x11)



        x2 = (self.convT2(x1))
        x2 = nn.LeakyReLU()(x2)
        #x2 =self.Batch_norm_64(x2)
        #print("decoder2:", x2.shape)
        #x22 = nn.functional.interpolate(x11, (12, 12))
        #x22 = (self.x22(x22))

        x3 = (self.convT3(x2))
        x3 = nn.LeakyReLU()(x3)
        #x3 = self.Batch_norm_32(x3)
        #print("decoder3:", x3.shape)
        #x33 = nn.functional.interpolate(x22, (24, 24))
        #x33 = (self.x33(x33))

        x4 = (self.convT4(x3))
        x4 = nn.LeakyReLU()(x4)
        #x4 = self.Batch_norm_16(x4)
        #print("decoder4:", x4.shape)
        #x44 = nn.functional.interpolate(x33, (49, 49))
        #x44 = nn.Tanh()(self.x44(x44))

        x5 = (self.convT5(x4))
        x5 =nn.Tanh()(x5)

        #print("decoder5:", x5.shape)

        return x5, C


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
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
    #Generator.apply(weights_init)
    # summary(Generator, (1, 49, 49), batch_size=1)
    return Generator


if __name__ == '__main__':
    GEN = Gen_AE()








