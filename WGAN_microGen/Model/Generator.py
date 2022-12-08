import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary


# pixel-wise feature normalization
class pixel_wise_Normalization(nn.Module):
    def __init__(self, eps=1e-8):
        super(pixel_wise_Normalization, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

    def __repr(self):
        return self.__class__.__name__ + '(eps = %s)' % (self.eps)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        ngf = 1
        self.features = nn.Sequential(

            nn.Conv2d(1, 256, 3, 1, 1, bias=False),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(True),

            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(True),

            nn.Conv2d(512, 1024, 3, 2, 1, bias=False),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(True),
        ) # 13 x13
        self.h = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )


        self.gen = nn.Sequential (

            nn.ConvTranspose2d(128+1024, 512, 3, 2, 1, bias=False),
            #nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 64, 3, 2, 1, bias=False),
            #nn.BatchNorm2d(32),
            #nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 3, 1, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, input1,input2):
        print(input2.shape)
        out1 = self.features(input1)

        h = torch.unsqueeze(input2, -1)
        #h = torch.unsqueeze(h,-1)
        h = torch.unsqueeze(h,-1).repeat(1, 1, 49, 49)
        h_f = self.h(h)
        all_features = torch.cat((out1,h_f),dim=1)
        rec_image = self.gen(all_features)
        print(rec_image.shape)

        return rec_image


def GEN():
    torch.cuda.is_available()

    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("Device", device)
    GEN = Generator().to(device)
    #GEN.apply(weights_init)
    #summary(GEN, [(1, 49, 49), ()], batch_size=1)
    return GEN


if __name__ == '__main__':
    batch_size = 1

    gen = GEN()
