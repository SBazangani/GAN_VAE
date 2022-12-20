from torchsummary import summary
import torch.nn  as nn
import torch

def conv(input_chanel,output_chanel):
    return nn.Conv2d(input_chanel,output_chanel,kernel_size=(5, 5),stride=2,padding=0)

def upsampling(input_channel,output_channel):
    out = nn.Sequential(
    nn.Upsample(scale_factor=2,mode='nearest'),
        nn.Conv2d(input_channel,output_channel, 3, 1, 1)
    )
    return out

def Pixel_shuffle(input_channel,output_channel):
    out = nn.Sequential(
        nn.PixelShuffle(upscale_factor=2),
        nn.Conv2d(int(input_channel/4), output_channel, 3, 1, 1),
    )
    return out





# class U-Net

def dual_conv(input_channel,output_channel):
    conv = nn.Sequential(
        nn.Conv2d(input_channel, output_channel,kernel_size=3,stride=1,padding=1),
        #nn.BatchNorm2d(output_channel),
        nn.LeakyReLU(inplace=True),

        nn.Conv2d(output_channel,output_channel,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(output_channel),
        nn.LeakyReLU(inplace=True),
    )
    return conv

def crop_tensor(target_tensor, tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


def out_conv_gen(input_channel,output_channel):
    out_conv = nn.Sequential(
        nn.Conv2d(input_channel, output_channel,kernel_size=2,stride=1,padding=1),
        nn.Tanh()
    )
    return out_conv





class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()

        self.dwn_conv1 = dual_conv(3,64)
        self.dwn_conv2 = dual_conv(64,128)
        self.dwn_conv3 = dual_conv(128,256)
        self.dwn_conv4 = dual_conv(256,512)
        self.dwn_conv5 = dual_conv(512,1024)
        self.maxpool = nn.MaxPool2d (kernel_size=2,stride =2)

        # right side
        self.Upsample1 = upsampling(1024,512)
        self.Pixelsh1 = Pixel_shuffle(1024,512)
        #self.trans1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.up_conv1 = dual_conv(1024,512)

        #self.trans2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.Upsample2 = upsampling(512,256)
        self.Pixelsh2 = Pixel_shuffle(512, 256)
        self.up_conv2 = dual_conv(512,256)

        #self.trans3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride =2)
        self.Upsample3 = upsampling(256,128)
        self.Pixelsh3 = Pixel_shuffle(256, 128)
        self.up_conv3 = dual_conv(256,128)

        #self.trans4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.Upsample4 = upsampling(128,64)
        self.Pixelsh4 = Pixel_shuffle(128, 64)
        self.up_conv4 =dual_conv(128,64)


        self.out = out_conv_gen(64,1)


    def forward(self,image):
            # left side
            x1 = self.dwn_conv1(image) #dual cov 3x3x3 witout padding and stride
            #print('shape of x1',x1.shape)
            x2 = self.maxpool(x1)  # Maxpooling with kernel 2x2x2 and stride 2
            #print('shape of x2', x2.shape)
            x3 = self.dwn_conv2(x2)
            #print('shape of x3', x3.shape)
            x4 = self.maxpool(x3)
            #print('shape of x4', x4.shape)
            x5 = self.dwn_conv3(x4)
            #print('shape of x5', x5.shape)
            x6 = self.maxpool(x5)
            #print('shape of x6', x6.shape)
            x7 = self.dwn_conv4(x6)
            #print('shape of x7', x7.shape)
            x8 = self.maxpool(x7)
            #print('shape of x8', x8.shape)
            x9 = self.dwn_conv5(x8)
            #print('shape of x9', x9.shape)
            # right side
        # forward pass for Right side
            #print(x9.shape)
            #x = self.trans1(x9)
           # x = self.Upsample1(x9)
            x = self.Pixelsh1(x9)

            y = crop_tensor(x, x7)
            #print('shape of x11', y.shape)
            #print(torch.cat([x, y], 1).shape)
            x = self.up_conv1(torch.cat([x, y], 1))
            #print('x12', x.shape)


            #x = self.Upsample2(x)
            x = self.Pixelsh2(x)
            #print('shape of x13', x.shape)
            y = crop_tensor(x, x5)
            #print('shape of crop tensor', y.shape)
            x = self.up_conv2(torch.cat([x, y], 1))
            #print('shape of x14', x.shape)


            x = self.Pixelsh3(x)
            #x = self.Upsample3(x)
            #print('shape of x15', x.shape)
            y = crop_tensor(x, x3)
            #print('crop_tensor y', y.shape)
            x = self.up_conv3(torch.cat([x, y], 1))
            #print('shape of x', x.shape)


            #x = self.Upsample4(x)
            x = self.Pixelsh4(x)
            #print('shape of x', x.shape)
            y = crop_tensor(x, x1)
            #print('shape of y', y.shape)

            x = self.up_conv4(torch.cat([x, y[:, :, 0:48, 0:48]], 1))
            #print('shape of x', x.shape)

            x = self.out(x)

            return x

def init_weight(m):
    if type(m) == nn.Conv3d:
        nn.init.normal_(m.weight,mean=0.0,std=0.02)
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,mean=0.0,std=0.02)
    if type(m) == nn.BatchNorm3d:
        nn.init.normal_(m.weight,mean=0.0,std=0.02)

def GEN_UNET():
    torch.cuda.is_available()
    if torch.cuda.is_available():dev = "cuda"
    else:dev = "cpu"
    device = torch.device(dev)
    print("Device", device)
    GEN = Unet().to(device)
    GEN.apply(init_weight)
    print(summary(GEN, (3, 49, 49), batch_size=10))
    return GEN

if __name__ == '__main__':
    batch_size = 1
    gen = GEN_UNET()