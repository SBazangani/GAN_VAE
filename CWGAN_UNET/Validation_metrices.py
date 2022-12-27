import torch
import numpy as np
import os
from Model.preprocessing import data_loader
import math
from skimage.metrics import structural_similarity
from torchvision import transforms
from Model.main import MODEL
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
import os
import sys

import math
#from skimage.measure import compare_ssim
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
## read the np arrays result
import nibabel as nib
from skimage import io, transform


# read the test data from the dataloader
batch_size =30
data_loader = data_loader(batch_size)
print(len(data_loader.dataset),"Samples")

batch_size = 100
num_epochs = 2000

image_size = 49  # image size should be 20x20
torch.cuda.is_available()
torch.cuda.empty_cache()
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)
print(device)

def load_ckp_GEN(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['gen_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizerd2_state_dict'])

    return model, optimizer, checkpoint['epoch']


# reload a model
def load_ckp_DIS(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['discrim_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizerd1_state_dict'])

    return model, optimizer, checkpoint['epoch']


# Model

gen, discrim, criteriond1, criteriond2, optimizerd1, optimizerd2 = MODEL()

# load pretrain model

ckp_path_gen = "/home/bazanganif/Desktop/PhD/GAN_VAE/gen.pth"
gen, optimizerd2, start_epoch = load_ckp_GEN(ckp_path_gen, gen, optimizerd2)

ckp_path_dis = "/home/bazanganif/Desktop/PhD/GAN_VAE/dicrim.pth"
discrim, optimizerd1, start_epoch = load_ckp_DIS(ckp_path_dis, discrim, optimizerd1)


# compute the PSNR

# 1. compute the maximum intensity for the generated images


# read the test dataset
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)

# compute the PNSR for each image and then return the mean o=value over the batch
PNSR_Array = []
SSIM_Array = []
MAE_Array = []

for batch_idx, image_batch in enumerate(data_loader):
    if batch_idx + 1 == len(data_loader):
        break
    inputs = image_batch['image'].float().to(device).detach()


    h1 = image_batch['H1'].float().to(device).detach()
    h2 = image_batch['H2'].float().to(device).detach()
    H = torch.cat((h1, h1), dim=1)

    tensor_H = torch.unsqueeze(torch.unsqueeze(H, -1), -1).repeat(1, 1, 49, 49)
    conditional_input = torch.cat([inputs, tensor_H], dim=1)
    noise = torch.normal(0, 1, (inputs.shape), device=device).detach()
    noise_conditional = torch.cat((noise, tensor_H), dim=1)
    generated_images = gen(noise_conditional)

    # compute max intensity of the generated images
    max_intens = torch.max(generated_images)
    print(generated_images.shape)
    # compute the Mean square error of between the generated sample and the real sample
    real_sample = inputs

    fig, axs = plt.subplots(ncols=2, figsize=(5, 5))
    img_array_one = np.array(generated_images[15, 0, :, :].detach().cpu().numpy())
    axs[0].imshow((img_array_one).reshape(49, 49))
    img_array_two = np.array(real_sample[1, 0, :, :].detach().cpu().numpy()) # red patch in upper left
    axs[1].imshow((img_array_two).reshape(49, 49))

    #file_name = '/home/bazanganif/Downloads/P_GAN/gan-result/VAl' + str(batch_idx)
    #plt.savefig('/home/bazanganif/Downloads/P_GAN/gan-result/' + ".png", dpi=200)

    plt.show()









    diff = np.square(real_sample.detach().cpu().numpy() - generated_images.detach().cpu().numpy())
    MSE = diff.mean()
    print(MSE)
    sqr_Max = math.sqrt(max_intens)
    print(sqr_Max)
    # compute the PSNR
    PNSR = 20 * math.log10(sqr_Max / MSE)
    print(PNSR)

    PNSR_Array.append(PNSR)

    # SSIM

    real_sample = torch.squeeze(real_sample, 1)
    real_sample = torch.squeeze(real_sample, 0).detach().cpu().numpy()

    generated_images = torch.squeeze(generated_images, 1)
    generated_images = torch.squeeze(generated_images, 0).detach().cpu().numpy()

    (SSIM_score, SSIM_diff) = structural_similarity(real_sample, generated_images, full=True)
    SSIM_Array.append(SSIM_score)

    # MAE
    MAE = (np.sum(np.absolute((real_sample.astype("float"), generated_images.astype("float"))))) / (95 * 95 * 95)
    MAE_Array.append(MAE)

PSNR_rate = np.sum(PNSR_Array) / len(PNSR_Array)
SSIM_rate = np.sum(SSIM_Array) / len(SSIM_Array)
MAE_rate = np.sum(MAE_Array) / len(MAE_Array)
print("PNSR: ", max(PNSR_Array))
print("SSIM: ", max(SSIM_Array))
print("MAE: ", MAE_rate)

