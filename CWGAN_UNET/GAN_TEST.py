import torch
import numpy as np
import torchvision
from plotly import express
from torch.autograd import Variable
from Model.main import MODEL
from Model.main import criteriond1_JSD
from Model.main import criteriond2_JSD
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
from Model.preprocessing import data_loader
import warnings
import wandb
import os
from sklearn.manifold import TSNE
import gc
# reload a model
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


if not sys.warnoptions:
    warnings.simplefilter("ignore")

torch.cuda.empty_cache()
epochs = 2000
discriminator_loss_history = []

wess_dis_history = []

generator_loss_history = []

# dataloader
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

wandb.init()

#################################

batch_size = 80
latent_dims = 800
T_dimes = latent_dims

m = 8  # number of block
l = 100  # size of each block
eigen_values = m * l
num_epochs = 100
lr = 1e-5
image_size = 49  # image size should be 20x20

latent_dime_sqr = int(np.sqrt(latent_dims))
latent_dims_T = int(latent_dims / 2)
torch.cuda.is_available()
torch.cuda.empty_cache()
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)
print(device)


dataset = torchvision.datasets.ImageFolder('/home/bazanganif/Desktop/PhD/GAN_VAE/DATA',
                                           transforms.Compose([
                                               transforms.Grayscale(num_output_channels=1),
                                               #transforms.RandomCrop(20),
                                               transforms.Resize((image_size, image_size)),
                                               transforms.ToTensor(),
                                               #transforms.Normalize(mean=[0.1],std=[0.9]),

                                           ]))
print(len(dataset), " images loaded")

data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          )

latent_dims = 800
T_dimes = latent_dims

m = 100  # number of block
l = 8  # size of each block
eigen_values = m * l
num_epochs = 2000
lr = 1e-5
image_size = 49  # image size should be 20x20

latent_dime_sqr = int(np.sqrt(latent_dims))
latent_dims_T = int(latent_dims / 2)



# Model

gen, discrim, criteriond1, criteriond2, optimizerd1, optimizerd2 = MODEL()

# load pretrain model

ckp_path_gen = "/home/bazanganif/Desktop/PhD/GAN_VAE/gen.pth"
gen, optimizerd2, start_epoch = load_ckp_GEN(ckp_path_gen, gen, optimizerd2)

ckp_path_dis = "/home/bazanganif/Desktop/PhD/GAN_VAE/dicrim.pth"
discrim, optimizerd1, start_epoch = load_ckp_DIS(ckp_path_dis, discrim, optimizerd1)

# Architecture_Path_gen = "/home/bazanganif/Downloads/DCGAN_NEW/gan-result/model/gen.onnx"
# dummy_input = torch.randn(5,1,95,95,95,device ='cuda')
# noise = torch.randn((5,16,12,12),device=device)
# out_generator = gen(dummy_input,noise)
# torch.onnx.export(gen, args=(dummy_input,noise) ,f=Architecture_Path_gen,input_names=['PET_Image','Noise'],output_names=['MRI_Image'] )

# Architecture_Path_disc= "/home/bazanganif/Downloads/DCGAN_NEW/gan-result/model/discrim.onnx"
# dummy_input = torch.randn(5,1,95,95,95,device ='cuda')
# torch.onnx.export(discrim,dummy_input,Architecture_Path_disc)


image_size = 64

tsne = TSNE(n_components=2)

with torch.no_grad():
    noise = torch.normal(0, 1, (batch_size,3,49,49), device=device).detach()


    gen_image = gen(noise)
    #print(gen_image.shape)
    gen_image =gen_image.to("cpu")

    #grid_image = torchvision.utils.make_grid(gen_image,nrow=10)
    #plt.imshow(grid_image.permute(1, 2, 0),cmap='hot')
    #plt.grid(False)
    #plt.show()
    print("TSNE starts")
    tsne =TSNE(n_components=2)
    tsne_result = tsne.fit_transform(gen_image.flatten().reshape(-1,1))
    print(tsne_result.shape)



for batch_idx, (image_batch,_) in enumerate(data_loader):
    inputs = image_batch.flatten().numpy()
    tsne_result_originial = tsne.fit_transform(inputs.reshape(-1, 1))
    color_original = np.zeros(192080)
    color_fake = np.ones(192080)
    color = np.concatenate([color_fake,color_original])
    print(tsne_result_originial.shape)
    tsne_result = np.concatenate([tsne_result,tsne_result_originial])

    fig = express.scatter(tsne_result, x=0, y=1,color=color)
    fig.show()
