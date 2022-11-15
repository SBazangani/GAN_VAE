import torch
import numpy as np
import torchvision
from torch.autograd import Variable
from Model.main import MODEL
from Model.main import criteriond1_JSD
from Model.main import criteriond2_JSD
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import warnings
import wandb
import os
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

batch_size = 50
latent_dims = 800
T_dimes = latent_dims

m = 100 # number of block
l = 8 # size of each block
eigen_values = m * l
num_epochs = 100
lr = 1e-5
image_size = 49  # image size should be 20x20

latent_dime_sqr = int(np.sqrt(latent_dims))
latent_dims_T = int(latent_dims / 2)


dataset = torchvision.datasets.ImageFolder('C:/Users/farideh/Desktop/Rnadom field/VAE and RF/DATA',
                                           transforms.Compose([
                                               transforms.Grayscale(num_output_channels=1),
                                               transforms.Resize((image_size, image_size)),
                                               transforms.ToTensor(),


                                           ]))
print(len(dataset), " images loaded")

data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          )


latent_dims = 800
T_dimes = latent_dims

m = 100 # number of block
l = 8 # size of each block
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

# Model

gen, discrim, criteriond1, criteriond2, optimizerd1, optimizerd2 = MODEL()

# load pretrain model

#ckp_path_gen = "/home/bazanganif/Downloads/DCGAN_NEW/gan-result/model/gen.pth"
#gen, optimizerd2, start_epoch = load_ckp_GEN(ckp_path_gen, gen, optimizerd2)

#ckp_path_dis = "/home/bazanganif/Downloads/DCGAN_NEW/gan-result/model/dicrim.pth"
#discrim, optimizerd1, start_epoch = load_ckp_DIS(ckp_path_dis, discrim, optimizerd1)

# Architecture_Path_gen = "/home/bazanganif/Downloads/DCGAN_NEW/gan-result/model/gen.onnx"
# dummy_input = torch.randn(5,1,95,95,95,device ='cuda')
# noise = torch.randn((5,16,12,12),device=device)
# out_generator = gen(dummy_input,noise)
# torch.onnx.export(gen, args=(dummy_input,noise) ,f=Architecture_Path_gen,input_names=['PET_Image','Noise'],output_names=['MRI_Image'] )

# Architecture_Path_disc= "/home/bazanganif/Downloads/DCGAN_NEW/gan-result/model/discrim.onnx"
# dummy_input = torch.randn(5,1,95,95,95,device ='cuda')
# torch.onnx.export(discrim,dummy_input,Architecture_Path_disc)


image_size = 64

activations = {}
test = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()

    return hook


gc.disable()
wandb.init()
wandb.watch(gen)
wandb.watch(discrim)

fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(5, 5))
# Training
figx1, x1ss = plt.subplots(ncols=1)
figs, axs = plt.subplots(ncols=5, nrows=2)
figeig, axeigen = plt.subplots()
figcov, axcov = plt.subplots(ncols=5)
print("Training is started !")
for epoch in range(num_epochs):

    torch.cuda.empty_cache()
    gc.collect()

    discriminator_batch_loss = 0.0
    generator_batch_loss = 0.0

    batch_idx = 0

    torch.backends.cudnn.benchmark = True

    for batch_idx, (image_batch, _) in enumerate(data_loader):
        if batch_idx + 1 == len(data_loader):
            break

        # get samples from dataloader
        inputs = image_batch
        discrim.zero_grad()

        print("Discriminator training !")
        # 1.1 Train discriminator on real data
        dis_real_out = discrim(inputs)
        label_one = Variable(torch.ones(batch_size, 1, device=device))
        label_one_smoothed = Variable(label_one - 0.3 + (np.random.random(label_one.shape)*0.5)).float().to(device)
        dis_real_loss = criteriond1(dis_real_out, label_one_smoothed)  # given label to the discriminator
        dis_real_loss_JSD = criteriond1_JSD(dis_real_out, label_one)

        # 1.2 Train discriminator on fake data from generator
        dis_inp_fake_x, C = gen(inputs)
        dis_fake_out = discrim(dis_inp_fake_x)
        label_zero = Variable(torch.zeros(batch_size, 1, device=device))
        dis_fake_loss = criteriond1(dis_fake_out, label_zero)
        dis_fake_loss_JSD = criteriond1_JSD(dis_fake_out, label_zero)
        gc.collect()

        # 1.3 combine real loss and fake loss for discriminator
        discriminator_loss = 0.5 * dis_real_loss + 0.5 * dis_fake_loss + 0.5 * dis_real_loss_JSD + 0.5 * dis_fake_loss_JSD
        #discriminator_loss = - torch.mean(dis_real_out) + torch.mean(dis_fake_out)
        discriminator_batch_loss += discriminator_loss.item()
        discriminator_loss.backward()
        optimizerd1.step()
        gc.collect()

        # for p in discrim.parameters():
        #     p.data.clamp_(-0.01, 0.01)

        # 2.1 Training the generator
        print("generator training !")
        gen.zero_grad()
        gen_out, C = gen(inputs)
        dis_out_gen_training = discrim(gen_out)
        gen_loss = criteriond2(dis_out_gen_training, Variable(torch.ones(batch_size, 1, device=device)))
        gen_loss_JSD = criteriond2_JSD(dis_out_gen_training, Variable(torch.ones(batch_size, 1, device=device)))
        gen_loss = 0.5*gen_loss + 0.5*gen_loss_JSD
        #gen_loss = -torch.mean(dis_out_gen_training)
        generator_batch_loss += gen_loss.item()
        gen_loss.backward()
        optimizerd2.step()
        gc.collect()

        print("=======================================================================================================")

        print(f'Epoch [{epoch}/{epochs}]  Batch {batch_idx + 1}/{len(data_loader)} \
                    Loss D: {discriminator_loss:.4f}, Loss G: {gen_loss:.4f}')

        print("=======================================================================================================")

        wandb.log({"loss disc": discriminator_loss})
        wandb.log({"loss gen": gen_loss})

        print("Figures loading !")

        reconstructed_img_one = gen_out.detach().numpy()
        axs[0, 0].imshow(inputs[0, 0, :, :])
        axs[0, 0].grid(False)
        axs[0, 1].imshow(inputs[1, 0, :, :])
        axs[0, 1].grid(False)

        axs[0, 2].imshow(inputs[2, 0, :, :])
        axs[0, 2].grid(False)
        axs[0, 3].imshow(inputs[3, 0, :, :])
        axs[0, 3].grid(False)
        axs[0, 4].imshow(inputs[4, 0, :, :])
        axs[0, 4].grid(False)

        axs[1, 0].imshow(reconstructed_img_one[0,0, :, :])
        axs[1, 0].grid(False)
        axs[1, 1].imshow(reconstructed_img_one[1,0, :, :])
        axs[1, 1].grid(False)
        axs[1, 2].imshow(reconstructed_img_one[2,0, :, :])
        axs[1, 2].grid(False)
        axs[1, 3].imshow(reconstructed_img_one[3,0, :, :])
        axs[1, 3].grid(False)
        axs[1, 4].imshow(reconstructed_img_one[4,0, :, :])
        axs[1, 4].grid(False)
        plt.grid(b=None)
        figs.savefig('sample.png')
        wandb.log({"Image_recons": wandb.Image(figs)})
        plt.close()

        c = (C[:, :, :]).detach().numpy()
        axcov[0].matshow(c[0, :, :].reshape(T_dimes, T_dimes))
        axcov[0].axes.get_xaxis().set_visible(False)
        axcov[0].axes.get_yaxis().set_visible(False)
        axcov[0].grid(False)
        axcov[1].matshow(c[1, :, :].reshape(T_dimes, T_dimes))
        axcov[1].axes.get_xaxis().set_visible(False)
        axcov[1].axes.get_yaxis().set_visible(False)
        axcov[1].grid(False)
        axcov[2].matshow(c[2, :, :].reshape(T_dimes, T_dimes))
        axcov[2].axes.get_xaxis().set_visible(False)
        axcov[2].axes.get_yaxis().set_visible(False)
        axcov[2].grid(False)
        axcov[3].matshow(c[3, :, :].reshape(T_dimes, T_dimes))
        axcov[3].axes.get_xaxis().set_visible(False)
        axcov[3].axes.get_yaxis().set_visible(False)
        axcov[3].grid(False)
        axcov[4].matshow(c[4, :, :].reshape(T_dimes, T_dimes))
        axcov[4].axes.get_xaxis().set_visible(False)
        axcov[4].axes.get_yaxis().set_visible(False)
        axcov[4].grid(False)

        plt.grid(b=None)
        plt.close()

        figcov.savefig("COVARIANCE.png")
        wandb.log({"COVARIANCE": wandb.Image(figcov)})
        plt.close()

        torch.cuda.empty_cache()
        gc.collect()

        # save the model

        torch.save({
            'epoch': epoch,
            'gen_state_dict': gen.state_dict(),
            'optimizerd2_state_dict': optimizerd2.state_dict(),
            'loss_gen': gen_loss}, "gen.pth")

        torch.save({
            'epoch': epoch,
            'discrim_state_dict': discrim.state_dict(),
            'optimizerd1_state_dict': optimizerd1.state_dict(),
            'loss_dis': discriminator_loss}, "dicrim.pth")

