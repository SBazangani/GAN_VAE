import torch

from torch.autograd import Variable
from Model.main import MODEL
from Model.main import criteriond1_JSD
from Model.main import criteriond2_JSD

import matplotlib.pyplot as plt
import sys
from Model.preprocessing import data_loader
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

batch_size = 399
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




# Model

gen, discrim, criteriond1, criteriond2, optimizerd1, optimizerd2 = MODEL()

# load pretrain model

ckp_path_gen = "/home/bazanganif/Desktop/PhD/GAN_VAE/gen.pth"
#gen, optimizerd2, start_epoch = load_ckp_GEN(ckp_path_gen, gen, optimizerd2)

ckp_path_dis = "/home/bazanganif/Desktop/PhD/GAN_VAE/dicrim.pth"
#discrim, optimizerd1, start_epoch = load_ckp_DIS(ckp_path_dis, discrim, optimizerd1)
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()

    return hook


gc.disable()
wandb.init()
wandb.watch(gen)
wandb.watch(discrim)

figs, axs = plt.subplots(ncols=5, nrows=2, figsize=(5, 5))
# Training
figx1, x1ss = plt.subplots(ncols=1)

print("Training is started !")
torch.cuda.empty_cache()
data_loader = data_loader(batch_size)
print(len(data_loader.dataset),"Samples")

for epoch in range(num_epochs):

    torch.cuda.empty_cache()
    gc.collect()

    discriminator_batch_loss = 0.0
    generator_batch_loss = 0.0

    batch_idx = 0

    torch.backends.cudnn.benchmark = True

    for batch_idx, image_batch in enumerate(data_loader):
        if batch_idx + 1 == len(data_loader):
            break

        inputs = image_batch['image'].float().to(device).detach()
        h1 = image_batch['H1'].float().to(device).detach()
        h2 = image_batch['H2'].float().to(device).detach()
        H = torch.cat((h1, h1),dim=1)
        tensor_H = torch.unsqueeze(torch.unsqueeze(H, -1), -1).repeat(1, 1, 49, 49)
        conditional_input = torch.cat([inputs, tensor_H], dim=1)


        discrim.zero_grad()

        print("Discriminator training !")
        # 1.1 Train discriminator on real data

        dis_real_out = discrim(conditional_input)
        label_one = Variable(torch.ones(batch_size, 1, device=device)).detach()
        label_one_smoothed = Variable(label_one - 0.3 + (torch.rand(label_one.shape) * 0.5).to(device)).float().to(device).detach()
        dis_real_loss = criteriond1(label_one, dis_real_out)  # given label to the discriminator
        dis_real_loss_JSD = criteriond1_JSD(label_one,dis_real_out)
        torch.cuda.empty_cache()

        # 1.2 Train discriminator on fake data from generator

        noise = torch.normal(0, 1, (inputs.shape), device=device).detach()
        noise_conditional = torch.cat((noise,tensor_H),dim=1)

        #dis_inp_fake_x = gen(noise,h)
        dis_inp_fake_x = gen(noise_conditional)
        dis_inp_fake_x_conditional = torch.cat((dis_inp_fake_x, tensor_H), dim=1)

        dis_fake_out = discrim(dis_inp_fake_x_conditional)
        label_zero = Variable(torch.zeros(batch_size, 1, device=device))
        dis_fake_loss = criteriond1(label_zero,dis_fake_out)
        dis_fake_loss_JSD = criteriond1_JSD(label_one_smoothed, dis_fake_out)
        gc.collect()
        torch.cuda.empty_cache()

        # 1.3 combine real loss and fake loss for discriminator
        #discriminator_loss =  dis_real_loss +  dis_fake_loss + dis_real_loss_JSD + dis_fake_loss_JSD
        discriminator_loss = - torch.mean(dis_real_out) + torch.mean(dis_fake_out)

        discriminator_batch_loss += discriminator_loss.item()
        discriminator_loss.backward()
        optimizerd1.step()
        gc.collect()
        torch.cuda.empty_cache()

        for p in discrim.parameters():
             p.data.clamp_(-0.01, 0.01)

        # 2.1 Training the generator
        print("generator training !")
        gen.zero_grad()
        #noise = torch.normal(0, 1, (inputs.shape), device=device).detach()

        #gen_out = gen(noise, h)
        gen_out = gen(noise_conditional)
        gen_out_conditional = torch.cat((gen_out,tensor_H),dim=1)
        dis_out_gen_training = discrim(gen_out_conditional)
        gen_loss = criteriond2( Variable(torch.ones(batch_size, 1, device = device)),dis_out_gen_training)
        gen_loss_JSD = criteriond2_JSD(dis_out_gen_training, Variable(torch.ones(batch_size, 1, device = device)))
        #gen_loss =  2*gen_loss +  2*gen_loss_JSD

        gen_loss = -torch.mean(dis_out_gen_training)
        generator_batch_loss += gen_loss.item()
        gen_loss.backward()
        optimizerd2.step()
        gc.collect()
        #wandb.watch(gen,log='all',log_freq=1000)
        #wandb.watch(discrim,log='all',log_freq=1000)

        print("=======================================================================================================")

        print(f'Epoch [{epoch}/{epochs}]  Batch {batch_idx + 1}/{len(data_loader)} \
                    Loss D: {discriminator_loss:.4f}, Loss G: {gen_loss:.4f}')

        print("=======================================================================================================")

        wandb.log({"loss disc": discriminator_loss})
        wandb.log({"loss gen": gen_loss})

        print("Figures loading !")

        reconstructed_img_one = gen_out.cpu().detach().numpy()
        axs[0, 0].imshow(inputs[0, 0, :, :].cpu().detach().numpy())
        axs[0, 0].grid(False)
        axs[0, 1].imshow(inputs[1, 0, :, :].cpu().detach().numpy())
        axs[0, 1].grid(False)

        axs[0, 2].imshow(inputs[2, 0, :, :].cpu().detach().numpy())
        axs[0, 2].grid(False)
        axs[0, 3].imshow(inputs[3, 0, :, :].cpu().detach().numpy())
        axs[0, 3].grid(False)
        axs[0, 4].imshow(inputs[4, 0, :, :].cpu().detach().numpy())
        axs[0, 4].grid(False)

        axs[1, 0].imshow(reconstructed_img_one[0, 0, :, :])
        axs[1, 0].grid(False)
        axs[1, 1].imshow(reconstructed_img_one[1, 0, :, :])
        axs[1, 1].grid(False)
        axs[1, 2].imshow(reconstructed_img_one[2, 0, :, :])
        axs[1, 2].grid(False)
        axs[1, 3].imshow(reconstructed_img_one[3, 0, :, :])
        axs[1, 3].grid(False)
        axs[1, 4].imshow(reconstructed_img_one[4, 0, :, :])
        axs[1, 4].grid(False)
        plt.grid(b=None)
        figs.savefig('sample.png')
        wandb.log({"Image_recons": wandb.Image(figs)})
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
