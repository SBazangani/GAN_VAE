import torch.nn.functional as F
import torch
from Model.Generator_VAE import Gen_AE
from Model.Generator import GEN
from Model.Discriminator import Disc
import torch.optim as optim
import matplotlib.pyplot as plt

torch.cuda.empty_cache()


def criteriond1_JSD(logits1, logits2):
    softmax1 = torch.softmax(logits1 + 1e-11, 1)
    softmax2 = (torch.softmax(logits2 + 1e-11, 1))
    M = 0.5 * (softmax1 + softmax2)
    loss = 0.0
    return 0.5 * (F.kl_div(M.log(), softmax1) + F.kl_div(M.log(), softmax2))


def criteriond2_JSD(logits1, logits2):
    softmax1 = torch.softmax(logits1 + 1e-10, 1)
    softmax2 = (torch.softmax(logits2 + 1e-10, 1))
    M = 0.5 * (softmax1 + softmax2)
    loss = 0.0
    return 0.5 * (F.kl_div(M.log(), softmax1) + F.kl_div(M.log(), softmax2))


def MODEL():
    gen = GEN()
    gen = gen.float()
    discrim = Disc()
    discrim = discrim.float()

    beta1 = 0.5
    beta2 = 0.999
    # Hyper param

    # Creates a criterion that measures the Binary Cross Entropy between the target and the output
    criteriond1 = torch.nn.BCEWithLogitsLoss()
    #criteriond1 = torch.nn.MSELoss()
    criteriond1_Kl = torch.nn.KLDivLoss()

    #criteriond2 = torch.nn.MSELoss()
    criteriond2 = torch.nn.BCEWithLogitsLoss()
    criteriond2_Kl = torch.nn.KLDivLoss()
    # criteriond2= torch.nn.KLDivLoss(reduction= 'batchmean')

    optimizerd1 = optim.RMSprop(discrim.parameters(), lr=0.00001,alpha=0.00005)
    optimizerd2 = optim.RMSprop(gen.parameters(), lr=0.00001, alpha=0.00005)
    #optimizerd2 = optim.(gen.parameters(), lr=0.00001, alpha=0.00005)

    return gen, discrim, criteriond1, criteriond2, optimizerd1, optimizerd2


if __name__ == '__main__':
    gen, discrim, criteriond1, criteriond2, optimizerd1, optimizerd2 = MODEL()