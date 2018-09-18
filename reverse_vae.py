import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

class REVERSE_VAE(nn.Module):
    def __init__(self):
        super(REVERSE_VAE, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2_mu = nn.Linear(512, 20)
        self.fc2_sigma = nn.Linear(512, 20)
        self.fc3 = nn.Linear(20, 512)
        self.fc4 = nn.Linear(512, 784)

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        h2_mu = self.fc2_mu(h1)
        h2_sigma = self.fc2_sigma(h1)
        return h2_mu, h2_sigma

    def reparameterize(self, mu, log_sigma):
        if self.training:
            std = torch.exp(0.5*log_sigma)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            # detecting and reversing
            score_adv = np.mean(KL_divergence(standard_Gassian, z_distribution(mu_adv, log_sigma_adv)))
            print("score_adv: ", score_adv)
            tao = score_normal
            enta = 1
            if score_adv <= tao:
                print("The sample is legitimate")
                return mu
            else:
                print("The sample is adversarial, reversing it")
                rev_d = score_adv - tao
                mu_rev = mu_adv - min(rev_d, enta)
                sigma_rev = sigma_adv - min(rev_d, enta)
                z_rev = mu_rev + sigma_rev * normal
                return z_rev

    def decoder(self, z):
        h3 = F.relu(self.fc3(z))
        recon_x = torch.sigmoid(self.fc4(h3))
        return recon_x

    def forward(self, x):
        exp, var = self.encoder(x.view(-1, 784))
        z = self.reparameterize(exp, var)
        recon_x = self.decoder(z)
        return recon_x, exp, var
