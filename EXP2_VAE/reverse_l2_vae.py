import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

class REVERSE_L2_VAE(nn.Module):
    def __init__(self, tao):
        super(REVERSE_L2_VAE, self).__init__()

        self.tao = tao

        self.fc1 = nn.Linear(784, 512)
        self.fc2_mu = nn.Linear(512, 20)
        self.fc2_sigma = nn.Linear(512, 20)
        self.fc3 = nn.Linear(20, 512)
        self.fc4 = nn.Linear(512, 784)

    def KL_divergence(logvar, mu):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD


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
            mu_distance = torch.mean(torch.sqrt(torch.sum(torch.pow(mu,2), 1)))
            if mu_distance > 3:
                print('The sample is legitimate')
                return mu
            else:
                mu += 10
                std = torch.exp(0.5 * log_sigma)
                eps = torch.randn_like(std)
                return eps.mul(std).add_(mu)


    def decoder(self, z):
        h3 = F.relu(self.fc3(z))
        recon_x = torch.sigmoid(self.fc4(h3))
        return recon_x

    def forward(self, x):
        exp, var = self.encoder(x.view(-1, 784))
        """
        # detecting and reversing
        if self.eval:
            Dkl = -0.5 * torch.sum(1 + var - exp.pow(2) - var.exp())
            print("Dkl_score: ", Dkl)

            if Dkl <= self.tao:
                print("The sample is legitimate")

            else:
                print("The sample is adversarial, reversing it")
                rev_optim = optim.Adam([var, exp], lr=1e-4)

                while (Dkl <= self.tao):
                    Dkl.backward()
                    rev_optim.zero_grad()
                    rev_optim.step()
                    # Dkl = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
            """

        z = self.reparameterize(exp, var)
        recon_x = self.decoder(z)
        return recon_x, exp, var
