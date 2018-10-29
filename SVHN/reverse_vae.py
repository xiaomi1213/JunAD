import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np


class REVERSE_VAE(nn.Module):
    def __init__(self, nb_latents=20):
        super(REVERSE_VAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)

        self.fc_mean = nn.Linear(256, nb_latents)
        self.fc_std = nn.Linear(256, nb_latents)

        self.fc2 = nn.Linear(nb_latents, 256)
        self.fc3 = nn.Linear(256, 64 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def rev_vol(self, mu_adv, log_sigma_adv):
        KLD = -0.5 * torch.sum(1 + log_sigma_adv - mu_adv.pow(2) - log_sigma_adv.exp(),1)
        return KLD

    def reversing(self, mu, log_sigma):
        lr = 1e-2
        update_mu = mu.clone()
        update_log_sigma = log_sigma.clone()
        init_rev_vol = self.rev_vol(update_mu, update_log_sigma)
        print("init_rev_vol: ", init_rev_vol)
        for i in range(50):
            reversing_vol = self.rev_vol(update_mu, update_log_sigma)
            reversed_vol = reversing_vol - init_rev_vol
            if (reversed_vol) >= 40:
                break
            mu_grad, sigma_grad = torch.autograd.grad(reversing_vol, [update_mu, update_log_sigma], \
                                                      allow_unused=True, retain_graph=True)
            update_mu += lr * mu_grad
            update_log_sigma += lr * sigma_grad

        print("reversed_vol: ", reversed_vol)

        std = torch.exp(0.5 * update_log_sigma)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(update_mu)

    def encoder(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.fc1(x.view(-1, 64 * 4 * 4)))
        return self.fc_mean(x), self.fc_std(x)

    def reparameterize(self, mu, log_sigma):
        if self.training:
            std = torch.exp(0.5 * log_sigma)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            # detecting and reversing
            z = self.reversing(mu, log_sigma)
            return z

    def decoder(self, z):
        x = self.relu(self.fc2(z))
        x = self.relu(self.fc3(x))
        x = self.relu(self.deconv1(x.view(-1, 64, 4, 4)))
        x = self.relu(self.deconv2(x))
        return self.sigmoid(self.deconv3(x))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar