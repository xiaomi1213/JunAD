import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np


class REV_BETA_VAE(nn.Module):
    def __init__(self, nb_latents=10):
        super(REV_BETA_VAE, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)

        self.fc_mean = nn.Linear(256, nb_latents)
        self.fc_std = nn.Linear(256, nb_latents)

        self.fc2 = nn.Linear(nb_latents, 256)
        self.fc3 = nn.Linear(256, 64 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def rev_vol(self, mu_adv, log_sigma_adv):
        """
        mu_normal = torch.from_numpy(np.array([0.0258, -0.0733, 0.0023, -0.0938, 0.0200, -0.0434, -0.0560, -0.1042,
                                               -0.0839, 0.0274, -0.0602, -0.0827, -0.0023, -0.0131, -0.0048, -0.0554,
                                               -0.1199, -0.0337, -0.0227, -0.0301])).type(torch.FloatTensor).cuda()
        log_sigma_normal = torch.from_numpy(
            np.array([-3.2315, -1.1976, -1.9496, -3.4011, -2.8279, -3.4072, -2.8475, -2.9930,
                      -2.2279, -2.5063, -3.0007, -3.0093, -2.1659, -1.7154, -1.5497, -2.5019,
                      -1.3952, -3.8016, -1.5189, -1.9021])).type(torch.FloatTensor).cuda()
        KLD = 0.5 * torch.sum(log_sigma_normal - log_sigma_adv \
                               + torch.pow(mu_adv - mu_adv, 2) * (1 / torch.exp(log_sigma_normal)) \
                               + torch.exp(log_sigma_adv) * (1 / torch.exp(log_sigma_normal)) - 1, 1)
        norm_mu = torch.pow(torch.norm(mu_adv - mu_normal, 2), 2)
        norm_log_sigma = torch.pow(torch.norm(log_sigma_adv - log_sigma_normal, 2), 2)
        vol = KLD \
              - 0 * norm_mu \
              - 0 * norm_log_sigma
        return vol / len(mu_adv)
        """
        KLD = -0.5 * torch.sum(1 + log_sigma_adv - mu_adv.pow(2) - log_sigma_adv.exp())
        return KLD

    def reversing(self, mu, log_sigma):
        lr = 0.1
        update_mu = mu.clone()
        update_log_sigma = log_sigma.clone()
        for i in range(100):
            reversing_vol = self.rev_vol(update_mu, update_log_sigma)
            mu_grad, sigma_grad = torch.autograd.grad(reversing_vol, [update_mu, update_log_sigma], \
                                                      allow_unused=True, retain_graph=True)
            update_mu += lr * mu_grad
            update_log_sigma += lr * sigma_grad
            print("reversing_vol: ", reversing_vol)

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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar