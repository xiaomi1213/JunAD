import torch
from torch import nn, optim
from torch.nn import functional as F

class DPVAE(nn.Module):
    def __init__(self):
        super(DPVAE, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2_mu = nn.Linear(512, 2)
        self.fc2_sigma = nn.Linear(512, 2)
        self.fc3 = nn.Linear(2, 512)
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
            std = torch.exp(0.5 * log_sigma)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)

    def decoder(self, z):
        h3 = F.relu(self.fc3(z))
        recon_x = torch.sigmoid(self.fc4(h3))
        return recon_x

    def forward(self, x):
        exp, var = self.encoder(x.view(-1, 784))
        z = self.reparameterize(exp, var)
        recon_x = self.decoder(z)
        return recon_x, exp, var, z

