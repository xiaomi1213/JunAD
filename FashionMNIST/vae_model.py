import torch
from torch import nn, optim
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nb_latents=10):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        #self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        #self.norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)

        self.fc_mean = nn.Linear(512, nb_latents)
        self.fc_std = nn.Linear(512, nb_latents)

        self.fc2 = nn.Linear(nb_latents, 512)
        self.fc3 = nn.Linear(512, 128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        #self.norm3 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        #self.norm4 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.relu(self.conv1(x))
        #x = self.norm1(x)
        x = self.relu(self.conv2(x))
        #x = self.norm2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.fc1(x.view(-1, 128 * 4 * 4)))
        return self.fc_mean(x), self.fc_std(x)

    def reparameterize(self, mu, log_sigma):
        if self.training:
            std = torch.exp(0.5*log_sigma)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.relu(self.fc2(z))
        x = self.relu(self.fc3(x))
        x = self.relu(self.deconv1(x.view(-1, 128, 4, 4)))
        #x = self.norm3(x)
        x = self.relu(self.deconv2(x))
        #x = self.norm4(x)
        return self.sigmoid(self.deconv3(x))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

