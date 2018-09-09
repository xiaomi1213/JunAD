import torch
from torch.nn import Module, Linear
from torch.distributions.multivariate_normal import MultivariateNormal


class VAE(Module):
    def __init__(self, hidden=2):
        self.layer1 = Linear(784, 512)
        self.mu = Linear(512, hidden)
        self.log_sigma = Linear(525, hidden)

        self.normal = MultivariateNormal(torch.zeros(2), torch.eye(2))

        self.layer2 = Linear(hidden, 512)
        self.out = Linear(512, 784)


    def forward(self,x):
        x = self.layer1(x)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)

        eps = self.normal.sample()
        sample_z = mu + torch.exp(log_sigma/2) * eps

        x = self.layer2(sample_z)
        x = self.out(x)

        return x



