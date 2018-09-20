import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

class REG_VAE(nn.Module):
    def __init__(self, tao):
        super(REG_VAE, self).__init__()

        self.tao = tao
        #self.mu_normal = mu_normal
        #self.log_sigma_normal = log_sigma_normal
        self.fc1 = nn.Linear(784, 512)
        self.fc2_mu = nn.Linear(512, 20)
        self.fc2_sigma = nn.Linear(512, 20)
        self.fc3 = nn.Linear(20, 512)
        self.fc4 = nn.Linear(512, 784)

    def KL_divergence(self, logvar, mu):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD

    def L2_reg_distance(self, mu, log_sigma, mu_normal, log_sigma_normal):
        mu_distance = torch.mean(torch.sqrt(torch.sum(torch.pow(mu-mu_normal, 2), 1)))
        sigma_distance = torch.mean(torch.sqrt(torch.sum(torch.pow((torch.exp(log_sigma) - torch.exp(log_sigma_normal)), 2), 1)))
        return mu_distance, sigma_distance

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
            #return mu
            # detecting and reversing
            mu_normal = torch.from_numpy(np.array()).cuda()
            log_sigma_normal = torch.from_numpy(np.array()).cuda()
            reg_Dkl = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp()) \
                      - self.L2_reg_distance(mu, log_sigma, mu_normal, log_sigma_normal)[0] \
                            - self.L2_reg_distance(mu, log_sigma, mu_normal, log_sigma_normal)[1]
            reg_Dkl = reg_Dkl/len(mu)
            print("Dkl_score: ", reg_Dkl)

            if reg_Dkl <= self.tao:
                print("The sample is legitimate")
                return mu

            else:
                print("The sample is adversarial, reversing it")
                #update_mu = mu.clone()
                #update_log_sigma = log_sigma.clone()
                update_score = reg_Dkl.clone()
                LR = 1e-2
                for i in range(5000):
                    gradients = torch.autograd.grad(update_score, [mu, log_sigma],  allow_unused=True,retain_graph=True)
                    #print(gradients)
                    mu += torch.mul(gradients[0],LR)
                    log_sigma += torch.mul(gradients[1],LR)

                    update_score = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp()) \
                      - self.L2_reg_distance(mu, log_sigma, mu_normal, log_sigma_normal)[0] \
                            - self.L2_reg_distance(mu, log_sigma, mu_normal, log_sigma_normal)[1]
                    update_score = update_score / len(mu)
                    print("update_score: ", update_score)


                std = torch.exp(0.5 * log_sigma)
                eps = torch.randn_like(std)
                return eps.mul(std).add_(mu)
                #return update_mu


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
