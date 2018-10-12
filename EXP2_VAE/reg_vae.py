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

    """
    def KL_divergence(self, logvar, mu):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD
    
    def L2_reg_distance(self, mu, log_sigma, mu_normal, log_sigma_normal):
        mu_distance = torch.mean(torch.sqrt(torch.sum(torch.pow(mu-mu_normal, 2), 1)))
        sigma_distance = torch.mean(torch.sqrt(torch.sum(torch.pow((torch.exp(log_sigma) - torch.exp(log_sigma_normal)), 2), 1)))
        return mu_distance, sigma_distance
    """


    def rev_vol(self, mu_adv, log_sigma_adv):
        mu_normal = torch.from_numpy(np.array([0.0258, -0.0733,  0.0023, -0.0938,  0.0200, -0.0434, -0.0560, -0.1042,
        -0.0839,  0.0274, -0.0602, -0.0827, -0.0023, -0.0131, -0.0048, -0.0554,
        -0.1199, -0.0337, -0.0227, -0.0301])).type(torch.FloatTensor).cuda()
        log_sigma_normal = torch.from_numpy(
            np.array([-3.2315, -1.1976, -1.9496, -3.4011, -2.8279, -3.4072, -2.8475, -2.9930,
        -2.2279, -2.5063, -3.0007, -3.0093, -2.1659, -1.7154, -1.5497, -2.5019,
        -1.3952, -3.8016, -1.5189, -1.9021])).type(torch.FloatTensor).cuda()
        KLD1 = 0.5 * torch.sum(1 + log_sigma_adv - mu_adv.pow(2) - log_sigma_adv.exp())
        KLD2 = 0.5 * torch.sum(log_sigma_normal - log_sigma_adv \
                + torch.pow(mu_adv-mu_adv,2)*(1/torch.exp(log_sigma_normal)) \
                + torch.exp(log_sigma_adv)*(1/torch.exp(log_sigma_normal)) - 1, 1)
        norm_mu = torch.pow(torch.norm(mu_adv - mu_normal, 2), 2)
        norm_log_sigma = torch.pow(torch.norm(log_sigma_adv - log_sigma_normal,2),2)
        vol = KLD2 \
              - 0*norm_mu\
              - 0*norm_log_sigma


        return vol/len(mu_adv)

    def reversing(self, mu, log_sigma):
        lr = 1
        update_mu = mu.clone()
        update_log_sigma = log_sigma.clone()
        for i in range(100):
            # gradients = torch.autograd.grad(update_score, [mu, log_sigma], allow_unused=True, retain_graph=True)
            # # print(gradients)
            # mu -= torch.mul(gradients[0], LR)
            # log_sigma -= torch.mul(gradients[1], LR)
            #
            # # update_score = 0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp()) \
            # # + 0.1*self.L2_reg_distance(mu, log_sigma, mu_normal, log_sigma_normal)[0] \
            # # + 0.1*self.L2_reg_distance(mu, log_sigma, mu_normal, log_sigma_normal)[1]
            #
            # update_score = self.L2_reg_distance(mu, log_sigma, mu_normal, log_sigma_normal)[0] \
            #                + self.L2_reg_distance(mu, log_sigma, mu_normal, log_sigma_normal)[1]
            #
            # update_score = update_score / len(mu)
            # print("update_score: ", update_score)
            reversing_vol = self.rev_vol(update_mu, update_log_sigma)
            mu_grad, sigma_grad = torch.autograd.grad(reversing_vol, [update_mu, update_log_sigma], \
                                                      allow_unused=True, retain_graph=True)
            update_mu += lr*mu_grad
            update_log_sigma +=lr*sigma_grad
            print("reversing_vol: ", reversing_vol)

        std = torch.exp(0.5 * update_log_sigma)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(update_mu)


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
            z = self.reversing(mu, log_sigma)
            return z



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
