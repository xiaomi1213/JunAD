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
            mu_normal = torch.from_numpy(np.array([ 0.3480, -0.1050,  0.0468, -0.3077, -0.0858,  0.0928,  0.0886,  0.1352,
        -0.1193,  0.3233, -0.3130, -0.1185,  0.2065, -0.4571,  0.0012, -0.3993,
        -0.3814,  0.1663, -0.0608,  0.0035,  0.1776, -0.1510, -0.2439,  0.2294,
        -0.2825, -0.2733,  0.0477,  0.0784, -0.3014,  0.2121,  0.0520, -0.0477,
         0.0870, -0.1208,  0.2554, -0.1111, -0.0871,  0.0559, -0.3548,  0.0662,
         0.0439, -0.0959, -0.1112, -0.1471,  0.3845,  0.0193,  0.2070,  0.1193,
         0.0535,  0.0903,  0.0068, -0.2066,  0.3892, -0.3527, -0.0975, -0.1206,
         0.2637, -0.0526,  0.1320,  0.2722,  0.0253,  0.1636, -0.1688,  0.2057,
         0.4805,  0.1210,  0.0016,  0.0530, -0.0563, -0.3043,  0.2559, -0.3621,
        -0.0038,  0.1283,  0.1991,  0.1688, -0.1201, -0.0797,  0.2793,  0.1904,
        -0.0105, -0.4763, -0.1211, -0.0235, -0.0221,  0.1442,  0.1626, -0.0151,
        -0.2517,  0.2028, -0.3451, -0.1992,  0.3869, -0.0196, -0.0066, -0.2654,
         0.1254, -0.0969, -0.4034, -0.3729])).cuda()
            log_sigma_normal = torch.from_numpy(np.array([-2.2643, -2.8146, -1.5796, -2.6966, -2.4679, -1.5462, -2.4373, -2.6706,
        -3.0956, -2.6162, -2.5269, -2.6727, -2.3953, -2.5427, -1.7464, -2.5107,
        -2.4663, -2.5219, -3.1516, -2.4011, -2.4866, -2.4784, -2.6602, -2.5634,
        -2.4028, -2.7931, -2.3710, -2.5464, -2.4784, -1.7874, -2.6247, -1.7504,
        -2.6035, -2.5811, -2.4427, -2.8318, -2.3325, -1.6016, -2.3075, -1.7176,
        -1.6063, -2.4488, -2.3609, -2.7131, -2.5438, -2.4578, -2.1115, -2.7321,
        -2.5798, -2.5000, -2.4738, -2.9205, -2.6922, -2.3187, -3.1208, -2.5000,
        -2.5940, -1.4643, -2.5341, -2.2839, -2.5769, -3.0377, -2.3435, -2.7316,
        -2.7387, -2.2442, -2.8491, -2.8348, -2.6846, -2.8087, -2.5267, -2.4942,
        -2.6217, -2.5356, -1.7189, -2.6012, -2.2414, -2.6386, -2.2202, -2.5812,
        -2.3785, -2.4909, -2.7311, -2.2526, -2.7489, -2.5867, -2.3445, -2.4392,
        -2.5805, -1.8482, -2.6246, -2.2658, -2.2864, -2.6104, -2.2056, -3.0322,
        -2.0202, -2.8292, -2.5012, -2.6471])).cuda()
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
