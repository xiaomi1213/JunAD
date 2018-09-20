import torch
import numpy as np

"""
mu = torch.autograd.Variable(torch.FloatTensor([0, 0]), requires_grad=True)
#mu = torch.autograd.Variable(mu, requires_grad=True)
log_sigma = torch.autograd.Variable(torch.FloatTensor([0, 0]), requires_grad=True)
#log_sigma = torch.autograd.Variable(log_sigma, requires_grad=True)

score = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
#score = torch.autograd.Variable(score)
print("score: ", score)



#optimiser = torch.optim.Adam([mu, log_sigma], lr=1e-3)

update_mu = mu.clone()
update_log_sigma = log_sigma.clone()

for i in range(250):
    gradients = torch.autograd.grad(score, [mu, log_sigma],retain_graph=True)
    #print(gradients)
    update_mu += -1 * (1e-3) * gradients[0]
    update_log_sigma += -1 * (1e-3) * gradients[1]

update_score = -0.5 * torch.sum(1 + update_log_sigma - update_mu.pow(2) - update_log_sigma.exp())

print("update_score: ", update_score)
print('update_mu:', update_mu)
print('update_log_sigma',update_log_sigma)

print("score: ", score)
print('mu:', mu)
print('sigma',log_sigma)



a = torch.autograd.Variable(torch.FloatTensor([[1, 2, 3], [4, 5, 6]]), requires_grad=1)
b = torch.mul(a, 2)
c = b.sum()
d = torch.autograd.grad(c, a, create_graph=True)

print("d gradients are fine")
print(d)
"""

a = np.array([ 0.0187, -0.3948,  0.2555, -0.9905, -0.8428,  0.0551,  0.3222, -0.6254,
         -0.3385, -0.0504,  0.1580,  0.5903,  0.0995,  0.4910, -0.4610, -0.0508,
         -0.7808,  0.0267, -0.5674,  0.8931])
c = [1,2]
#b = 1 * a
d = 0.1 * c
print(d)
