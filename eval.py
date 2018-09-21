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


def L2_reg_distance(mu, log_sigma, mu_normal, log_sigma_normal):
    mu_distance = torch.mean(torch.sum(torch.pow(mu - mu_normal, 2), 1))
    sigma_distance = torch.mean(
        torch.sum(torch.pow((torch.exp(log_sigma) - torch.exp(log_sigma_normal)), 2), 1))
    return mu_distance, sigma_distance

mu_normal = torch.from_numpy(np.array([ 0, 0, 0, 0, 0])).type(torch.FloatTensor).cuda()

log_sigma_normal = torch.from_numpy(np.array([0, 0, 0, 0, 0])).type(torch.FloatTensor).cuda()

mu = torch.from_numpy(np.array([ -0.5,  -0.5,  -0.5, -0.5,  -0.5])).type(torch.FloatTensor).cuda()
#mu = torch.autograd.Variable(mu, requires_grad=True)
log_sigma = torch.from_numpy(np.array([0, 0, 0, 0, 0])).type(torch.FloatTensor).cuda()
#log_sigma = torch.autograd.Variable(log_sigma, requires_grad=True)

reg_Dkl = torch.pow(torch.norm((mu-mu_normal),2),2)+torch.pow(torch.norm((log_sigma-log_sigma_normal),2),2)
reg_Dkl = torch.autograd.Variable(reg_Dkl,requires_grad=True)

opt = torch.optim.Adam((mu, log_sigma), lr=1e-2)
for i in range(5):
    opt.zero_grad()
    reg_Dkl.backward()
    opt.step()
    print(mu.grad)

    print(mu)
    print(reg_Dkl)


a = torch.FloatTensor([2])
x = torch.FloatTensor([2,3,4])
target = torch.FloatTensor([1])
y = a*torch.sum(x)
loss = y-target

loss = torch.autograd.Variable(loss, requires_grad=True)
opt1 = torch.optim.Adam([a,x],lr=1e-2)
for i in range(5):
    opt1.zero_grad()
    loss.backward()
    opt1.step()
    print(loss)

