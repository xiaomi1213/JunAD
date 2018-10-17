import torch
import numpy as np
import torchvision
import torch.utils.data as Data
"""




a = torch.autograd.Variable(torch.FloatTensor([[1, 2, 3], [4, 5, 6]]), requires_grad=1)
b = torch.mul(a, 2)
c = b.sum()
d = torch.autograd.grad(c, a, create_graph=True)

print("d gradients are fine")
print(d)



def L2_reg_distance(mu, log_sigma, mu_normal, log_sigma_normal):
    mu_distance = torch.mean(torch.sum(torch.pow(mu - mu_normal, 2), 1))
    sigma_distance = torch.mean(
        torch.sum(torch.pow((torch.exp(log_sigma) - torch.exp(log_sigma_normal)), 2), 1))
    return mu_distance, sigma_distance

mu_normal = torch.from_numpy(np.array([ 0, 0, 0, 0, 0])).type(torch.FloatTensor).cuda()

log_sigma_normal = torch.from_numpy(np.array([0, 0, 0, 0, 0])).type(torch.FloatTensor).cuda()

mu = torch.from_numpy(np.array([ -0.5,  -0.5,  -0.5, -0.5,  -0.5])).type(torch.FloatTensor).cuda()
mu = torch.autograd.Variable(mu, requires_grad=True)
log_sigma = torch.from_numpy(np.array([0, 0, 0, 0, 0])).type(torch.FloatTensor).cuda()
log_sigma = torch.autograd.Variable(log_sigma, requires_grad=True)


#reg_Dkl = torch.autograd.Variable(reg_Dkl,requires_grad=True)

opt = torch.optim.Adam((mu, log_sigma), lr=1e-2)
for i in range(5):
    opt.zero_grad()
    reg_Dkl = torch.pow(torch.norm((mu - mu_normal), 2), 2) + torch.pow(torch.norm((log_sigma - log_sigma_normal), 2),2)
    reg_Dkl.backward()
    opt.step()
    print(mu.grad)
    print(reg_Dkl)



from torch import FloatTensor
from torch.autograd import Variable


# Define the leaf nodes
a = Variable(FloatTensor([4]), requires_grad=True)

weights = [Variable(FloatTensor([i]), requires_grad=True) for i in (2, 5, 9, 7)]
#weights = [FloatTensor([i]) for i in (2, 5, 9, 7)]
# unpack the weights for nicer assignment
w1, w2, w3, w4 = weights

b = w1 * a
c = w2 * a
d = w3 * b + w4 * c
L = d

L.backward()

for index, weight in enumerate(weights, start=1):
    gradient, *_ = weight.grad.data
    print(f"Gradient of w{index} w.r.t to L: {gradient}")
print(a.grad.data)


import torch.nn.functional as F
loss_fn = torch.nn.BCELoss(reduce=False, size_average=False)
input = torch.autograd.Variable(torch.randn(3, 4))
target = torch.autograd.Variable(torch.FloatTensor(3, 4).random_(2))
loss = loss_fn(F.sigmoid(input), target)
print(input); print(target); print(loss)

m = torch.nn.LogSoftmax()
input = torch.randn(2, 3)
output = m(input)
print(output)


a = torch.FloatTensor([1,2,4])
b = 1/a
print(b)


def MD(x, mu, log_sigma):
    x = x.data.cpu().numpy()
    mu = mu.data.cpu().numpy()
    sigma = torch.exp(log_sigma).data.cpu().numpy()
    sigma = np.diag(sigma)
    dist = np.sqrt((x-mu).T.dot(np.linalg.inv(sigma)).dot((x-mu)))
    return dist



print(distance)


def MD_torch(x, mu, log_sigma):
    sigma = torch.diag(torch.exp(log_sigma))
    d = x - mu
    d = d.unsqueeze(1)
    dsigma = torch.matmul(d, torch.inverse(sigma))
    dT = torch.transpose(d, 1, 2)
    dsigmadT = torch.matmul(dsigma, dT)
    dist = torch.sqrt(dsigmadT)
    return dist

a = np.array([[0.,0.],[1.,1.]])
a = torch.from_numpy(a)
mu = np.array([0.,0.])
mu = torch.from_numpy(mu)
log_sigma = np.array([0.,0.])
log_sigma = torch.from_numpy(log_sigma)
distance = MD_torch(a, mu, log_sigma)
print(distance)


mu = torch.autograd.Variable(torch.FloatTensor([[0, 0], [1, 1]]), requires_grad=True)
#mu = torch.autograd.Variable(mu, requires_grad=True)
log_sigma = torch.autograd.Variable(torch.FloatTensor([[0, 0], [1, 1]]), requires_grad=True)
#log_sigma = torch.autograd.Variable(log_sigma, requires_grad=True)

score = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(),1)
#score = torch.autograd.Variable(score)
print("score: ", score)
print(score.size())



#optimiser = torch.optim.Adam([mu, log_sigma], lr=1e-3)

update_mu = mu.clone()
update_log_sigma = log_sigma.clone()

for i in range(250):
    gradients = torch.autograd.grad(score, [mu, log_sigma],retain_graph=True)
    #print(gradients)
    update_mu += -1 * (1e-3) * gradients[0]
    update_log_sigma += -1 * (1e-3) * gradients[1]

update_score = -0.5 * torch.sum(1 + update_log_sigma - update_mu.pow(2) - update_log_sigma.exp())

print(update_score.size())
print("update_score: ", update_score)
print('update_mu:', update_mu)
print('update_log_sigma',update_log_sigma)

print("score: ", score)
print('mu:', mu)
print('sigma',log_sigma)

"""
num_test = 10
reduced_num_test = list(range(num_test))
for i in range(num_test):
    choice_num = np.random.choice(reduced_num_test, 1).squeeze()
    reduced_num_test.remove(choice_num)
    print(reduced_num_test)
    print(choice_num)