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


import torch
import torchvision
import foolbox
import numpy as np
import matplotlib.pyplot as plt

# load data and model
num_test = 10
test_data = torchvision.datasets.MNIST(
    root='/home/junhang/Projects/DataSet/MNIST',
    train=True
)
test_x = torch.unsqueeze(test_data.train_data, dim=1).type(torch.FloatTensor)/255.
test_x = test_x[:num_test].cuda()
test_y = test_data.train_labels
test_y = test_y[:num_test].cuda()

cnn_model = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP3/cnn.pkl').eval()
vae_model = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP3/vae.pkl').eval()
#beta_vae_b = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP3/beta_vae_b.pkl').eval()
#beta_vae_h = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP3/beta_vae_h.pkl').eval()


# evaluate the cnn model
print("-------------------------evaluating cnn model-----------------------------")
cnn_test_output = cnn_model(test_x)
pred_y = torch.max(cnn_test_output, 1)[1].data.squeeze().cpu().numpy()
cnn_accuracy = float((pred_y == test_y.data.cpu().numpy()).astype(int).sum())/float(test_y.size(0))
print('CNN accuracy: %.4f' % cnn_accuracy)


# select the correctly classified samples indices
print("\n-------------------------selecting samples-----------------------------\n")
a = (pred_y == test_y.data.cpu().numpy()).astype(int)
correct_indice = []
for i in range(num_test):
    if a[i] == 1:
        correct_indice.append(i)


print("-------------------------generating adversarial examples-----------------------------")
#mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
#std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
fmodel = foolbox.models.PyTorchModel(
    cnn_model, bounds=(0, 1), num_classes=10, preprocessing=(0, 1))
attack = foolbox.attacks.FGSM(fmodel)

cnn_adv_test_x_cpu = test_x.cpu().data.numpy()
cnn_adv_test_y_cpu = test_y.cpu().data.numpy()
cnn_adv_xs = []
cnn_adv_ys = []

for i, idx in enumerate(correct_indice):
    cnn_adv_test_x = cnn_adv_test_x_cpu[idx]
    cnn_adv_test_y = cnn_adv_test_y_cpu[idx]
    cnn_adv_x = attack(cnn_adv_test_x, cnn_adv_test_y)
    if cnn_adv_x is None:
        continue
    cnn_adv_xs.append(cnn_adv_x)
    cnn_adv_ys.append(cnn_adv_test_y)


cnn_adv_xs_arr = np.array(cnn_adv_xs)
cnn_adv_ys_arr = np.array(cnn_adv_ys)
print(cnn_adv_xs_arr.shape)



# define KL divergence function
def KL_divergence(logvar, mu):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),1)
    return KLD


# vanilla vae model
# mu and sigma of normal examples
_, mu, log_sigma = vae_model(test_x)
# meausre Dkl between N(0, I) and normal examples
score_normal_1 = KL_divergence(log_sigma, mu)
# mu and sigma of adversarial examples
_, mu_adv, log_sigma_adv = vae_model(torch.from_numpy(cnn_adv_xs_arr).cuda())
# meausre Dkl between N(0, I) and adv examples
score_adv_1 = KL_divergence(log_sigma_adv, mu_adv)
plt.hist(score_normal_1.data.cpu().numpy(), bins=100, alpha=0.5, label='normal_1')
plt.hist(score_adv_1.data.cpu().numpy(), bins=100, alpha=0.5, label='adv_1')
plt.legend(loc='upper right')
plt.title('VAE')
plt.show()


"""
# beta vae h model 0.001
# mu and sigma of normal examples
_, mu, log_sigma = beta_vae_h_1(test_x)
# meausre Dkl between N(0, I) and normal examples
score_normal_1 = KL_divergence(log_sigma, mu)
# mu and sigma of adversarial examples
_, mu_adv, log_sigma_adv = beta_vae_h_1(torch.from_numpy(cnn_adv_xs_arr).cuda())
# meausre Dkl between N(0, I) and adv examples
score_adv_1 = KL_divergence(log_sigma_adv, mu_adv)
plt.hist(score_normal_1.data.cpu().numpy(), bins=100, alpha=0.5, label='normal_1')
plt.hist(score_adv_1.data.cpu().numpy(), bins=100, alpha=0.5, label='adv_1')
plt.legend(loc='upper right')
plt.title('beta-VAE-h-0.001')
plt.show()

# beta vae h model 0.01
# mu and sigma of normal examples
_, mu, log_sigma = beta_vae_h_2(test_x)
# meausre Dkl between N(0, I) and normal examples
score_normal_2 = KL_divergence(log_sigma, mu)
# mu and sigma of adversarial examples
_, mu_adv, log_sigma_adv = beta_vae_h_2(torch.from_numpy(cnn_adv_xs_arr).cuda())
# meausre Dkl between N(0, I) and adv examples
score_adv_2 = KL_divergence(log_sigma_adv, mu_adv)
plt.hist(score_normal_2.data.cpu().numpy(), bins=100, alpha=0.5, label='normal_2')
plt.hist(score_adv_2.data.cpu().numpy(), bins=100, alpha=0.5, label='adv_2')
plt.legend(loc='upper right')
plt.title('beta-VAE-h-0.01')
plt.show()

# beta vae h model 0.1
# mu and sigma of normal examples
_, mu, log_sigma = beta_vae_h_3(test_x)
# meausre Dkl between N(0, I) and normal examples
score_normal_3 = KL_divergence(log_sigma, mu)
# mu and sigma of adversarial examples
_, mu_adv, log_sigma_adv = beta_vae_h_3(torch.from_numpy(cnn_adv_xs_arr).cuda())
# meausre Dkl between N(0, I) and adv examples
score_adv_3 = KL_divergence(log_sigma_adv, mu_adv)
plt.hist(score_normal_3.data.cpu().numpy(), bins=100, alpha=0.5, label='normal_3')
plt.hist(score_adv_3.data.cpu().numpy(), bins=100, alpha=0.5, label='adv_3')
plt.legend(loc='upper right')
plt.title('beta-VAE-h-0.1')
plt.show()

# beta vae h model 1
# mu and sigma of normal examples
_, mu, log_sigma = beta_vae_h_4(test_x)
# meausre Dkl between N(0, I) and normal examples
score_normal_4 = KL_divergence(log_sigma, mu)
# mu and sigma of adversarial examples
_, mu_adv, log_sigma_adv = beta_vae_h_4(torch.from_numpy(cnn_adv_xs_arr).cuda())
# meausre Dkl between N(0, I) and adv examples
score_adv_4 = KL_divergence(log_sigma_adv, mu_adv)
plt.hist(score_normal_4.data.cpu().numpy(), bins=100, alpha=0.5, label='normal_4')
plt.hist(score_adv_4.data.cpu().numpy(), bins=100, alpha=0.5, label='adv_4')
plt.legend(loc='upper right')
plt.title('beta-VAE-h-1')
plt.show()

# beta vae h model 100
# mu and sigma of normal examples
_, mu, log_sigma = beta_vae_h_6(test_x)
# meausre Dkl between N(0, I) and normal examples
score_normal_6 = KL_divergence(log_sigma, mu)
# mu and sigma of adversarial examples
_, mu_adv, log_sigma_adv = beta_vae_h_6(torch.from_numpy(cnn_adv_xs_arr).cuda())
# meausre Dkl between N(0, I) and adv examples
score_adv_6 = KL_divergence(log_sigma_adv, mu_adv)
plt.hist(score_normal_6.data.cpu().numpy(), bins=100, alpha=0.5, label='normal_6')
plt.hist(score_adv_6.data.cpu().numpy(), bins=100, alpha=0.5, label='adv_6')
plt.legend(loc='upper right')
plt.title('beta-VAE-h-100')
plt.show()

# beta vae h model 1000
# mu and sigma of normal examples
_, mu, log_sigma = beta_vae_h_7(test_x)
# meausre Dkl between N(0, I) and normal examples
score_normal_7 = KL_divergence(log_sigma, mu)
# mu and sigma of adversarial examples
_, mu_adv, log_sigma_adv = beta_vae_h_7(torch.from_numpy(cnn_adv_xs_arr).cuda())
# meausre Dkl between N(0, I) and adv examples
score_adv_7 = KL_divergence(log_sigma_adv, mu_adv)
plt.hist(score_normal_7.data.cpu().numpy(), bins=100, alpha=0.5, label='normal_7')
plt.hist(score_adv_7.data.cpu().numpy(), bins=100, alpha=0.5, label='adv_7')
plt.legend(loc='upper right')
plt.title('beta-VAE-h-1000')
plt.show()



"""



"""

# beta vae h model 10
# mu and sigma of normal examples
_, mu, log_sigma = beta_vae_h(test_x)
# meausre Dkl between N(0, I) and normal examples
score_normal_5 = KL_divergence(log_sigma, mu)
# mu and sigma of adversarial examples
_, mu_adv, log_sigma_adv = beta_vae_h(torch.from_numpy(cnn_adv_xs_arr).cuda())
# meausre Dkl between N(0, I) and adv examples
score_adv_5 = KL_divergence(log_sigma_adv, mu_adv)
plt.hist(score_normal_5.data.cpu().numpy(), bins=100, alpha=0.5, label='normal_5')
plt.hist(score_adv_5.data.cpu().numpy(), bins=100, alpha=0.5, label='adv_5')
plt.legend(loc='upper right')
plt.title('beta-VAE-H')
plt.show()


# beta vae b model
# mu and sigma of normal examples
_, mu, log_sigma = beta_vae_b(test_x)
# meausre Dkl between N(0, I) and normal examples
score_normal = KL_divergence(log_sigma, mu)
# mu and sigma of adversarial examples
_, mu_adv, log_sigma_adv = beta_vae_b(torch.from_numpy(cnn_adv_xs_arr).cuda())
# meausre Dkl between N(0, I) and adv examples
score_adv = KL_divergence(log_sigma_adv, mu_adv)
plt.hist(score_normal.data.cpu().numpy(), bins=100, alpha=0.5, label='normal')
plt.hist(score_adv.data.cpu().numpy(), bins=100, alpha=0.5, label='adv')
plt.legend(loc='upper right')
plt.title('beta-VAE-B')
plt.show()
"""
