import torch
import torchvision
import foolbox
import numpy as np
import matplotlib.pyplot as plt

# load data and model
num_test = 100
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
beta_vae_b = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP3/beta_vae_b.pkl').eval()
beta_vae_h = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP3/beta_vae_h.pkl').eval()
rev_beta_vae = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP3/rev_beta_vae.pkl').eval()
limit_vae = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP3/limit_vae.pkl').eval()

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

# beta vae h model
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


# rev beta vae  model 0.01
# mu and sigma of normal examples
_, mu, log_sigma = rev_beta_vae(test_x)
# meausre Dkl between N(0, I) and normal examples
score_normal_rev = KL_divergence(log_sigma, mu)
# mu and sigma of adversarial examples
_, mu_adv, log_sigma_adv = rev_beta_vae(torch.from_numpy(cnn_adv_xs_arr).cuda())
# meausre Dkl between N(0, I) and adv examples
score_adv_rev = KL_divergence(log_sigma_adv, mu_adv)
plt.hist(score_normal_rev.data.cpu().numpy(), bins=100, alpha=0.5, label='normal_rev')
plt.hist(score_adv_rev.data.cpu().numpy(), bins=100, alpha=0.5, label='adv_rev')
plt.legend(loc='upper right')
plt.title('rev_beta-VAE')
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


# limit vae model
# mu and sigma of normal examples
_, mu, log_sigma = limit_vae(test_x)
# meausre Dkl between N(0, I) and normal examples
score_normal = KL_divergence(log_sigma, mu)
# mu and sigma of adversarial examples
_, mu_adv, log_sigma_adv = limit_vae(torch.from_numpy(cnn_adv_xs_arr).cuda())
# meausre Dkl between N(0, I) and adv examples
score_adv = KL_divergence(log_sigma_adv, mu_adv)
plt.hist(score_normal.data.cpu().numpy(), bins=100, alpha=0.5, label='normal')
plt.hist(score_adv.data.cpu().numpy(), bins=100, alpha=0.5, label='adv')
plt.legend(loc='upper right')
plt.title('limit_vae')
plt.show()



