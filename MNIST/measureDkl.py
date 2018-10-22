import torch
import torchvision
import foolbox
import numpy as np
import pickle
import matplotlib.pyplot as plt

# load data and model
num_test = 10000
test_data = torchvision.datasets.MNIST(
    root='/home/junhang/Projects/DataSet/MNIST',
    train=False
)
test_x = torch.unsqueeze(test_data.train_data, dim=1).type(torch.FloatTensor)/255.
test_x = test_x[:num_test].cuda()
test_y = test_data.train_labels
test_y = test_y[:num_test].cuda()

vae_model = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP2/vae.pkl').eval()
cnn_model = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP2/cnn.pkl').eval()


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


# mu and sigma of normal examples
_, mu, log_sigma,_ = vae_model(test_x)

# meausre Dkl between N(0, I) and normal examples
score_normal = KL_divergence(log_sigma, mu)
print("score_normal: ", score_normal)
plt.hist(score_normal.data.cpu().numpy(), bins=100)
plt.show()

# mu and sigma of adversarial examples
_, mu_adv, log_sigma_adv, _ = vae_model(torch.from_numpy(cnn_adv_xs_arr).cuda())
#sigma_adv = np.exp(log_sigma_adv)

# meausre Dkl between N(0, I) and adv examples
score_adv = KL_divergence(log_sigma_adv, mu_adv)
print("score_adv: ", score_adv)
plt.hist(score_adv.data.cpu().numpy(), bins=100)
plt.show()

print("-------------------------get average mu and sigma of normal examples-----------------------------")
_, mu, log_sigma,_ = vae_model(test_x)
average_mu = torch.mean(mu,0).type(torch.FloatTensor)
average_log_sigma = torch.mean(log_sigma,0).type(torch.FloatTensor)
print("average_mu", average_mu, "average_log_sigma", average_log_sigma)





