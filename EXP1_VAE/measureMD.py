import torch
import torchvision
import foolbox
import numpy as np
import pickle
import matplotlib.pyplot as plt

# load data and model
num_test = 10
test_data = torchvision.datasets.MNIST(
    root='/home/junhang/Projects/DataSet/MNIST',
    train=False
)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)/255.
test_x = test_x[:num_test].cuda()
test_y = test_data.test_labels
test_y = test_y[:num_test].cuda()

vae_model = torch.load('/home/junhang/Projects/Scripts/saved_model/vae.pkl').eval()
cnn_model = torch.load('/home/junhang/Projects/Scripts/saved_model/cnn.pkl').eval()


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


def MD_torch(x, mu, log_sigma):
    sigma = torch.diag(torch.exp(log_sigma))
    d = x - mu
    d = d.unsqueeze(1)
    dsigma = torch.matmul(d, torch.inverse(sigma))
    dT = torch.transpose(d, 1, 2)
    dsigmadT = torch.matmul(dsigma, dT)
    dist = torch.sqrt(dsigmadT)
    return dist

# measure the distances of adv zs from normal distribution average mu and sigma
mu_normal = torch.from_numpy(np.array([0.0258, -0.0733,  0.0023, -0.0938,  0.0200, -0.0434, -0.0560, -0.1042,
        -0.0839,  0.0274, -0.0602, -0.0827, -0.0023, -0.0131, -0.0048, -0.0554,
        -0.1199, -0.0337, -0.0227, -0.0301])).type(torch.FloatTensor).cuda()
log_sigma_normal = torch.from_numpy(
            np.array([-3.2315, -1.1976, -1.9496, -3.4011, -2.8279, -3.4072, -2.8475, -2.9930,
        -2.2279, -2.5063, -3.0007, -3.0093, -2.1659, -1.7154, -1.5497, -2.5019,
        -1.3952, -3.8016, -1.5189, -1.9021])).type(torch.FloatTensor).cuda()


_, mu, var, z = vae_model(test_x)
dist = MD_torch(var, mu_normal, log_sigma_normal)
dist = dist.squeeze(1).squeeze(1).data.cpu().numpy()
plt.hist(dist, bins=50)
plt.show()

_, mu_adv, var_adv, z_adv = vae_model(torch.from_numpy(cnn_adv_xs_arr).cuda())
adv_dist = MD_torch(var_adv, mu_normal, log_sigma_normal)
adv_dist = adv_dist.squeeze(1).squeeze(1).data.cpu().numpy()
plt.hist(adv_dist, bins=50)
plt.show()