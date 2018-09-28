import torch
import torchvision
import foolbox
import numpy as np
import pickle


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
"""
# measure the distances of adv zs from normal distribution average mu and sigma
mu_normal = torch.from_numpy(np.array([-0.0195, -0.0150,  0.0929, -0.0248, -0.0667,  0.1013, -0.0445, -0.0642,
        -0.0300,  0.0416, -0.0164, -0.0241, -0.1101, -0.0743, -0.1053, -0.0615,
        -0.0278,  0.0295, -0.0182,  0.0100])).type(torch.FloatTensor).cuda()
log_sigma_normal = torch.from_numpy(
            np.array([-1.8928, -1.6224, -2.4975, -1.4891, -1.7866, -3.4991, -2.5238, -2.0383,
        -2.0469, -2.7938, -1.6060, -3.0628, -2.7930, -3.3543, -3.0112, -2.2415,
        -2.8561, -2.6268, -1.9465, -4.1907])).type(torch.FloatTensor).cuda()
"""
mu_normal = torch.from_numpy(np.array([0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,
                                       0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0,  0.0, 0.0,  0.0])).type(torch.FloatTensor).cuda()
log_sigma_normal = torch.from_numpy(
            np.array([0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,
                                       0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0,  0.0, 0.0,  0.0])).type(torch.FloatTensor).cuda()

_, _, _, z = vae_model(test_x)
dist = torch.mean(MD_torch(z, mu_normal, log_sigma_normal))
print(dist)

_, _, _, z_adv = vae_model(torch.from_numpy(cnn_adv_xs_arr).cuda())
adv_dist = torch.mean(MD_torch(z_adv, mu_normal, log_sigma_normal))
print(adv_dist)
