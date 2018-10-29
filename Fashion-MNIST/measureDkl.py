import torch
import torchvision
import foolbox
import numpy as np
import matplotlib.pyplot as plt

num_test = 10

def normalize(tensor, mean, std):
# Site from https://pytorch.org/docs/0.2.0/_modules/torchvision/transforms.html#Normalize
    for sample in tensor:
        for c, m, s, in zip(sample, mean, std):
            c.sub_(m).div_(s)
    return tensor

print("-------------------------loading data-----------------------------")
test_data = torchvision.datasets.FashionMNIST(
        root='/home/junhang/Projects/DataSet/FashionMNIST',
        train='False',
        download=False
    )

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
test_x = torch.FloatTensor(test_data.test_data.transpose((0, 3, 1, 2)))/255.
test_x = normalize(test_x, mean, std)
test_y = torch.LongTensor(test_data.test_labels)

test_x = test_x[:num_test].cuda()
test_y = test_y[:num_test].cuda()
#print(test_x[0])




print("\n-------------------------loading models-----------------------------\n")
vae_model = torch.load('/home/junhang/Projects/Scripts/saved_model/FashionMNIST/vae.pkl').eval()
densenet = torch.load('/home/junhang/Projects/Scripts/saved_model/FashionMNIST/densenet.pkl').eval()



# evaluate the cnn model
print("-------------------------evaluating cnn model-----------------------------")
cnn_test_output = densenet(test_x)
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
    densenet, bounds=(-1, 1), num_classes=10, preprocessing=(0, 1))
attack = foolbox.attacks.FGSM(fmodel)
#attack = foolbox.attacks.BIM(fmodel)
#attack = foolbox.attacks.DeepFoolAttack(fmodel)
#attack = foolbox.attacks.SaliencyMapAttack(fmodel)
#attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel)

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


# beta vae h model
# mu and sigma of normal examples
_, mu, log_sigma = vae_model(test_x)
# meausre Dkl between N(0, I) and normal examples
score_normal_5 = KL_divergence(log_sigma, mu)
print(score_normal_5)
# mu and sigma of adversarial examples
_, mu_adv, log_sigma_adv = vae_model(torch.from_numpy(cnn_adv_xs_arr).cuda())
# meausre Dkl between N(0, I) and adv examples
score_adv_5 = KL_divergence(log_sigma_adv, mu_adv)
print(score_adv_5)

plt.hist(score_normal_5.data.cpu().numpy(), bins=100, alpha=0.5, label='normal')
plt.hist(score_adv_5.data.cpu().numpy(), bins=100, alpha=0.5, label='adv')
plt.legend(loc='upper right')
plt.title('REV_VAE')
plt.show()

