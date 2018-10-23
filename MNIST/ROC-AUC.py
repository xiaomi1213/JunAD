import torch
import torchvision
import foolbox
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle

# load data and model
num_test = 10000
test_data = torchvision.datasets.MNIST(
    root='/home/junhang/Projects/DataSet/MNIST',
    train=False
)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)/255.
test_x = test_x[:num_test].cuda()
test_y = test_data.test_labels
test_y = test_y[:num_test].cuda()


cnn_model = torch.load('/home/junhang/Projects/Scripts/saved_model/MNIST/cnn.pkl').eval()
rev_vae = torch.load('/home/junhang/Projects/Scripts/saved_model/MNIST/rev_vae.pkl').eval()

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


print("-------------------------evaluate cnn with adv-----------------------------")
# evaluate the cnn model with cnn_adv_x
test_output = cnn_model(torch.from_numpy(cnn_adv_xs_arr).cuda())
cnn_pred_adv_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
cnn_adv_accuracy = float((cnn_pred_adv_y == cnn_adv_ys_arr).astype(int).sum())/float(cnn_adv_ys_arr.shape[0])
print('CNN_adv accuracy: %.4f' % cnn_adv_accuracy)


print("-------------------------evaluate vae+cnn with adv-----------------------------")
vae_x,_,_ = rev_vae(torch.from_numpy(cnn_adv_xs_arr).cuda())
test_output = cnn_model(vae_x.view(-1, 1, 28, 28))
vae_cnn_pred_adv_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
vae_cnn_adv_accuracy = float((vae_cnn_pred_adv_y == cnn_adv_ys_arr).astype(int).sum())/float(cnn_adv_ys_arr.shape[0])
print('VAE+CNN_adv accuracy: %.4f' % vae_cnn_adv_accuracy)


print("-------------------------detect normal and adversarial samples with VAE-----------------------------")
# define the label of adversarial example is 1, normal is 0
ones_test_y = torch.ones(test_y.size())
label_normal_y = np.dstack((test_y, ones_test_y)).squeeze()
zeros_adv_y = torch.zeros(cnn_adv_ys_arr.shape)
label_adv_y = np.dstack((cnn_adv_ys_arr, zeros_adv_y)).squeeze()
#print(label_adv_y)



# mix normal and adversarial examples into one dataset
mix_dataset_x = np.concatenate((test_x, cnn_adv_xs_arr),axis=0)
mix_dataset_y = np.concatenate((label_normal_y, label_adv_y),axis=0)
mix_dataset_x, mix_dataset_y = shuffle(mix_dataset_x, mix_dataset_y, random_state=0)


# define KL divergence function
def KL_divergence(logvar, mu):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),1)
    return KLD

# Batch inputs condition efficient to calculate AUC-ROC

x = torch.from_numpy(mix_dataset_x).cuda()
y = mix_dataset_y[:,1]
_, mu, log_sigma = rev_vae(x)
Dkl = KL_divergence(log_sigma, mu).data.cpu().numpy()
fpr, tpr, threshold = roc_curve(y, Dkl)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Single input condition with difficulty to compute the AUC-ROC
"""
# detect normal examples
Boundaries = np.linspace(30, 80, 100)
TPRs = []
FPRs = []
for B in Boundaries:
    normal_count = 0
    adv_count = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    correct_count = 0
    for idx, data in enumerate(list(zip(mix_dataset_x, mix_dataset_y))):
        x = torch.from_numpy(data[0]).unsqueeze(0).cuda()
        y = data[1]
        _, mu, log_sigma = rev_vae(x)
        Dkl = KL_divergence(log_sigma, mu)
        if Dkl >= B:
            print('The input is normal')
            normal_count += 1
            if y[1] == 0:
                TN += 1
            else:
                FN += 1
        else:
            print('The input is adversarial')
            adv_count += 1
            if y[1] == 1:
                TP += 1
            else:
                FP += 1
            print('Reversing the adversarial examples')
            reversed_samples, _, _ = rev_vae(x)
            test_output = cnn_model(reversed_samples.view(-1,1,28,28))
            pred_rev_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
            if (pred_rev_y == y[0]):
                correct_count += 1
                print('Classify reversed image %d, correct!' % idx)
            else:
                print('Classify reversed image %d, wrong!' % idx)
    TPR = TP/float(TP+FN)
    TPRs.append(TPR)
    FPR = FP/float(FP+TN)
    FPRs.append(FPR)
"""
"""
plt.title('Receiver Operating Characteristic')
plt.plot(FPRs, TPRs, 'b', label = 'AUC' )#= %0.2f' % roc_auc
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
"""
