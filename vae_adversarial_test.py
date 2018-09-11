import torch
import torchvision
import torch.utils.data as Data

import foolbox
import numpy as np

EPOCH = 1
BATCH_SIZE = 100
LR = 0.001
DOWNLOAD_MNIST = False
num_train = 6000
num_test = 1000


test_data = torchvision.datasets.MNIST(
    root='/home/junhang/Projects/DataSet/MNIST',
    train=False
)
print(test_data.test_data.size(), test_data.test_labels.size())
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)/255.
test_y = test_data.test_labels


vae_model = torch.load('/home/junhang/Projects/Scripts/saved_model/vae.pkl').eval()
cnn_model = torch.load('/home/junhang/Projects/Scripts/saved_model/cnn.pkl').eval()


# evaluate the cnn model
cnn_test_output = cnn_model(test_x)
pred_y = torch.max(cnn_test_output, 1)[1].data.squeeze().numpy()
cnn_accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum())/float(test_y.size(0))
print('CNN accuracy: %.3f' % cnn_accuracy)

# select the correctly classified samples indices
a = (pred_y == test_y.data.numpy()).astype(int)
correct_indice = []
for i in range(num_test):
    if a[i] == 1:
        correct_indice.append(i)

cnn_adv_test_x = test_x.data.numpy()[correct_indice]
cnn_adv_test_y = test_y.data.numpy()[correct_indice]

#mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
#std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
fmodel = foolbox.models.PyTorchModel(
    cnn_model, bounds=(0, 1), num_classes=10, preprocessing=(0, 1))
attack = foolbox.attacks.FGSM(fmodel)

cnn_adv_xs = []
for i in range(len(correct_indice)):
    cnn_adv_x = attack(cnn_adv_test_x[i],cnn_adv_test_y[i])
    cnn_adv_xs.append(cnn_adv_x)

cnn_adv_xs_arr = np.array(cnn_adv_xs)
print(cnn_adv_xs_arr.shape)





