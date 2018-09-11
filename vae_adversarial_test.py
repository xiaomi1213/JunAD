import torch
import torchvision
import torch.utils.data as Data

import foolbox
import numpy as np


test_data = torchvision.datasets.MNIST(
    root='/home/junhang/Projects/DataSet/MNIST',
    train=False
)
print(test_data.test_data.size(), test_data.test_labels.size())

vae_model = torch.load('/home/junhang/Projects/Scripts/saved_model/vae.pkl').eval()

cnn_model = torch.load('/home/junhang/Projects/Scripts/saved_model/cnn.pkl').eval()

adv_test_data = torch.unsqueeze(test_data.test_data, 1).type(torch.FloatTensor)/255.0
adv_test_labels = test_data.test_labels
print(adv_test_data.shape, adv_test_labels.shape)

mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
fmodel = foolbox.models.PyTorchModel(
    cnn_model, bounds=(0, 1), num_classes=10, preprocessing=(0, 1))
attack = foolbox.attacks.FGSM(fmodel)
adversarial = attack(adv_test_data.data[0], adv_test_labels.data[0])





