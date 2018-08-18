import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import os
from show_image import show_a_image
from svm_loss_test import svm_l1loss, svm_l2loss
from sklearn import svm
import numpy as np
import foolbox

from margin_Dkl_loss import Margin_Dkl

torch.manual_seed(1)

GAMA = 1.0
EPOCH = 1
LR = 0.001
BATCH_SIZE = 100
num_train = 6000
num_test = 1000

mnist_train = torchvision.datasets.MNIST(
    root=r'E:\Bluedon\2DataSet\mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

train_loader = Data.DataLoader(dataset=mnist_train, batch_size=BATCH_SIZE, shuffle=True)

train_x = mnist_train.train_data[:num_train]
train_y = mnist_train.train_labels[:num_train]

test_data = torchvision.datasets.MNIST(root=r'E:\Bluedon\2DataSet\mnist',train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)/255.
test_y = test_data.test_labels
test_x = test_x[:num_test]
test_y = test_y[:num_test]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
        self.probs = nn.Softmax()

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        probility = self.probs(output)
        return probility

#pretrained_cnn = torch.load(r'E:\Bluedon\3Code\pytorch_models\cnn.pkl')

cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
#loss_func = Margin_Dkl(gama=GAMA)

print('-------------CNN training--------------------')
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = Margin_Dkl(output, b_y, gama=GAMA)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print('epoch %d - step %d - CNN training loss: %.3f' % (epoch, step, loss.data.numpy()))

# save the cnn model
#torch.save(cnn,r'E:\Bluedon\3Code\pytorch_models\cnn.pkl')

cnn_test_x = test_x
cnn_test_y = test_y
cnn_test_output = cnn(cnn_test_x)
pred_y = torch.max(cnn_test_output, 1)[1].data.squeeze().numpy()
cnn_accuracy = float((pred_y == cnn_test_y.data.numpy()).astype(int).sum())/float(cnn_test_y.size(0))
print('CNN accuracy: %.3f' % cnn_accuracy)