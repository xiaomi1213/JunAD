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


torch.manual_seed(1)

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False
num_train = 6000
num_test = 1000

# Import dataset
#if not(os.path.exists(r'E:\Bluedon\2DataSet\mnist')) or not os.listdir(r'E:\Bluedon\2DataSet\mnist'):
 #   DOWNLOAD_MNIST = True

mnist_train = torchvision.datasets.MNIST(
    root=r'E:\Bluedon\2DataSet\mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
train_x = mnist_train.train_data[:num_train]
train_y = mnist_train.train_labels[:num_train]
print(train_x.size())
print(train_y.size())
#show_a_image(mnist_train.train_data[0].numpy())

train_loader = Data.DataLoader(dataset=mnist_train, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root=r'E:\Bluedon\2DataSet\mnist',train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)/255.
test_y = test_data.test_labels
test_x = test_x[:num_test]
test_y = test_y[:num_test]


"""

# build the cnn model
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

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x

# train the cnn model
cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

loss_func = nn.CrossEntropyLoss()

print('-------------CNN training--------------------')
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print('epoch %d - step %d - CNN training loss: %.3f' % (epoch, step, loss.data.numpy()))

# evaluate the cnn model
test_output, last_layer = cnn(test_x)
pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
cnn_accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum())/float(test_y.size(0))
print('CNN accuracy: %.3f' % cnn_accuracy)
"""

# train and evaluate svm
svm_train_x = train_x.data.numpy()
svm_train_x = np.reshape(svm_train_x,[num_train,28*28])
svm_train_y = train_y.data.numpy()
#svm_train_y = svm_train_y[:np.newaxis]
svm_test_x = test_x.data.numpy()
svm_test_x = np.reshape(svm_test_x,[num_test,28*28])
svm_test_y = test_y.data.numpy()
print('-------------svm training------------------')
clf = svm.SVC(C=1.0)
clf.fit(svm_train_x, svm_train_y)
svm_preds =[]
for i in range(num_test):
    svm_pred = clf.predict([svm_test_x[i]])
    svm_preds.append(svm_pred)
svm_preds = np.array(svm_preds)
print(svm_preds.shape)
print(svm_preds)
accu = np.equal(np.squeeze(svm_preds),test_y)
print(accu.shape)
print(accu)
svm_accuracy = np.mean(accu)
print('SVM accuracy: %.3f' % svm_accuracy)


# build and train CNN+SVM
class CNN_SVM(nn.Module):
    def __init__(self):
        super(CNN_SVM, self).__init__()
        self.conv1_svm = nn.Sequential(
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
        self.conv2_svm = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out_svm = nn.Linear(32 * 7 * 7, 10)

    def forward(self,x):
        x = self.conv1_svm(x)
        x = self.conv2_svm(x)
        x = x.view(x.size(0), -1)
        output = self.out_svm(x)
        return output, x

cnn_svm = CNN_SVM()
optimizer_svm = torch.optim.Adam(cnn_svm.parameters(), lr=LR)
loss_svm = svm_l1loss()
#loss_func = svm_l2loss()

print('-------------CNN_SVM training--------------------')
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn_svm(b_x)[0]
        loss_svm = loss_svm(output, b_y)
        optimizer_svm.zero_grad()
        loss_svm.backward()
        optimizer_svm.step()

        if step % 50 == 0:
            print('epoch %d - step %d - CNN_SVM training loss: %.3f' % (epoch, step, loss_svm.data.numpy()))

# evaluate the cnn_svm model
test_output, last_layer = cnn_svm(test_x)
cnn_svm_pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
cnn_svm_accuracy = float((cnn_svm_pred_y == test_y.data.numpy()).astype(int).sum())/float(test_y.size(0))
print('CNN_SVM accuracy: %.3f' % cnn_svm_accuracy)



# generate adv_x with CNN

mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
cnn_model = foolbox.models.PyTorchModel(cnn, bounds=(0, 1), num_classes=10, preprocessing=(mean, std))
cnn_attack = foolbox.attacks.FGSM(cnn_model)
cnn_adv_x = cnn_attack(test_x, test_y)

# evaluate the cnn model with cnn_adv_x
test_output, last_layer = cnn(cnn_adv_x)
pred_adv_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
cnn_adv_accuracy = float((pred_adv_y == test_y.data.numpy()).astype(int).sum())/float(test_y.size(0))
print('CNN_adv accuracy: %.3f' % cnn_adv_accuracy)

# evaluate svm with cnn_adv_x
svm_x_adv = np.reshape(cnn_adv_x, [1000,28*28])
svm_adv_preds =[]
for i in range(1000):
    svm_pred = clf.predict([svm_x_adv[i]])
    svm_adv_preds.append(svm_pred)
svm_adv_preds = np.array(svm_adv_preds)
#print(svm_preds[0].shape)
#print(svm_preds[0])
#print(svm_preds.shape)
svm_adv_accu = np.equal(np.squeeze(svm_adv_preds),np.argmax(svm_test_y,1))
#print(accu)
#print(accu.shape)
svm_adv_accuracy = np.mean(svm_adv_accu)
print('SVM with cnn_adv_x accuracy: ',svm_adv_accuracy)


# generate adv_x with CNN+SVM
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
cnn_svm_model = foolbox.models.PyTorchModel(cnn_svm, bounds=(0, 1), num_classes=10, preprocessing=(mean, std))
cnn_svm_attack = foolbox.attacks.FGSM(cnn_svm_model)
cnn_svm_adv_x = cnn_svm_attack(test_x, test_y)

# evaluate the cnn_svm model with cnn_svm_adv_x
test_output, last_layer = cnn_svm(cnn_svm_adv_x)
pred_svm_adv_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
cnn_svm_adv_accuracy = float((pred_svm_adv_y == test_y.data.numpy()).astype(int).sum())/float(test_y.size(0))
print('CNN_SVM_adv with cnn_svm_adv_x accuracy: %.3f' % cnn_svm_adv_accuracy)


# evaluate the cnn_svm model with cnn_adv_x
test_output, last_layer = cnn_svm(cnn_adv_x)
pred_svm_adv_y2 = torch.max(test_output, 1)[1].data.squeeze().numpy()
cnn_svm_adv_accuracy2 = float((pred_svm_adv_y2 == test_y.data.numpy()).astype(int).sum())/float(test_y.size(0))
print('CNN_SVM_adv with cnn_adv_x accuracy: %.3f' % cnn_svm_adv_accuracy2)

print('\n-------------------------------------all accuracy--------------------------------------------')
print('CNN accuracy: %.3f' % cnn_accuracy)
print('SVM accuracy: %.3f' % svm_accuracy)
print('CNN_SVM accuracy: %.3f' % cnn_svm_accuracy)
print('CNN_adv accuracy: %.3f' % cnn_adv_accuracy)
print('SVM with cnn_adv_x accuracy: ',svm_adv_accuracy)
print('CNN_SVM_adv with cnn_svm_adv_x accuracy: %.3f' % cnn_svm_adv_accuracy)
print('CNN_SVM_adv with cnn_adv_x accuracy: %.3f' % cnn_svm_adv_accuracy2)



