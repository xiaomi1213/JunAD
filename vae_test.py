import torch
import torchvision
import numpy as np

EPOCH = 1
BATCH_SIZE = 100
LR = 0.001
DOWNLOAD_MNIST = False
num_train = 6000
num_test = 1000

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


# evaluate the cnn model
cnn_test_x = test_x
cnn_test_y = test_y
cnn_test_output = pretrained_cnn(cnn_test_x)
pred_y = torch.max(cnn_test_output, 1)[1].data.squeeze().numpy()
cnn_accuracy = float((pred_y == cnn_test_y.data.numpy()).astype(int).sum())/float(cnn_test_y.size(0))
print('CNN accuracy: %.3f' % cnn_accuracy)

# select the correctly classified samples indices
a = (pred_y == cnn_test_y.data.numpy()).astype(int)
cnn_indice = []
for i in range(num_test):
    if a[i] == 1:
        cnn_indice.append(i)


# generate adv_x with CNN
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
cnn_model = foolbox.models.PyTorchModel(pretrained_cnn.eval(), bounds=(0, 1), num_classes=10, preprocessing=(0, 1))
cnn_attack = foolbox.attacks.FGSM(cnn_model)
cnn_adv_test_x = test_x.data.numpy()[cnn_indice]
cnn_adv_test_y = test_y.data.numpy()[cnn_indice]

cnn_adv_xs = []
for i in range(len(cnn_indice)):
    cnn_adv_x = cnn_attack(cnn_adv_test_x[i],cnn_adv_test_y[i])
    cnn_adv_xs.append(cnn_adv_x)

cnn_adv_xs_arr = np.array(cnn_adv_xs)
print(cnn_adv_xs_arr.shape)
#cnn_adv_xs = np.transpose(cnn_adv_xs,(0,2,3,1))
print(cnn_adv_xs_arr.shape)
show_a_image(np.squeeze(cnn_adv_xs[8], 0))

