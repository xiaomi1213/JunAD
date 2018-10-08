import torch
import torchvision
import foolbox
import numpy as np


# define normal distribution
def standard_Gassian():
    pass

def z_distribution():
    pass


# load data and model
num_test = 100
test_data = torchvision.datasets.MNIST(
    root='/home/junhang/Projects/DataSet/MNIST',
    train=False
)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)/255.
test_x = test_x[:num_test].cuda()
test_y = test_data.test_labels
test_y = test_y[:num_test].cuda()
"""
num_training = 50000
training_data = torchvision.datasets.MNIST(
    root='/home/junhang/Projects/DataSet/MNIST',
    train=True
)
training_x = torch.unsqueeze(training_data.training_data, dim=1).type(torch.FloatTensor)/255.
training_x = training_x[:num_training].cuda()
training_y = training_data.training_labels
training_y = training_y[:num_training].cuda()
"""

cnn_model = torch.load('/home/junhang/Projects/Scripts/saved_model/cnn.pkl').eval()
vae_model = torch.load('/home/junhang/Projects/Scripts/saved_model/vae.pkl').eval()
rev_vae_model = torch.load('/home/junhang/Projects/Scripts/saved_model/rev_vae.pkl').eval()
rev_l2_vae_model = torch.load('/home/junhang/Projects/Scripts/saved_model/rev_l2_vae.pkl').eval()
reg_vae_model = torch.load('/home/junhang/Projects/Scripts/saved_model/reg_vae.pkl').eval()

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
vae_x,_,_,_ = vae_model(torch.from_numpy(cnn_adv_xs_arr).cuda())
test_output = cnn_model(vae_x.view(-1, 1, 28, 28))
vae_cnn_pred_adv_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
vae_cnn_adv_accuracy = float((vae_cnn_pred_adv_y == cnn_adv_ys_arr).astype(int).sum())/float(cnn_adv_ys_arr.shape[0])
print('VAE+CNN_adv accuracy: %.4f' % vae_cnn_adv_accuracy)

"""



print("-------------------------evaluate rev_l2_vae+cnn with adv-----------------------------")
rev_l2_vae_x,_,_ = rev_l2_vae_model(torch.from_numpy(cnn_adv_xs_arr).cuda())
test_output = cnn_model(rev_l2_vae_x.view(-1, 1, 28, 28))
rev_l2_vae_cnn_pred_adv_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
rev_l2_vae_cnn_adv_accuracy = float((rev_l2_vae_cnn_pred_adv_y == cnn_adv_ys_arr).astype(int).sum())/float(cnn_adv_ys_arr.shape[0])
print('REVERSE_L2_VAE+CNN_adv accuracy: %.4f' % rev_l2_vae_cnn_adv_accuracy)


print("-------------------------evaluate reg_vae+cnn with adv-----------------------------")
reg_vae_x,_,_ = reg_vae_model(torch.from_numpy(cnn_adv_xs_arr).cuda())
reg_test_output = cnn_model(reg_vae_x.view(-1, 1, 28, 28))
reg_vae_cnn_pred_adv_y = torch.max(reg_test_output, 1)[1].data.squeeze().cpu().numpy()
reg_vae_cnn_adv_accuracy = float((reg_vae_cnn_pred_adv_y == cnn_adv_ys_arr).astype(int).sum())/float(cnn_adv_ys_arr.shape[0])
print('REVERSE_VAE+CNN_adv accuracy: %.4f' % reg_vae_cnn_adv_accuracy)



"""



print("-------------------------evaluate rev_vae+cnn with adv-----------------------------")
num = 1
rev_vae_x,_,_ = rev_vae_model(torch.from_numpy(cnn_adv_xs_arr[num:num+1]).cuda())
rev_test_output = cnn_model(rev_vae_x.view(-1, 1, 28, 28))
rev_vae_cnn_pred_adv_y = torch.max(rev_test_output, 1)[1].data.squeeze().cpu().numpy()
rev_vae_cnn_adv_accuracy = float((rev_vae_cnn_pred_adv_y == cnn_adv_ys_arr).astype(int).sum())/float(cnn_adv_ys_arr.shape[0])
print('REVERSE_VAE+CNN_adv accuracy: %.4f' % rev_vae_cnn_adv_accuracy)

