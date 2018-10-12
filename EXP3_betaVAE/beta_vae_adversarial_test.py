import torch
import torchvision
import torch.utils.data as Data

import foolbox
import numpy as np

num_test = 10000

print("-------------------------loading data-----------------------------")
test_data = torchvision.datasets.MNIST(
    root='/home/junhang/Projects/DataSet/MNIST',
    train=False
)


print(test_data.test_data.size(), test_data.test_labels.size())
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)/255.
test_x = test_x[:num_test].cuda()
test_y = test_data.test_labels
test_y = test_y[:num_test].cuda()

print("\n-------------------------loading models-----------------------------\n")
vae_model = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP3/vae.pkl').eval()
cnn_model = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP3/cnn.pkl').eval()
ae_model = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP3/ae.pkl').eval()
beta_vae_h_6 = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP3/beta_vae_h_100.pkl').eval()
beta_vae_b = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP3/beta_vae_b.pkl').eval()

# evaluate the cnn model
print("-------------------------evaluating cnn model-----------------------------")
cnn_test_output = cnn_model(test_x)
pred_y = torch.max(cnn_test_output, 1)[1].data.squeeze().cpu().numpy()
cnn_accuracy = float((pred_y == test_y.data.cpu().numpy()).astype(int).sum())/float(test_y.size(0))
print('CNN accuracy: %.4f' % cnn_accuracy)

print("-------------------------evaluate ae+cnn with normal-----------------------------")
_,ae_x = ae_model(test_x.view(-1, 784).cuda())
test_output = cnn_model(ae_x.view(-1, 1, 28, 28))
ae_cnn_pred_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
ae_cnn_accuracy = float((ae_cnn_pred_y == test_y.data.cpu().numpy()).astype(int).sum())/float(test_y.shape[0])
print('AE+CNN accuracy: %.4f' % ae_cnn_accuracy)

print("-------------------------evaluate vae+cnn with adv-----------------------------")
vae_x,_,_,_ = vae_model(test_x.view(-1, 784).cuda())
test_output = cnn_model(vae_x.view(-1, 1, 28, 28))
vae_cnn_pred_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
vae_cnn_accuracy = float((vae_cnn_pred_y == test_y.data.cpu().numpy()).astype(int).sum())/float(test_y.shape[0])
print('VAE+CNN accuracy: %.4f' % vae_cnn_accuracy)

print("-------------------------evaluate beta_vae_h_6+cnn with normal-----------------------------")
vae_x,_,_ = beta_vae_h_6(test_x.cuda())
test_output = cnn_model(vae_x)
vae_cnn_pred_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
vae_cnn_accuracy = float((vae_cnn_pred_y == test_y.data.cpu().numpy()).astype(int).sum())/float(test_y.shape[0])
print('BETA_VAE_H_6+CNN accuracy: %.4f' % vae_cnn_accuracy)

print("-------------------------evaluate beta_vae_b+cnn with normal-----------------------------")
vae_x,_,_ = beta_vae_b(test_x.cuda())
test_output = cnn_model(vae_x)
vae_cnn_pred_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
vae_cnn_accuracy = float((vae_cnn_pred_y == test_y.data.cpu().numpy()).astype(int).sum())/float(test_y.shape[0])
print('BETA_VAE_b+CNN accuracy: %.4f' % vae_cnn_accuracy)


# select the correctly classified samples indices
print("\n-------------------------selecting samples-----------------------------\n")
a = (pred_y == test_y.data.cpu().numpy()).astype(int)
correct_indice = []
for i in range(num_test):
    if a[i] == 1:
        correct_indice.append(i)

#cnn_adv_test_x = test_x.cpu().data.numpy()[correct_indice]
#cnn_adv_test_y = test_y.cpu().data.numpy()[correct_indice]

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
"""
for i in range(len(correct_indice)):
    cnn_adv_x = attack(cnn_adv_test_x[i],cnn_adv_test_y[i])
    if cnn_adv_x is None:
        continue
    cnn_adv_xs.append(cnn_adv_x)
"""
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

print("-------------------------evaluate ae+cnn with adv-----------------------------")
_,ae_x = ae_model(torch.from_numpy(cnn_adv_xs_arr).view(-1, 784).cuda())
test_output = cnn_model(ae_x.view(-1, 1, 28, 28))
ae_cnn_pred_adv_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
ae_cnn_adv_accuracy = float((ae_cnn_pred_adv_y == cnn_adv_ys_arr).astype(int).sum())/float(cnn_adv_ys_arr.shape[0])
print('AE+CNN_adv accuracy: %.4f' % ae_cnn_adv_accuracy)

print("-------------------------evaluate vae+cnn with adv-----------------------------")
vae_x,_,_,_ = vae_model(torch.from_numpy(cnn_adv_xs_arr).cuda())
test_output = cnn_model(vae_x.view(-1, 1, 28, 28))
vae_cnn_pred_adv_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
vae_cnn_adv_accuracy = float((vae_cnn_pred_adv_y == cnn_adv_ys_arr).astype(int).sum())/float(cnn_adv_ys_arr.shape[0])
print('VAE+CNN_adv accuracy: %.4f' % vae_cnn_adv_accuracy)


print("-------------------------evaluate beta_vae_h_6+cnn with normal-----------------------------")
beta_vae_h_x,_,_ = beta_vae_h_6(torch.from_numpy(cnn_adv_xs_arr).cuda())
test_output = cnn_model(beta_vae_h_x)
beta_vae_h_cnn_pred_adv_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
beta_vae_h_cnn_accuracy = float((beta_vae_h_cnn_pred_adv_y == cnn_adv_ys_arr).astype(int).sum())/float(test_y.shape[0])
print('BETA_VAE_H_6+CNN_adv accuracy: %.4f' % beta_vae_h_cnn_accuracy)

print("-------------------------evaluate beta_vae_b+cnn with normal-----------------------------")
beta_vae_b_x,_,_ = beta_vae_b(torch.from_numpy(cnn_adv_xs_arr).cuda())
test_output = cnn_model(beta_vae_b_x)
beta_vae_b_cnn_pred_adv_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
beta_vae_b_cnn_accuracy = float((beta_vae_b_cnn_pred_adv_y == cnn_adv_ys_arr).astype(int).sum())/float(test_y.shape[0])
print('BETA_VAE_b+CNN_adv accuracy: %.4f' % beta_vae_b_cnn_accuracy)



