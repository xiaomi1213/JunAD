import torch
import torchvision
import foolbox
import numpy as np

#1 load model
cnn_model = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP3/cnn.pkl').eval()
beta_vae_h = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP3/beta_vae_h.pkl').eval()
rev_beta_vae = torch.load('/home/junhang/Projects/Scripts/saved_model/EXP3/rev_beta_vae.pkl').eval()

#2 load data
num_test = 4
test_data = torchvision.datasets.MNIST(
    root='/home/junhang/Projects/DataSet/MNIST',
    train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)/255.
test_x = test_x[:num_test].cuda()
test_y = test_data.test_labels
test_y = test_y[:num_test].cuda()

correct_count = 0
generate_count = 0
reduced_num_test = list(range(num_test))
print("-------------------------Looping-----------------------------")
for i in range(num_test):
    #3 select a image randomly and not redunperly

    choice_num = np.random.choice(reduced_num_test, 1).squeeze()
    reduced_num_test.remove(choice_num)
    single_x = test_x[choice_num].unsqueeze(0)
    single_y = test_y[choice_num].unsqueeze(0)


    #4 classify and the image verify the result
    cnn_test_output = cnn_model(single_x)
    pred_y = torch.max(cnn_test_output, 1)[1].data.squeeze().cpu().numpy()
    if (pred_y != single_y.data.squeeze().cpu().numpy()):
        continue#go back to step 3 to select another image
    else:
        #5 generate adversarial examples
        # mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        # std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        fmodel = foolbox.models.PyTorchModel(
            cnn_model, bounds=(0, 1), num_classes=10, preprocessing=(0, 1))
        attack = foolbox.attacks.FGSM(fmodel)

        single_x_cpu = single_x.cpu().numpy().squeeze(0)
        single_y_cpu = single_y.cpu().numpy().squeeze(0)

        single_adv_x = attack(single_x_cpu, single_y_cpu)# generating an adversarial example
        if single_adv_x is None:
            continue
        generate_count += 1

    #6 reverse the adversarial example with VAE
    rev_beta_vae_x, _, _ = rev_beta_vae(torch.from_numpy(single_adv_x).unsqueeze(0).cuda())

    #7 classify the reversed image and verify the result
    test_output = cnn_model(rev_beta_vae_x)
    pred_rev_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
    if (pred_rev_y == single_y_cpu):
        correct_count += 1
        print('image %d, correct!' % choice_num)
    else:
        print('image %d, wrong!' % choice_num)

#8 calculate the accuracy
accuracy = correct_count / float(generate_count)
print('have generated adversarial examples: %d' % generate_count)
print('reverse_beta_vae+CNN_adv accuracy: %.4f' % accuracy)




