import torch
import torchvision
import torch.utils.data as Data

test_data = torchvision.datasets.MNIST(
    root='/home/junhang/Projects/DataSet/MNIST',
    train=False
)
#print(test_data.test_data.size(), test_data.test_labels.size())

#vae_model = torch.load('/home/junhang/Projects/Scripts/saved_model/vae.pkl')



