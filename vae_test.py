import torch
import torch.utils.data as Data
import torchvision

from vae_model import VAE
from train import train_vae, vae_loss

train_data = torchvision.datasets.MNIST(
    root='/home/junhang/Projects/DataSet/MNIST',
    train=True,
    transform = torchvision.transforms.ToTensor(),
    download=False
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE()

model = model.to(device)

train_vae(model, train_loader, vae_loss, device, num_epoch=20)

torch.save(model, '/home/junhang/Projects/Scripts/saved_model/vae.pkl')

