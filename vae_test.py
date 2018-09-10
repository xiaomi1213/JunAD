import torch
import torch.utils.data as Data
import torchvision

from vae_model import VAE
from train import train_vae, vae_loss

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform = torchvision.transforms.ToTensor(),
    download=True
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=50, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE()

train_vae(model, train_loader, vae_loss, device, num_epoch=2)



