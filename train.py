import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data




def vae_loss(recon_x, x, mu, log_sigma):
    Recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784))
    Dkl = 0.5 * torch.sum(mu.pow(2) + log_sigma.exp() - 1.0 - log_sigma)

    return Recon_loss + Dkl


def train_vae(model, train_loader, loss_func, device, num_epoch=2, lr=1e-3):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        train_loss = 0
        for step, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_sigma = model(batch_x)
            loss = loss_func(recon_batch, batch_x, mu, log_sigma)
            loss.backward()
            train_loss = loss.item()
            optimizer.step()
            if step%10 == 0:
                print("epoch {} - step {} - step_loss: {:.3f}".format(
                    epoch, step, train_loss
                ))

def train_cnn(model, train_loader, device, num_epoch=2, lr=1e-3):
    model.train()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        train_loss = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_func(preds, batch_y)
            loss.backward()
            train_loss = loss.item()
            optimizer.step()
            if step%10 == 0:
                print("epoch {} - step {} - step_loss: {:.3f}".format(
                    epoch, step, train_loss
                ))








if __name__ == "__main__":
    import torch
    import torch.utils.data as Data
    import torchvision

    from vae_model import VAE
    from base_cnn import CNN
    #from train import train_vae, vae_loss

    train_data = torchvision.datasets.MNIST(
        root='/home/junhang/Projects/DataSet/MNIST',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False
    )
    train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """
    print("--------------------VAE training--------------------------------")
    vae = VAE()
    vae = vae.to(device)
    train_vae(vae, train_loader, vae_loss, device, num_epoch=20)
    torch.save(vae, '/home/junhang/Projects/Scripts/saved_model/vae.pkl')
    """
    print("--------------------CNN training--------------------------------")
    cnn = CNN()
    cnn = cnn.to(device)
    train_cnn(cnn, train_loader, device, num_epoch=20)
    torch.save(cnn, '/home/junhang/Projects/Scripts/saved_model/cnn.pkl')
