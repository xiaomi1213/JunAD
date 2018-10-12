import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


def train_beta_vae_h(model, train_loader, device, num_epoch=2, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(recon_x, x, mu, logvar, beta):
        BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        beta_vae_loss = BCE + beta * KLD
        return beta_vae_loss

    train_loss = 0

    for epoch in range(num_epoch):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, beta=10)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

def train_beta_vae_b(model, train_loader, device, C_max, C_stop_iter, gamma, num_epoch=2, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(recon_x, x, mu, logvar, C_max, C_stop_iter, global_iter, gamma):
        C_max = torch.tensor(C_max, dtype=torch.float32).to(device)
        C_stop_iter = torch.tensor(C_stop_iter, dtype=torch.float32).to(device)
        global_iter = torch.tensor(global_iter, dtype=torch.float32).to(device)
        #batch_size = x.size(0)
        BCE = F.binary_cross_entropy_with_logits(recon_x, x, size_average=False)
        #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        C = torch.clamp(C_max / C_stop_iter * global_iter, 0, C_max)
        beta_vae_loss = BCE + gamma * (KLD - C).abs()
        return beta_vae_loss

    train_loss = 0

    for epoch in range(num_epoch):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, C_max=C_max, C_stop_iter=C_stop_iter, global_iter=epoch, gamma=gamma)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))


if __name__ == "__main__":
    import torch
    import torch.utils.data as Data
    import torchvision
    from EXP3_betaVAE.beta_vae_model import BetaVAE

    train_data = torchvision.datasets.MNIST(
        root='/home/junhang/Projects/DataSet/MNIST',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False
    )
    train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    
    print("--------------------BETA_VAE_H training--------------------------------")
    beta_vae_h = BetaVAE()
    beta_vae_h = beta_vae_h.to(device)
    train_beta_vae_h(beta_vae_h, train_loader, device, num_epoch=10)
    torch.save(beta_vae_h, '/home/junhang/Projects/Scripts/saved_model/beta_vae_h.pkl')
    
    """
    print("--------------------BETA_VAE_B training--------------------------------")
    beta_vae_b = BetaVAE()
    beta_vae_b = beta_vae_b.to(device)
    train_beta_vae_b(beta_vae_b, train_loader, device, C_max=25, C_stop_iter=10, gamma=1000, num_epoch=10)
    torch.save(beta_vae_b, '/home/junhang/Projects/Scripts/saved_model/beta_vae_b.pkl')


