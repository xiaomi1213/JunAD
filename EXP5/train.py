import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.autograd import Variable

def train_cnn(model, train_loader, device, num_epoch=2, lr=1e-3):
    model.train()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        train_loss = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_func(preds, batch_y)
            loss.backward()
            train_loss = loss.item()
            optimizer.step()
            if step%50 == 0:
                print("epoch {} - step {} - step_loss: {:.4f}".format(
                    epoch, step, train_loss
                ))
def train_ae(model, train_loader, device, num_epoch=2, lr=1e-3):
    model.train()
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        train_loss = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            #batch_x = batch_x.view(-1, 784)
            optimizer.zero_grad()
            _,preds = model(batch_x)
            loss = loss_func(preds, batch_x)
            loss.backward()
            train_loss = loss.item()
            optimizer.step()
            if step % 50 == 0:
                print("epoch {} - step {} - step_loss: {:.4f}".format(
                    epoch, step, train_loss))
def train_vae(model, train_loader, device, num_epoch=2, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    train_loss = 0

    for epoch in range(num_epoch):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / len(data)))




if __name__ == "__main__":
    import torch
    import torch.utils.data as Data
    import torchvision
    from EXP5.base_cnn import CNN
    from EXP5.ae_model import AutoEncoder
    from EXP5.vae_model import VAE



    train_data = torchvision.datasets.MNIST(
        root='/home/junhang/Projects/DataSet/MNIST',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False
    )
    train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    print("--------------------CNN training--------------------------------")
    cnn = CNN()
    cnn = cnn.to(device)
    train_cnn(cnn, train_loader, device, num_epoch=2)
    torch.save(cnn, '/home/junhang/Projects/Scripts/saved_model/EXP5/cnn.pkl')

    print("--------------------AE training--------------------------------")
    ae = AutoEncoder()
    ae = ae.to(device)
    train_ae(ae, train_loader, device, num_epoch=10, lr=1e-3)
    torch.save(ae, '/home/junhang/Projects/Scripts/saved_model/EXP5/ae.pkl')

    print("--------------------VAE training--------------------------------")
    vae = VAE()
    vae = vae.to(device)
    train_vae(vae, train_loader, device, num_epoch=10)
    torch.save(vae, '/home/junhang/Projects/Scripts/saved_model/EXP5/vae.pkl')

    





