import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data




def vae_loss(recon_x, x, mu, log_sigma):
    Recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    Dkl = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())

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
            train_loss += loss.item()
            optimizer.step()
            if step%5 == 0:
                print("epoch {} - step {} - step_loss: {:.3f}".format(
                    epoch, step, train_loss/step
                ))









