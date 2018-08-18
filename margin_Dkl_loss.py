import numpy as np
import torch


def Dkl(preds, y):
    a = torch.zeros([1], dtype=torch.float32)
    b = a+0.1
    #return torch.sum((preds * torch.log(preds/(y+b))), dtype=torch.long)
    #c = torch.sum((preds * torch.log(preds / (y + b))), dtype=torch.long).float()
    c = torch.sum((y * torch.log(y / preds)), dtype=torch.float32)
    return c
    #c = np.sum(preds.data.numpy() * np.log(preds.data.numpy()/(y.data.numpy() + b.data.numpy())))
    #return torch.from_numpy(c)


def Margin_Dkl(preds, y, gama):
    gama += torch.zeros([1], dtype=torch.float32)
    total_loss = torch.zeros([1], dtype=torch.float32)

    for i in range(y.size()[0]):
        if torch.argmax(y[i]) != torch.argmax(preds[i]):
            loss = gama + Dkl(preds[i], y[i])
        else:
            if gama < Dkl(preds[i], y[i]):
                loss = gama - Dkl(preds[i], y[i])
            else:
                loss = 0
        total_loss += loss
    return total_loss



preds = np.array([[0.1, 0.2, 0.6, 0.05, 0.05],[0.2, 0.5, 0.1, 0.1, 0.1]], np.float32)
y = np.array([[0.0, 0.0, 0.0, 1.0, 0.0],[0.0, 1.0, 0.0, 0.0, 0.0]], np.float32)

print(torch.argmax(torch.from_numpy(y),1).data.numpy() )

#print(torch.from_numpy(preds), torch.from_numpy(y))
print(Margin_Dkl(torch.from_numpy(preds), torch.from_numpy(y), gama=1.0))

#y = torch.from_numpy(y)
#gama = torch.zeros(y.size(), dtype=torch.float32)
#gama = 2.0 + gama
#print(gama)