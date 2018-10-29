import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time


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
def train_densenet(model, train_loader, device, num_epoch=30, lr=1e-1):
    # Site from here:https://github.com/andreasveit/densenet-pytorch/blob/master/train.py
    model.train()
    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=1e-4)

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
        lr = 1e-1 * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    class AverageMeter(object):
        """Computes and stores the average and current value"""

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    for epoch in range(0, num_epoch):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            target = target.cuda(async=True)
            input = input.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    loss=losses, top1=top1))
def train_ae(model, train_loader, device, num_epoch=2, lr=1e-3):
    model.train()
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        train_loss = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_x = batch_x
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
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(recon_x, x, mu, logvar, beta):
        BCE = F.binary_cross_entropy(recon_x,
                                 x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + beta*KLD

    train_loss = 0

    for epoch in range(num_epoch):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, beta=1)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / len(data)))
def train_reverse_vae(model, train_loader, device, num_epoch=2, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(recon_x, x, mu, logvar, beta):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta*KLD

    train_loss = 0

    for epoch in range(num_epoch):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, beta=1)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / len(data)))


class Config(object):
    def __init__(self,z_dim,c_dim,num_channels,std):
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.num_channels = num_channels
        self.std = std

if __name__ == "__main__":
    import torch
    import torch.utils.data as Data
    import torchvision
    import torchvision.transforms as transforms

    from CIFAR.vae_model import VAE
    from CIFAR.ae_model import AutoEncoder
    from CIFAR.reverse_vae import REVERSE_VAE
    from CIFAR.densenet import DenseNet3
    from CIFAR.base_cnn import CNN

    # normalize site from https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py#L28
    # and https://discuss.pytorch.org/t/normalization-of-cifar-and-mnist-datasets/829
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #normalize,
    ])
    train_data = torchvision.datasets.SVHN(
        root='/home/junhang/Projects/DataSet/SVHN',
        split='train',
        transform=transform_train,
        download=True,
    )
    #print(train_data.train_data.shape)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, **kwargs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """  """
    print("--------------------AE training--------------------------------")
    ae = AutoEncoder()
    ae = ae.to(device)
    train_ae(ae, train_loader, device, num_epoch=10, lr=1e-3)
    torch.save(ae, '/home/junhang/Projects/Scripts/saved_model/CIFAR/ae.pkl')
    
    print("--------------------REVERSE VAE training--------------------------------")
    rev_vae = REVERSE_VAE()
    rev_vae = rev_vae.to(device)
    train_reverse_vae(rev_vae, train_loader, device, num_epoch=10)
    torch.save(rev_vae, '/home/junhang/Projects/Scripts/saved_model/CIFAR/rev_vae.pkl')
    
    
    print("--------------------DenseNet training--------------------------------")
    densenet = DenseNet3(100, 10, 12, reduction=0.5,
                         bottleneck=True, dropRate=0)
    densenet = densenet.to(device)
    train_densenet(densenet, train_loader, device, num_epoch=300, lr=1e-1)
    torch.save(densenet, '/home/junhang/Projects/Scripts/saved_model/CIFAR/densenet.pkl')

    print("--------------------CNN training--------------------------------")
    cnn = CNN()
    cnn = cnn.to(device)
    train_cnn(cnn, train_loader, device, num_epoch=5)
    torch.save(cnn, '/home/junhang/Projects/Scripts/saved_model/CIFAR/cnn.pkl')


    print("--------------------VAE training--------------------------------")
    vae = VAE()
    vae = vae.to(device)
    train_vae(vae, train_loader, device, num_epoch=10)
    torch.save(vae, '/home/junhang/Projects/Scripts/saved_model/CIFAR/vae.pkl')






