import torch

def vae_eval(model, eval_loader):
    model.eval()
    eval_loss = 0
