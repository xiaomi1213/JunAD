import numpy as np
import torch
import torch.nn.functional as nn

def svm_l1loss(a, y, weight, C=1.0):
    relu = nn.relu(1 - a * y)
    loss = 0.5 * (weight * weight).sum() + C * relu.sum()
    return loss

def svm_l2loss(a, y, weight, C=1.0):
    relu = nn.relu(1 - a * y)
    loss = 0.5 * (weight * weight).sum() + C * relu.sum()
    return loss


def svm_l3loss(a, y, weight, C=1.0):
    relu = nn.relu(1 - a*y)
