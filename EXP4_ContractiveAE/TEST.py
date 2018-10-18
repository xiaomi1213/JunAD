
import torch
"""
def jacobian(inputs, outputs):
    return torch.stack([grad([outputs[:, i].sum()], [inputs], retain_graph=True, create_graph=True)[0]
                        for i in range(outputs.size(1))], dim=-1)

"""


def FrobiusNorm(jacobian):
    norm = torch.pow(torch.norm(jacobian, 2),2)
    return norm



