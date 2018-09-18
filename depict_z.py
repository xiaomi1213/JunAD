import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt

# define Gaussian function for display
def Gaussian(x, mu, sigma):
    k = len(mu)
    a = 1.0 / (np.sqrt(np.power(2*np.pi, k) * np.linalg.det(sigma)))
    b = np.matmul(np.transpose(x-mu), np.matmul(np.linalg.inv(sigma), (x-mu)))
    z = a * np.exp(-0.5 * b)
    return z

mu = np.array([0,0])
sigma = np.array([[1,0],[0,1]])
y = Gaussian(np.array([0,0]), mu, sigma)
print(y)
# latent variable contours of normal examples






# latent variable contours of adversarial examples





# the distance between two distribution(normal and adversarial)





# evaluate the adversarial samples with VAE+CNN





# reverse the adversarial towards normal ones





# evaluate the reversed adversarial samples with VAE+CNN