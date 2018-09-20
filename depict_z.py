import torch
import torchvision
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation

# define Gaussian function for display
def Gaussian(x, mu, sigma):
    k = len(mu)
    a = 1.0 / (np.sqrt(np.power(2.0*np.pi, k) * np.linalg.det(sigma)))
    b = np.matmul(np.transpose(x-mu), np.matmul(np.linalg.inv(sigma), (x-mu)))
    z = a * np.exp(-0.5 * b)
    return z


x1 = np.linspace(-2.0, 2.0, 10)
x2 = np.linspace(-2.0, 2.0, 10)
X1, X2 = np.meshgrid(x1, x2)



# load test data
num_test = 10000
test_data = torchvision.datasets.MNIST(
    root = '/home/junhang/Projects/DataSet/MNIST',
    train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)/255.
test_x = test_x[:num_test].cuda()


# latent variable contours of normal examples
vae_depict_model = torch.load('/home/junhang/Projects/Scripts/saved_model/vae_depict_model.pkl')
#for point in test_data:
_, mu, var = vae_depict_model(test_x[0])

mu_vector = np.transpose(mu.cpu().data.numpy())
sigma_matrix = np.diag(np.squeeze(np.exp(var.cpu().data.numpy()), 0))
X_vectors = np.vstack((x1, x2))

G_func = Gaussian(X_vectors, mu_vector, sigma_matrix)
#fig, ax = plt.subplots()
C = plt.contour(X1, X2, G_func, 6, linewidths=0.5, colors='k')
plt.clabel(C, inline=True, fontsize=10)
plt.xticks(())
plt.yticks(())
plt.show()
"""
    
    def animate(i):
        c = ax.contour(X1, X2, z, 6, linewidths=0.5, colors='k')
        return c

    def init():
        return gauss

    ani = animation.FuncAnimation(fig=fig,
                                  func=animate,
                                  frames=100,
                                  init_func=init,
                                  interval=20,
                                  blit=False)
    plt.show()
"""




# latent variable contours of adversarial examples





# the distance between two distribution(normal and adversarial)





# evaluate the adversarial samples with VAE+CNN





# reverse the adversarial towards normal ones





# evaluate the reversed adversarial samples with VAE+CNN