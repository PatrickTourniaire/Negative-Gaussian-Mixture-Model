from models.mul_gmm import GaussianMixture
import torch
import numpy as np
import matplotlib.pyplot as plt

n_coordinates = 2

n_gold_clusters = 3
n_gold_samples = 1000

# The prior distribution on clusters is not uniform!
gold_cluster_prior = [0.2, 0.4, 0.4]

gold_x_means = np.array(np.array([ [-1, -1], [-1, 1], [1, 1] ]))
gold_x_cov = np.array([
    [
        [0.5**2, 0],
        [0, 0.2**2],
    ],
    [
        [0.3**2, 0],
        [0, 0.3**2],
    ],
    [
        [0.4**2, 0.8 * 0.4 * 0.5],
        [0.8 * 0.4 * 0.5, 0.5**2],
    ],
])

# sampling

# first we sample the cluster for each point
gold_clusters = np.random.choice(n_gold_clusters, n_gold_samples, p=gold_cluster_prior)

# then we sample the coordinates
gold_data = np.empty((n_gold_samples, n_coordinates))
for i in range(n_gold_samples):
    gold_data[i] = np.random.multivariate_normal(gold_x_means[gold_clusters[i]], gold_x_cov[gold_clusters[i]])

colors = np.array(["blue", "green", "red"])

fig, ax = plt.subplots()
ax.set_xlim((-2, 2))
ax.set_ylim((-2, 2))

ax.scatter(gold_data[:,0], gold_data[:,1], color=colors[gold_clusters])
plt.savefig(f'out/simple_gmm/gold_data.pdf')

# On of the cluster in the dataset has a full covariance matrix,
# we cannot approximate it with a single Gaussian that has independent coordinate,
# but we can increase the number of cluster in our model to try to fit it with 
# several bivariate Gaussians!
n_clusters = 2
g = GaussianMixture(n_clusters)

optimizer = torch.optim.SGD(g.parameters(), lr=0.01, momentum=0.9)

for _ in range(1000):
    optimizer.zero_grad()
    loss = -torch.sum(g(torch.from_numpy(gold_data))) / len(gold_data)
    print(loss)
    loss.backward()
    optimizer.step()

# sample from the learned distribution
pred_prior = np.random.choice(g.prior.shape[0], 1000, p=g.prior.softmax(0).detach().numpy())

x_means = g.mean.squeeze(0).detach().numpy()
x_cov = np.empty((g.prior.shape[0], 2, 2))
for i in range(g.prior.shape[0]):
    std1 = g.log_std[0, i, 0].exp().item()
    std2 = g.log_std[0, i, 1].exp().item()
    p = g.reparameterized_coef[0, i].tanh().item()

    x_cov[i, 0, 0] = std1 * std1
    x_cov[i, 1, 1] = std2 * std2
    x_cov[i, 0, 1] = std1 * std2 * p
    x_cov[i, 1, 0] = std1 * std2 * p

# then we sample the coordinates
pred = np.empty((1000, 2))
for i in range(1000):
    pred[i] = np.random.multivariate_normal(x_means[pred_prior[i]], x_cov[pred_prior[i]])

colors = np.array([c for c in "bgrcmyk"])
# you can do this if you want to visualize with more clusters,
# but several clusters will have the same color...
#colors = np.array([c for c in "bgrcmykbgrcmykbgrcmykbgrcmyk"])

fig, ax = plt.subplots()

ax.set_xlim((-2, 2))
ax.set_ylim((-2, 2))

ax.scatter(pred[:,0], pred[:,1], color=colors[pred_prior])
plt.savefig(f'out/simple_gmm/model_plot.pdf')

plt.clf()