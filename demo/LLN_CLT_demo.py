# Quantitative Economics with Python

## This is a simulation that demonstrates Law of Large Numbers and Central Limit Theorem


import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import t, beta, lognorm, expon, gamma, uniform, cauchy
from scipy.stats import gaussian_kde, poisson,binom, norm,chi2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from scipy.linalg import inv, sqrtm

# classical LLN
# iid variables Kolmogorov's strong law
# X_1, X_2, ... X_n iid ~ F
# mu = E[X] = integral[xF](dx)
# X_bar = 1/n * sum(X_i)
# then, Prob(X_bar->mu as n-> inf) = 1

# Law of Large Number Demo
n = 100
distributions = {
    "student's t with 10 df ": t(10),
    "beta(2,2)": beta(2, 2),
    "lognormal": lognorm(0.5),
    "gamma(5,1/2)": gamma(5, scale=2),
    "poisson(4)": poisson(4),
    "exponential with lambda =1": expon(1)
}
num_plots = 3
fig, axes = plt.subplots(num_plots, 1, figsize=(8, 8))
bbox = (0., 1.02, 1, .102)
legend_args = {'ncol': 2, 'bbox_to_anchor': bbox, "loc": 3, 'mode': 'expand'}

plt.subplots_adjust(hspace=0.5)

for ax in axes:
    # pick a distribution
    name = random.choice(list(distributions.keys()))
    distribution = distributions.pop(name)

    # random draw
    data = distribution.rvs(n)

    # sample mean
    sample_mean = np.empty(n)
    for i in range(n):
        sample_mean[i] = np.mean(data[:i + 1])

    # plot
    ax.plot(list(range(n)), data, 'o', color='grey', alpha=0.5)
    axlabel = '$\\bar X_n$ for $x_i \sim$' + name
    ax.plot(list(range(n)), sample_mean, 'g-', lw=3, alpha=0.6, label=axlabel)
    m = distribution.mean()
    ax.plot(list(range(n)), [m] * n, 'k--', lw=1.5, label='$\mu$')
    ax.vlines(list(range(n)), m, data, lw=0.2)
    ax.legend(**legend_args)

plt.show(block=False)

# Non convergence
# cauchy population mean is undefined
# so sample mean does NOT converge
distribution = cauchy()
fig, ax = plt.subplots(figsize =(10,6))
data= distribution.rvs(n)
ax.plot(list(range(n)),data,linestyle ='', marker ='o', alpha=0.5)
ax.vlines(list(range(n)), 0, data, lw=0.2)
ax.set_title("{} observations from Cauchy distribution".format(n))
plt.show()

# CLT
# sample mean
n = 1000
fig, ax = plt.subplots(figsize=(10, 6))
data = distribution.rvs(n)

sample_mean = np.empty(n)

for i in range(1, n):
    sample_mean[i] = np.mean(data[:i])

ax.plot(list(range(n)), sample_mean, 'r-', lw=3, alpha=0.6, label="$\\bar X_n$")
ax.plot(list(range(n)), [0] * n, 'k--', lw=0.5)
ax.legend()

# CLT
fig, axes = plt.subplots(2,2,figsize=(10,6))
plt.subplots_adjust(hspace=0.5)
axes = axes.flatten()
ns = [1,2,4,8]
dom = list(range(9))

for ax,n in zip(axes,ns):
    b= binom(n,0.5)
    ax.bar(dom,b.pmf(dom),alpha =0.6,align='center')
    ax.set(xlim=(-0.5,8.5),ylim=(0,0.55), xticks= list(range(9)),yticks=(0,0.2,0.4),title='$n={}$'.format(n))
plt.show()

n = 250
k = 100000 # draws
distribution = expon(2)
mu, sigma = distribution.mean(), distribution.std()

data = distribution.rvs((k,n))
sample_means = data.mean(axis=1)

Y = np.sqrt(n)*(sample_means-mu)

#plot
fig,ax = plt.subplots(figsize=(10,6))
xmin,xmax = -3*sigma, 3*sigma
ax.set_xlim(xmin,xmax)
ax.hist(Y,bins=50,alpha=0.5,density=True)
xgrid = np.linspace(xmin,xmax,200)
ax.plot(xgrid,norm.pdf(xgrid,scale=sigma),'k-',lw=2,label='$N(0,\sigma^2)$')
ax.legend()

plt.show()