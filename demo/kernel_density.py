# use kernel density to approximate distribution
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt

# true data generating process
N = 500
mu, sigma = 80, 5
mu2, sigma2 = 40, 25
X1 = np.random.normal(mu, sigma, 500)
X2 = np.random.normal(mu2, sigma2, 500)

X = np.concatenate([X1, X2])

# setup X range for density calculation and plotting
X_plot = np.linspace(0, 100, 1000)

# use kernel to smooth the histogram
# [:, np.newaxis] creates one extra bracket for each entry
kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(X[:, np.newaxis])

# log density mapped from X_plot
log_dens = kde.score_samples(X_plot[:, np.newaxis])

# compare two density in a figure

fig = plt.figure()

ax1 = plt.subplot(3, 1, 1)
plt.hist(X, bins = 100)
#ax1.set_title("original DGP")

ax2 = plt.subplot(3, 1, 2)
plt.plot(X_plot, np.exp(log_dens), color="tab:red")
#ax2.set_title("histogram")
plt.xlim(0,120)

ax3 = plt.subplot(3, 1, 3)
plt.hist(X, bins=100, density=True)
plt.plot(X_plot, np.exp(log_dens), color="tab:red")
#ax3.set_title("fit kernel to distribution")


plt.subplots_adjust(hspace=0.8)
fig.suptitle("Use kernel density to approximate distribution")
plt.show()