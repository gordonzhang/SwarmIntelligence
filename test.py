import numpy as np
from matplotlib.pylab import *
import matplotlib.animation as animation

# generate 4 random variables from the random, gamma, exponential, and uniform distribution
x1 = np.random.normal(-2.5, 1, 10000)
x2 = np.random.gamma(2, 1.5, 10000)
x3 = np.random.exponential(2, 10000)+7
x4 = np.random.uniform(14,20, 10000)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

def updateData(curr):
    if curr <=2: return
    for ax in (ax1, ax2, ax3, ax4):
        ax.clear()
    ax1.hist(x1[:curr], normed=True, bins=np.linspace(-6,1, num=21), alpha=0.5)
    ax2.hist(x2[:curr], normed=True, bins=np.linspace(0,15,num=21), alpha=0.5)
    ax3.hist(x3[:curr], normed=True, bins=np.linspace(7,20,num=21), alpha=0.5)
    ax4.hist(x4[:curr], normed=True, bins=np.linspace(14,20,num=21), alpha=0.5)

simulation = animation.FuncAnimation(fig, updateData, interval=50, repeat=False)

plt.show()