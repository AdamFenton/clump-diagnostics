# ================================== #
# plot_cublic_spline.py
# ================================== #
# A small script the plot the M4
# cublic spline with its first and
# second derivatives. I will use this
# in the numerical methods chapter

# Author = Adam Fenton
# Date = 20220706
# =============================== #

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
q = np.arange(0., 3.05, 0.05)

fig, ax = plt.subplots(figsize=(8,8))

def cubic_spline(q):
    return (1/np.pi)*np.piecewise(q,[((q >=0) & (q<1)),((q >=1) & (q<2)),q>=2],
               [lambda q : 1/4*(2-q)**3 - (1-q)**3,
                lambda q : 1/4*(2-q)**3,
                lambda q : 0*q])


def cubic_spline_first_deriv(q):
    return (1/np.pi)*np.piecewise(q,[((q >=0) & (q<1)),((q >=1) & (q<2)),q>=2],
               [lambda q : 3/4*q*(3*q-4),
                lambda q : -3/4*(2-q)**2,
                lambda q : 0*q])

def cubic_spline_second_deriv(q):
    return (1/np.pi)*np.piecewise(q,[((q >=0) & (q<1)),((q >=1) & (q<2)),q>=2],
               [lambda q : (9*q/2)-3,
                lambda q : 3 - (3*q)/2,
                lambda q : 0*q])



# x_values = np.arange(0., 3.05, 0.05)
# for mu, sig in [(0, 1/(np.pi))]:
#     ax.plot(x_values, gaussian(x_values, mu, sig))




ax.axvline(x=2,c='black',linestyle='--',linewidth=1)
# ax.axhline(y=0,c='black',linestyle='-',linewidth=2)
# ax.spines['bottom'].set_position('zero')

ax.plot(q,cubic_spline(q),linewidth=2,c='k')
ax.plot(q,cubic_spline_first_deriv(q),linewidth=2,c='k',linestyle="--")
ax.plot(q,cubic_spline_second_deriv(q),linewidth=2,c='k',linestyle="-.")
ax.set_ylabel(r'$W_{ij}$',fontsize=15)
ax.set_xlabel('q',fontsize=15)
ax.set_xlim(0,3)
ax.set_ylim(-1,1)

# # ax.tick_params(axis="x", labelsize=15,width=2,length=10)
ax.tick_params(axis="x", labelsize=15,length=8,which='minor',width=2)

# ax.tick_params(axis="y", labelsize=15,width=2,length=10)
ax.tick_params(axis="y", labelsize=15,length=8,which='minor',width=2)
ax.set_xticks(np.arange(min(q), max(q)+1, 1.0))
# ax.set_yticks(np.arange(-1, 1.0, 1.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.2))
# ax.yaxis.set_minor_locator(MultipleLocator(0.1))

ax.tick_params(right=True, top=True,labelsize=15,width=2,length=10)

plt.figtext(0.90, 0.87, r'M$_{4}$ cubic' ,fontsize=20,va="top", ha="right")
fig.align_ylabels()
fig.tight_layout(pad=0.40)
plt.show()
# plt.savefig("m4_cubic_spline.pdf")
