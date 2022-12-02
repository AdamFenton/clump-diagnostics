import numpy as np
import matplotlib.pyplot as plt

R = np.linspace(1,300,100)


def sigma_wo_softening(R):
    return 1.53E3 * (R/10)**(-1.5)
def sigma_w_softening(R):
    return 1.53E3 * (R/10)**(-1.5)*(1-np.sqrt(10/R))

plt.plot(R,sigma_w_softening(R),c='k',linestyle='dashed')
plt.plot(R,sigma_wo_softening(R),c='k',linestyle='solid')
plt.yscale('log')

plt.xlabel("R [AU]")
plt.ylabel("Σ g$\,cm^{-3}$")
plt.show()
