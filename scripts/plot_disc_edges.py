import numpy as np
import matplotlib.pyplot as plt
R = np.linspace(10,300,100)
G = 6.67E-8
omega = np.sqrt(1*(0.8)/(R)**3)


def calculate_cs0(T):
    return np.sqrt((1.38E-16 * T)/(2.381 * 1.67E-24)) / 2.978E6


labels = [r'T$_{1\rm{AU}} = 200\,$K',r'T$_{1\rm{AU}} = 150\,$K',r'T$_{1\rm{AU}} = 10\,$K']
for T,style,label in zip([200,150,10],['-','--',':'],labels):
    cs0 = calculate_cs0(T)
    cs = cs0*(R)**(-0.25)
    HHsqrt2 = cs/omega * np.sqrt(2)
    zmax = (3.0*HHsqrt2)
    zmin = (-3.0*HHsqrt2)

    plt.plot(R,zmin,c='k',linestyle=style,label = label)
    plt.plot(R,zmax,c='k',linestyle=style)
plt.xlabel('Radius [AU]')
plt.ylabel('Z [AU]')
plt.legend()
plt.savefig('/Users/adamfenton/Documents/PhD/thesis/images /zlimits.pdf')
