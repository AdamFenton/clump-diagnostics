import plonk
import numpy as np
import math
import matplotlib.pyplot as plt
from decimal import *
getcontext().prec = 100
x = 1
y = 2
z = 3

au = plonk.units('au')
fig, axs = plt.subplots(figsize=(7,8))
snap = plonk.load_snap('run1.012.0962764.030.h5')
sinks = snap.sinks
snap.set_units(position='au', density='g/cm^3',smoothing_length='au',velocity='km/s')
accreted_mask = snap['smoothing_length'] > 0
h = snap['smoothing_length']

x,y,z = -0.43473436487793210 * au    ,   823.43709657320983  * au  ,  0.60831784701906388*au
x,y,z = snap['x'][962763],snap['y'][962763],snap['z'][962763]

max_elem = np.amax(snap['density'][h>0])
id = np.where(snap['density']== max_elem)
# print(snap['density'][id])
# print(id)
# print(max(snap['density'][h>0]))







A, B, C = snap['x']-x, snap['y']-y, snap['z']-z
snap['radius'] = np.sqrt(A**2 + B ** 2 + C**2)
snap_active = snap[accreted_mask]


axs.scatter(snap_active['radius'],snap_active['density'])
axs.set_xscale('log')
axs.set_xlim(1e-3,1e-1)
axs.set_yscale('log')
plt.show()
