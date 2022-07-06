# ================================== #
# plot_toomre_parameter.py
# ================================== #
# Calculate and plot the value of Q
# as a function of disc radius
# Author = Adam Fenton
# Date = 20220706
# =============================== #

import plonk
import matplotlib.pyplot as plt
import numpy as np
from plonk.utils.strings import time_string # Required for timestamping in plot
import os # Required for getcwd()
import sys # Required for command line arguments
from pathlib import Path # Required to create plot directory
import warnings
import pint


if hasattr(pint, 'UnitStrippedWarning'):
    warnings.simplefilter('ignore', category=pint.UnitStrippedWarning)
np.seterr(divide='ignore', invalid='ignore')

# Author : Adam Fenton
# Date   : 07/12/2021

Path("%s/plots/" % os.getcwd()).mkdir(parents=True, exist_ok=True )


kb_on_mu_mh = 34705892.71 # kb/(μ*Mh) for the calculation of sound speed
G           = 6.67E-8     # Gravitational constant in cgs


if len(sys.argv) != 3:
    print('Requires a filename and value for r_out')
    exit(0)

filename = str(sys.argv[1]) # Get filename from command line
r_out = str(sys.argv[2]) #

snap = plonk.load_snap(filename)
time_stamp = time_string(snap, 'year', 'yr')



snap.set_units(position='au', density='g/cm^3',smoothing_length='au',velocity='km/s')
sink = snap.sinks[0]
snap['rad'] = np.sqrt((snap['x'] - sink['position'][0]) ** 2 + (snap['y']-sink['position'][1]) ** 2)
snap.add_quantities('disc')
snap.set_central_body(0)
accreted_mask = snap['smoothing_length'] > 0
snap = snap[accreted_mask]
prof = plonk.load_profile(snap,cmin='1 au', cmax='%s au' % r_out ,n_bins=100)
prof.set_units(position='au',surface_density='g/cm^2',radius='au')

prof['my_sound_speed'] = np.sqrt(kb_on_mu_mh * prof['my_temp'])
prof['toomre_Q'] = ((prof['keplerian_frequency']* prof['my_sound_speed'])/(np.pi * G * prof['surface_density'].to('g/cm^2')))


plt.plot(prof['radius'],prof['toomre_Q'],c='k')
plt.ylabel("Toomre Q")
plt.xlabel("Radius [AU]")
plt.ylim(0,4)
plt.xlim(0,int(r_out))

plt.figtext(0.7, 0.175, '%s' % time_stamp,fontsize=12,va="top", ha="left")
plt.savefig('%s/plots/toomre_q_%s.png' % (os.getcwd(),filename),dpi = 200)
