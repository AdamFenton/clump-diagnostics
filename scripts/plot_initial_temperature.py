# ================================== #
# plot_initial_temperature.py
# ================================== #
# Plot the initial temperature as a
# function of the radius
# Author = Adam Fenton
# Date = 20221103
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
import glob
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rcParams['text.usetex'] = True
fig,axs = plt.subplots(figsize=(7,4.5))
plt.rcParams["font.family"] = "serif"

if hasattr(pint, 'UnitStrippedWarning'):
    warnings.simplefilter('ignore', category=pint.UnitStrippedWarning)
np.seterr(divide='ignore', invalid='ignore')

# Author : Adam Fenton
# Date   : 07/12/2021

Path("%s/plots/" % os.getcwd()).mkdir(parents=True, exist_ok=True )


kb_on_mu_mh = 34705892.71 # kb/(μ*Mh) for the calculation of sound speed
G           = 6.67E-8     # Gravitational constant in cgs


# if len(sys.argv) != 3:
#     print('Requires a filename and value for r_out')
#     exit(0)

filenames = glob.glob("*.h5") # Get filename from command line
filenames = sorted(filenames, key = lambda x: x.split('_')[1].split('.')[0])

names_tmp = [r'T$_{1\rm{AU}} = 200\,$K',r'T$_{1\rm{AU}} = 150\,$K']
linestyles = ['solid','dashed']
r_out = 300


for filename,label,style in zip(filenames,names_tmp,linestyles):
    snap = plonk.load_snap(filename)

    time_stamp = time_string(snap, 'year', 'yr')



    snap.set_units(position='au', density='g/cm^3',smoothing_length='au',velocity='km/s')
    sink = snap.sinks[0]
    snap['rad'] = np.sqrt((snap['x'] - sink['position'][0]) ** 2 + (snap['y']-sink['position'][1]) ** 2)
    snap.add_quantities('disc')
    snap.set_central_body(0)
    accreted_mask = snap['smoothing_length'] > 0
    snap = snap[accreted_mask]
    prof = plonk.load_profile(snap,cmin='10 au', cmax='%s au' % r_out ,n_bins=500)
    prof.set_units(position='au',surface_density='g/cm^2',radius='au')

    # prof['my_sound_speed'] = np.sqrt(kb_on_mu_mh * prof['my_temp'])
    # prof['toomre_Q'] = ((prof['keplerian_frequency']* prof['sound_speed'].to('cm/s'))/(np.pi * G * prof['surface_density'].to('g/cm^2')))


    axs.plot(prof['radius'],prof['temperature'],c='k',linewidth=2,label="%s" %label,linestyle='%s' %style)
    # axs.axhline(y=1,c='red',linestyle='dotted',linewidth=2)
    axs.set_ylabel("Temperature [K]",fontsize=15)
    axs.set_xlabel("Radius [AU]",fontsize=15)
    axs.set_ylim(0,70)
    axs.set_xlim(0,int(r_out))

# plt.show()
# axs.figtext(0.7, 0.175, '%s' % time_stamp,fontsize=12,va="top", ha="left")
#handles,labels = axs.get_legend_handles_labels()
#labels, handles = zip(*sorted(zip(labels,handles),key=lambda t: t[0]))
plt.legend(loc='upper right')
plt.savefig('%s/plots/initial_temperature.pdf' % (os.getcwd()))
