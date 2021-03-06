# ================================== #
# plot_snapshot_profiles.py
# ================================== #
# Produce profiles and surface density
# render for a specific snapshot given
# by the CLI argument. Converts the
# snapshot to H5 format if required.
# Generally used to check progress of
# simulation during a run.


# Author = Adam Fenton
# Date = 20220706
# =============================== #



import matplotlib.pyplot as plt
import plonk
import sys
import subprocess
import numpy as np
import warnings
import pint
from matplotlib.gridspec import GridSpec
from os.path import exists
from plonk.utils.strings import time_string # Required for timestamping in plot
from pathlib import Path # Required to create plot directory
import os

Path("%s/plots/" % os.getcwd()).mkdir(parents=True, exist_ok=True )

cwd = os.getcwd()
if hasattr(pint, 'UnitStrippedWarning'):
    warnings.simplefilter('ignore', category=pint.UnitStrippedWarning)
np.seterr(divide='ignore', invalid='ignore')

def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.tick_params(labelbottom=False, labelleft=False)

def convert_to_h5():
    file_from_cmd_line = sys.argv[1]
    convert = subprocess.run(['phantom2hdf5',str(file_from_cmd_line)], stdout=subprocess.PIPE)
    h5_file = file_from_cmd_line + ".h5"
    return h5_file

def prepare_snapshots(h5_file):
    ''' Load full snapshot as plonk object and initialise subsnap centred on clump.
        Also apply filter to exclude dead and accreted particles with `accreted_mask`
        and load units
    '''
    kb_on_mu_mh = 34705892.71 # kb/(μ*Mh) for the calculation of sound speed
    G           = 6.67E-8     # Gravitational constant in cgs

    snap = plonk.load_snap(h5_file)
    sinks = snap.sinks
    if type(sinks['m'].magnitude) == np.float64:
        central_star_mass = sinks['m']
    else:
        central_star_mass = sinks['m'][0]

    snap.set_units(position='au', density='g/cm^3',smoothing_length='au',velocity='km/s')
    snap['radius'] = np.sqrt(snap['x']**2 + snap['y']**2 + snap['z']**2)
    r_out = max(snap['radius'].magnitude)
    snap['vmag'] = np.sqrt(snap['v_x']**2 + snap['v_y']**2 + snap['v_z']**2)
    snap.add_quantities('disc')
    snap.set_central_body(0)
    prof = plonk.load_profile(snap,cmin='10 au', cmax='%s au' % r_out ,n_bins=200)

    prof['my_sound_speed'] = np.sqrt(kb_on_mu_mh * prof['my_temp'])
    prof['toomre_Q'] = ((prof['keplerian_frequency']* prof['my_sound_speed'])/(np.pi * G * prof['surface_density'].to('g/cm^2')))


    accreted_mask = snap['smoothing_length'] > 0
    snap_active = snap[accreted_mask]

    return snap_active, prof,r_out




try:
    file_exists = exists(sys.argv[1]+".h5")
    if file_exists == True:
        print('='*65)
        print("Snapshot", str(sys.argv[1])+".h5 already exists, no need to convert")
        print('='*65)
        h5_file = sys.argv[1]+".h5"
        snap = prepare_snapshots(h5_file)[0]
        prof = prepare_snapshots(h5_file)[1]
    else:
        print('='*45)
        print("Converting", str(sys.argv[1]), "to HDF5 format")
        print('='*45)
        h5_file = convert_to_h5()
        snap = prepare_snapshots(h5_file)[0]
        prof = prepare_snapshots(h5_file)[1]
except IndexError:
    print("File not provided...exiting")
    sys.exit(1)
n_alive = len(snap['x'])
n_sinks = len(snap.sinks)
time_stamp = time_string(snap, 'year', 'yr')
file_name = sys.argv[1]


fig = plt.figure(constrained_layout=True,figsize=(14.5,8))

gs = GridSpec(3, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, -2])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, -2])
ax5 = fig.add_subplot(gs[1:, -1])
ax6 = fig.add_subplot(gs[-1, 0])
ax7 = fig.add_subplot(gs[-1, -2])
r_out = prepare_snapshots(h5_file)[2]
extent = (-r_out, r_out, -r_out, r_out)



y_labels = ['Q','Z','ρ','T','y','V','Σ']
x_labels = ['R','R','R','R','x','R','R']
y_limits = [(0,5),(-100,100),(1E-18,1e-7),(1e0,5E3),(-1000,1000),(0,15)]
axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7]

for axes,y_label,x_label,y_limit in zip(axes,y_labels,x_labels,y_limits):
    axes.set_ylabel(y_label)
    axes.set_xlabel(x_label)
    axes.set_ylim(y_limit)
    axes.set_xlim(0,r_out-50)

ax1.plot(prof['radius'],prof['toomre_Q'],c='k')
ax2.scatter(snap['radius'],snap['z'],s=0.5,c='k')
ax3.scatter(snap['radius'],snap['density'],s=0.5,c='k')
ax4.scatter(snap['radius'],snap['my_temp'],s=0.5,c='k')
snap.image(quantity='density', extent=extent, cmap='gist_heat', ax=ax5,norm='log',vmin=3e-3)

ax6.scatter(snap['radius'],snap['vmag'],s=0.5,c='k')
ax7.plot(prof['radius'],prof['surface_density'].to('g/cm^2'),c='k')
ax3.set_yscale('log')
ax4.set_yscale('log')
ax5.set_xlim(-r_out,r_out)
ax5.set_ylim(-r_out,r_out)
ax5.set_aspect('equal', 'box')
plt.figtext(0.79, 0.97, '%s' % file_name,fontsize=12,va="top", ha="right")
plt.figtext(0.79, 0.94, '%s' % time_stamp,fontsize=12,va="top", ha="right")
plt.figtext(0.79, 0.91, 'N alive = %s' % n_alive,fontsize=12,va="top", ha="right")
plt.figtext(0.79, 0.88, 'N sinks = %s' % n_sinks,fontsize=12,va="top", ha="right")


plt.savefig('%s/plots/snapshot_profiles_%s.png' % (os.getcwd(),file_name),dpi = 200)
