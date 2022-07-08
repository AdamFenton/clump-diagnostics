# ---------------------------------------- #
# plot_clump_accretion.py
# ---------------------------------------- #
# Read in all the snapshots from clump dir
# and calculate the mass contained within
# the checkpoint radii from the centre. Then
# divide these masses by the time between
# snapshots to calculate an average mass
# accretion rate.
# ---------------------------------------- #
# Author: Adam Fenton
# Date: 20220708
# ---------------------------------------- #

import plonk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from scipy import stats

fig,axs = plt.subplots(figsize=(8,8))

complete_file_list = glob.glob("run*") # Use glob to collect all files in direc
def sortKeyFunc(s):
    return int(os.path.basename(s)[-6:-4])
complete_file_list.sort(key=sortKeyFunc,reverse=True)

bins=np.logspace(np.log10(0.001),np.log10(11),75)

# It is safe to index the complete file list here because all the files are the
# same clump and so have the same .dat file.
clump_data_file = complete_file_list[0].split(".")[1]+".dat"

# Read in the clump location data as a pandas dataframe
clump_location_data = pd.read_csv(clump_data_file,names=["rho", "x", "y", "z",
                                                         "vx", "vy", "vz"]
                                                         )
# Define units for plonk so calculated values are in correct units
au = plonk.units('au')
kms = plonk.units('km/s')


def prepare_snapshot(snapshot_file):
    '''Load the snapshot_file with plonk and prepare it for working. This means
       changing it's units and removing all of the accreted particles
    '''

    snap = plonk.load_snap(snapshot_file)
    snap.set_units(position='au', density='g/cm^3',smoothing_length='au',
                   velocity='km/s')

    # There will be at least one sink (host star) in the file but likely more so
    # we have to identify which one is the host star and assign it's mass for
    # use later on.

    sinks = snap.sinks
    if type(sinks['m'].magnitude) == np.float64:
        central_star_mass = sinks['m']
    else:
        central_star_mass = sinks['m'][0]

    accreted_mask = snap['smoothing_length'] > 0
    snap_active = snap[accreted_mask]

    return snap_active

def retrieve_clump_locations(snapshot_file,clump_location_data):
    ''' Collect the clump's position and velocity from the clump location
        dataframe and return it for recentering proceedure
    '''

    snapshot_density = 10**(-int(snapshot_file.split(".")[-2].rstrip('0')))
    index = clump_location_data['rho'].sub(snapshot_density).abs().idxmin()
    clump_row_in_df = clump_location_data.iloc[index]
    clump_position = clump_row_in_df.iloc[1:4].values * au
    clump_velocity = clump_row_in_df.iloc[4:].values * kms


    return clump_position, clump_velocity


def define_clump_subsnap(snap,clump_position):
    ''' For each of the loaded full disc snapshots, take the position of the
        clump from the retrieve_clump_locations function and define a new
        subsnap centred on the clump.
    '''
    subSnap=plonk.analysis.filters.sphere(snap=snap,radius = (10*au),center=clump_position)

    return subSnap


def calculate_number_in_bin(binned_quantity,mean_quantity,bins):
    return stats.binned_statistic(binned_quantity, mean_quantity, 'count', bins=bins)



class Clump:
    def __init__(self):
        self.time  = None
        self.M1AU  = None
        self.M2AU  = None
        self.M5AU  = None
        self.M10AU = None

for file in complete_file_list:
    clump = Clump()
    snap = prepare_snapshot(file)
    clump.time = snap.properties['time']

    clump_position = retrieve_clump_locations(file,clump_location_data)[0]
    clump_subsnap = define_clump_subsnap(snap,clump_position)
    r_clump_centred = np.sqrt((clump_subsnap['x']-(clump_position[0]))**2 +
                              (clump_subsnap['y']-(clump_position[1]))**2 +
                              (clump_subsnap['z']-(clump_position[2]))**2)


    # Calculate the cumulative mass of the clump out to 10 AU by summing the
    # number of particles in 75 bins between 1e-3 AU and 10 AU then multiplying
    # by the mass of each particle in jupiter masses.
    mass_in_clump = np.cumsum(
                    calculate_number_in_bin(r_clump_centred,clump_subsnap['m'],bins)[0]
                    * clump_subsnap['m'][0].to('jupiter_mass')
                    )
    checkpoint_1AU = np.digitize(1, bins) - 1
    checkpoint_2AU = np.digitize(2, bins) - 1
    checkpoint_5AU = np.digitize(5, bins) - 1
    checkpoint_10AU = np.digitize(10, bins) - 1
    clump.M1AU  = mass_in_clump[checkpoint_1AU]
    clump.M2AU  = mass_in_clump[checkpoint_2AU]
    clump.M5AU  = mass_in_clump[checkpoint_5AU]
    clump.M10AU = mass_in_clump[checkpoint_10AU]

    attrs = vars(clump)
    print(', '.join("%s: %s" % item for item in attrs.items()))
