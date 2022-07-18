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
from os.path import exists

fig,axs = plt.subplots(figsize=(8,8))
cwd = os.getcwd()
complete_file_list = glob.glob("run*") # Use glob to collect all files in direc

def sortKeyFunc(s):
    ''' We want to sort the filenames in reverse order by their density so we
        use the -6th to the -4th index of the filename e.g. 030, 040 etc
    '''
    return int(os.path.basename(s)[-6:-4])

complete_file_list.sort(key=sortKeyFunc,reverse=True)
final_array = np.zeros((len(complete_file_list), 15))

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


    return clump_position, clump_velocity,snapshot_density


def define_clump_subsnaps(snap,clump_position):
    ''' For each of the loaded full disc snapshots, take the position of the
        clump from the retrieve_clump_locations function and define a new
        subsnap centred on the clump.
    '''
    subSnap_1AU=plonk.analysis.filters.sphere(snap=snap,radius = (1*au),center=clump_position)
    subSnap_2AU=plonk.analysis.filters.sphere(snap=snap,radius = (2*au),center=clump_position)
    subSnap_5AU=plonk.analysis.filters.sphere(snap=snap,radius = (5*au),center=clump_position)
    subSnap_10AU=plonk.analysis.filters.sphere(snap=snap,radius = (10*au),center=clump_position)

    return subSnap_1AU,subSnap_2AU,subSnap_5AU,subSnap_10AU






class Clump:
    # Define class to store the clump information
    def __init__(self):
        self.density = None
        self.time    = None
        self.M1AU    = None
        self.M2AU    = None
        self.M5AU    = None
        self.M10AU   = None



file_exists = exists("clump_accretion_rates.csv")
if file_exists == True:
    print('='*65)
    print("File containing clump accretion rates already exists")
    print('='*65)
    df = pd.read_csv("clump_accretion_rates.csv")
    df = df.iloc[: , 1:]
    print(df)
else:
    print('='*65)
    print("Calculating clump accretion rates...")
    print('='*65)

    for file,n in zip(complete_file_list,enumerate(complete_file_list)):
        print("Calculating accretion rates for %s" % str(file))
        clump = Clump()
        snap = prepare_snapshot(file)
        particle_mass = snap['m'][0].to('jupiter_mass').magnitude
        clump_position = retrieve_clump_locations(file,clump_location_data)[0]
        clump_subsnaps = define_clump_subsnaps(snap,clump_position)

        clump.density = retrieve_clump_locations(file,clump_location_data)[2]
        clump.time = snap.properties['time'].to('year').magnitude
        # Rather than binning one array of particles up to certain radii and relying
        # on the bin resolution to be high enough to get the correct 'checkpoint
        # radii', I have used 4 individual subsnaps (see define_clump_subsnaps func)
        # and then just counted the number of particles inside each subsnap
        clump.M1AU  = len(clump_subsnaps[0]['m']) * particle_mass
        clump.M2AU  = len(clump_subsnaps[1]['m']) * particle_mass
        clump.M5AU  = len(clump_subsnaps[2]['m']) * particle_mass
        clump.M10AU = len(clump_subsnaps[3]['m']) * particle_mass


        final_array[n[0]] = [clump.density,
                             clump.time,
                             clump.M1AU,
                             None,              # Change in mass from previous snap
                             None,              # Mass accretion rate
                             clump.M2AU,
                             None,              # Change in mass from previous snap
                             None,              # Mass accretion rate
                             clump.M5AU,
                             None,              # Change in mass from previous snap
                             None,              # Mass accretion rate
                             clump.M10AU,
                             None,              # Change in mass from previous snap
                             None,              # Mass accretion rate
                             None]              # Change in time from previous snap


    # convert to dataframe
    df = pd.DataFrame(final_array, columns = ['Density','Time (years)',
                                              '1 AU Mass (Mj)',
                                              'Delta M (1AU)',
                                              'Mdot (1AU)',
                                              '2 AU Mass (Mj)',
                                              'Delta M (2AU)',
                                              'Mdot (2AU)',
                                              '5 AU Mass (Mj)',
                                              'Delta M (5AU)',
                                              'Mdot (5AU)',
                                              '10 AU Mass (Mj)',
                                              'Delta M (10AU)',
                                              'Mdot (10AU)',
                                              'Delta Time (years)'
                                              ])


    # Calculate differences and mass accretion rates for each radii at each step.
    df['Delta M (1AU)'] = df['1 AU Mass (Mj)'].diff()
    df['Delta M (2AU)'] = df['2 AU Mass (Mj)'].diff()
    df['Delta M (5AU)'] = df['5 AU Mass (Mj)'].diff()
    df['Delta M (10AU)'] = df['10 AU Mass (Mj)'].diff()
    df['Delta Time (years)'] = df['Time (years)'].diff()
    df['Mdot (1AU)'] = df['Delta M (1AU)'] / df['Delta Time (years)']
    df['Mdot (2AU)'] = df['Delta M (2AU)'] / df['Delta Time (years)']
    df['Mdot (5AU)'] = df['Delta M (5AU)'] / df['Delta Time (years)']
    df['Mdot (10AU)'] = df['Delta M (10AU)'] / df['Delta Time (years)']

    print("Saving data to clump_accretion_rates.csv")
    df.to_csv('%s/clump_accretion_rates.csv' % cwd)
