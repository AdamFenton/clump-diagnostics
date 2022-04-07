import plonk
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import processing
import glob
import os
import pandas as pd
from pathlib import Path
import sys
from scipy import stats
import pint
from tqdm.auto import tqdm, trange
import warnings
from scipy.ndimage import gaussian_filter

cwd = os.getcwd()
kms =  plonk.units('km/s')
mean_bins_radial = np.logspace(np.log10(0.001),np.log10(50.0),120)
au = plonk.units('au')
density_to_load = None
fig_radial, f_radial_axs = plt.subplots(nrows=3,ncols=2,figsize=(7,8))

clump_results = open('clump-results.dat', 'w')

if hasattr(pint, 'UnitStrippedWarning'):
    warnings.simplefilter('ignore', category=pint.UnitStrippedWarning)

def calculate_sum(binned_quantity,summed_quantity,bins):
    return stats.binned_statistic(binned_quantity, summed_quantity, 'sum', bins=bins)


def calculate_mean(binned_quantity,mean_quantity,bins):
    return stats.binned_statistic(binned_quantity, mean_quantity, 'mean', bins=bins)

def calculate_number_in_bin(binned_quantity,mean_quantity,width):
    bins=np.logspace(np.log10(0.001),np.log10(width), 120)
    return stats.binned_statistic(binned_quantity, mean_quantity, 'count', bins=bins)

def calculate_thermal_energy(subSnap):
    U = 3/2 * 1.38E-16 * subSnap['my_temp'] * ((subSnap['m'][0].to('g'))/(1.67E-24))
    U = U.magnitude
    U *= ((U>2000)*(1/1.2)+(U<2000)*(1/2.381))
    return U

def calculate_gravitational_energy(subSnap,r_clump_centred):
    mass_interior = []
    for radius in r_clump_centred:
        N_interior = (r_clump_centred < radius).sum()
        mass_interior.append(N_interior * subSnap['m'][0].to('g').magnitude)

    omega = (6.67E-8 * np.asarray(mass_interior) * subSnap['m'][0].to('g').magnitude)/(r_clump_centred.magnitude * 1.496E13)
    return omega


def collect_clump_positions(snapshot,density_to_load=1e-3):
    parent_path = Path(snapshot).parent
    clump_info_file = glob.glob('%s/*.dat' % str(parent_path))[0]
    clump_info = pd.read_csv(clump_info_file,engine='python',names=['time','density','x','y','z','vx','vy','vz'],delim_whitespace=True)
    # index = np.where(clump_info['density'] == np.int(np.abs(np.log10(density_to_load)) * 10))[0][0]
    index = 6

    x = clump_info['x'][index]
    y = clump_info['y'][index]
    z = clump_info['z'][index]
    vx = clump_info['vx'][index]
    vy = clump_info['vy'][index]
    vz = clump_info['vz'][index]

    return x,y,z,vx,vy,vz


def prepare_snapshots(snapshot,x,y,z):

    snap = plonk.load_snap(snapshot)
    sinks = snap.sinks
    if type(sinks['m'].magnitude) == np.float64:
        central_star_mass = sinks['m']
    else:
        central_star_mass = sinks['m'][0]
    snap.set_units(position='au', density='g/cm^3',smoothing_length='au',velocity='km/s')
    accreted_mask = snap['smoothing_length'] > 0
    snap_active = snap[accreted_mask]
    subSnap=plonk.analysis.filters.sphere(snap=snap_active,radius = (50*au),center=(x,y,z) * au)

    subSnap.set_units(position='au', density='g/cm^3',smoothing_length='au',velocity='km/s')

    return subSnap,snap_active




if sys.argv[1] == 'clump':
    print("Loading all clump files from current directory")
    check = input("This should only be used if there is one clump present, proceed? [y/n] ")
    complete_file_list = glob.glob("run*")

elif sys.argv[1] == 'density' and len(sys.argv) == 2:
    print('No density provided, using a default value of 1E-3 g/cm')
    density_to_load = 1E-3
    pattern = str(int(np.abs(np.log10(density_to_load)) * 10)).zfill(3)
    complete_file_list = glob.glob("**/*%s.h5" % pattern)

elif sys.argv[1] == 'density'  and len(sys.argv) == 3:
    density_to_load = float(sys.argv[2])
    pattern = str(int(np.abs(np.log10(density_to_load)) * 10)).zfill(3)
    complete_file_list = glob.glob("**/*%s.h5" % pattern)

for file in tqdm(complete_file_list):
    x,y,z,vx,vy,vz = collect_clump_positions(file,density_to_load)

    subSnap = prepare_snapshots(file,x,y,z)[0]

    fullSnap = prepare_snapshots(file,x,y,z)[1]

    r_clump_centred = np.sqrt((subSnap['x']-(x*au))**2 +(subSnap['y']-(y * au))**2 + (subSnap['z']-(z* au))**2)
    r_clump_centred_midplane = np.hypot(subSnap['x']-(x*au),subSnap['y']-(y*au))

    ORIGIN = (x,y,z)  * au # The position of the clump centre
    radius_clump = np.sqrt(x**2 + y**2 + z**2)
    vel_ORIGIN = (vx,vy,vz) * kms # The velocity of the clump centre


    count = calculate_number_in_bin(r_clump_centred,subSnap['density'],100)
    mass_in_bin = np.cumsum(count[0]) * subSnap['mass'][0].to('jupiter_mass')
    mid_plane_radius = plonk.analysis.particles.mid_plane_radius(subSnap,ORIGIN,ignore_accreted=True)
    rotational_velocity_radial = plonk.analysis.particles.rotational_velocity(subSnap,vel_ORIGIN,ignore_accreted=True)
    infall_velocity_radial = plonk.analysis.particles.velocity_radial_spherical_altered(subSnap,ORIGIN,vel_ORIGIN,ignore_accreted=True)
    averaged_infall_radial = calculate_mean(r_clump_centred,infall_velocity_radial,mean_bins_radial)
    averaged_rotational_velocity = calculate_mean(r_clump_centred_midplane,rotational_velocity_radial,mean_bins_radial)
    averaged_density_radial = calculate_mean(r_clump_centred,subSnap['density'],mean_bins_radial)
    averaged_temperature_radial = calculate_mean(r_clump_centred,subSnap['my_temp'],mean_bins_radial)

    rotational_energy = 0.5 * subSnap['m'][0].to('g') * rotational_velocity_radial.to('cm/s') **2
    rotational_energy_binned = calculate_sum(r_clump_centred_midplane,rotational_energy,mean_bins_radial)
    cumsum_erot = np.cumsum(rotational_energy_binned[0])

    thermal_energy = calculate_thermal_energy(subSnap)
    thermal_energy_binned = calculate_sum(r_clump_centred_midplane,thermal_energy,mean_bins_radial)
    cumsum_etherm = np.cumsum(thermal_energy_binned[0])

    gravitational_energy = calculate_gravitational_energy(subSnap,r_clump_centred)
    gravitational_energy_binned = calculate_sum(r_clump_centred_midplane,gravitational_energy,mean_bins_radial)
    cumsum_egrav = np.cumsum(gravitational_energy_binned[0])

    with np.errstate(invalid='ignore'):
        alpha = cumsum_etherm / cumsum_egrav
    with np.errstate(invalid='ignore'):
        beta = cumsum_erot / cumsum_egrav

    smoothed_infall = gaussian_filter(averaged_infall_radial[0], sigma=1.75)

    peaks, _ = find_peaks(smoothed_infall,prominence=0.1,distance= 30)

    smoothed_rotational = gaussian_filter(averaged_rotational_velocity[0], sigma=1.75)
    smoothed_temperature = gaussian_filter(averaged_temperature_radial[0], sigma=1.75)

    for i in range(0,3):
        for j in range(0,2):
            f_radial_axs[i,j].set_xscale('log')
            f_radial_axs[i,j].set_xlim(1E-3,50)

    for i in [0,2]:
        for j in [0,1]:
            f_radial_axs[i,j].set_yscale('log')


    f_radial_axs[0,0].plot(averaged_density_radial[1][:-1],averaged_density_radial[0],linewidth=0.75)
    f_radial_axs[0,1].plot(averaged_temperature_radial[1][:-1],smoothed_temperature,linewidth=0.75)
    f_radial_axs[1,0].plot(averaged_rotational_velocity[1][:-1],smoothed_rotational,linewidth=0.75)
    f_radial_axs[1,1].plot(averaged_infall_radial[1][:-1],smoothed_infall,linewidth=0.75)
    f_radial_axs[1,1].plot(averaged_infall_radial[1][:-1][peaks],smoothed_infall[peaks],'+',c='red')

    f_radial_axs[2,0].plot(count[1][1:],mass_in_bin,linewidth=0.75)
    f_radial_axs[2,0].set_yscale('linear')

    f_radial_axs[2,1].plot(gravitational_energy_binned[1][1:],alpha,linewidth=0.75)
    f_radial_axs[2,1].plot(gravitational_energy_binned[1][1:],beta,linewidth=0.75)
    f_radial_axs[2,1].axhline(y=1,c='black',linestyle='--',linewidth=1.5)
    f_radial_axs[2,1].set_xscale('log')
    f_radial_axs[2,1].set_yscale('log')
    f_radial_axs[2,1].set_xlim(1E-3,50)
    f_radial_axs[2,1].set_ylim(1E-2,1E2)


    f_radial_axs[0,0].set_ylim(1E-13,1E-1)
    f_radial_axs[0,1].set_ylim(10,8000)
    f_radial_axs[1,1].set_ylim(0,10)
    f_radial_axs[2,0].set_ylim(0.1,40)


    f_radial_axs[0,0].set_ylabel('Density (g/cm^3)')
    f_radial_axs[1,0].set_ylabel('Rotational Velocity (km/s)')
    f_radial_axs[2,0].set_ylabel('Mass [Jupiter Masses]')
    f_radial_axs[0,1].set_ylabel('Temperature (K)')
    f_radial_axs[1,1].set_ylabel('Infall Velocity (km/s)')
    f_radial_axs[2,1].set_ylabel('Energy ratio')

    f_radial_axs[0,0].set_xlabel('R (AU)')
    f_radial_axs[1,0].set_xlabel('R (AU)')
    f_radial_axs[2,0].set_xlabel('R (AU)')
    f_radial_axs[0,1].set_xlabel('R (AU)')
    f_radial_axs[1,1].set_xlabel('R (AU)')
    f_radial_axs[2,1].set_xlabel('R (AU)')
    fig_radial.align_ylabels()
    fig_radial.tight_layout(pad=0.35)

    if sys.argv[1] == 'density':
        clump_density = '{0:.2e}'.format(density_to_load)
    else:
        clump_density = 'clump'
    second_core_radius = 0.0000000
    second_core_count  = 0.0000000
    second_core_mass   = 0.0000000
    first_core_radius  = 0.0000000
    first_core_count   = 0.0000000
    first_core_mass    = 0.0000000
    weak_fc = 0.000000



    if len(peaks) == 1:
        first_core_radius = float('{0:.5e}'.format(averaged_infall_radial[1][:-1][peaks[0]]))
        first_core_count   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(first_core_radius))[0]
        first_core_mass =   float('{0:.5e}'.format(np.cumsum(first_core_count)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))

        first_core_count   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(first_core_radius))[0]
        first_core_mass =   float('{0:.5e}'.format(np.cumsum(first_core_count)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))
    if len(peaks) == 2:
        first_core_radius = float('{0:.5e}'.format(averaged_infall_radial[1][:-1][peaks][1]))
        first_core_count   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(first_core_radius))[0]
        first_core_mass =   float('{0:.5e}'.format(np.cumsum(first_core_count)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))

        if smoothed_infall[peaks][1] < 0.5:
            weak_fc = 1
        #
        second_core_radius = float('{0:.5e}'.format(averaged_infall_radial[1][:-1][peaks][0]))
        second_core_count   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(second_core_radius))[0]
        second_core_mass =   float('{0:.5e}'.format(np.cumsum(second_core_count)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))
        #
    clump_results.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % \
                       (file.split("/")[-1],\
                       clump_density,\
                       second_core_radius,\
                       second_core_mass,\
                       first_core_radius,\
                       first_core_mass,
                       radius_clump,
                       weak_fc))
plt.savefig("%s/clump_profiles.png" % cwd,dpi = 500)
