import plonk
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from scipy import stats
import pint
from tqdm.auto import tqdm
import warnings
from scipy.signal import savgol_filter
from digitize import solution

import time
# Author: Adam Fenton
start_time = time.time()
cwd = os.getcwd()
# Load units for use later, useful for derived quantities.
au = plonk.units('au')




density_to_load = None
mean_bins_radial = np.logspace(np.log10(0.001),np.log10(50),120)


# Initalise the figure and output file which is written to later on
fig_radial, f_radial_axs = plt.subplots(nrows=3,ncols=2,figsize=(7,8))


clump_results = open('clump-results.dat', 'w')

# Ignore pesky warnings when stripping unit off of pint quantity when downcasting to array
if hasattr(pint, 'UnitStrippedWarning'):
    warnings.simplefilter('ignore', category=pint.UnitStrippedWarning)
np.seterr(divide='ignore', invalid='ignore')

def calculate_sum(binned_quantity,summed_quantity,bins):
    return stats.binned_statistic(binned_quantity, summed_quantity, 'sum', bins=bins)

def calculate_mean(binned_quantity,mean_quantity):
    bins = np.logspace(np.log10(0.001),np.log10(50),120)
    return stats.binned_statistic(binned_quantity, mean_quantity, 'mean', bins=bins)


def calculate_number_in_bin(binned_quantity,mean_quantity,width):
    bins=np.logspace(np.log10(0.001),np.log10(width),120)
    return stats.binned_statistic(binned_quantity, mean_quantity, 'count', bins=bins)

def calculate_thermal_energy(subSnap):
    U = 3/2 * 1.38E-16 * subSnap['my_temp'] * ((subSnap['m'][0].to('g'))/(1.67E-24))

    U = U.magnitude

    U *= ((subSnap['my_temp'].magnitude>2000)*(1/1.2)+(subSnap['my_temp'].magnitude<2000)*(1/2.381))
    return U





def prepare_snapshots(snapshot):
    ''' Load full snapshot as plonk object and initialise subsnap centred on clump.
        Also apply filter to exclude dead and accreted particles with `accreted_mask`
        and load units
    '''

    snap = plonk.load_snap(snapshot)
    sinks = snap.sinks
    if type(sinks['m'].magnitude) == np.float64:
        central_star_mass = sinks['m']
    else:
        central_star_mass = sinks['m'][0]
    snap.set_units(position='au', density='g/cm^3',smoothing_length='au',velocity='km/s')
    h = snap['smoothing_length']
    max_elem = np.amax(snap['density'][h>0])
    id = np.where(snap['density']== max_elem)
    clump_centre = snap['position'][id]
    clump_velocity = snap['velocity'][id]
    accreted_mask = snap['smoothing_length'] > 0
    snap_active = snap[accreted_mask]
    subSnap=plonk.analysis.filters.sphere(snap=snap_active,radius = (50*au),center=clump_centre)

    subSnap.set_units(position='au', density='g/cm^3',smoothing_length='au',velocity='km/s')

    return subSnap,snap_active,clump_centre,clump_velocity



# Catch case where user does not supply mode argument when running script from the command line
try:
    print('Running script in',sys.argv[1], 'mode')

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

except IndexError:
    print("Plotting mode not provided...exiting")
    sys.exit(1)






mpl_colour_defaults=plt.rcParams['axes.prop_cycle'].by_key()['color']
for file in tqdm(complete_file_list):
    index = complete_file_list.index(file)
    line_colour = mpl_colour_defaults[index]
    prepared_snapshots = prepare_snapshots(file)
    ORIGIN = prepared_snapshots[2][0] # The position of the clump centre
    x,y,z = ORIGIN[0],ORIGIN[1],ORIGIN[2]
    vel_ORIGIN = prepared_snapshots[3][0]# The velocity of the clump centre

    subSnap = prepared_snapshots[0]


    r_clump_centred = np.sqrt((subSnap['x']-(x))**2 +(subSnap['y']-(y))**2 + (subSnap['z']-(z))**2)
    r_clump_centred_midplane = np.hypot(subSnap['x']-(x),subSnap['y']-(y))

    radius_clump = np.sqrt((x)**2 + (y)**2 + (z)**2)


    count = calculate_number_in_bin(r_clump_centred,subSnap['m'],50)
    mass_in_bin = np.cumsum(count[0]) * subSnap['mass'][0].to('jupiter_mass')
    mid_plane_radius = plonk.analysis.particles.mid_plane_radius(subSnap,ORIGIN,ignore_accreted=True)
    rotational_velocity_radial = plonk.analysis.particles.rotational_velocity(subSnap,vel_ORIGIN,ignore_accreted=True)
    infall_velocity_radial = plonk.analysis.particles.velocity_radial_spherical_altered(subSnap,ORIGIN,vel_ORIGIN,ignore_accreted=True)

    averaged_infall_radial = calculate_mean(r_clump_centred,infall_velocity_radial)
    averaged_rotational_velocity = calculate_mean(r_clump_centred_midplane,rotational_velocity_radial)
    averaged_density_radial = calculate_mean(r_clump_centred,subSnap['density'])
    averaged_temperature_radial = calculate_mean(r_clump_centred,subSnap['my_temp'])


    elems = [i for i, a in enumerate(count[0]) if a <= 50]

    def highlight_low_confidence_bins(quantity,elems):
        ''' A function that, when provided with an input quantity in the form of
            a binned_statistic and an array of elements, returns two arrays that
            are copies of the input array but with bins with fewer than 50 particles
            in changed to nan values. We copy the arrays so the originals remain
            unaltered.
        '''
        R_y = quantity[0].copy()     # Resulting arrays are initally copies of the input arrays.
        R_x = quantity[1][1:].copy()

        for elem in elems:
            R_x[elem] = np.nan
            R_y[elem] = np.nan

        return R_x, R_y



    binned_r_clump_with_nans = highlight_low_confidence_bins(averaged_infall_radial,elems)[0]
    average_temp_with_nans = highlight_low_confidence_bins(averaged_temperature_radial,elems)[1]
    average_density_with_nans = highlight_low_confidence_bins(averaged_density_radial,elems)[1]
    average_infall_with_nans = highlight_low_confidence_bins(averaged_infall_radial,elems)[1]
    average_rotational_with_nans = highlight_low_confidence_bins(averaged_rotational_velocity,elems)[1]


    averaged_infall_radial_interp = np.interp(np.arange(len(averaged_infall_radial[0])),
                                    np.arange(len(averaged_infall_radial[0]))[np.isnan(averaged_infall_radial[0]) == False],
                                    averaged_infall_radial[0][np.isnan(averaged_infall_radial[0]) == False])

    averaged_rotational_velocity_interp = np.interp(np.arange(len(averaged_rotational_velocity[0])),
                                    np.arange(len(averaged_rotational_velocity[0]))[np.isnan(averaged_rotational_velocity[0]) == False],
                                    averaged_rotational_velocity[0][np.isnan(averaged_rotational_velocity[0]) == False])

    averaged_density_radial_interp = np.interp(np.arange(len(averaged_density_radial[0])),
                                    np.arange(len(averaged_density_radial[0]))[np.isnan(averaged_density_radial[0]) == False],
                                    averaged_density_radial[0][np.isnan(averaged_density_radial[0]) == False])

    averaged_temperature_radial_interp = np.interp(np.arange(len(averaged_temperature_radial[0])),
                                    np.arange(len(averaged_temperature_radial[0]))[np.isnan(averaged_temperature_radial[0]) == False],
                                    averaged_temperature_radial[0][np.isnan(averaged_temperature_radial[0]) == False])









    rotational_energy = 0.5 * subSnap['m'][0].to('g') * rotational_velocity_radial.to('cm/s') **2
    rotational_energy_binned = calculate_sum(r_clump_centred_midplane,rotational_energy,mean_bins_radial)
    cumsum_erot = np.cumsum(rotational_energy_binned[0])
    grav_rad = r_clump_centred.magnitude

    gravitational_energy = solution(grav_rad,mean_bins_radial,subSnap['m'][0].to('g'))
    gravitational_energy_binned = calculate_sum(r_clump_centred_midplane,gravitational_energy,mean_bins_radial)
    cumsum_egrav = np.cumsum(gravitational_energy_binned[0])

    thermal_energy = calculate_thermal_energy(subSnap)
    thermal_energy_binned = calculate_sum(r_clump_centred_midplane,thermal_energy,mean_bins_radial)
    cumsum_etherm = np.cumsum(thermal_energy_binned[0])

    with np.errstate(invalid='ignore'):
        alpha = cumsum_etherm / cumsum_egrav

    with np.errstate(invalid='ignore'):
        beta = cumsum_erot / cumsum_egrav





    # smoothed_rotational     = savgol_filter(averaged_rotational_velocity[0],11,5)
    # smoothed_temperature    = savgol_filter(averaged_temperature_radial[0],11,5)
    # smoothed_density        = savgol_filter(averaged_density_radial[0],11,5)
    smoothed_infall              = savgol_filter(averaged_infall_radial[0],15,3)
    smoothed_infall_nans         = savgol_filter(average_infall_with_nans ,15,3)
    peaks, _ = find_peaks(smoothed_infall,width=3,distance=25,prominence=0.5)

    # Tidily set axes limits and scale types
    for i in range(0,3):
        for j in range(0,2):
            f_radial_axs[i,j].set_xscale('log')
            f_radial_axs[i,j].set_xlim(1E-4,50)

    for i in [0,2]:
        for j in [0,1]:
            f_radial_axs[i,j].set_yscale('log')


    f_radial_axs[0,0].plot(averaged_density_radial[1][1:],averaged_density_radial[0],c = line_colour,linestyle="--",linewidth = 1)
    f_radial_axs[0,0].plot(binned_r_clump_with_nans,average_density_with_nans,c = line_colour)
    f_radial_axs[0,1].plot(averaged_temperature_radial[1][1:],averaged_temperature_radial[0],c = line_colour,linestyle="--",linewidth =1)
    f_radial_axs[0,1].plot(binned_r_clump_with_nans,average_temp_with_nans,c = line_colour)
    f_radial_axs[1,0].plot(averaged_rotational_velocity[1][1:],averaged_rotational_velocity[0],c = line_colour,linestyle="--",linewidth = 1)
    f_radial_axs[1,0].plot(binned_r_clump_with_nans,average_rotational_with_nans,c = line_colour)

    f_radial_axs[1,1].plot(averaged_infall_radial[1][1:],smoothed_infall,c = line_colour,linestyle="--",linewidth = 1)
    f_radial_axs[1,1].plot(binned_r_clump_with_nans,smoothed_infall_nans ,c = line_colour)




    f_radial_axs[1,1].plot(averaged_infall_radial[1][1:][peaks],smoothed_infall[peaks],'+',c='red')

    f_radial_axs[2,0].plot(count[1][1:],mass_in_bin,linewidth=1)
    f_radial_axs[2,0].set_yscale('linear')

    f_radial_axs[2,1].plot(gravitational_energy_binned[1][1:],alpha,linewidth=1,c=line_colour)
    f_radial_axs[2,1].plot(gravitational_energy_binned[1][1:],beta,linewidth=1,c=line_colour)
    f_radial_axs[2,1].axhline(y=1,c='black',linestyle='--',linewidth=1.5)
    f_radial_axs[2,1].set_xscale('log')
    f_radial_axs[2,1].set_yscale('log')
    f_radial_axs[2,1].set_xlim(1E-4,50)
    f_radial_axs[2,1].set_ylim(1E-2,1E2)


    f_radial_axs[0,0].set_ylim(1E-13,1E-1)
    f_radial_axs[0,1].set_ylim(10,8000)
    f_radial_axs[1,1].set_ylim(-5,10)
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
    rhocritID = 1


    # Write core information to the clump_results file for plotting later on.
    if len(peaks) == 1:
        first_core_radius = float('{0:.5e}'.format(averaged_infall_radial[1][:-1][peaks[0]]))
        first_core_count   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(first_core_radius))[0]
        first_core_mass =   float('{0:.5e}'.format(np.cumsum(first_core_count)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))

        first_core_count   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(first_core_radius))[0]
        first_core_mass =   float('{0:.5e}'.format(np.cumsum(first_core_count)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))
    if len(peaks) >= 2:
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

    clump_results.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % \
                       (file.split("/")[-1],\
                       clump_density,\
                       second_core_radius,\
                       second_core_mass,\
                       first_core_radius,\
                       first_core_mass,
                       radius_clump,
                       weak_fc,
                       rhocritID))

print(time.time()-start_time)
plt.show()
# plt.savefig("%s/clump_profiles.png" % cwd,dpi = 500)
