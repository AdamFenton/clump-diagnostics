import plonk
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys
import glob
from pathlib import Path
from mpl_toolkits import mplot3d
from tqdm.auto import tqdm, trange
cwd=os.getcwd()
Path("%s/plots/clump_profiles" % cwd).mkdir(parents=True, exist_ok=True )

mpl_colour_defaults=plt.rcParams['axes.prop_cycle'].by_key()['color'] # MLP default colours

au = plonk.units('au')

kms =  plonk.units('km/s')

def calculate_mean(binned_quantity,mean_quantity):
    ''' This function, when called, calculates the mean value of "mean_quantity"
        in each bin, binned from "binned_quantity" and returns:

        result[0] = statistic - the mean values of "mean_quantity"
        result[1] = bin_edges - the edges of the bins in the x direction
        result[2] = bin_number - the indices of the bin edges

        e.g:

        averaged_density = calculate_mean(subSnap['radius_new'],subSnap['density'])
        averaged_density[0] are the mean values on density
        averaged_density[1] are the bin edges, in this case these are the radii values.

        Since this function uses bin edges, we need to ignore the final value in
        averaged_density[1] as this is the very outer edge. Ignore this when plotting
        by indexing this array with "averaged_density[1][:-1]"

    '''
    bins=np.logspace(np.log10(0.001),np.log10(50.0), 100)
    return stats.binned_statistic(binned_quantity, mean_quantity, 'mean', bins=bins)

def calculate_number_in_bin(binned_quantity,mean_quantity):
    ''' This function, when called, calculates the mean value of "mean_quantity"
        in each bin, binned from "binned_quantity" and returns:

        result[0] = statistic - the mean values of "mean_quantity"
        result[1] = bin_edges - the edges of the bins in the x direction
        result[2] = bin_number - the indices of the bin edges

        e.g:

        averaged_density = calculate_mean(subSnap['radius_new'],subSnap['density'])
        averaged_density[0] are the mean values on density
        averaged_density[1] are the bin edges, in this case these are the radii values.

        Since this function uses bin edges, we need to ignore the final value in
        averaged_density[1] as this is the very outer edge. Ignore this when plotting
        by indexing this array with "averaged_density[1][:-1]"

    '''
    bins=np.logspace(np.log10(0.001),np.log10(50.0), 50)
    return stats.binned_statistic(binned_quantity, mean_quantity, 'count', bins=bins)
def calculate_sum(binned_quantity,summed_quantity,bins):
    return stats.binned_statistic(binned_quantity, summed_quantity, 'sum', bins=bins)

def calculate_mean(binned_quantity,mean_quantity):
    bins = np.logspace(np.log10(0.001),np.log10(50),75)
    return stats.binned_statistic(binned_quantity, mean_quantity, 'mean', bins=bins)

def calculate_number_in_bin(binned_quantity,mean_quantity,width):
    bins=np.logspace(np.log10(0.001),np.log10(width),75)
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
    x,y,z = clump_centre[0][0].magnitude,clump_centre[0][1].magnitude,clump_centre[0][2].magnitude

    clump_velocity = snap['velocity'][id]
    accreted_mask = snap['smoothing_length'] > 0
    snap_active = snap[accreted_mask]
    subSnap   = plonk.analysis.filters.sphere(snap=snap_active,radius = (50*au),center=clump_centre)
    subSnap_x = plonk.analysis.filters.tube(snap=subSnap,radius = (0.1*au),length=(10*au),orientation='x',center= (x,y,z) * au)
    subSnap_y = plonk.analysis.filters.tube(snap=subSnap,radius = (0.1*au),length=(10*au),orientation='y',center= (x,y,z) * au)
    subSnap_z = plonk.analysis.filters.cylinder(snap=subSnap,radius = (0.1*au),height=(10*au),center= (x,y,z) * au)

    subSnap.set_units(position='au', density='g/cm^3',smoothing_length='au',velocity='km/s')

    return subSnap,clump_centre,clump_velocity,subSnap_x,subSnap_y,subSnap_z


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



for file in tqdm(complete_file_list):
    # fig7, f7_axs = plt.subplots(nrows=3,figsize=(8,8))
    # fig1, ax1 = plt.subplots(figsize=(12,8))
    fig, ax = plt.subplots(nrows=3,figsize=(7,8))
    index = complete_file_list.index(file)
    line_colour = mpl_colour_defaults[index]
    prepared_snapshots = prepare_snapshots(file)
    ORIGIN = prepared_snapshots[1][0] # The position of the clump centre
    x,y,z = ORIGIN[0],ORIGIN[1],ORIGIN[2]
    vel_ORIGIN = prepared_snapshots[1][0]# The velocity of the clump centre

    subSnap = prepared_snapshots[0]
    SS_x = prepared_snapshots[3]
    SS_y = prepared_snapshots[4]
    SS_z = prepared_snapshots[5]
    # r_clump_centred_SSX = np.sqrt((SS_x['x']-x)**2 + \
    #                               (SS_x['y']-y)**2 + \
    #                               (SS_x['z']-z)**2)
    #
    # r_clump_centred_SSY = np.sqrt((SS_y['x']-x)**2 + \
    #                               (SS_y['y']-y)**2 + \
    #                               (SS_y['z']-z)**2)
    #
    # r_clump_centred_SSZ = np.sqrt((SS_z['x']-x)**2 + \
    #                               (SS_z['y']-y)**2 + \
    #                               (SS_z['z']-z)**2)
    r_clump_centred_SSX = np.abs(SS_x['x']-x)
    r_clump_centred_SSY = np.abs(SS_y['y']-y)
    r_clump_centred_SSZ = np.abs(SS_z['z']-z)


    averaged_density_radial_SSX = calculate_mean(r_clump_centred_SSX,SS_x['density'])
    averaged_density_radial_SSY = calculate_mean(r_clump_centred_SSY,SS_y['density'])
    averaged_density_radial_SSZ = calculate_mean(r_clump_centred_SSZ,SS_z['density'])

    averaged_temperature_radial_SSX = calculate_mean(r_clump_centred_SSX,SS_x['my_temp'])
    averaged_temperature_radial_SSY = calculate_mean(r_clump_centred_SSY,SS_y['my_temp'])
    averaged_temperature_radial_SSZ = calculate_mean(r_clump_centred_SSZ,SS_z['my_temp'])

    averaged_density_radial_SSX = calculate_mean(r_clump_centred_SSX,SS_x['density'])
    averaged_density_radial_SSY = calculate_mean(r_clump_centred_SSY,SS_y['density'])
    averaged_density_radial_SSZ = calculate_mean(r_clump_centred_SSZ,SS_z['density'])




    # ax.scatter(r_clump_centred_SSX,SS_x['density'],s=5)
    ax[0].plot(averaged_density_radial_SSX[1][1:],averaged_density_radial_SSX[0])
    ax[0].plot(averaged_density_radial_SSY[1][1:],averaged_density_radial_SSY[0])
    ax[0].plot(averaged_density_radial_SSZ[1][1:],averaged_density_radial_SSZ[0])

    ax[1].plot(averaged_temperature_radial_SSX[1][1:],averaged_temperature_radial_SSX[0])
    ax[1].plot(averaged_temperature_radial_SSY[1][1:],averaged_temperature_radial_SSY[0])
    ax[1].plot(averaged_temperature_radial_SSZ[1][1:],averaged_temperature_radial_SSZ[0])

    ax[1].scatter(r_clump_centred_SSZ,SS_z['my_temp'],s=5)

    # ax.scatter(r_clump_centred_SSZ,SS_z['density'],s=5,marker='+')


    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xscale('log')


    # ax.scatter(subSnap['x'],subSnap['z'],c='k',s=0.2)
    # ax.scatter(SS_x['x'],SS_x['z'])
    # ax.set_aspect('equal', 'box')
    plt.show()

    # clump_ids = (subSnap['id'].magnitude).copy()
    #
    #
    # r_clump_centred = np.sqrt((subSnap['x']-position[0][0])**2 +(subSnap['y']-position[0][1])**2 + (subSnap['z']-position[0][2])**2)
    #
    #
    #
    #
    #
    #
    # @snap.add_array()
    # def plonk_infall_velocity_altered(subSnap):
    #     x = position[0][0].to('km')
    #     y = position[0][1].to('km')
    #     z = position[0][2].to('km')
    #     vx = velocity[0][0].to('km/s')
    #     vy = velocity[0][1].to('km/s')
    #     vz = velocity[0][2].to('km/s')
    #     pos = subSnap['position'].to('km')
    #     vel = subSnap['velocity']
    #
    #     ORIGIN = (position[0].to('km'))
    #     vel_ORIGIN = (velocity[0].to('km/s'))
    #     pos = pos - ORIGIN
    #     vel = vel - vel_ORIGIN
    #
    #     x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    #     vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]
    #
    #     return (x * vx + y * vy + z * vz) / np.sqrt(x ** 2 + y ** 2 + z ** 2) * -1
    #
    #
    #
    #
    # averaged_density_radial = calculate_mean(r_clump_centred,subSnap['density'])
    # averaged_temperature_radial = calculate_mean(r_clump_centred,subSnap['my_temp'])
    # averaged_infall_radial = calculate_mean(r_clump_centred,subSnap['plonk_infall_velocity_altered'])
    #
    #
    # f7_axs[0].plot(averaged_density_radial[:-1],averaged_density_radial,c='r')
    # f7_axs[0].scatter(r_clump_centred,subSnap['density'].to('g/cm^3'),s=0.1,c='k')
    # f7_axs[1].plot(averaged_temperature_radial[:-1],averaged_temperature_radial,c='r')
    # f7_axs[1].scatter(r_clump_centred,subSnap['my_temp'],s=0.1,c='k')
    # f7_axs[2].plot(averaged_infall_radial[:-1],averaged_infall_radial,c='r')
    # f7_axs[2].scatter(r_clump_centred,subSnap['plonk_infall_velocity_altered'],s=0.1,c='k')
    #
    #
    #
    #
    # f7_axs[0].set_ylabel('Density [g/cm^3]')
    # f7_axs[0].set_yscale('log')
    # f7_axs[0].set_xscale('log')
    # f7_axs[0].set_xlim(1e-3,15)
    # f7_axs[0].set_ylim(1e-14,1e-3)
    # f7_axs[0].tick_params(labelbottom=False)
    # #
    #
    # f7_axs[1].set_ylabel('Temperature [K]')
    # f7_axs[1].set_yscale('log')
    # f7_axs[1].set_xscale('log')
    # f7_axs[1].set_xlim(1e-3,15)
    # f7_axs[1].set_ylim(1e1,4e3)
    # f7_axs[1].tick_params(labelbottom=False)
    #
    # f7_axs[2].set_ylabel('Infall Velocity [km/s]')
    # f7_axs[2].set_xlabel('R [AU] (from clump centre)')
    # f7_axs[2].set_xscale('log')
    # f7_axs[2].set_xlim(1e-3,15)
    # f7_axs[2].set_ylim(0,5)
    #
    #
    #
    # # plt.show()
    # plt.savefig("%s/plots/clump_profiles/%s.png" % (cwd,i))
    # plt.close()
