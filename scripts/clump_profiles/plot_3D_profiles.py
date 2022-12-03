# ---------------------------- #
# plot_3D_profiles.py
# ---------------------------- #
# Plot the profiles of fragments
# in x y and z to get an
# understanding of the 3D
# structure of the fragments
# ---------------------------- #
# Author: Adam Fenton
# Date 20221123
# ---------------------------- #
import plonk
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import pint
from scipy.signal import savgol_filter
# Define constants to convert to physical units
au = plonk.units('au')
kms = plonk.units('km/s')
bins = np.logspace(np.log10(5e-4),np.log10(50),75) # change the number of bins ?

if hasattr(pint, 'UnitStrippedWarning'):
    warnings.simplefilter('ignore', category=pint.UnitStrippedWarning)
np.seterr(divide='ignore', invalid='ignore')

# Define axes
def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:

            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list
def calculate_number_in_bin(binned_quantity,mean_quantity):
    return stats.binned_statistic(binned_quantity, mean_quantity, 'count', bins=bins)
def calculate_sum(binned_quantity,summed_quantity):
    return stats.binned_statistic(binned_quantity, summed_quantity, 'sum', bins=bins)

def calculate_SPH_mean(subsnap,clump_centre,clump_velocity,bins):
    bin_counter = np.zeros(len(bins)) # Keep track of number of particles in each bin
    # Keep track of total temperature, density, rotational velocity and infall
    # velocity in each bin
    temp_in_bin = np.zeros(len(bins))
    density_in_bin =  np.zeros(len(bins))
    infall_in_bin =  np.zeros(len(bins))
    rotational_in_bin =  np.zeros(len(bins))
    avg_temp = np.zeros(len(bins))
    avg_density =  np.zeros(len(bins))
    avg_infall =  np.zeros(len(bins))
    avg_rotational =  np.zeros(len(bins))


    n_part = len(subsnap['m'])
    particles_ids = [[] for _ in range(len(bins))]

    x = subsnap['x'].magnitude - clump_centre[0].magnitude
    y = subsnap['y'].magnitude - clump_centre[1].magnitude
    z = subsnap['z'].magnitude - clump_centre[2].magnitude

    R = np.sqrt(z**2 + y**2)


    rotational = plonk.analysis.particles.rotational_velocity(subsnap,
                                                              clump_velocity,
                                                              ignore_accreted=True)
    infall = plonk.analysis.particles.velocity_radial_spherical_altered(subsnap,
                                                                        clump_centre,
                                                                        clump_velocity,
                                                                        ignore_accreted=True)

    for part in range(n_part):
        for bin in range(len(bins)-1):
            if x[part] < bins[bin+1] and x[part] > bins[bin] \
            and R[part] < 10*(bins[bin+1] - bins[bin]) :
                particles_ids[bin].append(part)
                bin_counter[bin] += 1
                infall_in_bin[bin] += infall[part].magnitude



    ids = flatten_list(particles_ids)

    avg_infall = infall_in_bin/bin_counter

    return avg_infall, infall[ids], x[ids]
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
    accreted_mask = snap['smoothing_length'] > 0
    snap_active = snap[accreted_mask]

    # Find the position and velocity of the clump centre using the PID from the
    # filename. Because python indicies start at 0 (and fortran at 1), we have
    # to subtract 1 from the PID to get the particle in the arrays.
    PID_index = np.where(snap_active['id'] == (int(snapshot.split('.')[2]) - 1))
    clump_centre = snap_active['position'][PID_index][0]
    clump_velocity= snap_active['velocity'][PID_index][0]

    # Create subsnaps for the x, y and z componants
    x_comp = plonk.analysis.filters.tube(snap = snap_active,
                                         radius = (0.5*au),
                                         length = (50*au),
                                         orientation = 'x',
                                         center = (clump_centre)
                                         )



    return x_comp,clump_centre, clump_velocity

x_comp,clump_centre, clump_velocity = prepare_snapshots('run1.001.0878138.030.h5')
print('Completed snapshot preparation')
avg_infall, infall, x =calculate_SPH_mean(x_comp,clump_centre,clump_velocity,bins)
plt.scatter(x,infall,s=0.1)
plt.plot(bins,avg_infall,c='red')
plt.xscale('log')
plt.show()
#                                                              clump_velocity,bins)
# avg_density_x, avg_temp_x, avg_infall_x, avg_rotational_x, density_x, \
# temperature_x, rotational_x, infall_x, x = calculate_SPH_mean(x_comp,clump_centre,
#                                                              clump_velocity,bins)
#
# print('Completed X component averages')
# figure_indexes = [(0,0),(0,1),(1,0),(1,1)]
# figure_ylimits = [(1E-13,1E-2),(10,8000),(0,10),(0,10)]
# figure_ylabels = ['Density $(\\rm g\,cm^{-3})$','Temperature (K)','Rotational Velocity $(\\rm km\,s^{-1})$',
#                   'Infall Velocity $(\\rm km\,s^{-1})$']
#
# for index,label,limit in zip(figure_indexes,figure_ylabels,figure_ylimits):
#     axx[index].set_ylabel(label,fontsize=10)
#     axx[index].set_ylim(limit)
#
# for i in range(0,2):
#     for j in range(0,2):
#         axx[i,j].set_xscale('log')
#         axx[i,j].set_xlim(5E-4,50)
#         axx[i,j].set_xlabel('x (AU)',fontsize=10)
#         axx[i,j].tick_params(axis="x", labelsize=8)
#         axx[i,j].tick_params(axis="y", labelsize=8)
#
# for index,label,limit in zip(figure_indexes,figure_ylabels,figure_ylimits):
#     axy[index].set_ylabel(label,fontsize=10)
#     axy[index].set_ylim(limit)
#
#
# print('Plotting...')
# axx[0,0].scatter(x,density_x,s=0.1,c='black')
# axx[0,0].plot(bins,avg_density_x,c='red')
# axx[0,0].set_yscale('log')
# axx[0,1].scatter(x,temperature_x,s=0.1,c='black')
# axx[0,1].plot(bins,avg_temp_x,c='red')
# axx[0,1].set_yscale('log')
# axx[1,0].scatter(x,rotational_x,s=0.1,c='black')
# axx[1,0].plot(bins,avg_rotational_x,c='red')
# axx[1,1].scatter(x,infall_x,s=0.1,c='black')
# axx[1,1].plot(bins,avg_infall_x,c='red')
# figx.savefig('X_profiles.png',dpi=200)
