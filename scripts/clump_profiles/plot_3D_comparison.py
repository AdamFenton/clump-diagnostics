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
import sys
from scipy.signal import savgol_filter
# Define figure
figx, axx = plt.subplots(ncols=2,nrows=2,figsize=(9,9))
figy, axy = plt.subplots(ncols=2,nrows=2,figsize=(9,9))
figz, axz = plt.subplots(ncols=2,nrows=2,figsize=(9,9))
fig_3, ax_3 = plt.subplots(figsize=(9,9))
# fig_3, ax = plt.subplots(figsize=(4,4))
# Define constants to convert to physical units
au = plonk.units('au')
kms = plonk.units('km/s')
bins = np.logspace(np.log10(5e-4),np.log10(50),120) # change the number of bins ?

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

def interpolate_across_nans(array):
     ''' Interpolate accross NaN values in arrays - this is useful when plotting
         the average of a binned quantity when the resolution is too low and the
         number of particles in the bin is zero - resulting in a NaN values for
         that bin
     '''

     interpolated= np.interp(np.arange(len(array)),
                   np.arange(len(array))[np.isnan(array) == False],
                   array[np.isnan(array) == False])

     return interpolated

def calculate_SPH_mean_x(subsnap,clump_centre,clump_velocity,bins,smoothing_factor,NEGATIVE):
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

    if NEGATIVE == True:
        multiplier = -1
    else:
        multiplier = 1

    x = (subsnap['x'].magnitude - clump_centre[0].magnitude)*multiplier
    y = subsnap['y'].magnitude - clump_centre[1].magnitude
    z = subsnap['z'].magnitude - clump_centre[2].magnitude




    R = np.sqrt(y**2 + z**2)


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
            and R[part] < smoothing_factor*(bins[bin+1] - bins[bin]):
                particles_ids[bin].append(part)
                bin_counter[bin] += 1
                infall_in_bin[bin] += infall[part].magnitude
                rotational_in_bin[bin] += rotational[part].magnitude
                temp_in_bin[bin] += subsnap['my_temp'][part].magnitude
                density_in_bin[bin] += subsnap['density'][part].magnitude


    ids = flatten_list(particles_ids)

    avg_infall = infall_in_bin/bin_counter
    avg_rotational = rotational_in_bin/bin_counter
    avg_temp = temp_in_bin/bin_counter
    avg_density = density_in_bin/bin_counter


    interp_start = np.where(avg_density == (avg_density[np.isfinite(avg_density)][0]))[0][0] - 1
    interp_end = np.where(avg_density == (avg_density[np.isfinite(avg_density)][-1]))[0][0] + 1

    return avg_infall,avg_rotational,avg_temp,avg_density, infall[ids], \
           rotational[ids],subsnap['my_temp'][ids],subsnap['density'][ids], \
           x[ids],interp_start,interp_end


def calculate_SPH_mean_y(subsnap,clump_centre,clump_velocity,bins,smoothing_factor,NEGATIVE):
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

    if NEGATIVE == True:
        multiplier = -1
    else:
        multiplier = 1

    x = subsnap['x'].magnitude - clump_centre[0].magnitude
    y = (subsnap['y'].magnitude - clump_centre[1].magnitude)*multiplier
    z = subsnap['z'].magnitude - clump_centre[2].magnitude


    R = np.sqrt(x**2 + z**2)


    rotational = plonk.analysis.particles.rotational_velocity(subsnap,
                                                              clump_velocity,
                                                              ignore_accreted=True)
    infall = plonk.analysis.particles.velocity_radial_spherical_altered(subsnap,
                                                                        clump_centre,
                                                                        clump_velocity,
                                                                        ignore_accreted=True)

    for part in range(n_part):
        for bin in range(len(bins)-1):
            if y[part] < bins[bin+1] and y[part] > bins[bin] \
            and R[part] < smoothing_factor*(bins[bin+1] - bins[bin]) :
                particles_ids[bin].append(part)
                bin_counter[bin] += 1
                infall_in_bin[bin] += infall[part].magnitude
                rotational_in_bin[bin] += rotational[part].magnitude
                temp_in_bin[bin] += subsnap['my_temp'][part].magnitude
                density_in_bin[bin] += subsnap['density'][part].magnitude


    ids = flatten_list(particles_ids)

    avg_infall = infall_in_bin/bin_counter
    avg_rotational = rotational_in_bin/bin_counter
    avg_temp = temp_in_bin/bin_counter
    avg_density = density_in_bin/bin_counter

    interp_start = np.where(avg_density == (avg_density[np.isfinite(avg_density)][0]))[0][0] - 1
    interp_end = np.where(avg_density == (avg_density[np.isfinite(avg_density)][-1]))[0][0] + 1

    return avg_infall,avg_rotational,avg_temp,avg_density, infall[ids], \
           rotational[ids],subsnap['my_temp'][ids],subsnap['density'][ids], \
           y[ids],interp_start,interp_end

def calculate_SPH_mean_z(subsnap,clump_centre,clump_velocity,bins,smoothing_factor,NEGATIVE):
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

    if NEGATIVE == True:
        multiplier = -1
    else:
        multiplier = 1

    x = subsnap['x'].magnitude - clump_centre[0].magnitude
    y = subsnap['y'].magnitude - clump_centre[1].magnitude
    z = (subsnap['z'].magnitude - clump_centre[2].magnitude)*multiplier

    R = np.sqrt(x**2 + y**2)


    rotational = plonk.analysis.particles.rotational_velocity(subsnap,
                                                              clump_velocity,
                                                              ignore_accreted=True)
    infall = plonk.analysis.particles.velocity_radial_spherical_altered(subsnap,
                                                                        clump_centre,
                                                                        clump_velocity,
                                                                        ignore_accreted=True)

    for part in range(n_part):
        for bin in range(len(bins)-1):
            if z[part] < bins[bin+1] and z[part] > bins[bin] \
            and R[part] < smoothing_factor*(bins[bin+1] - bins[bin]) :
                particles_ids[bin].append(part)
                bin_counter[bin] += 1
                infall_in_bin[bin] += infall[part].magnitude
                rotational_in_bin[bin] += rotational[part].magnitude
                temp_in_bin[bin] += subsnap['my_temp'][part].magnitude
                density_in_bin[bin] += subsnap['density'][part].magnitude


    ids = flatten_list(particles_ids)

    avg_infall = infall_in_bin/bin_counter
    avg_rotational = rotational_in_bin/bin_counter
    avg_temp = temp_in_bin/bin_counter
    avg_density = density_in_bin/bin_counter

    interp_start = np.where(avg_density == (avg_density[np.isfinite(avg_density)][0]))[0][0] - 1
    interp_end = np.where(avg_density == (avg_density[np.isfinite(avg_density)][-1]))[0][0] + 1

    return avg_infall,avg_rotational,avg_temp,avg_density, infall[ids], \
           rotational[ids],subsnap['my_temp'][ids],subsnap['density'][ids], \
           z[ids],interp_start,interp_end

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
    y_comp = plonk.analysis.filters.tube(snap = snap_active,
                                         radius = (0.5*au),
                                         length = (50*au),
                                         orientation = 'y',
                                         center = (clump_centre)
                                         )
    z_comp = plonk.analysis.filters.cylinder(snap = snap_active,
                                         radius = (0.5*au),
                                         height = (50*au),
                                         center = (clump_centre)
                                         )



    return x_comp,y_comp,z_comp,clump_centre,clump_velocity

x_comp,y_comp,z_comp,clump_centre,clump_velocity = prepare_snapshots('%s' % sys.argv[1])

print('Completed snapshot preparation')

avg_infall_x_pos,avg_rotational_x_pos,avg_temp_x_pos,avg_density_x_pos, \
infall_x_pos,rotational_x_pos,temperature_x_pos,density_x_pos, x_pos,interp_start_x_pos,interp_end_x_pos = calculate_SPH_mean_x(x_comp,
                                                                        clump_centre,
                                                                        clump_velocity,bins,10,False)

avg_infall_y_pos,avg_rotational_y_pos,avg_temp_y_pos,avg_density_y_pos, \
infall_y_pos,rotational_y_pos,temperature_y_pos,density_y_pos, y_pos,interp_start_y_pos,interp_end_y_pos = calculate_SPH_mean_y(y_comp,
                                                                        clump_centre,
                                                                        clump_velocity,bins,10,False)
avg_infall_z_pos,avg_rotational_z_pos,avg_temp_z_pos,avg_density_z_pos, \
infall_z_pos,rotational_z_pos,temperature_z_pos,density_z_pos, z_pos,interp_start_z_pos,interp_end_z_pos = calculate_SPH_mean_z(z_comp,
                                                                        clump_centre,
                                                                        clump_velocity,bins,10,False)

avg_infall_x_neg,avg_rotational_x_neg,avg_temp_x_neg,avg_density_x_neg, \
infall_x_neg,rotational_x_neg,temperature_x_neg,density_x_neg, x_neg,interp_start_x_neg,interp_end_x_neg = calculate_SPH_mean_x(x_comp,
                                                                        clump_centre,
                                                                        clump_velocity,bins,10,True)

avg_infall_y_neg,avg_rotational_y_neg,avg_temp_y_neg,avg_density_y_neg, \
infall_y_neg,rotational_y_neg,temperature_y_neg,density_y_neg, y_neg,interp_start_y_neg,interp_end_y_neg = calculate_SPH_mean_y(y_comp,
                                                                        clump_centre,
                                                                        clump_velocity,bins,10,True)
avg_infall_z_neg,avg_rotational_z_neg,avg_temp_z_neg,avg_density_z_neg, \
infall_z_neg,rotational_z_neg,temperature_z_neg,density_z_neg, z_neg,interp_start_z_neg,interp_end_z_neg = calculate_SPH_mean_z(z_comp,
                                                                        clump_centre,
                                                                        clump_velocity,bins,10,True)




figure_indexes = [(0,0),(0,1),(1,0),(1,1)]
figure_ylimits = [(1E-13,1E-2),(10,8000),(0,10),(-1,10)]
figure_ylabels = ['Density $(\\rm g\,cm^{-3})$','Temperature (K)','Rotational Velocity $(\\rm km\,s^{-1})$',
                  'Infall Velocity $(\\rm km\,s^{-1})$']

for index,label,limit in zip(figure_indexes,figure_ylabels,figure_ylimits):
    axx[index].set_ylabel(label,fontsize=10)
    axx[index].set_ylim(limit)
    axx[index].set_xlim(5e-4,30)
for index,label,limit in zip(figure_indexes,figure_ylabels,figure_ylimits):
    axy[index].set_ylabel(label,fontsize=10)
    axy[index].set_ylim(limit)
    axy[index].set_xlim(5e-4,30)
for index,label,limit in zip(figure_indexes,figure_ylabels,figure_ylimits):
    axz[index].set_ylabel(label,fontsize=10)
    axz[index].set_ylim(limit)
    axz[index].set_xlim(5e-4,30)

for i in range(2):
    for j in range(2):
        axx[i,j].set_xscale('log')
        axx[i,j].set_xlabel('x (AU)',fontsize=10)
for i in range(2):
    for j in range(2):
        axy[i,j].set_xscale('log')
        axy[i,j].set_xlabel('y (AU)',fontsize=10)
for i in range(2):
    for j in range(2):
        axz[i,j].set_xscale('log')
        axz[i,j].set_xlabel('z (AU)',fontsize=10)

axx[0,0].plot(bins[interp_start_x_pos:interp_end_x_pos],interpolate_across_nans(avg_density_x_pos)[interp_start_x_pos:interp_end_x_pos],c='red',linestyle='dotted',linewidth=0.75)
axx[0,0].plot(bins,avg_density_x_pos,c='red',alpha=0.5)
axx[0,0].set_yscale('log')
axx[0,1].plot(bins[interp_start_x_pos:interp_end_x_pos],interpolate_across_nans(avg_temp_x_pos)[interp_start_x_pos:interp_end_x_pos],c='red',linestyle='dotted',linewidth=0.75)
axx[0,1].plot(bins,avg_temp_x_pos,c='red',alpha=1)
axx[0,1].set_yscale('log')
axx[1,0].plot(bins[interp_start_x_pos:interp_end_x_pos],interpolate_across_nans(avg_rotational_x_pos)[interp_start_x_pos:interp_end_x_pos],c='red',linestyle='dotted',linewidth=0.75)
axx[1,0].plot(bins,avg_rotational_x_pos,c='red',alpha=1)
axx[1,1].plot(bins[interp_start_x_pos:interp_end_x_pos],interpolate_across_nans(avg_infall_x_pos)[interp_start_x_pos:interp_end_x_pos],c='red',linestyle='dotted',linewidth=0.75)
axx[1,1].plot(bins,avg_infall_x_pos,c='red',alpha=1)

axx[0,0].plot(bins[interp_start_x_neg:interp_end_x_neg],interpolate_across_nans(avg_density_x_neg)[interp_start_x_neg:interp_end_x_neg],c='blue',linestyle='dotted',linewidth=0.75)
axx[0,0].plot(bins,avg_density_x_neg,c='blue',alpha=0.5)
axx[0,0].set_yscale('log')
axx[0,1].plot(bins[interp_start_x_neg:interp_end_x_neg],interpolate_across_nans(avg_temp_x_neg)[interp_start_x_neg:interp_end_x_neg],c='blue',linestyle='dotted',linewidth=0.75)
axx[0,1].plot(bins,avg_temp_x_neg,c='blue',alpha=1)
axx[0,1].set_yscale('log')
axx[1,0].plot(bins[interp_start_x_neg:interp_end_x_neg],interpolate_across_nans(avg_rotational_x_neg)[interp_start_x_neg:interp_end_x_neg],c='blue',linestyle='dotted',linewidth=0.75)
axx[1,0].plot(bins,avg_rotational_x_neg,c='blue',alpha=1)
axx[1,1].plot(bins[interp_start_x_neg:interp_end_x_neg],interpolate_across_nans(avg_infall_x_neg)[interp_start_x_neg:interp_end_x_neg],c='blue',linestyle='dotted',linewidth=0.75)
axx[1,1].plot(bins,avg_infall_x_neg,c='blue',alpha=1)

axy[0,0].plot(bins[interp_start_y_pos:interp_end_y_pos],interpolate_across_nans(avg_density_y_pos)[interp_start_y_pos:interp_end_y_pos],c='red',linestyle='dotted',linewidth=0.75)
axy[0,0].plot(bins,avg_density_y_pos,c='red',alpha=1)
axy[0,0].set_yscale('log')
axy[0,1].plot(bins[interp_start_y_pos:interp_end_y_pos],interpolate_across_nans(avg_temp_y_pos)[interp_start_y_pos:interp_end_y_pos],c='red',linestyle='dotted',linewidth=0.75)
axy[0,1].plot(bins,avg_temp_y_pos,c='red',alpha=1)
axy[0,1].set_yscale('log')
axy[1,0].plot(bins[interp_start_y_pos:interp_end_y_pos],interpolate_across_nans(avg_rotational_y_pos)[interp_start_y_pos:interp_end_y_pos],c='red',linestyle='dotted',linewidth=0.75)
axy[1,0].plot(bins,avg_rotational_y_pos,c='red',alpha=1)
axy[1,1].plot(bins[interp_start_y_pos:interp_end_y_pos],interpolate_across_nans(avg_infall_y_pos)[interp_start_y_pos:interp_end_y_pos],c='red',linestyle='dotted',linewidth=0.75)
axy[1,1].plot(bins,avg_infall_y_pos,c='red',alpha=1)

axy[0,0].plot(bins[interp_start_y_neg:interp_end_y_neg],interpolate_across_nans(avg_density_y_neg)[interp_start_y_neg:interp_end_y_neg],c='blue',linestyle='dotted',linewidth=0.75)
axy[0,0].plot(bins,avg_density_y_neg,c='blue',alpha=1)
axy[0,0].set_yscale('log')
axy[0,1].plot(bins[interp_start_y_neg:interp_end_y_neg],interpolate_across_nans(avg_temp_y_neg)[interp_start_y_neg:interp_end_y_neg],c='blue',linestyle='dotted',linewidth=0.75)
axy[0,1].plot(bins,avg_temp_y_neg,c='blue',alpha=1)
axy[0,1].set_yscale('log')
axy[1,0].plot(bins[interp_start_y_neg:interp_end_y_neg],interpolate_across_nans(avg_rotational_y_neg)[interp_start_y_neg:interp_end_y_neg],c='blue',linestyle='dotted',linewidth=0.75)
axy[1,0].plot(bins,avg_rotational_y_neg,c='blue',alpha=1)
axy[1,1].plot(bins[interp_start_y_neg:interp_end_y_neg],interpolate_across_nans(avg_infall_y_neg)[interp_start_y_neg:interp_end_y_neg],c='blue',linestyle='dotted',linewidth=0.75)
axy[1,1].plot(bins,avg_infall_y_neg,c='blue',alpha=1)


axz[0,0].plot(bins[interp_start_z_pos:interp_end_z_pos],interpolate_across_nans(avg_density_z_pos)[interp_start_z_pos:interp_end_z_pos],c='red',linestyle='dotted',linewidth=0.75)
axz[0,0].plot(bins,avg_density_z_pos,c='red',alpha=1)
axz[0,0].set_yscale('log')
axz[0,1].plot(bins[interp_start_z_pos:interp_end_z_pos],interpolate_across_nans(avg_temp_z_pos)[interp_start_z_pos:interp_end_z_pos],c='red',linestyle='dotted',linewidth=0.75)
axz[0,1].plot(bins,avg_temp_z_pos,c='red',alpha=1)
axz[0,1].set_yscale('log')
axz[1,0].plot(bins[interp_start_z_pos:interp_end_z_pos],interpolate_across_nans(avg_rotational_z_pos)[interp_start_z_pos:interp_end_z_pos],c='red',linestyle='dotted',linewidth=0.75)
axz[1,0].plot(bins,avg_rotational_z_pos,c='red',alpha=1)
axz[1,1].plot(bins[interp_start_z_pos:interp_end_z_pos],interpolate_across_nans(avg_infall_z_pos)[interp_start_z_pos:interp_end_z_pos],c='red',linestyle='dotted',linewidth=0.75)
axz[1,1].plot(bins,avg_infall_z_pos,c='red',alpha=1)

axz[0,0].plot(bins[interp_start_z_neg:interp_end_z_neg],interpolate_across_nans(avg_density_z_neg)[interp_start_z_neg:interp_end_z_neg],c='blue',linestyle='dotted',linewidth=0.75)
axz[0,0].plot(bins,avg_density_z_neg,c='blue',alpha=1)
axz[0,0].set_yscale('log')
axz[0,1].plot(bins[interp_start_z_neg:interp_end_z_neg],interpolate_across_nans(avg_temp_z_neg)[interp_start_z_neg:interp_end_z_neg],c='blue',linestyle='dotted',linewidth=0.75)
axz[0,1].plot(bins,avg_temp_z_neg,c='blue',alpha=1)
axz[0,1].set_yscale('log')
axz[1,0].plot(bins[interp_start_z_neg:interp_end_z_neg],interpolate_across_nans(avg_rotational_z_neg)[interp_start_z_neg:interp_end_z_neg],c='blue',linestyle='dotted',linewidth=0.75)
axz[1,0].plot(bins,avg_rotational_z_neg,c='blue',alpha=1)
axz[1,1].plot(bins[interp_start_z_neg:interp_end_z_neg],interpolate_across_nans(avg_infall_z_neg)[interp_start_z_neg:interp_end_z_neg],c='blue',linestyle='dotted',linewidth=0.75)
axz[1,1].plot(bins,avg_infall_z_neg,c='blue',alpha=1)

figx.savefig('x_profiles-.png',dpi=200)
figy.savefig('y_profiles-.png',dpi=200)
figz.savefig('z_profiles-.png',dpi=200)
