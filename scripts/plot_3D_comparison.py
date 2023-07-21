# ---------------------------- #
# plot_3D_profiles.py
# ---------------------------- #
# Plot the 3D structure of the
# fragments on the same plot
# and compare with the spherical
# average
# ---------------------------- #
# Author: Adam Fenton
# Date 20221123
# ---------------------------- #
import plonk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import pint
import sys
import csv
from scipy.interpolate import splev, splrep
from scipy.signal import savgol_filter, find_peaks
from threading import Thread
from tqdm.auto import tqdm
# Define figure
fig_main, ax_main = plt.subplots(ncols=2,nrows=2,figsize=(9,9))
fig_test, ax_test = plt.subplots(figsize=(9,9))

# Define constants to convert to physical units
au = plonk.units('au')
kms = plonk.units('km/s')
bins = np.logspace(np.log10(5e-4),np.log10(50),100) # change the number of bins ?
x_smooth = np.logspace(np.log10(5e-4),np.log10(50),2000)
if hasattr(pint, 'UnitStrippedWarning'):
    warnings.simplefilter('ignore', category=pint.UnitStrippedWarning)
np.seterr(divide='ignore', invalid='ignore')
def apply_filter(array):

    smoothed_array = savgol_filter(array,10,2)
    return smoothed_array

def format_core_radii(peaks,minima):
    Rsc,Rfci,Rfco,delta_shock = x_smooth[peaks][0],\
                                x_smooth[minima], \
                                x_smooth[peaks][1],\
                                x_smooth[peaks][1] - x_smooth[minima]


    return Rsc,Rfci,Rfco,delta_shock


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

def calculate_mean(binned_quantity,mean_quantity):
    return stats.binned_statistic(binned_quantity, mean_quantity, 'mean', bins=bins)

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

    regular_avg_infall = calculate_mean(x,infall)
    regular_avg_rotational = calculate_mean(x,rotational)
    regular_avg_temp = calculate_mean(x,subsnap['my_temp'])
    regular_avg_density = calculate_mean(x,subsnap['density'])

    interp_start = np.where(avg_density == (avg_density[np.isfinite(avg_density)][0]))[0][0] - 1
    interp_end = np.where(avg_density == (avg_density[np.isfinite(avg_density)][-1]))[0][0] + 1

    return avg_infall,avg_rotational,avg_temp,avg_density, infall[ids], \
           rotational[ids],subsnap['my_temp'][ids],subsnap['density'][ids], \
           x[ids],interp_start,interp_end,regular_avg_infall,regular_avg_rotational,\
           regular_avg_temp,regular_avg_density

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

    interp_start = np.where(apply_filter(avg_density) == (apply_filter(avg_density)[np.isfinite(apply_filter(avg_density))][0]))[0][0] - 1
    interp_end = np.where(apply_filter(avg_density)== (apply_filter(avg_density)[np.isfinite(apply_filter(avg_density))][-1]))[0][0] + 1

    # return apply_filter(avg_infall),apply_filter(avg_rotational),apply_filter(avg_temp),apply_filter(avg_density), infall[ids], \
    #        rotational[ids],subsnap['my_temp'][ids],subsnap['density'][ids], \
    #        z[ids],interp_start,interp_end

    return avg_infall,avg_rotational,avg_temp,avg_density, infall[ids], \
        rotational[ids],subsnap['my_temp'][ids],subsnap['density'][ids], \
        z[ids],interp_start,interp_end


def spherical_average(subsnap,bins):
    x = subsnap['x'].magnitude - clump_centre[0].magnitude
    y = subsnap['y'].magnitude - clump_centre[1].magnitude
    z = subsnap['z'].magnitude - clump_centre[2].magnitude

    R = np.sqrt(x**2 + y**2 + z**2)

    rotational = plonk.analysis.particles.rotational_velocity(subsnap,
                                                              clump_velocity,
                                                              ignore_accreted=True)
    infall = plonk.analysis.particles.velocity_radial_spherical_altered(subsnap,
                                                                        clump_centre,
                                                                        clump_velocity,
                                                                        ignore_accreted=True)

    avg_infall = calculate_mean(R,infall)[0]
    avg_rotational = calculate_mean(R,rotational)[0]
    avg_density = calculate_mean(R,subsnap['density'])[0]
    avg_temperature = calculate_mean(R,subsnap['my_temp'])[0]
    R = calculate_mean(R,infall)[1][1:]

    interp_start = np.where(avg_density == (avg_density[np.isfinite(avg_density)][0]))[0][0] - 1
    interp_end = np.where(avg_density == (avg_density[np.isfinite(avg_density)][-1]))[0][0] + 1


    return avg_infall,avg_rotational,avg_density,avg_temperature, R,interp_start,interp_end

def axisym_average(subsnap,clump_centre,clump_velocity,bins,smoothing_factor):
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

    R = np.sqrt(x**2 + y**2)


    rotational = plonk.analysis.particles.rotational_velocity(subsnap,
                                                              clump_velocity,
                                                              ignore_accreted=True)
    infall = plonk.analysis.particles.velocity_radial_spherical_altered(subsnap,
                                                                        clump_centre,
                                                                        clump_velocity,
                                                                        ignore_accreted=True)




    R_binned = np.histogram(R,bins)
    R_digitized = np.digitize(R,bins)

    for bin in tqdm(range(len(bins)-1),miniters=1):
        for particle in np.where(R_digitized==bin)[0]:
            if np.abs(z[particle]) < bins[bin+1]-bins[bin]:
                particles_ids[bin].append(particle)
                bin_counter[bin] += 1
                infall_in_bin[bin] += infall[particle].magnitude
                rotational_in_bin[bin] += rotational[particle].magnitude
                temp_in_bin[bin] += subsnap['my_temp'][particle].magnitude
                density_in_bin[bin] += subsnap['density'][particle].magnitude

    ids = flatten_list(particles_ids)

    avg_infall = infall_in_bin/bin_counter
    avg_rotational = rotational_in_bin/bin_counter
    avg_temp = temp_in_bin/bin_counter
    avg_density = density_in_bin/bin_counter




    # interp_start = np.where(avg_density == (avg_density[np.isfinite(avg_density)][0]))[0][0] - 1
    # interp_end = np.where(avg_density == (avg_density[np.isfinite(avg_density)][-1]))[0][0] + 1

    return avg_infall,avg_rotational,avg_temp,avg_density, infall[ids], \
           rotational[ids],subsnap['my_temp'][ids],subsnap['density'][ids], \
           R[ids]

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
                                         length = (100*au),
                                         orientation = 'x',
                                         center = (clump_centre)
                                         )
    y_comp = plonk.analysis.filters.tube(snap = snap_active,
                                         radius = (0.5*au),
                                         length = (100*au),
                                         orientation = 'y',
                                         center = (clump_centre)
                                         )
    z_comp = plonk.analysis.filters.cylinder(snap = snap_active,
                                         radius = (0.5*au),
                                         height = (100*au),
                                         center = (clump_centre)
                                         )
    axi_comp = plonk.analysis.filters.cylinder(snap = snap_active,
                                         radius = (50*au),
                                         height = (1*au),
                                         center = (clump_centre)
                                         )

    full_clump = plonk.analysis.filters.sphere(snap=snap_active,
                                               radius = (50*au),
                                               center = (clump_centre))


    return axi_comp,x_comp,y_comp,z_comp,clump_centre,clump_velocity, full_clump

def find_infall_peaks(array,interp_start=None,interp_end=None):
    # Do we need to interpolate - i.e. are there any NaN values in the data?
    if np.isnan(array).any() == True:
        array = interpolate_across_nans(array)[interp_start:interp_end]
        bspl = splrep(bins[interp_start:interp_end],array)
    elif len(array) == 99:
        bspl = splrep(bins[1:],array)

    bspl_y = splev(x_smooth, bspl)
    peak_search = np.where(x_smooth == x_smooth[np.abs(x_smooth-3e-3).argmin()])[0][0]
    peaks, _ = find_peaks(bspl_y[peak_search:],height=1,distance=500)
    peaks = peaks + peak_search
    minima_search = np.where(x_smooth == x_smooth[np.abs(x_smooth-0.1).argmin()])[0][0]
    y_inv = (bspl_y[minima_search:peaks[1]] * -1)
    minima, _ = find_peaks(y_inv,distance=300)
    minima = minima + minima_search
    #
    return bspl_y, peaks, minima


axi_comp,x_comp,y_comp,z_comp,clump_centre,clump_velocity,full_clump = prepare_snapshots('%s' % sys.argv[1])


print('Completed snapshot preparation')

avg_infall_axi,avg_rotational_axi,avg_temperature_axi,avg_density_axi, \
infall_axi,rotational_axi,temperature_axi,density_axi,\
R_axi = axisym_average(axi_comp,
                                                           clump_centre,
                                                            clump_velocity,bins,3)

print('Completed axisymmetric averages...')

avg_infall_sphere, avg_rotational_sphere, avg_density_sphere, avg_temperature_sphere, \
                                    R_sphere,interp_start_sphere,interp_end_sphere = spherical_average(full_clump,bins)
print('Completed spherical averages...')

avg_infall_x_pos,avg_rotational_x_pos,avg_temperature_x_pos,avg_density_x_pos, \
infall_x_pos,rotational_x_pos,temperature_x_pos,density_x_pos, \
x_pos,interp_start_x_pos,interp_end_x_pos,regular_avg_infall_x,\
regular_avg_rotational_x,regular_avg_temperature_x,regular_avg_density_x = calculate_SPH_mean_x(x_comp,
                                                                        clump_centre,
                                                                        clump_velocity,bins,3,False)

print('Completed positive x averages...')

avg_infall_y_pos,avg_rotational_y_pos,avg_temperature_y_pos,avg_density_y_pos, \
infall_y_pos,rotational_y_pos,temperature_y_pos,density_y_pos,\
y_pos,interp_start_y_pos,interp_end_y_pos = calculate_SPH_mean_y(y_comp,
                                                                        clump_centre,
                                                                        clump_velocity,bins,3,False)


print('Completed positive y averages...')
avg_infall_z_pos,avg_rotational_z_pos,avg_temperature_z_pos,avg_density_z_pos, \
infall_z_pos,rotational_z_pos,temperature_z_pos,density_z_pos,\
z_pos,interp_start_z_pos,interp_end_z_pos = calculate_SPH_mean_z(z_comp,
                                                                        clump_centre,
                                                                        clump_velocity,bins,3,False)
print('Completed positive z averages...')
avg_infall_x_neg,avg_rotational_x_neg,avg_temperature_x_neg,avg_density_x_neg, \
infall_x_neg,rotational_x_neg,temperature_x_neg,density_x_neg, x_neg,interp_start_x_neg,interp_end_x_neg,regular_avg_infall_x,\
regular_avg_rotational_x,regular_avg_temp_x,regular_avg_density_x = calculate_SPH_mean_x(x_comp,
                                                                        clump_centre,
                                                                        clump_velocity,bins,3,True)


print('Completed negative x averages...')
avg_infall_y_neg,avg_rotational_y_neg,avg_temperature_y_neg,avg_density_y_neg, \
infall_y_neg,rotational_y_neg,temperature_y_neg,density_y_neg, y_neg,interp_start_y_neg,interp_end_y_neg = calculate_SPH_mean_y(y_comp,
                                                                        clump_centre,
                                                                        clump_velocity,bins,3,True)
print('Completed negative y averages...')
avg_infall_z_neg,avg_rotational_z_neg,avg_temperature_z_neg,avg_density_z_neg, \
infall_z_neg,rotational_z_neg,temperature_z_neg,density_z_neg, z_neg,interp_start_z_neg,interp_end_z_neg = calculate_SPH_mean_z(z_comp,
                                                                        clump_centre,
                                                                        clump_velocity,bins,3,True)
print('Completed negative z averages...')




figure_indexes = [(0,0),(0,1),(1,0),(1,1)]
figure_ylimits = [(1E-13,1E-2),(10,12000),(0,12),(0,12)]
figure_ylabels = ['Density $(\\rm g\,cm^{-3})$','Temperature (K)','Rotational Velocity $(\\rm km\,s^{-1})$',
                  'Infall Velocity $(\\rm km\,s^{-1})$']

for index,label,limit in zip(figure_indexes,figure_ylabels,figure_ylimits):
    ax_main[index].set_ylabel(label,fontsize=10)
    ax_main[index].set_ylim(limit)
    ax_main[index].set_xlim(5e-4,50)
for i in range(2):
    for j in range(2):
        ax_main[i,j].set_xscale('log')
        ax_main[i,j].set_xlabel('r (AU)',fontsize=10)

print('Calculating peaks and minima...')
# x_pos_peaks,x_pos_mins = find_infall_peaks(avg_infall_x_pos,interp_start_x_pos,interp_end_x_pos)[1:]
# y_pos_peaks,y_pos_mins = find_infall_peaks(avg_infall_y_pos,interp_start_y_pos,interp_end_y_pos)[1:]
# z_pos_peaks,z_pos_mins = find_infall_peaks(avg_infall_z_pos,interp_start_z_pos,interp_end_z_pos)[1:]
#
# x_neg_peaks,x_neg_mins = find_infall_peaks(avg_infall_x_neg,interp_start_x_neg,interp_end_x_neg)[1:]
# y_neg_peaks,y_neg_mins = find_infall_peaks(avg_infall_y_neg,interp_start_y_neg,interp_end_y_neg)[1:]
# R_axi_peaks,R_axi_mins = find_infall_peaks(avg_infall_axi)[1:]
# R_sphere_peaks,R_sphere_mins = find_infall_peaks(avg_infall_sphere,interp_start_sphere,interp_end_sphere)[1:]

fragment_id = sys.argv[1].split('.')[1]
# with open('fragment_cores_%s.csv' % fragment_id, 'w') as f:
#     csv.writer(f).writerow(format_core_radii(x_pos_peaks,x_pos_mins))
#     csv.writer(f).writerow(format_core_radii(x_neg_peaks,x_neg_mins))
#     csv.writer(f).writerow(format_core_radii(y_pos_peaks,y_pos_mins))
#     csv.writer(f).writerow(format_core_radii(y_neg_peaks,y_neg_mins))
#     csv.writer(f).writerow(format_core_radii(z_pos_peaks,z_pos_mins))
#     csv.writer(f).writerow(format_core_radii(R_axi_peaks,R_axi_mins))
#     csv.writer(f).writerow(format_core_radii(R_sphere_peaks,R_sphere_mins))

print('Applying smoothing to z direction...')

print('Plotting...')
#------------------------------------------------------------------------------#
ax_main[0,0].plot(bins[interp_start_x_pos:interp_end_x_pos],interpolate_across_nans(avg_density_x_pos)[interp_start_x_pos:interp_end_x_pos],c='red',linestyle='-',linewidth=0.75,alpha=0.5)
ax_main[0,0].plot(bins[interp_start_y_pos:interp_end_y_pos],interpolate_across_nans(avg_density_y_pos)[interp_start_y_pos:interp_end_y_pos],c='green',linestyle='-',linewidth=0.75,alpha=0.5)
ax_main[0,0].plot(bins[interp_start_z_pos:interp_end_z_pos],interpolate_across_nans(avg_density_z_pos)[interp_start_z_pos:interp_end_z_pos],c='blue',linestyle='-',linewidth=0.75,alpha=0.5)

ax_main[0,0].plot(bins[interp_start_x_neg:interp_end_x_neg],interpolate_across_nans(avg_density_x_neg)[interp_start_x_neg:interp_end_x_neg],c='red',linestyle='--',linewidth=0.75,alpha=0.5)
ax_main[0,0].plot(bins[interp_start_y_neg:interp_end_y_neg],interpolate_across_nans(avg_density_y_neg)[interp_start_y_neg:interp_end_y_neg],c='green',linestyle='--',linewidth=0.75,alpha=0.5)
ax_main[0,0].plot(bins[interp_start_z_neg:interp_end_z_neg],interpolate_across_nans(avg_density_z_neg)[interp_start_z_neg:interp_end_z_neg],c='blue',linestyle='--',linewidth=0.75,alpha=0.5)

ax_main[0,0].plot(bins,avg_density_x_pos,c='red')
ax_main[0,0].plot(bins,avg_density_y_pos,c='green')
ax_main[0,0].plot(bins,avg_density_z_pos,c='blue')

ax_main[0,0].plot(bins,avg_density_x_neg,c='red',linestyle='--')
ax_main[0,0].plot(bins,avg_density_y_neg,c='green',linestyle='--')
ax_main[0,0].plot(bins,avg_density_z_neg,c='blue',linestyle='--')

ax_main[0,0].plot(R_sphere,avg_density_sphere,c='black',linestyle='dashdot')
ax_main[0,0].plot(bins,avg_density_axi,c='black',linestyle='dotted')

# ax_main[0,0].scatter(R_cont,full_clump['density'],s=0.1)
ax_main[0,0].set_yscale('log')
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
ax_main[0,1].plot(bins[interp_start_x_pos:interp_end_x_pos],interpolate_across_nans(avg_temperature_x_pos)[interp_start_x_pos:interp_end_x_pos],c='red',linestyle='-',linewidth=0.75,alpha=0.5)
ax_main[0,1].plot(bins[interp_start_y_pos:interp_end_y_pos],interpolate_across_nans(avg_temperature_y_pos)[interp_start_y_pos:interp_end_y_pos],c='green',linestyle='-',linewidth=0.75,alpha=0.5)
ax_main[0,1].plot(bins[interp_start_z_pos:interp_end_z_pos],interpolate_across_nans(avg_temperature_z_pos)[interp_start_z_pos:interp_end_z_pos],c='blue',linestyle='-',linewidth=0.75,alpha=0.5)

ax_main[0,1].plot(bins[interp_start_x_neg:interp_end_x_neg],interpolate_across_nans(avg_temperature_x_neg)[interp_start_x_neg:interp_end_x_neg],c='red',linestyle='--',linewidth=0.75,alpha=0.5)
ax_main[0,1].plot(bins[interp_start_y_neg:interp_end_y_neg],interpolate_across_nans(avg_temperature_y_neg)[interp_start_y_neg:interp_end_y_neg],c='green',linestyle='--',linewidth=0.75,alpha=0.5)
ax_main[0,1].plot(bins[interp_start_z_neg:interp_end_z_neg],interpolate_across_nans(avg_temperature_z_neg)[interp_start_z_neg:interp_end_z_neg],c='blue',linestyle='--',linewidth=0.75,alpha=0.5)

ax_main[0,1].plot(bins,avg_temperature_x_pos,c='red')
ax_main[0,1].plot(bins,avg_temperature_y_pos,c='green')
ax_main[0,1].plot(bins,avg_temperature_z_pos,c='blue')

ax_main[0,1].plot(bins,avg_temperature_x_neg,c='red',linestyle='--')
ax_main[0,1].plot(bins,avg_temperature_y_neg,c='green',linestyle='--')
ax_main[0,1].plot(bins,avg_temperature_z_neg,c='blue',linestyle='--')

ax_main[0,1].plot(R_sphere,avg_temperature_sphere,c='black',linestyle='dashdot')
ax_main[0,1].plot(bins,avg_temperature_axi,c='black',linestyle='dotted')
ax_main[0,1].set_yscale('log')
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
ax_main[1,0].plot(bins[interp_start_x_pos:interp_end_x_pos],interpolate_across_nans(avg_rotational_x_pos)[interp_start_x_pos:interp_end_x_pos],c='red',linestyle='-',linewidth=0.75,alpha=0.5)
ax_main[1,0].plot(bins[interp_start_y_pos:interp_end_y_pos],interpolate_across_nans(avg_rotational_y_pos)[interp_start_y_pos:interp_end_y_pos],c='green',linestyle='-',linewidth=0.75,alpha=0.5)
# ax_main[1,0].plot(bins[interp_start_z_pos:interp_end_z_pos],interpolate_across_nans(avg_rotational_z_pos)[interp_start_z_pos:interp_end_z_pos],c='blue',linestyle='-',linewidth=0.75,alpha=0.5)

ax_main[1,0].plot(bins[interp_start_x_neg:interp_end_x_neg],interpolate_across_nans(avg_rotational_x_neg)[interp_start_x_neg:interp_end_x_neg],c='red',linestyle='--',linewidth=0.75,alpha=0.5)
ax_main[1,0].plot(bins[interp_start_y_neg:interp_end_y_neg],interpolate_across_nans(avg_rotational_y_neg)[interp_start_y_neg:interp_end_y_neg],c='green',linestyle='--',linewidth=0.75,alpha=0.5)
# ax_main[1,0].plot(bins[interp_start_z_neg:interp_end_z_neg],interpolate_across_nans(avg_rotational_z_neg)[interp_start_z_neg:interp_end_z_neg],c='blue',linestyle='--',linewidth=0.75,alpha=0.5)

ax_main[1,0].plot(bins,avg_rotational_x_pos,c='red')
ax_main[1,0].plot(bins,avg_rotational_y_pos,c='green')
# ax_main[1,0].plot(bins,avg_rotational_z_pos,c='blue')

ax_main[1,0].plot(bins,avg_rotational_x_neg,c='red',linestyle='--')
ax_main[1,0].plot(bins,avg_rotational_y_neg,c='green',linestyle='--')
# ax_main[1,0].plot(bins,avg_rotational_z_neg,c='blue',linestyle='--')

ax_main[1,0].plot(R_sphere,avg_rotational_sphere,c='black',linestyle='dashdot')
ax_main[1,0].plot(bins,avg_rotational_axi,c='black',linestyle='dotted')
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
ax_main[1,1].plot(bins[interp_start_x_pos:interp_end_x_pos],interpolate_across_nans(avg_infall_x_pos)[interp_start_x_pos:interp_end_x_pos],c='red',linestyle='-',linewidth=0.75,alpha=0.5)
ax_main[1,1].plot(bins[interp_start_y_pos:interp_end_y_pos],interpolate_across_nans(avg_infall_y_pos)[interp_start_y_pos:interp_end_y_pos],c='green',linestyle='-',linewidth=0.75,alpha=0.5)
# ax_main[1,1].plot(bins[interp_start_z_pos:interp_end_z_pos],interpolate_across_nans(avg_infall_z_pos)[interp_start_z_pos:interp_end_z_pos],c='blue',linestyle='-',linewidth=0.75,alpha=0.5)

ax_main[1,1].plot(bins[interp_start_x_neg:interp_end_x_neg],interpolate_across_nans(avg_infall_x_neg)[interp_start_x_neg:interp_end_x_neg],c='red',linestyle='--',linewidth=0.75,alpha=0.5)
ax_main[1,1].plot(bins[interp_start_y_neg:interp_end_y_neg],interpolate_across_nans(avg_infall_y_neg)[interp_start_y_neg:interp_end_y_neg],c='green',linestyle='--',linewidth=0.75,alpha=0.5)
# ax_main[1,1].plot(bins[interp_start_z_neg:interp_end_z_neg],interpolate_across_nans(avg_infall_z_neg)[interp_start_z_neg:interp_end_z_neg],c='blue',linestyle='--',linewidth=0.75,alpha=0.5)

ax_main[1,1].plot(bins,avg_infall_x_pos,c='red',label='+x direction')
ax_main[1,1].plot(bins,avg_infall_y_pos,c='green',label='+y direction')
ax_main[1,1].plot(bins,apply_filter(avg_infall_z_pos),c='blue',label='+z direction')

ax_main[1,1].plot(bins,avg_infall_x_neg,c='red',linestyle='--',label='-x direction')
ax_main[1,1].plot(bins,avg_infall_y_neg,c='green',linestyle='--',label='-y direction')
ax_main[1,1].plot(bins,apply_filter(avg_infall_z_neg),c='blue',linestyle='--',label='-z direction')


ax_main[1,1].plot(R_sphere,avg_infall_sphere,c='black',linestyle='dashdot',label='Spherical average')
ax_main[1,1].plot(bins,avg_infall_axi,c='black',linestyle='dotted',label='Axisymmetric average')

ax_main[1,1].legend(loc='upper center', bbox_to_anchor=(-0.1, -0.2),
            fancybox=True, shadow=True, ncol=3)
#------------------------------------------------------------------------------#
print('Saving figure...')
fig_main.subplots_adjust(bottom=0.175)

fig_main.savefig('3D_comparison_%s.png' % fragment_id,dpi=200)
