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
from termcolor import colored
from threading import Thread
from tqdm.auto import tqdm
import time
from matplotlib.lines import Line2D
from digitize import calculate_gravitational_energy

start_time = time.time()
# Define figure
fig_main, ax_main = plt.subplots(ncols=2,nrows=3,figsize=(7,8))
fig_test, ax_test = plt.subplots(figsize=(9,9))
mpl_colour_defaults=plt.rcParams['axes.prop_cycle'].by_key()['color'] # MLP default colours

# Define constants to convert to physical units
au = plonk.units('au')
kms = plonk.units('km/s')
bins = np.logspace(np.log10(5e-4),np.log10(50),100) # change the number of bins ?

bin_widths= np.diff(bins,prepend=0.0)
x_smooth = np.logspace(np.log10(3e-4),np.log10(50),2000)
if hasattr(pint, 'UnitStrippedWarning'):
    warnings.simplefilter('ignore', category=pint.UnitStrippedWarning)
np.seterr(divide='ignore', invalid='ignore')

def calculate_number_in_bin(binned_quantity,mean_quantity):
    return stats.binned_statistic(binned_quantity, mean_quantity, 'count', bins=bins)

def calculate_sum(binned_quantity,summed_quantity):
    return stats.binned_statistic(binned_quantity, summed_quantity, 'sum', bins=bins)

def apply_filter(array):
    smoothed_array = savgol_filter(array,5,3)
    return smoothed_array

def format_core_radii(peaks,minima):
    Rsc,Rfci,Rfco,delta_shock = x_smooth[peaks][0],\
                                x_smooth[minima], \
                                x_smooth[peaks][1],\
                                x_smooth[peaks][1] - x_smooth[minima]


    return Rsc,Rfci,Rfco,delta_shock

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

def calculate_thermal_energy(subSnap):
    U = 3/2 * 1.38E-16 * subSnap['my_temp'] * ((subSnap['m'][0].to('g'))/(1.67E-24))
    U = U.magnitude
    U *= ((subSnap['my_temp'].magnitude>2000)*(1/1.2)+(subSnap['my_temp'].magnitude<2000)*(1/2.381))
    return U

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
            if np.abs(z[part]) < bins[bin+1] and np.abs(z[part]) > bins[bin] \
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

    interp_start = np.where(avg_density == (avg_density[np.isfinite(avg_density)][0]))[0][0]
    interp_end = np.where(avg_density == (avg_density[np.isfinite(avg_density)][-1]))[0][0] + 1

    return avg_infall,avg_rotational,avg_temp,avg_density, infall[ids], \
           rotational[ids],subsnap['my_temp'][ids],subsnap['density'][ids], \
           z[ids],interp_start,interp_end

def spherical_average(subsnap,bins):
    x = subsnap['x'].magnitude - clump_centre[0].magnitude
    y = subsnap['y'].magnitude - clump_centre[1].magnitude
    z = subsnap['z'].magnitude - clump_centre[2].magnitude

    R_full = np.sqrt(x**2 + y**2 + z**2)

    rotational = plonk.analysis.particles.rotational_velocity(subsnap,
                                                              clump_velocity,
                                                              ignore_accreted=True)
    infall = plonk.analysis.particles.velocity_radial_spherical_altered(subsnap,
                                                                        clump_centre,
                                                                        clump_velocity,
                                                                        ignore_accreted=True)

    mass_in_bin = np.cumsum(calculate_number_in_bin(R_full,subsnap['m'])[0]) * subsnap['m'][0].to('jupiter_mass')
    avg_infall = calculate_mean(R_full,infall)[0]
    avg_rotational = calculate_mean(R_full,rotational)[0]
    avg_density = calculate_mean(R_full,subsnap['density'])[0]
    avg_temperature = calculate_mean(R_full,subsnap['my_temp'])[0]
    R = calculate_mean(R_full,infall)[1][1:]

    interp_start = np.where(avg_density == (avg_density[np.isfinite(avg_density)][0]))[0][0] - 1
    interp_end = np.where(avg_density == (avg_density[np.isfinite(avg_density)][-1]))[0][0] + 1


    return avg_infall,avg_rotational,avg_density,avg_temperature, R,interp_start,interp_end, mass_in_bin,R_full

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


    return axi_comp, z_comp, clump_centre, clump_velocity, full_clump

def find_infall_peaks(array,interp_start=None,interp_end=None):
    # Do we need to interpolate - i.e. are there any NaN values in the data?
    if np.isnan(array).any() == True:
        array = interpolate_across_nans(array)[interp_start:interp_end]
        bspl = splrep(bins[interp_start:interp_end],array)
    # elif len(array) == 99:
    #     bspl = splrep(bins[1:],array)
    #
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



fragments_to_plot = sys.argv[1:]
for fragment in fragments_to_plot:
    index = fragments_to_plot.index(fragment)
    line_colour = mpl_colour_defaults[index]
    axi_comp,z_comp,clump_centre,clump_velocity,full_clump = prepare_snapshots('%s' % fragment)


    print(colored('Completed snapshot preparation','green'))

    avg_infall_axi,avg_rotational_axi,avg_temperature_axi,avg_density_axi, \
    infall_axi,rotational_axi,temperature_axi,density_axi,\
    R_axi = axisym_average(axi_comp,
                                                               clump_centre,
                                                                clump_velocity,bins,3)


    print(colored('Completed axisymmetric averages...','green'))

    avg_infall_sphere, avg_rotational_sphere, avg_density_sphere, avg_temperature_sphere, \
                                        R_sphere,interp_start_sphere,interp_end_sphere, mass,R_full= spherical_average(full_clump,bins)
    print(colored('Completed spherical averages...','green'))

    print(colored('Fitting B-spline to mass to increase sampling resolution...','green'))

    bspl_mass = splrep(bins[1:],mass)
    bsply_mass = splev(x_smooth, bspl_mass)




    avg_infall_z_pos,avg_rotational_z_pos,avg_temperature_z_pos,avg_density_z_pos, \
    infall_z_pos,rotational_z_pos,temperature_z_pos,density_z_pos,\
    z_pos,interp_start_z_pos,interp_end_z_pos = calculate_SPH_mean_z(z_comp,
                                                                     clump_centre,
                                                                     clump_velocity,bins,3,False)
    print(colored('Completed z averages...','green'))


    figure_indexes = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]
    figure_ylimits = [(1E-13,1E-2),(10,12000),(0,8),(0,8),(0,80),(0,10)]
    figure_ylabels = ['Density $(\\rm g\,cm^{-3})$','Temperature (K)','Rotational Velocity $(\\rm km\,s^{-1})$',
                      'Infall Velocity $(\\rm km\,s^{-1})$',r'Mass (M$_{J}$)','Energy ratio']

    for index,label,limit in zip(figure_indexes,figure_ylabels,figure_ylimits):
        ax_main[index].set_ylabel(label,fontsize=10)
        ax_main[index].set_ylim(limit)
        ax_main[index].set_xlim(1e-3,50)
    for i in range(3):
        for j in range(2):
            ax_main[i,j].set_xscale('log')
            ax_main[i,j].set_xlabel('r (AU)',fontsize=10)

    print(colored('Calculating peaks and minima...','green'))
    # z_tmp,z_pos_peaks,z_pos_mins = find_infall_peaks(avg_infall_z_pos,interp_start_z_pos,interp_end_z_pos)
    # tmp_axi,R_axi_peaks,R_axi_mins = find_infall_peaks(avg_infall_axi)

    print(colored('Calculating energy ratios...','green'))

    rotational_energy = 0.5 * axi_comp['m'][0].to('g') * rotational_axi.to('cm/s') **2
    rotational_energy_binned = calculate_sum(R_axi,rotational_energy)
    cumsum_erot = np.cumsum(rotational_energy_binned[0])

    thermal_energy = calculate_thermal_energy(full_clump)
    thermal_energy_binned = calculate_sum(R_full,thermal_energy)
    cumsum_etherm = np.cumsum(thermal_energy_binned[0])

    grav_rad = R_full
    gravitational_energy = calculate_gravitational_energy(grav_rad,bins,full_clump['m'][0].to('g'))
    gravitational_energy_binned = calculate_sum(R_full,gravitational_energy)
    cumsum_egrav = np.cumsum(gravitational_energy_binned[0])

    with np.errstate(invalid='ignore'):
        alpha = cumsum_etherm / cumsum_egrav

    with np.errstate(invalid='ignore'):
        beta = cumsum_erot / cumsum_egrav


    print(colored('Plotting...','green'))
    r_tmp = np.sqrt((axi_comp['x']-clump_centre[0])**2 + (axi_comp['y']-clump_centre[1])**2)
    #------------------------------------------------------------------------------#
    ax_main[0,0].plot(bins[interp_start_z_pos:interp_end_z_pos],interpolate_across_nans(avg_density_z_pos)[interp_start_z_pos:interp_end_z_pos],c=line_colour,linestyle='dotted',linewidth=0.75,alpha=0.5)
    ax_main[0,0].plot(bins,avg_density_z_pos,c=line_colour,ls='dotted')
    ax_main[0,0].plot(bins,avg_density_axi,c=line_colour,linestyle='-')
    ax_main[0,0].set_yscale('log')
    #------------------------------------------------------------------------------#
    #------------------------------------------------------------------------------#
    ax_main[0,1].plot(bins[interp_start_z_pos:interp_end_z_pos],interpolate_across_nans(avg_temperature_z_pos)[interp_start_z_pos:interp_end_z_pos],c=line_colour,linestyle='dotted',linewidth=0.75,alpha=0.5)
    ax_main[0,1].plot(bins,avg_temperature_z_pos,c=line_colour,ls='dotted')
    ax_main[0,1].plot(bins,avg_temperature_axi,c=line_colour,linestyle='-')
    ax_main[0,1].set_yscale('log')
    #------------------------------------------------------------------------------#
    #------------------------------------------------------------------------------#
    ax_main[1,0].plot(bins,avg_rotational_axi,c=line_colour,linestyle='-')
    #------------------------------------------------------------------------------#
    #------------------------------------------------------------------------------#
    ax_main[1,1].plot(bins[interp_start_z_pos:interp_end_z_pos],interpolate_across_nans(avg_infall_z_pos)[interp_start_z_pos:interp_end_z_pos],c=line_colour,linestyle='dotted',linewidth=0.75,alpha=0.5)
    ax_main[1,1].plot(bins,avg_infall_z_pos,c=line_colour,label='z direction',ls='dotted')
    ax_main[1,1].plot(bins,avg_infall_axi,c=line_colour,linestyle='-', label='Axisymmetric average')
    #------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------#
    ax_main[2,0].plot(bins[1:],mass,c=line_colour,ls='dashed')
    #------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------#
    ax_main[2,1].plot(gravitational_energy_binned[1][1:],alpha,linewidth=1,c=line_colour,ls='dashed')
    ax_main[2,1].plot(gravitational_energy_binned[1][1:],beta,linewidth=1,c=line_colour,ls='dashed')

    ax_main[2,1].axhline(y=1,c='black',linestyle='-',linewidth=1.5,alpha=0.4)
    ax_main[2,1].set_xscale('log')
    ax_main[2,1].set_yscale('log')
    ax_main[2,1].set_ylim(1e-3,4)

    legend_elements = [Line2D([0], [0], color='black', lw=1.5,ls='dotted',
                              label='z Direction'),
                       Line2D([0], [0], color='black', lw=1.5,ls='-',
                              label='Axisymmetric average'),
                      Line2D([0], [0], color='black', lw=1.5,ls='dashed',
                              label='Spherical average')]

    fig_main.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(.5,.05), fontsize='small',ncol=3)

    #------------------------------------------------------------------------------#
    # print(colored('Saving figure...','green'))
    # print(colored('Script execution time = %s minutes' % str(total_time/60),'green'))
    # print(colored('-' * 120,'green'))
    # print(colored('Rsc (z) = %s | Rfci (z) = %s | Rfio (z) = %s' %(x_smooth[z_pos_peaks[0]],x_smooth[z_pos_mins[0]],x_smooth[z_pos_peaks[1]]),'green'))
    # print(colored('Msc (z) = %s | Mfci (z) = %s | Mfio (z) = %s' %(bsply_mass[z_pos_peaks[0]],bsply_mass[z_pos_mins[0]],bsply_mass[z_pos_peaks[1]]),'green'))
    #
    # print(colored('Rsc (axi) = %s | Rfci (axi) = %s | Rfio (axi) = %s' %(x_smooth[R_axi_peaks[0]],x_smooth[R_axi_mins[0]],x_smooth[R_axi_peaks[1]]),'green'))
    # print(colored('Msc (axi) = %s | Mfci (axi) = %s | Mfio (axi) = %s' %(bsply_mass[R_axi_peaks[0]],bsply_mass[R_axi_mins[0]],bsply_mass[R_axi_peaks[1]]),'green'))
    #
    # print(colored('-' * 120,'green'))
fig_main.align_ylabels()
fig_main.subplots_adjust(wspace=0.45,hspace=0.3)
fig_main.subplots_adjust(bottom=0.175)
fig_main.savefig('clump_profiles.png',dpi=200)
total_time = time.time() - start_time
