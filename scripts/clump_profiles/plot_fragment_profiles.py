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
from numpy import inf
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
from itertools import zip_longest
import os
start_time = time.time()
# Define figure
fig_main, ax_main = plt.subplots(ncols=2,nrows=3,figsize=(7,8))
fig_test, ax_test = plt.subplots(figsize=(9,9))
fig_mom, axs_mom = plt.subplots(figsize=(7,8))

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


run_set_dict = {'first':1,'second':2,'third':3,'fourth':4,'fifth':5,'sixth':6,\
                'seventh':7,'eighth':8,'ninth':9,'tenth':10}


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


    return avg_infall,avg_rotational,avg_density,avg_temperature, R, mass_in_bin,R_full

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

    bspl_y = splev(x_smooth, bspl)
    peak_search = np.where(x_smooth == x_smooth[np.abs(x_smooth-3e-3).argmin()])[0][0]
    peaks, _ = find_peaks(bspl_y[peak_search:],height=1,distance=500)
    if len(peaks)>1:
        peaks = peaks + peak_search
        minima_search = np.where(x_smooth == x_smooth[np.abs(x_smooth-0.1).argmin()])[0][0]
        y_inv = (bspl_y[minima_search:peaks[1]] * -1)
        minima, _ = find_peaks(y_inv,distance=300)
        minima = minima + minima_search
    else:
        peaks = [None,None]
        minima = [None]
    return bspl_y, peaks, minima

def calculate_momentum(subsnap,clump_centre,clump_velocity,R_full):
    count = calculate_number_in_bin(R_full,subsnap['m'])
    specific_angular_momentum = plonk.analysis.particles.specific_angular_momentum(subsnap,clump_centre,clump_velocity,ignore_accreted=True).to('cm**2/s')
    total_L = np.sqrt(specific_angular_momentum[:,0]**2 + specific_angular_momentum[:,1]**2 +specific_angular_momentum[:,2]**2)
    spec_mom_binned_2 = calculate_sum(R_full,total_L)
    tmp1 = spec_mom_binned_2[0][spec_mom_binned_2[0] != 0]
    tmp2 = count[0][count[0] != 0]
    spec_mom_sum_2 = (np.cumsum(tmp1/tmp2))
    pad_width = len(spec_mom_binned_2[1][1:]) - len(spec_mom_sum_2)

    L = np.pad(spec_mom_sum_2, (pad_width,0), 'constant')
    return spec_mom_binned_2[1][1:],L

with open("fragment_results.csv", "wt") as f:
    headers = ['Rsc (axi)', 'Rfci (axi)', 'Rfco (axi)', 'Msc (axi)', \
               'Mfci (axi)', 'Mfco (axi)', 'Lsc (axi)', 'Lfci (axi)', \
               'Lfco (axi)','alphasc (axi)', 'alphafci (axi)', \
               'alphafco (axi)','betasc (axi)', 'betafci (axi)', \
               'betafco (axi)',\

               'Rsc (z)', 'Rfci (z)', 'Rfco (z)', 'Msc (z)', \
               'Mfci (z)', 'Mfco (z)', 'Lsc (z)', 'Lfci (z)', \
               'Lfco (z)','alphasc (z)', 'alphafci (z)', \
               'alphafco (z)','betasc (z)', 'betafci (z)', \
               'betafco (z)','R','Nsc (axi)','Nfci (axi)','Nfco (axi)',\
               'Nsc (z)','Nfci (z)','Nfco (z)','ID']
    for heading in headers:
        f.write("# %s \n" % heading)


f.close()

fragments_to_plot = sys.argv[1:]
for fragment in fragments_to_plot:
    run_set = os.path.abspath(fragment).split('/')[5].split('_')[0] # Which initial particle distribution?
    param_set = os.path.abspath(fragment).split('/')[6].split('_')[1].lstrip('0') # Which parameter set (run_001,run_002 etc)
    fragment_number = fragment.split('.')[1].lstrip('0') # Fragment number
    ID_code = '%s_%s_%s' % (run_set_dict[run_set],param_set,fragment_number)

    index = fragments_to_plot.index(fragment)
    line_colour = mpl_colour_defaults[index]
    print(colored('Preparing snapshots...','green'),end="", flush=True)

    axi_comp,z_comp,clump_centre,clump_velocity,full_clump = prepare_snapshots('%s' % fragment)


    print(colored('Done','green'))




    avg_infall_axi,avg_rotational_axi,avg_temperature_axi,avg_density_axi, \
    infall_axi,rotational_axi,temperature_axi,density_axi,\
    R_axi = axisym_average(axi_comp,
                           clump_centre,
                           clump_velocity,bins,3)


    print(colored('Calculating spherical averages...','green'),end="", flush=True)
    avg_infall_sphere, avg_rotational_sphere, avg_density_sphere, avg_temperature_sphere, \
    R_sphere, mass,R_full= spherical_average(full_clump,bins)
    print(colored('Done','green'))



    print(colored('Calculating z averages...','green'),end="", flush=True)
    avg_infall_z_pos,avg_rotational_z_pos,avg_temperature_z_pos,avg_density_z_pos, \
    infall_z_pos,rotational_z_pos,temperature_z_pos,density_z_pos,\
    z_pos,interp_start_z_pos,interp_end_z_pos = calculate_SPH_mean_z(z_comp,
                                                                     clump_centre,
                                                                     clump_velocity,bins,3,False)


    print(colored('Done','green'))


    print(colored('Calculating specific angular momentum...','green'),end="", flush=True)
    momentum_bins, momentum = calculate_momentum(full_clump,clump_centre,clump_velocity,R_full)
    print(colored('Done','green'))





    figure_indexes = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]
    figure_ylimits = [(1E-13,1E-2),(10,12000),(0,12),(0,12),(0,80),(0,10)]
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

    print(colored('Calculating energy ratios...','green'),end="", flush=True)

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
        alpha[alpha == inf] = np.nan
        alpha = interpolate_across_nans(alpha)
    with np.errstate(invalid='ignore'):
        beta = cumsum_erot / cumsum_egrav
        beta[beta == inf] = np.nan
        beta = interpolate_across_nans(beta)

    print(colored('Done','green'))
    print(colored('Fitting B-spline to arrays to increase sampling resolution...','green'),end="", flush=True)

    bspl_mass = splrep(bins[1:],mass)
    bsply_mass = splev(x_smooth, bspl_mass)
    bspl_mom = splrep(bins[1:],momentum)
    bsply_mom = splev(x_smooth, bspl_mom)
    bspl_alpha = splrep(bins[1:],alpha)
    bsply_alpha = splev(x_smooth, bspl_alpha)
    bspl_beta = splrep(bins[1:],beta)
    bsply_beta = splev(x_smooth, bspl_beta)


    print(colored('Done','green'))

    print(colored('Plotting...','green'),end="", flush=True)
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
    ax_main[2,1].plot(gravitational_energy_binned[1][1:],alpha,c=line_colour,ls='dashed')
    ax_main[2,1].plot(gravitational_energy_binned[1][1:],beta,c=line_colour,ls='dashed')

    ax_main[2,1].axhline(y=1,c='black',linestyle='-',linewidth=1.5,alpha=0.4)
    ax_main[2,1].set_xscale('log')
    ax_main[2,1].set_yscale('log')
    ax_main[2,1].set_ylim(1e-3,4)

    #------------------------------------------------------------------------------#
    axs_mom.plot(momentum_bins,momentum,c=line_colour)

    axs_mom.set_xscale('log')
    axs_mom.set_xlabel('R (AU)')
    axs_mom.set_ylabel('J $(\\rm cm^{2}\,s^{-1})$')
    axs_mom.set_xlim(5E-4,50)
    axs_mom.set_yscale('log')
    print(colored('Done','green'))

    position_fragment = np.sqrt(clump_centre[0]**2+clump_centre[1]**2+clump_centre[2]**2)


    print(colored('Calculating peaks and minima...','green'),end="", flush=True)
    try:
        infall_z_spline,z_pos_peaks,z_pos_mins = find_infall_peaks(avg_infall_z_pos,interp_start_z_pos,interp_end_z_pos)
    except:
        print(colored('No peaks found in z profile','green'))
        z_pos_peaks[0] = None
        z_pos_peaks[1] = None
        z_pos_mins[0] = None

    try:
        infall_axi_spline,R_axi_peaks,R_axi_mins = find_infall_peaks(avg_infall_axi)
    except:
        print(colored('No peaks found in axisymmetric profile','green'))
        R_axi_peaks[0] = None
        R_axi_peaks[1] = None
        R_axi_mins[0] = None

    print(colored('Done','green'))
    ax_main[0,0].annotate("a)", xy=(0.9, 0.9), xycoords="axes fraction")
    ax_main[0,1].annotate("b)", xy=(0.9, 0.9), xycoords="axes fraction")
    ax_main[1,0].annotate("c)", xy=(0.9, 0.9), xycoords="axes fraction")
    ax_main[1,1].annotate("d)", xy=(0.9, 0.9), xycoords="axes fraction")
    ax_main[2,0].annotate("e)", xy=(0.9, 0.9), xycoords="axes fraction")
    ax_main[2,1].annotate("f)", xy=(0.9, 0.9), xycoords="axes fraction")
    fig_main.align_ylabels()
    fig_main.subplots_adjust(wspace=0.45,hspace=0.3)
    fig_main.savefig('clump_profiles.png',dpi=200)
    fig_mom.savefig('angular_momentum.png',dpi=200)

    if R_axi_peaks[0] is not None and R_axi_peaks[1] is not None and z_pos_peaks[0] is not None and z_pos_peaks[1] is not None:


        Nsc_axi = bsply_mass[R_axi_peaks[0]] / full_clump['m'][0].to('jupiter_mass')
        Nfci_axi = bsply_mass[R_axi_mins[0]] / full_clump['m'][0].to('jupiter_mass')
        Nfco_axi = bsply_mass[R_axi_peaks[1]] / full_clump['m'][0].to('jupiter_mass')
        Nsc_z = bsply_mass[z_pos_peaks[0]] / full_clump['m'][0].to('jupiter_mass')
        Nfci_z = bsply_mass[z_pos_mins[0]] / full_clump['m'][0].to('jupiter_mass')
        Nfco_z = bsply_mass[z_pos_peaks[1]] / full_clump['m'][0].to('jupiter_mass')

        with open("fragment_results.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([x_smooth[R_axi_peaks[0]],x_smooth[R_axi_mins[0]],\
                             x_smooth[R_axi_peaks[1]],\

                             bsply_mass[R_axi_peaks[0]], \
                             bsply_mass[R_axi_mins[0]],bsply_mass[R_axi_peaks[1]],\

                             bsply_mom[R_axi_peaks[0]], \
                             bsply_mom[R_axi_mins[0]],bsply_mom[R_axi_peaks[1]],\

                             bsply_alpha[R_axi_peaks[0]], \
                             bsply_alpha[R_axi_mins[0]],bsply_alpha[R_axi_peaks[1]],\

                             bsply_beta[R_axi_peaks[0]], \
                             bsply_beta[R_axi_mins[0]],bsply_beta[R_axi_peaks[1]],\

                             x_smooth[z_pos_peaks[0]],x_smooth[z_pos_mins[0]],\
                             x_smooth[z_pos_peaks[1]],\

                             bsply_mass[z_pos_peaks[0]], \
                             bsply_mass[z_pos_mins[0]],bsply_mass[z_pos_peaks[1]],\

                             bsply_mom[z_pos_peaks[0]], \
                             bsply_mom[z_pos_mins[0]],bsply_mom[z_pos_peaks[1]],\

                             bsply_alpha[z_pos_peaks[0]], \
                             bsply_alpha[z_pos_mins[0]],bsply_alpha[z_pos_peaks[1]],\

                             bsply_beta[z_pos_peaks[0]], \
                             bsply_beta[z_pos_mins[0]],bsply_beta[z_pos_peaks[1]],\

                             position_fragment, Nsc_axi, Nfci_axi, Nfco_axi, Nsc_z, \
                             Nfci_z, Nfco_z,ID_code])
            f.close()
        print(colored('Done','green'))
    else:
        print(colored('No entry made for fragment %s as no peaks were found in the profiles % fragment','green'))
