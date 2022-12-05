import plonk
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from scipy import stats
from scipy.interpolate import splev, splrep
import pint
from tqdm.auto import tqdm
import warnings
from scipy.signal import savgol_filter
from digitize import calculate_gravitational_energy
import time
import pandas as pd

# Author: Adam Fenton
cwd = os.getcwd()
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# plt.rcParams['text.usetex'] = True
# fig,axs = plt.subplots(figsize=(7,4.5))
# plt.rcParams["font.family"] = "serif"
# Load units for use later, useful for derived quantities.
au = plonk.units('au')
kms = plonk.units('km/s')
density_to_load = None #
mean_bins_radial = np.logspace(np.log10(0.0005),np.log10(50),100)
mpl_colour_defaults=plt.rcParams['axes.prop_cycle'].by_key()['color'] # MLP default colours

# Initalise the figure and output file which is written to later on
fig_radial, f_radial_axs = plt.subplots(nrows=3,ncols=2,figsize=(7,8))
fig_ang_mom, axs_ang_mom = plt.subplots(figsize=(7,8))
fig_test, axs_test = plt.subplots(figsize=(7,8))

clump_results = open('clump-results.dat', 'w')
# Ignore pesky warnings when stripping unit off of pint quantity when downcasting to array
if hasattr(pint, 'UnitStrippedWarning'):
    warnings.simplefilter('ignore', category=pint.UnitStrippedWarning)
np.seterr(divide='ignore', invalid='ignore')
# @profile
def calculate_sum(binned_quantity,summed_quantity,bins):
    return stats.binned_statistic(binned_quantity, summed_quantity, 'sum', bins=bins)
# @profile
def calculate_mean(binned_quantity,mean_quantity):
    bins = np.logspace(np.log10(0.0005),np.log10(50),100)
    return stats.binned_statistic(binned_quantity, mean_quantity, 'mean', bins=bins)
# @profile
def calculate_number_in_bin(binned_quantity,mean_quantity,width):
    bins=np.logspace(np.log10(0.0005),np.log10(width),100)
    return stats.binned_statistic(binned_quantity, mean_quantity, 'count', bins=bins)
# @profile
def calculate_thermal_energy(subSnap):
    U = 3/2 * 1.38E-16 * subSnap['my_temp'] * ((subSnap['m'][0].to('g'))/(1.67E-24))
    U = U.magnitude
    U *= ((subSnap['my_temp'].magnitude>2000)*(1/1.2)+(subSnap['my_temp'].magnitude<2000)*(1/2.381))
    return U
# @profile
def prepare_snapshots(snapshot,density_to_load,clump_number,clump_data_file,density_flag):
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
    snapshot_file_path = os.path.dirname(os.path.abspath(snapshot))

    clump_data_file_name = snapshot_file_path.split("/")[-2].split("p")[-1]+".dat"

    clump_data_file = snapshot_file_path + "/"+clump_data_file_name


    ### This is where the issue is, it works for the E-3 because there is no
    ### particle in the snapshot with a higher density but when plotting lower
    ### densities, the recentering is off

    clump_location_data = pd.read_csv(clump_data_file,names=["rho", "x", "y", "z",
                                                             "vx", "vy", "vz"])

    if flag == 1:
        index = clump_location_data['rho'].sub(density_to_load).abs().idxmin()
        row_of_interest = clump_location_data.iloc[[index]]
        clump_centre = row_of_interest.values[0][1:4]
        velocity_conversion = 2.978E6/1e5
        clump_velocity = row_of_interest.values[0][4:] * velocity_conversion * kms
        x,y,z = clump_centre[0],clump_centre[1],clump_centre[2]

    if flag == 0:
        snapshot_density = 10**(-(int(snapshot.split(".")[3].split("0")[1])))
        index = clump_location_data['rho'].sub(snapshot_density).abs().idxmin()
        row_of_interest = clump_location_data.iloc[[index]]
        clump_centre = row_of_interest.values[0][1:4]
        velocity_conversion = 2.978E6/1e5
        clump_velocity = row_of_interest.values[0][4:] * velocity_conversion * kms
        x,y,z = clump_centre[0],clump_centre[1],clump_centre[2]

    # clump_velocity = snap['velocity'][id]
    accreted_mask = snap['smoothing_length'] > 0
    snap_active = snap[accreted_mask]
    PID = int(file.split('.')[2]) - 1
    PID_index = np.where(snap_active['id'] == PID)
    clump_centre_new = snap_active['position'][PID_index][0].magnitude
    clump_velocity_new = snap_active['velocity'][PID_index][0]
    subSnap=plonk.analysis.filters.sphere(snap=snap_active,radius = (50*au),center=clump_centre_new *au)
    subSnap_temp=plonk.analysis.filters.sphere(snap=snap_active,radius = (15*au),center=clump_centre_new *au)


    subSnap_rotvel=plonk.analysis.filters.cylinder(snap=snap_active,radius = (50*au),height=(0.75*au),center=clump_centre_new * au)

    subSnap.set_units(position='au', density='g/cm^3',smoothing_length='au',velocity='km/s')
    return subSnap,snap_active,clump_centre_new,clump_velocity_new,subSnap_rotvel,len(subSnap_temp['m'])
# Catch case where user does not supply mode argument when running script from the command line
try:
    print('Running script in',sys.argv[1], 'mode')

    if sys.argv[1] == 'clump':
        flag = 0
        print("Loading all clump files from current directory")
        check = input("This should only be used if there is one clump present, proceed? [y/n] ")
        complete_file_list = glob.glob("run*")

        clump_data_file = complete_file_list[0].split(".")[1]+".dat" # it is safe to index the complete file list here
                                                                     # because all the files are the same clump and so
                                                                     # have the same .dat file.
        complete_file_list = sorted(complete_file_list, key = lambda x: x.split('/')[2].split('.')[1])


    elif sys.argv[1] == 'density' and len(sys.argv) == 2:

        flag = 1
        print('No density provided, using a default value of 1E-3 g/cm')
        density_to_load = 1E-3
        pattern = str(int(np.abs(np.log10(density_to_load)) * 10)).zfill(3)
        complete_file_list = glob.glob("**/HDF5_outputs/*%s.h5" % pattern)
        clump_data_file = complete_file_list[0].split("/")[0]+"/"+complete_file_list[0].split("/")[0].split('p')[1]+".dat"
        complete_file_list = sorted(complete_file_list, key = lambda x: x.split('/')[2].split('.')[1])

    elif sys.argv[1] == 'density'  and len(sys.argv) == 3:
        flag = 1
        density_to_load = float(sys.argv[2])
        pattern = str(int(np.abs(np.log10(density_to_load)) * 10)).zfill(3)
        complete_file_list = glob.glob("**/HDF5_outputs/*%s.h5" % pattern)
        complete_file_list = sorted(complete_file_list, key = lambda x: x.split('/')[2].split('.')[1])
        clump_data_file = complete_file_list[0].split("/")[0]+"/"+complete_file_list[0].split("/")[0].split('p')[1]+".dat"


except IndexError:
    print("Plotting mode not provided...exiting")
    #sys.exit(1)
# @profile
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
        R_x[elem] = np.nan       # We exclude the elements where the bins contain less than
        R_y[elem] = np.nan       # the threshold number of particles

    return R_x, R_y

# @profile
def find_first_non_nan(array):
    for i in array:
        if math.isnan(i) == False:
            return array.index(i) - 1


for file in tqdm(complete_file_list):

    index = complete_file_list.index(file)
    clump_number = file.split("/")[0].split("p")[-1]

    line_colour = mpl_colour_defaults[index]

    prepared_snapshots = prepare_snapshots(file,density_to_load,clump_number,clump_data_file,flag)

    ORIGIN = prepared_snapshots[2]# The position of the clump centre
    x,y,z = ORIGIN[0],ORIGIN[1],ORIGIN[2]
    vel_ORIGIN = prepared_snapshots[3]  # The velocity of the clump centre
    # vx,vy,vz = ORIGIN[0]*kms,ORIGIN[1]*kms,ORIGIN[2]*kms
    subSnap = prepared_snapshots[0]
    rot_vel_snap = prepared_snapshots[4]
    r_clump_centred_midplane_rotvel = np.hypot(rot_vel_snap['x']-(x*au),rot_vel_snap['y']-(y*au))


    r_clump_centred = np.sqrt((subSnap['x']-(x*au))**2 +(subSnap['y']-(y*au))**2 + (subSnap['z']-(z*au))**2)
    r_clump_centred_midplane = np.hypot(subSnap['x']-(x*au),subSnap['y']-(y*au))

    radius_clump = np.sqrt((x)**2 + (y)**2 + (z)**2)

    count = calculate_number_in_bin(r_clump_centred,subSnap['m'],50)


    mass_in_bin = np.cumsum(count[0]) * subSnap['mass'][0].to('jupiter_mass')
    mid_plane_radius = plonk.analysis.particles.mid_plane_radius(subSnap,ORIGIN,ignore_accreted=True)
    rotational_velocity_radial = plonk.analysis.particles.rotational_velocity(subSnap,vel_ORIGIN,ignore_accreted=True)

    rotational_velocity_radial_cyl = plonk.analysis.particles.rotational_velocity(rot_vel_snap,vel_ORIGIN,ignore_accreted=True)
    specific_angular_momentum = plonk.analysis.particles.specific_angular_momentum(subSnap,ORIGIN*au,vel_ORIGIN,ignore_accreted=True).to('cm**2/s')
    total_L = np.sqrt(specific_angular_momentum[:,0]**2 + specific_angular_momentum[:,1]**2 +specific_angular_momentum[:,2]**2)


    spec_mom_binned_2 = calculate_sum(r_clump_centred,total_L,mean_bins_radial)
    tmp1 = spec_mom_binned_2[0][spec_mom_binned_2[0] != 0]
    tmp2 = count[0][count[0] != 0]
    spec_mom_sum_2 = (np.cumsum(tmp1/tmp2))
    # spec_mom_sum_2 = (np.cumsum(spec_mom_binned_2[0]/count[0]))
    # spec_mom_sum_2 = np.cumsum(spec_mom_binned_2[0])


    pad_width = len(spec_mom_binned_2[1][1:]) - len(spec_mom_sum_2)
    spec_mom_sum_2 = np.pad(spec_mom_sum_2, (pad_width,0), 'constant')





    axs_ang_mom.plot(spec_mom_binned_2[1][1:],spec_mom_sum_2,c=line_colour)

    axs_ang_mom.set_xscale('log')
    axs_ang_mom.set_xlabel('R (AU)')
    axs_ang_mom.set_ylabel('J $(\\rm cm^{2}\,s^{-1})$')
    axs_ang_mom.set_xlim(5E-4,50)
    axs_ang_mom.set_yscale('log')


    infall_velocity_radial = plonk.analysis.particles.velocity_radial_spherical_altered(subSnap,ORIGIN*au,vel_ORIGIN,ignore_accreted=True)

    averaged_infall_radial = calculate_mean(r_clump_centred,infall_velocity_radial)
    averaged_rotational_velocity = calculate_mean(r_clump_centred_midplane_rotvel,rotational_velocity_radial_cyl)
    averaged_density_radial = calculate_mean(r_clump_centred,subSnap['density'])
    averaged_temperature_radial = calculate_mean(r_clump_centred,subSnap['my_temp'])


    # We can find the indexes of the bins that hold fewer than 50 partcles. We
    # use this to define an area of confidence which will show in the plots.
    low_confidence_values = [i for i, a in enumerate(count[0]) if a <= 50]




    binned_r_clump_with_nans = highlight_low_confidence_bins(averaged_infall_radial,low_confidence_values)[0]
    average_temp_with_nans = highlight_low_confidence_bins(averaged_temperature_radial,low_confidence_values)[1]
    average_density_with_nans = highlight_low_confidence_bins(averaged_density_radial,low_confidence_values)[1]
    average_infall_with_nans = highlight_low_confidence_bins(averaged_infall_radial,low_confidence_values)[1]
    average_rotational_with_nans = highlight_low_confidence_bins(averaged_rotational_velocity,low_confidence_values)[1]


    averaged_infall_radial_interp = np.interp(np.arange(len(averaged_infall_radial[0])),
                                    np.arange(len(averaged_infall_radial[0]))[np.isnan(averaged_infall_radial[0]) == False],
                                    averaged_infall_radial[0][np.isnan(averaged_infall_radial[0]) == False])

    rotational_energy = 0.5 * subSnap['m'][0].to('g') * rotational_velocity_radial_cyl.to('cm/s') **2
    rotational_energy_binned = calculate_sum(r_clump_centred_midplane_rotvel,rotational_energy,mean_bins_radial)
    cumsum_erot = np.cumsum(rotational_energy_binned[0])
    grav_rad = r_clump_centred.magnitude
    gravitational_energy = calculate_gravitational_energy(grav_rad,mean_bins_radial,subSnap['m'][0].to('g'))
    gravitational_energy_binned = calculate_sum(r_clump_centred_midplane,gravitational_energy,mean_bins_radial)
    cumsum_egrav = np.cumsum(gravitational_energy_binned[0])
    thermal_energy = calculate_thermal_energy(subSnap)
    thermal_energy_binned = calculate_sum(r_clump_centred_midplane,thermal_energy,mean_bins_radial)
    cumsum_etherm = np.cumsum(thermal_energy_binned[0])
    with np.errstate(invalid='ignore'):
        alpha = cumsum_etherm / cumsum_egrav

    with np.errstate(invalid='ignore'):
        beta = cumsum_erot / cumsum_egrav




    smoothed_infall         = savgol_filter(averaged_infall_radial[0],15,3)
    smoothed_infall_nans    = savgol_filter(average_infall_with_nans ,15,3)







    y = smoothed_infall.copy()
    x = averaged_infall_radial[1][1:]

    y[np.isnan(y)] = 0
    y_2 = savgol_filter(averaged_infall_radial[0],8,3)
    x = averaged_infall_radial[1][1:]



    x_smooth = np.logspace(np.log10(min(averaged_infall_radial[1])), np.log10(max(averaged_infall_radial[1])), 2000)
    bspl = splrep(averaged_infall_radial[1][10:-10],y_2[9:-10])
    bspl_y = splev(x_smooth, bspl)

    peaks, _ = find_peaks(bspl_y,height=1,distance=300)
    sub = -bspl_y[peaks[0]:peaks[1]]
    minima, _ = find_peaks(-bspl_y[:peaks[1]],distance=300)

    # Tidily set axes limits and scale types
    for i in range(0,3):
        for j in range(0,2):
            f_radial_axs[i,j].set_xscale('log')
            f_radial_axs[i,j].set_xlim(5E-4,50)
            f_radial_axs[i,j].set_xlabel('R (AU)',fontsize=10)
            f_radial_axs[i,j].tick_params(axis="x", labelsize=8)
            f_radial_axs[i,j].tick_params(axis="y", labelsize=8)

    for i in [0,2]:
        for j in [0,1]:
            f_radial_axs[i,j].set_yscale('log')

    figure_indexes = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]
    figure_ylimits = [(1E-13,1E-2),(10,8000),(0,7.5),(-10,10),(0.1,80),(1E-5,10000)]
    figure_ylabels = ['Density $(\\rm g\,cm^{-3})$','Temperature (K)','Rotational Velocity $(\\rm km\,s^{-1})$',
                      'Infall Velocity $(\\rm km\,s^{-1})$',r'Mass (M$_{J}$)','Energy ratio']

    for index,label,limit in zip(figure_indexes,figure_ylabels,figure_ylimits):
        f_radial_axs[index].set_ylabel(label,fontsize=10)
        f_radial_axs[index].set_ylim(limit)

    f_radial_axs[0,0].plot(averaged_density_radial[1][1:],averaged_density_radial[0],
                           c = line_colour,linestyle="--",linewidth = 1,alpha=0.5)
    f_radial_axs[0,0].plot(binned_r_clump_with_nans,average_density_with_nans,
                           c = line_colour)

    axs_test.plot(averaged_density_radial[1][1:],averaged_density_radial[0],
                           c = line_colour,linestyle="--",linewidth = 2,alpha=0.5)
    axs_test.plot(binned_r_clump_with_nans,average_density_with_nans,linewidth=2,
                           c = line_colour,alpha=0.5)
    #
    # axs_test.set_ylabel('Density (g/cm^3)')
    # axs_test.set_xlabel('R (AU)')
    # axs_test.set_yscale('log')
    # axs_test.set_xscale('log')
    #
    # axs_test.set_ylim(1e-13,1e-2)
    f_radial_axs[0,1].plot(averaged_temperature_radial[1][1:],
                           averaged_temperature_radial[0],c = line_colour,
                           linestyle="--",linewidth =1,alpha=0.5)
    f_radial_axs[0,1].plot(binned_r_clump_with_nans,average_temp_with_nans,
                           c = line_colour)

    f_radial_axs[1,0].plot(averaged_rotational_velocity[1][1:],
                           averaged_rotational_velocity[0],c = line_colour,
                           linestyle="--",linewidth = 1,alpha=0.5)
    f_radial_axs[1,0].plot(binned_r_clump_with_nans,average_rotational_with_nans,
                           c = line_colour)


    # infall_with_nans_eq_0 = averaged_infall_radial[0].copy()
    # infall_with_nans_eq_0[np.isnan(infall_with_nans_eq_0)] = 0
    # x_smooth = averaged_infall_radial[1][1:]
    # bspl_y = averaged_infall_radial[0]
    # f_radial_axs[1,1].plot(averaged_infall_radial[1][1:],smoothed_infall,
    #                        c = line_colour,linestyle="--",linewidth = 1,alpha=0.5)
    # f_radial_axs[1,1].plot(binned_r_clump_with_nans,smoothed_infall_nans,
    #                        c = line_colour)
    # f_radial_axs[1,1].plot(x_smooth[peaks],bspl_y[peaks],'+',c=line_colour)
    # f_radial_axs[1,1].plot(averaged_infall_radial[1][1:],averaged_infall_radial[0],
    #                        c = line_colour,linestyle="--",linewidth = 1,alpha=0.5)
    # f_radial_axs[1,1].plot(binned_r_clump_with_nans,average_infall_with_nans,
    #                        c = line_colour)
    # f_radial_axs[1,1].plot(x_smooth[peaks],bspl_y[peaks],'+',c=line_colour)

    # f_radial_axs[1,1].plot(x_smooth[minima],bspl_y[minima],'+',c=line_colour)
    f_radial_axs[1,1].plot(averaged_infall_radial[1][1:],averaged_infall_radial[0],
                           c = line_colour,linestyle="-",linewidth = 1,alpha=0.5)
    # f_radial_axs[1,1].plot(x_smooth,bspl_y,
    #                        c = line_colour,linewidth = 1,alpha=1)

    f_radial_axs[1,1].plot(x_smooth[peaks],bspl_y[peaks],'+',c=line_colour)
    # f_radial_axs[1,1].plot(x_smooth,-bspl_y,c='red')
    # f_radial_axs[1,1].plot(x_smooth[minima],-bspl_y[minima],'*',c='red')
    # f_radial_axs[1,1].plot(x,y_2,c='green')






    # f_radial_axs[2,0].plot(count[1][1:],mass_in_bin,linewidth=1)

    f_radial_axs[2,0].plot(count[1][1:],mass_in_bin,linewidth=1,label = 'Fragment %s' % clump_number)
    f_radial_axs[2,0].set_yscale('linear')

    f_radial_axs[2,0].legend(loc='upper left')
    f_radial_axs[2,1].plot(gravitational_energy_binned[1][1:],alpha,linewidth=1,
                           c=line_colour)
    f_radial_axs[2,1].plot(gravitational_energy_binned[1][1:],beta,linewidth=1,
                           c=line_colour)
    f_radial_axs[2,1].axhline(y=1,c='black',linestyle='--',linewidth=1.5)
    f_radial_axs[2,1].set_xscale('log')
    f_radial_axs[2,1].set_yscale('log')
    f_radial_axs[2,1].set_ylim(1e-3,4)

    f_radial_axs[0,0].annotate("a)", xy=(0.9, 0.9), xycoords="axes fraction")
    f_radial_axs[0,1].annotate("b)", xy=(0.9, 0.9), xycoords="axes fraction")
    f_radial_axs[1,0].annotate("c)", xy=(0.9, 0.9), xycoords="axes fraction")
    f_radial_axs[1,1].annotate("d)", xy=(0.9, 0.9), xycoords="axes fraction")
    f_radial_axs[2,0].annotate("e)", xy=(0.9, 0.9), xycoords="axes fraction")
    f_radial_axs[2,1].annotate("f)", xy=(0.9, 0.9), xycoords="axes fraction")
    fig_radial.align_ylabels()
    fig_radial.tight_layout(pad=0.40)


    # Put a legend to the right of the current axis
    if sys.argv[1] == 'density':
        clump_density = '{0:.2e}'.format(density_to_load)
    else:
        clump_density = 'clump'

    second_core_radius = 0.0000000
    second_core_count  = 0.0000000
    second_core_mass   = 0.0000000
    N_sc               = 0.0000000
    L_sc               = 0.0000000
    egrav_sc           = 0.0000000
    etherm_sc           = 0.0000000
    erot_sc           = 0.0000000
    first_core_radius_outer  = 0.0000000
    first_core_radius_inner  = 0.0000000
    first_core_count   = 0.0000000
    L_fco               = 0.0000000
    L_fci               = 0.0000000
    egrav_fc           = 0.0000000
    etherm_fc           = 0.0000000
    erot_fc           = 0.0000000
    first_core_mass    = 0.0000000
    N_fc               = 0.0000000
    alpha_fc               = 0.0000000
    beta_fc               = 0.0000000
    alpha_sc               = 0.0000000
    beta_sc               = 0.0000000
    only_one_core      = False
    weak_fc = 0.000000
    rhocritID = 1


    # Write core information to the clump_results file for plotting later on.
    if len(peaks) == 1:
        first_core_radius_outer = float('{0:.5e}'.format(x_smooth[peaks[0]]))
        first_core_radius_inner = float('{0:.5e}'.format(x_smooth[minima[0]]))

        first_core_count   = calculate_number_in_bin(r_clump_centred,
                                                     subSnap['density'],
                                                     float(first_core_radius_outer))[0]
        first_core_mass =   float('{0:.5e}'.format(np.cumsum(first_core_count)[-1]
                                                   * subSnap['m'][0].to('jupiter_mass').magnitude))

        first_core_count_outer   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(first_core_radius_outer))[0]
        first_core_count_inner   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(first_core_radius_inner))[0]

        first_core_mass_outer =   float('{0:.5e}'.format(np.cumsum(first_core_count_outer)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))
        first_core_mass_inner =   float('{0:.5e}'.format(np.cumsum(first_core_count_inner)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))

        first_core_bin_outer = np.digitize(first_core_radius_outer,mean_bins_radial)-1
        first_core_bin_inner = np.digitize(first_core_radius_inner,mean_bins_radial)-1

        L_fco = spec_mom_sum_2[first_core_bin_outer]
        L_fci = spec_mom_sum_2[first_core_bin_inner]
        egrav_fco = cumsum_egrav[first_core_bin_outer]
        etherm_fco = cumsum_etherm[first_core_bin_outer]
        erot_fco = cumsum_erot[first_core_bin_outer]
        egrav_fci = cumsum_egrav[first_core_bin_inner]
        etherm_fci = cumsum_etherm[first_core_bin_inner]
        erot_fci = cumsum_erot[first_core_bin_inner]



        axs_ang_mom.axvline(x=x_smooth[peaks[0]],c=line_colour,linestyle='dotted',linewidth=1)
        N_fc = np.cumsum(first_core_count)[-1]
        alpha_fco = etherm_fco/egrav_fco
        beta_fco = erot_fco/egrav_fco
        alpha_fci = etherm_fci/egrav_fci
        beta_fci = erot_fci/egrav_fci
        # If the peak finding algorithm only finds one peak, we flag this clump
        # as having only one core for checking later on.
        only_one_core = True
    if len(peaks) >= 2:
        first_core_radius_outer = float('{0:.5e}'.format(x_smooth[peaks[1]]))
        first_core_radius_inner = float('{0:.5e}'.format(x_smooth[minima[0]]))
        first_core_count_outer   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(first_core_radius_outer))[0]
        first_core_count_inner   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(first_core_radius_inner))[0]

        first_core_mass_outer =   float('{0:.5e}'.format(np.cumsum(first_core_count_outer)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))
        first_core_mass_inner =   float('{0:.5e}'.format(np.cumsum(first_core_count_inner)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))


        if bspl_y[peaks][1] < 0.5:
            weak_fc = 1
        #
        second_core_radius = float('{0:.5e}'.format(x_smooth[peaks[0]]))
        second_core_count   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(second_core_radius))[0]
        second_core_mass =   float('{0:.5e}'.format(np.cumsum(second_core_count)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))

        second_core_L_R = r_clump_centred[np.abs(spec_mom_binned_2[1]-second_core_radius).argmin()]
        second_core_L = np.where(r_clump_centred == second_core_L_R)
        N_fc = np.cumsum(first_core_count_outer)[-1]
        N_sc = np.cumsum(second_core_count)[-1]
        first_core_bin_outer = np.digitize(first_core_radius_outer,mean_bins_radial)-1
        first_core_bin_inner = np.digitize(first_core_radius_inner,mean_bins_radial)-1

        second_core_bin = np.digitize(second_core_radius,mean_bins_radial)-1
        L_fco = spec_mom_sum_2[first_core_bin_outer] #* ((subSnap['m'][0])/(np.cumsum(first_core_count)[-1] * subSnap['m'][0]))
        L_fci = spec_mom_sum_2[first_core_bin_inner] #* ((subSnap['m'][0])/(np.cumsum(first_core_count)[-1] * subSnap['m'][0]))

        L_sc = spec_mom_sum_2[second_core_bin]#* ((subSnap['m'][0])/(np.cumsum(second_core_count)[-1] * subSnap['m'][0]))
        egrav_fco = cumsum_egrav[first_core_bin_outer]
        etherm_fco = cumsum_etherm[first_core_bin_outer]
        erot_fco = cumsum_erot[first_core_bin_outer]
        egrav_fci = cumsum_egrav[first_core_bin_inner]
        etherm_fci = cumsum_etherm[first_core_bin_inner]
        erot_fci = cumsum_erot[first_core_bin_inner]

        egrav_sc = cumsum_egrav[second_core_bin]
        etherm_sc = cumsum_etherm[second_core_bin]
        erot_sc = cumsum_erot[second_core_bin]
        alpha_fci =   etherm_fci/egrav_fci
        beta_fci = erot_fci/egrav_fci
        alpha_fco =   etherm_fco/egrav_fco
        beta_fco = erot_fco/egrav_fco

        alpha_sc = etherm_sc/egrav_sc
        beta_sc = erot_sc/egrav_sc
        axs_ang_mom.axvline(x=x_smooth[peaks[0]],c=line_colour,linestyle='--',linewidth=1)
        axs_ang_mom.axvline(x=x_smooth[peaks[1]],c=line_colour,linestyle='dotted',linewidth=1)

    clump_results.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % \
                       (file.split("/")[-1],\
                       clump_density,\
                       second_core_radius * 2092.51,
                       L_sc,\
                       egrav_sc,\
                       etherm_sc,\
                       erot_sc,\
                       alpha_sc,\
                       beta_sc,\
                       second_core_mass,\
                       N_sc,\
                       first_core_radius_inner,\
                       first_core_radius_outer,\
                       L_fci,\
                       egrav_fci,\
                       etherm_fci,\
                       erot_fci,\
                       alpha_fci,\
                       beta_fci,\
                       L_fco,\
                       egrav_fco,\
                       etherm_fco,\
                       erot_fco,\
                       alpha_fco,\
                       beta_fco,\
                       first_core_mass_inner,
                       first_core_mass_outer,
                       N_fc,\
                       radius_clump,
                       only_one_core,
                       rhocritID))

fig_radial.savefig("%s/clump_profiles_new.png" % cwd,dpi = 200)
fig_ang_mom.savefig("%s/specific_angular_momentum.png" % cwd,dpi = 200)

# fig_test.savefig("%s/infall.png" % cwd,dpi = 500)

# import plonk
# from scipy.signal import find_peaks
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# import os
# import sys
# from scipy import stats
# from scipy.interpolate import splev, splrep
# import pint
# from tqdm.auto import tqdm
# import warnings
# from scipy.signal import savgol_filter
# from digitize import calculate_gravitational_energy
# import time
# import pandas as pd
#
# # Author: Adam Fenton
# cwd = os.getcwd()
#
# # Load units for use later, useful for derived quantities.
# au = plonk.units('au')
# kms = plonk.units('km/s')
# density_to_load = None #
# mean_bins_radial = np.logspace(np.log10(0.001),np.log10(50),40)
# mpl_colour_defaults=plt.rcParams['axes.prop_cycle'].by_key()['color'] # MLP default colours
#
# # Initalise the figure and output file which is written to later on
# fig_radial, f_radial_axs = plt.subplots(nrows=3,ncols=2,figsize=(7,8))
# fig_ang_mom, axs_ang_mom = plt.subplots(figsize=(7,8))
# fig_test, axs_test = plt.subplots(figsize=(7,8))
# labels = []
# clump_results = open('clump-results.dat', 'w')
# # Ignore pesky warnings when stripping unit off of pint quantity when downcasting to array
# if hasattr(pint, 'UnitStrippedWarning'):
#     warnings.simplefilter('ignore', category=pint.UnitStrippedWarning)
# np.seterr(divide='ignore', invalid='ignore')
#
# def calculate_sum(binned_quantity,summed_quantity,bins):
#     return stats.binned_statistic(binned_quantity, summed_quantity, 'sum', bins=bins)
#
# def calculate_mean(binned_quantity,mean_quantity):
#     bins = np.logspace(np.log10(0.0001),np.log10(50),40)
#     return stats.binned_statistic(binned_quantity, mean_quantity, 'mean', bins=bins)
#
# def calculate_number_in_bin(binned_quantity,mean_quantity,width):
#     bins=np.logspace(np.log10(0.0001),np.log10(width),40)
#     return stats.binned_statistic(binned_quantity, mean_quantity, 'count', bins=bins)
#
# def calculate_thermal_energy(subSnap):
#     U = 3/2 * 1.38E-16 * subSnap['my_temp'] * ((subSnap['m'][0].to('g'))/(1.67E-24))
#     U = U.magnitude
#     U *= ((subSnap['my_temp'].magnitude>2000)*(1/1.2)+(subSnap['my_temp'].magnitude<2000)*(1/2.381))
#     return U
#
# def prepare_snapshots(snapshot,density_to_load,clump_number,clump_data_file,density_flag):
#     ''' Load full snapshot as plonk object and initialise subsnap centred on clump.
#         Also apply filter to exclude dead and accreted particles with `accreted_mask`
#         and load units
#     '''
#
#     snap = plonk.load_snap(snapshot)
#     sinks = snap.sinks
#     if type(sinks['m'].magnitude) == np.float64:
#         central_star_mass = sinks['m']
#     else:
#         central_star_mass = sinks['m'][0]
#     snap.set_units(position='au', density='g/cm^3',smoothing_length='au',velocity='km/s')
#     h = snap['smoothing_length']
#     snapshot_file_path = os.path.dirname(os.path.abspath(snapshot))
#
#     clump_data_file_name = snapshot_file_path.split("/")[-2].split("p")[-1]+".dat"
#
#     clump_data_file = snapshot_file_path + "/"+clump_data_file_name
#
#
#
#
#     ### This is where the issue is, it works for the E-3 because there is no
#     ### particle in the snapshot with a higher density but when plotting lower
#     ### densities, the recentering is off
#
#     clump_location_data = pd.read_csv(clump_data_file,names=["rho", "x", "y", "z",
#                                                              "vx", "vy", "vz"])
#
#     if flag == 1:
#         index = clump_location_data['rho'].sub(density_to_load).abs().idxmin()
#         row_of_interest = clump_location_data.iloc[[index]]
#         clump_centre = row_of_interest.values[0][1:4]
#         velocity_conversion = 2.978E6/1e5
#         clump_velocity = row_of_interest.values[0][4:] * velocity_conversion * kms
#         x,y,z = clump_centre[0],clump_centre[1],clump_centre[2]
#
#     if flag == 0:
#         snapshot_density = 10**(-(int(snapshot.split(".")[3].split("0")[1])))
#         index = clump_location_data['rho'].sub(snapshot_density).abs().idxmin()
#         row_of_interest = clump_location_data.iloc[[index]]
#         clump_centre = row_of_interest.values[0][1:4]
#         velocity_conversion = 2.978E6/1e5
#         clump_velocity = row_of_interest.values[0][4:] * velocity_conversion * kms
#         x,y,z = clump_centre[0],clump_centre[1],clump_centre[2]
#
#     # clump_velocity = snap['velocity'][id]
#     accreted_mask = snap['smoothing_length'] > 0
#     snap_active = snap[accreted_mask]
#     PID = int(file.split('.')[2]) - 1
#     PID_index = np.where(snap_active['id'] == PID)
#     clump_centre_new = snap_active['position'][PID_index][0].magnitude
#
#     subSnap=plonk.analysis.filters.sphere(snap=snap_active,radius = (50*au),center=clump_centre *au)
#
#     subSnap_rotvel=plonk.analysis.filters.cylinder(snap=snap_active,radius = (50*au),height=(0.75*au),center=clump_centre * au)
#
#     subSnap.set_units(position='au', density='g/cm^3',smoothing_length='au',velocity='km/s')
#     return subSnap,snap_active,clump_centre,clump_velocity,subSnap_rotvel
# # Catch case where user does not supply mode argument when running script from the command line
#
# try:
#     print('Running script in',sys.argv[1], 'mode')
#
#     if sys.argv[1] == 'clump':
#         flag = 0
#         print("Loading all clump files from current directory")
#         check = input("This should only be used if there is one clump present, proceed? [y/n] ")
#         complete_file_list = glob.glob("run*")
#         clump_data_file = complete_file_list[0].split(".")[1]+".dat" # it is safe to index the complete file list here
#                                                                      # because all the files are the same clump and so
#                                                                      # have the same .dat file.
#
#
#     elif sys.argv[1] == 'density' and len(sys.argv) == 2:
#
#         flag = 1
#         print('No density provided, using a default value of 1E-3 g/cm')
#         density_to_load = 1E-3
#         pattern = str(int(np.abs(np.log10(density_to_load)) * 10)).zfill(3)
#         complete_file_list = glob.glob("**/HDF5_outputs/*%s.h5" % pattern)
#         clump_data_file = complete_file_list[0].split("/")[0]+"/"+complete_file_list[0].split("/")[0].split('p')[1]+".dat"
#
#     elif sys.argv[1] == 'density'  and len(sys.argv) == 3:
#         flag = 1
#         density_to_load = float(sys.argv[2])
#         pattern = str(int(np.abs(np.log10(density_to_load)) * 10)).zfill(3)
#         complete_file_list = glob.glob("**/HDF5_outputs/*%s.h5" % pattern)
#         clump_data_file = complete_file_list[0].split("/")[0]+"/"+complete_file_list[0].split("/")[0].split('p')[1]+".dat"
#
#
# except IndexError:
#     print("Plotting mode not provided...exiting")
#     sys.exit(1)
#
# def highlight_low_confidence_bins(quantity,elems):
#     ''' A function that, when provided with an input quantity in the form of
#         a binned_statistic and an array of elements, returns two arrays that
#         are copies of the input array but with bins with fewer than 50 particles
#         in changed to nan values. We copy the arrays so the originals remain
#         unaltered.
#     '''
#     R_y = quantity[0].copy()     # Resulting arrays are initally copies of the input arrays.
#     R_x = quantity[1][1:].copy()
#
#     for elem in elems:
#         R_x[elem] = np.nan       # We exclude the elements where the bins contain less than
#         R_y[elem] = np.nan       # the threshold number of particles
#
#     return R_x, R_y
#
#
# def find_first_non_nan(array):
#     for i in array:
#         if math.isnan(i) == False:
#             return array.index(i) - 1
#
# for file in tqdm(complete_file_list):
#     index = complete_file_list.index(file)
#     clump_number = file.split("/")[0].split("p")[-1]
#     labels.append('clump_'+ clump_number)
#
#     PID = int(file.split('.')[2]) - 1
#     line_colour = mpl_colour_defaults[index]
#
#     prepared_snapshots = prepare_snapshots(file,density_to_load,clump_number,clump_data_file,flag)
#     ORIGIN = prepared_snapshots[2]# The position of the clump centre
#     ##### TESTING #####
#     # fullsnap = prepared_snapshots[1]
#     # PID_index = np.where(fullsnap['id'] == PID)
#     # ORIGIN = fullsnap['position'][PID_index].magnitude
#
#     # x,y,z = fullsnap['x'][PID_index].magnitude,fullsnap['y'][PID_index].magnitude,fullsnap['z'][PID_index].magnitude
#
#
#     x,y,z = ORIGIN[0],ORIGIN[1],ORIGIN[2]
#
#     vel_ORIGIN = prepared_snapshots[3]  # The velocity of the clump centre
#     # vx,vy,vz = ORIGIN[0]*kms,ORIGIN[1]*kms,ORIGIN[2]*kms
#     subSnap = prepared_snapshots[0]
#     rot_vel_snap = prepared_snapshots[4]
#     r_clump_centred_midplane_rotvel = np.hypot(rot_vel_snap['x']-(x*au),rot_vel_snap['y']-(y*au))
#
#     r_clump_centred = np.sqrt((subSnap['x']-(x*au))**2 +(subSnap['y']-(y*au))**2 + (subSnap['z']-(z*au))**2)
#
#     # r_clump_centred = np.sqrt((subSnap['x']-(x*au))**2 +(subSnap['y']-(y*au))**2 + (subSnap['z']-(z*au))**2)
#     r_clump_centred_midplane = np.hypot(subSnap['x']-(x*au),subSnap['y']-(y*au))
#
#
#     radius_clump = np.sqrt((x)**2 + (y)**2 + (z)**2)
#
#
#     count = calculate_number_in_bin(r_clump_centred,subSnap['m'],50)
#
#     mass_in_bin = np.cumsum(count[0]) * subSnap['mass'][0].to('jupiter_mass')
#     mid_plane_radius = plonk.analysis.particles.mid_plane_radius(subSnap,ORIGIN,ignore_accreted=True)
#     rotational_velocity_radial = plonk.analysis.particles.rotational_velocity(subSnap,vel_ORIGIN,ignore_accreted=True)
#
#     rotational_velocity_radial_cyl = plonk.analysis.particles.rotational_velocity(rot_vel_snap,vel_ORIGIN,ignore_accreted=True)
#     specific_angular_momentum = plonk.analysis.particles.specific_angular_momentum(subSnap,ORIGIN*au,vel_ORIGIN,ignore_accreted=True).to('cm**2/s')
#     total_L = np.sqrt(specific_angular_momentum[:,0]**2 + specific_angular_momentum[:,1]**2 +specific_angular_momentum[:,2]**2)
#
#
#     spec_mom_binned_2 = calculate_sum(r_clump_centred,total_L,mean_bins_radial)
#     spec_mom_sum_2 = (np.cumsum(spec_mom_binned_2[0]/count[0]))
#     print(count[0])
#     stop
#     # spec_mom_sum_2 = np.cumsum(spec_mom_binned_2[0])
#
#
#
#
#
#     axs_ang_mom.plot(spec_mom_binned_2[1][1:],spec_mom_sum_2,label="Total Magnitude",c=line_colour)
#
#     axs_ang_mom.set_xscale('log')
#     axs_ang_mom.set_xlabel('R (AU)')
#     axs_ang_mom.set_ylabel('J $(\\rm cm^{2}\,s^{-1})$')
#     axs_ang_mom.set_xlim(1e-4,100)
#     axs_ang_mom.set_yscale('log')
#     infall_velocity_radial = plonk.analysis.particles.velocity_radial_spherical_altered(subSnap,ORIGIN*au,vel_ORIGIN,ignore_accreted=True)
#
#     averaged_infall_radial = calculate_mean(r_clump_centred,infall_velocity_radial)
#     averaged_rotational_velocity = calculate_mean(r_clump_centred_midplane_rotvel,rotational_velocity_radial_cyl)
#     averaged_density_radial = calculate_mean(r_clump_centred,subSnap['density'])
#     averaged_temperature_radial = calculate_mean(r_clump_centred,subSnap['my_temp'])
#
#     # We can find the indexes of the bins that hold fewer than 50 partcles. We
#     # use this to define an area of confidence which will show in the plots.
#     low_confidence_values = [i for i, a in enumerate(count[0]) if a <= 50]
#
#
#     # plt.scatter(r_clump_centred,subSnap['density'],s=0.1)
#     # plt.plot(averaged_density_radial[1][1:],averaged_density_radial[0])
#     # plt.xscale('log')
#     # plt.yscale('log')
#     # plt.show()
#     # stop
#     binned_r_clump_with_nans = highlight_low_confidence_bins(averaged_infall_radial,low_confidence_values)[0]
#     average_temp_with_nans = highlight_low_confidence_bins(averaged_temperature_radial,low_confidence_values)[1]
#     average_density_with_nans = highlight_low_confidence_bins(averaged_density_radial,low_confidence_values)[1]
#     average_infall_with_nans = highlight_low_confidence_bins(averaged_infall_radial,low_confidence_values)[1]
#     average_rotational_with_nans = highlight_low_confidence_bins(averaged_rotational_velocity,low_confidence_values)[1]
#
#
#     averaged_infall_radial_interp = np.interp(np.arange(len(averaged_infall_radial[0])),
#                                     np.arange(len(averaged_infall_radial[0]))[np.isnan(averaged_infall_radial[0]) == False],
#                                     averaged_infall_radial[0][np.isnan(averaged_infall_radial[0]) == False])
#
#     rotational_energy = 0.5 * subSnap['m'][0].to('g') * rotational_velocity_radial_cyl.to('cm/s') **2
#     rotational_energy_binned = calculate_sum(r_clump_centred_midplane_rotvel,rotational_energy,mean_bins_radial)
#     cumsum_erot = np.cumsum(rotational_energy_binned[0])
#     grav_rad = r_clump_centred.magnitude
#     gravitational_energy = calculate_gravitational_energy(grav_rad,mean_bins_radial,subSnap['m'][0].to('g'))
#     gravitational_energy_binned = calculate_sum(r_clump_centred_midplane,gravitational_energy,mean_bins_radial)
#     cumsum_egrav = np.cumsum(gravitational_energy_binned[0])
#     thermal_energy = calculate_thermal_energy(subSnap)
#     thermal_energy_binned = calculate_sum(r_clump_centred_midplane,thermal_energy,mean_bins_radial)
#     cumsum_etherm = np.cumsum(thermal_energy_binned[0])
#     with np.errstate(invalid='ignore'):
#         alpha = cumsum_etherm / cumsum_egrav
#
#     with np.errstate(invalid='ignore'):
#         beta = cumsum_erot / cumsum_egrav
#
#
#
#
#     smoothed_infall         = savgol_filter(averaged_infall_radial[0],15,3)
#     smoothed_infall_nans    = savgol_filter(average_infall_with_nans ,15,3)
#
#
#
#
#
#
#
#     y = smoothed_infall.copy()
#     x = averaged_infall_radial[1][1:]
#
#     y[np.isnan(y)] = 0
#     y_2 = savgol_filter(averaged_infall_radial_interp ,15,3)
#     x = averaged_infall_radial[1][1:]
#
#
#
#     x_smooth = np.logspace(np.log10(min(averaged_infall_radial[1])), np.log10(max(averaged_infall_radial[1])), 2000)
#     bspl = splrep(x,y_2)
#     bspl_y = splev(x_smooth, bspl)
#     peaks, _ = find_peaks(bspl_y,height=0.1,distance=100,prominence=0.05)
#
#
#     # Tidily set axes limits and scale types
#     for i in range(0,3):
#         for j in range(0,2):
#             f_radial_axs[i,j].set_xscale('log')
#             f_radial_axs[i,j].set_xlim(1E-4,100)
#             f_radial_axs[i,j].set_xlabel('R (AU)',fontsize=6.5)
#             f_radial_axs[i,j].tick_params(axis="x", labelsize=8)
#             f_radial_axs[i,j].tick_params(axis="y", labelsize=8)
#
#     for i in [0,2]:
#         for j in [0,1]:
#             f_radial_axs[i,j].set_yscale('log')
#
#     figure_indexes = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]
#     figure_ylimits = [(1E-13,1E-2),(10,8000),(0,7),(-0.5,7),(0.1,40),(1E-5,10000)]
#     figure_ylabels = ['Density $(\\rm g\,cm^{-3})$','Temperature (K)','Rotational Velocity $(\\rm km\,s^{-1})$',
#                       'Infall Velocity $(\\rm km\,s^{-1})$','Mass $(\\rm M_{J})$','Energy ratio']
#
#     for index,label,limit in zip(figure_indexes,figure_ylabels,figure_ylimits):
#         f_radial_axs[index].set_ylabel(label,fontsize=6.5)
#         f_radial_axs[index].set_ylim(limit)
#
#     for density in [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]:
#         f_radial_axs[0,0].axhline(y=density,c='black',linestyle='--',linewidth=0.5)
#     f_radial_axs[0,0].plot(averaged_density_radial[1][1:],averaged_density_radial[0],
#                            c = line_colour,linestyle="--",linewidth = 1,alpha=0.5)
#     f_radial_axs[0,0].plot(binned_r_clump_with_nans,average_density_with_nans,
#                            c = line_colour,alpha=1)
#
#     f_radial_axs[0,1].plot(averaged_temperature_radial[1][1:],
#                            averaged_temperature_radial[0],c = line_colour,
#                            linestyle="--",linewidth =1,alpha=0.5)
#     f_radial_axs[0,1].plot(binned_r_clump_with_nans,average_temp_with_nans,
#                            c = line_colour,alpha=1)
#
#     f_radial_axs[1,0].plot(averaged_rotational_velocity[1][1:],
#                            averaged_rotational_velocity[0],c = line_colour,
#                            linestyle="--",linewidth = 1,alpha=0.5)
#     f_radial_axs[1,0].plot(binned_r_clump_with_nans,average_rotational_with_nans,
#                            c = line_colour,alpha=1)
#
#
#     # infall_with_nans_eq_0 = averaged_infall_radial[0].copy()
#     # infall_with_nans_eq_0[np.isnan(infall_with_nans_eq_0)] = 0
#
#
#     f_radial_axs[1,1].plot(averaged_infall_radial[1][1:],smoothed_infall,
#                            c = line_colour,linestyle="--",linewidth = 1,alpha=0.5)
#     f_radial_axs[1,1].plot(binned_r_clump_with_nans,smoothed_infall_nans,
#                            c = line_colour,alpha=1)
#     f_radial_axs[1,1].plot(x_smooth[peaks],bspl_y[peaks],'+',c=line_colour)
#
#
#
#
#
#
#     f_radial_axs[2,0].plot(count[1][1:],mass_in_bin,linewidth=1)
#     f_radial_axs[2,0].set_yscale('linear')
#
#     f_radial_axs[2,1].plot(gravitational_energy_binned[1][1:],alpha,linewidth=1,
#                            c=line_colour)
#     f_radial_axs[2,1].plot(gravitational_energy_binned[1][1:],beta,linewidth=1,
#                            c=line_colour)
#     f_radial_axs[2,1].axhline(y=1,c='black',linestyle='--',linewidth=1.5)
#     f_radial_axs[2,1].set_xscale('log')
#     f_radial_axs[2,1].set_yscale('log')
#     f_radial_axs[2,1].set_ylim(1e-2,2)
#
#
#     f_radial_axs[0,0].annotate("a)", xy=(0.05, 0.9), xycoords="axes fraction")
#     f_radial_axs[0,1].annotate("b)", xy=(0.05, 0.9), xycoords="axes fraction")
#     f_radial_axs[1,0].annotate("c)", xy=(0.05, 0.9), xycoords="axes fraction")
#     f_radial_axs[1,1].annotate("d)", xy=(0.05, 0.9), xycoords="axes fraction")
#     f_radial_axs[2,0].annotate("e)", xy=(0.05, 0.9), xycoords="axes fraction")
#     f_radial_axs[2,1].annotate("f)", xy=(0.05, 0.9), xycoords="axes fraction")
#     fig_radial.align_ylabels()
#     fig_radial.tight_layout(pad=0.40)
#     # fig_radial.subplots_adjust(bottom=0.1)
#     # fig_radial.legend(labels=labels, loc="lower center", ncol=6)
#
#     if sys.argv[1] == 'density':
#         clump_density = '{0:.2e}'.format(density_to_load)
#     else:
#         clump_density = 'clump'
#
#     second_core_radius = 0.0000000
#     second_core_count  = 0.0000000
#     second_core_mass   = 0.0000000
#     L_sc               = 0.0000000
#     egrav_sc           = 0.0000000
#     etherm_sc           = 0.0000000
#     erot_sc           = 0.0000000
#     first_core_radius  = 0.0000000
#     first_core_count   = 0.0000000
#     L_fc               = 0.0000000
#     egrav_fc           = 0.0000000
#     etherm_fc           = 0.0000000
#     erot_fc           = 0.0000000
#     first_core_mass    = 0.0000000
#     weak_fc = 0.000000
#     rhocritID = 1
#
#
#     # Write core information to the clump_results file for plotting later on.
#     if len(peaks) == 1:
#         first_core_radius = float('{0:.5e}'.format(x_smooth[peaks[0]]))
#         first_core_count   = calculate_number_in_bin(r_clump_centred,
#                                                      subSnap['density'],
#                                                      float(first_core_radius))[0]
#         first_core_mass =   float('{0:.5e}'.format(np.cumsum(first_core_count)[-1]
#                                                    * subSnap['m'][0].to('jupiter_mass').magnitude))
#
#         first_core_count   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(first_core_radius))[0]
#         first_core_mass =   float('{0:.5e}'.format(np.cumsum(first_core_count)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))
#         first_core_bin = np.digitize(first_core_radius,mean_bins_radial)-1
#         L_fc = spec_mom_sum_2[first_core_bin]
#         egrav_fc = cumsum_egrav[first_core_bin]
#         etherm_fc = cumsum_etherm[first_core_bin]
#         erot_fc = cumsum_erot[first_core_bin]
#
#     if len(peaks) >= 2:
#         first_core_radius = float('{0:.5e}'.format(x_smooth[peaks[1]]))
#         first_core_count   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(first_core_radius))[0]
#         first_core_mass =   float('{0:.5e}'.format(np.cumsum(first_core_count)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))
#
#         if bspl_y[peaks][1] < 0.5:
#             weak_fc = 1
#         #
#         second_core_radius = float('{0:.5e}'.format(x_smooth[peaks[0]]))
#         second_core_count   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(second_core_radius))[0]
#         second_core_mass =   float('{0:.5e}'.format(np.cumsum(second_core_count)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))
#
#         second_core_L_R = r_clump_centred[np.abs(spec_mom_binned_2[1]-second_core_radius).argmin()]
#         second_core_L = np.where(r_clump_centred == second_core_L_R)
#
#
#         first_core_bin = np.digitize(first_core_radius,mean_bins_radial)-1
#         second_core_bin = np.digitize(second_core_radius,mean_bins_radial)-1
#         L_fc = spec_mom_sum_2[first_core_bin] #* ((subSnap['m'][0])/(np.cumsum(first_core_count)[-1] * subSnap['m'][0]))
#         L_sc = spec_mom_sum_2[second_core_bin]#* ((subSnap['m'][0])/(np.cumsum(second_core_count)[-1] * subSnap['m'][0]))
#         egrav_fc = cumsum_egrav[first_core_bin]
#         etherm_fc = cumsum_etherm[first_core_bin]
#         erot_fc = cumsum_erot[first_core_bin]
#         egrav_sc = cumsum_egrav[second_core_bin]
#         etherm_sc = cumsum_etherm[second_core_bin]
#         erot_sc = cumsum_erot[second_core_bin]
#         axs_ang_mom.axvline(x=x_smooth[peaks[0]],c='black',linestyle='--',linewidth=1)
#         axs_ang_mom.axvline(x=x_smooth[peaks[1]],c='black',linestyle='--',linewidth=1)
#
#     clump_results.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % \
#                        (file.split("/")[-1],\
#                        clump_density,\
#                        second_core_radius,\
#                        L_sc,\
#                        egrav_sc,\
#                        etherm_sc,\
#                        erot_sc,\
#                        second_core_mass,\
#                        first_core_radius,\
#                        L_fc,\
#                        egrav_fc,\
#                        etherm_fc,\
#                        erot_fc,\
#                        first_core_mass,
#                        radius_clump,
#                        weak_fc,
#                        rhocritID))
#
# fig_radial.savefig("%s/clump_profiles.png" % cwd,dpi = 500)
# fig_ang_mom.savefig("%s/specific_angular_momentum.png" % cwd,dpi = 500)
# fig_test.savefig("%s/infall.png" % cwd,dpi = 500)
