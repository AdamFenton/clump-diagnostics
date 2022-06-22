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

# Author: Adam Fenton

cwd = os.getcwd()

# Load units for use later, useful for derived quantities.
au = plonk.units('au')

density_to_load = None #
mean_bins_radial = np.logspace(np.log10(0.001),np.log10(50),120)
mpl_colour_defaults=plt.rcParams['axes.prop_cycle'].by_key()['color'] # MLP default colours

# Initalise the figure and output file which is written to later on
fig_radial, f_radial_axs = plt.subplots(nrows=3,ncols=2,figsize=(7,8))
fig_ang_mom, axs_ang_mom = plt.subplots(figsize=(7,8))

clump_results = open('clump-results.dat', 'w')

# Ignore pesky warnings when stripping unit off of pint quantity when downcasting to array
if hasattr(pint, 'UnitStrippedWarning'):
    warnings.simplefilter('ignore', category=pint.UnitStrippedWarning)
np.seterr(divide='ignore', invalid='ignore')

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
    subSnap=plonk.analysis.filters.sphere(snap=snap_active,radius = (50*au),center=clump_centre)
    subSnap_rotvel=plonk.analysis.filters.cylinder(snap=snap_active,radius = (50*au),height=(0.1*au),center=(x,y,z) * au)

    subSnap.set_units(position='au', density='g/cm^3',smoothing_length='au',velocity='km/s')

    return subSnap,snap_active,clump_centre,clump_velocity,subSnap_rotvel

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


def find_first_non_nan(array):
    for i in array:
        if math.isnan(i) == False:
            return array.index(i) - 1


for file in tqdm(complete_file_list):
    index = complete_file_list.index(file)
    line_colour = mpl_colour_defaults[index]
    prepared_snapshots = prepare_snapshots(file)
    ORIGIN = prepared_snapshots[2][0] # The position of the clump centre
    x,y,z = ORIGIN[0],ORIGIN[1],ORIGIN[2]
    vel_ORIGIN = prepared_snapshots[3][0]# The velocity of the clump centre

    subSnap = prepared_snapshots[0]
    rot_vel_snap = prepared_snapshots[4]
    r_clump_centred_midplane_rotvel = np.hypot(rot_vel_snap['x']-(x),rot_vel_snap['y']-(y))


    r_clump_centred = np.sqrt((subSnap['x']-(x))**2 +(subSnap['y']-(y))**2 + (subSnap['z']-(z))**2)
    r_clump_centred_midplane = np.hypot(subSnap['x']-(x),subSnap['y']-(y))

    radius_clump = np.sqrt((x)**2 + (y)**2 + (z)**2)


    count = calculate_number_in_bin(r_clump_centred,subSnap['m'],100)
    mass_in_bin = np.cumsum(count[0]) * subSnap['mass'][0].to('jupiter_mass')
    mid_plane_radius = plonk.analysis.particles.mid_plane_radius(subSnap,ORIGIN,ignore_accreted=True)
    rotational_velocity_radial = plonk.analysis.particles.rotational_velocity(subSnap,vel_ORIGIN,ignore_accreted=True)
    rotational_velocity_radial_cyl = plonk.analysis.particles.rotational_velocity(rot_vel_snap,vel_ORIGIN,ignore_accreted=True)
    specific_angular_momentum = plonk.analysis.particles.specific_angular_momentum(subSnap,ORIGIN,vel_ORIGIN,ignore_accreted=True).to('cm**2/s')
    total_L = np.sqrt(specific_angular_momentum[:,0]**2 + specific_angular_momentum[:,1]**2 +specific_angular_momentum[:,2]**2)



    spec_mom_binned_2 = calculate_sum(r_clump_centred,total_L,mean_bins_radial)
    spec_mom_sum_2 = np.cumsum(spec_mom_binned_2[0])
    axs_ang_mom.plot(spec_mom_binned_2[1][1:],spec_mom_sum_2,label="Total Magnitude",c=line_colour)

    # spec_mom_binned_1 = calculate_sum(r_clump_centred,specific_angular_momentum[:,2],mean_bins_radial)
    # spec_mom_sum_1= np.cumsum(spec_mom_binned_1[0])
    # axs_ang_mom.plot(spec_mom_binned_1[1][1:],spec_mom_sum_1,label="Z Componant",c=line_colour,linestyle='--')

    axs_ang_mom.set_xscale('log')
    axs_ang_mom.set_xlabel('R (AU)')
    axs_ang_mom.set_ylabel('J (cm^2/s)')
    axs_ang_mom.set_xlim(1e-4,50)
    axs_ang_mom.set_yscale('log')
    plt.legend()
    infall_velocity_radial = plonk.analysis.particles.velocity_radial_spherical_altered(subSnap,ORIGIN,vel_ORIGIN,ignore_accreted=True)

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

    averaged_rotational_velocity_interp = np.interp(np.arange(len(averaged_rotational_velocity[0])),
                                    np.arange(len(averaged_rotational_velocity[0]))[np.isnan(averaged_rotational_velocity[0]) == False],
                                    averaged_rotational_velocity[0][np.isnan(averaged_rotational_velocity[0]) == False])

    averaged_density_radial_interp = np.interp(np.arange(len(averaged_density_radial[0])),
                                    np.arange(len(averaged_density_radial[0]))[np.isnan(averaged_density_radial[0]) == False],
                                    averaged_density_radial[0][np.isnan(averaged_density_radial[0]) == False])

    averaged_temperature_radial_interp = np.interp(np.arange(len(averaged_temperature_radial[0])),
                                    np.arange(len(averaged_temperature_radial[0]))[np.isnan(averaged_temperature_radial[0]) == False],
                                    averaged_temperature_radial[0][np.isnan(averaged_temperature_radial[0]) == False])

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




    smoothed_rotational     = savgol_filter(averaged_rotational_velocity[0],15,3)
    smoothed_temperature    = savgol_filter(averaged_temperature_radial[0],15,3)
    smoothed_density        = savgol_filter(averaged_density_radial[0],15,3)
    smoothed_infall         = savgol_filter(averaged_infall_radial[0],15,3)
    smoothed_infall_nans    = savgol_filter(average_infall_with_nans ,15,3)








    y = smoothed_infall.copy()
    x = averaged_infall_radial[1][1:]

    y[np.isnan(y)] = 0


    # starting_id = np.where(smoothed_infall[np.isnan(smoothed_infall)])[0][-1] + 1
    # y = smoothed_infall[starting_id:]
    # x = x[starting_id:]

    x_smooth = np.logspace(np.log10(min(averaged_infall_radial[1])), np.log10(max(averaged_infall_radial[1])), 2000)
    bspl = splrep(x,y, s=0.1)
    bspl_y = splev(x_smooth, bspl)

    peaks, _ = find_peaks(bspl_y,height = 0.1)




    # Tidily set axes limits and scale types
    for i in range(0,3):
        for j in range(0,2):
            f_radial_axs[i,j].set_xscale('log')
            f_radial_axs[i,j].set_xlim(1E-3,50)
            f_radial_axs[i,j].set_xlabel('R (AU)')

    for i in [0,2]:
        for j in [0,1]:
            f_radial_axs[i,j].set_yscale('log')

    figure_indexes = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]
    figure_ylimits = [(1E-13,1E-1),(10,8000),(0,7),(-1,10),(0.1,40),(1E-5,10000)]
    figure_ylabels = ['Density (g/cm^3)','Temperature (K)','Rotational Velocity (km/s)',
                      'Infall Velocity (km/s)','Mass [Jupiter Masses]','Energy ratio']

    for index,label,limit in zip(figure_indexes,figure_ylabels,figure_ylimits):
        f_radial_axs[index].set_ylabel(label)
        f_radial_axs[index].set_ylim(limit)

    f_radial_axs[0,0].plot(averaged_density_radial[1][1:],averaged_density_radial[0],c = line_colour,linestyle="--",linewidth = 1)
    f_radial_axs[0,0].plot(binned_r_clump_with_nans,average_density_with_nans,c = line_colour)

    f_radial_axs[0,1].plot(averaged_temperature_radial[1][1:],averaged_temperature_radial[0],c = line_colour,linestyle="--",linewidth =1)
    f_radial_axs[0,1].plot(binned_r_clump_with_nans,average_temp_with_nans,c = line_colour)

    f_radial_axs[1,0].plot(averaged_rotational_velocity[1][1:],averaged_rotational_velocity[0],c = line_colour,linestyle="--",linewidth = 1)
    f_radial_axs[1,0].plot(binned_r_clump_with_nans,average_rotational_with_nans,c = line_colour)

    f_radial_axs[1,1].plot(averaged_infall_radial[1][1:],smoothed_infall,c = line_colour,linestyle="--",linewidth = 1)
    f_radial_axs[1,1].plot(binned_r_clump_with_nans,smoothed_infall_nans ,c = line_colour)
    # f_radial_axs[1,1].plot(x_smooth,bspl_y ,c = 'green')
    f_radial_axs[1,1].plot(x_smooth[peaks],bspl_y[peaks],'+',c='red')

    f_radial_axs[2,0].plot(count[1][1:],mass_in_bin,linewidth=1)
    f_radial_axs[2,0].set_yscale('linear')

    f_radial_axs[2,1].plot(gravitational_energy_binned[1][1:],alpha,linewidth=1,c=line_colour)
    f_radial_axs[2,1].plot(gravitational_energy_binned[1][1:],beta,linewidth=1,c=line_colour)
    f_radial_axs[2,1].axhline(y=1,c='black',linestyle='--',linewidth=1.5)
    f_radial_axs[2,1].set_xscale('log')
    f_radial_axs[2,1].set_yscale('log')
    f_radial_axs[2,1].set_ylim(1e-2,11)

    fig_radial.align_ylabels()
    fig_radial.tight_layout(pad=0.40)

    if sys.argv[1] == 'density':
        clump_density = '{0:.2e}'.format(density_to_load)
    else:
        clump_density = 'clump'

    second_core_radius = 0.0000000
    second_core_count  = 0.0000000
    second_core_mass   = 0.0000000
    L_sc               = 0.0000000
    egrav_sc           = 0.0000000
    etherm_sc           = 0.0000000
    erot_sc           = 0.0000000
    first_core_radius  = 0.0000000
    first_core_count   = 0.0000000
    L_fc               = 0.0000000
    egrav_fc           = 0.0000000
    etherm_fc           = 0.0000000
    erot_fc           = 0.0000000
    first_core_mass    = 0.0000000
    weak_fc = 0.000000
    rhocritID = 1


    # Write core information to the clump_results file for plotting later on.
    if len(peaks) == 1:
        first_core_radius = float('{0:.5e}'.format(x_smooth[peaks[0]]))
        first_core_count   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(first_core_radius))[0]
        first_core_mass =   float('{0:.5e}'.format(np.cumsum(first_core_count)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))

        first_core_count   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(first_core_radius))[0]
        first_core_mass =   float('{0:.5e}'.format(np.cumsum(first_core_count)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))
        first_core_bin = np.digitize(first_core_radius,mean_bins_radial)-1
        L_fc = spec_mom_sum_2[first_core_bin]
        egrav_fc = cumsum_egrav[first_core_bin]
        etherm_fc = cumsum_etherm[first_core_bin]
        erot_fc = cumsum_erot[first_core_bin]

    if len(peaks) >= 2:
        first_core_radius = float('{0:.5e}'.format(x_smooth[peaks[1]]))
        first_core_count   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(first_core_radius))[0]
        first_core_mass =   float('{0:.5e}'.format(np.cumsum(first_core_count)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))

        if bspl_y[peaks][1] < 0.5:
            weak_fc = 1
        #
        second_core_radius = float('{0:.5e}'.format(x_smooth[peaks[0]]))
        second_core_count   = calculate_number_in_bin(r_clump_centred,subSnap['density'],float(second_core_radius))[0]
        second_core_mass =   float('{0:.5e}'.format(np.cumsum(second_core_count)[-1] * subSnap['m'][0].to('jupiter_mass').magnitude))

        second_core_L_R = r_clump_centred[np.abs(spec_mom_binned_2[1]-second_core_radius).argmin()]
        second_core_L = np.where(r_clump_centred == second_core_L_R)


        first_core_bin = np.digitize(first_core_radius,mean_bins_radial)-1
        second_core_bin = np.digitize(second_core_radius,mean_bins_radial)-1
        L_fc = spec_mom_sum_2[first_core_bin]
        L_sc = spec_mom_sum_2[second_core_bin]
        egrav_fc = cumsum_egrav[first_core_bin]
        etherm_fc = cumsum_etherm[first_core_bin]
        erot_fc = cumsum_erot[first_core_bin]
        egrav_sc = cumsum_egrav[second_core_bin]
        etherm_sc = cumsum_etherm[second_core_bin]
        erot_sc = cumsum_erot[second_core_bin]


    clump_results.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % \
                       (file.split("/")[-1],\
                       clump_density,\
                       second_core_radius,\
                       L_sc,\
                       egrav_sc,\
                       etherm_sc,\
                       erot_sc,\
                       second_core_mass,\
                       first_core_radius,\
                       L_fc,\
                       egrav_fc,\
                       etherm_fc,\
                       erot_fc,\
                       first_core_mass,
                       radius_clump,
                       weak_fc,
                       rhocritID))

fig_radial.savefig("%s/clump_profiles.png" % cwd,dpi = 500)
fig_ang_mom.savefig("%s/specific_angular_momentum.png" % cwd,dpi = 500)
