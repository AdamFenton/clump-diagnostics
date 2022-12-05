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
# Define figure
figx, axx = plt.subplots(ncols=2,nrows=2,figsize=(9,9))
figy, axy = plt.subplots(ncols=2,nrows=2,figsize=(9,9))
figz, axz = plt.subplots(ncols=2,nrows=2,figsize=(9,9))
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

def calculate_SPH_mean_x(subsnap,clump_centre,clump_velocity,bins):
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
            and R[part] < 10*(bins[bin+1] - bins[bin]) :
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

    return avg_infall,avg_rotational,avg_temp,avg_density, infall[ids], \
           rotational[ids],subsnap['my_temp'][ids],subsnap['density'][ids], \
           x[ids]

def calculate_SPH_mean_y(subsnap,clump_centre,clump_velocity,bins):
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
            and R[part] < 10*(bins[bin+1] - bins[bin]) :
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

    return avg_infall,avg_rotational,avg_temp,avg_density, infall[ids], \
           rotational[ids],subsnap['my_temp'][ids],subsnap['density'][ids], \
           y[ids]

def calculate_SPH_mean_z(subsnap,clump_centre,clump_velocity,bins):
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

    for part in range(n_part):
        for bin in range(len(bins)-1):
            if z[part] < bins[bin+1] and z[part] > bins[bin] \
            and R[part] < 10*(bins[bin+1] - bins[bin]) :
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

    return avg_infall,avg_rotational,avg_temp,avg_density, infall[ids], \
           rotational[ids],subsnap['my_temp'][ids],subsnap['density'][ids], \
           z[ids]

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

x_comp,y_comp,z_comp,clump_centre,clump_velocity = prepare_snapshots('run1.001.0878138.030.h5')

print('Completed snapshot preparation')




avg_infall_x,avg_rotational_x,avg_temp_x,avg_density_x, \
infall_x,rotational_x,temperature_x,density_x, x = calculate_SPH_mean_x(x_comp,
                                                                        clump_centre,
                                                                        clump_velocity,bins)

avg_infall_y,avg_rotational_y,avg_temp_y,avg_density_y, \
infall_y,rotational_y,temperature_y,density_y, y = calculate_SPH_mean_y(y_comp,
                                                                        clump_centre,
                                                                        clump_velocity,bins)
avg_infall_z,avg_rotational_z,avg_temp_z,avg_density_z, \
infall_z,rotational_z,temperature_z,density_z, z = calculate_SPH_mean_z(z_comp,
                                                                        clump_centre,
                                                                        clump_velocity,bins)




for i in range(2):
    for j in range(2):
        axx[i,j].set_xscale('log')
        axx[i,j].set_xlabel('x (AU)')
for i in range(2):
    for j in range(2):
        axy[i,j].set_xscale('log')
        axy[i,j].set_xlabel('y (AU)')
for i in range(2):
    for j in range(2):
        axz[i,j].set_xscale('log')
        axz[i,j].set_xlabel('z (AU)')


axx[0,0].scatter(x,density_x,s=0.1)
axx[0,0].plot(bins,avg_density_x,c='red')
axx[0,0].set_yscale('log')
axx[0,1].scatter(x,temperature_x,s=0.1)
axx[0,1].plot(bins,avg_temp_x,c='red')
axx[0,1].set_yscale('log')
axx[1,0].scatter(x,rotational_x,s=0.1)
axx[1,0].plot(bins,avg_rotational_x,c='red')
axx[1,1].scatter(x,infall_x,s=0.1)
axx[1,1].plot(bins,avg_infall_x,c='red')

axy[0,0].scatter(y,density_y,s=0.1)
axy[0,0].plot(bins,avg_density_y,c='red')
axy[0,0].set_yscale('log')
axy[0,1].scatter(y,temperature_y,s=0.1)
axy[0,1].plot(bins,avg_temp_y,c='red')
axy[0,1].set_yscale('log')
axy[1,0].scatter(y,rotational_y,s=0.1)
axy[1,0].plot(bins,avg_rotational_y,c='red')
axy[1,1].scatter(y,infall_y,s=0.1)
axy[1,1].plot(bins,avg_infall_y,c='red')

axz[0,0].scatter(z,density_z,s=0.1)
axz[0,0].plot(bins,avg_density_z,c='red')
axz[0,0].set_yscale('log')
axz[0,1].scatter(z,temperature_z,s=0.1)
axz[0,1].plot(bins,avg_temp_z,c='red')
axz[0,1].set_yscale('log')
axz[1,0].scatter(z,rotational_z,s=0.1)
axz[1,0].plot(bins,avg_rotational_z,c='red')
axz[1,1].scatter(z,infall_z,s=0.1)
axz[1,1].plot(bins,avg_infall_z,c='red')

axx.savefig('x_profiles.png',dpi=200)
axy.savefig('y_profiles.png',dpi=200)
axz.savefig('z_profiles.png',dpi=200)
