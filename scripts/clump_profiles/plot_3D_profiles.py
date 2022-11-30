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
bins = np.logspace(np.log10(5e-4),np.log10(50),100) # change the number of bins ?

if hasattr(pint, 'UnitStrippedWarning'):
    warnings.simplefilter('ignore', category=pint.UnitStrippedWarning)
np.seterr(divide='ignore', invalid='ignore')

# Define axes

figx, axx = plt.subplots(ncols=2,nrows=2,figsize=(8,8))
figy, axy = plt.subplots(ncols=2,nrows=2,figsize=(8,8))
figz, axz = plt.subplots(ncols=2,nrows=2,figsize=(8,8))

def calculate_number_in_bin(binned_quantity,mean_quantity):
    return stats.binned_statistic(binned_quantity, mean_quantity, 'count', bins=bins)
def calculate_sum(binned_quantity,summed_quantity):
    return stats.binned_statistic(binned_quantity, summed_quantity, 'sum', bins=bins)

def calculate_SPH_mean(subsnap,clump_centre,clump_velocity,bins,orientation):
    ''' Calculate the average of density, temperature, rotational velocity and
        infall velocity using adaptive binning.

        Particles are assigned a bin manually if they lie between the two bin
        edges and are within 3 times the bin width. This prevents narrow and
        elongated bins at small radii.

        Once the particles are binned, the average of the quantity is calculated
        by taking it's total in a bin and dividing it by the number of particles
        in that bin.

        ---------|..|---------              Bins closer to the centre of the
        ---------|..|---------              fragment are made smaller to make
              |.. .. ..|                    sure that only particles within
        ----------------------              the fragment are plotted. Without
            | ... ... ...|                  this, there is a large spread in
            |. .. .. .. .|                  the quantity values due to
            |... . ...  .|                  'over sampling' at small radii.
        ----------------------
        |. . .. . .. .. ..   |
        |.. . . .  ...  .. . |
        |... . .. . . .. .. .|
        |. . .. . . . .. . ..|
        |..  .   . .  . . . .|
        ----------------------
    '''

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
    # Keep track of which particles satisfy the bin conditions
    particles_ids = []
    # Recentre the arrays on the centre of the clump
    x = subsnap['x'].magnitude - clump_centre[0].magnitude
    y = subsnap['y'].magnitude - clump_centre[1].magnitude
    z = subsnap['z'].magnitude - clump_centre[2].magnitude


    # This section cleanly determines which arrays to use for the calculation
    # depending on the orientation of the tube
    if orientation == 'z':
        R = np.sqrt(x**2 + y**2) # R is used to check if the particle is within bin width
        array_to_check = z
    elif orientation == 'y':
        R = np.sqrt(x**2 + z**2)
        array_to_check = y
    else:
        R = np.sqrt(y**2 + z**2)
        array_to_check = x

    # -------------------------------- #
    # Are these calculations correct?
    # -------------------------------- #
    # Calculate the rotational velocity
    # Rotational Velocity = np.sqrt(vx**2 + vy**2)
    rotational = plonk.analysis.particles.rotational_velocity(subsnap,
                                                              clump_velocity,
                                                              ignore_accreted=True)
    # Calculate the infall velocity
    # Infall Velocity = (((x * vx )+ (y * vy) +(z * vz)) / np.sqrt(x ** 2 + y ** 2 + z ** 2)) * -1
    infall = plonk.analysis.particles.velocity_radial_spherical_altered(subsnap,
                                                                        clump_centre,
                                                                        clump_velocity,
                                                                        ignore_accreted=True)


    # Perform the bin check, if the particle is within the bin and 3 times the
    # width of the bin then include it in the average calculation
    for bin in range(0,len(bins)-1):
        # The check checks bin+1 so the loop has to stop at len(bins) - 1
        n_in_bin = 0
        for part in range(0,n_part):
            # Check if particle's position (x,y or z depending on which component
            # is being calculated) is within the two bin edges and also if it's
            # position is within 3 times the bin width
            if np.abs(array_to_check[part]) > bins[bin] \
            and np.abs(array_to_check[part]) < bins[bin+1]  \
            and R[part] < 3*(bins[bin+1] - bins[bin]): # Change this to 0.1 AU to check behav0opr
            # if all 3 checks are passed, add that particle's temperature etc to
            # the relevant array and increment the number of particles in that
            # bin by one
                temp_in_bin[bin] += subsnap['my_temp'][part].magnitude
                density_in_bin[bin] += subsnap['density'][part].magnitude
                infall_in_bin[bin] += infall[part].magnitude
                rotational_in_bin[bin] += rotational[part].magnitude
                particles_ids.append(part)
                n_in_bin += 1
        avg_temp[bin] = temp_in_bin[bin] / n_in_bin
        avg_density[bin] = density_in_bin[bin] / n_in_bin
        avg_infall[bin] = infall_in_bin[bin] / n_in_bin
        avg_rotational[bin] = rotational_in_bin[bin] / n_in_bin
        bin_counter[bin] = n_in_bin

    # Calculate the averages by dividing the total quantity by the number of
    # particles in each bin



    # Return all average arrays as well as the arrays containing only the
    # particles that passed the bin checks i.e. those stored in particle_ids
    return avg_density, avg_temp, avg_infall, avg_rotational, \
           subsnap['density'][particles_ids], subsnap['my_temp'][particles_ids], \
           rotational[particles_ids], infall[particles_ids], array_to_check[particles_ids]

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

def apply_savgol_filter(array):
    '''Apply a Savitzky - Golay filter to the array to smoothing the data'''
    return savgol_filter(array,15,3)

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
    full_clump = plonk.analysis.filters.sphere(snap=snap_active,
                                               radius = (50*au),
                                               center = clump_centre)

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

    z_comp = plonk.analysis.filters.cylinder(snap=snap_active,
                                             radius = (0.5*au),
                                             height = (50*au),
                                             center = (clump_centre)
                                             )

    return x_comp, y_comp, z_comp, clump_centre, clump_velocity, full_clump

x_comp, y_comp, z_comp, clump_centre, clump_velocity, full_clump = prepare_snapshots('run1.001.0878138.030.h5')
print('Completed snapshot preparation')
avg_density_y, avg_temp_y, avg_infall_y, avg_rotational_y, density_y, \
temperature_y, rotational_y, infall_y, y = calculate_SPH_mean(y_comp,clump_centre,
                                                             clump_velocity,bins,
                                                             orientation = 'y')
print('Completed Y component averages')


avg_density_x, avg_temp_x, avg_infall_x, avg_rotational_x, density_x, \
temperature_x, rotational_x, infall_x, x = calculate_SPH_mean(x_comp,clump_centre,
                                                             clump_velocity,bins,
                                                             orientation = 'x')

print('Completed X component averages')
avg_density_z, avg_temp_z, avg_infall_z, avg_rotational_z, density_z, \
temperature_z, rotational_z, infall_z, z = calculate_SPH_mean(z_comp,clump_centre,
                                                             clump_velocity,bins,
                                                             orientation = 'z')
print('Completed Z component averages')
figure_indexes = [(0,0),(0,1),(1,0),(1,1)]
figure_ylimits = [(1E-13,1E-2),(10,8000),(0,10),(0,10)]
figure_ylabels = ['Density $(\\rm g\,cm^{-3})$','Temperature (K)','Rotational Velocity $(\\rm km\,s^{-1})$',
                  'Infall Velocity $(\\rm km\,s^{-1})$']

for index,label,limit in zip(figure_indexes,figure_ylabels,figure_ylimits):
    axx[index].set_ylabel(label,fontsize=10)
    axx[index].set_ylim(limit)

for i in range(0,2):
    for j in range(0,2):
        axx[i,j].set_xscale('log')
        axx[i,j].set_xlim(5E-4,50)
        axx[i,j].set_xlabel('x (AU)',fontsize=10)
        axx[i,j].tick_params(axis="x", labelsize=8)
        axx[i,j].tick_params(axis="y", labelsize=8)

for index,label,limit in zip(figure_indexes,figure_ylabels,figure_ylimits):
    axy[index].set_ylabel(label,fontsize=10)
    axy[index].set_ylim(limit)

for i in range(0,2):
    for j in range(0,2):
        axy[i,j].set_xscale('log')
        axy[i,j].set_xlim(5E-4,50)
        axy[i,j].set_xlabel('y (AU)',fontsize=10)
        axy[i,j].tick_params(axis="x", labelsize=8)
        axy[i,j].tick_params(axis="y", labelsize=8)

for index,label,limit in zip(figure_indexes,figure_ylabels,figure_ylimits):
    axz[index].set_ylabel(label,fontsize=10)
    axz[index].set_ylim(limit)

for i in range(0,2):
    for j in range(0,2):
        axz[i,j].set_xscale('log')
        axz[i,j].set_xlim(5E-4,50)
        axz[i,j].set_xlabel('z (AU)',fontsize=10)
        axz[i,j].tick_params(axis="x", labelsize=8)
        axz[i,j].tick_params(axis="y", labelsize=8)


print('Plotting...')
axx[0,0].scatter(x,density_x,s=0.1,c='black')
axx[0,0].plot(bins,avg_density_x,c='red')
axx[0,0].set_yscale('log')
axx[0,1].scatter(x,temperature_x,s=0.1,c='black')
axx[0,1].plot(bins,avg_temp_x,c='red')
axx[0,1].set_yscale('log')
axx[1,0].scatter(x,rotational_x,s=0.1,c='black')
axx[1,0].plot(bins,avg_rotational_x,c='red')
axx[1,1].scatter(x,infall_x,s=0.1,c='black')
axx[1,1].plot(bins,avg_infall_x,c='red')
# for x_bin in bins:
#     ax_.scatter(x,infall_x,s=5,c='black')
#     ax_.plot(bins,avg_infall_x,c='red',alpha=0.5)
#     ax_.axvline(x=x_bin,c='black',linestyle='-',linewidth=0.5)
#     ax_.set_xscale('log')
#     ax_.set_xlim(5e-4,50)

axy[0,0].scatter(y,density_y,s=0.1,c='black')
axy[0,0].plot(bins,avg_density_y,c='red')
axy[0,0].set_yscale('log')
axy[0,1].scatter(y,temperature_y,s=0.1,c='black')
axy[0,1].plot(bins,avg_temp_y,c='red')
axy[0,1].set_yscale('log')
axy[1,0].scatter(y,rotational_y,s=0.1,c='black')
axy[1,0].plot(bins,avg_rotational_y,c='red')
axy[1,1].scatter(y,infall_y,s=0.1,c='black')
axy[1,1].plot(bins,avg_infall_y,c='red')

axz[0,0].scatter(z,density_z,s=0.1,c='black')
axz[0,0].plot(bins,avg_density_z,c='red')
axz[0,0].set_yscale('log')
axz[0,1].scatter(z,temperature_z,s=0.1,c='black')
axz[0,1].plot(bins,avg_temp_z,c='red')
axz[0,1].set_yscale('log')
axz[1,0].scatter(z,rotational_z,s=0.1,c='black')
axz[1,0].plot(bins,avg_rotational_z,c='red')
axz[1,1].scatter(z,infall_z,s=0.1,c='black')
axz[1,1].plot(bins,avg_infall_z,c='red')

figx.savefig('X_profiles.png',dpi=200)
figy.savefig('Y_profiles.png',dpi=200)
figz.savefig('Z_profiles.png',dpi=200)
