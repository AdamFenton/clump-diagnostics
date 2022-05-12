import numpy as np
from scipy import stats



def calculate_gravitational_energy(A,bins,particle_mass):
    ''' Our philosophy here is to reduce execution time for gravitational energy
        calculation where the number of particle interior to each particle needs
        to be considered. Doing this with a loop through every particle is expensive
        so instead we construct a histogram and then, for every particle, find the
        number of particles up to the last bin edge (np.cumsum). This process is only
        done once outside the for loop. Then we count the number of particles with radii
        less than that of the particle under inspection but only down to the last bin edge.
    '''

    R = []
    result = []
    count = np.histogram(A,bins)
    cumulative = np.cumsum(count[0])
    bin = np.digitize(A, bins) # Which bin to elements fall into

    LE = count[1][bin-1]       # Create arrays for left and right edges for each particle. Doing this outside the loop cuts down comp time
    RE = count[1][bin]
    A = np.multiply(A,1.496E13)

    constant = 6.67E-8 * particle_mass # Define the constant, for gravitational energy this is G * particle mass

    array_1 = (np.true_divide(A,constant)) ** -1 # Do array division for the 'radii' array and then raise to power of -1 to flip



    for i in range(len(cumulative)+1):
        if i == 0:
            R.append(0)
        else:
            R.append(cumulative[i-1])


    for i in range(len(A)):
        LE_i = LE[i]
        RE_i = RE[i]

        B = A[A>LE_i]
        C = B[B<RE_i]
        if bin[i] - 1 == 0:
            result.append((C < A[i]).sum())
        else:
            n_upto_edge = R[bin[i]-1]

            total_upto_element = (n_upto_edge + (C < A[i]).sum()) * particle_mass
            result.append(total_upto_element)


    return np.multiply(result,array_1) # Return the resulting array (interior masses) multiplied by the array_1 we defined earlier to give E_grav
