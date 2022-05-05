import numpy as np
import matplotlib.pyplot as plt
import random
import time
start_time = time.time()
# A = [16,12,5,10]
# B = [2, 3, 1, 4]

    # U = 3/2 * 1.38E-16 * subSnap['my_temp'] * ((subSnap['m'][0].to('g'))/(1.67E-24))
    # U = U.magnitude
    # U *= ((U>2000)*(1/1.2)+(U<2000)*(1/2.381))
    # return U

def solution_temp(T,R,particle_mass):

    thermal_energy = np.zeros(len(T))
    perm = np.asarray(T).argsort() # Permuation of array that sorts T array
    T_sorted = np.sort(T)
    index = np.argmax(T_sorted >= 2000) # Index of array where element is greater than threshold
    thermal_energy[:index] = 3/2 * 1.38E-16 * T_sorted[:index] * ((particle_mass)/(1.67E-24 * 2.381)) # Calculate thermal energy in first regime
    thermal_energy[index:] = 3/2 * 1.38E-16 * T_sorted[index:] * ((particle_mass)/(1.67E-24 * 1.2)) # Calculate thermal energy in second regime


    # We need to create an array for the radii that has the same permutation of
    # the sorted temperature array so that the energies are correct at the correct R

    radii = np.asarray(R)[perm]

    result = zip(radii,thermal_energy)
    return list(result)
