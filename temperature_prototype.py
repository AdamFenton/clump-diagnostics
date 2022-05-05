import numpy as np
import matplotlib.pyplot as plt
import random
A = [16,12,5,10]
B = [2, 3, 1, 4]


def solution(T,R):
    thermal_energy = np.zeros(len(T))
    perm = np.asarray(T).argsort() # Permuation of array that sorts T array
    T_sorted = np.sort(T)
    index = np.argmax(T_sorted > 10) # Index of array where element is greater than threshold
    thermal_energy[:index] = 1 # Calculate thermal energy in first regime
    thermal_energy[index:] = 10 # Calculate thermal energy in second regime

    # We need to create an array for the radii that has the same permutation of
    # the sorted temperature array so that the energies are correct at the correct R

    radii = np.asarray(R)[perm]

    result = zip(radii,thermal_energy)
    return list(result)



print(solution(A,B))
