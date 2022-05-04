import numpy as np
import matplotlib.pyplot as plt
import random
A = [16,12,10,5]
B = [2, 3, 1, 4]


def solution(T,R):
    zipped_arrays = zip(T,R) # sorting a zipped object by the first list.
    sorted_pairs  = sorted(zipped_arrays)
    print(np.argmax(np.asarray(T.sort())>2))

    energy_below = T[:threshold] * 1
    energy_above = T[threshold:] * 10

    x,y = zip(*sorted_pairs)
    plt.plot(x,y)
    plt.show()
    # tuples = zip(*sorted_pairs)
    #
    # for tuple in tuples:
    #     print(tuple)



solution(A,B)
