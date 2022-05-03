import numpy as np
from scipy import stats

# x = np.array([0.2,0.4,0.5,0.2,0.4, 6.4,5.2,4,9,2,6,3, 3.0,6,4,7,3.4,5, 1.6])
x = np.array([2,3,6,7,12])

bins =  np.array([1,5,10,15])
# inds = np.digitize(x, bins)



def solution(A,bins):

    count = np.histogram(A,bins)
    cumulative = np.cumsum(count[0])
    print(cumulative)
    # Find bin a value is in:

    for particle in A:
        bin_id = np.digitize(A, bins)-1
        print(cumulative[bin_id-1])



    # # A value's ind is going to be the bin array index + 1. For example, the value 1.2 has an inds of 2 and lies in the 1st bin (not zeroth)
    # inds  = np.digitize(x, bins)
    # for i in range(len(A)):
    #     bin_in = inds[i] - 1
    #     print(bin_in)

solution(x,bins)
