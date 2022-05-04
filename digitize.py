import numpy as np
from scipy import stats

# x = np.array([0.2,0.4,0.5,0.2,0.4, 6.4,5.2,4,9,2,6,3, 3.0,6,4,7,3.4,5, 1.6])
x = np.array([2,3,6,7,11,11.5,12,12.2,12.6,12.7,12.9,12.7,13])

bins =  np.array([1,5,10,15])
# inds = np.digitize(x, bins)



def solution(A,bins):
    R = []
    count = np.histogram(A,bins)
    cumulative = np.cumsum(count[0])
    bin = np.digitize(A, bins)

    for i in range(len(A)):
        if bin[i] - 1 == 0:
            continue
        else:
            n_upto_edge = cumulative[bin[i]-1] - count[0][bin[i]-1]
            LE = count[1][[bin[i]-1]]
            RE = count[1][[bin[i]]]

            B = A[A>LE]
            R = B[B<RE]

            total_upto_element = n_upto_edge + (R < A[i]).sum()
            print(total_upto_element,A[i])


    # # Find bin a value is in:
    #
    # for particle in A:
    #     bin_id = np.digitize(A, bins)-1
    #     print(cumulative[bin_id-1])


solution(x,bins)
