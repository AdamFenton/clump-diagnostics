import numpy as np
from scipy import stats
import time
start_time = time.time()
# x = np.array([0.2,0.4,0.5,0.2,0.4, 6.4,5.2,4,9,2,6,3, 3.0,6,4,7,3.4,5, 1.6])
x = np.linspace(1,14,75000)

bins =  np.array([1,5,10,15])
# inds = np.digitize(x, bins)



def solution(A,bins):
    constant = 6.67E-8 * 1E-6
    array_1 = (np.true_divide(A,constant)) ** -1

    R = []
    result = []
    count = np.histogram(A,bins)
    cumulative = np.cumsum(count[0])
    bin = np.digitize(A, bins)
    LE = count[1][bin-1]
    RE = count[1][bin]



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

            total_upto_element = (n_upto_edge + (C < A[i]).sum())
            result.append(total_upto_element)


    return np.multiply(result,array_1)


    # for i in range(len(A)):
    #     if bin[i] - 1 == 0:
    #         continue
    #     else:
    #         n_upto_edge = cumulative[bin[i]-1] - count[0][bin[i]-1]
    #         LE = count[1][[bin[i]-1]]
    #         RE = count[1][[bin[i]]]
    #
    #         B = A[A>LE]
    #         R = B[B<RE]
    #
    #         total_upto_element = n_upto_edge + (R < A[i]).sum()
    #         print(total_upto_element,A[i])


    # # Find bin a value is in:
    #
    # for particle in A:
    #     bin_id = np.digitize(A, bins)-1
    #     print(cumulative[bin_id-1])


print(solution(x,bins))
print("--- %s seconds ---" % (time.time() - start_time))
