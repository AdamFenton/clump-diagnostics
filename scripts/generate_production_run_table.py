import itertools
import numpy as np
from calculate_barotropic_EOS import calculate_eos_rhocrit
rho_c1 = [1e-13,3e-13,6e-13]
T_1AU = [200,150]
gamma1 = [1.4,1.66,1.8,1.2]

run_params = open('run_parameters.dat', 'w')
combinations = np.array(list(itertools.product(rho_c1,gamma1,T_1AU)))

for i in combinations:
    rc1 = calculate_eos_rhocrit(i[0],i[1],i[1])[0]
    rc2 = calculate_eos_rhocrit(i[0],i[1],i[1])[1]
    rc3 = calculate_eos_rhocrit(i[0],i[1],i[1])[2]
    g1 = g2 = i[1]
    g3 = 1.10
    T_1 = i[2]
    T_inf = 10.0
    run_params.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % \
                       (rc1,\
                        rc2,\
                        rc3,\
                        g1, \
                        g2, \
                        g3, \
                        T_1,\
                        T_inf))
