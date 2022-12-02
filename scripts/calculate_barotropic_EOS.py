# ---------------------------------------- #
# calculate_barotropic_EOS.py
# ---------------------------------------- #
# From the CLI, read in values for rhocrit1
# gamma1 and gamma2 then return the values
# for rhocrit2 (where T = 100 K) and rhocrit3
# (where T = 2000 K)
# ---------------------------------------- #
# Author: Adam Fenton
# Date: 20220718
# ---------------------------------------- #



import numpy as np
import sys
kb = 1.38E-16
mu = 2.381
mh = 1.67E-24
unit_vel = 2.978E6
R_ref = 300
q = 0.25
m_star = 0.8

def calculate_H_on_R(T_1AU):
    Tref = T_1AU*R_ref**(-2*q)
    cs0 = (np.sqrt((Tref * kb)/(mu*mh)) / unit_vel) / R_ref **-q
    H_on_R = (cs0 / R_ref **q) / np.sqrt((1 * m_star)/(R_ref))
    return H_on_R

def calculate_eos_params(rhocrit1,gamma1,gamma2):
    first_turnoff_temp = 100 # Kelvin
    second_turnoff_temp = 2000 # Kelvin
    T_iso = 10 # Kelvin
    cs2 = ((kb*first_turnoff_temp)/(mu * mh))
    cs02 = ((kb*T_iso)/(mu * mh))
    rhocrit2 = ((cs2/cs02) ** ((gamma1 - 1)**-1)) * rhocrit1
    cs2 = ((kb*second_turnoff_temp)/(mu * mh))
    rhocrit3 = ((cs2/cs02)/((rhocrit2)/(rhocrit1))**(gamma1 - 1)) ** ((gamma2 - 1)**-1) * rhocrit2
    return rhocrit1,rhocrit2, rhocrit3

# calculate_eos_rhocrit(float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]))
print(calculate_H_on_R(200))
# print(calculate_eos_params(1e-13,1.667,1.4))
