##############################
# plot_formation_distance.py #
# Plots the distance at which
# each clump formed (at e-9)
# on a histogram
##############################
# Author: Adam Fenton
# Date 20221024
##############################

import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import os

data = []
bins = np.linspace(65,200,25)


complete_file_list = glob.glob('./clump0*/0*.dat')
complete_file_list = sorted(complete_file_list, key = lambda x: x.split('/')[1])


def collect_data(file):
    open_file = open(file,'r')
    data.append(open_file.readlines()[0])


if len(complete_file_list) != 0:
    print('Plotting histogram of formation distance...')
    for file in complete_file_list:
        collect_data(file)
    df = pd.DataFrame([sub.split(",") for sub in data])
    df = df.astype(float)
    df.columns = ['dens','x','y','z','vx','vy','vz']

    df['R'] = np.sqrt(df['x']**2 + df['y']**2+df['z']**2)
    plt.hist(df['R'], bins=bins)
    plt.savefig('./radial_distribution.pdf')

else:
    print('No clumps in file...skipping')
