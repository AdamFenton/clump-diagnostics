# ---------------------------------------- #
# plot_clump_accretion.py
# ---------------------------------------- #
# Read CSV file containing clump accretion
# rates at 4 different radii and plot them
# ---------------------------------------- #
# Author: Adam Fenton
# Date: 20220718
# ---------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(7,8))

complete_file_list = glob.glob('**/clump_accretion_rates.csv', recursive=True)

for file in complete_file_list:
    df = pd.read_csv(file)
    df = df.iloc[: , 1:]
    ax[0,0].plot(df['Density'],df['Mdot (1AU)'],'o-')
    ax[0,0].set_ylabel(r'$\dot{M}$ at 1 AU ')
    ax[0,1].plot(df['Density'],df['Mdot (2AU)'],'o-')
    ax[0,1].set_ylabel(r'$\dot{M}$ at 2 AU ')
    ax[1,0].plot(df['Density'],df['Mdot (5AU)'],'o-')
    ax[1,0].set_ylabel(r'$\dot{M}$ at 5 AU ')
    ax[1,1].plot(df['Density'],df['Mdot (10AU)'],'o-')
    ax[1,1].set_ylabel(r'$\dot{M}$ at 10 AU ')

    for i in range(0,2):
        for j in range(0,2):
            ax[i,j].set_xscale('log')
fig.align_ylabels()
fig.tight_layout(pad=0.40)
plt.show()
