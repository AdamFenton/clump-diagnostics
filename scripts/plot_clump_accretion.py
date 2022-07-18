# ---------------------------------------- #
# plot_clump_accretion.py
# ---------------------------------------- #
# Read CSV file containing clump accretion
# rates at 4 different radii and plot them
# ---------------------------------------- #
# CSV file column names:
# 'Density','Time (years)',
# '1 AU Mass (Mj)',
# 'Delta M (1AU)',
# 'Mdot (1AU)',
# '2 AU Mass (Mj)',
# 'Delta M (2AU)',
# 'Mdot (2AU)',
# '5 AU Mass (Mj)',
# 'Delta M (5AU)',
# 'Mdot (5AU)',
# '10 AU Mass (Mj)',
# 'Delta M (10AU)',
# 'Mdot (10AU)',
# 'Delta Time (years)'
# ---------------------------------------- #
# Author: Adam Fenton
# Date: 20220718
# ---------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
cwd = os.getcwd()
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(7,8))
fig2, ax2 = plt.subplots(nrows=2,ncols=2,figsize=(7,8))

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
    ax2[0,0].plot(df['Density'],df['1 AU Mass (Mj)'],'o-')
    ax2[0,0].set_ylabel('M at 1 AU ')
    ax2[0,1].plot(df['Density'],df['2 AU Mass (Mj)'],'o-')
    ax2[0,1].set_ylabel('M at 2 AU ')
    ax2[1,0].plot(df['Density'],df['5 AU Mass (Mj)'],'o-')
    ax2[1,0].set_ylabel('M at 5 AU ')
    ax2[1,1].plot(df['Density'],df['10 AU Mass (Mj)'],'o-')
    ax2[1,1].set_ylabel('M at 10 AU ')


    for i in range(0,2):
        for j in range(0,2):
            ax[i,j].set_xscale('log')
            ax[i,j].set_xlabel('Density (g/cm^3)')
            ax2[i,j].set_xscale('log')
            ax2[i,j].set_xlabel('Density (g/cm^3)')

fig.align_ylabels()
fig.tight_layout(pad=0.40)
fig2.align_ylabels()
fig2.tight_layout(pad=0.40)

fig.suptitle('Μass accretion rates for clumps with no first cores')
fig.subplots_adjust(top=0.95)

fig.savefig("%s/clump_accretion_rates.png" % cwd,dpi = 500)
fig2.savefig("%s/clump_mass_inside.png" % cwd,dpi = 500)
