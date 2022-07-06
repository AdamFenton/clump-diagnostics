# ================================== #
# plot_core_data.py
# ================================== #
# Read data from file created by clump
# profiles script. Plot the first and
# second hydrostatic core information

# Author = Adam Fenton
# Date = 20220706
# =============================== #


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option("display.max_rows", None, "display.max_columns", None)
fig_one,axs = plt.subplots(nrows=3,ncols=2,figsize=(8,8))
fig_two, axs_energies = plt.subplots(nrows=3,ncols=2,figsize=(8,8))

cwd = os.getcwd()
raw_data = pd.read_csv("%s/clump-results.dat" % cwd,sep='\t',
                       names=['file','density','Rsc','Lsc','egrav_sc',"etherm_sc",
                       "erot_sc",'Msc','Rfc','Lfc','egrav_fc',"etherm_fc","erot_fc"
                       ,'Mfc','rad','weak_fc','rhocritID'])



raw_data = raw_data.iloc[1:]
numeric_columns = raw_data.columns.drop('file')
raw_data[numeric_columns] = raw_data[numeric_columns].apply(pd.to_numeric)
dropped = raw_data.drop(raw_data[raw_data.Rsc == 0].index)



tmp = raw_data.loc[raw_data['Rsc'] == 0]
tmp ['Rsc'] = tmp['Rfc']
tmp ['Msc'] = tmp['Mfc']
tmp ['Lsc'] = tmp['Lfc']
tmp ['Rfc'] = 0.000
tmp ['Mfc'] = 0.000
tmp ['Lfc'] = 0.000

filtered = pd.concat([tmp, dropped], ignore_index=True, sort=False)
# filtered = filtered.drop(filtered[filtered.Rfc == 0].index)

filtered['Rsc'] = filtered['Rsc']  * 2092.51
df1 = filtered.loc[filtered['rhocritID'] == 1]
df2 = filtered.loc[filtered['rhocritID'] == 2]
df3 = filtered.loc[filtered['rhocritID'] == 3]
df4 = filtered.loc[filtered['rhocritID'] == 4]
df5 = filtered.loc[filtered['rhocritID'] == 5]
df6 = filtered.loc[filtered['rhocritID'] == 6]

# df1_avgs, df1_stds = df1[['Rsc','Rfc','Msc','Mfc','rad']].mean(), df1[['Rsc','Rfc','Msc','Mfc','rad']].std()
# df2_avgs, df2_stds = df2[['Rsc','Rfc','Msc','Mfc','rad']].mean(), df2[['Rsc','Rfc','Msc','Mfc','rad']].std()
# df3_avgs, df3_stds = df3[['Rsc','Rfc','Msc','Mfc','rad']].mean(), df3[['Rsc','Rfc','Msc','Mfc','rad']].std()
# df4_avgs, df4_stds = df4[['Rsc','Rfc','Msc','Mfc','rad']].mean(), df4[['Rsc','Rfc','Msc','Mfc','rad']].std()
# df5_avgs, df5_stds = df5[['Rsc','Rfc','Msc','Mfc','rad']].mean(), df5[['Rsc','Rfc','Msc','Mfc','rad']].std()
# df6_avgs, df6_stds = df6[['Rsc','Rfc','Msc','Mfc','rad']].mean(), df6[['Rsc','Rfc','Msc','Mfc','rad']].std()
#
# print(df1_avgs, df1_stds )

df1.name = "Run 1"
df2.name = "Run 2"
df3.name = "Run 3"
# df4.name = "Run 4"
# df5.name = "Run 5"
# df6.name = "Run 6"
# df4e_13.name = "4E-13"
# df1e_12.name = "5E-13"

dfs = [df1,df2,df3]
markers = ['.','^','s']

for df,marker in zip(dfs,markers):
    axs[0,0].scatter(df['rad'],df['Mfc'],s=12,label='%s' % df.name,marker=marker)
    # axs[0,0].scatter(tmp['rad'],tmp['Mfc'],s=12,label='%s' % df.name,marker="o",alpha=0.5)
    axs[0,1].scatter(df['rad'],df['Rfc'],s=12,label='%s' % df.name,marker=marker)
    # axs[0,1].scatter(tmp['rad'],tmp['Rfc'],s=12,label='%s' % df.name,marker="o",alpha=0.5)
    axs[1,0].scatter(df['rad'],df['Msc'],s=12,label='%s' % df.name,marker=marker)
    # axs[1,0].scatter(tmp['rad'],tmp['Msc'],s=12,label='%s' % df.name,marker="o",alpha=0.5)
    axs[1,1].scatter(df['rad'],df['Rsc'],s=12,label='%s' % df.name,marker=marker)

    # axs[1,1].scatter(tmp['rad'],tmp['Rsc'],s=12,label='%s' % df.name,marker="o",alpha=0.5)
    axs[2,0].scatter(df['rad'],df['Lsc'],s=8,label='%s' % df.name,marker=marker)
    # axs[2,0].scatter(tmp['rad'],tmp['Lsc'],s=12,label='%s' % df.name,marker="o",alpha=0.5)
    axs[2,1].scatter(df['rad'],df['Lfc'],s=8,label='%s' % df.name,marker=marker)
    # axs[2,1].scatter(tmp['rad'],tmp['Lfc'],s=12,label='%s' % df.name,marker="o",alpha=0.5)
    axs_energies[0,0].scatter(df['rad'],df['egrav_fc'],s=8,label='%s' % df.name,marker=marker)
    # axs_energies[0,0].scatter(tmp['rad'],tmp['Mfc'],s=12,label='%s' % df.name,marker="o",alpha=0.5)
    axs_energies[0,1].scatter(df['rad'],df['egrav_sc'],s=8,label='%s' % df.name,marker=marker)
    # axs_energies[0,1].scatter(tmp['rad'],tmp['Rfc'],s=12,label='%s' % df.name,marker="o",alpha=0.5)
    axs_energies[1,0].scatter(df['rad'],df['etherm_fc'],s=8,label='%s' % df.name,marker=marker)
    # axs_energies[1,0].scatter(tmp['rad'],tmp['Msc'],s=12,label='%s' % df.name,marker="o",alpha=0.5)
    axs_energies[1,1].scatter(df['rad'],df['etherm_sc'],s=8,label='%s' % df.name,marker=marker)
    # axs_energies[1,1].scatter(tmp['rad'],tmp['Rsc'],s=12,label='%s' % df.name,marker="o",alpha=0.5)
    axs_energies[2,0].scatter(df['rad'],df['erot_fc'],s=8,label='%s' % df.name,marker=marker)
    # axs_energies[2,0].scatter(tmp['rad'],tmp['Lsc'],s=12,label='%s' % df.name,marker="o",alpha=0.5)
    axs_energies[2,1].scatter(df['rad'],df['erot_sc'],s=8,label='%s' % df.name,marker=marker)
    # axs_energies[2,1].scatter(tmp['rad'],tmp['Lfc'],s=12,label='%s' % df.name,marker="o",alpha=0.5)





axs[0,0].set_ylabel("First Core Mass (Mj)",fontsize=12)
axs[0,1].set_ylabel("First Core Radius (AU)",fontsize=12)
axs[1,0].set_ylabel("Second Core Mass (Mj)",fontsize=12)
axs[1,1].set_ylabel("Second Core Radius (Rj)",fontsize=12)
axs[1,0].set_ylim(1.25,3)
axs[1,1].set_ylim(30,50)
axs[2,0].set_ylabel("Second Core J (cm^2/s)",fontsize=12)
axs[2,1].set_ylabel("First Core J (cm^2/s)",fontsize=12)
axs[2,0].set_yscale('log')
axs[2,1].set_yscale('log')
axs[2,1].set_ylim(bottom=4E20)
axs[0,0].legend()
# Shrink current axis by 20%
axs_energies[0,0].set_ylabel("First Core egrav",fontsize=12)
axs_energies[0,1].set_ylabel("Second Core egrav",fontsize=12)
axs_energies[1,0].set_ylabel("First Core etherm",fontsize=12)
axs_energies[1,1].set_ylabel("Second Core etherm",fontsize=12)
axs_energies[2,0].set_ylabel("First Core erot",fontsize=12)
axs_energies[2,1].set_ylabel("Second Core erot",fontsize=12)
axs[0,0].set_ylim(top=60)
axs[0,1].set_ylim(top=40)
axs[1,0].set_ylim(top=2.5)
axs[1,1].set_ylim(top=50)
axs[2,0].set_ylim(top=5E20)
axs[2,1].set_ylim(top=5E23)

for i in range(0,3):
    for j in range(0,2):
        axs[i,j].set_xlabel('R (AU)',fontsize=12)
        axs[i,j].set_xlim(0,500)
        axs[i,j].tick_params(axis="x", labelsize=12)
        axs[i,j].tick_params(axis="y", labelsize=12)
        axs_energies[i,j].set_xlabel('R (AU)')
        axs_energies[i,j].set_xlim(0,500)




fig_one.align_ylabels()
fig_one.tight_layout(pad=0.35)

fig_two.align_ylabels()
fig_two.tight_layout(pad=0.35)

fig_one.savefig("%s/clump-core-info.png" % cwd, dpi = 500)
fig_two.savefig("%s/clump-core-info_energies.png" % cwd, dpi = 500)
