# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:27:48 2022

@author: vjohn
"""

#%% path
import sys
import os

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
sys.path.append(script_dir)
save_path = os.path.join(script_dir, 'Figures')

#%% imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pickle
from utils.notebook_tools import get_data_from, get_mw_prop, plot_sequence

#%% path
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

save_path = os.path.join(script_dir, 'Figures')
fig_size_single = 3.37
fig_size_double = 6.69

#%% defining style
plt.rcParams.update({'font.size': 8})
plt.rcParams['font.sans-serif']= 'Arial'
plt.rcParams["figure.figsize"] = (fig_size_single, 3)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.default'] = 'it' #'regular'

plt.rcParams['legend.frameon']= False
plt.rcParams['legend.fontsize']= 'small'
plt.rcParams['legend.scatterpoints']= 1
plt.rcParams['axes.labelpad'] = 4 #-2

#%% Load data

start_time_fq1 = '2022-07-06\\11-43-01'
start_time_fq2 = '2022-07-12\\11-40-36'

end_time_fq1 = start_time_fq1 #'2021-06-24\\18-03-01' #'2021-06-10\\18-32-47'
datadir = os.path.join(script_dir, 'measurements')
datfiles_fq1, fnames_fq1 = get_data_from(start_time_fq1, end_time_fq1, num = 1, rootfolder=datadir, only_complete = False) 
datfile_fq1 = datfiles_fq1[0]

end_time_fq2 = start_time_fq2 #'2021-06-24\\18-03-01' #'2021-06-10\\18-32-47'
datadir = os.path.join(script_dir, 'measurements')
datfiles_fq2, fnames_fq2 = get_data_from(start_time_fq2, end_time_fq2, num = 1, rootfolder=datadir, only_complete = False) 
datfile_fq2 = datfiles_fq2[0]

#%% Calibrated Rabi frequencies
with open(os.path.join(script_dir, 'measurements\config_freq_rabi.txt'), "rb") as file:
    config_freq_rabi = pickle.load(file)

vP1 = -10
vP2 = 10
P2_pwr = -5
P4_pwr = 3

fq2p2 = config_freq_rabi['P2'][P2_pwr]['fq2'][(vP1,vP2)]
fq2 = fq2p2 #/1e9
fq2_p2 = config_freq_rabi['P2'][P2_pwr]['fq2_'][(vP1,vP2)]
fq2_ = fq2_p2 #/1e9
fq1p4 = config_freq_rabi['P4'][P4_pwr]['fq1'][(vP1,vP2)]
fq1 = fq1p4 #/1e9


#%% PLotting

fig, axs = plt.subplots(1,2, sharey=True)

vP2s1 = datfile_fq1.vPdetuning_set
vP1s1 = -1 * np.array(vP2s1) 

vP2s2 = datfile_fq2.vpB12_set
vP1s2 = -1 * np.array(vP2s2) 

detuning1 = (np.array(datfile_fq1.vPdetuning_set)**2 * 2)**0.5
detuning2 = (np.array(datfile_fq2.vpB12_set)**2 * 2)**0.5

norm = mpl.colors.Normalize(vmin=0.1, vmax=0.9)

axs[0].pcolor(datfile_fq1.frequency_set[0],
              vP1s1,
              datfile_fq1.su0,
              shading='auto', cmap='hot', norm=norm)

# axs[0].pcolor(datfile_fq1.frequency_set[0],
#               - detuning1[round(len(detuning1)/2):],
#               datfile_fq1.su0[round(len(detuning1)/2):],
#               shading='auto', cmap='hot', norm=norm)

axs[1].pcolor(datfile_fq2.frequency_set[0],
              vP1s2,
              datfile_fq2.su0[::-1], # inverse order
              shading='auto', cmap='hot', norm=norm)

c = axs[1].pcolor(datfile_fq2.frequency_set[0],
              - detuning2[round(len(detuning2)/2):],
              datfile_fq2.su0[::-1][round(len(detuning2)/2):], # inverse order
              shading='auto', cmap='hot', norm=norm)

axs[0].scatter(fq1, -10, marker='*', color='green', s=50, label='operation point')
axs[1].scatter(fq2, -10, marker='*', color='green', s=50)

axs[0].set_ylabel(r'$\epsilon$ [mV]')
axs[0].set_xlabel(r'$f_{P4}$ [GHz]')
axs[1].set_xlabel(r'$f_{P2}$ [GHz]')

axs[1].set_ylim(-17,17)

divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(c, cax=cax)
cbar.ax.set_ylabel(r'$1-P_{\downdownarrows}$', rotation=-90, labelpad=15)
cbar.ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])

plt.tight_layout()
axs[0].legend(loc=2, frameon=True)
plt.savefig(save_path+'\\figureS2_res_freq.png', dpi=300)

plt.show()

#%%

fig, axs = plt.subplots(2,1)

det_point = (14,-14)
plot_sequence(datfile_fq1, ['vP1', 'vP2'], ['MW_p4', 'MW_p2'], ax=axs[0], xlim=(3.4e4, 3.7e4), legend=True)
axs[0].hlines(det_point[0], 3.4e4, 3.7e4, ls='--', lw=1, color='black', label=str(det_point))
axs[0].hlines(det_point[1], 3.4e4, 3.7e4, ls='--', lw=1, color='black')
# axs[0].legend()

det_point = (12,-12)
plot_sequence(datfile_fq2, ['vP1', 'vP2'], ['MW_p4', 'MW_p2'], ax=axs[1], xlim=(3.4e4, 3.7e4), legend=True)
axs[1].hlines(det_point[0], 3.4e4, 3.7e4, ls='--', lw=1, color='black', label=str(det_point))
axs[1].hlines(det_point[1], 3.4e4, 3.7e4, ls='--', lw=1, color='black')
# axs[1].legend()

plt.tight_layout()
plt.show()


mw_prop = get_mw_prop(datfile_fq1, ['p2', 'p4'])
f_drve = mw_prop['p4']['rf'] - mw_prop['p2']['rf']