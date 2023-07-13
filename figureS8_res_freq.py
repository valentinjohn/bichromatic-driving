# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:27:48 2022

@author: vjohn
"""

# %% Imports
from utils.settings import *
from utils.delft_tools import *

# %% Save path
save_path = get_save_path('FigureS6')
plt.rcParams['axes.labelpad'] = 4  # -2

# %% Load data

start_time_fq1 = '2022-07-06\\11-43-01'
start_time_fq2 = '2022-07-12\\11-40-36'

datfile_fq1 = load_dat(start_time_fq1)
datfile_fq2 = load_dat(start_time_fq2)

# %% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_PWR = -5
P4_PWR = 3

fq1, fq1_, fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_PWR, P4_PWR)

# %% PLotting

fig, axs = plt.subplots(1, 2, sharey=True)

vP2s1 = datfile_fq1.vPdetuning_set
vP1s1 = -1 * np.array(vP2s1)

vP2s2 = datfile_fq2.vpB12_set
vP1s2 = -1 * np.array(vP2s2)

detuning1 = (np.array(datfile_fq1.vPdetuning_set)**2 * 2)**0.5
detuning2 = (np.array(datfile_fq2.vpB12_set)**2 * 2)**0.5

norm = mpl.colors.Normalize(vmin=0.1, vmax=0.9)

axs[0].pcolor(datfile_fq1.frequency_set[0]/1e9,
              vP1s1,
              datfile_fq1.su0,
              shading='auto', cmap='hot', norm=norm)

# axs[0].pcolor(datfile_fq1.frequency_set[0],
#               - detuning1[round(len(detuning1)/2):],
#               datfile_fq1.su0[round(len(detuning1)/2):],
#               shading='auto', cmap='hot', norm=norm)

axs[1].pcolor(datfile_fq2.frequency_set[0]/1e9,
              vP1s2,
              datfile_fq2.su0[::-1],  # inverse order
              shading='auto', cmap='hot', norm=norm)

c = axs[1].pcolor(datfile_fq2.frequency_set[0]/1e9,
                  - detuning2[round(len(detuning2)/2):],
                  # inverse order
                  datfile_fq2.su0[::-1][round(len(detuning2)/2):],
                  shading='auto', cmap='hot', norm=norm)

axs[0].scatter(fq1, -10, marker='*', color='green',
               s=50, label='operation point')
axs[1].scatter(fq2, -10, marker='*', color='green', s=50)

axs[0].set_ylabel(r'$\epsilon$ [mV]')
axs[0].set_xlabel(r'$f_{P4}$ [GHz]')
axs[1].set_xlabel(r'$f_{P2}$ [GHz]')

axs[1].set_ylim(-17, 17)

divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(c, cax=cax)
cbar.ax.set_ylabel(r'$1-P_{\downdownarrows}$', rotation=-90, labelpad=15)
cbar.ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])

plt.tight_layout()
axs[0].legend(loc=2, frameon=True)
plt.savefig(save_path+'\\figureS8_res_freq.png', dpi=300)

plt.show()

# %%

fig, axs = plt.subplots(2, 1)

det_point = (14, -14)
plot_sequence(datfile_fq1, ['vP1', 'vP2'], [
              'MW_p4', 'MW_p2'], ax=axs[0], xlim=(3.4e4, 3.7e4), legend=True)
axs[0].hlines(det_point[0], 3.4e4, 3.7e4, ls='--',
              lw=1, color='black', label=str(det_point))
axs[0].hlines(det_point[1], 3.4e4, 3.7e4, ls='--', lw=1, color='black')
# axs[0].legend()

det_point = (12, -12)
plot_sequence(datfile_fq2, ['vP1', 'vP2'], [
              'MW_p4', 'MW_p2'], ax=axs[1], xlim=(3.4e4, 3.7e4), legend=True)
axs[1].hlines(det_point[0], 3.4e4, 3.7e4, ls='--',
              lw=1, color='black', label=str(det_point))
axs[1].hlines(det_point[1], 3.4e4, 3.7e4, ls='--', lw=1, color='black')
# axs[1].legend()

plt.tight_layout()
plt.show()


mw_prop = get_mw_prop(datfile_fq1, ['p2', 'p4'])
f_drve = mw_prop['p4']['rf'] - mw_prop['p2']['rf']
