# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 08:55:11 2022

@author: vjohn
"""

#%% Imports
from utils.settings import *

#%% Save path
save_path = get_save_path('Figure4')

#%% load data
start_times = ['2022-07-11\\12-03-11',
               '2022-07-11\\13-12-41',
               '2022-07-11\\18-00-58',
               '2022-07-11\\20-20-02',
               '2022-07-12\\09-12-02']

datfiles_dict = {}
detuning_list = []
for start_time in start_times:
    datfile = load_data(start_time)
    # detuning position from metadata, p6 is the sequence during which we
    # drive
    vP1 = datfile.metadata['pc0']['vP1_baseband']['p6']['v_start']
    vP2 = datfile.metadata['pc0']['vP2_baseband']['p6']['v_start']
    detuning = (vP1, vP2)
    detuning_list.append(detuning)
    datfiles_dict[detuning] = datfile

#%% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_pwr = -5
P4_pwr = 3

fq1, fq1_,fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_pwr, P4_pwr)

fq = fq1/1e9

#%% Plotting
fig, axs = plt.subplots(1, 5, figsize=(fig_size_single, 3), sharex=True)
n = 0
for detuning in detuning_list:
    datfile = datfiles_dict[detuning]

    delta = datfile.delta_set.ndarray[0,:]
    mixing = datfile.mixing_set.ndarray
    fp2 = mixing
    fp4 = abs(fp2 - fq)

    axs[n].pcolor(delta/1e6, fp2/1e9, datfile.su0, shading='auto', cmap='hot', zorder=0)
    axs[n].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[n].tick_params('x', labelrotation=45)
    axs[n].set_xticks([-30,0,+30])
    # axs[n].set_title(detuning, fontsize=8)
    for spine in axs[n].spines.values():
        spine.set_edgecolor('lightskyblue')

    if n > 0:
         axs[n].set_yticks([])

    ax2 = axs[n].twinx()
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    fp2_max, fp2_min = axs[n].get_ylim()
    ax2.set_ylim(fp2_max - fq, fp2_min - fq)
    if n < len(detuning_list)-1:
        ax2.set_yticklabels([])
        ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_edgecolor('deepskyblue')

    n = n + 1

axs[0].set_ylabel(r'$f_{p4}$ [GHz]')
axs[2].set_xlabel(r'$\Delta f_{p2}$ [MHz]')
ax2.set_ylabel('$f_{P2}$ [GHz]')

plt.tight_layout(w_pad = -0.5)
# plt.subplots_adjust(hspace=0.01)

plt.savefig(os.path.join(save_path, 'figure4a.pdf'), dpi=300)
# plt.savefig(os.path.join(save_path, 'figure4a.svg'), dpi=300)
plt.savefig(os.path.join(save_path, 'figure4a.png'), dpi=300)

plt.show()
