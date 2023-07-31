# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 08:55:11 2022

@author: vjohn
"""

# %% Imports
from utils.settings import *
from utils.delft_tools import *
from utils.budapest_tools import *

# %% Save path
save_path = get_save_path('Figure4')

# %% load data
start_times = ['2022-07-11\\12-03-11',
               '2022-07-11\\13-12-41',
               '2022-07-11\\18-00-58',
               '2022-07-11\\20-20-02',
               '2022-07-12\\09-12-02']

datfiles_dict = {}
detuning_list = []
for start_time in start_times:
    datfile = load_dat(start_time)
    # detuning position from metadata, p6 is the sequence during which we
    # drive
    vP1 = datfile.metadata['pc0']['vP1_baseband']['p6']['v_start']
    vP2 = datfile.metadata['pc0']['vP2_baseband']['p6']['v_start']
    detuning = (vP1, vP2)
    detuning_list.append(detuning)
    datfiles_dict[detuning] = datfile

# %% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_pwr = -5
P4_pwr = 3

fq1, fq1_, fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_pwr, P4_pwr)

fq = fq1

# %% Plotting
fig, axs = plt.subplots(1, 5, figsize=(fig_size_single, 3.7), sharex=True)
n = 0
vmin = 0.15
vmax = 0.9

VP1 = 1000*np.linspace(7, 11, 5)
t_avg=18.096097943069715

y0s = [2.66626806,
       2.65246803,
       2.64365476,
       2.62758725,
       2.61686396]

x0s = [7.03532953*10**(-3),
       6.57388819*10**(-3),
       -8.98660776*10**(-4),
       -5.58537923*10**(-3),
       -7.42040056*10**(-3)]

ts = [t_avg,
      t_avg,
       t_avg,
       t_avg,
       t_avg]

y1ds = [2.67,
       2.655,
        2.645,
        2.6277,
        2.6168]

y1us = [3.8,
        3.8,
        3.8,
        3.8,
        3.8]

y2ds = [2.3,
        2.3,
        2.3,
        2.3,
        2.3]

y2us = [2.66,
        2.65,
        2.642,
        2.627,
        2.6165]

for detuning in detuning_list:
    datfile = datfiles_dict[detuning]

    delta = datfile.delta_set.ndarray[0, :]
    mixing = datfile.mixing_set.ndarray
    fp2 = mixing
    fp4 = abs(fp2 - fq)

    axs[n].pcolor(delta/1e6, fp2/1e9, datfile.su0,
                  shading='auto', cmap='hot', zorder=0,
                  vmin=vmin, vmax=vmax)
    axs[n].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[n].tick_params('x', labelrotation=45)
    axs[n].set_xticks([-30, 0, +30])
    axs[n].set_yticks([2.4, 2.8, 3.4, 3.8])
    for spine in axs[n].spines.values():
        spine.set_edgecolor('blue')

    xi = np.linspace(-40*10**6, 39*10**6, 5000)
    plotting(xi, axs[n], -11.0+n, color=color_Q2sum)
    y1 = np.linspace(y1ds[n], y1us[n], 5000)
    y2 = np.linspace(y2ds[n], y2us[n], 5000)
    axs[n].plot(x_1(y1, x0s[n], y0s[n], ts[n],detuning[1]*1000)*1000, y1, color=color_Q2sum, linewidth=0.8, linestyle=(0, (3, 3)))
    axs[n].plot(x_1(y2, x0s[n], y0s[n], ts[n],detuning[1]*1000)*1000, y2, color=color_Q2sum, linewidth=0.8, linestyle=(0, (3, 3)))
    axs[n].set_xlim(-40, 39)
    axs[n].set_ylim(2.3, 3.8)

    if n > 0:
        axs[n].set_yticks([])

    ax2 = axs[n].twinx()
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    fp2_max, fp2_min = axs[n].get_ylim()
    ax2.set_ylim(fp2_max - fq, fp2_min - fq)
    if n < len(detuning_list)-1:
        ax2.set_yticks([])
    else:
        ax2.set_yticks([0.8, 1.2, 1.8, 2.2])
    for spine in ax2.spines.values():
        spine.set_edgecolor('deepskyblue')

    n = n + 1
fontsize_label = 7
axs[0].text(-10, 2.62, '$\mathrm{AC2}$', fontsize=7, color='white')
axs[0].set_ylabel(r'$f_{\mathrm{P4}}$' +
                  f' {unit_style("GHz")}',
                  labelpad=-10)
axs[2].set_xlabel(r'$\Delta f_{\mathrm{P2}}$' +
                  f' {unit_style("MHz")}')
axs[0].text(-90, 2.65, '$\mathrm{Q2^{P4}}$',
            fontsize=fontsize_label, c='black')

fig.suptitle('$\mathrm{Q1^{-P2,P4}}$')

ax2.set_ylabel('$f_{\mathrm{P2}}$' +
               f' {unit_style("GHz")}',
               labelpad=-10)

plt.subplots_adjust(left=0.07,
                    bottom=0.05,
                    right=0.93,
                    top=0.95,
                    hspace=0)

plt.tight_layout(w_pad=-0.5)
plt.show()

fig.savefig(os.path.join(save_path, 'Figure4_overlayed.pdf'),
            format='pdf', dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(save_path, 'Figure4_overlayed.png'),
            format='png', dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(save_path, 'Figure4_overlayed.svg'),
            format='svg', bbox_inches="tight")
