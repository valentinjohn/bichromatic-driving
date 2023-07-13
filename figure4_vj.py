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
VP2 = -VP1
ratio_avg = t_avg**2/U
chi3_avg = chi3_AC2(VP1, ratio_avg)*1000/4.13566553853809*10**6  # Hz

y0s = [2675040111.7585015,
       2660330415.3005323,
       2650364760.5178795,
       2631532967.6893964,
       2616048804.3389482]

x0s = [2000000.0,
       1500000.0,
       -3000000.0,
       -6000000.0,
       -10000000.0]

chi3s = [21630160.0,
         10914770.0,
         5669362.0,
         5362999.0,
         2039367.0]

As = [11847479.048683455,
      4907393.14487201,
      2605955.8783992277,
      942515.3988537926,
      1835671.9874591588]

fq1s = [1478719177.586693,
        1501973721.8806822,
        1519687724.192985,
        1534636830.5540218,
        1547998214.7768426]

BB = [5.0*10**(-9),
      5.0*10**(-9),
      5.0*10**(-9),
      5.0*10**(-9),
      5.0*10**(-9)]

fcs = [1.05*10**9,
       1.05*10**9,
       1.05*10**9,
       1.05*10**9,
       1.05*10**9]

y1ds = [2.667*10**9,
        2.656*10**9,
        2.648*10**9,
        2.6313*10**9,
        2.6143*10**9]

y1us = [3.8*10**9,
        3.8*10**9,
        3.8*10**9,
        3.8*10**9,
        3.8*10**9]

y2ds = [2.3*10**9,
        2.3*10**9,
        2.3*10**9,
        2.3*10**9,
        2.3*10**9]

y2us = [2.63*10**9,
        2.64*10**9,
        2.642*10**9,
        2.629*10**9,
        2.6122*10**9]

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
    axs[n].plot(x_1(y1, y0s[n], x0s[n], As[n], chi3s[n], fq1s[n], fcs[n]) /
                1e6, y1/1e9, color=color_Q2sum, linewidth=0.8, linestyle=(0, (3, 3)))
    axs[n].plot(x_2(y2, y0s[n], x0s[n], As[n], chi3s[n], fq1s[n], fcs[n]) /
                1e6, y2/1e9, color=color_Q2sum, linewidth=0.8, linestyle=(0, (3, 3)))
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
