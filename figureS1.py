# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:27:48 2022

@author: vjohn
"""

# %% Imports

import __main__
from utils.settings import *
from utils.delft_tools import *
from config import FIGURE_DIR

# %% Save path

save_path = FIGURE_DIR / 'FigureS1'

# %% Load data

start_time_fq1 = '2022-07-07\\20-30-00'
start_time_fq2 = '2022-07-07\\18-14-31'

datfile_fq1 = datfile = load_dat(start_time_fq1)
datfile_fq2 = load_dat(start_time_fq2)

start_time_q1dif = '2022-07-07\\22-36-07'
datfile_Q1dif = load_dat(start_time_q1dif)

mixing_regime = 'difference'

# %% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_pwr = -5
P4_pwr = 3

fq1, fq1_, fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_pwr, P4_pwr)

fq = fq1/1e9

# %% Obtain driving frequencies

delta = datfile_Q1dif.delta_set.ndarray[0, :]
mixing = datfile_Q1dif.mixing_set.ndarray

# Obtain driving frequencies of the power sweep measurements
plot_seq = False

if plot_seq:
    plot_sequence(datfile_fq2, ['vP1', 'vP2'], [
                  'MW_p4', 'MW_p2'], xlim=(3.4e4, 3.7e4), legend=True)
    plt.hlines(12, 3.4e4, 3.7e4, ls='--', lw=1,
               color='black', label='(-12,12)')
    plt.hlines(-12, 3.4e4, 3.7e4, ls='--', lw=1, color='black')
    plt.legend()
    if hasattr(__main__, '__file__') is False:
        plt.show()

mw_prop = get_mw_prop(datfile_fq2, ['p2', 'p4'])
# mw_prop = get_mw_prop(datfile_fq1, ['p2', 'p4'])

# f_drive = mw_prop['p4']['rf'] - mw_prop['p2']['rf']
P4_drive = mw_prop['p4']['rf']
P2_drive = mw_prop['p2']['rf']


# Calculate delta_P2_drive for the power sweep to plot frequency combination on bichromatic bias spectroscopy plot

mw_prop_Q1dif = get_mw_prop(datfile_Q1dif, ['p2', 'p4'])
f_drive = mw_prop_Q1dif['p4']['rf'] - \
    (mw_prop_Q1dif['p2']['rf'] - delta[0]/1e9)

delta_P2_drive = (P2_drive - (P4_drive - f_drive))*1e3

# %% Plot Q1 difference spectrocopy

fig, axes = plt.subplot_mosaic([["spec", "pwr P2"],
                                ["spec", "pwr P4"]])

vmin = 0.15
vmax = 0.8

# fig, axes = plt.subplot_mosaic([["spec", "spec", "pwr P2", "pwr P4"]])

n = 1
# ax = []
# colors = ['#1f77b4', 'purple', 'turquoise', 'lightskyblue']


# ax = fig1.add_subplot(gs1[0,n-1])


# p4_LO = datfile_Q1dif.metadata['LOs']['MW_p4'] / 1e9
# p2_LO = datfile_Q1dif.metadata['LOs']['MW_p2'] / 1e9
# print(p4_LO, p2_LO)
# p4_freq = datfile_Q1dif.metadata['station']['instruments']['sig_gen2']['parameters']['frequency']['value'] /1e9
# p2_freq = datfile_Q1dif.metadata['station']['instruments']['sig_gen3']['parameters']['frequency']['value'] /1e9
# print(p4_freq, p2_freq)

fp2 = mixing

cm = axes['spec'].pcolor(delta/1e6, fp2/1e9, datfile_Q1dif.su0,
                         shading='auto', cmap='hot', zorder=0,
                         vmin=vmin, vmax=vmax,
                         rasterized=True)
# for spine in ax.spines.values():
#     spine.set_edgecolor(colors[n])
axes['spec'].set_xlabel(r'$\Delta f_{\mathrm{P2}}$' +
                        f' {unit_style("MHz")}')
axes['spec'].set_ylabel(r'$f_{\mathrm{P4}}$' +
                        f' {unit_style("GHz")}')

axes['spec'].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axes['spec'].set_xticks([-200, 0, 200])
ax2 = axes['spec'].twinx()
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
fp2_max, fp2_min = axes['spec'].get_ylim()
if mixing_regime == 'difference':
    ax2.set_ylim(fp2_max - f_drive, fp2_min - f_drive)
ax2.set_ylabel(r'$f_{\mathrm{P2}}$' +
               f' {unit_style("MHz")}')
axes['spec'].scatter([delta_P2_drive], [P4_drive], marker='d')


time1 = datfile_fq1.time_set[0]
time2 = datfile_fq2.time_set[0]

power1 = datfile_fq1.sig_gen2_power_set
power2 = datfile_fq2.sig_gen3_power_set

# fp4 = datfile_fq1.metadata['station']['instruments']['sig_gen2']['parameters']['frequency']['value']/1e9
# fp2 = datfile_fq1.metadata['station']['instruments']['sig_gen3']['parameters']['frequency']['value']/1e9

# fp4_2 = datfile_fq2.metadata['station']['instruments']['sig_gen2']['parameters']['frequency']['value']/1e9
# fp2_2 = datfile_fq2.metadata['station']['instruments']['sig_gen3']['parameters']['frequency']['value']/1e9

axes['pwr P4'].pcolor(time1,
                      power1,
                      datfile_fq1.su0,
                      shading='auto', cmap='hot',
                      vmin=vmin, vmax=vmax,
                      rasterized=True)
axes['pwr P2'].pcolor(time2,
                      power2,
                      datfile_fq2.su0,
                      shading='auto', cmap='hot',
                      vmin=vmin, vmax=vmax,
                      rasterized=True)

axes['pwr P2'].set_xticks([0, 200])
axes['pwr P4'].set_xticks([0, 200])

axes['pwr P4'].set_ylabel(r'$P_{\mathrm{P4}}$' +
                          f' {unit_style("dBm")}')
axes['pwr P4'].set_xlabel(f'time {unit_style("ns")}')
axes['pwr P2'].set_ylabel(r'$P_{\mathrm{P2}}$' +
                          f' {unit_style("dBm")}')
axes['pwr P2'].set_xlabel(f'time {unit_style("ns")}')


plt.tight_layout()

# CREATE SAVE PATH IF NOT EXIST
if not os.path.exists(save_path):
    os.makedirs(save_path)

plt.savefig(os.path.join(save_path, 'FigureS1_power.png'), dpi=300)
plt.savefig(os.path.join(save_path, 'FigureS1_power.pdf'), dpi=300)

if hasattr(__main__, '__file__') is False:
    plt.show()

# %% colorbar

fig, ax = plt.subplots(figsize=cm2inch(2.0, 1.6), nrows=1)

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))
ax.imshow(gradient, aspect='auto', cmap=cm.cmap)
pos = list(ax.get_position().bounds)
ax.set_yticks([])
ax.set_xticks([0, 256])
ax.set_xlabel(r'$1 - P_{\downdownarrows}$')
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
ax.set_xticklabels([vmin, vmax])
ax.xaxis.set_label_position('top')

# ax.set_xticklabels([])
# ax.set_xticks([])
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'FigureS1_cbar.pdf'),
            dpi=300, transparent=True)
if hasattr(__main__, '__file__') is False:
    plt.show()


# %% Plot traces of power sweep with Rabi fit
m = - 1  # 20
pulse_seg = 'p6'
det_point = (int(datfile_fq1.metadata['pc0']['vP1_baseband']['p6']['v_start']),
             int(datfile_fq1.metadata['pc0']['vP2_baseband']['p6']['v_start']))

ms = [-1, -20]

fig, axs = plt.subplots(len(ms), 1)

n = 0
for m in ms:
    power = datfile_fq1.sig_gen2_power_set[m]
    time_set = datfile_fq1.time_set[0]
    signal = datfile_fq1.su0[m]

    p0 = [0.4,  # amplitude
          0.015,  # freq
          0.4,  # alpha
          0.51,  # y0
          0.9*np.pi]  # phase

    fit_start = 5
    fit_end = -40
    fit_par = fit_data(time_set[fit_start:fit_end],
                       signal[fit_start:fit_end], p0=p0, func=Rabi, plot=False)
    freq = fit_par[1]
    rabi_time = 1/freq*0.5

    axs[n].plot(time_set, signal)
    axs[n].plot(time_set[fit_start:], Rabi(time_set[fit_start:], fit_par[0], fit_par[1],
                fit_par[2], fit_par[3], fit_par[4]), label=f'fit: {np.round(freq*1e3,2)} MHz')
    axs[n].set_title(f'$P_{4}$ = {power} dBm at (vP1, vP2) = {det_point}')
    axs[n].legend()
    axs[n].set_ylabel('$1-P_{\downdownarrows}$')

    n = n + 1


plt.xlabel('time (ns)')
plt.tight_layout()
if hasattr(__main__, '__file__') is False:
    plt.show()

# fit_par, t_rabi = cal_rabi_t(datfile, p0=p0)
# fit_rabi = Rabi(np.array(x), fit_par[0],fit_par[1],fit_par[2],fit_par[3],fit_par[4])
# ax.plot(x,fit_rabi, label='fit: t_rabi = {} ns'.format(round(t_rabi,2)))
