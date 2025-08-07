# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:59:25 2022

@author: TUD278427
"""

# %% Imports

from utils.settings import *
from utils.delft_tools import *
from config import FIGURE_DIR

# %% Save path

save_path = FIGURE_DIR / 'FigureS4'
# CREATE SAVE PATH IF NOT EXIST
if not os.path.exists(save_path):
    os.makedirs(save_path)

# %% load data
START_TIME_Q1DIF = '2022-07-11\\13-12-41'
START_TIME_Q2DIF = '2022-07-12\\14-28-39'
START_TIME_Q2SUM = '2022-07-12\\15-39-11'

start_time_list = [START_TIME_Q1DIF, START_TIME_Q2DIF, START_TIME_Q2SUM]

datfile = {}
mixing_regime = {START_TIME_Q1DIF: 'difference',
                 START_TIME_Q2DIF: 'difference',
                 START_TIME_Q2SUM: 'sum'}

qubit = {START_TIME_Q1DIF: 'Q1',
         START_TIME_Q2DIF: 'Q2',
         START_TIME_Q2SUM: 'Q2'}

for START_TIME in start_time_list:
    datfile[START_TIME] = load_dat(START_TIME)

start_time_rabi_q1dif_list = ['2022-07-13\\15-26-45',
                              '2022-07-13\\15-44-07',
                              '2022-07-13\\15-56-21']
start_time_rabi_q2dif_list = ['2022-07-13\\14-27-14',
                              '2022-07-13\\14-34-26',
                              '2022-07-13\\14-10-52',
                              '2022-07-13\\14-40-14']
start_time_rabi_q2sum_list = ['2022-07-13\\15-07-05',
                              '2022-07-13\\14-51-17',
                              '2022-07-13\\14-55-58']

start_time_rabi_list = (start_time_rabi_q1dif_list +
                        start_time_rabi_q2dif_list +
                        start_time_rabi_q2sum_list)
start_time_rabi_list_list = [start_time_rabi_q1dif_list,
                             start_time_rabi_q2dif_list,
                             start_time_rabi_q2sum_list]

fp4_fp2 = {}
datfiles_rabi = {}

for start_time_rabi in start_time_rabi_list:
    datfile_rabi = load_dat(start_time_rabi)
    datfiles_rabi[start_time_rabi] = datfile_rabi

    fp4 = (datfile_rabi.metadata['station']['instruments']['sig_gen2']
           ['parameters']['frequency']['value'])/1e9
    fp2 = (datfile_rabi.metadata['station']['instruments']['sig_gen3']
           ['parameters']['frequency']['value'])/1e9
    fp4_fp2[start_time_rabi] = (fp4, fp2)

    if start_time_rabi in start_time_rabi_q2sum_list:
        mixing_regime[start_time_rabi] = 'sum'
    elif start_time_rabi in (start_time_rabi_q1dif_list +
                             start_time_rabi_q2dif_list):
        mixing_regime[start_time_rabi] = 'difference'


# %% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_PWR = -5
P4_PWR = 3

fq1, fq1_, fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_PWR, P4_PWR)

# %%
# mixing_regime = ['difference', 'difference', 'sum']

scatter_points = {}

for n in [0, 1, 2]:
    START_TIME = start_time_list[n]
    scatter_points[START_TIME] = []
    for m in range(len(start_time_rabi_list_list[n])):
        delta = datfile[START_TIME].delta_set.ndarray[0, :]
        mixing = datfile[START_TIME].mixing_set.ndarray

        # Obtain driving frequencies of the Rabi oscillations
        start_time_rabi = start_time_rabi_list_list[n][m]
        mw_prop = get_mw_prop(datfiles_rabi[start_time_rabi], ['p2', 'p4'])
        P4_drive = mw_prop['p4']['rf']
        P2_drive = mw_prop['p2']['rf']

        # Calculate delta_P2_drive for the Rabi to plot frequency combination
        # on BICHROmatic bias spectroscopy plot
        mw_prop_Q = get_mw_prop(datfile[START_TIME], ['p2', 'p4'])

        if mixing_regime[START_TIME] == 'difference':
            f_drive = (mw_prop_Q['p4']['rf'] -
                       (mw_prop_Q['p2']['rf'] - delta[0]/1e9))
            delta_P2_drive = (P2_drive - (P4_drive - f_drive))*1e3
        elif mixing_regime[START_TIME] == 'sum':
            f_drive = (mw_prop_Q['p4']['rf'] +
                       (mw_prop_Q['p2']['rf'] - delta[0]/1e9))
            delta_P2_drive = (P2_drive + (P4_drive - f_drive))*1e3

        # print(f_drive-fq1)
        scatter_points[START_TIME].append([delta_P2_drive, P4_drive])
        # print([delta_P2_drive, P4_drive])


# %% Calculate values of all possible resonances

factors = [0, 1, 2, 3, -1, -2, -3]
factors_pos = [0, 1]

fq_sum = (fq1+fq1_+fq2+fq2_)/2
fq_dif = abs(fq2-fq1)
f_res_list = [fq1, fq2,  # fq2-fq1,
              fq_sum]
f_res_dict = {fq1: [1, 0],
              fq2: [0, 1],
              fq2-fq1: [-1, 1],
              (fq1+fq1_+fq2+fq2_)/2: [1, 1]}
f_res_list.sort()

f_res_list = list(dict.fromkeys(f_res_list))  # remove duplicates

f_res_list = [item for item in f_res_list
              if item > 0.8 and item < 5]  # remove small and high values

# %% Plotting

vmin = 0.15
vmax = 0.8

fig, axes = plt.subplot_mosaic([["Q1_dif", "Q1_dif_sim", "empty", "Q2_dif", "Q2_dif_sim", "empty2", "Q2_sum", "Q2_sum_sim"],
                                ["Rabi_Q1d1", "Rabi_Q1d1", "empty", "Rabi_Q2d1",
                                    "Rabi_Q2d1", "empty2", "Rabi_Q2s1", "Rabi_Q2s1"],
                                ["Rabi_Q1d2", "Rabi_Q1d2", "empty", "Rabi_Q2d2",
                                    "Rabi_Q2d2", "empty2", "Rabi_Q2s2", "Rabi_Q2s2"],
                                ["Rabi_Q1d3", "Rabi_Q1d3", "empty", "Rabi_Q2d3",
                                    "Rabi_Q2d3", "empty2", "Rabi_Q2s3", "Rabi_Q2s3"],
                                ["Rabi_Q1d4", "Rabi_Q1d4", "empty", "Rabi_Q2d4", "Rabi_Q2d4", "empty2", "Rabi_Q2s4", "Rabi_Q2s4"]],
                               figsize=cm2inch(15.24, 17.39), sharex=False, sharey=False,
                               gridspec_kw={'height_ratios': [6, 1, 1, 1, 1],
                                            'width_ratios': [1, 1, 1, 1, 1, 1, 1, 1]})

axes['empty'].set_visible(False)
axes['empty2'].set_visible(False)
axes['Rabi_Q1d4'].set_visible(False)
axes['Rabi_Q2s4'].set_visible(False)

n = 0
colors = ['purple', 'turquoise', 'lightskyblue', '#1f77b4', '#ff7f0e',
          '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf']
SHOW_DRIVING_PROP = False

# *****************************************************************************
# Plotting colour maps

for START_TIME in start_time_list:
    ax = axes[f'Q{qubit[START_TIME][-1]}_{mixing_regime[START_TIME][:3]}']

    delta = datfile[START_TIME].delta_set.ndarray[0, :]

    mixing = datfile[START_TIME].mixing_set.ndarray
    fp2 = mixing

    if qubit[START_TIME] == 'Q1':
        fq = fq1/1e9
    elif qubit[START_TIME] == 'Q2':
        fq = fq2/1e9

    cm = ax.pcolor(delta/1e6, fp2/1e9, datfile[START_TIME].su0,
                   shading='auto', cmap='hot', zorder=1,
                   vmin=vmin, vmax=vmax,
                   rasterized=True)

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("top", size="5%", pad="2%")
    # cbar = plt.colorbar(c, cax=cax, orientation="horizontal")
    # cbar.ax.set_xlabel(r'$1-P_{\downdownarrows}$', labelpad=-40)
    # cbar.ax.set_yticks([0, 0.2, 0.4, 0.6])
    # cax.xaxis.set_ticks_position("top")

    j = 0
    for scatter_point in scatter_points[START_TIME]:
        delta_P2_drive, P4_drive = scatter_point
        ax.scatter([delta_P2_drive], [P4_drive], s=50, zorder=1,
                   marker='d', color=colors[3+j], edgecolors='black')
        j = j + 1

    for spine in ax.spines.values():
        spine.set_edgecolor(colors[n])

    if n == 0:
        ax.set_ylabel('$f_{\mathrm{P4}}$' +
                      f' {unit_style("GHz")}')
    ax.set_xlabel(r'$\Delta f_{\mathrm{P2}}$' +
                  f' {unit_style("MHz")}')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ax2 = ax.twinx()
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    fp2_max, fp2_min = ax.get_ylim()
    ax2.yaxis.set_tick_params(labelright=False)

    ax.set_xlim(delta.min()/1e6, delta.max()/1e6)
    ax.set_ylim(mixing.min()/1e9, mixing.max()/1e9)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    for spine in ax2.spines.values():
        spine.set_edgecolor(colors[n])

    n = n + 1

# *****************************************************************************
# Q1 dif simulated lines
BICHRO = 'Q1 dif'
N_HARM = 3

if BICHRO[1] == '1':
    fq = fq1
elif BICHRO[1] == '2':
    fq = fq2
if BICHRO[-3:] == 'dif':
    m = -1
elif BICHRO[-3:] == 'sum':
    m = 1

START_TIME = START_TIME_Q1DIF
delta = datfile[START_TIME].delta_set.ndarray[0, :]

mixing = datfile[START_TIME].mixing_set.ndarray
fp2 = mixing

fp4_start = mixing.min()/1e9
fp4_stop = mixing.max()/1e9

fp2_start = (fq - fp4_start)/m
fp2_stop = (fq - fp4_stop)/m

fp4_delta_span = 0.08

fp2_delta = np.linspace(-40, 40, 81)  # MHz

colormap = plt.cm.nipy_spectral
c = 0

fres_nP2_nP4 = [[fq1, 1, -1],
                [fq1, 1, 0],
                [fq2, 0, 1],
                [fq2, 2, 0],
                # [fq_sum, 1, 1],
                [fq_sum, 2, 0],
                [fq_sum, 3, 0]
                ]

for [f_res, n_P2, n_P4] in fres_nP2_nP4:
    n_Q1 = f_res_dict[f_res][0]
    n_Q2 = f_res_dict[f_res][1]

    if n_Q1 == 0:
        label_Q = f'{n_Q2}Q2'
    elif n_Q2 == 0:
        label_Q = f'{n_Q1}Q1'
    else:
        label_Q = f'({n_Q1}Q1, {n_Q2}Q2)'

    if n_P2 == 0:
        label_P = f'{n_P4}P4'
    elif n_P4 == 0:
        label_P = f'{n_P2}P2'
    else:
        label_P = f'({n_P2}P2, {n_P4}P4)'

    if n_P4-m*n_P2 == 0:
        SLOPE = 'inf'
        fp2_delta_vert = (fq1 - f_res/n_P2)*1e3
        if abs(fp2_delta_vert) < fp4_delta_span/2*1e3:
            axes['Q1_dif_sim'].vlines(fp2_delta_vert, fp4_start, fp4_stop,
                                      label=f'{label_Q}^{label_P}, SLOPE = {SLOPE}',
                                      color='black', ls='-', lw=1, alpha=0.1)
            c = c + 1
    else:
        SLOPE = - n_P2 / (n_P4-m*n_P2)

        fp4 = (f_res/(-m*n_P2+n_P4) +
               n_P2/(-m*n_P2+n_P4) * (-m*fq1-fp2_delta*1e-3))

        if any(fp4 > fp4_start) and any(fp4 < fp4_stop):
            axes['Q1_dif_sim'].plot(fp2_delta, fp4,
                                    label=(f'{label_Q}^{label_P}, '
                                           f'SLOPE = {np.round(SLOPE,2)}'),
                                    color='black', ls='-', lw=1)
            c = c + 1

axes['Q1_dif_sim'].set_ylim(fp4_start, fp4_stop)
axes['Q1_dif_sim'].set_xlim(-fp4_delta_span/2*1e3, fp4_delta_span/2*1e3)
axes['Q1_dif_sim'].set_xlabel(r'$\Delta f_{\mathrm{P2}}$' +
                              f' {unit_style("MHz")}')


ax2 = axes['Q1_dif_sim'].twinx()
ax2.set_ylim(fp2_start, fp2_stop)

# *****************************************************************************
# Q2 dif simulated lines

BICHRO = 'Q2 dif'

if BICHRO[1] == '1':
    fq = fq1
elif BICHRO[1] == '2':
    fq = fq2
if BICHRO[-3:] == 'dif':
    m = -1
elif BICHRO[-3:] == 'sum':
    m = 1
N_HARM = 3

START_TIME = START_TIME_Q2DIF
delta = datfile[START_TIME].delta_set.ndarray[0, :]

mixing = datfile[START_TIME].mixing_set.ndarray
fp2 = mixing

fp4_start = mixing.min()/1e9
fp4_stop = mixing.max()/1e9

fp2_start = (fq - fp4_start)/m
fp2_stop = (fq - fp4_stop)/m

fp4_delta_span = 0.08

fp2_delta = np.linspace(-40, 40, 81)  # MHz

c = 0

fres_nP2_nP4 = [[fq1, -2, 1],
                [fq1, 1, -1],
                [fq1, 1, 0],
                [fq2, 2, 0],
                [fq_sum, 0, 1],
                [fq_sum, 3, 0],
                [fq_dif, 1, 0]
                ]

for [f_res, n_P2, n_P4] in fres_nP2_nP4:
    n_Q1 = f_res_dict[f_res][0]
    n_Q2 = f_res_dict[f_res][1]

    if n_Q1 == 0:
        label_Q = f'{n_Q2}Q2'
    elif n_Q2 == 0:
        label_Q = f'{n_Q1}Q1'
    else:
        label_Q = f'({n_Q1}Q1, {n_Q2}Q2)'

    if n_P2 == 0:
        label_P = f'{n_P4}P4'
    elif n_P4 == 0:
        label_P = f'{n_P2}P2'
    else:
        label_P = f'({n_P2}P2, {n_P4}P4)'

    if n_P4-m*n_P2 == 0:
        SLOPE = 'inf'
        fp2_delta_vert = (fq1 - f_res/n_P2)*1e3
        if abs(fp2_delta_vert) < fp4_delta_span/2*1e3:
            axes['Q2_dif_sim'].vlines(fp2_delta_vert, fp4_start, fp4_stop,
                                      label=f'{label_Q}^{label_P}, SLOPE = {SLOPE}',
                                      color='black', ls='-', lw=1, alpha=0.1)
            c = c + 1
    else:
        SLOPE = - n_P2 / (n_P4-m*n_P2)

        fp4 = (f_res/(-m*n_P2+n_P4) +
               n_P2/(-m*n_P2+n_P4) * (-m*fq-fp2_delta*1e-3))

        if any(fp4 > fp4_start) and any(fp4 < fp4_stop):
            axes['Q2_dif_sim'].plot(fp2_delta, fp4,
                                    label=(f'{label_Q}^{label_P}, '
                                           'SLOPE = {np.round(SLOPE,2)}'),
                                    color='black', ls='-', lw=1)
            c = c + 1

axes['Q2_dif_sim'].set_ylim(fp4_start, fp4_stop)
axes['Q2_dif_sim'].set_xlim(-fp4_delta_span/2*1e3, fp4_delta_span/2*1e3)
axes['Q2_dif_sim'].set_xlabel(r'$\Delta f_{\mathrm{P2}}$' +
                              f' {unit_style("MHz")}')

ax2 = axes['Q2_dif_sim'].twinx()
ax2.set_ylim(fp2_start, fp2_stop)


# *****************************************************************************
# Q2 sum simulated plot

BICHRO = 'Q2 sum'

if BICHRO[1] == '1':
    fq = fq1
elif BICHRO[1] == '2':
    fq = fq2
if BICHRO[-3:] == 'dif':
    m = -1
elif BICHRO[-3:] == 'sum':
    m = 1
N_HARM = 3

START_TIME = START_TIME_Q2SUM
delta = datfile[START_TIME].delta_set.ndarray[0, :]

mixing = datfile[START_TIME].mixing_set.ndarray
fp2 = mixing

fp4_start = mixing.min()/1e9
fp4_stop = mixing.max()/1e9

fp2_start = (fq - fp4_start)/m
fp2_stop = (fq - fp4_stop)/m

fp4_delta_span = 0.08

fp2_delta = np.linspace(-40, 40, 81)  # MHz

c = 0

fres_nP2_nP4 = [[fq1, -1, 2],
                [fq1, 0, 1],
                [fq1, 1, 0],
                [fq1, 1, 1],
                [fq1, 2, 1],
                [fq1, 2, -1],
                [fq2, 0, 2],
                [fq2, 2, 0],
                [fq_sum, 1, 2],
                [fq_sum, 2, 1],
                [fq_sum, 3, 0]
                ]

for [f_res, n_P2, n_P4] in fres_nP2_nP4:
    n_Q1 = f_res_dict[f_res][0]
    n_Q2 = f_res_dict[f_res][1]

    if n_Q1 == 0:
        label_Q = f'{n_Q2}Q2'
    elif n_Q2 == 0:
        label_Q = f'{n_Q1}Q1'
    else:
        label_Q = f'({n_Q1}Q1, {n_Q2}Q2)'

    if n_P2 == 0:
        label_P = f'{n_P4}P4'
    elif n_P4 == 0:
        label_P = f'{n_P2}P2'
    else:
        label_P = f'({n_P2}P2, {n_P4}P4)'

    if n_P4-m*n_P2 == 0:
        SLOPE = 'inf'
        fp2_delta_vert = (fq1 - f_res/n_P2)*1e3
        if abs(fp2_delta_vert) < fp4_delta_span/2*1e3:
            axes['Q2_sum_sim'].vlines(fp2_delta_vert, fp4_start, fp4_stop,
                                      label=f'{label_Q}^{label_P}, SLOPE = {SLOPE}',
                                      color='black', ls='-', lw=1, alpha=0.1)
            c = c + 1
    else:
        SLOPE = - n_P2 / (n_P4-m*n_P2)

        fp4 = (f_res/(-m*n_P2+n_P4) +
               n_P2/(-m*n_P2+n_P4) * (-m*fq-fp2_delta*1e-3))

        if any(fp4 > fp4_start) and any(fp4 < fp4_stop):
            axes['Q2_sum_sim'].plot(fp2_delta, fp4,
                                    label=(f'{label_Q}^{label_P}, '
                                           'SLOPE = {np.round(SLOPE,2)}'),
                                    color='black', ls='-', lw=1)
            c = c + 1

axes['Q2_sum_sim'].set_ylim(fp4_start, fp4_stop)
axes['Q2_sum_sim'].set_xlim(-fp4_delta_span/2*1e3, fp4_delta_span/2*1e3)
axes['Q2_sum_sim'].set_xlabel(r'$\Delta f_{\mathrm{P2}}$' +
                              f' {unit_style("MHz")}')

ax2 = axes['Q2_sum_sim'].twinx()
ax2.set_ylim(fp2_start, fp2_stop)
ax2.set_ylabel(r'$f_{\mathrm{P2}}$ (GHz)')

# *****************************************************************************
# Plotting Rabi oscillations
m = 1
bbox_to_anchor = (0.5, -0.15)
frabi_list = []
p0_list = [[0.3, 1/240, 0.12, 0.6, np.pi],
           [0.3, 1/170, 0.12, 0.55, np.pi],
           [0.3, 1/170, 0.12, 0.6, np.pi]]
for start_time_rabi in start_time_rabi_q1dif_list:
    ax2 = axes[f'Rabi_Q1d{m}']

    time = np.array(datfiles_rabi[start_time_rabi].time_set)
    prob = np.array(datfiles_rabi[start_time_rabi].su0)

    popt, pcov = curve_fit(Rabi, time, prob, p0=p0_list[m-1])
    perr = np.sqrt(np.diag(pcov))
    CROP = 1
    frabi = np.round(popt[1]*1e3, 2)
    frabi_list.append(frabi)

    ax2.plot(time[CROP:], Rabi(time, *popt)[CROP:], 'black',
             lw=1, zorder=10, label=f'{frabi} MHz'
             )
    ax2.legend(loc='lower center', bbox_to_anchor=bbox_to_anchor)

    ax2.plot(time, prob, color=colors[2+m])
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.set_ylim(0.01, 0.99)
    ax2.set_xlabel(f'time {unit_style("ns")}')

    for spine in ax2.spines.values():
        spine.set_edgecolor(colors[0])

    ax2.set_ylabel(r'$1-P_\downdownarrows$')
    if SHOW_DRIVING_PROP:
        mw_prop = get_mw_prop(datfiles_rabi[start_time_rabi], ['p2', 'p4'])
        title_ax2 = (f"({mw_prop['p2']['PWR']}, "
                     f"{mw_prop['p4']['PWR']}) dBm, "
                     f"({mw_prop['p2']['rf']}, "
                     f"{mw_prop['p4']['rf']}) GHz")
        ax2.set_title(title_ax2)
    m = m + 1

m = 1
p0_list = [[0.3, 1/600, 0.12, 0.55, np.pi],
           [0.3, 1/800, 0.12, 0.55, np.pi],
           [0.3, 1/700, 0.12, 0.55, np.pi],
           [0.3, 1/250, 0.12, 0.65, np.pi]]
for start_time_rabi in start_time_rabi_q2dif_list:
    ax3 = axes[f'Rabi_Q2d{m}']

    time = np.array(datfiles_rabi[start_time_rabi].time_set)
    prob = np.array(datfiles_rabi[start_time_rabi].su0)

    popt, pcov = curve_fit(Rabi, time, prob, p0=p0_list[m-1])
    perr = np.sqrt(np.diag(pcov))
    CROP = 1
    frabi = np.round(popt[1]*1e3, 2)
    frabi_list.append(frabi)

    ax3.plot(time[CROP:], Rabi(time, *popt)[CROP:], 'black',
             lw=1, zorder=10, label=f'{frabi} MHz')
    ax3.legend(loc='lower center', bbox_to_anchor=bbox_to_anchor)

    ax3.plot(time, prob, color=colors[2+m])
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax3.set_ylim(0.01, 0.99)
    ax3.set_xlabel(f'time {unit_style("ns")}')

    for spine in ax3.spines.values():
        spine.set_edgecolor(colors[1])

    ax3.set_ylabel(r'$1-P_\downdownarrows$')
    mw_prop = get_mw_prop(datfiles_rabi[start_time_rabi], ['p2', 'p4'])
    if SHOW_DRIVING_PROP:
        mw_prop = get_mw_prop(datfiles_rabi[start_time_rabi], ['p2', 'p4'])
        title_ax3 = (f"({mw_prop['p2']['PWR']}, "
                     f"{mw_prop['p4']['PWR']}) dBm,  "
                     f"({mw_prop['p2']['rf']}, "
                     f"{mw_prop['p4']['rf']}) GHz")
        ax3.set_title(title_ax3)
    m = m + 1

m = 1
p0_list = [[0.3, 2e-3, 0.5, 0.55, np.pi],
           [0.3, 4e-3, 0.12, 0.55, np.pi],
           [0.3, 1/280, 0.12, 0.55, np.pi]]
for start_time_rabi in start_time_rabi_q2sum_list:
    ax4 = axes[f'Rabi_Q2s{m}']

    time = np.array(datfiles_rabi[start_time_rabi].time_set)
    prob = np.array(datfiles_rabi[start_time_rabi].su0)

    popt, pcov = curve_fit(Rabi, time, prob, p0=p0_list[m-1])
    perr = np.sqrt(np.diag(pcov))
    CROP = 1
    frabi = np.round(popt[1]*1e3, 2)
    frabi_list.append(frabi)

    ax4.plot(time[CROP:], Rabi(time, *popt)[CROP:], 'black',
             lw=1, zorder=10, label=f'{frabi} MHz')
    ax4.legend(loc='lower center', bbox_to_anchor=bbox_to_anchor)

    ax4.plot(time, prob, color=colors[2+m])
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax4.set_ylim(0.01, 0.99)
    ax4.set_xlabel(f'time {unit_style("ns")}')
    for spine in ax4.spines.values():
        spine.set_edgecolor(colors[2])

    ax4.set_ylabel(r'$1-P_\downdownarrows$')
    if SHOW_DRIVING_PROP:
        mw_prop = get_mw_prop(datfiles_rabi[start_time_rabi], ['p2', 'p4'])
        title_ax4 = (f"({mw_prop['p2']['PWR']}, "
                     f"{mw_prop['p4']['PWR']}) dBm,  "
                     f"({mw_prop['p2']['rf']}, "
                     f"{mw_prop['p4']['rf']}) GHz")
        ax4.set_title(title_ax3)
    m = m + 1

axes["Q1_dif_sim"].sharey(axes["Q1_dif"])
axes["Q1_dif_sim"].set_yticklabels([])

axes["Q2_dif_sim"].sharey(axes["Q2_dif"])
axes["Q2_dif_sim"].set_yticklabels([])

axes["Q2_sum_sim"].sharey(axes["Q2_sum"])
axes["Q2_sum_sim"].set_yticklabels([])

fig.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.6)
plt.savefig(os.path.join(save_path, 'FigureS4_more_Rabis.png'), dpi=300)
plt.savefig(os.path.join(save_path, 'FigureS4_more_Rabis.pdf'), dpi=300)

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

plt.savefig(os.path.join(save_path, 'FigureS4_cbar.pdf'),
            dpi=300, transparent=True)

plt.show()
