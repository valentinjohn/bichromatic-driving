# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 08:54:40 2023

@author: vjohn
"""

# %% Imports
from utils.settings import *

# %% Save path
save_path = get_save_path('FigureS6')

# %% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_pwr = -5
P4_pwr = 3

fq1, fq1_, fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_pwr, P4_pwr)

# %% load data
start_time_q1dif = '2022-07-11\\13-12-41'
datfile = {}

qubit = {start_time_q1dif: 'Q1'}

start_time = start_time_q1dif
datfile[start_time] = load_dat(start_time)

# %% Calculate values of all possible resonances

factors = [0, 1, 2, 3, -1, -2, -3]
factors_pos = [0, 1]

fq_sum = (fq1+fq1_+fq2+fq2_)/2
f_res_list = [fq1, fq2,  # fq2-fq1,
              fq_sum]
f_res_dict = {fq1: [1, 0],
              fq2: [0, 1],
              # fq2-fq1: [-1, 1],
              (fq1+fq1_+fq2+fq2_)/2: [1, 1]}

f_res_list.sort()
f_res_list = list(dict.fromkeys(f_res_list))  # remove duplicates


f_res_list = [item for item in f_res_list if item >
              0.8 and item < 5]  # remove small and high values


# %%
bichro = 'Q1 dif'
n_harm = 4
n_lines = 7  # 12

if bichro[1] == '1':
    fq = fq1
elif bichro[1] == '2':
    fq = fq2
if bichro[-3:] == 'dif':
    m = -1
elif bichro[-3:] == 'sum':
    m = 1

fig, axes = plt.subplot_mosaic([["Q1_dif", "Q1_dif_sim", "Q1_dif_leg", "Q1_dif_leg"]],
                               figsize=(6, 4), sharex=True, sharey=True)
axes['Q1_dif_leg'].set_visible(False)


# *****************************************************************************
# Q1 dif colour plot

start_time = start_time_q1dif
delta = datfile[start_time].delta_set.ndarray[0, :]

mixing = datfile[start_time].mixing_set.ndarray
fp2 = mixing

fp4_start = mixing.min()/1e9
fp4_stop = mixing.max()/1e9

fp2_start = (fq - fp4_start)/m
fp2_stop = (fq - fp4_stop)/m

fp4_delta_span = 0.08

fp4 = np.linspace(fp4_start, fp4_stop, 501)


c = axes['Q1_dif'].pcolor(
    delta/1e6, fp2/1e9, datfile[start_time].su0, shading='auto', cmap='hot', zorder=0)

axes['Q1_dif'].set_ylabel('$f_{P4}$ [GHz]')
axes['Q1_dif'].set_xlabel(r'$\Delta f_{P2}$ [MHz]')


axes['Q1_dif'].set_xlim(delta.min()/1e6, delta.max()/1e6)
axes['Q1_dif'].set_ylim(fp4_start, fp4_stop)

ax2 = axes['Q1_dif'].twinx()
ax2.set_ylim(fp2_start, fp2_stop)
ax2.yaxis.set_tick_params(labelright=False)

# *****************************************************************************
# Q1 dif simulated lines
fp2_delta = np.linspace(-40, 40, 81)  # MHz

colormap = plt.cm.nipy_spectral
colors = colormap(np.linspace(0, 1, n_lines))
c = 0

fres_nP2_nP4 = [[fq1, 1, -1],
                [fq1, 1, 0],
                [fq2, 0, 1],
                [fq2, 2, 0],
                [fq_sum, 1, 1],
                [fq_sum, 2, 0],
                [fq_sum, 3, 0]
                ]

# for f_res in f_res_list:
#     for n_P2 in range(-n_harm,n_harm):
#         for n_P4 in range(-n_harm,n_harm):
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
        slope = 'inf'
        fp2_delta_vert = (fq1 - f_res/n_P2)*1e3
        if abs(fp2_delta_vert) < fp4_delta_span/2*1e3:
            axes['Q1_dif_sim'].vlines(fp2_delta_vert, fp4_start, fp4_stop,
                                      label=f'{label_Q}^{label_P}, slope = {slope}',
                                      color=colors[c])
            c = c + 1
    else:
        slope = - n_P2 / (n_P4-m*n_P2)

        fp4 = f_res/(-m*n_P2+n_P4) + n_P2/(-m*n_P2+n_P4) * \
            (-m*fq1-fp2_delta*1e-3)

        if any(fp4 > fp4_start) and any(fp4 < fp4_stop):
            axes['Q1_dif_sim'].plot(fp2_delta, fp4,
                                    label=f'{label_Q}^{label_P}, slope = {np.round(slope,2)}',
                                    color=colors[c])
            c = c + 1

axes['Q1_dif_sim'].set_ylim(fp4_start, fp4_stop)
axes['Q1_dif_sim'].set_xlim(-fp4_delta_span/2*1e3, fp4_delta_span/2*1e3)
axes['Q1_dif_sim'].set_xlabel(r'$\Delta f_{P2}$ [MHz]')


ax2 = axes['Q1_dif_sim'].twinx()
ax2.set_ylim(fp2_start, fp2_stop)
ax2.set_ylabel(r'$f_{p2}$ (GHz)')

plt.figlegend(loc=(0.65, 0.4))
plt.tight_layout()
plt.show()
