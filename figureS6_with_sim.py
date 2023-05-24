# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:32:06 2023

@author: vjohn
"""

# %% Imports
from utils.settings import *

# %% Save path
save_path = get_save_path('Figure2')

# %% load data
start_time_q1dif = '2022-07-11\\13-12-41'
start_time_q2dif = '2022-07-12\\14-28-39'
start_time_q2sum = '2022-07-12\\15-39-11'

start_time_list = [start_time_q1dif, start_time_q2dif, start_time_q2sum]

datfile = {}
mixing_regime = {start_time_q1dif: 'difference',
                 start_time_q2dif: 'difference',
                 start_time_q2sum: 'sum'}

qubit = {start_time_q1dif: 'Q1',
         start_time_q2dif: 'Q2',
         start_time_q2sum: 'Q2'}

for start_time in start_time_list:
    datfile[start_time] = load_data(start_time)

start_time_rabi_q1dif_list = ['2022-07-13\\15-26-45',
                              '2022-07-13\\15-44-07',
                              '2022-07-13\\15-56-21']
start_time_rabi_q2dif_list = ['2022-07-13\\14-34-26',
                              '2022-07-13\\14-10-52',
                              '2022-07-13\\14-40-14']
start_time_rabi_q2sum_list = ['2022-07-13\\15-07-05',
                              '2022-07-13\\14-55-58']

start_time_rabi_list = start_time_rabi_q1dif_list + \
    start_time_rabi_q2dif_list + start_time_rabi_q2sum_list
start_time_rabi_list_list = [start_time_rabi_q1dif_list,
                             start_time_rabi_q2dif_list, start_time_rabi_q2sum_list]

fp4_fp2 = {}
datfile_rabi = {}

for start_time_rabi in start_time_rabi_list:
    datfiles = load_data(start_time_rabi)
    datfile_rabi[start_time_rabi] = datfiles

    fp4 = datfiles[0].metadata['station']['instruments']['sig_gen2']['parameters']['frequency']['value']/1e9
    fp2 = datfiles[0].metadata['station']['instruments']['sig_gen3']['parameters']['frequency']['value']/1e9
    fp4_fp2[start_time_rabi] = (fp4, fp2)
    if start_time_rabi in start_time_rabi_q2sum_list:
        mixing_regime[start_time_rabi] = 'sum'
    elif start_time_rabi in (start_time_rabi_q1dif_list + start_time_rabi_q2dif_list):
        mixing_regime[start_time_rabi] = 'difference'

# %% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_pwr = -5
P4_pwr = 3

fq1, fq1_, fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_pwr, P4_pwr)

# %% Calculate values of all possible resonances

factors = [0, 1, 2, 3, -1, -2, -3]
factors_pos = [0, 1]

f_res_list = [fq1, fq2, fq2-fq1, (fq1+fq1_+fq2+fq2_)/2]
f_res_dict = {fq1: [1, 0],
              fq2: [0, 1],
              fq2-fq1: [-1, 1],
              (fq1+fq1_+fq2+fq2_)/2: [1, 1]}

f_res_list.sort()
f_res_list = list(dict.fromkeys(f_res_list))  # remove duplicates


f_res_list = [item for item in f_res_list if item >
              0.8 and item < 5]  # remove small and high values


# %% Q1 dif: fq1 = fp4 - fq2

fp4_start = 2.4
fp4_stop = 3.8

fp2_start = fp4_start - fq1
fp2_stop = fp4_stop - fq1

fp4_delta_span = 0.08

fp4 = np.linspace(fp4_start, fp4_stop, 501)

fig, (ax, ax2) = plt.subplots(2, 2, figsize=(4, 4))

n_harm = 2

colormap = plt.cm.nipy_spectral
colors = colormap(np.linspace(0, 1, 9))

c = 0
for fres in f_res_list:
    for i in np.linspace(-n_harm, n_harm, 2*n_harm+1, dtype=int):
        for j in np.linspace(-n_harm, n_harm, 2*n_harm+1, dtype=int):
            fp4 = np.linspace(fp4_start, fp4_stop, 501)
            fres_ref = fp4 - fq1
            if j != 0:
                fp2 = 1/j * (fres - i*fp4)
                fp2_delta = fp2-fres_ref
                if any(fp2 > fp2_start) and any(fp2 < fp2_stop) and any(abs(fp2_delta) < fp4_delta_span/2):
                    plt.plot(fp2_delta, fp4, color=colors[c], label='{}, {}, {}, {}'.format(f_res_dict[fres][0],
                                                                                            f_res_dict[fres][1],
                                                                                            i, j
                                                                                            ))
                    c = c + 1

            else:
                if i != 0:
                    fp4 = fres/i
                    # print(fp4)
                    if fp4 > fp4_start and fp4 < fp4_stop:
                        # print(fp4)
                        plt.hlines(fp4, -fp4_delta_span/2, fp4_delta_span/2, color=colors[c], label='{}, {}, {}, {}'.format(f_res_dict[fres][0],
                                                                                                                            f_res_dict[
                                                                                                                                fres][1],
                                                                                                                            i, j
                                                                                                                            ))
                        c = c + 1

ax2 = ax.twinx()
ax2.set_ylim(fp2_start, fp2_stop)

# DOUBLE CHECK LINES
# fp2_delta = np.linspace(-0.05, 0.05, 11)
# plt.plot(fp2_delta, fp4)


plt.xlabel(r'$\Delta f_{p2}$ (GHz)')
ax.set_ylabel(r'$f_{p4}$ (GHz)')
ax2.set_ylabel(r'$f_{p2}$ (GHz)')
# lgnd = plt.legend(title='             fq1, fq2, fp4, fp2', bbox_to_anchor=(2., 1), scatterpoints=1, fontsize=10)
# for handle in lgnd.legendHandles:
#     handle._sizes = [100]

# plt.ylim(fp2_start,fp2_stop)
# plt.axis('scaled')
plt.xlim(-fp4_delta_span/2, fp4_delta_span/2)

ax.legend(loc='center left', bbox_to_anchor=(1.3, 0.4))
fig.tight_layout()
plt.show()

# %% Q2 dif: fq2 = fp4 - fq2

fp4_start = 3.7
fp4_stop = 4.4

fp2_start = fp4_start - fq2
fp2_stop = fp4_stop - fq2

fp4_delta_span = 0.08

fp4 = np.linspace(fp4_start, fp4_stop, 501)

fig, ax = plt.subplots(figsize=(4, 4))

n_harm = 3

colormap = plt.cm.nipy_spectral
colors = colormap(np.linspace(0, 1, 10))

c = 0
for fres in f_res_list:
    for i in np.linspace(-n_harm, n_harm, 2*n_harm+1, dtype=int):
        for j in np.linspace(-n_harm, n_harm, 2*n_harm+1, dtype=int):
            fp4 = np.linspace(fp4_start, fp4_stop, 501)
            fres_ref = fp4 - fq2
            if j != 0:
                fp2 = 1/j * (fres - i*fp4)
                fp2_delta = fp2-fres_ref
                if any(fp2 > fp2_start) and any(fp2 < fp2_stop) and any(abs(fp2_delta) < fp4_delta_span/2):
                    plt.plot(fp2_delta, fp4, color=colors[c], label='({}Q1+{}Q2)^({}P2+{}P4)'.format(f_res_dict[fres][0],
                                                                                                     f_res_dict[fres][1],
                                                                                                     j, i
                                                                                                     ))
                    c = c + 1

            else:
                if i != 0:
                    fp4 = fres/i
                    # print(fp4)
                    if fp4 > fp4_start and fp4 < fp4_stop:
                        # print(fp4)
                        plt.hlines(fp4, -fp4_delta_span/2, fp4_delta_span/2, color=colors[c], label='({}Q1+{}Q2)^({}P2+{}P4)'.format(f_res_dict[fres][0],
                                                                                                                                     f_res_dict[
                                                                                                                                         fres][1],
                                                                                                                                     j, i
                                                                                                                                     ))
                        c = c + 1

ax2 = ax.twinx()
ax2.set_ylim(fp2_start, fp2_stop)

plt.xlabel(r'$\Delta f_{p2}$ (GHz)')
ax.set_ylabel(r'$f_{p4}$ (GHz)')
ax2.set_ylabel(r'$f_{p2}$ (GHz)')
# lgnd = plt.legend(title='             fq1, fq2, fp4, fp2', loc='center left', bbox_to_anchor=(1.1, 0.75), scatterpoints=1, fontsize=10)
# for handle in lgnd.legendHandles:
#     handle._sizes = [100]

plt.xlim(-fp4_delta_span/2, fp4_delta_span/2)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1.7, 0.4))
fig.tight_layout()
plt.show()

# %% Q2 sum fq2 = fp4 + fq2

fp4_start = 1
fp4_stop = 1.65

fp2_start = fq2 - fp4_start
fp2_stop = fq2 - fp4_stop

fp2_delta_span = 0.08

fp4 = np.linspace(fp4_start, fp4_stop, 501)

fig, ax = plt.subplots(figsize=(4, 4))

n_harm = 2

colormap = plt.cm.nipy_spectral
colors = colormap(np.linspace(0, 1, 15))

c = 0
for fres in f_res_list:
    for i in np.linspace(-n_harm, n_harm, 2*n_harm+1, dtype=int):
        for j in np.linspace(-n_harm, n_harm, 2*n_harm+1, dtype=int):
            fp4 = np.linspace(fp4_start, fp4_stop, 501)
            fres_ref = fq2 - fp4
            if j != 0:
                fp2 = 1/j * (fres - i*fp4)
                fp2_delta = fp2 - fres_ref
                if fres == fq2 and i+j == 0:
                    print(i, j, fp2_delta[0], fp2_delta[1])

                if any(fp2 > fp2_stop) and any(fp2 < fp2_start) and any(abs(fp2_delta) < fp2_delta_span/2):
                    plt.plot(fp2_delta, fp4, color=colors[c], label='({}Q1+{}Q2)^({}P2+{}P4)'.format(f_res_dict[fres][0],
                                                                                                     f_res_dict[fres][1],
                                                                                                     j, i
                                                                                                     ))
                    c = c + 1

            else:
                if i != 0:
                    fp4 = fres/i
                    # print(fp4)
                    if fp4 > fp4_start and fp4 < fp4_stop:
                        # print(fp4)
                        plt.hlines(fp4, -fp2_delta_span/2, fp2_delta_span/2, color=colors[c], label='({}Q1+{}Q2)^({}P2+{}P4)'.format(f_res_dict[fres][0],
                                                                                                                                     f_res_dict[
                                                                                                                                         fres][1],
                                                                                                                                     j, i
                                                                                                                                     ))
                        c = c + 1

ax2 = ax.twinx()
ax2.set_ylim(fp2_start, fp2_stop)

ax.set_xlabel(r'$\Delta f_{p2}$ (GHz)')
ax.set_ylabel(r'$f_{p4}$ (GHz)')
ax2.set_ylabel(r'$f_{p2}$ (GHz)')
# lgnd = plt.legend(title='             fq1, fq2, fp4, fp2', loc='center left', bbox_to_anchor=(1.1, 0.9), scatterpoints=1, fontsize=12)
# for handle in lgnd.legendHandles:
#     handle._sizes = [100]

# plt.ylim(fp2_start,fp2_stop)
# plt.axis('scaled')
plt.xlim(-fp2_delta_span/2, fp2_delta_span/2)
fig.tight_layout()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1.7, 0.4))
plt.show()
