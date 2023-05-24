# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:59:25 2022

@author: TUD278427
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
mixing_regime = ['difference', 'difference', 'sum']

datfile = {}

for start_time in start_time_list:
    end_time = start_time  # '2021-06-24\\18-03-01' #'2021-06-10\\18-32-47'
    datfile[start_time] = load_data(start_time)

# start_time_rabi_q1dif = '2022-07-13\\15-26-45'
# start_time_rabi_q2dif = '2022-07-13\\14-27-14'
# start_time_rabi_q2sum = '2022-07-13\\14-51-17'

# start_time_rabi_list = [start_time_rabi_q1dif, start_time_rabi_q2dif, start_time_rabi_q2sum]
# mixing_regime = ['difference', 'difference', 'sum']
# fp4_fp2 = [(2.6, 1.10064), (4.2, 1.5539), (1.4, 1.2472)]

# datfile_rabi = {}

# for start_time_rabi in start_time_rabi_list:
#     end_time = start_time_rabi #'2021-06-24\\18-03-01' #'2021-06-10\\18-32-47'
#     datfiles, fnames = get_data_from(start_time_rabi, end_time, num = 1, rootfolder=datadir, only_complete = False)
#     datfile_rabi[start_time_rabi] = datfiles


# %% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_pwr = -5
P4_pwr = 3

fq1, fq1_, fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_pwr, P4_pwr)

fq = [fq1/1e9, fq2/1e9, fq2/1e9]

# %% calculate resonances
factors = [0, 1, 2, 3, -1, -2, -3]
factors_pos = [0, 1]

f_res_list = []
f_res_dict = {}

for n in factors_pos:
    for m in factors_pos:
        f_res = n*fq1
        f_res_list.append(f_res)
        f_res_dict[f_res] = [round(n, 2), round(m, 2)]

        f_res = m*fq2
        f_res_list.append(f_res)
        f_res_dict[f_res] = [round(n, 2), round(m, 2)]

        f_res = abs(n*fq1-m*fq2)
        f_res_list.append(f_res)
        f_res_dict[f_res] = [round(n, 2), round(-m, 2)]

f_res = (fq1+fq1_+fq2+fq2_)/2
f_res_list.append(f_res)
f_res_dict[f_res] = [1, 1]

f_res_list.sort()
f_res_list = list(dict.fromkeys(f_res_list))  # remove duplicates


f_res_list = [item for item in f_res_list if item >
              0.8 and item < 5]  # remove small and high values

# %% Plotting
# fig, ax = plt.subplots(1,3, figsize=(6,4))
fig, ax = plt.subplots(1, 2, figsize=(4, 3), sharex=True, sharey=True)
# fig2 = plt.figure(figsize=(1.5, 4))

n = 0
colors = ['#1f77b4', 'purple', 'turquoise', 'lightskyblue']
#
# for start_time in start_time_list[1:]:

start_time = start_time_list[n]


delta = datfile[start_time].delta_set.ndarray[0, :]

mixing = datfile[start_time].mixing_set.ndarray
fp2 = mixing

ax[0].pcolor(delta/1e6, fp2/1e9, datfile[start_time].su0,
             shading='auto', cmap='hot', zorder=0)


for spine in ax[0].spines.values():
    spine.set_edgecolor(colors[n])

if n == 1:
    ax[0].set_ylabel('$f_{P4}$ [GHz]')
if n == 2:
    ax[0].set_ylim(0.8, 1.65)
ax[0].set_xlabel(r'$\Delta f_{P2}$ [MHz]')
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# ax2 = ax[0].twinx()
# ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# fp2_max, fp2_min = ax[0].get_ylim()

# ax2.set_ylim(fp2_max - fq1, fp2_min - fq1)

for axis in ['top', 'bottom', 'left', 'right']:
    ax[0].spines[axis].set_linewidth(2)
for spine in ax2.spines.values():
    spine.set_edgecolor(colors[n])

n = n + 1


fp4_start = fp2.min()/1e9
fp4_stop = fp2.max()/1e9

fp2_start = fp4_start - fq1
fp2_stop = fp4_stop - fq1

fp4_delta_span = 0.08

fp4 = np.linspace(fp4_start, fp4_stop, 501)

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
                    ax[1].plot(fp2_delta*1e3, fp4, color=colors[c], label='{}, {}, {}, {}'.format(f_res_dict[fres][0],
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
                        ax[1].hlines(fp4, -fp4_delta_span/2*1e3, fp4_delta_span/2*1e3, color=colors[c], label='{}, {}, {}, {}'.format(f_res_dict[fres][0],
                                                                                                                                      f_res_dict[
                                                                                                                                          fres][1],
                                                                                                                                      i, j
                                                                                                                                      ))
                        c = c + 1

ax2_2 = ax[1].twinx()
ax2_2.set_ylim(fp2_start, fp2_stop)

ax[1].set_xlabel(r'$\Delta f_{p2}$ (MHz)')
ax[0].set_ylabel(r'$f_{p4}$ (GHz)')
ax2_2.set_ylabel(r'$f_{p2}$ (GHz)')
lgnd = ax[1].legend(title='fq1, fq2, fp4, fp2', loc='center left',
                    bbox_to_anchor=(1.1, 0.75), scatterpoints=1, fontsize=10)
for handle in lgnd.legendHandles:
    handle._sizes = [100]

# plt.ylim(fp2_start,fp2_stop)
# plt.ax[1]is('scaled')
ax[1].set_xlim(-fp4_delta_span/2*1e3, fp4_delta_span/2*1e3)
box = ax[1].get_position()
ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax[1].legend(loc='center left', bbox_to_anchor=(1.6, 0.4))

# plt.subplots_adjust(left=0.1,
#                     bottom=0.1,
#                     right=0.9,
#                     top=0.9,
#                     wspace=0.4,
#                     hspace=0.4)
plt.tight_layout()

plt.subplots_adjust(wspace=0.1, hspace=0)

plt.show()
