# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:59:25 2022

@author: TUD278427
"""

#%% Imports 
from utils.settings import *

#%% Save path
save_path = get_save_path('Figure3')

#%% load data
start_time_q1dif = '2022-07-11\\13-12-41'
start_time_q2dif = '2022-07-12\\14-28-39'
start_time_q2sum = '2022-07-12\\15-39-11'

start_time_list = [start_time_q1dif, start_time_q2dif, start_time_q2sum]
mixing_regime = ['difference', 'difference', 'sum']

datfile = {}
for start_time in start_time_list:
    datfile[start_time] = load_data(start_time)


#%% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_pwr = -5
P4_pwr = 3

fq1, fq1_,fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_pwr, P4_pwr)

fq = [fq1, fq2, fq2]

#%% Plotting
# fig, ax = plt.subplots(1,3, figsize=(6,4))
fig1 = plt.figure(figsize=(fig_size_single, 3))
# fig2 = plt.figure(figsize=(1.5, 4))

gs1 = GridSpec(nrows=1, ncols=2)
# gs2 = GridSpec(nrows=3, ncols=1)

n = 1
ax = []
colors = ['#1f77b4', 'purple', 'turquoise', 'lightskyblue']

for start_time in start_time_list[1:]:
    ax = fig1.add_subplot(gs1[0,n-1])
    
    delta = datfile[start_time].delta_set.ndarray[0,:]
    
    mixing = datfile[start_time].mixing_set.ndarray
    
    p4_LO = datfile[start_time].metadata['LOs']['MW_p4'] / 1e9
    p2_LO = datfile[start_time].metadata['LOs']['MW_p2'] / 1e9
    print(p4_LO, p2_LO)
    p4_freq = datfile[start_time].metadata['station']['instruments']['sig_gen2']['parameters']['frequency']['value'] /1e9
    p2_freq = datfile[start_time].metadata['station']['instruments']['sig_gen3']['parameters']['frequency']['value'] /1e9
    print(p4_freq, p2_freq)
    
    fp2 = mixing
    # if mixing_regime[n] == 'difference':
    #     fp4 = abs(fp2 - fq[n])
    #     ax.scatter([(fq[n]-(fp4_fp2[n][0]-fp4_fp2[n][1]))*1e3],[fp4_fp2[n][0]], s=50, zorder=1, marker='d', color=colors[n], edgecolors='black')
    # elif mixing_regime[n] == 'sum':
    #     fp4 = abs(fq[n] - fp2)
    #     ax.scatter([(fq[n]-(fp4_fp2[n][0]+fp4_fp2[n][1]))*1e3],[fp4_fp2[n][0]], s=50, zorder=1, marker='d', color=colors[n], edgecolors='black')
    #     ax.set_ylim(min(fp4)/1e9, 1.65)
    
    
    ax.pcolor(delta/1e6, fp2/1e9, datfile[start_time].su0, shading='auto', cmap='hot', zorder=0)
    
    
    for spine in ax.spines.values():
        spine.set_edgecolor(colors[n])
                            
    if n == 1:
        ax.set_ylabel('$f_{P4}$ [GHz]')
    if n == 2:
        ax.set_ylim(0.8, 1.65)
    ax.set_xlabel(r'$\Delta f_{P2}$ [MHz]')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    ax2 = ax.twinx()
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    fp2_max, fp2_min = ax.get_ylim()
    if mixing_regime[n] == 'difference':
        ax2.set_ylim(fp2_max - fq[n], fp2_min - fq[n])
    elif mixing_regime[n] == 'sum':
        ax2.set_ylim(fq[n] - fp2_max,
                     fq[n] - fp2_min)
    if n == 2:
        ax2.set_ylabel('$f_{P2}$ [GHz]')
   
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    for spine in ax2.spines.values():
        spine.set_edgecolor(colors[n]) 
   
    n = n + 1

# m = 1
# p0_list = [[0.5, 0.004, 0.55, 0.577, np.pi],
#            [0.2, 1/500, 0.2, 0.55, np.pi],
#            [0.4, 1/200, 0.12, 0.55, np.pi]]

# for start_time_rabi in start_time_rabi_list[1:]:
#     ax = fig1.add_subplot(gs1[1,m-1])
    
#     time = np.array(datfile_rabi[start_time_rabi][0].time_set)
#     prob = np.array(datfile_rabi[start_time_rabi][0].su0)
    
#     if m == 0:
#         popt, pcov = curve_fit(Rabi, time[5:100], prob[5:100], p0=p0_list[m])
#         crop = 5
#         # popt = [0.5, 0.004, 0.6, 0.577, np.pi]
#     else:
#         popt, pcov = curve_fit(Rabi, time, prob, p0=p0_list[m])
#         crop = 1
    
#     ax.plot(time, prob, color=colors[m])
#     ax.plot(time[crop:], Rabi(time, *popt)[crop:], 'black', lw=1, 
#             label=r'fit: A=%5.3f, f=%5.3f, $\alpha$=%5.3f, $y_0$=%5.3f, , $\phi$=%5.3f' % tuple(popt))
#     print(r'fit: A=%5.3f, f=%5.3f, $\alpha$=%5.3f, $y_0$=%5.3f, , $\phi$=%5.3f' % tuple(popt))
#     # ax.legend()
    
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#     ax.set_ylim(0.1,0.9)
#     ax.set_xlabel('time [ns]')
#     ax.xaxis.set_major_locator(MaxNLocator(3)) 
#     for spine in ax.spines.values():
#         spine.set_edgecolor(colors[m])
        
#     if m == 0:
#         ax.set_ylabel(r'$\upuparrows$-probability')
#     m = m + 1
fig1.tight_layout()
gs1.tight_layout(fig1)
plt.savefig(os.path.join(save_path, 'figure3a.png'), dpi=300)
plt.savefig(os.path.join(save_path, 'figure3a.pdf'), dpi=300)
# plt.savefig(os.path.join(save_path, 'figure3a.svg'), dpi=300)
# fig2.tight_layout(h_pad=-0.1)
plt.show()

