# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 08:22:42 2022

@author: TUD278427
"""

#%% Imports 
from utils.settings import *

#%% Save path
save_path = get_save_path('Figure2')

#%% Load data
show_Rabi = False # 3 different bichromatic Rabi drives with indication on color plot

start_time = '2022-07-12\\17-59-02'
datfile = load_data(start_time)

start_time2 = '2022-07-13\\17-27-20'
datfile2 = load_data(start_time2)

if show_Rabi:
    start_time_rabi_q1dif = '2022-07-13\\15-56-21'
    start_time_rabi_q2dif = '2022-07-13\\14-27-14'
    start_time_rabi_q2sum = '2022-07-13\\14-51-17'
    start_time_rabi_list = [start_time_rabi_q1dif, start_time_rabi_q2dif, start_time_rabi_q2sum]
    
    datfile_rabi = {}
    for start_time_rabi in start_time_rabi_list:
        datfile_rabi[start_time_rabi] = load_data(start_time_rabi)

#%% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_pwr = -5
P4_pwr = 3

fq1, fq1_,fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_pwr, P4_pwr)

#%% Data manipulation

# P2 and P4 frequency axes
P2_frequency = datfile.sig_gen3_frequency_set.ndarray/1e9
P4_frequency = datfile.sig_gen2_frequency_set.ndarray[0,:]/1e9
P2_frequency2 = datfile2.sig_gen3_frequency_set.ndarray/1e9
P4_frequency2 = datfile2.sig_gen2_frequency_set.ndarray[0,:]/1e9

# (1 - downdown)-probabilities
prob = datfile.su0
prob2 = datfile2.su0

# Remove NaN since measurement aborted
P2_frequency2 = P2_frequency2[0:-90]
prob2 = prob2[0:-90]

# level background of two consecutive measurements
problev = prob-prob.min()
problev2 = prob2-prob2.min()

#%% Bichromatic line wrapper

fp4_q2_dif = np.linspace(0, 4.2, 10)
q2_dif_low = Q_dif(fp4_q2_dif, fq2-100e-3)
q2_dif_up = Q_dif(fp4_q2_dif, fq2+100e-3)

fp4_q1_dif = np.linspace(0, 4.2, 10)
q1_dif_low = Q_dif(fp4_q1_dif, fq1-100e-3)
q1_dif_up = Q_dif(fp4_q1_dif, fq1+100e-3)

fp4_q2_sum = np.linspace(0.8, 1.8, 10)
q2_sum_low = Q_sum(fp4_q2_sum, fq2-100e-3)
q2_sum_up = Q_sum(fp4_q2_sum, fq2+100e-3)

#%% Plot settings

figure_size = 'small'
figsize = (fig_size_single ,2.5)
linestyles = ['-', ':', '--', '-.']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors = ['lightskyblue', 'purple', 'turquoise']
vmin = 0
vmax = 0.6
lw = 1

#%% Plotting

fig, ax1, = plt.subplots(1,1,figsize=figsize)
cm = ax1.pcolor(P4_frequency, P2_frequency, problev, shading='auto', cmap='hot', zorder=1, vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cm, fraction=0.046, pad=0.04)

ax1.pcolor(P4_frequency2, P2_frequency2, problev2, shading='auto', cmap='hot', zorder=1, vmin=vmin, vmax=vmax)

ax1.set_ylabel('$f_{\mathrm{P2}}$ [GHz]')
ax1.set_xlabel('$f_{\mathrm{P4}}$ [GHz]')

ax1.set_ylim(bottom=0)
ax1.set_xlim(left=0)

ymin, ymax = ax1.get_ylim()
xmin, xmax = ax1.get_xlim()
ymax = 4.3e9
xmax = 4.3e9
# *****************************************************************************
# Indication of position of 3 different bichromatic Rabi drives
if show_Rabi:
    n = 0
    for start_time_rabi in start_time_rabi_list:
        fp4 = np.round(datfile_rabi[start_time_rabi].metadata['station']['instruments']['sig_gen2']['parameters']['frequency']['value']/1e9, 3)
        fp2 = np.round(datfile_rabi[start_time_rabi].metadata['station']['instruments']['sig_gen3']['parameters']['frequency']['value']/1e9, 3)
        ax1.scatter([fp4],[fp2], s=50, zorder=1, marker='d', color=colors[n], edgecolors='black', clip_on=False)
    n = n + 1

# *****************************************************************************

ax1.axis('square')
ax1.set_ylim(0.9,3.5)
ax1.set_xlim(0.9,4.2)

# *****************************************************************************
# Bichromatic line wrapper plot
ax1.plot(fp4_q2_sum, q2_sum_low, lw = 1, ls=linestyles[2], color='turquoise', zorder=2)
ax1.plot(fp4_q2_sum, q2_sum_up, lw = 1, ls=linestyles[2], color='turquoise', zorder=2)

ax1.plot(fp4_q2_dif, q2_dif_low, lw = 1, ls=linestyles[2], color='violet', zorder=2)
ax1.plot(fp4_q2_dif, q2_dif_up, lw = 1, ls=linestyles[2], color='violet', zorder=2)

ax1.plot(fp4_q1_dif, q1_dif_low, lw = 1, ls=linestyles[2], color='lightskyblue', zorder=2)
ax1.plot(fp4_q1_dif, q1_dif_up, lw = 1, ls=linestyles[2], color='lightskyblue', zorder=2)

# *****************************************************************************

plt.tight_layout()
plt.savefig(os.path.join(save_path, 'figure2a.png'), dpi=300)
plt.savefig(os.path.join(save_path, 'figure2a.pdf'), dpi=300)
plt.show()

#%% Plotting Rabi drives in seperate plot

if show_Rabi:
    colors = ['lightskyblue', 'purple', 'turquoise']
    fig, axs = plt.subplots(3, figsize=(fig_size_single/2, 3))
    
    m = 0
    p0_list = [[0.5, 0.006, 0.1, 0.55, np.pi],
               [0.2, 1/500, 0.2, 0.55, np.pi],
               [0.4, 1/200, 0.12, 0.55, np.pi]]
    
    for start_time_rabi in start_time_rabi_list:
        time = np.array(datfile_rabi[start_time_rabi].time_set)
        prob = np.array(datfile_rabi[start_time_rabi].su0)
        
        fp4 = np.round(datfile_rabi[start_time_rabi].metadata['station']['instruments']['sig_gen2']['parameters']['frequency']['value']/1e9, 3)
        fp2 = np.round(datfile_rabi[start_time_rabi].metadata['station']['instruments']['sig_gen3']['parameters']['frequency']['value']/1e9, 3)
        print((fp4, fp2))
        
        popt, pcov = curve_fit(Rabi, time, prob, p0=p0_list[m])
        perr = np.sqrt(np.diag(pcov))
        crop = 1
    
        axs[m].plot(time, prob, color=colors[m])
        axs[m].plot(time[crop:], Rabi(time, *popt)[crop:], 'black', lw=0.5, 
                label=r'fit: A=%5.3f, f=%5.6f, $\alpha$=%5.3f, $y_0$=%5.3f, , $\phi$=%5.3f' % tuple(popt))
        # print(r'fit: A=%5.3f, f=%5.6f, $\alpha$=%5.4f, $y_0$=%5.3f, , $\phi$=%5.3f' % tuple(popt))
        # print(r'fit std: sigA=%5.3f, sigf=%5.6f, $sig\alpha$=%5.4f, $sigy_0$=%5.3f, , $sig\phi$=%5.3f' % tuple(perr))

        axs[m].set_ylim(0.1,0.9)
        axs[m].set_xlabel('time [ns]')
        axs[m].xaxis.set_major_locator(MaxNLocator(3)) 
        for spine in axs[m].spines.values():
            spine.set_edgecolor(colors[m])
            
        axs[m].set_ylabel(r'$1-P_{\downdownarrows}$')
        m = m + 1
    fig.tight_layout()
    
    if save:
        plt.savefig(os.path.join(save_path, 'figure2b.png'), dpi=300)
        plt.savefig(os.path.join(save_path, 'figure2b.png'), dpi=300)
    fig.tight_layout()
    plt.show()
