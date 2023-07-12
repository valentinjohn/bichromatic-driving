# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 08:22:42 2022

@author: TUD278427
"""

# %% Imports
from utils.settings import *
from utils.delft_tools import *

# %% Save path
save_path = get_save_path('FigureS4')

# %% Load data

start_time = '2022-07-01\\17-07-32'
datfile = load_data(start_time)

# %% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_pwr = -5
P4_pwr = 3

fq1, fq1_, fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_pwr, P4_pwr)
fq = fq1

# %% Plotting
figure_size = 'small'

figsize = {'big': (7.2, 4), 'small': (fig_size_single, 2.5)}

P2_frequency = datfile.sig_gen3_frequency_set.ndarray/1e9
P1_frequency = datfile.sig_gen_frequency_set.ndarray[0, :]/1e9

linestyles = ['-', ':', '--', '-.']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

fig, ax1 = plt.subplots(1, 1, figsize=figsize[figure_size])

vmin = 0
vmax = 0.6
c = ax1.pcolor(P1_frequency, P2_frequency, datfile.su0-datfile.su0.min(),
               shading='auto', cmap='hot', zorder=1, vmin=vmin, vmax=vmax,
               )

ax1.set_ylabel('$f_{\mathrm{P2}}$' +
               f' {unit_style("GHz")}')
ax1.set_xlabel('$f_{\mathrm{P1}}$' +
               f' {unit_style("GHz")}')

ax1.set_ylim(bottom=0)
ax1.set_xlim(left=0)

ymin, ymax = ax1.get_ylim()
xmin, xmax = ax1.get_xlim()
ymax = 4.3e9
xmax = 4.3e9
lw = 1


ax1.axis('square')
ax1.set_ylim(0.9, 3.5)
ax1.set_xlim(0.9, 4.2)

divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(c, cax=cax)
cbar.ax.set_ylabel(r'$1-P_{\downdownarrows}$', rotation=-90, labelpad=15)
cbar.ax.set_yticks([0, 0.2, 0.4, 0.6])

# ax1.plot([-100e-3,fq2-100e-3], [fq2,0], lw = 1, ls=linestyles[2], color='turquoise', zorder=2)
# ax1.plot([+100e-3,fq2+100e-3], [fq2,0], lw = 1, ls=linestyles[2], color='turquoise', zorder=2)
# ax1.plot([fq2-100e-3,2*fq2-100e-3], [0,fq2], lw = 1, ls=linestyles[2], color='violet', zorder=2)
# ax1.plot([fq2+100e-3,2*fq2+100e-3], [0,fq2], lw = 1, ls=linestyles[2], color='violet', zorder=2)
# ax1.plot([fq1-100e-3,4*fq1-100e-3], [0,3*fq1], lw = 1, ls=linestyles[2], color='lightskyblue', zorder=2)
# ax1.plot([fq1+100e-3,4*fq1+100e-3], [0,3*fq1], lw = 1, ls=linestyles[2], color='lightskyblue', zorder=2)

fig.tight_layout()


plt.savefig(save_path+'\\FigureS4.png', dpi=300)
plt.savefig(save_path+'\\FigureS4.pdf', dpi=300)
