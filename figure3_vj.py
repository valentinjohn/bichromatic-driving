# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:59:25 2022

@author: TUD278427
"""

# %% Imports
from utils.settings import *
from utils.delft_tools import *
from utils.budapest_tools import *

# %% Save path
save_path = get_save_path('Figure3')

# %% load data
start_time_q1dif = '2022-07-11\\13-12-41'
start_time_q2dif = '2022-07-12\\14-28-39'
start_time_q2sum = '2022-07-12\\15-39-11'

start_time_list = [start_time_q1dif, start_time_q2dif, start_time_q2sum]
mixing_regime = ['difference', 'difference', 'sum']

datfile = {}
for start_time in start_time_list:
    datfile[start_time] = load_dat(start_time)


# %% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_pwr = -5
P4_pwr = 3

fq1, fq1_, fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_pwr, P4_pwr)

fq = [fq1, fq2, fq2]

# %% Plotting

fig1 = plt.figure(figsize=(fig_size_double, 3.6))
gs1 = GridSpec(nrows=2, ncols=5, wspace=0.7)


vmin = 0.15
vmax = 0.9

n = 1
ax = []
colors = ['#1f77b4', 'violet', 'turquoise', 'lightskyblue']

ax_ticks = [[3.9, 4.2],
            [1.0, 1.3, 1.6]]
ax2_ticks = [[1.1, 1.4, 1.7],
             [1.2, 1.7]]

ax1 = fig1.add_subplot(gs1[:, 0])
ax2 = fig1.add_subplot(gs1[:, 2])
ax3 = fig1.add_subplot(gs1[:, 1])
ax4 = fig1.add_subplot(gs1[:, 3])
ax7 = fig1.add_subplot(gs1[0, 4])
ax8 = fig1.add_subplot(gs1[1, 4])
axs = [ax1, ax2]

for idx, start_time in enumerate(start_time_list[1:]):
    ax = axs[idx]

    delta = datfile[start_time].delta_set.ndarray[0, :]
    mixing = datfile[start_time].mixing_set.ndarray

    p4_LO = datfile[start_time].metadata['LOs']['MW_p4'] / 1e9
    p2_LO = datfile[start_time].metadata['LOs']['MW_p2'] / 1e9

    p4_freq = (datfile[start_time].metadata['station']['instruments']
               ['sig_gen2']['parameters']['frequency']['value']) / 1e9
    p2_freq = (datfile[start_time].metadata['station']['instruments']
               ['sig_gen3']['parameters']['frequency']['value']) / 1e9

    fp2 = mixing

    cm = ax.pcolor(delta/1e6, fp2/1e9,
                   datfile[start_time].su0, shading='auto', cmap='hot', zorder=0,
                   vmin=vmin, vmax=vmax)

    for spine in ax.spines.values():
        spine.set_edgecolor(colors[n])

    if n == 1:
        ax.set_ylabel('$f_{\mathrm{P4}}$' +
                      f' {unit_style("GHz")}',
                      labelpad=-10)
    if n == 2:
        ax.set_ylim(0.8, 1.65)
    ax.set_xlabel(r'$\Delta f_{\mathrm{P2}}$' +
                  f' {unit_style("MHz")}')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax.set_yticks(ax_ticks[n-1])

    ax2 = ax.twinx()
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax2.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax2.set_yticks(ax2_ticks[n-1])
    fp2_max, fp2_min = ax.get_ylim()
    if mixing_regime[n] == 'difference':
        ax2.set_ylim(fp2_max - fq[n], fp2_min - fq[n])
    elif mixing_regime[n] == 'sum':
        ax2.set_ylim(fq[n] - fp2_max,
                     fq[n] - fp2_min)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    for spine in ax2.spines.values():
        spine.set_edgecolor(colors[n])

    n = n + 1

n = 1

x = np.linspace(-40*10**6, 38*10**6)
y1 = get_fq1(-10)*10**9-x
y2 = get_fq2(-10)*10**9/2-x

y3 = get_fq1(-10)*10**9+x
y4 = get_fq2(-10)*10**9/2+x

s = 0

ax3.set_ylim(3.7, 4.395)
ax3.set_xlim(-40, 40)

ax4.set_ylim(0.8, 1.65)
ax4.set_xlim(-39, 39)

ax3.set_xlabel(r'$\Delta f_{\mathrm{P2}}$' +
               f' {unit_style("MHz")}')
ax4.set_xlabel(r'$\Delta f_{\mathrm{P2}}$' +
               f' {unit_style("MHz")}')

ax5 = ax3.twinx()
ax6 = ax4.twinx()

ax3.yaxis.set_major_locator(plt.MaxNLocator(2))
ax4.yaxis.set_major_locator(plt.MaxNLocator(2))
ax5.yaxis.set_major_locator(plt.MaxNLocator(2))
ax6.yaxis.set_major_locator(plt.MaxNLocator(2))

ax6.set_ylabel('$f_{\mathrm{P2}}$' +
               f' {unit_style("GHz")}',
               labelpad=-10)

ax5.set_ylim(3.7-fq2, 4.395-fq2)
ax6.set_ylim(fq2-0.8, fq2-1.65)

for spine in ax5.spines.values():
    spine.set_linewidth(2)
    spine.set_edgecolor('violet')

for spine in ax6.spines.values():
    spine.set_linewidth(2)
    spine.set_edgecolor('turquoise')

x = np.linspace(-40*10**6, 40*10**6)
y1 = get_fq1(-10)*10**9-x
y2 = get_fq2(-10)*10**9/2-x

ax6.plot(x*10**(-6), y1*10**(-9), color='black', linewidth=0.6)
ax6.plot(x*10**(-6), y2*10**(-9), color='black', linewidth=0.6)

y3 = get_fq1(-10)*10**9+x
y4 = get_fq2(-10)*10**9/2+x

ax5.plot(x*10**(-6), y1*10**(-9), color='black', linewidth=0.6)
ax5.plot(x*10**(-6), y2*10**(-9), color='black', linewidth=0.6)


b0_AC5 = 0.0010476570790074575
b1_AC5 = -3.795398142141938

t_avg=18.090348165326574
O_avg=14.278008900473601

b0 = b0_AC5
b1 = b1_AC5

y_u=np.linspace(3.7,3.97,5000)
y_d=np.linspace(3.7,3.92,5000)

ax3.plot(x_curve_AC5_up(y_u,b0_AC5,b1_AC5,t_avg)*1000,y_u, color='black', linewidth=0.6)  # lower
ax3.plot(x_curve_AC5_down(y_d,b0_AC5,b1_AC5,t_avg)*1000,y_d, color='black', linewidth=0.6)

x0_AC3 = -6.72780021*10**(-4)
y0_AC3 = 4.23352197
tO_AC3=t_avg*O_avg

y1=np.linspace(3.97,4.232,5000)
y2=np.linspace(4.235,4.4,5000)
ax3.plot(x_curve_AC3(y1, x0_AC3, y0_AC3, tO_AC3)*1000,y1,color='black', linewidth=0.6)  # uppper
ax3.plot(x_curve_AC3(y2, x0_AC3, y0_AC3, tO_AC3)*1000,y2, color='black', linewidth=0.6)
ax3.set_title(r'$\mathrm{Q2^{-P2,P4}}$')

x0_AC1 = -5.68320293*10**(-4)
y0_AC1 = 1.50340012

y1=np.linspace(1.145,1.5,5000)
y2=np.linspace(1.506,1.7,5000)
ax4.plot(x_curve_AC1(y1, x0_AC1, y0_AC1, t_avg)*1000,y1,color='black', linewidth=0.6)  # upper
ax4.plot(x_curve_AC1(y2, x0_AC1, y0_AC1, t_avg)*1000,y2,color='black', linewidth=0.6)  # lower
ax4.set_title(r'$\mathrm{Q2^{P2,P4}}$')

b0_AC4 = 0.0030126932276232947
b1_AC4 = -1.0636075643177132
tO_AC4=t_avg*O_avg

y_u=np.linspace(0.8,1.146,5000)
y_d=np.linspace(0.8,1.15,5000)

ax4.plot(x_curve_AC4_up(y_u, b0_AC4, b1_AC4, tO_AC4)*1000,y_u,color='black', linewidth=0.6)  # lower
ax4.plot(x_curve_AC4_down(y_d, b0_AC4, b1_AC4, tO_AC4)*1000,y_d,color='black', linewidth=0.6)  # lower

fontsize_label = 8
fontsize_label_AC = 8

# ax4.text(-7, 0.88, '$\mathrm{Q2^{P2,P4}}$',
#          rotation=85, fontsize=fontsize_label)
# ax4.text(15, 1.17, '$\mathrm{Q1^{P2}}$', fontsize=fontsize_label)
# ax4.text(15, 1.355, '$\mathrm{Q2^{2P2}}$', fontsize=fontsize_label)
# ax4.text(-36, 1.49, '$\mathrm{Q1^{P4}}$', fontsize=fontsize_label)
# ax4.text(-39, 0.96,
#          '$(\mathrm{Q1}+\mathrm{Q2}\_)^{\mathrm{2P2,P4}}$', rotation=33.69, fontsize=fontsize_label)

# ax3.text(-9, 4, '$\mathrm{Q2^{-P2,P4}}$', rotation=85, fontsize=fontsize_label)
# ax3.text(13, 4.135, '$\mathrm{Q1^{P2}}$', fontsize=fontsize_label)
# ax3.text(13, 3.955, '$\mathrm{Q2^{2P2}}$', fontsize=fontsize_label)
# ax3.text(
#     0, 4.24, '$(\mathrm{Q1}+\mathrm{Q2}\_)^{\mathrm{P4}}$', fontsize=fontsize_label)
# ax3.text(-39, 3.83, '$\mathrm{Q1^{-2P2,P4}}$',
#          rotation=-39, fontsize=fontsize_label)

# ax4.text(-2, 1.48, '$\mathrm{AC_1}$', fontsize=fontsize_label_AC, color='red')
# ax4.text(0, 1.09, '$\mathrm{AC_4}$', fontsize=fontsize_label_AC, color='red')

# ax3.text(-11, 4.222, '$\mathrm{AC_3}$',
#          fontsize=fontsize_label_AC, color='red')
# ax3.text(-7, 3.8, '$\mathrm{AC_5}$', fontsize=fontsize_label_AC, color='red')

axs = [ax7, ax8]
for n in range(2):
    axs[n].hlines(0, -5, -1, color='black')
    axs[n].hlines(0, 5, 1, color='black')
    axs[n].hlines(fq1, -5, -1, color='black')
    axs[n].hlines(fq1, 5, 1, color='black')
    axs[n].hlines(fq2, -5, -1, color='black')
    axs[n].hlines(fq2, 5, 1, color='black')
    axs[n].hlines(fq1+fq2_, -5, -1, color='black')
    axs[n].hlines(fq1+fq2_, 5, 1, color='black')
    axs[n].axes.get_xaxis().set_ticks([])
    axs[n].axes.get_yaxis().set_ticks([])
    axs[n].set_ylim(0-0.5, fq1+fq2_+1)

for spine in axs[0].spines.values():
    spine.set_linewidth(2)
    spine.set_edgecolor('violet')

for spine in axs[1].spines.values():
    spine.set_linewidth(2)
    spine.set_edgecolor('turquoise')
# axs[0].set_ylabel('AC1')
# axs[1].set_ylabel('AC3')
# axs[2].set_ylabel('AC4')
# axs[3].set_ylabel('AC5')

# arrow = patches.FancyArrowPatch((0, 0), (0, fq1), arrowstyle='<|-|>',
#                         mutation_scale=20)
# ax.add_patch(arrow)

ax3.set_yticks([3.9, 4.2])
ax4.set_yticks([1.0, 1.3, 1.6])
ax5.set_yticks([1.1, 1.4, 1.7])
ax6.set_yticks([1.7, 1.2])

plt.subplots_adjust(left=0.07,
                    bottom=0.12,
                    right=0.93,
                    top=0.94,
                    hspace=0)

fig1.tight_layout()
gs1.tight_layout(fig1)
plt.savefig(os.path.join(save_path, 'figure3a.png'), dpi=300)
plt.savefig(os.path.join(save_path, 'figure3a.pdf'), dpi=300)
# plt.savefig(os.path.join(save_path, 'figure3a.svg'), dpi=300)
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
plt.savefig(os.path.join(save_path, 'figure3_cbar.pdf'),
            dpi=300, transparent=True)
plt.show()
