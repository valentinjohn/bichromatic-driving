# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 08:22:42 2022

@author: TUD278427
"""

# %% Imports

from utils.settings import *
from utils.delft_tools import *
from utils.budapest_tools import *
from config import FIGURE_DIR

# %% Save path

save_path = FIGURE_DIR / 'Figure2'
# CREATE SAVE PATH IF NOT EXIST
if not os.path.exists(save_path):
    os.makedirs(save_path)

# %% Load data
show_Rabi = True  # 3 different bichromatic Rabi drives with indication on color plot

print('Loading this file takes a long time, please be patient...')
start_time = '2022-07-12\\17-59-02'
datfile = load_dat(start_time)

print('Loading this file takes a long time, please be patient...')
start_time2 = '2022-07-13\\17-27-20'
datfile2 = load_dat(start_time2)

start_time_rabi_q1dif = '2022-07-13\\15-56-21'
start_time_rabi_q2dif = '2022-07-13\\14-27-14'
start_time_rabi_q2sum = '2022-07-13\\14-51-17'
start_time_rabi_list = [start_time_rabi_q1dif,
                        start_time_rabi_q2dif, start_time_rabi_q2sum]

datfile_rabi = {}
for start_time_rabi in start_time_rabi_list:
    datfile_rabi[start_time_rabi] = load_dat(start_time_rabi)

# %% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_pwr = -5
P4_pwr = 3

fq1, fq1_, fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_pwr, P4_pwr)

# %% Data manipulation

# P2 and P4 frequency axes
P2_frequency = datfile.sig_gen3_frequency_set.ndarray/1e9
P4_frequency = datfile.sig_gen2_frequency_set.ndarray[0, :]/1e9
P2_frequency2 = datfile2.sig_gen3_frequency_set.ndarray/1e9
P4_frequency2 = datfile2.sig_gen2_frequency_set.ndarray[0, :]/1e9

# (1 - downdown)-probabilities
prob = datfile.su0
prob2 = datfile2.su0

# Remove NaN since measurement aborted
P2_frequency2 = P2_frequency2[0:-90]
prob2 = prob2[0:-90]

# level background of two consecutive measurements
problev = prob-prob.min()
problev2 = prob2-prob2.min()

# %% Bichromatic line wrapper

fp4_q2_dif = np.linspace(0, 4.2, 10)
q2_dif_low = Q_dif(fp4_q2_dif, fq2-100e-3)
q2_dif_up = Q_dif(fp4_q2_dif, fq2+100e-3)

fp4_q1_dif = np.linspace(0, 4.2, 10)
q1_dif_low = Q_dif(fp4_q1_dif, fq1-100e-3)
q1_dif_up = Q_dif(fp4_q1_dif, fq1+100e-3)

fp4_q2_sum = np.linspace(0.8, 1.8, 10)
q2_sum_low = Q_sum(fp4_q2_sum, fq2-100e-3)
q2_sum_up = Q_sum(fp4_q2_sum, fq2+100e-3)

# %% Plot settings

figure_size = 'small'
figsize = (1.5*fig_size_single, 4.5)
linestyles = ['-', ':', '--', '-.']
colors = [color_Q1dif, color_Q2dif, color_Q2sum]
vmin = 0
vmax = 0.6
lw = 1

# %% Plotting

fig, [[ax1, ax2], [ax3, ax_empty]], = plt.subplots(2, 2, figsize=figsize,
                                                   sharex=True,
                                                   sharey=True)
ax_empty.axis('off')

cm = ax1.pcolor(P4_frequency, P2_frequency, problev, shading='auto',
                cmap='hot', zorder=1, vmin=vmin, vmax=vmax, rasterized=True)
# cbar = fig.colorbar(cm, fraction=0.03, shrink=1, pad=0.05, ax=ax_empty, location='right')
# cbar.set_label(r'$1 - P_{\downdownarrows}$', rotation=0)

ax1.pcolor(P4_frequency2, P2_frequency2, problev2, shading='auto',
           cmap='hot', zorder=1, vmin=vmin, vmax=vmax, rasterized=True)

ax1.set_ylabel('$f_{\mathrm{P2}}$' +
               f' {unit_style("GHz")}')
ax1.set_xlabel('$f_{\mathrm{P4}}$' +
               f' {unit_style("GHz")}')
ax1.set_ylim(bottom=0)
ax1.set_xlim(left=0)

ymin, ymax = ax1.get_ylim()
xmin, xmax = ax1.get_xlim()
ymax = 4.3e9
xmax = 4.3e9
# *****************************************************************************
# Indication of position of 3 different bichromatic Rabi drives

for n, start_time_rabi in enumerate(start_time_rabi_list[1:]):
    n = n + 1
    fp4 = np.round(datfile_rabi[start_time_rabi].metadata['station']
                   ['instruments']['sig_gen2']['parameters']['frequency']
                   ['value']/1e9, 3)
    fp2 = np.round(datfile_rabi[start_time_rabi].metadata['station']
                   ['instruments']['sig_gen3']['parameters']['frequency']
                   ['value']/1e9, 3)
    ax1.scatter([fp4], [fp2], s=50, zorder=3, marker='d',
                color=colors[n], edgecolors='black', clip_on=False)

# *****************************************************************************

ax1.axis('square')
ax1.set_ylim(0.9, 3.5)
ax1.set_xlim(0.9, 4.2)

# *****************************************************************************
# Bichromatic line wrapper plot
ax1.plot(fp4_q2_sum, q2_sum_low, lw=1, ls=linestyles[2], color=color_Q2sum,
         zorder=2)
ax1.plot(fp4_q2_sum, q2_sum_up, lw=1, ls=linestyles[2], color=color_Q2sum,
         zorder=2)

ax1.plot(fp4_q2_dif, q2_dif_low, lw=1, ls=linestyles[2], color=color_Q2dif,
         zorder=2)
ax1.plot(fp4_q2_dif, q2_dif_up, lw=1, ls=linestyles[2], color=color_Q2dif,
         zorder=2)

ax1.plot(fp4_q1_dif, q1_dif_low, lw=1, ls=linestyles[2],
         color=color_Q1dif, zorder=2)
ax1.plot(fp4_q1_dif, q1_dif_up, lw=1, ls=linestyles[2],
         color=color_Q1dif, zorder=2)

# *****************************************************************************

Q1 = fq1
Q1b = fq1_
Q2 = fq2
Q2b = fq2_

'''
Plotting the monochromatic transitions
    First entry: plunger
    Second entry: frequency
    Third entry: plotting axis
'''
mono(2, Q1, ax2)
mono(2, Q2, ax2)
mono(2, Q1+Q2b, ax2)

mono(4, Q1, ax2)
mono(4, Q2, ax2)
mono(4, Q1+Q2b, ax2)

'''
Plotting the bichromatic transitions
    First entry: plunger P2 factor
    Second entry: plunger P4 factor
    Third entry: (1 -> fq1,
                  2 -> fq2,
                  3 -> fq1+fq2_)
    Fourth entry: plotting axis
'''
bichro(1, 1, 1, ax2)
bichro(1, 1, 2, ax2)
bichro(1, 1, 3, ax2)

bichro(-1, 1, 1, ax2)
bichro(-1, 1, 2, ax2)
bichro(-1, 1, 3, ax2)

bichro(1, -1, 1, ax2)
bichro(1, -1, 2, ax2)
bichro(1, -1, 3, ax2)


# monochromatic transitions, C2*P2+C4*P4=Q_i, driven by P4 or P2
mon4 = [[0, 1, 1], [0, 1, 2], [0, 1, 3]]
mon2 = [[1, 0, 1], [1, 0, 2], [1, 0, 3]]
# similar to the monochromatic transitions
bikro = [[1, 1, 1], [1, 1, 2], [1, 1, 3],
         [1, -1, 1], [1, -1, 2], [1, -1, 3],
         [-1, 1, 1], [-1, 1, 2], [-1, 1, 3]]

# bichromatic, when the sum of the frequencies is resonant
bichro_sum = [[1, 1, 1], [1, 1, 2], [1, 1, 3]]
# bichromatic, when the difference is resonant
bichro_diff = [[1, -1, 1], [1, -1, 2], [1, -1, 3],
               [-1, 1, 1], [-1, 1, 2], [-1, 1, 3]]

# circles around the analysed anticrossings

# intersection point of a monochromatic driven by P2 and a bichromatic
x = (Q(2)-1*Q(1))/1
y = Q(1)
ax2.scatter(y, x, s=68, marker='o', facecolors='none', edgecolors='red',
            zorder=2,)  # draw circles around the analysed anticrossings

# intersection point of a monochromatic driven by P2 and a bichromatic
x = (Q(1)-1*Q(2))/(-1)
y = Q(2)
ax2.scatter(y, x, s=68, marker='o', facecolors='none',
            edgecolors='red', zorder=2,)

# intersection point of a monochromatic driven by P2 and a bichromatic
x = (Q(2)-1*Q(3))/(-1)
y = Q(3)
ax2.scatter(y, x, s=68, marker='o', facecolors='none',
            edgecolors='red', zorder=2,)

m = 20*2/3  # markersize

for i in bichro_sum:  # intersection of a bichromatic with sum and a bichromatic with diff
    for j in bichro_diff:
        C21 = i[0]
        C41 = i[1]
        Qi = Q(i[2])

        C22 = j[0]
        C42 = j[1]
        Qj = Q(j[2])

        # calculates the intersection point
        x = (Qj-C22*Qi/C21)/(C42-C41*C22/C21)
        y = (Qj-C42*x)/C22

for j in mon4:
    for i in bikro:
        # intersection point of a monochromatic driven by P4 and a bichromatic
        x = (Q(i[2])-i[0]*Q(j[2]))/i[1]
        y = Q(j[2])

for j in mon2:
    for i in bikro:
        # intersection point of a monochromatic driven by P2 and a bichromatic
        x = (Q(i[2])-i[1]*Q(j[2]))/i[0]
        y = Q(j[2])
        if i[1] == 1:
            ax2.scatter(y, x, s=m, marker='x', color='blue', zorder=2)

twophoton2 = [1, 2, 3]
twophoton4 = [1, 2, 3]

mon4 = [1, 2, 3]
mon2 = [1, 2, 3]

# intersection point of a monochromatic driven by P2 and a bichromatic
x = (Q(3)-1*Q(1))/1
y = Q(1)
# ax1.scatter(y,x,s=m,marker='x',color='blue',zorder=2,label='strong anticrossing, size$\sim$$\Omega t$')  #we put labels

x = intersection_mono_bi([4, 1], [1, 1, 2])[0]
y = intersection_mono_bi([4, 1], [1, 1, 2])[1]
# ax1.scatter(x,y,s=m,marker='x',color='purple',zorder=3,label='strong anticrossing, size$\sim$$t^2$')    #labels

x = intersection_mono_bi([4, 1], [1, -1, 1])[0]
y = intersection_mono_bi([4, 1], [1, -1, 1])[1]
# ax1.scatter(x,y,s=m,marker='x',color='black',label='crossing',zorder=3)    #labels

C21 = -1
C41 = 1
Qi = Q(1)

C22 = 1
C42 = 1
Qj = Q(3)

x = (Qj-C22*Qi/C21)/(C42-C41*C22/C21)
y = (Qj-C42*x)/C22

ax2.set_ylabel('$f_{\mathrm{P2}}$' +
               f' {unit_style("GHz")}')
ax2.set_xlabel(r'$f_{\mathrm{P4}}$' +
               f' {unit_style("GHz")}')

fontsize_text = 7

ax2.text(1.1, 2, r'$\mathrm{Q1^{P4}}$', fontsize=fontsize_text)
ax2.text(2.23, 2.95, r'$\mathrm{Q2^{P4}}$', fontsize=fontsize_text)

ax2.text(1.65, 1.55, r'$\mathrm{Q1^{P2}}$', fontsize=fontsize_text)
ax2.text(2.8, 2.7, r'$\mathrm{Q2^{P2}}$', fontsize=fontsize_text)

ax2.text(0.85, 0.99, r'$\mathrm{Q2^{P2,P4}}$',
         fontsize=fontsize_text, rotation=-45)
ax2.text(1.47, 1.49, r'$\mathrm{(Q1+Q2\_)^{P2,P4}}$',
         fontsize=fontsize_text, rotation=-45)

ax2.text(3.0, 1.65, r'$\mathrm{Q1^{-P2,P4}}$',
         fontsize=fontsize_text, rotation=45)

ax2.text(3.2, 3.25, r'$\mathrm{(Q1+Q2\_)^{P4}}$', fontsize=fontsize_text)
ax2.text(3.42, 0.92, r'$\mathrm{Q2^{-P2,P4}}$',
         fontsize=fontsize_text, rotation=45)
# ax2.arrow(3.3, 1.16, 0.25, -0.08,
#          head_width = 0.07,
#          width = 0.01,
#          ec ='black')


# these are the trichromatic processes, without the specification of Qi
trikro_l = [[1, 2], [2, 1], [-1, 2], [2, -1], [1, -2], [-2, 1]]
trikro_l = np.array(trikro_l)
trikro = []
for i in trikro_l:
    for j in range(1, 4):
        cucc = []
        cucc = i
        cucc = np.append(cucc, j)
        # print(cucc)
        # this adds the Qi term, so we will get 18 transitions, every element of trikro_l will have 3
        trikro.append(cucc)
        # different Qi values

bichro2(1, 1, 1, ax3)  # plotting the bichromatic transitions
bichro2(1, 1, 2, ax3)
bichro2(1, 1, 3, ax3)

bichro2(-1, 1, 1, ax3)
bichro2(-1, 1, 2, ax3)
bichro2(-1, 1, 3, ax3)

bichro2(1, -1, 1, ax3)
bichro2(1, -1, 2, ax3)
bichro2(1, -1, 3, ax3)

m = 18*2/3  # markersize

for i in trikro:  # we plot all of the trichromatic transitions
    trichro(i, ax3)

bikro = [[1, 1, 1], [1, 1, 2], [1, 1, 3],
         [1, -1, 1], [1, -1, 2], [1, -1, 3],
         [-1, 1, 1], [-1, 1, 2], [-1, 1, 3]]

# there are six different kind of resonance lines

# bichromatic, when the sum of the frequencies is resonant
bichro_sum = [[1, 1, 1], [1, 1, 2], [1, 1, 3]]
bichro_diff = [[1, -1, 1], [1, -1, 2], [1, -1, 3],
               [-1, 1, 1], [-1, 1, 2], [-1, 1, 3]]  # bichromatic, when the difference is resonant
trichro_big = [[1, -2, 2], [1, -2, 1], [-1, 2, 1], [-1, 2, 2],
               [-1, 2, 3]]  # trichromatic transitions with a slope of 2
trichro_small = [[2, -1, 3], [2, -1, 2], [2, -1, 1], [-2, 1, 1],
                 [-2, 1, 2]]  # trichromatic transitions with a slope of 1/2
trichro_neg1 = [[1, 2, 3]]  # trichromatic transitions with a slope of -2
trichro_neg2 = [[2, 1, 3]]  # trichromatic transitions with a slope of -1/2

# when we are interested only in the intersection of bicchromatic 2photon with bichro 3photon
lines = [bikro, trikro]

# using these numbers we generate all possible pairs of transition groups to intersect
numbers = [0, 1]
pairs = []  # these are in the pairs
for i in range(len(numbers)):
    for j in range(i+1, len(numbers)):
        pairs.append([numbers[i], numbers[j]])

for i in pairs:
    set1 = lines[i[0]]  # from the pairs we choose two sets of transitions
    set2 = lines[i[1]]
    for j in set1:  # we take two elements, intersect
        for k in set2:
            anti = anticrossing(j, k)  # see if it is an anticrossing
            x = intersection(j, k)[0]  # calculate intersection points
            y = intersection(j, k)[1]

            if anti[0] == 1:
                if anti[1] == 2:
                    if anti[3] == 2:
                        # second order anticrossing driven by P2
                        ax3.scatter(x, y, s=m, marker='x',
                                    color='green', zorder=2)
                    # if anti[3] == 4:
                    #     # second order anticrossing driven by P4
                    #     ax3.scatter(x, y, s=m, marker='x',
                    #                 color='green', zorder=2)
                    if anti[3] == 0:
                        # second order anticrossing driven by P4
                        ax3.scatter(x, y, s=m, marker='x',
                                    color='red', zorder=2)
                if anti[1] == 1:  # first order anticrossing, driven by P2
                    if anti[3] == 2:
                        if anti[2] == 'Ot':
                            strong_ac = ax3.scatter(x, y, s=m, marker='x',
                                                    color='blue', zorder=2)
                        if anti[2] == 't2':
                            ax3.scatter(x, y, s=m, marker='x',
                                        color='blue', zorder=2)
                    # else:  # first order anticrossing, driven by P4
                    #     if anti[2] == 'Ot':
                    #         weak_ac = ax3.scatter(x, y, s=m, marker='x',
                    #                               color='green', zorder=2)
                        # if anti[2] == 't2':
                        #     ax3.scatter(x, y, s=m, marker='x',
                        #                 color='green', zorder=2)


x = intersection([2, 1, 3], [1, 1, 2])[0]
y = intersection([2, 1, 3], [1, 1, 2])[1]
ax3.scatter(x, y, s=68, marker='o', facecolors='none', edgecolors='red',
            zorder=2)  # draw circles around the analysed anticrossings

x = intersection([-1, 1, 2], [-2, 1, 1])[0]
y = intersection([-1, 1, 2], [-2, 1, 1])[1]
ax3.scatter(x, y, s=68, marker='o', facecolors='none', edgecolors='red',
            zorder=2)  # draw circles around the analysed anticrossings

x = intersection([-1, 2, 1], [1, -1, 1])[0]
y = intersection([-1, 2, 1], [1, -1, 1])[1]

x = intersection([2, 1, 3], [1, 1, 2])[0]
y = intersection([2, 1, 3], [1, 1, 2])[1]
ax3.scatter(x, y, s=m, marker='x', color='blue',
            zorder=2, label='strong anticrossing')

x = intersection([-1, 1, 2], [-2, 1, 1])[0]
y = intersection([-1, 1, 2], [-2, 1, 1])[1]

# x = intersection([1, -1, 2], [1, -2, 1])[0]
# y = intersection([1, -1, 2], [1, -2, 1])[1]
# ax3.scatter(x, y, s=m, marker='x', color='green',
#             zorder=2, label='weak anticrossing')

ax3.set_xlabel(r'$f_{\mathrm{P4}}$' +
               f' {unit_style("GHz")}')
ax3.set_ylabel('$f_{\mathrm{P2}}$' +
               f' {unit_style("GHz")}')

ax3.arrow(1.75, 0.55, 0, 0.6,
          head_width=0.07,
          width=0.01,
          ec='black', fc='black', clip_on=False)

ax3.arrow(4.15, 0.55, 0, 0.67,
          head_width=0.07,
          width=0.01,
          ec='black', fc='black', clip_on=False)


ax3.text(0.8, 0.32, r'$\mathrm{(Q1+Q2\_)^{2P2,P4}}$', fontsize=fontsize_text)
# ax3.text(0.35, 1.7, r'$\mathrm{Q2^{P2,P4}}$', fontsize=fontsize_text)
ax3.text(3.7, 1.25, r'$\mathrm{Q2^{-P2,P4}}$',
         fontsize=fontsize_text, rotation=45)
ax3.text(3.7, 0.32, r'$\mathrm{Q1^{-2P2,P4}}$',
         fontsize=fontsize_text, rotation=0)

ax2.axis('square')
ax3.axis('square')

ax2.set_ylim(0.9, 3.5)
ax2.set_xlim(0.9, 4.3)
ax3.set_ylim(0.9, 3.5)
ax3.set_xlim(0.9, 4.3)

# ax2.legend(loc='upper left', bbox_to_anchor=(-0.13, 1.2),
#            fancybox=True, shadow=False, ncol=2)

ax1.tick_params(axis='y', pad=0.5)
ax2.tick_params(axis='y', pad=0.5)
ax3.tick_params(axis='y', pad=0.5)

ax1.set_yticks([1, 2, 3])
ax2.set_yticks([1, 2, 3])
ax1.xaxis.set_tick_params(labelbottom=True)
ax2.xaxis.set_tick_params(labelbottom=True)
ax2.yaxis.set_tick_params(labelbottom=True)

ax2.text(1.65, 1.25, r'$\mathrm{AC1}$', fontsize=fontsize_text)
ax2.text(2.2, 1.2, r'$\mathrm{AC2}$', fontsize=fontsize_text)
ax2.text(3.75, 1.6, r'$\mathrm{AC3}$', fontsize=fontsize_text)

ax3.text(0.9, 1.3, r'$\mathrm{AC4}$', fontsize=fontsize_text)
ax3.text(3.35, 1.2, r'$\mathrm{AC5}$', fontsize=fontsize_text)


# *****************************************************************************
# ax2.legend([strong_ac, weak_ac], ['strong AC', 'weak AC'], ncol=2,
#            loc='upper left', bbox_to_anchor=(0.1, 1.2))

plt.tight_layout()
plt.subplots_adjust(left=0.1,
                    bottom=0.05,
                    right=0.97,
                    top=0.95,
                    hspace=-0.05,
                    wspace=0.3)
plt.savefig(os.path.join(save_path, 'figure2_plots.png'), dpi=300)
plt.savefig(os.path.join(save_path, 'figure2_plots.pdf'), dpi=300)
if hasattr(__main__, '__file__') is False:
    plt.show()


# %%


# define custom colorbars
# Blues = mpl.cm.get_cmap('Blues', 256)
# New_Blues = ListedColormap(Blues(np.linspace(0.0, 0.90, int(256*0.90))))
# Reds = mpl.cm.get_cmap('Reds', 256)
# New_Reds = ListedColormap(Reds(np.linspace(0.0, 0.90, int(256*0.90))))

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
plt.savefig(os.path.join(save_path, 'figure2_cbar.pdf'),
            dpi=300, transparent=True)
if hasattr(__main__, '__file__') is False:
    plt.show()


# %% Plotting Rabi drives in seperate plot
save = True
fig, axs = plt.subplots(2, 1, figsize=(
    0.38*fig_size_single, 2.06), sharey=True)

p0_list = [[0.5, 0.006, 0.1, 0.55, np.pi],
           [0.2, 1/500, 0.2, 0.55, np.pi],
           [0.4, 1/200, 0.12, 0.55, np.pi]]

for n, start_time_rabi in enumerate(start_time_rabi_list[1:]):
    m = n + 1
    time = np.array(datfile_rabi[start_time_rabi].time_set)
    prob = np.array(datfile_rabi[start_time_rabi].su0)

    fp4 = np.round(datfile_rabi[start_time_rabi].metadata['station']
                   ['instruments']['sig_gen2']['parameters']['frequency']
                   ['value']/1e9, 3)
    fp2 = np.round(datfile_rabi[start_time_rabi].metadata['station']
                   ['instruments']['sig_gen3']['parameters']['frequency']
                   ['value']/1e9, 3)
    print((fp4, fp2))

    popt, pcov = curve_fit(Rabi, time, prob, p0=p0_list[m])
    perr = np.sqrt(np.diag(pcov))
    crop = 1

    axs[n].plot(time, prob, color=colors[m])
    axs[n].plot(time[crop:], Rabi(time, *popt)[crop:], 'black', lw=0.5,
                label=r'fit: A=%5.3f, f=%5.6f, $\alpha$=%5.3f, $y_0$=%5.3f, , $\phi$=%5.3f' % tuple(popt))
    # print(r'fit: A=%5.3f, f=%5.6f, $\alpha$=%5.4f, $y_0$=%5.3f, , $\phi$=%5.3f' % tuple(popt))
    # print(r'fit std: sigA=%5.3f, sigf=%5.6f, $sig\alpha$=%5.4f, $sigy_0$=%5.3f, , $sig\phi$=%5.3f' % tuple(perr))

    axs[n].set_ylim(0.1, 0.9)
    axs[n].set_xlabel(f'time {unit_style("ns")}')
    axs[n].xaxis.set_major_locator(MaxNLocator(2))
    # axs[n].tick_params(axis='x', labelrotation=45)
    for spine in axs[n].spines.values():
        spine.set_edgecolor(colors[m])

    axs[n].set_ylabel(r'$1-P_{\downdownarrows}$')
fig.tight_layout()

if save:
    plt.savefig(os.path.join(save_path, 'figure2b.pdf'),
                dpi=300, transparent=True)
fig.tight_layout()
plt.show()
