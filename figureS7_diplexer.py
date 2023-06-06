# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 17:14:20 2023

@author: vjohn
"""

# %% Imports
from utils.settings import *

# %% Save path
save_path = get_save_path('FigureS7')

# %% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_pwr = -5
P4_pwr = 3

fq1, fq1_, fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_pwr, P4_pwr)

mixing_regime = 'difference'
fq = fq1

# %% import

path = './data/2023-01-20/'
path2 = './data/2023-02-24/'

# %% Load data
name = 'a12_rf_line_w_diplexer.csv'
df_wd = pd.read_csv(path+name, comment='!')

name = 'diplexer2.csv'
df_dip = pd.read_csv(path+name, comment='!')

name = 'a12_rf_line.csv'
df_wod = pd.read_csv(path+name, comment='!')

name = 'attenuation_vaughan_cryo.csv'
df_wod_cryo = pd.read_csv(path2+name, comment='!')

name = 'cable.csv'
df_cab = pd.read_csv(path+name, comment='!')

# %%
plt.figure(figsize=(6, 3))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 1, 2)

f_span = 0.3


ax1.plot(df_wd['TR1.FREQ.MHZ']/1e3, df_wd['TR1.TRANSMISSION (2-PORT).DB'],
         label='signal arriving at device')
ax1.plot(df_dip['TR1.FREQ.MHZ']/1e3,
         df_dip['TR1.TRANSMISSION (2-PORT).DB']-20, label='diplexer alone')
ax1.set_ylim(-31, -27)
f_centre = (fq1+fq1_)/2
ax1.set_xlim(f_centre-f_span/2, f_centre+f_span/2)
ax1.set_xlabel('f (GHz)')
ax1.set_ylabel('P (dBm)')
ax1.vlines(fq1, -40, -20, color='black', ls='--', label='fq1')
ax1.vlines(fq1_, -40, -20, color='black', ls='-.', label='fq1_')
# ax1.set_title('Diplexer dependance in Vaughan fridge')


ax2.plot(df_wd['TR1.FREQ.MHZ']/1e3, df_wd['TR1.TRANSMISSION (2-PORT).DB'],
         label='signal arriving at device')
ax2.plot(df_dip['TR1.FREQ.MHZ']/1e3,
         df_dip['TR1.TRANSMISSION (2-PORT).DB']-22.5, label='diplexer alone')
ax2.set_ylim(-31, -27)
f_centre = (fq2+fq2_)/2
ax2.set_xlim(f_centre-f_span/2, f_centre+f_span/2)
ax2.set_xlabel('f (GHz)')
ax2.set_ylabel('P (dBm)')
ax2.vlines(fq2, -40, -20, color='purple', ls='--', label='fq2')
ax2.vlines(fq2_, -40, -20, color='purple', ls='-.', label='fq2_')
# ax2.set_title('Diplexer dependance in Vaughan fridge')

ax3.plot(df_wd['TR1.FREQ.MHZ']/1e3, df_wd['TR1.TRANSMISSION (2-PORT).DB'],
         label='signal arriving at device')
ax3.plot(df_dip['TR1.FREQ.MHZ']/1e3,
         df_dip['TR1.TRANSMISSION (2-PORT).DB']-22.5, label='diplexer alone')
ax3.vlines(fq1, -60, -20, color='black', ls='--', label='fq1')
ax3.vlines(fq1_, -60, -20, color='black', ls='-.', label='fq1_')
ax3.vlines(fq2, -60, -20, color='purple', ls='--', label='fq2')
ax3.vlines(fq2_, -60, -20, color='purple', ls='-.', label='fq2_')
ax3.set_xlabel('f (GHz)')
ax3.set_ylabel('P (dBm)')

plt.tight_layout()
plt.legend(bbox_to_anchor=(1.3, 1.6))
plt.show()

# %%
lw = 1

fig, axs = plt.subplot_mosaic("AA;BC", figsize=(fig_size_single, 2.5))

ax1 = axs['B']
ax2 = axs['C']
ax3 = axs['A']


f_span = 0.3


ax1.plot(df_wd['TR1.FREQ.MHZ']/1e3, df_wd['TR1.TRANSMISSION (2-PORT).DB'] +
         7, label='diplexer + fridge lines', lw=lw)
ax1.plot(df_dip['TR1.FREQ.MHZ']/1e3,
         df_dip['TR1.TRANSMISSION (2-PORT).DB']-13, label='diplexer alone', lw=lw)
ax1.plot(df_wod['TR1.FREQ.MHZ']/1e3,
         df_wod['TR1.TRANSMISSION (2-PORT).DB'], label='fridge lines', lw=lw)
ax1.plot(df_wod_cryo['TR1.FREQ.MHZ']/1e3, df_wod_cryo['TR1.TRANSMISSION (2-PORT).DB'] +
         25, label='fridge lines at cryo T', lw=lw)
# ax1.plot(df_cab['TR1.FREQ.MHZ']/1e3, df_cab['TR1.TRANSMISSION (2-PORT).DB']-20, label='cable', lw = lw)
ax1.set_ylim(-24, -20)
f_centre = (fq1+fq1_)/2
ax1.set_xlim(f_centre-f_span/2, f_centre+f_span/2)
ax1.set_xlabel('f (GHz)')
ax1.set_ylabel('P (dBm)')
ax1.vlines(fq1, -40, -20, color='black', ls='--', lw=lw, label='fq1')
ax1.vlines(fq1_, -40, -20, color='black', ls='-.', lw=lw, label='fq1_')
# ax1.set_title('Diplexer dependance in Vaughan fridge')


ax2.plot(df_wd['TR1.FREQ.MHZ']/1e3, df_wd['TR1.TRANSMISSION (2-PORT).DB'] +
         3.5, label='diplexer + fridge lines', lw=lw)
ax2.plot(df_dip['TR1.FREQ.MHZ']/1e3, df_dip['TR1.TRANSMISSION (2-PORT).DB'] -
         19.5, label='diplexer alone', lw=lw)
ax2.plot(df_wod['TR1.FREQ.MHZ']/1e3,
         df_wod['TR1.TRANSMISSION (2-PORT).DB'], label='fridge lines', lw=lw)
ax2.plot(df_wod_cryo['TR1.FREQ.MHZ']/1e3, df_wod_cryo['TR1.TRANSMISSION (2-PORT).DB'] +
         26.5, label='fridge lines at cryo T', lw=lw)
# ax2.plot(df_cab['TR1.FREQ.MHZ']/1e3, df_cab['TR1.TRANSMISSION (2-PORT).DB']-22.5, label='cable', lw = lw)
ax2.set_ylim(-28, -24)
f_centre = (fq2+fq2_)/2
ax2.set_xlim(f_centre-f_span/2, f_centre+f_span/2)
ax2.set_xlabel('f (GHz)')
ax2.set_ylabel('P (dBm)')
ax2.vlines(fq2, -40, -20, color='purple', ls='--', lw=lw, label='fq2')
ax2.vlines(fq2_, -40, -20, color='purple', ls='-.', lw=lw, label='fq2_')
# ax2.set_title('Diplexer dependance in Vaughan fridge')

ax3.plot(df_wd['TR1.FREQ.MHZ']/1e3, df_wd['TR1.TRANSMISSION (2-PORT).DB'] +
         7, label='diplexer + fridge lines', lw=lw)
ax3.plot(df_dip['TR1.FREQ.MHZ']/1e3,
         df_dip['TR1.TRANSMISSION (2-PORT).DB']-13, label='diplexer', lw=lw)
ax3.plot(df_wod['TR1.FREQ.MHZ']/1e3,
         df_wod['TR1.TRANSMISSION (2-PORT).DB'], label='fridge lines', lw=lw)
ax3.plot(df_wod_cryo['TR1.FREQ.MHZ']/1e3, df_wod_cryo['TR1.TRANSMISSION (2-PORT).DB'] +
         15, label='fridge lines at cryo T', lw=lw)
# ax3.plot(df_cab['TR1.FREQ.MHZ']/1e3, df_cab['TR1.TRANSMISSION (2-PORT).DB']-22.5, label='measurement cable', lw = lw)
ax3.vlines(fq1, -60, -15, color='black', ls='--', lw=lw, label='fq1')
ax3.vlines(fq1_, -60, -15, color='black', ls='-.', lw=lw, label='fq1_')
ax3.vlines(fq2, -60, -15, color='purple', ls='--', lw=lw, label='fq2')
ax3.vlines(fq2_, -60, -15, color='purple', ls='-.', lw=lw, label='fq2_')
ax3.set_xlabel('f (GHz)')
ax3.set_ylabel('P (dBm)')
ax3.set_ylim(-55, -15)
ax3.set_xlim(0.6, 5)

plt.tight_layout()
ax3.legend(bbox_to_anchor=(0.3, 1.3))
plt.savefig(save_path+'\\figureS7_diplexer.png', dpi=300)

plt.show()
