# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:14:30 2023

@author: vjohn
"""

# %% Imports
from utils.notebook_tools import ExpDec
from scipy.optimize import curve_fit
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

name = 'a12_rf_line.csv'
df_wod = pd.read_csv(path+name, comment='!')

name = 'attenuation_vaughan_cryo.csv'
df_wod_cryo = pd.read_csv(path2+name, comment='!')

# %% fit line attenuation dBm


def att(f, a, c):
    return a*f + c


def dBm2V_conversion(data_dBm, Z=50):
    V = (Z/1e3)**0.5 * 10**(np.array(data_dBm)/20)
    return V

# n_start = 0
# n_stop = -1

# xdata = np.array(df_wod['TR1.FREQ.MHZ']/1e3)[n_start:n_stop]
# ydata = np.array(df_wod['TR1.TRANSMISSION (2-PORT).DB'])[n_start:n_stop]
# popt, pcov = curve_fit(att, xdata, ydata)
# plt.plot(xdata, ydata, label='RT: measured')
# plt.plot(xdata, att(xdata, *popt), label=f'RT: slope = {np.round(popt[0],2)} dB/GHz, offset = {np.round(popt[1],2)} dB')


n_start = 100
n_stop = 600

# plt.figure(figsize=(4,3))
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(fig_size_single, 2))

xdata = np.array(df_wod_cryo['TR1.FREQ.MHZ']/1e3)[n_start:n_stop]
# +15 dB attenuation at RT and 2 lines in series measured
ydata_dBm = np.array(
    (df_wod_cryo['TR1.TRANSMISSION (2-PORT).DB']+15)/2)[n_start:n_stop]
popt, pcov = curve_fit(att, xdata, ydata_dBm)

slope = 0.4*(4.6-1.5)/(10-1)
att_cor = 1.5*slope*(xdata-1)
ydata_dBm_cor = ydata_dBm + att_cor
popt_cor, pcov_cor = curve_fit(att, xdata, ydata_dBm_cor)

# plt.plot(xdata, ydata_dBm, label='cryo: measured')
ax.plot(xdata, ydata_dBm_cor, label='data',
        marker='.', ls=None, markersize=0.1)
# plt.plot(xdata, att(xdata, *popt), label=f'cryo: slope = {np.round(popt[0],2)} dB/GHz, offset = {np.round(popt[1],2)} dB')
ax.plot(xdata, att(xdata, *popt_cor), label=f'{np.round(popt_cor[0],2)} dB/GHz',
        lw=0.9)  # = {np.round(popt_cor[0],2)} dB/GHz

ax.set_xlabel('f (GHz)')
ax.set_ylabel('P (dBm)')
# ax.title('Attenuation of fridge lines in Vaughan fridge')
ax.legend()
# plt.tight_layout()
# plt.show()


n_start = 0
n_stop = -1

p_sig_gen2 = 3  # dBm
att_rt2 = -20  # dB

# p_sig_gen3 = -5 #dBm
# att_rt3 = -12 #dB

# xdata = np.array(df_wod['TR1.FREQ.MHZ']/1e3)[n_start:n_stop]
ydata2_dBm = p_sig_gen2+att_rt2+ydata_dBm
ydata2_dBm_cor = p_sig_gen2+att_rt2+ydata_dBm_cor


# ydata3_dBm = p_sig_gen3+att_rt3+ydata
impedance = 50
ydata2_mV = dBm2V_conversion(ydata2_dBm)*1e3
ydata2_mV_cor = dBm2V_conversion(ydata2_dBm_cor)*1e3
# ydata3_mV = dBm2V_conversion(ydata3_dBm)*1e3
# popt, pcov = curve_fit(ExpDec, xdata, ydata2_mV)
# plt.plot(xdata, ydata2_mV, label='measured')
# plt.plot(xdata, ExpDec(xdata, *popt), label=f'decay = {np.round(1/popt[1],2)} 1/GHz')

popt_cor, pcov_cor = curve_fit(ExpDec, xdata, ydata2_mV_cor)
ax2.plot(xdata, ydata2_mV_cor, label='data',
         marker='.', ls=None, markersize=0.1)
# A * np.exp(-x / m) + y0
ax2.plot(xdata, ExpDec(xdata, *popt_cor), label=f'fit: att_mV',
         lw=0.9)  # = {np.round(np.exp(-1/popt_cor[1]),2)}/GHz

mV_fq1 = ExpDec(fq1, *popt_cor)
mV_fq1_ = ExpDec(fq1_, *popt_cor)
mV_fq2 = ExpDec(fq2, *popt_cor)
mV_fq2_ = ExpDec(fq2_, *popt_cor)

# = {np.round(mV_fq1,2)} mV
ax2.scatter([fq1], [mV_fq1], label=r'$A_{fq1}$', marker='+')
# = {np.round(mV_fq1_,2)} mV
ax2.scatter([fq1_], [mV_fq1_], label=r'$A_{fq1\_}$', marker='+')
# = {np.round(mV_fq2,2)} mV
ax2.scatter([fq2], [mV_fq2], label=r'$A_{fq2}$', marker='+')
# = {np.round(mV_fq2_,2)} mV
ax2.scatter([fq2_], [mV_fq2_], label=r'$A_{fq2\_}$', marker='+')

ax2.legend()
ax2.set_xlabel('f (GHz)')
ax2.set_ylabel('Amplitude (mV)')
# plt.title('Amplitude arriving at fridge')
plt.tight_layout()
plt.savefig(save_path+'\\figureS7_fridge_lines_cryo.png', dpi=300)

plt.show()
# plt.ylim(0, 3)

# n_start = 0
# n_stop = 700

# xdata = np.array(df_wod_cryo['TR1.FREQ.MHZ']/1e3)[n_start:n_stop]
# ydata = np.array(df_wod_cryo['TR1.TRANSMISSION (2-PORT).DB']+15)[n_start:n_stop]
# popt, pcov = curve_fit(att, xdata, ydata)
# plt.plot(xdata, ydata, label='cryo: measured')
# plt.plot(xdata, att(xdata, *popt), label=f'cryo: slope = {np.round(popt[0],2)} dB/GHz, offset = {np.round(popt[1],2)} dB')

# plt.xlabel('f (GHz)')
# plt.ylabel('P (dBm)')
# plt.title('Attenuation of fridge lines in Vaughan fridge')
# plt.legend()
# plt.tight_layout()
# plt.show()
