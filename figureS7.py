# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 17:14:20 2023

@author: vjohn
"""

# %% Imports
from utils.delft_tools import *
from utils.budapest_tools import *
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

# %% cryo cables

n_start = 100
n_stop = 600

# plt.figure(figsize=(4,3))
fig, ax = plt.subplots(1, 1, figsize=(fig_size_single, 2))

xdata_cryo = np.array(df_wod_cryo['TR1.FREQ.MHZ']/1e3)[n_start:n_stop]
# +15 dB attenuation at RT and 2 lines in series measured
ydata_cryo_dBm = np.array(
    (df_wod_cryo['TR1.TRANSMISSION (2-PORT).DB']+15)/2)[n_start:n_stop]
popt, pcov = curve_fit(att, xdata_cryo, ydata_cryo_dBm)

# correction for difference in cable type
# this data: 0.4 m SCuNi
# experimental data: 0.4 m superconducting
# difference in attenuation is 4.6 and 1.5 dB at 10 and 1 GHz respectively
slope = 0.4*(4.6-1.5)/(10-1)
# attenuation sets in at 1 GHz
# att_cor = 1.5*slope*(xdata_cryo-1) old, why 1.5 ???????
att_cor = slope*(xdata_cryo-1)
ydata_cryo_dBm_cor = ydata_cryo_dBm + att_cor
popt_cor, pcov_cor = curve_fit(att, xdata_cryo, ydata_cryo_dBm_cor)

# ax.plot(xdata_cryo, ydata_cryo_dBm, label='cryo: measured')
ax.plot(xdata_cryo, ydata_cryo_dBm_cor, label='data',
        marker='.', ls=None, markersize=0.1)

ax.plot(xdata_cryo, att(xdata_cryo, *popt_cor),
        label=f'{np.round(popt_cor[0],2)} dB/GHz',
        lw=0.9)

ax.set_xlabel('f (GHz)')
ax.set_ylabel('P (dB)')
ax.legend()
plt.title('Fridge lines with cryo attenuators')
plt.tight_layout()
plt.show()


# %% diplexer

lw = 1

fig, ax3 = plt.subplots(1, 1, figsize=(fig_size_single, 2))

f_span = 0.3

n_start = 130
n_end = 240

ax3.plot(df_dip['TR1.FREQ.MHZ']/1e3,
         df_dip['TR1.TRANSMISSION (2-PORT).DB'],
         label='diplexer', lw=lw)

xdata = df_dip['TR1.FREQ.MHZ'][n_start:n_end]/1e3
ydata_dipl = df_dip['TR1.TRANSMISSION (2-PORT).DB'][n_start:n_end]
popt_cor, pcov_cor = curve_fit(att, xdata, ydata_dipl)

freq = df_dip['TR1.FREQ.MHZ']/1e3
P_att_cryo_0 = - 7.5
P_att_cryo = func_lpf(freq, fth=1.02, slope=118, offset=P_att_cryo_0)
ax3.plot(freq, P_att_cryo,
         label=f'{np.round(popt_cor[0],2)} dB/GHz',
         lw=0.9)  # = {np.round(popt_cor[0],2)} dB/GHz

ax3.set_xlabel('f (GHz)')
ax3.set_ylabel('P (dB)')
ax3.set_ylim(-40, 0)
# ax3.set_xlim(0.6, 5)

plt.title('Diplexer')
plt.tight_layout()
plt.savefig(save_path+'\\figureS7.png', dpi=300)
plt.legend()
plt.show()

# %% Adding up datasets for diplexer and cryo fridge lines

# diplexeer
freq1 = df_dip['TR1.FREQ.MHZ']
power1 = df_dip['TR1.TRANSMISSION (2-PORT).DB']

# cryo fridge lines
cryo_max = df_wod_cryo['TR1.TRANSMISSION (2-PORT).DB'].max()
freq2 = df_wod_cryo['TR1.FREQ.MHZ']
# power at device (in only) is half that power measured (in and out), 15 dB RT attenuation in measurement
power2 = (df_wod_cryo['TR1.TRANSMISSION (2-PORT).DB'] + 15)/2

# Identify the overlapping range
start_freq = max(freq1[0], freq2[0])
end_freq = min(list(freq1)[-1], list(freq2)[-1])

# Find the overlapping indices for each dataset
start_idx1 = np.argmax(freq1 >= start_freq)
end_idx1 = np.argmax(freq1 >= end_freq)
start_idx2 = np.argmax(freq2 >= start_freq)
end_idx2 = np.argmax(freq2 >= end_freq)

# Resample the datasets within the overlapping range
desired_step = (end_freq - start_freq) / (end_idx2 - start_idx2)
resampled_freq = np.linspace(start_freq, end_freq, end_idx2 - start_idx2)
resampled_power1 = np.interp(
    resampled_freq, freq1[start_idx1:end_idx1], power1[start_idx1:end_idx1], left=0, right=0)
resampled_power2 = np.interp(
    resampled_freq, freq2[start_idx2:end_idx2], power2[start_idx2:end_idx2], left=0, right=0)

# Combine the datasets by summing the power values
combined_power = resampled_power1 + resampled_power2


# %% Plotting combined datasets
lw = 1

fig, ax3 = plt.subplots(1, 1, figsize=(fig_size_single, 2))

ax3.plot(resampled_freq/1e3,
         combined_power,
         label=r'$P_{att,meas}$', lw=lw)

offset = -20

hpf = -2.45*resampled_freq/1e3
hpf[hpf > 0] = 0

fcut_lpf = 1.02
lpf = 118*(resampled_freq/1e3 - fcut_lpf)
lpf[lpf > 0] = 0

att_tot = hpf+lpf+offset
ax3.plot(resampled_freq/1e3, att_tot, ls='--', lw=lw,
         label=r'$P_{att,fit}$')

ax3.set_xlabel('f (GHz)')
ax3.set_ylabel(r'$P_{att}$ (dB)')
ax3.set_ylim(-65, -15)
ax3.set_xlim(0.6, 5)

plt.title('Diplexer with cryo fridge lines')
plt.tight_layout()
plt.legend()
plt.savefig(save_path+'\\figureS7.png', dpi=300)
plt.show()

# %% Total attenuation (now including RT attenuation and P_siggen)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_size_single, 2.5))

impedance = 50

xdata = resampled_freq/1e3

p_sig_gen2 = 2.5  # dBm
att_rt2 = -20  # dB
ydata2_dBm = combined_power + p_sig_gen2 + att_rt2
ydata2_mV = dBm2V_conversion(ydata2_dBm)*1e3


ydata3_dBm = att_tot + p_sig_gen2 + att_rt2
ydata3_mV = dBm2V_conversion(ydata3_dBm)*1e3


popt_cor, pcov_cor = curve_fit(ExpDec, xdata, ydata2_mV)

ax1.plot(xdata[:-2], ydata2_dBm[:-2], label='data',
         marker='.', ls=None, markersize=0.1)
ax1.plot(xdata, power(xdata, plunger='P4', P_siggen=2.5),
         label='fit', ls='--', lw='1', marker='.', markersize=0.1)

ax1.set_ylim(-70, -35)

ax2.plot(xdata[:-2], ydata2_mV[:-2], label='data',
         marker='.', ls=None, markersize=0.1)
ax2.plot(xdata, dBm2V_conversion(power(xdata, plunger='P4', P_siggen=2.5))*1e3,
         label='fit', ls='--', lw='1', marker='.', markersize=0.1)


ax1.legend()

ax1.set_xlim(0.6, 5)
ax2.set_xlim(0.6, 5)
# ax2.legend()
ax2.set_xlabel('f (GHz)')
ax2.set_ylabel('$A_{RMS}$ (mV)')
ax1.set_ylabel('$P_{att}$ (dB)')
# plt.title('Amplitude arriving at fridge')
plt.tight_layout()
plt.savefig(save_path+'\\figureS7_fridge_lines_cryo.png', dpi=300)
plt.savefig(save_path+'\\figureS7_fridge_lines_cryo.pdf', dpi=300)

plt.show()

# %% Plotting difference between data and fit

fig, ax3 = plt.subplots(1, 1, figsize=(fig_size_single, 1.5))
dif_dat_fit = ydata2_mV - dBm2V_conversion(power(xdata, plunger='P4',
                                                 P_siggen=2.5, slope_cryo=-3))*1e3
ax3.plot(xdata[:-2], dif_dat_fit[:-2], label='dif',
         marker='.', ls=None, markersize=0.1)


ax3.set_xlabel('f (GHz)')
ax3.set_ylabel(r'$\Delta A_{RMS}$ (mV)')
plt.title('Difference between attenuation data and fit')
plt.tight_layout()
plt.show()

# %% power when driving (according to fit)

pwr_q1_p4 = power(fq1, plunger='P4', P_siggen=2.5)
pwr_q1__p4 = power(fq1_, plunger='P4', P_siggen=2.5)
pwr_q2_p2 = power(fq2, plunger='P2', P_siggen=-6.0)
pwr_q2__p2 = power(fq2_, plunger='P2', P_siggen=-5.8)

amp_q1_p4 = dBm2V_conversion(pwr_q1_p4)*1e3
amp_q1__p4 = dBm2V_conversion(pwr_q1__p4)*1e3
amp_q2_p2 = dBm2V_conversion(pwr_q2_p2)*1e3
amp_q2__p2 = dBm2V_conversion(pwr_q2__p2)*1e3

print(f'Q1(P4): ({"%.2f" %fq1} GHz, ' +
      f'{"%.2f" %pwr_q1_p4[0]} dBm, ' +
      f'{"%.2f" %amp_q1_p4[0]} mV, ' +
      f'{"%.2f" %(2**0.5*amp_q1_p4[0])} mV)')
print(f'Q1_(P4): ({"%.2f" %fq1_} GHz, ' +
      f'{"%.2f" %pwr_q1__p4[0]} dBm, ' +
      f'{"%.2f" %amp_q1__p4[0]} mV, ' +
      f'{"%.2f" %(2**0.5*amp_q1__p4[0])} mV)')
print(f'Q2(P2): ({"%.2f" %fq2} GHz, ' +
      f'{"%.2f" %pwr_q2_p2[0]} dBm, ' +
      f'{"%.2f" %amp_q2_p2[0]} mV, ' +
      f'{"%.2f" %(2**0.5*amp_q2_p2[0])} mV)')
print(f'Q2_(P2): ({"%.2f" %fq2_} GHz, ' +
      f'{"%.2f" %pwr_q2__p2[0]} dBm, ' +
      f'{"%.2f" %amp_q2__p2[0]} mV, ' +
      f'{"%.2f" %(2**0.5*amp_q2__p2[0])} mV)')
