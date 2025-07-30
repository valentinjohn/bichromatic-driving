# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:45:40 2023

@author: ZoltanGyorgy
"""

# %% Imports

import __main__
from utils.settings import *
from utils.budapest_tools import *
from utils.delft_tools import *

# %% Save path
save_path = get_save_path('FigureS5')

# %% Import data
attenuation = np.loadtxt("data/attenuation_lovelace_fridge/attenuation.txt")
A_RMS = attenuation[1, :]
A = attenuation[1, :]*np.sqrt(2)
freq = attenuation[0, :]

# %% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_PWR = -5
P4_PWR = 3

fq1, fq1_, fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_PWR, P4_PWR)

# %% Plotting

fig, ax = plt.subplots()
f = np.linspace(min(freq), max(freq), 5000)
ax.plot(freq, A, label='data')
ax.plot(f, Amplitude(f), label='filtered data', linestyle='dashed')
ax.set_xlabel(r'$f$ (GHz)')
ax.set_ylabel(r'$A$ (mV)')
ax.legend()
plt.savefig(os.path.join(save_path, 'FigureS5.pdf'),
            dpi=300, transparent=True)

if hasattr(__main__, '__file__') is False:
    plt.show()

print(f'A(f_Q1)={Amplitude(fq1):.1f}')
print(f'A(f_Q1_)={Amplitude(fq1_):.1f}')
print(f'A(f_Q2)={Amplitude(fq2):.1f}')
print(f'A(f_Q2)={Amplitude(fq2_):.1f}')
