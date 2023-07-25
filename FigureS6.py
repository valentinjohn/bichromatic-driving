# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:45:40 2023

@author: ZoltanGyorgy
"""

# %% Imports
from utils.settings import *
from utils.budapest_tools import *
from utils.delft_tools import *

# %% Save path
save_path = get_save_path('FigureS6')

# %% Import data
attenuation=np.loadtxt("attenuation.txt")
A_RMS=attenuation[1,:]
A=attenuation[1,:]*np.sqrt(2)
freq=attenuation[0,:]

# %% Plotting

fig, ax = plt.subplots()
f=np.linspace(min(freq),max(freq),5000)
ax.plot(freq,A,label='data')
ax.plot(f,Amplitude(f),label='filtered data',linestyle='dashed')
ax.set_xlabel(r'$f$ (GHz)')
ax.set_ylabel(r'$A$ (mV)')
ax.legend()
plt.savefig(os.path.join(save_path, 'FigureS6.pdf'),
            dpi=300, transparent=True)