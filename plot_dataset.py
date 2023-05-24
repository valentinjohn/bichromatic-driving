# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 08:34:11 2022

@author: vjohn
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
from projects.notebook_tools.notebook_tools import get_data_from, Gauss
import pickle
import os
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.patches as patches

#%% path
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
script_dir = 'M:\\tnw\\ns\\qt\\spin-qubits\\data\\stations\\LD400top\\measurements\\SQ20_111\\PSB11_2204'

save_path = os.path.join(script_dir, 'Figures')
fig_size_single = 3.37
fig_size_double = 6.69

#%% defining style
plt.rcParams.update({'font.size': 8})
plt.rcParams['font.sans-serif']= 'Arial'
plt.rcParams["figure.figsize"] = (fig_size_single, 3)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.default'] = 'it' #'regular'

plt.rcParams['legend.frameon']= False
plt.rcParams['legend.fontsize']= 'small'
plt.rcParams['legend.scatterpoints']= 1
plt.rcParams['axes.labelpad'] = 4 #-2

#%% Load data
datfile = {}

start_time_1 = '2022-07-07\\15-37-56'
start_time_2 = '2022-07-07\\15-43-46'
start_time_3 = '2022-07-07\\16-14-18'
start_time_4 = '2022-07-07\\16-15-38'

end_time_1 = start_time_1 #'2021-06-24\\18-03-01' #'2021-06-10\\18-32-47'
datadir = os.path.join(script_dir)
datfiles_fq1, fnames_fq1 = get_data_from(start_time_1, end_time_1, num = 1, rootfolder=datadir, only_complete = False) 
datfile[start_time_1] = datfiles_fq1[0]

end_time_2 = start_time_2 #'2021-06-24\\18-03-01' #'2021-06-10\\18-32-47'
datadir = os.path.join(script_dir)
datfiles_fq1_, fnames_fq1_ = get_data_from(start_time_2, end_time_2, num = 1, rootfolder=datadir, only_complete = False) 
datfile[start_time_2] = datfiles_fq1_[0]

end_time_3 = start_time_3 #'2)
datfiles_fq2, fnames_fq2 = get_data_from(start_time_3, end_time_3, num = 1, rootfolder=datadir, only_complete = False) 
datfile[start_time_3] = datfiles_fq2[0]

end_time_4 = start_time_4
datadir = os.path.join(script_dir)
datfiles_fq2_, fnames_fq2_ = get_data_from(start_time_4, end_time_4, num = 1, rootfolder=datadir, only_complete = False) 
datfile[start_time_4] = datfiles_fq2_[0]

fp4 = datfile[start_time_4].metadata['station']['instruments']['sig_gen2']['parameters']['frequency']['value']/1e9
fp2 = datfile[start_time_4].metadata['station']['instruments']['sig_gen3']['parameters']['frequency']['value']/1e9
pp4 = datfile[start_time_4].metadata['station']['instruments']['sig_gen2']['parameters']['power']['value']
pp2 = datfile[start_time_4].metadata['station']['instruments']['sig_gen3']['parameters']['power']['value']

#%% Plotting
fig, axs = plt.subplots(2,2, sharey='row')


axs[0,0].plot(datfile[start_time_1].time_set,
            datfile[start_time_1].su0)
axs[0,0].set_ylabel(r'$\downdownarrows$-probability')
axs[0,0].set_ylim(0.1, 0.95)
axs[0,0].axes.xaxis.set_ticklabels([])

axs[1,0].plot(datfile[start_time_2].time_set,
             datfile[start_time_2].su0)
axs[1,0].set_xlabel(datfile[start_time_2].time_set.label)
axs[1,0].set_ylabel(r'$\downdownarrows$-probability')
axs[1,0].set_ylim(0.1, 0.95)

axs[0,1].plot(datfile[start_time_3].time_set,
            datfile[start_time_3].su0)
axs[0,1].axes.xaxis.set_ticklabels([])

axs[1,1].plot(datfile[start_time_4].time_set,
              datfile[start_time_4].su0)
axs[1,1].set_xlabel(datfile[start_time_4].time_set.label)

plt.tight_layout()
# plt.savefig(save_path+'\\figureS1_exchange.png', dpi=300)

plt.show()
