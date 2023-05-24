# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:19:10 2023

@author: vjohn
"""

#%% Imports 
from utils.settings import *

#%% Loading CSD data
start_time = '2022-06-29\\22-16-44'
datfile = load_data(start_time)

#%% Detuning virtual gate matrix (full)

vg_matrix_set = 'EU1234_sep'

vg_names = datfile.metadata['station']['instruments']['hardwareLD400top']['virtual_gates'][vg_matrix_set ]['virtual_gate_names']
g_names = datfile.metadata['station']['instruments']['hardwareLD400top']['virtual_gates'][vg_matrix_set ]['real_gate_names']

vg_matrix = datfile.metadata['station']['instruments']['hardwareLD400top']['virtual_gates'][vg_matrix_set ]['virtual_gate_matrix']
vg_matrix = json.loads(vg_matrix)

df_det = pd.DataFrame(vg_matrix, columns=g_names, index=vg_names)
df_det_inv = pd.DataFrame(np.linalg.pinv(df_det.values), df_det.columns, df_det.index)

#%% Detuning virtual gate matrix (only e and U vs. P1 and P2)

virtual_gates = ['e12', 'U12']
real_gates = ['P1', 'P2']

df_det12 = df_det.loc[virtual_gates, real_gates].round(3)
df_det12_inv = df_det_inv.loc[real_gates, virtual_gates].round(3)

#%% Plunger virtual gate matrix (full)

vg_matrix_set = 'P1P2_set'

vg_names = datfile.metadata['station']['instruments']['hardwareLD400top']['virtual_gates'][vg_matrix_set ]['virtual_gate_names']
g_names = datfile.metadata['station']['instruments']['hardwareLD400top']['virtual_gates'][vg_matrix_set ]['real_gate_names']

vg_matrix = datfile.metadata['station']['instruments']['hardwareLD400top']['virtual_gates'][vg_matrix_set ]['virtual_gate_matrix']
vg_matrix = json.loads(vg_matrix)

df_pl = pd.DataFrame(vg_matrix, columns=g_names, index=vg_names)
df_pl_inv = pd.DataFrame(np.linalg.pinv(df_pl.values), df_pl.columns, df_pl.index)

#%% Plunger virtual gate matrix (only vP1 and vP2 vs. P1 and P2)

virtual_gates = ['vP1', 'vP2']
real_gates = ['P1', 'P2']

df_pl12 = df_pl.loc[virtual_gates, real_gates].round(3)
df_pl12_inv = df_pl_inv.loc[real_gates, virtual_gates].round(3)

#%% Detuning vs vP virtual gate matrix

df_det12_pl12 = df_det12.dot(df_pl12_inv)