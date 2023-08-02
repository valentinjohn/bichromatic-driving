# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:50:38 2022

@author: Valentin John
@email: v.john@tudelft.nl
"""

# %% Imports
from utils.settings import *
from utils.delft_tools import *

# %% Save path
save_path = get_save_path('Figure1')

# %% Loading CSD data
start_time = '2022-06-29\\22-16-44'
datfile = load_dat(start_time)

# %% Loading reference data set to centralize the CSD according to the measurements performed later

# Loading for the scan along the bichromatiic driving lines of Q1 driven by P4 - P2
start_time_q1dif = '2022-07-11\\13-12-41'
datfile_q1dif = load_dat(start_time_q1dif)

# %% Data manipulation

# absolute values on vP1 and vP2 for CSD measurement
vP1_abs = datfile.metadata['station']['instruments']['gates']['parameters']['vP1']['value']
vP2_abs = datfile.metadata['station']['instruments']['gates']['parameters']['vP2']['value']
# absolute values on vP1 and vP2 for the scans along the bichromatiic driving lines
vP1_abs_ref = datfile_q1dif.metadata['station']['instruments']['gates']['parameters']['vP1']['value']
vP2_abs_ref = datfile_q1dif.metadata['station']['instruments']['gates']['parameters']['vP2']['value']

# relative difference in absolute values of vP1 and vP2 that we have to correct when plotting the charge stability diagram
delta_vP1 = vP1_abs - vP1_abs_ref
delta_vP2 = vP2_abs - vP2_abs_ref

# Corrected measurement data
charge_sensor = datfile.ch2
P1 = datfile.vP1_set[0] + delta_vP1
P2 = np.array(datfile.vP2_set) + delta_vP2

# %% Plunger virtual gate matrix (full)

vg_matrix_set = 'P1P2_set'

vg_names = datfile.metadata['station']['instruments']['hardwareLD400top']['virtual_gates'][vg_matrix_set]['virtual_gate_names']
g_names = datfile.metadata['station']['instruments']['hardwareLD400top']['virtual_gates'][vg_matrix_set]['real_gate_names']

vg_matrix = datfile.metadata['station']['instruments']['hardwareLD400top']['virtual_gates'][vg_matrix_set]['virtual_gate_matrix']
vg_matrix = json.loads(vg_matrix)

df_pl = pd.DataFrame(vg_matrix, columns=g_names, index=vg_names)
df_pl_inv = pd.DataFrame(np.linalg.pinv(df_pl.values),
                         df_pl.columns, df_pl.index)

virtual_gates = ['vP1', 'vP2']
real_gates = ['P1', 'P2', 'P4']

df_pl12 = df_pl.loc[virtual_gates, real_gates]  # .round(3)
df_pl12_inv = df_pl_inv.loc[real_gates, virtual_gates]  # .round(3)

# %% Plotting

# Amplitudes
A_P4 = 2.8*5
A_P2 = 1.9*5

plt.figure(figsize=(2, 2.7))  # 1.7))

# ******************************************************************************
# charge stability diagram
plt.pcolor(P1, P2, charge_sensor, shading='auto',
           cmap='pink',
           rasterized=True)

# ******************************************************************************
# P2 and P4 axis plot
offset = [-10, 10]

plt.quiver(-10, 10, vP1(df_pl12, A_P2, 0), vP2(df_pl12, A_P2, 0),
           angles='xy', scale_units='xy', scale=1, headwidth=8,
           label='P2 axis', color=color_P2)
plt.quiver(-10, 10, -vP1(df_pl12, A_P2, 0), -vP2(df_pl12, A_P2, 0),
           angles='xy', scale_units='xy', scale=1, headwidth=8,
           color=color_P2)

plt.quiver(-10, 10, vP1(df_pl12, 0, A_P4), vP2(df_pl12, 0, A_P4),
           angles='xy', scale_units='xy', scale=1, headwidth=8,
           label='P4 axis', color=color_P4)
plt.quiver(-10, 10, -vP1(df_pl12, 0, A_P4), -vP2(df_pl12, 0, A_P4),
           angles='xy', scale_units='xy', scale=1, headwidth=8,
           color=color_P4)

# ******************************************************************************
# detuning axis plot
plt.plot([50, -50], [-50, 50], ls='-', c='black',
         zorder=1, lw=1, label=r'$ϵ_{12}$')

# ******************************************************************************
# center 1,1 and operation point indication
plt.scatter([0], [0], marker='.', color='black', zorder=2, s=50)
plt.scatter([-10], [10], marker='*', color='white', edgecolor='black', linewidths=0.3,
            zorder=2, label=r'$ϵ_{12}=-20$ mV', s=70)
# ******************************************************************************

xlabel = datfile.vP1_set.label + unit_style(datfile.vP1_set.unit, blank=True)
ylabel = datfile.vP2_set.label + unit_style(datfile.vP2_set.unit, blank=True)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.axis('scaled')
plt.ylim(-27, 27)
plt.xlim(-27, 27)
plt.legend(loc='upper center', bbox_to_anchor=(0.4, -0.3), ncol=2)
plt.tight_layout()

plt.savefig(os.path.join(save_path, 'figure1_csd.png'), dpi=300)
plt.savefig(os.path.join(save_path, 'figure1_csd.pdf'), dpi=300)

plt.show()
