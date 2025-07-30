# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:45:50 2023

@author: vjohn
"""


# %% Imports
import matplotlib.colors as mcolors
from utils.settings import *
from utils.delft_tools import *
from utils.budapest_tools import *

script_dir = get_script_directory()
sys.path.append(script_dir)

path_alldat = "M:\\tnw\\ns\\qt\\spin-qubits\\data\\stations\\LD400top\\measurements\\SQ20_111\\PSB11_2204"

# %% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_pwr = -5
P4_pwr = 3

fq1, fq1_, fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_pwr, P4_pwr)

fq = [fq1, fq2, fq2]


# %% Load data

start_time = '2022-07-13\\09-41-51'
end_time = start_time
datadir = os.path.join(script_dir, 'measurements')
datfiles, fnames = get_data_from(start_time, end_time, num=1,
                                 rootfolder=path_alldat, only_complete=False)
datfile = datfiles[0]

start_time2 = '2022-07-13\\09-42-47'
end_time2 = start_time2
datfiles2, fnames2 = get_data_from(start_time2, end_time2, num=1,
                                   rootfolder=path_alldat, only_complete=False)
datfile2 = datfiles2[0]

# %%

dates = ['2022-07-11', '2022-07-12', '2022-07-13']
time_sweeps = []

for date in dates:
    path = os.path.join(path_alldat, date)
    files = os.listdir(path)

    for file in files:
        if file[-12:] == 'sweep1D_time':
            time_sweeps.append('{}\\{}'.format(date,
                                               file[:8]))

# %%

mono_sweeps = ['2022-07-13\\15-56-21',
               '2022-07-07\\14-23-08',
               '2022-07-07\\15-08-59',
               '2022-07-07\\15-29-58',
               ]

for start_time in mono_sweeps:
    end_time = start_time  # '2021-06-24\\18-03-01' #'2021-06-10\\18-32-47'
    datfiles, fnames = get_data_from(
        start_time, end_time, num=1, rootfolder=path_alldat, only_complete=False)
    datfile = datfiles[0]
    vp1_vp2 = (round(datfile.metadata['station']['instruments']['gates']['parameters']['vP1']['value'], 2),
               round(datfile.metadata['station']['instruments']['gates']['parameters']['vP2']['value'], 2))

    x = datfile.time_set
    y = datfile.su0
    xlabel = datfile.time_set.label
    ylabel = datfile.su0.label

    freq_dict = get_mw_prop(datfile, ['p2', 'p4'])

    fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(fig_size_double, 3))

    fig.delaxes(ax4)

    # ax.hlines(10, 3.5e4, 3.6e4)

    plot_sequence(datfile, ['vP1', 'vP2'], ['MW_p4', 'MW_p2'],
                  ('vP1', 3), ('vP1', 9),
                  xlim=True,
                  ax=ax,
                  show_inset=False,
                  legend=False,
                  bbox_to_anchor=(-0.1, 1.1),
                  show_plot=False)

    plot_sequence(datfile, ['vP1', 'vP2'], ['MW_p4', 'MW_p2'],
                  ('vP1', 3), ('vP1', 9),
                  xlim=False,
                  ax=ax3,
                  show_inset=False,
                  show_mw_prop=True,
                  legend=True,
                  bbox_to_anchor=(1.1, 0.6),
                  show_plot=False)

    colors = list(mcolors.TABLEAU_COLORS.values())
    m = 4
    color = colors[m]
    m = m + 1
    ax2.plot(x, y, color=color, label='data')

    try:
        p0 = [0.3,  # amplitude
              0.012,  # freq
              0.01,  # alpha
              0.5,  # y0
              -np.pi/2]  # phase

        color = colors[m]
        m = m + 1
        (cov, fit_par), t_rabi = cal_rabi_t(datfile, p0=p0, return_cov=True)
        f_rabi = fit_par[1]*1e3
        f_rabi_std = cov[1]*1e3
        fit_rabi = Rabi(
            np.array(x), fit_par[0], fit_par[1], fit_par[2], fit_par[3], fit_par[4])
        ax2.plot(x, fit_rabi, ls='--', color='black',
                 label=f'fit: f_rabi = {round(f_rabi,2)} $\pm$ {round(f_rabi_std,2)} MHz')
    except:
        pass

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.legend(bbox_to_anchor=(0.53, -0.3))
    # ax2.set_title('Rabi driving with fit at {}'.format(vp1_vp2))

    fig.suptitle(datfile.location)
    plt.show()
