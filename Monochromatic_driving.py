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

fig, ax = plt.subplots(1, 2)

x = datfile.frequency_set
y = datfile.su0
xlabel = datfile.frequency_set.label
ylabel = datfile.su0.label

x2 = datfile2.time_set
y2 = datfile2.su0
x2label = datfile2.time_set.label
y2label = datfile2.su0.label

sig_gen_dict = {'p1': 'sig_gen',
                'p2': 'sig_gen3',
                'p4': 'sig_gen2',
                }


plunger = 'p2'
RF = round(datfile.metadata['station']['instruments']
           [sig_gen_dict[plunger]]['parameters']['frequency']['value'] / 1e9, 3)
LO = round(datfile.metadata['LOs']['MW_{}'.format(plunger)] / 1e9, 3)
IF = abs(LO - RF)
pwr = datfile.metadata['station']['instruments'][sig_gen_dict[plunger]
                                                 ]['parameters']['power']['value']
if LO == 0:
    print('RF is swept by sweeping LO')
else:
    print('RF is swept by sweeping IF')

plunger2 = plunger
RF2 = round(datfile2.metadata['station']['instruments']
            [sig_gen_dict[plunger2]]['parameters']['frequency']['value'] / 1e9, 3)
LO2 = round(datfile2.metadata['LOs']['MW_{}'.format(plunger2)] / 1e9, 3)
IF2 = abs(LO - RF)
pwr2 = datfile2.metadata['station']['instruments'][sig_gen_dict[plunger2]
                                                   ]['parameters']['power']['value']
if LO2 == 0:
    print('RF is swept by sweeping LO')
else:
    print('RF is swept by sweeping IF')

ax[0].plot(x, y, label='(RF, LO, IF) = ({}, {}, {}) GHz'.format(RF, LO, IF))
ax[0].set_xlabel(xlabel)
ax[0].set_ylabel(ylabel)
ax[0].legend()

ax[1].plot(x2, y2, label='(RF, LO, IF) = ({}, {}, {}) GHz'.format(RF2, LO2, IF2))
ax[1].set_xlabel(x2label)
ax[1].set_ylabel(y2label)
ax[1].legend()

fig.tight_layout()
plt.show()

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
sig_gen_dict = {'p1': 'sig_gen',
                'p2': 'sig_gen3',
                'p4': 'sig_gen2'
                }

freq_dict = {'p1': {},
             'p2': {},
             'p4': {}}

mono_sweeps = ['2022-07-13\\08-27-57',
               '2022-07-13\\08-53-49',
               '2022-07-12\\15-05-40',
               '2022-07-13\\08-45-57']

for start_time in mono_sweeps:  # time_sweeps: #

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

    if freq_dict['p2']['pwr'] == -5 and freq_dict['p4']['pwr'] == 3:
        fig, (ax, ax2) = plt.subplots(2, 1)

        ax.plot(x, y, label='data')
        try:
            p0 = [0.3,  # amplitude
                  0.012,  # freq
                  0.01,  # alpha
                  0.5,  # y0
                  -np.pi/2]  # phase

            fit_par, t_rabi = cal_rabi_t(datfile, p0=p0)
            fit_rabi = Rabi(
                np.array(x), fit_par[0], fit_par[1], fit_par[2], fit_par[3], fit_par[4])
            ax.plot(x, fit_rabi, label='fit: t_rabi = {} ns'.format(
                round(t_rabi, 2)))
        except:
            continue
        # ax.plot(x,y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(bbox_to_anchor=(1.1, 0.7))
        ax.set_title('Rabi driving with fit at {}'.format(vp1_vp2))

        for pl in ['p2', 'p4']:
            try:
                mw_start = datfile.metadata['pc0']['MW_{}_pulses'.format(
                    pl)]['p0']['start']
                mw_stop = datfile.metadata['pc0']['MW_{}_pulses'.format(
                    pl)]['p0']['stop']
                mw_duration = round(mw_stop - mw_start, 1)
                mw_amp = datfile.metadata['pc0']['MW_{}_pulses'.format(
                    pl)]['p0']['amplitude']
                mw_freq = freq_dict[pl]['rf']
                mw_time = np.linspace(mw_start, mw_stop, 1001)
                mw = mw_amp*np.sin(mw_time*mw_freq)
                ax2.plot(mw_time, mw, lw=0.1, label='{} {} ns: (RF, LO, IF) = ({}, {}, {}) GHz, PWR = {} dBm'.format(pl, mw_duration,
                                                                                                                     freq_dict[pl]['rf'],
                                                                                                                     freq_dict[pl]['lo'],
                                                                                                                     freq_dict[pl]['if'],
                                                                                                                     freq_dict[pl]['pwr']))
            except:
                continue

        ax2.set_xlabel('time [ns]')
        ax2.set_ylabel('amplitude [mV]')
        ax2.set_title('Pulse sequence')
        ax2.legend(bbox_to_anchor=(1.1, 0.7))

        # plot_sequence(datfile, ['vP1', 'vP2'], ['MW_p4', 'MW_p2'], ('vP1', 3), ('vP1', 9), ax = ax2,
        #               inset_pos = [1.1, 0, 0.5, 1])

        fig.suptitle(datfile.location)
        fig.tight_layout()
        # if round(mw_p2_freq) == round(fq2,2) or round(mw_p2_freq) == round(fq2_) or round(mw_p4_freq) == round(fq1) or round(mw_p4_freq) == round(fq1_):
        plt.show()


# %% Plot sequence with an inset

fig, ax = plot_sequence(datfile, ['vP1', 'vP2'], ['MW_p4', 'MW_p2'],
                        ('vP1', 3), ('vP1', 9),
                        xlim=False,
                        ax=None,
                        show_inset=False,
                        inset_pos=[0, 1.4, 1, 1],
                        legend=True,
                        show_mw_prop=True,
                        figsize=(fig_size_double, 3),
                        bbox_to_anchor=(1.2, 3.2),
                        show_plot=False,)

fig.suptitle(datfile.location)
fig.tight_layout()
plt.show()


# %%

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

    # %%
chan = [1, 2, 3, 4]
awg = [1, 2, 3, 4, 5, 6]
for aw in awg:
    for ch in chan:
        f = datfile.metadata['station']['instruments'][f'AWG{aw}'][
            'parameters'][f'frequency_channel_{ch}']['value']
        print(aw, ch, f)
