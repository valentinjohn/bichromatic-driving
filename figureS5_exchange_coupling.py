# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 09:01:25 2022

@author: vjohn
"""

# %% Imports
from utils.settings import *
from utils.delft_tools import *

# %% Save path
save_path = get_save_path('FigureS5')

# %% Load data
datfile = {}

START_TIME_FQ1 = '2022-07-13\\09-47-07'
START_TIME_FQ1_ = '2022-07-13\\09-48-32'
START_TIME_FQ2 = '2022-07-13\\09-52-36'
START_TIME_FQ2_ = '2022-07-13\\09-53-42'

datfile['fq1'] = load_data(START_TIME_FQ1)
datfile['fq1_'] = load_data(START_TIME_FQ1_)
datfile['fq2'] = load_data(START_TIME_FQ2)
datfile['fq2_'] = load_data(START_TIME_FQ2_)

# %% Fitting

parameters, covariance = curve_fit(Gauss,
                                   np.array(datfile['fq1'].frequency_set),
                                   np.array(datfile['fq1'].su0),
                                   p0=[0.6, 2e6, 1.52e9, 0.2])
fq1_amp = parameters[0]
fq1_sigma = parameters[1]
fq1 = parameters[2]
fq1_yoffset = parameters[3]
fit_fq1 = Gauss(datfile['fq1'].frequency_set,
                fq1_amp, fq1_sigma, fq1, fq1_yoffset)

parameters, covariance = curve_fit(Gauss,
                                   np.array(datfile['fq1_'].frequency_set),
                                   np.array(datfile['fq1_'].su0),
                                   p0=[0.6, 2e6, 1.57e9, 0.2])
fq1__amp = parameters[0]
fq1__sigma = parameters[1]
fq1_ = parameters[2]
fq1__yoffset = parameters[3]
fit_fq1_ = Gauss(datfile['fq1'].frequency_set,
                 fq1__amp, fq1__sigma, fq1_, fq1__yoffset)

parameters, covariance = curve_fit(Gauss,
                                   np.array(datfile['fq2'].frequency_set),
                                   np.array(datfile['fq2'].su0),
                                   p0=[0.6, 2e6, 2.65e9, 0.2])
fq2_amp = parameters[0]
fq2_sigma = parameters[1]
fq2 = parameters[2]
fq2_yoffset = parameters[3]
fit_fq2 = Gauss(datfile['fq2'].frequency_set,
                fq2_amp, fq2_sigma, fq2, fq2_yoffset)

parameters, covariance = curve_fit(Gauss,
                                   np.array(datfile['fq2_'].frequency_set),
                                   np.array(datfile['fq2_'].su0),
                                   p0=[0.6, 2e6, 2.71e9, 0.2])
fq2__amp = parameters[0]
fq2__sigma = parameters[1]
fq2_ = parameters[2]
fq2__yoffset = parameters[3]
fit_fq2_ = Gauss(datfile['fq2'].frequency_set,
                 fq2__amp, fq2__sigma, fq2_, fq2__yoffset)

exchange_q1 = round((fq1_ - fq1)/1e6)
# exchange_q1_sigma = (fq1_sigma + fq1__sigma)/1e6

exchange_q2 = round((fq2_ - fq2)/1e6)
# exchange_q2_sigma = (fq2_sigma + fq2__sigma)/1e6

UNIT_CONVERSION = 1e-9
fq1 = fq1*UNIT_CONVERSION
fq1_ = fq1_*UNIT_CONVERSION
fq2 = fq2*UNIT_CONVERSION
fq2_ = fq2_*UNIT_CONVERSION

# %% Plotting
UNIT = 'GHz'
unit_dict = {'meV': 1e6, 'GHz': 1e-9}
# FREQ2ENERGY = hbar/e*unit_dict[UNIT]
FREQ2ENERGY = unit_dict[UNIT]

DD = 0
UD = fq1
DU = fq2
UU = (fq1 + fq2)

STYLE = "Simple, tail_width=0.5, head_width=4, head_length=8"
kw = dict(arrowSTYLE=STYLE)

fig, axs = plt.subplots(4, 2, sharey='row',
                        gridspec_kw={'height_ratios': [1, 4, 1, 4]})

fq1_freq_scan = np.array(datfile['fq1'].frequency_set)*UNIT_CONVERSION
fq2_freq_scan = np.array(datfile['fq2'].frequency_set)*UNIT_CONVERSION

axs[0, 0].set_xticks([])
axs[0, 0].set_yticks([])
axs[0, 1].set_xticks([])
axs[0, 1].set_yticks([])
axs[2, 0].set_xticks([])
axs[2, 0].set_yticks([])
axs[2, 1].set_xticks([])
axs[2, 1].set_yticks([])

axs[0, 0].set_xlim([0, 100])
axs[0, 0].set_xlim([0, 100])
axs[0, 1].set_xlim([0, 100])
axs[0, 1].set_xlim([0, 100])
axs[2, 0].set_xlim([0, 100])
axs[2, 0].set_xlim([0, 100])
axs[2, 1].set_xlim([0, 100])
axs[2, 1].set_xlim([0, 100])

axs[0, 0].set_ylim([-1.5, 1.5])
axs[0, 0].set_ylim([-1.5, 1.5])
axs[0, 1].set_ylim([-1.5, 1.5])
axs[0, 1].set_ylim([-1.5, 1.5])
axs[2, 0].set_ylim([-1.5, 1.5])
axs[2, 0].set_ylim([-1.5, 1.5])
axs[2, 1].set_ylim([-1.5, 1.5])
axs[2, 1].set_ylim([-1.5, 1.5])

axs[0, 0].plot(np.arange(35, 65.1, 0.1),
               np.sin(np.arange(0, 30.1, 0.1)),
               c='tab:blue')
axs[0, 1].plot(np.arange(35, 65.1, 0.1),
               np.sin(np.arange(0, 30.1, 0.1)),
               c='tab:green')

axs[1, 0].plot(fq1_freq_scan, fit_fq1, '-', label='fit', color='black', lw=1)
axs[1, 0].plot(fq1_freq_scan,
               datfile['fq1'].su0)
axs[1, 0].plot([fq1, fq1], [0.1, 0.9], '--', color='black', lw=1)
axs[1, 0].plot([fq1_, fq1_], [0.1, 0.9], '--', color='black', lw=1)
axs[1, 0].annotate("", xy=(fq1, 0.75), xytext=(
    fq1_, 0.75), arrowprops=dict(arrowstyle="<->"))
axs[1, 0].annotate(str(exchange_q1)+' MHz', xy=(fq1+0.007, 0.8))
axs[1, 0].set_ylabel(r'$1 - P_\downdownarrows$')
axs[1, 0].set_ylim(0.1, 0.95)
axs[1, 0].axes.xaxis.set_ticklabels([])

axs[2, 0].plot(np.arange(0, 35, 0.1),
               np.zeros(int(35/0.1)),
               c='tab:blue')
axs[2, 0].plot(np.arange(35, 65.1, 0.1),
               np.sin(np.arange(0, 30.1, 0.1)),
               c='tab:blue')
axs[2, 0].plot(np.arange(65.1, 100, 0.1),
               np.zeros(int(35/0.1)),
               c='tab:blue')
axs[2, 0].plot(np.arange(10, 25.1, 0.1),
               np.sin(np.arange(0, 15.1, 0.1)),
               c='tab:green')
axs[2, 0].plot(np.arange(75, 90.1, 0.1),
               np.sin(np.arange(0, 15.1, 0.1)),
               c='tab:green')

axs[2, 1].plot(np.arange(10, 25.1, 0.1),
               np.sin(np.arange(0, 15.1, 0.1)),
               c='tab:blue')
axs[2, 1].plot(np.arange(35, 65.1, 0.1),
               np.sin(np.arange(0, 30.1, 0.1)),
               c='tab:green')
axs[2, 1].plot(np.arange(75, 90.1, 0.1),
               np.sin(np.arange(0, 15.1, 0.1)),
               c='tab:blue')


axs[3, 0].plot(fq1_freq_scan, fit_fq1_, '-', label='fit', color='black', lw=1)
axs[3, 0].plot(fq1_freq_scan,
               datfile['fq1_'].su0)
axs[3, 0].plot([fq1, fq1], [0.1, 0.9], '--', color='black', lw=1)
axs[3, 0].plot([fq1_, fq1_], [0.1, 0.9], '--', color='black', lw=1)
axs[3, 0].set_xlabel('$f_{\mathrm{P4}}$' +
                     f' {unit_style("GHz")}')
axs[3, 0].set_ylabel(r'$1 - P_\downdownarrows$')
axs[3, 0].set_ylim(0.1, 0.95)


axs[1, 1].plot(fq2_freq_scan, fit_fq2, '-', label='fit', color='black', lw=1)
axs[1, 1].plot(fq2_freq_scan,
               datfile['fq2'].su0)
axs[1, 1].plot([fq2, fq2], [0.1, 0.9], '--', color='black', lw=1)
axs[1, 1].plot([fq2_, fq2_], [0.1, 0.9], '--', color='black', lw=1)
axs[1, 1].annotate("", xy=(fq2, 0.75), xytext=(
    fq2_, 0.75), arrowprops=dict(arrowstyle="<->"))
axs[1, 1].annotate(str(exchange_q2)+' MHz', xy=(fq2+0.007, 0.8))
axs[1, 1].axes.xaxis.set_ticklabels([])

axs[3, 1].plot(fq2_freq_scan, fit_fq2_, '-', label='fit', color='black', lw=1)
axs[3, 1].plot(fq2_freq_scan,
               datfile['fq2_'].su0)
axs[3, 1].plot([fq2, fq2], [0.1, 0.9], '--', color='black', lw=1)
axs[3, 1].plot([fq2_, fq2_], [0.1, 0.9], '--', color='black', lw=1)
axs[3, 1].set_xlabel('$f_{\mathrm{P2}}$' +
                     f' {unit_style("GHz")}')

plt.tight_layout()
plt.savefig(os.path.join(save_path, 'figureS5_exchange.pdf'), dpi=300)
plt.savefig(os.path.join(save_path, 'figureS5_exchange.png'), dpi=300)

plt.show()

# %%


n_mw_start = 35
n_mw_end = 65
n_step = 0.1
n_end = 100

x1 = np.arange(0, n_mw_start, n_step)
y1 = np.zeros(int(n_mw_start/n_step))

x2 = np.arange(n_mw_start, n_mw_end+n_step, n_step)
y2 = np.sin(np.arange(0, n_mw_end-n_mw_start+n_step, n_step))

x3 = np.arange(n_mw_end+n_step, n_end, n_step)
y3 = np.zeros(int((n_end-n_mw_end)/n_step))

x = np.append(x1, [x2, x3]).flatten()
y = np.append(y1, [y2, y3]).flatten()

plt.plot(x, y)
plt.show()
