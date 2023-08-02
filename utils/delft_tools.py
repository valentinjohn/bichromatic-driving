# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 08:36:39 2023

@author: Valentin John
"""

# %% imports

import sys
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit

import logging
from functools import partial
import inspect
import time

# %% definitions


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def get_script_directory():
    main_file = sys.modules['__main__'].__file__
    script_dir = os.path.dirname(os.path.abspath(main_file))
    return script_dir


def get_save_path(figure_name: str):
    script_dir = get_script_directory()
    path_figures = os.path.join(script_dir, 'Figures')
    save_path = os.path.join(path_figures, figure_name)

    # Create the 'Figures' folder if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    return save_path


def load_dat(start_time):
    script_dir = get_script_directory()
    end_time = start_time
    datadir = os.path.join(script_dir, 'data')
    datfiles, fnames = get_data_from(
        start_time, end_time, num=1, rootfolder=datadir, only_complete=False)
    datfile = datfiles[0]
    return datfile


def load_cal_rabi_freq(vP1, vP2, P2_pwr, P4_pwr):
    script_dir = get_script_directory()
    with open(os.path.join(script_dir, 'data\config_freq_rabi.txt'), "rb") as file:
        config_freq_rabi = pickle.load(file)

    fq2p2 = config_freq_rabi['P2'][P2_pwr]['fq2'][(vP1, vP2)]
    fq2 = fq2p2/1e9
    fq2_p2 = config_freq_rabi['P2'][P2_pwr]['fq2_'][(vP1, vP2)]
    fq2_ = fq2_p2/1e9
    fq1p4 = config_freq_rabi['P4'][P4_pwr]['fq1'][(vP1, vP2)]
    fq1 = fq1p4/1e9
    fq1_p4 = config_freq_rabi['P4'][P4_pwr]['fq1_'][(vP1, vP2)]
    fq1_ = fq1_p4/1e9
    return fq1, fq1_, fq2, fq2_


def draw_vector(start, amplitude, slope, ax=None):
    end = (start[0] + amplitude, start[1] + slope * amplitude)
    if ax == None:
        plt.plot([start[0], end[0]], [start[1], end[1]], 'b', linewidth=2)
    else:
        ax.plot([start[0], end[0]], [start[1], end[1]], 'b', linewidth=2)


def Q_dif(fp4, fq):
    fp2 = fp4 - fq
    return fp2


def Q_sum(fp4, fq):
    fp2 = fq - fp4
    return fp2


def generate_pulse(total_time, start_time, end_time, frequency, num_pulses, time_between_pulses):
    # Create time array
    t = np.linspace(0, total_time, 1000*total_time)

    # Initialize signal array
    signal = np.zeros_like(t)

    for i in range(num_pulses):
        # Calculate start and end indices for each pulse
        start_index = np.where(t >= start_time + i *
                               (end_time + time_between_pulses))[0][0]
        end_index = np.where(
            t < end_time + i*(end_time + time_between_pulses))[0][-1]

        # Generate pulse
        pulse_t = t[:end_index-start_index]
        pulse = np.sin(2 * np.pi * frequency * pulse_t)

        # Add pulse to signal
        signal[start_index:end_index] = pulse

    return t, signal

# %% Virtual plunger values vP1 and vP2 as a function of P2 and P4


def vP1(df_virtual_gate_matrix: pd.DataFrame,
        P2, P4):
    return (df_virtual_gate_matrix.loc['vP1']['P2']*P2
            + df_virtual_gate_matrix.loc['vP1']['P4']*P4)


def vP2(df_virtual_gate_matrix: pd.DataFrame,
        P2, P4):
    return (df_virtual_gate_matrix.loc['vP2']['P2']*P2
            + df_virtual_gate_matrix.loc['vP2']['P4']*P4)

# %% definitions power considerations


def att(f, a, c):
    return a*f + c


def func_lpf(freq, fth=1.02, slope=118, offset=-5):
    freq = np.array(freq)
    P = list(slope * (freq[freq < fth] - fth) + offset)
    P.extend(list(0 * freq[freq > fth] + offset))
    return P


def func_hpf(freq, fth=1.02, slope=118, offset=-5):
    freq = np.array(freq)
    P = list(0 * freq[freq < fth] + offset)
    P.extend(slope * (freq[freq > fth] - fth) + offset)
    return P


def dBm2V_conversion(data_dBm, Z=50):
    V = (Z/1e3)**0.5 * 10**(np.array(data_dBm)/20)
    return V


def parabola(freq, a, b, c):
    return a*freq**2 + b*freq + c


def power(freq,
          plunger,
          P_siggen=0,
          slope_dipl=118,
          slope_cryo=-2.5):

    if plunger == 'P2':
        att_rt = -12
    elif plunger == 'P4':
        att_rt = -20

    if not isinstance(freq, list):
        freq = np.array([freq])
    else:
        freq = np.array(freq)
    att_dipl = np.array(
        func_lpf(freq, fth=1.02, slope=slope_dipl, offset=-7.5))
    att_cryo = np.array(func_hpf(freq, fth=0, slope=slope_cryo, offset=-12.2))

    P_dBm = P_siggen + att_rt + att_dipl + att_cryo

    return P_dBm

# %% fitting functions


def Lorentz(x, A, FWHM, x0, y0):
    w = FWHM / 2
    return A / (((x - x0) / w)**2 + 1) + y0


def Gauss(x, A, FWHM, x0, y0):
    w = FWHM / (2 * np.sqrt(2 * np.log(2)))
    return A * np.exp(-(x - x0)**2 / (2 * w**2)) + y0


def Gauss2(x, A, A2, FWHM, FWHM2, x0, x0_2, y0):
    w = FWHM / (2 * np.sqrt(2 * np.log(2)))
    w2 = FWHM2 / (2 * np.sqrt(2 * np.log(2)))
    return A * np.exp(-(x - x0)**2 / (2 * w**2)) + A2 * np.exp(-(x - x0_2)**2 / (2 * w2**2)) + y0


def dGauss(x, A, FWHM, x0, y0):
    w = FWHM / (2 * np.sqrt(2 * np.log(2)))
    return (x0 - x) / w**2 * A * np.exp(-(x - x0)**2 / (2 * w**2)) + y0


def Rabi(t, A, f, alpha, y0, phi):
    return A*np.cos(2*np.pi*f*t+phi)/(2*np.pi*f*t)**alpha+y0


def Ramsey(t, A, f, T2, y0, phi):
    return A*np.cos(2*np.pi*t*f+phi)*np.exp(-t/T2)+y0


def CosRams(phi, A, y0, phi0):
    return A*np.cos(phi+phi0) + y0


def ExpDec(x, A, m, y0):
    return A * np.exp(-x / m) + y0


def DoubExpDec(x, A1, m1, A2, m2, y0):
    return A1 * np.exp(-x / m1) + A2 * np.exp(-x / m2) + y0


def Power(x, A, alpha, y0):
    return A*alpha**x+y0


def Line(x, m, y0):
    return m*x+y0


def Coulomb(x, alpha, Te, E0, y0):
    return alpha*np.cosh((x-E0)/(2*Te*8.617e-2))**(-2)+y0


def DoubCoulomb(x, alpha, alpha2, Te, Te2, E0, E02, y0):
    return alpha*np.cosh((x-E0)/(2*Te*8.617e-2))**(-2)+alpha2*np.cosh((x-E02)/(2*Te2*8.617e-2))**(-2)+y0

# %% notebook tools


def get_mw_prop(datfile, gates: list, sig_gen_dict={'p1': 'sig_gen', 'p2': 'sig_gen3', 'p4': 'sig_gen2'}):
    mw_prop = {}

    for gate in gates:
        mw_prop[gate] = {}
        try:
            RF = round(
                datfile.metadata['pc0'][f'MW_{gate}_pulses']['p0']['frequency'] / 1e9, 3)
        except:
            RF = 0

        LO = round(datfile.metadata['LOs'][f'MW_{gate}'] / 1e9, 3)
        IF = round(abs(LO - RF), 3)

        # if LO is deactivated, we have to fetch the RF value from sig_gen directly (I think)
        if RF == 0 and LO == 0:
            RF = round(datfile.metadata['station']['instruments']
                       [sig_gen_dict[gate]]['parameters']['frequency']['value'] / 1e9, 3)

        pwr = datfile.metadata['station']['instruments'][sig_gen_dict[gate]
                                                         ]['parameters']['power']['value']

        try:
            mw_start = datfile.metadata['pc0'][f'MW_{gate}_pulses']['p0']['start']
            mw_stop = datfile.metadata['pc0'][f'MW_{gate}_pulses']['p0']['stop']
            mw_duration = round(mw_stop - mw_start, 1)
        except:
            mw_duration = 0

        mw_prop[gate]['rf'] = RF
        mw_prop[gate]['lo'] = LO
        mw_prop[gate]['if'] = IF
        mw_prop[gate]['pwr'] = pwr
        mw_prop[gate]['time'] = mw_duration
    return mw_prop


def plot_sequence(datfile,
                  gates=['vP1', 'vP2'],
                  mws_p=['MW_p1'],
                  seg_zoom_start=('vP1', 3),
                  seg_zoom_stop=('vP1', -1),
                  ax=None,
                  legend=True,
                  figsize=[3.37, 2],
                  show_inset=False,
                  xlim=False,
                  inset_pos=[0, 1.15, 1, 1],
                  show_mw_prop=False,
                  bbox_to_anchor=(0, 1.5),
                  show_plot=False,
                  lw_mw=0.5):

    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    if show_inset:
        axin = ax.inset_axes(inset_pos)
    handles = []
    labels = gates

    n = 0
    for gate in gates:
        color = list(mcolors.TABLEAU_COLORS.values())[n]
        n = n + 1
        for seg in datfile.metadata['pc0'][f'{gate}_baseband'].keys():
            start = datfile.metadata['pc0'][f'{gate}_baseband'][seg]['start']
            stop = datfile.metadata['pc0'][f'{gate}_baseband'][seg]['stop']
            v_start = datfile.metadata['pc0'][f'{gate}_baseband'][seg]['v_start']
            v_stop = datfile.metadata['pc0'][f'{gate}_baseband'][seg]['v_stop']
            time = np.linspace(start, stop, 2)
            voltage = np.linspace(v_start, v_stop, 2)
            line = ax.plot(time, voltage, color=color)
            if show_inset:
                axin.plot(time, voltage, color=color)
            ax.set_xlabel('time (ns)')
            ax.set_ylabel('V (mV)')
        handles.append(line[0])

    bottom, top = ax.get_ylim()
    yaxis_max = max(abs(bottom), abs(top))

    mw_pl = []
    for mw_p in mws_p:
        mw_pl.append(mw_p[-2:])
    freq_dict = get_mw_prop(datfile, mw_pl)

    for mw_p in mws_p:
        color = list(mcolors.TABLEAU_COLORS.values())[n]
        n = n + 1
        try:
            mw_start = datfile.metadata['pc0'][f'{mw_p}_pulses']['p0']['start']
            mw_stop = datfile.metadata['pc0'][f'{mw_p}_pulses']['p0']['stop']
            mw_duration = round(mw_stop - mw_start, 1)
            # mw_amp = datfile.metadata['pc0']['MW_{}_pulses'.format(pl)]['p0']['amplitude']
            mw_freq = freq_dict[mw_p[-2:]]['rf']
            mw_time = np.linspace(mw_start, mw_stop, 1001)
            mw = yaxis_max*np.sin(mw_time*mw_freq)
            if show_mw_prop:
                mw_prop = get_mw_prop(datfile, [mw_p[-2:]])[mw_p[-2:]]
                labels.append(
                    f"{mw_p} {mw_prop['time']} ns: (RF, LO, IF) = ({mw_prop['rf']}, {mw_prop['lo']}, {mw_prop['if']}) GHz, PWR = {mw_prop['pwr']} dBm")
            else:
                labels.append(mw_p)

            line = ax.plot(mw_time, mw,
                           color=color, lw=lw_mw, alpha=0.5)
            if show_inset:
                axin.plot(mw_time, mw, color=color)
            handles.append(line[0])
        except:
            continue

    ax.set_ylim(-1.1*yaxis_max, 1.1*yaxis_max)
    if xlim:
        try:
            ax.set_xlim(xlim[0], xlim[1])
        except:
            ax.set_xlim(datfile.metadata['pc0'][f'{seg_zoom_start[0]}_baseband'][f'p{seg_zoom_start[1]}']['start'],
                        datfile.metadata['pc0'][f'{seg_zoom_stop[0]}_baseband'][f'p{seg_zoom_stop[1]}']['stop'])

    if show_inset:
        axin.set_xlim(datfile.metadata['pc0'][f'{seg_zoom_start[0]}_baseband'][f'p{seg_zoom_start[1]}']['start'],
                      datfile.metadata['pc0'][f'{seg_zoom_stop[0]}_baseband'][f'p{seg_zoom_stop[1]}']['stop'])
        axin.set_ylim(-1.1*yaxis_max, 1.1*yaxis_max)
        ax.indicate_inset_zoom(axin)

    if legend:
        ax.legend(handles, labels, bbox_to_anchor=bbox_to_anchor)

    if show_plot:
        plt.show()

    try:
        return fig, ax
    except:
        pass
    try:
        return fig
    except:
        pass


def fit_data(xdata, ydata, p0=None, func=dGauss,
             plot=True, return_cov=False, verbose=0, fix_params={}, **kwargs):

    x_range = np.linspace(np.min(xdata), np.max(xdata), num=500)
    p0dict = {}
    if p0 is None:
        if func in [Gauss, dGauss, Lorentz]:

            p0dict = {
                'A': np.max(ydata) - np.mean(ydata),
                'FWHM': (x_range[-1] - x_range[0]) * 0.2,
                'x0': xdata[np.argmax(ydata)],
                'y0': np.mean(ydata)
            }
            if verbose:
                logging.info('p0: ' + str(p0))
        elif func is Rabi:
            p0dict = {
                'A': np.max(ydata) - np.mean(ydata),
                'f': 1/(x_range[-1] - x_range[0])*2,
                'alpha': 0.1,
                'y0': np.mean(ydata),
                'phi': np.pi
            }

            if verbose:
                logging.info('p0: ' + str(p0))

        elif func is ExpDec:
            start = np.mean(ydata[:5])
            baseline = np.mean(ydata[int(3 * len(ydata) / 4):])
            p0dict = {'A': start - baseline,
                      'm': (xdata[1] - xdata[-1]) / 2,
                      'y0': baseline}
        elif func is DoubExpDec:
            start = np.mean(ydata[:5])
            baseline = np.mean(ydata[int(3 * len(ydata) / 4):])
            p0dict = {'A1': start - baseline,
                      'm1': (xdata[1] - xdata[0]) / 2,
                      'A2': start - baseline,
                      'm2': (xdata[1] - xdata[0]) / 4,
                      'y0': baseline}

    if fix_params:
        func = partial(func, **fix_params)

    try:
        if p0dict:
            p0 = [p0dict[arg] for arg in inspect.getfullargspec(func).args[1:]]

        # check p0 feasability
        if 'bounds' in kwargs.keys() and p0 is not None:
            for i, p in enumerate(p0):
                lower, upper = (kwargs['bounds'][0][i], kwargs['bounds'][1][i])
                if not lower <= p <= upper:
                    p = min([max([lower, p]), upper])
        p1, covar = curve_fit(func, xdata, ydata, p0=p0, **kwargs)
        if plot:
            plt.scatter(xdata, ydata, marker='.')
            plt.plot(x_range, func(x_range, *p1))

        if return_cov:
            return np.sqrt(np.diag(covar)), p1
        else:
            return p1
    except RuntimeError as Err:
        logging.warning(Err)
