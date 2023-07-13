# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 08:36:39 2023

@author: Valentin John
"""

# %% imports

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.notebook_tools import get_data_from
from scipy.optimize import curve_fit
import pickle

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


def load_data(start_time):
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
