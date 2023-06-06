# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:27:14 2017

@author: David Franke
"""

#%% Imports

import os
import time
from matplotlib import pyplot as plt
from qcodes.data.data_set import load_data
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import ipywidgets as widgets
import logging
import pickle
from functools import partial
import inspect
import qcodes

#%% Definitions

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

from IPython.display import display, Markdown
def mdtable(vdict, vertical = False):
    '''
    returns a markdown table that can be displayed in jupyter notebooks.
    vdict: dictionary

    '''
    mdlabels = '|'.join(list(vdict.keys()))+'\n'
    mddashes = '|'.join(['--- ' for v in vdict.values()])+'\n'
    mdvalues = '|'.join([str(v) for v in vdict.values()])

    mdstring = mdlabels + mddashes + mdvalues


    return(Markdown(mdstring))

def get_data_from(start_time, end_time = 'now', num = np.inf,
                  rootfolder=qcodes.data.data_set.DataSet.default_io.base_location,
                  verbose=0, only_complete = True):
    """
    Read a number of consecutive datasets. arguments:
        times:  time of first datset in %H-%M-%S format, or
                datetime of first dataset (%Y-%m-%d\\%H-%M-%S), or
                two element list with from and to datetimes
        num:    max number of datasets
        rootfolder: folder to scan for datasets
        only_complete: only read finished datasets
    """
    datfiles = []
    fnames = []

    times = [start_time, end_time]

    if isinstance(times, list):
        try:
            dtf = time.strptime(times[0], '%Y-%m-%d\\%H-%M-%S')
        except ValueError as Err:
            dtf = time.strptime(time.strftime(
                '%Y-%m-%d\\') + times[0], '%Y-%m-%d\\%H-%M-%S')
            logging.debug('defaulting dtf to today')

        try:
            dtt = time.strptime(times[1], '%Y-%m-%d\\%H-%M-%S')
        except ValueError:
            try:
                dtt = time.strptime(time.strftime(
                    '%Y-%m-%d\\') + times[1], '%Y-%m-%d\\%H-%M-%S')
                logging.debug('defaulting dtt to today')
            except ValueError:
                dtt = time.localtime()
                logging.debug('defaulting dtt to now')
    else:
        raise ValueError('Wrong input format.')
    rfs = []
    for fol in os.listdir(rootfolder):
        try:
            if dtt.tm_yday >= time.strptime(os.path.basename(
                    fol), '%Y-%m-%d').tm_yday >= dtf.tm_yday:
                rfs.append(rootfolder + '\\' + fol)
        except ValueError as Err:
            if verbose:
                print(Err)

    logging.debug('scanning folders %s' % str(rfs))
    # print('dtt',dtt,'dtf',dtf)
    for i, rf in enumerate(rfs):
        for fol in os.listdir(rf):
            try:
                t = time.strptime(rf[-10:] + '\\' +
                                  fol[0:8], '%Y-%m-%d\\%H-%M-%S')
                if dtt >= t >= dtf and len(datfiles) < num:
                    fnames.append(rf + '\\' + fol)
                    try:
                        datfiles.append(
                            load_data(location=rf + '\\' + fol, formatter=None, io=None))

                        if any([np.any(np.isnan(arr)) for arr in
                                datfiles[-1].arrays.values()]) and only_complete:
                            raise Exception('incomplete')
                    except Exception as Ex:
                        logging.info(
                                'file %s seems to be broken or incomplete, omitting: %s' % (fnames[-1], str(Ex)))
                        del datfiles[-1]
                        del fnames[-1]
            except ValueError as Err:
                print(Err)
    print('loaded %d files' % len(datfiles))
    logging.debug('\n'.join(fnames))
    return (datfiles, fnames)


def plot_average(dat, ratio=True, return_data=False, **args):
    if len(dat) is 2:
        datfiles = dat[0]
    else:
        datfiles = dat

    datkeys = datfiles[0].arrays.keys()
    datasum = {}
    avg_data = {}

    for dk in datkeys:
        datasum[dk] = datfiles[0].arrays[dk].ndarray
        for df in datfiles[1:len(datfiles)]:
            try:
                datasum[dk] += df.arrays[dk].ndarray
            except ValueError:
                logging.warning('shapes did not match: %s' % df.location)
        #print('averaging %d sweeps' %len(datfiles))
        avg_data[dk] = datasum[dk] / len(datfiles)

    meta = datfiles[-1].metadata
    plt.figure(figsize=(15, 5))
    xdata = avg_data['Frequency'] / 1e9
    if ratio:
        plt.plot(xdata, avg_data['readall'] / avg_data['init'])
        plt.ylabel('read/init ratio')
        plt.text(np.min(xdata), plt.ylim()[0], 'iinit: %.2f\n iread: %.2f' % (
            np.mean(avg_data['iinit']), np.mean(avg_data['iread'])))
    else:
        plt.plot(xdata, avg_data['init'])
        plt.plot(xdata, avg_data['readall'])
        plt.ylabel('avg events')

    plt.xlabel('frequency (GHz)')
    plt.title(datfiles[-1].location + ' (%d files)' % (len(datfiles)))
    if return_data:
        return avg_data, meta


def nice_plot(data, key, x='Frequency', fig=None, title=None):
    if fig is None:
        plt.figure(figsize=(15, 5))
    else:
        plt.figure(fig)

    if x == 'Frequency':
        div = 1e9
        unit = 'GHz'
    elif x == 'pulsewidth':
        div = 1e-6
        unit = 'us'
    else:
        div = 1
        unit = 'arb. units'

    xdata = data[x] / div
    if key is 'ratio':
        ydata = data['readall'] / data['init']
    else:
        ydata = data[key]

    plt.plot(xdata, ydata)
    plt.ylabel(key)

    plt.xlabel('%s (%s)' % (x, unit))
    if title:
        plt.title(title)


def get_average(dat):
    datfiles = dat

    if len(dat) == 2:
        if isinstance(dat[1], list):
            logging.debug('assuming [data, fnames]')
            datfiles = dat[0]

    datkeys = datfiles[0].arrays.keys()
    logging.info(', '.join(datkeys))
    datasum = {}
    for dk in datkeys:
        datasum[dk] = np.zeros(len(datfiles[0].arrays[dk]))
    avg_data = {}

#    for df in datfiles:
#        for dk in datkeys:
# try:
#            datasum[dk]+=df.arrays[dk].ndarray
# except ValueError:
##                logging.warning('shapes did not match: %s' % df.location)

    setpoint = [ds for ds in datfiles[0].arrays.keys() if getattr(
        datfiles[0], ds).is_setpoint == True][0]

    for df in datfiles:
        for ind, f in enumerate(sorted(df.arrays[setpoint])):
            ran_i = np.where(df.arrays[setpoint].ndarray == f)[0][0]
            for dk in (datkeys - [setpoint]):
                datasum[dk][ind] += df.arrays[dk].ndarray[ran_i]

    for dk in datkeys - [setpoint]:
        avg_data[dk] = datasum[dk] / len(datfiles)

    avg_data[setpoint] = np.array(sorted(datfiles[0].arrays[setpoint]))

    meta = datfiles[-1].metadata
    return avg_data, meta


def meta_gate_table(meta):
    gates = meta['station']['instruments']['gates']['parameters'].keys() - \
        ['IDN']
    res = '| gate | value |\n |---|---|\n'

    for g in sorted(gates):
        value = float(meta['station']['instruments']
                      ['gates']['parameters'][g]['value'])
        res += '|%s | %.2f mV|\n' % (g, value)
    return res

def meta_gate_string(meta):
    gates = meta['station']['instruments']['gates']['parameters'].keys() - \
        ['IDN']
    res = ''

    for g in sorted(gates):
        value = float(meta['station']['instruments']
                      ['gates']['parameters'][g]['value'])
        res += '%s: %.2f mV\n' % (g, value)
    return res


def meta_mw_string(meta):
    gates = meta['station']['instruments']['sig_gen']['parameters'].keys() - \
        ['IDN']
    res = ''

    for g in sorted(gates):
        value = meta['station']['instruments']['sig_gen']['parameters'][g]['value']
        unit = meta['station']['instruments']['sig_gen']['parameters'][g]['unit']
        res += '%s : %s %s\n' % (g, value, unit)
    return res

def meta_get_value(data, inst, parameter):
    return data.metadata['station']['instruments'][inst]['parameters'][parameter]['value']

def plot_all(time_from, **args):
    fnames, datfiles = get_data_from(time_from, **args)
    for i, df in enumerate(datfiles):
        plt.figure(figsize=(15, 5))
        xdata = df.arrays['Frequency'].ndarray / 1e9
        ydata = df.arrays['readall'].ndarray / df.arrays['init'].ndarray
        plt.plot(xdata, ydata)
        plt.text(np.min(xdata), plt.ylim()[0], 'average: %.2f\n std: %.2f' % (
            np.mean(ydata), np.std(ydata)))
        plt.xlabel('frequency (GHz)')
        plt.ylabel('signal (arb. units)')
        plt.title(fnames[i])

def get_mw_prop(datfile, gates:list, sig_gen_dict = {'p1':'sig_gen', 'p2':'sig_gen3', 'p4':'sig_gen2'}):
    mw_prop = {}

    for gate in gates:
        mw_prop[gate] = {}
        try:
            RF = round(datfile.metadata['pc0'][f'MW_{gate}_pulses']['p0']['frequency'] /1e9, 3)
        except:
            RF = 0

        LO = round(datfile.metadata['LOs'][f'MW_{gate}'] / 1e9, 3)
        IF = round(abs(LO - RF), 3)

        if RF == 0 and LO == 0: # if LO is deactivated, we have to fetch the RF value from sig_gen directly (I think)
            RF = round(datfile.metadata['station']['instruments'][sig_gen_dict[gate]]['parameters']['frequency']['value'] /1e9, 3)


        pwr = datfile.metadata['station']['instruments'][sig_gen_dict[gate]]['parameters']['power']['value']

        try:
            mw_start = datfile.metadata['pc0'][f'MW_{gate}_pulses']['p0']['start']
            mw_stop = datfile.metadata['pc0'][f'MW_{gate}_pulses']['p0']['stop']
            mw_duration = round(mw_stop - mw_start,1)
        except:
            mw_duration = 0

        mw_prop[gate]['rf'] = RF
        mw_prop[gate]['lo'] = LO
        mw_prop[gate]['if'] = IF
        mw_prop[gate]['pwr'] = pwr
        mw_prop[gate]['time'] = mw_duration
    return mw_prop

def plot_sequence(datfile,
                  gates = ['vP1', 'vP2'],
                  mws_p = ['MW_p1'],
                  seg_zoom_start = ('vP1', 3),
                  seg_zoom_stop = ('vP1', -1),
                  ax = None,
                  legend=True,
                  figsize=[3.37, 2],
                  show_inset = False,
                  xlim = False,
                  inset_pos = [0, 1.15, 1, 1],
                  show_mw_prop = False,
                  bbox_to_anchor=(0, 1.5),
                  show_plot = False,
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
            mw_duration = round(mw_stop - mw_start,1)
            # mw_amp = datfile.metadata['pc0']['MW_{}_pulses'.format(pl)]['p0']['amplitude']
            mw_freq = freq_dict[mw_p[-2:]]['rf']
            mw_time = np.linspace(mw_start, mw_stop, 1001)
            mw = yaxis_max*np.sin(mw_time*mw_freq)
            if show_mw_prop:
                mw_prop = get_mw_prop(datfile, [mw_p[-2:]])[mw_p[-2:]]
                labels.append(f"{mw_p} {mw_prop['time']} ns: (RF, LO, IF) = ({mw_prop['rf']}, {mw_prop['lo']}, {mw_prop['if']}) GHz, PWR = {mw_prop['pwr']} dBm")
            else:
                labels.append(mw_p)

            line = ax.plot(mw_time, mw ,
                           color=color, lw = lw_mw, alpha=0.5)
            if show_inset:
                axin.plot(mw_time, mw, color=color)
            handles.append(line[0])
        except: continue


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
#%% fitting functions


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

def Power(x,A,alpha,y0):
    return A*alpha**x+y0

def Line(x, m, y0):
    return m*x+y0

def Coulomb(x, alpha, Te, E0, y0):
    return alpha*np.cosh((x-E0)/(2*Te*8.617e-2))**(-2)+y0

def DoubCoulomb(x, alpha, alpha2, Te, Te2, E0, E02, y0):
    return alpha*np.cosh((x-E0)/(2*Te*8.617e-2))**(-2)+alpha2*np.cosh((x-E02)/(2*Te2*8.617e-2))**(-2)+y0

#%% fit routines


def fit_ratio_data(data, p0=None, func=Gauss, plot=True,
                   return_cov=False, verbose=0):
    ydata = data['readall'] / data['init']
    freq_range = np.linspace(
        np.min(data['Frequency'] / 1e9), np.max(data['Frequency'] / 1e9), num=500)
    if p0 is None:
        p0 = [np.max(ydata) - np.mean(ydata), (freq_range[-1] - freq_range[0])
              * 0.2, data['Frequency'][np.argmax(ydata)] / 1e9, np.mean(ydata)]
        if verbose:
            logging.debug('p0: ' + str(p0))
    try:
        pfit, covar = curve_fit(func, data['Frequency'] / 1e9, ydata, p0=p0)
        if plot:
            plt.plot(freq_range, func(
                freq_range, pfit[0], pfit[1], pfit[2], pfit[3]))
            plt.text(plt.xlim()[0] + (plt.xlim()[1] - plt.xlim()[0]) * 0.1, plt.ylim()[1] - (
                plt.ylim()[1] - plt.ylim()[0]) * 0.1, ' g-factor: %.4f' % (0.07144773 * pfit[2] / 0.6))
        if return_cov:
            return np.sqrt(np.diag(covar)), pfit
        else:
            return pfit
    except RuntimeError as Err:
        logging.warning(Err)


def fit_diff_data(data, p0=None, func=dGauss, plot=True,
                  return_cov=False, verbose=0):
    ydata = data['readdiff']
    freq_range = np.linspace(
        np.min(data['Frequency'] / 1e9), np.max(data['Frequency'] / 1e9), num=500)
    if p0 is None:
        p0 = [np.max(ydata) - np.mean(ydata), (freq_range[-1] - freq_range[0])
              * 0.2, data['Frequency'][np.argmax(ydata)] / 1e9, np.mean(ydata)]
        if verbose:
            logging.debug('p0: ' + str(p0))
    try:
        pfit, covar = curve_fit(func, data['Frequency'] / 1e9, ydata, p0=p0)
        if plot:
            plt.plot(freq_range, func(
                freq_range, pfit[0], pfit[1], pfit[2], pfit[3]))
            plt.text(plt.xlim()[0] + (plt.xlim()[1] - plt.xlim()[0]) * 0.1, plt.ylim()[1] - (plt.ylim()[1] - plt.ylim()[0])
                     * 0.2, ' g-factor: %.4f\n FWHM = %.2f MHz\n Amp: %E' % (0.07144773 * pfit[2] / 0.6, abs(pfit[1]) * 1e3, pfit[0]))
        if return_cov:
            return np.sqrt(np.diag(covar)), pfit
        else:
            return pfit
    except RuntimeError as Err:
        logging.warning(Err)


def fit_data(xdata, ydata, p0=None, func=dGauss,
             plot=True, return_cov=False, verbose=0,fix_params = {}, **kwargs):

    x_range = np.linspace(np.min(xdata), np.max(xdata), num=500)
    p0dict = {}
    if p0 is None:
        if func in [Gauss, dGauss, Lorentz]:

            p0dict = {
                    'A':np.max(ydata) - np.mean(ydata),
                    'FWHM': (x_range[-1] - x_range[0]) * 0.2,
                    'x0': xdata[np.argmax(ydata)],
                    'y0': np.mean(ydata)
                    }
            if verbose:
                logging.info('p0: ' + str(p0))
        elif func is Rabi:
            p0dict = {
                    'A':np.max(ydata) - np.mean(ydata),
                    'f' :1/(x_range[-1] - x_range[0])*2,
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
            for i,p in enumerate(p0):
                lower, upper = (kwargs['bounds'][0][i], kwargs['bounds'][1][i])
                if not lower <=p <= upper:
                    p=min([max([lower, p]),upper])
        p1, covar = curve_fit(func, xdata, ydata, p0=p0, **kwargs)
        if plot:
            plt.scatter(xdata,ydata,marker='.')
            plt.plot(x_range, func(x_range, *p1))

        if return_cov:
            return np.sqrt(np.diag(covar)), p1
        else:
            return p1
    except RuntimeError as Err:
        logging.warning(Err)


def avg_widget(times, plot1='ratio', return_data=False, **args):
    datasets, fnames = get_data_from(times, **args)
    setpoint = [ds for ds in datasets[0].arrays.keys() if getattr(
        datasets[0], ds).is_setpoint == True][0]

    tab = widgets.Tab()
    tab.children = [widgets.Output(), widgets.Output(), widgets.Output(
    ), widgets.Output(), widgets.Output(), widgets.Output()]
    tab.set_title(0, 'ratio')
    tab.set_title(1, 'diff')
    tab.set_title(2, 'Gates')
    tab.set_title(3, 'MW')
    tab.set_title(4, 'noise')
    tab.set_title(5, 'files')

    with tab.children[0]:
        avg_dat, meta = get_average(datasets)
        nice_plot(avg_dat, plot1, x=setpoint,
                  title=fnames[-1] + ' (%d files)' % (len(datasets)))
        try:
            plt.text(plt.xlim()[0], plt.ylim()[0], 'iinit: %.2f\n iread: %.2f' % (
                np.mean(avg_dat['iinit']), np.mean(avg_dat['iread'])))
        except BaseException:
            pass
        plt.show()
    with tab.children[1]:
        if 'readdiff' in avg_dat.keys():
            nice_plot(avg_dat, 'readdiff', x=setpoint,
                      title=fnames[-1] + ' (%d files)' % (len(datasets)))
            if setpoint is 'pulsewidth':
                print(fit_data(avg_dat['pulsewidth'] / 1e-6, avg_dat['readdiff'], func=Rabi, p0=[5e-03, 1.5, 1000,
                                                                                                 7.14333323e-03, 0.002], plot=True))
            plt.show()
        else:
            print('No readdiff')
    with tab.children[2]:
        print(meta_gate_string(meta))
    with tab.children[3]:
        print(meta_mw_string(meta))
    with tab.children[4]:
        plot_std_vs_sweeps(datasets)
        plt.show()
    with tab.children[5]:
        print('\n'.join(fnames))

    if return_data:
        display(tab)
        return avg_dat, meta
    else:
        return tab


def plot_std_vs_sweeps(dat):
    noise = {}
    datsum = dat[0].arrays
    for dk in dat[0].arrays.keys():
        noise[dk] = [np.std(dat[0].arrays[dk].ndarray)]
    for i, d in enumerate(dat[1:]):
        for dk in dat[0].arrays.keys():
            datsum[dk].ndarray += d.arrays[dk].ndarray
            noise[dk].append(np.std(datsum[dk].ndarray / (i + 2)))
    plt.plot(range(1, len(dat) + 1), noise['readall'])
    plt.plot(range(1, len(dat) + 1), noise['init'])
    plt.xlabel('sweeps')
    plt.ylabel('std deviation')


def load_traces_avg(times, int_range = None, int_times=[0, 12], plot_avgs=False,
                    rf=r'D:\measurements\Si28-IntelDelft\2017_11_17\Data\BOTTOM\Qcodes', return_data=False, **args):
    dat, fnames = get_data_from(times, rootfolder = rf, **args)
    meta = dat[-1].metadata
    setpoint = [key for key in dat[-1].arrays.keys()
                if dat[-1].arrays[key].is_setpoint][0]

    avgs = []
    avgs2 = []
    freq = []
    for lm in fnames:
        try:
            rawdata = pickle.load(open(lm + '\\traces.pkl', 'rb'))
            for rd in rawdata:
                freq.append(rd[0])
                tr1 = np.zeros(len(rd[1][0]))
                tr2 = np.zeros(len(rd[1][0]))
                for i, tr in enumerate(rd[1]):
                    if i % 2 is 0:
                        tr1 += tr
                    else:
                        tr2 += tr
                avgs.append(tr1 / (len(rd[1]) / 2))
                avgs2.append(tr2 / (len(rd[1]) / 2))
        except FileNotFoundError as Err:
            avg_data = pickle.load(open(lm + '\\avg_traces.pkl', 'rb'))
            avgs = [tr[2] for tr in avg_data[0]]
            avgs2 = [tr[2] for tr in avg_data[1]]
            freq = [tr[0] for tr in avg_data[0]]

    diff = []

    avgs_avg = []
    avgs2_avg = []
    avgs_avg_backsub = []
    for f in sorted(set(freq)):
        dps = np.array([[avgs[index], avgs2[index]]
                        for index in np.where(freq == f)[0]])
        avgs_avg.append(np.mean(dps, axis=0)[0])
        avgs2_avg.append(np.mean(dps, axis=0)[1])
        avgs_avg_backsub.append(np.mean(dps, axis=0)[
                                0] - np.mean(np.mean(dps, axis=0)[0][300:350]))
    avgs_avg = np.array(avgs_avg)
    avgs2_avg = np.array(avgs2_avg)
    avgs_avg_backsub = np.array(avgs_avg_backsub)

    diff = avgs_avg - avgs2_avg
    freq_arr = np.array(sorted(list(set(freq)))) / 1e9

    trace_len = len(avgs[0]) / 30.517
    trace_times = np.linspace(0, trace_len, num=len(avgs[0]))
    int_range = [int(np.where(trace_times > int_times[0])[0][0]), int(np.where(trace_times < int_times[1])[0][-1])]

    if plot_avgs:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        freq_arr = np.array(sorted(list(set(freq)))) / 1e9
        plt.pcolor(np.linspace(0, trace_len, num=len(
            avgs[0])), freq_arr, np.array(avgs_avg))
        plt.xlabel('time (ms)')
        plt.ylabel('%s (%s)' %
                   (dat[1].arrays[setpoint].label, dat[1].arrays[setpoint].unit))
        plt.colorbar()
        plt.subplot(1, 2, 2)
        freq_arr = np.array(sorted(list(set(freq)))) / 1e9
        plt.pcolor(np.linspace(0, trace_len, num=len(
            avgs[0])), freq_arr, np.array(avgs_avg_backsub))
        plt.xlabel('time (ms)')
        plt.ylabel('%s (%s)' %
                   (dat[1].arrays[setpoint].label, dat[1].arrays[setpoint].unit))
        plt.colorbar()
        plt.show()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.pcolor(trace_times, freq_arr, np.array(diff))
    plt.xlabel('time (ms)')
    plt.ylabel('%s (%s)' %
               (dat[1].arrays[setpoint].label, dat[1].arrays[setpoint].unit))
    plt.axvline(int_times[0], color='b')
    plt.axvline(int_times[1], color='r')
    plt.colorbar()


    plt.subplot(122)
    dat_int = np.sum([d[int_range[0]:int_range[1]] for d in diff], axis=1)
    plt.plot(freq_arr, dat_int)
    plt.xlabel('%s (%s)' %
               (dat[1].arrays[setpoint].label, dat[1].arrays[setpoint].unit))
    plt.suptitle(os.path.basename(dat[0].location) + ' - ' +
                 os.path.basename(dat[-1].location) + ' (%d files)' % len(fnames))
    plt.show()

    if return_data:
        return (np.linspace(0, trace_len, num=len(
            avgs[0])), freq_arr), avgs_avg, avgs2_avg, meta


def plot_data(alldata, plot_title='plot'):
    keysX = [k for k in alldata.arrays.keys() if getattr(
        alldata, k).is_setpoint == True]
    dataX = [alldata.arrays[k] for k in keysX][0]
    keysY = [k for k in alldata.arrays.keys() if getattr(
        alldata, k).is_setpoint == False]
    dataY = [alldata.arrays[k] for k in keysY]

    plt.figure(figsize=(15, 5))
    for i, dat in enumerate(dataY):
        rng = min([len(dat), len(dataX)])
        plt.subplot(1, len(dataY), i + 1)
        plt.title(plot_title + ': ' +
                  '\\'.join(alldata.location.split('\\')[-2:]))
        plt.plot(dataX[:rng], dat[:rng], color=plt.cm.rainbow(
            i / max([len(dataY) - 1, 1])))
        plt.xlabel(keysX[0])
        plt.ylabel(keysY[i])


def plot_map(alldata):
    keyY, keyX = [k for k in alldata.arrays.keys() if getattr(
        alldata, k).is_setpoint == True]
    px = getattr(alldata, keyX)
    py = getattr(alldata, keyY)

    keysZ = [k for k in alldata.arrays.keys() if getattr(
        alldata, k).is_setpoint == False]
    pz_list = [getattr(alldata, k) for k in keysZ]

    plt.figure(figsize=(15, 8))

    for i, pz in enumerate(pz_list):
        plt.subplot(1, len(pz_list), i + 1)

        plt.pcolor(px.ndarray[0], py.ndarray, pz.ndarray)
        plt.xlabel('%s (%s)' % (px.name, px.unit))
        plt.ylabel('%s (%s)' % (py.name, py.unit))
        plt.title('%s (%s)' % (pz.name, pz.unit))
        plt.colorbar()
    plt.show()


def gate_value(d, gate):
    try:
        return float(d.metadata['station']['instruments']
                 ['gates']['parameters'][gate]['value'])
    except:
        return float(d.metadata['gates']['parameters'][gate]['value'])


def sub_avg(data, Npoints_offset):
    '''
    Substract offset to digitizer traces, using first Npoints_offset of the traces.
    '''

    res = data.arrays[data.default_parameter_name()]

    res_new = np.zeros((len(res), len(res[0])))

    for i in range(0, len(res)):
        res_new[i] = res[i] - res[i][0:Npoints_offset].mean()

    arr = DataArray(name=res.name+'_sub', array_id=res.array_id+'_sub', label=res.label+'_sub', unit='V',shape =res.shape, set_arrays=res.set_arrays, preset_data=res_new)
    try:
        data.add_array(arr)
        data.write()
    except Exception as Err:
        print(Err)
