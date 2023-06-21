# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 08:36:39 2023

@author: Valentin John
"""

# %% imports

import sys
import os
import matplotlib.pyplot as plt
from utils.notebook_tools import get_data_from
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
