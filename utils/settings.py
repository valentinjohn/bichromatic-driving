# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:36:59 2023

@author: Valentin John
"""

# %% imports

import sys
import os
import inspect
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from utils.notebook_tools import get_data_from
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pickle
from scipy.optimize import curve_fit
from utils.notebook_tools import Rabi, Gauss, plot_sequence, get_mw_prop

# %% defining style

fig_size_single = 3.37
fig_size_double = 6.69

plt.rcParams.update({'font.size': 8})
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams["figure.figsize"] = (fig_size_single, 3)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.default'] = 'it'  # 'regular'

plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.fontsize'] = 'small'
plt.rcParams['legend.scatterpoints'] = 1
plt.rcParams['axes.labelpad'] = 4  # -2
