import os, math, sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import unwrap

import scipy.constants as constants


##########################################################################
##########################################################################

from farfield_plotting import *

##########################################################################
##########################################################################

plt.rcParams.update({'font.size': 14})


def update_base_plotting_directory(new_dir):
    global global_dict
    global_dict['base_plotting_directory'] = new_dir