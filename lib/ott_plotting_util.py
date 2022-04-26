import os, math, sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import unwrap

import scipy.constants as constants

plt.rcParams.update({'font.size': 14})


base_plotting_directory = os.path.expanduser('~/plots/')
global_dict = {'base_plotting_directory': base_plotting_directory}



# print()
# print('Plots will be saved with the following base path:')
# print(f'     < {base_plotting_directory} >')
# print()
# print('( to change, set the following variable:   )')
# print('(    ott_plotting.base_plotting_directory  )')







def _labelLine(line, x, x_offset=0.0, y_offset=0.0, \
               alpha=0.0, label=None, align=True, **kwargs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = math.degrees(math.atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5
        
    t = ax.text(x+x_offset, y+y_offset, label, rotation=trans_angle, **kwargs)
    t.set_bbox(dict(alpha=alpha))






def _make_all_pardirs(path, confirm=True):
    '''Function to help pickle from being shit. Takes a path
       and looks at all the parent directories etc and tries 
       making them if they don't exist.
       INPUTS: path, any path which needs a hierarchy already 
                     in the file system before being used
       OUTPUTS: none
       '''

    parts = path.split('/')
    parent_dir = '/'
    for ind, part in enumerate(parts):
        if ind == 0 or ind == len(parts) - 1:
            continue
        parent_dir += part
        parent_dir += '/'
        if not os.path.isdir(parent_dir):
            if confirm:
                print()
                print('Make this directory?')
                print(f'    {parent_dir}')
                print()
                answer = input('(Y/N): ')
            else:
                answer = 'Y'

            if ('y' in answer) or ('Y' in answer):
                os.mkdir(parent_dir)





def _build_title(beam, transmitted=True):

    if beam == 'inc':
        title = 'Incident Gaussian Beam'
    elif beam == 'scat':
        title = 'Scattered Beam'
    else:
        title = 'Total Beam'

    if transmitted:
        title += ', Transmitted Hemisphere'
    else:
        title += ', Back-reflected Hemisphere'

    return title



def _determine_Nmax(offset, Nmax_min=100, Nmax_max=300):
    return None




