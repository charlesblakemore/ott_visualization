import sys, os, time

from tqdm import tqdm

import numpy as np
import matlab.engine

import lib.ott_plotting as ott_plotting
ott_plotting.base_plotting_directory = '/home/cblakemore/plots/ott_farfield'



start = time.time()

base_data_path = '../raw_data/ztrans_movie_transmitted_test/'

### Pack up all the simulation parameters into a dictionary with string keys
### that match the expected MATLAB variable names. This is the only reasonably
### compact way of passing keyword arguments via the MATLAB/Python engine API
###
### REMEMBER: Cartesian offsets are in SI base units (meters) and correspond
### to the position of the BEAM relative to the microsphere/scatterer.
###    e.g. zOffset = +10um corresponds to a microsphere BELOW the focus
simulation_parameters = {
        'datapath': base_data_path, \
          'radius': 3.76e-6, \
      'n_particle': 1.39, \
        'n_medium': 1.00, \
      'wavelength': 1064.0e-9, \
              'NA': 0.095, \
         'xOffset': 0.0e-6, \
         'yOffset': 0.0e-6, \
         'zOffset': 0.0e-6, \
        'thetaMin': 0.0, \
        'thetaMax': float(np.pi/6.0), \
          'ntheta': 1001, \
            'nphi': 101, \
    'polarisation': 'X', \
            'Nmax': 100
}

param_to_sweep = 'zOffset'
# param_array = np.linspace(0.0, -100.0, 101)
param_array = np.linspace(0.0, -100.0, 3)
param_scale = 1e-6
save_suffix = '_um'
# save_suffix = ''

movie_name = 'zsweep_0-50um'



beam = 'tot'
transmitted = True
rmax = 0.005
save_fig = True
show_fig = True

# max_radiance_val = 8.4
max_radiance_val = 8.4**2
manual_phase_plot_lims = ()



##########################################################################
##########################################################################
##########################################################################

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

param_ind = 0
for param_ind in tqdm(range(len(param_array))):

    ### Get the current value of the swept parameter
    param_val = param_array[param_ind]
    param_str = f'{param_to_sweep}_{int(param_val):d}{save_suffix}'
    sys.stdout.flush()
    param_ind += 1

    ### Adjust the savepath to included information about the current
    ### parameter values
    simulation_parameters['datapath'] \
        = os.path.join(base_data_path, param_str)
    simulation_parameters[param_to_sweep] \
        = float(param_scale*param_val)

    ### Build the MATLAB formatted argument list from the dictionary
    ### defined at the top of this script
    arglist = [[key, simulation_parameters[key]] \
                for key in simulation_parameters.keys()]

    ### Start the MATLAB engine and run the computation
    engine = matlab.engine.start_matlab()
    matlab_datapath \
        = engine.compute_far_field(\
            *[arg for argtup in arglist for arg in argtup], \
            nargout=1, background=False)

    ### Load the data that MATLAB computed and saved
    theta_grid, r_grid, efield \
        = ott_plotting.load_farfield_data(\
                    matlab_datapath, beam=beam, \
                    transmitted=transmitted)

    ### Update this label for movie frames
    ms_position = [simulation_parameters['xOffset'], \
                   simulation_parameters['yOffset'], \
                   simulation_parameters['zOffset']]

    ### Plot everything!
    figname = os.path.join(ott_plotting.base_plotting_directory, \
                           movie_name, f'frame_{param_ind:04d}.png')
    ott_plotting.plot_2D_farfield(
        theta_grid, r_grid, efield, simulation_paramters, \
        ms_position=ms_position, rmax=rmax, title=title, \
        manual_phase_plot_lims=manual_phase_plot_lims, \
        figname=figname, save=save_fig, show=show_fig)

