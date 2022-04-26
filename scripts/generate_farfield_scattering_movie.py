import sys, os, time

from tqdm import tqdm

import numpy as np
import matlab.engine

import ott_plotting


ott_plotting.update_base_plotting_directory(\
        '/home/cblakemore/plots/ott_farfield/movies')


base_data_path = '../raw_data/movies/zsweep_test'

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
                  'halfCone': float(np.pi/6), \
                    'ntheta': 301, \
                      'nphi': 101, \
              'polarisation': 'X', \
                      'Nmax': 30, \
                'resimulate': True
}



plot_parameters = {
                      'beam': 'tot', \
                      'rmax': 0.004, \
                      'save': True, \
                      'show': False, \
                   'plot_2D': True, \
                   'plot_3D': True, \
                 'view_elev': -40.0, \
                 'view_azim': 20.0, \
          'max_radiance_val': 25.0, \
              'unwrap_phase': True, \
    'manual_phase_plot_lims': (-2.0*np.pi, 2.0*np.pi), \
            'label_position': True, \
                   'verbose': True
}






param_to_sweep = 'zOffset'
# param_array = np.linspace(0.0, -100.0, 101)
param_array = np.linspace(0.0, -50.0, 51)
param_scale = 1e-6
save_suffix = '_um'
# save_suffix = ''

movie_name = 'zsweep_0-50um'





##########################################################################
##########################################################################
##########################################################################


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
    engine.addpath('../lib', nargout=0)
    matlab_datapath \
        = engine.compute_far_field(\
            *[arg for argtup in arglist for arg in argtup], \
            nargout=1, background=False)

    ### Load the data that MATLAB computed and saved, handling the 
    ### transmitted and reflected cases separately since they may 
    ### propagate through distinct optical systems
    theta_grid_trans, r_grid_trans, efield_trans\
        = ott_plotting.load_farfield_data(\
                matlab_datapath, transmitted=True,\
                beam=plot_parameters['beam'])

    theta_grid_refl, r_grid_refl, efield_refl\
        = ott_plotting.load_farfield_data(\
                matlab_datapath, transmitted=False,\
                beam=plot_parameters['beam'])


    ray_tracing = ott_plotting.get_simple_ray_tracing_matrix()


    ### Plot everything!
    figname_trans = os.path.join(movie_name, 'trans', f'frame_{param_ind:04d}.png')
    ott_plotting.plot_2D_farfield(
        theta_grid_trans, r_grid_trans, efield_trans, simulation_parameters, \
        transmitted=True, ray_tracing_matrix=ray_tracing, \
        **{**plot_parameters, 'figname': figname_trans})

    figname_refl = os.path.join(movie_name, 'refl', f'frame_{param_ind:04d}.png')
    ott_plotting.plot_2D_farfield(
        theta_grid_refl, r_grid_refl, efield_refl, simulation_parameters, \
        transmitted=False, ray_tracing_matrix=ray_tracing, \
        **{**plot_parameters, 'figname': figname_refl})



    figname_trans_3D = os.path.join(movie_name, 'trans_3d', \
                                    f'frame_{param_ind:04d}.png')
    ott_plotting.plot_3D_farfield(
        theta_grid_trans, r_grid_trans, efield_trans, simulation_parameters, \
        transmitted=True, ray_tracing_matrix=ray_tracing, \
        **{**plot_parameters, 'figname': figname_trans_3D})

    figname_refl_3D = os.path.join(movie_name, 'refl_3d', \
                                   f'frame_{param_ind:04d}.png')
    ott_plotting.plot_3D_farfield(
        theta_grid_refl, r_grid_refl, efield_refl, simulation_parameters, \
        transmitted=True, ray_tracing_matrix=ray_tracing, \
        **{**plot_parameters, 'figname': figname_refl_3D})

