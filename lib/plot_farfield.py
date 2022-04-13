import os, sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from skimage.restoration import unwrap_phase
import unwrap

import scipy.constants as constants

import bead_util as bu

plt.rcParams.update({'font.size': 14})



try:
    radius = float(sys.argv[1])
    n_particle = float(sys.argv[2])
    NA = float(sys.argv[3])
    xOffset = float(sys.argv[4])
    yOffset = float(sys.argv[5])
    zOffset = float(sys.argv[6])
    sim_id = f'r{radius*1e6:0.2f}um_n{n_particle:0.2f}_na{NA:0.3f}_' + \
                f'x{xOffset*1e6:0.2f}_y{yOffset*1e6:0.2f}_z{zOffset*1e6:0.2f}'
    sim_id = sim_id.replace('.', '_')
except:
    xOffset = 0.0e-6
    yOffset = 0.0e-6
    zOffset = 0.0e-6
    sim_id = 'r3_76um_n1_39_na0_095_x0_00_y0_00_z0_00'


try:
    transmitted = int(sys.argv[7])
    beam = str(sys.argv[8])
    rmax = float(sys.argv[9])
    plot_sin_approx_breakdown = int(sys.argv[10])
    view_elev = float(sys.argv[11])
    view_azim = float(sys.argv[12])
except:
    transmitted = True
    beam = 'tot'
    rmax = 0.05
    plot_sin_approx_breakdown = False
    view_elev = +40.0
    view_azim = 20.0


try:
    plot_base = str(sys.argv[13])
    figname = str(sys.argv[14])
    save_fig = int(sys.argv[15])
    show_fig = int(sys.argv[16])
except:
    plot_base = ''
    figname = ''
    save_fig = False
    show_fig = True

if not len(plot_base):
    plot_base = '/Users/manifestation/Stanford/beads/plots/ott_farfield/'


##########################################################################
#####################      DATA      #####################################
##########################################################################

### These datafiles should be sampled on a regular (theta, phi) grid for 
### this to work properly
data_pts = \
    np.loadtxt(f'../raw_data/{sim_id}/farfield_points.txt', delimiter=',')

tot_real = \
    np.loadtxt(f'../raw_data/{sim_id}/farfield_tot_real.txt', delimiter=',')
tot_imag = \
    np.loadtxt(f'../raw_data/{sim_id}/farfield_tot_imag.txt', delimiter=',')

inc_real = \
    np.loadtxt(f'../raw_data/{sim_id}/farfield_inc_real.txt', delimiter=',')
inc_imag = \
    np.loadtxt(f'../raw_data/{sim_id}/farfield_inc_imag.txt', delimiter=',')

scat_real = \
    np.loadtxt(f'../raw_data/{sim_id}/farfield_scat_real.txt', delimiter=',')
scat_imag = \
    np.loadtxt(f'../raw_data/{sim_id}/farfield_scat_imag.txt', delimiter=',')




##########################################################################
#####################     OPTIONS      ###################################
##########################################################################

# print(beam)
# beam='inc'

# save_fig = True
# show_fig = True

# plot_3d_sphere = True
plot_3d_sphere = False

# manual_phase_plot_lims = (-np.pi, np.pi)
max_radiance_val = 8.4

manual_phase_plot_lims = ()
# max_radiance_val = 0.0


##########################################################################
#####################   RAY TRACING      #################################
##########################################################################

### Simple optical system with meridional ray tracing, given we make the
### assumption of a point source in this farfield visualization.
###
### The parabolic mirror and all lens have been assumed ideal for this 
### simple treatment. Zemax will do better eventually

T1 = np.array([[1, 0], [50.8e-2, 1]])     # Propagation to parabolic mirror
La = np.array([[1, -1/50.8e-2], [0, 1]])  # Recollimation
T2 = np.array([[1, 0], [90.8e-3, 1]])     # To telescope
Lb = np.array([[1, -1/40.0e-2], [0, 1]])  # First telescope lens
T3 = np.array([[1, 0], [50.0e-2, 1]])     # refracting telescope configuration
Lc = np.array([[1, -1/10.0e-2], [0, 1]])  # First telescope lens
T4 = np.array([[1, 0], [10.8e-2, 1]])     # projection onto QPD

ray_tracing_matrix = T4 @ Lc @ T3 @ Lb @ T2 @ La @ T1




##########################################################################
###########    SELECTING, RE-CASTING, AND NORMALZING DATA    #############
##########################################################################

all_theta = data_pts[0]
all_phi = data_pts[1]


### Select the appropriate hemisphere and find all the unique values
### of the azimuthal angle
theta_max = 1.05*np.abs(rmax/ray_tracing_matrix[1,0])
if transmitted:
    theta_inds = (all_theta < np.pi/2) * (all_theta < theta_max)
    thetapts = np.unique(all_theta[theta_inds])
else:
    theta_inds = all_theta > np.pi/2 * ((np.pi - all_theta) < theta_max)
    thetapts = np.pi - np.unique(all_theta[theta_inds])

# theta_inds = (all_theta == all_theta)

### Index all the data for our desired hemisphere
inc_real = inc_real[:,theta_inds]
inc_imag = inc_imag[:,theta_inds]
scat_real = scat_real[:,theta_inds]
scat_imag = scat_imag[:,theta_inds]
tot_real = tot_real[:,theta_inds]
tot_imag = tot_imag[:,theta_inds]

### Build meshgrids from the MATLAB-exported data
phipts = np.unique(all_phi)
rpts = np.abs(ray_tracing_matrix[1,0] * thetapts)

delta_phi = np.abs(np.mean(np.diff(phipts)))
delta_r = np.abs(np.mean(np.diff(rpts)))

theta_grid, phi_grid = np.meshgrid(thetapts, phipts, indexing='ij')
r_grid, phi_grid = np.meshgrid(rpts, phipts, indexing='ij')
resid_angle_grid = ray_tracing_matrix[0,0] * theta_grid

### Initialize grid-like arrays for the re-cast electric field
ntheta = len(thetapts)
nphi = len(phipts)
efield_inc = np.zeros((ntheta, nphi, 3), dtype=np.complex128)
efield_scat = np.zeros((ntheta, nphi, 3), dtype=np.complex128)
efield_tot = np.zeros((ntheta, nphi, 3), dtype=np.complex128)

### Extract the single column data to the gridded format
for i in range(ntheta):
    for k in range(nphi):
        efield_inc[i,k,:] = inc_real[:,k*ntheta+i] \
                                + 1.0j * inc_imag[:,k*ntheta+i]
        efield_scat[i,k,:] = scat_real[:,k*ntheta+i] \
                                + 1.0j * scat_imag[:,k*ntheta+i]
        efield_tot[i,k,:] = tot_real[:,k*ntheta+i] \
                                + 1.0j * tot_imag[:,k*ntheta+i]


### Select which field we want to look at (could probably do this earlier)
if beam == 'inc':
    efield_plot = efield_inc
    title = 'Incident Gaussian Beam'
elif beam == 'scat':
    efield_plot = efield_scat
    title = 'Scattered Beam'
else:
    efield_plot = efield_tot
    title = 'Total Beam'

if transmitted:
    title += ', Transmitted Hemisphere'
else:
    title += ', Back-reflected Hemisphere'


### HARDCODED ASSUMPTION THAT FIELD IS LINEARLY POLARIZED IN X
efield_plot_x = efield_plot[:,:,1] * np.cos(theta_grid) * np.cos(phi_grid) \
                + efield_plot[:,:,2] * (-1) * np.sin(phi_grid)

efield_plot_y = efield_plot[:,:,1] * np.cos(theta_grid) * np.sin(phi_grid) \
                + efield_plot[:,:,2] * np.cos(phi_grid)

radiance = np.sqrt( np.abs(efield_plot[:,:,0])**2 \
                    + np.abs(efield_plot[:,:,1])**2 \
                    + np.abs(efield_plot[:,:,2])**2 )

radiancex = np.abs(efield_plot_x[:,:])
phasex = np.angle(efield_plot_x[:,:])
phasex_unwrap = unwrap.unwrap(phasex)

radiancey = np.abs(efield_plot_y[:,:])
phasey = np.angle(efield_plot_y[:,:])
phasey_unwrap = unwrap.unwrap(phasey)

if transmitted:
    center_phasex = np.mean(phasex_unwrap[0,:])
    center_phasey = np.mean(phasey_unwrap[0,:])
else:
    center_phasex = np.mean(phasex_unwrap[-1,:])
    center_phasey = np.mean(phasey_unwrap[-1,:])

phasex_unwrap -= center_phasex
phasey_unwrap -= center_phasey

if not max_radiance_val:
    max_radiance_val = np.max([np.max(radiancex), np.max(radiancey)])


# max_val = np.max(np.abs(efield_inc).flatten())
# radiancex *= 1.0 / max_val
# max_val = 1.0

# eta = 1.0 / (constants.c * constants.epsilon_0)
# k = 2.0 * np.pi / (1064.0e-9)
# integral = np.sum(radiancex**2 * r_grid * delta_r * delta_phi)
# integral2 = np.sum(radiance**2 * r_grid * delta_r * delta_phi)
# print(eta)
# print(integral, integral2, integral/constants.c, integral2*eta/(2.0*np.pi))
# input()







##########################################################################
##########################    2D PLOTTING    #############################
##########################################################################

fig1, axarr = plt.subplots(1, 2, figsize=(12,6), sharex=True, sharey=True,\
                           subplot_kw=dict(projection='polar'))

if title:
    fig1.suptitle('Image from Output Optics: ' + title, \
                  fontsize=16, fontweight='bold')

### Plot a contour for the sine approximation breakdown (with a label)
### if so desired
if plot_sin_approx_breakdown:
    derp_phi = np.linspace(0, 2.0*np.pi, 1000)
    derp_r = np.abs(np.pi/6.0 * ray_tracing_matrix[1,0]) * np.ones(1000)
    for i in range(2):
        line = axarr[i].plot(derp_phi, derp_r, ls='--', lw=3, \
                             color='w', zorder=4)
        bu.labelLine(axarr[i].get_lines()[0], 3*np.pi/2, \
                     y_offset=-0.005, label='$\\pi/6$ half-cone', \
                     va='bottom', zorder=99)

### Plot the radiance and phase of the electric field specifically 
### definining the filled contour levels to allow for algorithmic 
### construction of the colorbars
rad_levels = np.linspace(0, max_radiance_val, 501)
# rad_cont = axarr[0].contourf(phi_grid, r_grid, radiancex, \
#                              levels=rad_levels, \
#                              cmap='plasma', zorder=3, )
rad_cont = axarr[0].pcolormesh(phi_grid, r_grid, radiancex, \
                               vmin=0, vmax=max_radiance_val, \
                               cmap='plasma', zorder=3, \
                               shading='gouraud')
# phase_levels = np.linspace(-np.pi, np.pi, 501)
# phase_cont = axarr[1].contourf(phi_grid, r_grid, phasex, \
#                                levels=phase_levels, \
#                                cmap='plasma', zorder=3)
inds = r_grid < rmax
if len(manual_phase_plot_lims):
    min_phase = np.floor(manual_phase_plot_lims[0]/np.pi)
    max_phase = np.ceil(manual_phase_plot_lims[1]/np.pi)
else:
    min_phase = np.floor(np.min(phasex_unwrap[inds])/np.pi)
    max_phase = np.ceil(np.max(phasex_unwrap[inds])/np.pi)
phase_levels = np.linspace(min_phase, max_phase, 501)*np.pi
phase_cont = axarr[1].contourf(phi_grid, r_grid, phasex_unwrap, \
                               levels=phase_levels, \
                               cmap='plasma', zorder=3)
phase_ticks = []
phase_ticklabels = []
for i in range(int(max_phase - min_phase)+1):
    phase_val = min_phase + i
    phase_ticks.append(phase_val*np.pi)
    if not phase_val:
        phase_ticklabels.append('0')
    elif phase_val == 1:
        phase_ticklabels.append('$\\pi$')
    elif phase_val == -1:
        phase_ticklabels.append('$-\\pi$')
    else:
        phase_ticklabels.append(f'{int(phase_val):d}$\\pi$')

### Clean up the axes and labels that we don't really care about
for i in range(2):
    axarr[i].set_rmax(rmax)
    axarr[i].set_yticks([rmax])
    axarr[i].set_yticklabels([])
    axarr[i].set_xticks([])
    axarr[i].grid(False)

### Add a note with the plotted value of rmax, i.e. the size of the
### circular aperture displayed at the end
fig1.text(0.5, 0.1, f'{100*rmax:0.1f} cm radius\naperture', fontsize=16, \
          ha='center', va='center')

### Add a note with the relative positions of beam and scatterer, noting
### that the offset in the filename/simulation is BEAM relative to the
### MS at the origin, so that we need to invert the coordinates. I also
### want consistent sizing so the plots can be combined into a movie
val_str = 'MS position:\n('
for var in [xOffset, yOffset, zOffset]:
    if var > 0:
        sign_str = '-'
    else:
        sign_str = ' '
    val_str += sign_str + f'{np.abs(var)*1e6:0.2f}, '
val_str = val_str[:-2] + ')'

ms_label = fig1.text(0.5, 0.85, f'{val_str} $\\mu$m', \
                     fontsize=12, ha='center', va='center')
ms_label.set(fontfamily='monospace')

### These labels need to have the same vertical extent otherwise they
### fuck up with the axis sizing
axarr[0].set_title('Radiance')
axarr[1].set_title('Phase')

### Do a tight_layout(), but then pull in the sides of the figure a bit
### to make room for colorbars
fig1.tight_layout()
fig1.subplots_adjust(left=0.075, right=0.925, top=0.85, bottom=0.05)

### Make the colorbars explicitly first by defining and inset axes
### and then plotting the colorbar in the new inset
rad_inset = inset_axes(axarr[0], width="4%", height="85%", \
                       loc='center left', \
                       bbox_to_anchor=(-0.07, 0, 1, 1), \
                       bbox_transform=axarr[0].transAxes, \
                       borderpad=0)
nlabel = 5
rad_ticks = np.linspace(0, max_radiance_val, nlabel)
rad_cbar = fig1.colorbar(rad_cont, cax=rad_inset, \
                         ticks=rad_ticks, format='%0.1f')
rad_inset.yaxis.set_ticks_position('left')

### Same thing for the phase
phase_inset = inset_axes(axarr[1], width="4%", height="85%", \
                         loc='center right', \
                         bbox_to_anchor=(0.07, 0, 1, 1), \
                         bbox_transform=axarr[1].transAxes, \
                         borderpad=0)
# phase_cbar = fig1.colorbar(phase_cont, cax=phase_inset, \
#                            ticks=[-np.pi, 0, np.pi])
# phase_cbar.ax.set_yticklabels(['$-\\pi$', '0', '$+\\pi$'])
phase_cbar = fig1.colorbar(phase_cont, cax=phase_inset, ticks=phase_ticks)
phase_cbar.ax.set_yticklabels(phase_ticklabels)

if save_fig:
    if not len(figname):
        if transmitted:
            figname = f'{sim_id}/{beam}beam_output_image.svg'
        else:
            figname = f'{sim_id}/{beam}beam_reflected_output_image.svg'
    savepath = os.path.join(plot_base, figname)
    print('Saving figure to:')
    print(f'     {savepath}')
    print()
    bu.make_all_pardirs(savepath,confirm=False)
    fig1.savefig(savepath, dpi=150)


plt.figure()
plt.plot(r_grid[:,0], phasex_unwrap[:,0])
plt.show()
















##########################################################################
##########################    2D PLOTTING    #############################
##########################################################################

fig2, axarr2 = plt.subplots(1, 2, figsize=(12,6), sharex=True, sharey=True,\
                           subplot_kw=dict(projection='polar'))

if title:
    fig2.suptitle('Image from Output Optics: ' + title, \
                  fontsize=16, fontweight='bold')

### Plot a contour for the sine approximation breakdown (with a label)
### if so desired
if plot_sin_approx_breakdown:
    derp_phi = np.linspace(0, 2.0*np.pi, 1000)
    derp_r = np.abs(np.pi/6.0 * ray_tracing_matrix[1,0]) * np.ones(1000)
    for i in range(2):
        line = axarr2[i].plot(derp_phi, derp_r, ls='--', lw=3, \
                             color='w', zorder=4)
        bu.labelLine(axarr2[i].get_lines()[0], 3*np.pi/2, \
                     y_offset=-0.005, label='$\\pi/6$ half-cone', \
                     va='bottom', zorder=99)

### Plot the radiance and phase of the electric field specifically 
### definining the filled contour levels to allow for algorithmic 
### construction of the colorbars
rad_levels2 = np.linspace(0, max_radiance_val, 501)
rad_cont2 = axarr2[0].pcolormesh(phi_grid, r_grid, radiancey, \
                                 vmin=0, vmax=max_radiance_val, \
                                 cmap='plasma', zorder=3, \
                                 shading='gouraud')
# phase_levels = np.linspace(-np.pi, np.pi, 501)
# phase_cont = axarr[1].contourf(phi_grid, r_grid, phasex, \
#                                levels=phase_levels, \
#                                cmap='plasma', zorder=3)
inds = r_grid < rmax
if len(manual_phase_plot_lims):
    min_phase = np.floor(manual_phase_plot_lims[0]/np.pi)
    max_phase = np.ceil(manual_phase_plot_lims[1]/np.pi)
else:
    min_phase = np.floor(np.min(phasey_unwrap[inds])/np.pi)
    max_phase = np.ceil(np.max(phasey_unwrap[inds])/np.pi)
phase_levels2 = np.linspace(min_phase, max_phase, 501)*np.pi
phase_cont2 = axarr2[1].contourf(phi_grid, r_grid, phasey_unwrap, \
                                 levels=phase_levels2, \
                                 cmap='plasma', zorder=3)
phase_ticks = []
phase_ticklabels = []
for i in range(int(max_phase - min_phase)+1):
    phase_val = min_phase + i
    phase_ticks.append(phase_val*np.pi)
    if not phase_val:
        phase_ticklabels.append('0')
    elif phase_val == 1:
        phase_ticklabels.append('$\\pi$')
    elif phase_val == -1:
        phase_ticklabels.append('$-\\pi$')
    else:
        phase_ticklabels.append(f'{int(phase_val):d}$\\pi$')

### Clean up the axes and labels that we don't really care about
for i in range(2):
    axarr2[i].set_rmax(rmax)
    axarr2[i].set_yticks([rmax])
    axarr2[i].set_yticklabels([])
    axarr2[i].set_xticks([])
    axarr2[i].grid(False)

### Add a note with the plotted value of rmax, i.e. the size of the
### circular aperture displayed at the end
fig2.text(0.5, 0.1, f'{100*rmax:0.1f} cm radius\naperture', fontsize=16, \
          ha='center', va='center')

### Add a note with the relative positions of beam and scatterer, noting
### that the offset in the filename/simulation is BEAM relative to the
### MS at the origin, so that we need to invert the coordinates. I also
### want consistent sizing so the plots can be combined into a movie
val_str = 'MS position:\n('
for var in [xOffset, yOffset, zOffset]:
    if var > 0:
        sign_str = '-'
    else:
        sign_str = ' '
    val_str += sign_str + f'{np.abs(var)*1e6:0.2f}, '
val_str = val_str[:-2] + ')'

ms_label = fig2.text(0.5, 0.85, f'{val_str} $\\mu$m', \
                     fontsize=12, ha='center', va='center')
ms_label.set(fontfamily='monospace')

### These labels need to have the same vertical extent otherwise they
### fuck up with the axis sizing
axarr2[0].set_title('Radiance')
axarr2[1].set_title('Phase')

### Do a tight_layout(), but then pull in the sides of the figure a bit
### to make room for colorbars
fig2.tight_layout()
fig2.subplots_adjust(left=0.075, right=0.925, top=0.85, bottom=0.05)

### Make the colorbars explicitly first by defining and inset axes
### and then plotting the colorbar in the new inset
rad_inset = inset_axes(axarr2[0], width="4%", height="85%", \
                       loc='center left', \
                       bbox_to_anchor=(-0.07, 0, 1, 1), \
                       bbox_transform=axarr2[0].transAxes, \
                       borderpad=0)
nlabel = 5
rad_ticks = np.linspace(0, max_radiance_val, nlabel)
rad_cbar = fig2.colorbar(rad_cont2, cax=rad_inset, \
                         ticks=rad_ticks, format='%0.1f')
rad_inset.yaxis.set_ticks_position('left')

### Same thing for the phase
phase_inset = inset_axes(axarr2[1], width="4%", height="85%", \
                         loc='center right', \
                         bbox_to_anchor=(0.07, 0, 1, 1), \
                         bbox_transform=axarr2[1].transAxes, \
                         borderpad=0)
# phase_cbar = fig1.colorbar(phase_cont, cax=phase_inset, \
#                            ticks=[-np.pi, 0, np.pi])
# phase_cbar.ax.set_yticklabels(['$-\\pi$', '0', '$+\\pi$'])
phase_cbar = fig2.colorbar(phase_cont2, cax=phase_inset, ticks=phase_ticks)
phase_cbar.ax.set_yticklabels(phase_ticklabels)

if save_fig:
    if not len(figname):
        if transmitted:
            figname = f'{sim_id}/{beam}beam_output_image_2.png'
        else:
            figname = f'{sim_id}/{beam}beam_reflected_output_image_2.png'
    figname = figname.replace('.svg', '_2.svg')
    figname = figname.replace('.png', '_2.png')
    savepath = os.path.join(plot_base, figname)
    print('Saving figure to:')
    print(f'     {savepath}')
    print()
    bu.make_all_pardirs(savepath,confirm=False)
    fig2.savefig(savepath, dpi=150)
















##########################################################################
##########################    3D PLOTTING    #############################
##########################################################################

if plot_3d_sphere:

    fig_3d, axarr_3d = plt.subplots(1, 2, figsize=(12,6), \
                                    subplot_kw=dict(projection='3d'))

    if title:
        fig_3d.suptitle('Far-field OTT Output: ' + title, \
                        fontsize=16, fontweight='bold')

    for i in range(2):
        axarr_3d[i].set_box_aspect([1,1,0.5])

    ### Basically, we're going to plot a spherical surface and then color
    ### it based on the intensity
    X = np.sin(theta_grid) * np.cos(phi_grid)
    Y = np.sin(theta_grid) * np.sin(phi_grid)
    Z = np.cos(theta_grid)

    ### Build the colormap for the radiance
    radiancex_norm = colors.Normalize(vmin=0, vmax=np.max(radiancex.flatten()))
    radiancex_smap = cm.ScalarMappable(norm=radiancex_norm, cmap='plasma')
    radiancex_colors = radiancex_smap.to_rgba(radiancex)

    ### Plot the radiance
    radiancex_surf = axarr_3d[0].plot_surface(X, Y, Z, rstride=1, cstride=1, \
                           facecolors=radiancex_colors, \
                           linewidth=0, antialiased=False, shade=False)

    ### Build the colormap for the phase
    phasex_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    phasex_smap = cm.ScalarMappable(norm=phasex_norm, cmap='plasma')
    phasex_colors = phasex_smap.to_rgba(phasex)

    ### plot the phase
    phasex_surf = axarr_3d[1].plot_surface(X, Y, Z, rstride=1, cstride=1, \
                           facecolors=phasex_colors, \
                           linewidth=0, antialiased=False, shade=False)

    ### Label some stuff and set a defined viewing angle
    for i in range(2):
        axarr_3d[i].set_xlabel('X')
        axarr_3d[i].set_ylabel('Y')
        axarr_3d[i].set_zlabel('Z')
        axarr_3d[i].view_init(elev=view_elev, azim=view_azim)

    axarr_3d[0].set_title('Radiance: $|\\vec{E}|^2$')
    axarr_3d[1].set_title('Phase: $\\angle \\vec{E}$')

    if save_fig:
        if transmitted:
            figname_3d = f'{sim_id}/{beam}beam_hemisphere_projection.svg'
        else:
            figname_3d = f'{sim_id}/{beam}beam_reflected_hemisphere_projection.svg'

        savepath_3d = os.path.join(plot_base, figname_3d)
        print('Saving figure to:')
        print(f'     {savepath_3d}')
        print()
        fig_3d.savefig(savepath_3d, dpi=150)


if show_fig:
    plt.show()


