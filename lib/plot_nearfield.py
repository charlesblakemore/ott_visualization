import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})



# sim_id = 'r3_76um_n1_39_na0_095_x0_00_y0_00_z-10_00'

# sim_id = 'r3_76um_n1_39_na0_500_x0_00_y0_00_z0_00'
sim_id = 'r3_76um_n1_39_na0_500_x0_00_y0_00_z3_00'
# sim_id = 'r3_76um_n1_39_na0_500_x2_00_y0_00_z2_00'
# sim_id = 'r3_76um_n1_39_na0_500_x0_00_y0_00_z10_00'
# sim_id = 'r3_76um_n1_39_na0_500_x0_00_y0_00_z-10_00'

data_pts = np.loadtxt(f'../raw_data/{sim_id}/nearfield_points.txt', delimiter=',')

tot_real = np.loadtxt(f'../raw_data/{sim_id}/nearfield_tot_real.txt', delimiter=',')
tot_imag = np.loadtxt(f'../raw_data/{sim_id}/nearfield_tot_imag.txt', delimiter=',')

inc_real = np.loadtxt(f'../raw_data/{sim_id}/nearfield_inc_real.txt', delimiter=',')
inc_imag = np.loadtxt(f'../raw_data/{sim_id}/nearfield_inc_imag.txt', delimiter=',')

scat_real = np.loadtxt(f'../raw_data/{sim_id}/nearfield_scat_real.txt', delimiter=',')
scat_imag = np.loadtxt(f'../raw_data/{sim_id}/nearfield_scat_imag.txt', delimiter=',')

int_real = np.loadtxt(f'../raw_data/{sim_id}/nearfield_int_real.txt', delimiter=',')
int_imag = np.loadtxt(f'../raw_data/{sim_id}/nearfield_int_imag.txt', delimiter=',')

block_sphere = False
plot_sphere = True
rbead = 3.76e-6

figsize=(10,6)

##########################################################################
##########################################################################
##########################################################################


all_x = data_pts[0]
all_z = data_pts[2]

xpts = np.unique(all_x)
zpts = np.unique(all_z)

nx = len(xpts)
nz = len(zpts)

efield_inc = np.zeros((nx, nz, 3), dtype=np.complex128)
efield_int = np.zeros((nx, nz, 3), dtype=np.complex128)
efield_scat = np.zeros((nx, nz, 3), dtype=np.complex128)
efield_tot = np.zeros((nx, nz, 3), dtype=np.complex128)

efield_derp = np.zeros((nx, nz, 3), dtype=np.complex128)

# plt.hist(np.abs(efield).flatten(), 100)
# plt.show()

# plt.hist(efield.real.flatten(), 100)
# plt.show()

for i in range(nx):
    for k in range(nz):
        internal = ( (xpts[i]**2 + zpts[k]**2) < (rbead)**2 ) 
        efield_int[i,k,:] = int_real[:,k*nx+i] + 1.0j * int_imag[:,k*nx+i]
        if internal:
            efield_inc[i,k,:] = 0.0 + 1.0j * 0.0
            efield_scat[i,k,:] = efield_int[i,k,:]
            efield_tot[i,k,:] = efield_int[i,k,:]
        else:
            efield_inc[i,k,:] = inc_real[:,k*nx+i] + 1.0j * inc_imag[:,k*nx+i]
            efield_scat[i,k,:] = scat_real[:,k*nx+i] + 1.0j * scat_imag[:,k*nx+i]
            efield_tot[i,k,:] = tot_real[:,k*nx+i] + 1.0j * tot_imag[:,k*nx+i]

efield_derp = efield_tot - 2*efield_scat - efield_inc

# abs_min = np.min(np.abs(efield).flatten())
# abs_max = np.max(np.abs(efield).flatten())

inc_min = np.min(np.abs(efield_inc).flatten())
inc_max = np.max(np.abs(efield_inc).flatten())

scat_min = np.min(np.abs(efield_scat).flatten())
scat_max = np.max(np.abs(efield_scat).flatten())

tot_min = np.min(np.abs(efield_tot).flatten())
tot_max = np.max(np.abs(efield_tot).flatten())

derp_min = np.min(np.abs(efield_derp).flatten())
derp_max = np.max(np.abs(efield_derp).flatten())

if np.abs(inc_min) > np.abs(inc_max):
    inc_max = -1.0 * inc_min
else:
    inc_min = -1.0 * inc_max

if np.abs(scat_min) > np.abs(scat_max):
    scat_max = -1.0 * scat_min
else:
    scat_min = -1.0 * scat_max

if np.abs(tot_min) > np.abs(tot_max):
    tot_max = -1.0 * tot_min
else:
    tot_min = -1.0 * tot_max

if np.abs(derp_min) > np.abs(derp_max):
    derp_max = -1.0 * derp_min
else:
    derp_min = -1.0 * derp_max

all_max = np.max([derp_max, inc_max, scat_max, tot_max])
for field in [efield_derp, efield_inc, efield_scat, efield_tot]:
    field *= (1.0 / all_max)

fig, axarr = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
cbar_ax = fig.add_axes([0.90, 0.1, 0.025, 0.8])
axarr[1].set_title('Incident Field')
Cs = []
for i in [0,1,2]:
    # Cs.append( axarr[i].contourf(xpts, zpts, efield_inc[:,:,i].T.real, 100, cmap='RdBu', 
    #                   vmin=inc_min, vmax=inc_max) )
    Cs.append( axarr[i].contourf(xpts, zpts, efield_inc[:,:,i].T.real, 100, cmap='RdBu', 
                      vmin=-1, vmax=1) )
    axarr[i].set_aspect('equal')
    if plot_sphere:
        bead = plt.Circle((0,0), rbead, ls=':', lw=3, ec='k', fc='none')
        axarr[i].add_patch(bead)
fig.colorbar(Cs[0], cax=cbar_ax)#, vmin=inc_min, vmax=inc_max)
fig.tight_layout()
fig.subplots_adjust(right=0.85)


fig2, axarr2 = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
cbar_ax2 = fig2.add_axes([0.90, 0.1, 0.025, 0.8])
axarr2[1].set_title('Scattered Field')
Cs2 = []
for i in [0,1,2]:
    # Cs2.append( axarr2[i].contourf(xpts, zpts, efield_scat[:,:,i].T.real, 100, cmap='RdBu',
    #                    vmin=scat_min, vmax=scat_max) )
    Cs2.append( axarr2[i].contourf(xpts, zpts, efield_scat[:,:,i].T.real, 100, cmap='RdBu',
                       vmin=-1, vmax=1) )
    axarr2[i].set_aspect('equal')
    if plot_sphere:
        bead = plt.Circle((0,0), rbead, ls=':', lw=3, ec='k', fc='none')
        axarr2[i].add_patch(bead)
fig2.colorbar(Cs2[0], cax=cbar_ax2)#, vmin=inc_min, vmax=inc_max)
fig2.tight_layout()
fig2.subplots_adjust(right=0.85)


fig3, axarr3 = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
cbar_ax3 = fig3.add_axes([0.90, 0.1, 0.025, 0.8])
axarr3[1].set_title('"Total" Field')
Cs3 = []
for i in [0,1,2]:
    # Cs3.append( axarr3[i].contourf(xpts, zpts, efield_tot[:,:,i].T.real, 100, cmap='RdBu',
    #                    vmin=tot_min, vmax=tot_max) )
    Cs3.append( axarr3[i].contourf(xpts, zpts, efield_tot[:,:,i].T.real, 100, cmap='RdBu',
                       vmin=-1, vmax=1) )
    axarr3[i].set_aspect('equal')
    if plot_sphere:
        bead = plt.Circle((0,0), rbead, ls=':', lw=3, ec='k', fc='none')
        axarr3[i].add_patch(bead)
fig3.colorbar(Cs3[0], cax=cbar_ax3)#, vmin=inc_min, vmax=inc_max)
fig3.tight_layout()
fig3.subplots_adjust(right=0.85)


fig4, axarr4 = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
cbar_ax4 = fig4.add_axes([0.90, 0.1, 0.025, 0.8])
axarr4[1].set_title('"Derp" Field')
Cs4 = []
for i in [0,1,2]:
    # Cs4.append( axarr4[i].contourf(xpts, zpts, efield_derp[:,:,i].T.real, 100, cmap='RdBu',
    #                    vmin=derp_min, vmax=derp_max) )
    Cs4.append( axarr4[i].contourf(xpts, zpts, efield_derp[:,:,i].T.real, 100, cmap='RdBu',
                       vmin=-1, vmax=1) )
    axarr4[i].set_aspect('equal')
    if plot_sphere:
        bead = plt.Circle((0,0), rbead, ls=':', lw=3, ec='k', fc='none')
        axarr4[i].add_patch(bead)
fig4.colorbar(Cs4[0], cax=cbar_ax4)#, vmin=inc_min, vmax=inc_max)
fig4.tight_layout()
fig4.subplots_adjust(right=0.85)

plt.show()


