import cmcrameri.cm as cmc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from plotcosmomap import plotcosmo_notick, pole

###############################################################################
# Data
###############################################################################
# ds = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS11_ctrl/3h/ASOD_T/01_ASOD_T.nc')
# PD = np.nanmean(ds.variables['ASOD_T'][...], axis=0)
# ds.close()
# ds = xr.open_dataset('/scratch/snx3000/rxiang/data/cosmo/EAS11_lgm/3h/ASOD_T/01_ASOD_T.nc')
# LGM = np.nanmean(ds.variables['ASOD_T'][...], axis=0)
# ds.close()

ds = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn/ASOD_T/01_ASOD_T_DJF.nc')
PD = np.nanmean(ds.variables['ASOD_T'][...], axis=0)
ds.close()
ds = xr.open_dataset('/scratch/snx3000/rxiang/data/cosmo/EAS11_lgm/szn/ASOD_T/01_ASOD_T_DJF.nc')
LGM = np.nanmean(ds.variables['ASOD_T'][...], axis=0)
ds.close()

###############################################################################
# %% Plot
###############################################################################
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()
rlon_, rlat_ = np.meshgrid(rlon, rlat)

levels = MaxNLocator(nbins=31).tick_values(100, 400)
cmap = cmc.roma_r
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

levels2 = MaxNLocator(nbins=20).tick_values(-5, 5)
cmap2 = cmc.vik
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

fig = plt.figure(figsize=(11, 3))

gs = gridspec.GridSpec(1, 3, left=0.05, bottom=0.07, right=0.98,
                       top=0.97, hspace=0.05, wspace=0.05, width_ratios=[1, 1, 1])

axs, cs = np.empty(shape=(1, 3), dtype='object'), np.empty(shape=(1, 3), dtype='object')
axs[0, 0] = fig.add_subplot(gs[0], projection=rot_pole_crs)
axs[0, 0] = plotcosmo_notick(axs[0, 0])
cs[0, 0] = axs[0, 0].pcolormesh(rlon, rlat, PD, shading="auto", cmap=cmap, norm=norm)
axs[0, 0].set_title("(a) PD", fontsize=13, loc='left')

axs[0, 1] = fig.add_subplot(gs[1], projection=rot_pole_crs)
axs[0, 1] = plotcosmo_notick(axs[0, 1])
cs[0, 1] = axs[0, 1].pcolormesh(rlon, rlat, LGM, shading="auto", cmap=cmap, norm=norm)
axs[0, 1].set_title("(b) LGM", fontsize=13, loc='left')

axs[0, 0].text(0.5, -0.13, '[W m$^{-2}$]', ha='left', va='top', transform=axs[0, 1].transAxes, fontsize=13)

axs[0, 2] = fig.add_subplot(gs[2], projection=rot_pole_crs)
axs[0, 2] = plotcosmo_notick(axs[0, 2])
cs[0, 2] = axs[0, 2].pcolormesh(rlon, rlat, LGM - PD, shading="auto", cmap=cmap2, norm=norm2)
axs[0, 2].set_title("(c) LGM - PD", fontsize=13, loc='left')

cax = fig.add_axes([axs[0, 0].get_position().x0+0.155, axs[0, 1].get_position().y0 - 0.13, axs[0, 1].get_position().width, 0.05])
cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='horizontal', extend='both', ticks=np.linspace(100, 400, 7, endpoint=True))
cbar.ax.tick_params(labelsize=13)

cax = fig.add_axes([axs[0, 2].get_position().x0, axs[0, 2].get_position().y0 - 0.13, axs[0, 2].get_position().width, 0.05])
cbar = fig.colorbar(cs[0, 2], cax=cax, orientation='horizontal', extend='both')
cbar.ax.tick_params(labelsize=13)

axs[0, 0].text(-0.008, 0.95, '50°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=13)
axs[0, 0].text(-0.008, 0.77, '40°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=13)
axs[0, 0].text(-0.008, 0.59, '30°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=13)
axs[0, 0].text(-0.008, 0.41, '20°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=13)
axs[0, 0].text(-0.008, 0.23, '10°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=13)
axs[0, 0].text(-0.008, 0.05, '0°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=13)

for i in range(3):
    axs[0, i].text(0.12, -0.02, '80°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)
    axs[0, i].text(0.32, -0.02, '100°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)
    axs[0, i].text(0.52, -0.02, '120°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)
    axs[0, i].text(0.72, -0.02, '140°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)
    axs[0, i].text(0.92, -0.02, '160°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)

fig.suptitle('TOA Solar downward radiation averaged during year 2001 DJF', fontsize=14)

plt.show()

plotpath = "/project/pr133/rxiang/figure/paper2/setup/"
fig.savefig(plotpath + 'lpalorb_DJF.png', dpi=500, transparent=True)
