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
# PD dust
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_EAS_ext_12km_merit_unmod_topo.nc')
dust_PD = ds.variables['AER_DUST12'][6, ...]
rlon_ = ds['rlon'].values
rlat_ = ds['rlat'].values
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_EAS_ext_12km_merit_lgm_dust_adj.nc')
dust_LGM = ds.variables['AER_DUST12'][6, ...]
ds.close()

PD_, LGM_ = [], []
for i in ("00090100", "00100100", "00110100", "00120100", "01010100", "01020100", "01030100", "01040100", "01050100", "01060100", "01070100", "01080100"):
    ds = xr.open_dataset(f'/store/c2sm/pr04/rxiang/data_lmp/{i}_EAS11_ctrl/lm_coarse/3h/ASOB_T.nc')
    dt = np.nanmean(ds.variables['ASOB_T'][...], axis=0)
    PD_.append(dt)
    ds.close()
    ds = xr.open_dataset(f'/store/c2sm/pr04/rxiang/data_lmp/{i}_EAS11_lgm_dust/lm_coarse/3h/ASOB_T.nc')
    dt = np.nanmean(ds.variables['ASOB_T'][...], axis=0)
    LGM_.append(dt)

a = np.array(PD_)
b = np.array(LGM_)
PD = np.nanmean(a, axis=0)
LGM = np.nanmean(b, axis=0)

###############################################################################
# %% Plot
###############################################################################
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

fig = plt.figure(figsize=(11, 5.7))

gs = gridspec.GridSpec(2, 3, left=0.05, bottom=0.08, right=0.98,
                       top=0.95, hspace=0.05, wspace=0.05, width_ratios=[1, 1, 1], height_ratios=[1, 1])

axs, cs = np.empty(shape=(2, 3), dtype='object'), np.empty(shape=(2, 3), dtype='object')

cmap = cmc.lapaz_r
levels = np.linspace(0, 0.6, 19, endpoint=True)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
tick = np.linspace(0, 0.6, 4, endpoint=True)
levels2 = MaxNLocator(nbins=24).tick_values(-0.3, 0.3)
cmap2 = cmc.vik
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)
tick2 = np.linspace(-0.3, 0.3, 7, endpoint=True)

axs[0, 0] = fig.add_subplot(gs[0, 0], projection=rot_pole_crs)
axs[0, 0] = plotcosmo_notick(axs[0, 0])
cs[0, 0] = axs[0, 0].pcolormesh(rlon_, rlat_, dust_PD, shading="auto", cmap=cmap, norm=norm)

axs[0, 1] = fig.add_subplot(gs[0, 1], projection=rot_pole_crs)
axs[0, 1] = plotcosmo_notick(axs[0, 1])
cs[0, 1] = axs[0, 1].pcolormesh(rlon_, rlat_, dust_LGM, shading="auto", cmap=cmap, norm=norm)

axs[0, 2] = fig.add_subplot(gs[0, 2], projection=rot_pole_crs)
axs[0, 2] = plotcosmo_notick(axs[0, 2])
cs[0, 2] = axs[0, 2].pcolormesh(rlon_, rlat_, dust_LGM - dust_PD, shading="auto", cmap=cmap2, norm=norm2)

cax = fig.add_axes(
    [axs[0, 0].get_position().x0 + 0.155, axs[0, 1].get_position().y0 - 0.05, axs[0, 1].get_position().width, 0.03])
cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='horizontal', extend='max', ticks=tick)
cbar.ax.tick_params(labelsize=13)

cax = fig.add_axes(
    [axs[0, 2].get_position().x0, axs[0, 2].get_position().y0 - 0.05, axs[0, 2].get_position().width, 0.03])
cbar = fig.colorbar(cs[0, 2], cax=cax, orientation='horizontal', extend='both', ticks=tick2)
cbar.ax.tick_params(labelsize=13)

axs[0, 0].set_title("(a) PD", fontsize=13, loc='left')
axs[0, 1].set_title("(b) LGM", fontsize=13, loc='left')
axs[0, 2].set_title("(c) LGM - PD", fontsize=13, loc='left')

levels = MaxNLocator(nbins=25).tick_values(0, 400)
cmap = cmc.roma_r
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

levels2 = MaxNLocator(nbins=20).tick_values(-20, 20)
cmap2 = cmc.vik
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

axs[1, 0] = fig.add_subplot(gs[1, 0], projection=rot_pole_crs)
axs[1, 0] = plotcosmo_notick(axs[1, 0])
cs[1, 0] = axs[1, 0].pcolormesh(rlon, rlat, PD, shading="auto", cmap=cmap, norm=norm)

axs[1, 1] = fig.add_subplot(gs[1, 1], projection=rot_pole_crs)
axs[1, 1] = plotcosmo_notick(axs[1, 1])
cs[1, 1] = axs[1, 1].pcolormesh(rlon, rlat, LGM, shading="auto", cmap=cmap, norm=norm)

axs[1, 2] = fig.add_subplot(gs[1, 2], projection=rot_pole_crs)
axs[1, 2] = plotcosmo_notick(axs[1, 2])
cs[1, 2] = axs[1, 2].pcolormesh(rlon, rlat, LGM - PD, shading="auto", cmap=cmap2, norm=norm2)

cax = fig.add_axes(
    [axs[1, 0].get_position().x0 + 0.155, axs[1, 1].get_position().y0 - 0.08, axs[1, 1].get_position().width, 0.03])
cbar = fig.colorbar(cs[1, 1], cax=cax, orientation='horizontal', extend='max', ticks=[0, 100, 200, 300, 400])
cbar.ax.tick_params(labelsize=13)
axs[1, 0].text(0.5, -0.14, '[W m$^{-2}$]', ha='left', va='top', transform=axs[1, 1].transAxes, fontsize=13)

cax = fig.add_axes(
    [axs[1, 2].get_position().x0, axs[1, 2].get_position().y0 - 0.08, axs[1, 2].get_position().width, 0.03])
cbar = fig.colorbar(cs[1, 2], cax=cax, orientation='horizontal', extend='both')
cbar.ax.tick_params(labelsize=13)

for i in range(2):
    axs[i, 0].text(-0.008, 0.95, '50°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.77, '40°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.59, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.41, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.23, '10°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.05, '0°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)

for i in range(3):
    axs[1, i].text(0.12, -0.02, '80°E', ha='center', va='top', transform=axs[1, i].transAxes, fontsize=13)
    axs[1, i].text(0.32, -0.02, '100°E', ha='center', va='top', transform=axs[1, i].transAxes, fontsize=13)
    axs[1, i].text(0.52, -0.02, '120°E', ha='center', va='top', transform=axs[1, i].transAxes, fontsize=13)
    axs[1, i].text(0.72, -0.02, '140°E', ha='center', va='top', transform=axs[1, i].transAxes, fontsize=13)
    axs[1, i].text(0.92, -0.02, '160°E', ha='center', va='top', transform=axs[1, i].transAxes, fontsize=13)

fig.suptitle('TOA net downward shortwave radiation', fontsize=14)

plt.show()

plotpath = "/project/pr133/rxiang/figure/paper2/setup/"
fig.savefig(plotpath + 'lgmdust_jul.png', dpi=500, transparent=True)
