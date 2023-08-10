import cmcrameri.cm as cmc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
from mycolor import custom_div_cmap

from plotcosmomap import plotcosmo_notick, pole

###############################################################################
# Data
###############################################################################
ds = xr.open_dataset('/project/pr133/rxiang/data/echam5_raw/PI/input/T159_jan_surf.nc')
PI = ds.variables['ALB'][...]
lon = ds['lon'].values
lat = ds['lat'].values
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/echam5_raw/LGM/input/T159_jan_surf.lgm.veg.nc')
LGM = ds.variables['ALB'][...]
ds.close()

###############################################################################
# %% Plot
###############################################################################
fig = plt.figure(figsize=(12, 3.3), constrained_layout=True)
gs = gridspec.GridSpec(1, 3)
gs.update(left=0.043, right=0.99, top=0.98, bottom=0.18, hspace=0.1, wspace=0.06)
axs, cs = np.empty(shape=(1, 3), dtype='object'), np.empty(shape=(1, 3), dtype='object')

cmap = cmc.lapaz_r
levels = np.linspace(0, 0.7, 15, endpoint=True)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
tick = np.linspace(0, 0.7, 8, endpoint=True)
levels2 = MaxNLocator(nbins=23).tick_values(-0.2, 0.2)
cmap2 = custom_div_cmap(27, cmc.vik)
tick2 = np.linspace(-0.2, 0.2, 5, endpoint=True)

axs[0, 0] = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
axs[0, 0].set_extent([60, 173, 0, 65], crs=ccrs.PlateCarree())
cs[0, 0] = axs[0, 0].pcolormesh(lon, lat, PI, shading="auto", cmap=cmap, norm=norm)

axs[0, 1] = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
axs[0, 1].set_extent([60, 173, 0, 65], crs=ccrs.PlateCarree())
cs[0, 1] = axs[0, 1].pcolormesh(lon, lat, LGM, shading="auto", cmap=cmap, norm=norm)

axs[0, 2] = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
axs[0, 2].set_extent([60, 173, 0, 65], crs=ccrs.PlateCarree())
cs[0, 2] = axs[0, 2].pcolormesh(lon, lat, LGM - PI, shading="auto", cmap=cmap2, clim=(-0.2, 0.2))

cax1 = fig.add_axes(
    [axs[0, 0].get_position().x0+0.17, axs[0, 0].get_position().y0 - 0.12, axs[0, 0].get_position().width, 0.035])
cbar = fig.colorbar(cs[0, 1], cax=cax1, orientation='horizontal', extend='max', ticks=tick)
cbar.ax.tick_params(labelsize=13)

cax2 = fig.add_axes(
    [axs[0, 2].get_position().x0, axs[0, 2].get_position().y0 - .12, axs[0, 2].get_position().width, 0.035])
cbar = fig.colorbar(cs[0, 2], cax=cax2, orientation='horizontal', extend='both', ticks=tick2)
cbar.ax.tick_params(labelsize=13)

axs[0, 0].set_title("(a) PI (ECHAM5)", fontsize=13, loc='left')
axs[0, 1].set_title("(b) LGM (ECHAM5)", fontsize=13, loc='left')
axs[0, 2].set_title("(c) LGM - PI", fontsize=13, loc='left')


for i in range(1):
    axs[i, 0].text(-0.008, 0.95, '50°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.77, '40°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.59, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.41, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.23, '10°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.05, '0°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)

for i in range(3):
    axs[0, i].text(0.12, -0.02, '80°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)
    axs[0, i].text(0.32, -0.02, '100°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)
    axs[0, i].text(0.52, -0.02, '120°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)
    axs[0, i].text(0.72, -0.02, '140°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)
    axs[0, i].text(0.92, -0.02, '160°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper2/setup/"
fig.savefig(plotpath + 'alb1.png', dpi=500, transparent=True)
plt.close()

ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_EAS_ext_12km_merit_LGM_consistent_TCL.nc')
COSMO = np.nanmean(ds.variables['ALB_DIF12'][...], axis=0)
rlon_ = ds['rlon'].values
rlat_ = ds['rlat'].values
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/echam5_raw/PD/input/T63oc_jan_surf_remap.nc')
PD = ds.variables['ALB'][...]
ds.close()

[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

fig = plt.figure(figsize=(12, 3.3), constrained_layout=True)
gs = gridspec.GridSpec(1, 3)
gs.update(left=0.043, right=0.99, top=0.98, bottom=0.18, hspace=0.1, wspace=0.06)
axs, cs = np.empty(shape=(1, 3), dtype='object'), np.empty(shape=(1, 3), dtype='object')

cmap = cmc.lapaz_r
levels = np.linspace(0, 0.7, 15, endpoint=True)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
tick = np.linspace(0, 0.7, 8, endpoint=True)
levels2 = MaxNLocator(nbins=23).tick_values(-0.2, 0.2)
cmap2 = custom_div_cmap(27, cmc.vik)
tick2 = np.linspace(-0.2, 0.2, 5, endpoint=True)

axs[0, 0] = fig.add_subplot(gs[0, 0], projection=rot_pole_crs)
axs[0, 0].set_extent([60, 173, 0, 65], crs=ccrs.PlateCarree())
cs[0, 0] = axs[0, 0].pcolormesh(rlon_, rlat_, COSMO, shading="auto", cmap=cmap, norm=norm)

axs[0, 1] = fig.add_subplot(gs[0, 1], projection=rot_pole_crs)
axs[0, 1].set_extent([60, 173, 0, 65], crs=ccrs.PlateCarree())
cs[0, 1] = axs[0, 1].pcolormesh(rlon_, rlat_, PD, shading="auto", cmap=cmap, norm=norm)

axs[0, 2] = fig.add_subplot(gs[0, 2], projection=rot_pole_crs)
axs[0, 2].set_extent([60, 173, 0, 65], crs=ccrs.PlateCarree())
cs[0, 2] = axs[0, 2].pcolormesh(rlon_, rlat_, PD-COSMO, shading="auto", cmap=cmap2, clim=(-0.2, 0.2))

cax1 = fig.add_axes(
    [axs[0, 0].get_position().x0+0.17, axs[0, 0].get_position().y0 - 0.12, axs[0, 0].get_position().width, 0.035])
cbar = fig.colorbar(cs[0, 1], cax=cax1, orientation='horizontal', extend='max', ticks=tick)
cbar.ax.tick_params(labelsize=13)

cax2 = fig.add_axes(
    [axs[0, 2].get_position().x0, axs[0, 2].get_position().y0 - .12, axs[0, 2].get_position().width, 0.035])
cbar = fig.colorbar(cs[0, 2], cax=cax2, orientation='horizontal', extend='both', ticks=tick2)
cbar.ax.tick_params(labelsize=13)

axs[0, 0].set_title("(a) PD (COSMO)", fontsize=13, loc='left')
axs[0, 1].set_title("(b) PD (ECHAM5)", fontsize=13, loc='left')
axs[0, 2].set_title("(c) ECHAM5 - COSMO", fontsize=13, loc='left')


for i in range(1):
    axs[i, 0].text(-0.008, 0.95, '50°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.77, '40°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.59, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.41, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.23, '10°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.05, '0°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)

for i in range(3):
    axs[0, i].text(0.12, -0.02, '80°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)
    axs[0, i].text(0.32, -0.02, '100°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)
    axs[0, i].text(0.52, -0.02, '120°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)
    axs[0, i].text(0.72, -0.02, '140°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)
    axs[0, i].text(0.92, -0.02, '160°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper2/setup/"
fig.savefig(plotpath + 'alb2.png', dpi=500, transparent=True)
