# -------------------------------------------------------------------------------
# modules
#
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo, pole, colorbar
import cartopy.crs as ccrs
from numpy import inf
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, hotcold
from pyproj import Transformer
import scipy.ndimage as ndimage
import matplotlib.colors as colors

# precipitation validation
mdpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn/TOT_PREC/"
erapath = "/project/pr133/rxiang/data/era5/pr/remap/"
crupath = "/project/pr133/rxiang/data/obs/pr/cru/remap/"

mdpr = xr.open_dataset(f'{mdpath}' + '2001-2005.TOT_PREC.JJA.nc')['TOT_PREC'].values[0, :, :]
np.seterr(divide='ignore', invalid='ignore')
erapr = xr.open_dataset(f'{erapath}' + 'era5.mo.2001-2005.JJA.remap.nc')['tp'].values[0, :, :]*1000
crupr = xr.open_dataset(f'{crupath}' + 'cru.2001-2005.05.JJA.remap.nc')['pre'].values[0, :, :]

erabias = np.nanmean(mdpr - erapr)
crubias = np.nanmean(mdpr - crupr)

# plot
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

ar = 1.0  # initial aspect ratio for first trial
wi = 15.5  # height in inches
hi = wi * ar  # width in inches
ncol = 3  # edit here
nrow = 1
axs, cs, ct = np.empty(3, dtype='object'), np.empty(3, dtype='object'), np.empty(3, dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.035, 0.14, 0.995, 0.95
gs = gridspec.GridSpec(1, 3, left=left, bottom=bottom, right=right, top=top, wspace=0.12, hspace=0.1)
for i in range(3):
    axs[i] = fig.add_subplot(gs[i], projection=rot_pole_crs)
    axs[i] = plotcosmo(axs[i])

# plot model
levels = MaxNLocator(nbins=20).tick_values(0, 20)
cmap = cmc.davos_r
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

cs[0] = axs[0].pcolormesh(rlon, rlat, mdpr, cmap=cmap, norm=norm, shading="auto")
# ct[0] = axs[0].contour(rlon, rlat, mdpr, levels=np.linspace(5, 20, 4, endpoint=True), colors='maroon', linewidths=1)
axs[0].text(0, 1.02, 'COSMO', ha='left', va='bottom', transform=axs[0].transAxes, fontsize=14)

# cax = fig.add_axes([axs[0].get_position().x0, axs[0].get_position().y0 - 0.35, axs[0].get_position().width, 0.05])
# cbar = fig.colorbar(cs[0], cax=cax, orientation='horizontal', extend='max')
# cbar.ax.tick_params(labelsize=13)
# cbar.ax.set_xlabel('mm/day', fontsize=13, labelpad=-0.01)
#
# # plot difference
# levels = MaxNLocator(nbins=15).tick_values(-5, 5)
# cmap = drywet(30, cmc.vik_r)
# norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

cs[1] = axs[1].pcolormesh(rlon, rlat, erapr, cmap=cmap, norm=norm, shading="auto")
# cs[1] = axs[1].pcolormesh(rlon, rlat, erapr, cmap=cmap, clim=(-15, 15), shading="auto")
# ct[1] = axs[1].contour(rlon, rlat, erapr, levels=np.linspace(-15, 15, 7, endpoint=True), colors='maroon',
#                        linewidths=1)
axs[1].text(0, 1.02, 'ERA5', ha='left', va='bottom', transform=axs[1].transAxes, fontsize=14)

cs[2] = axs[2].pcolormesh(rlon, rlat, crupr, cmap=cmap, norm=norm, shading="auto")
# cs[2] = axs[2].pcolormesh(rlon, rlat, crupr, cmap=cmap, clim=(-15, 15), shading="auto")
# ct[2] = axs[2].contour(rlon, rlat, crupr, levels=np.linspace(-15, 15, 7, endpoint=True), colors='maroon',
#                        linewidths=1)
axs[2].text(0, 1.02, 'CRU', ha='left', va='bottom', transform=axs[2].transAxes, fontsize=14)

cax = fig.add_axes([(axs[0].get_position().x0+axs[0].get_position().x1)/2, axs[1].get_position().y0 - 0.35,
                    (axs[2].get_position().x0+axs[2].get_position().x1)/2 - (axs[0].get_position().x0+axs[0].get_position().x1)/2, 0.05])
cbar = fig.colorbar(cs[2], cax=cax, orientation='horizontal', extend='max')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('mm/day', fontsize=13, labelpad=-0.01)

# add bias
n = str(round(erabias, 2))
t = axs[1].text(0.99, 0.98, f'bias = {n}', horizontalalignment='right', verticalalignment='top',
                transform=axs[1].transAxes, fontsize=14, zorder=4)
rect = plt.Rectangle((0.74, 0.91), width=0.255, height=0.08,
                         transform=axs[1].transAxes, zorder=3,
                         fill=True, facecolor="white", alpha=0.7, clip_on=False)
axs[1].add_patch(rect)

n = str(round(crubias, 2))
t = axs[2].text(0.99, 0.98, f'bias = {n}', horizontalalignment='right', verticalalignment='top',
                transform=axs[2].transAxes, fontsize=14, zorder=4)
rect = plt.Rectangle((0.74, 0.91), width=0.255, height=0.08,
                         transform=axs[2].transAxes, zorder=3,
                         fill=True, facecolor="white", alpha=0.7, clip_on=False)
axs[2].add_patch(rect)

fig.suptitle('Validation - Summer Precipitation (JJA)', fontsize=16, fontweight='bold')

xmin, xmax = axs[1].get_xbound()
ymin, ymax = axs[1].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol + 0.09
fig.set_figheight(wi * y2x_ratio)

fig.show()
plotpath = "/project/pr133/rxiang/figure/EAS11/validation/"
fig.savefig(plotpath + 'pr_jja.png', dpi=500)
plt.close(fig)

# temperature validation
mdpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn/T_2M/"
erapath = "/project/pr133/rxiang/data/era5/ot/remap/"
aphropath = "/project/pr133/rxiang/data/obs/tmp/APHRO/remap/"

mdt = xr.open_dataset(f'{mdpath}' + '2001-2005.T_2M.JJA.nc')['T_2M'].values[0, :, :] - 273.15
np.seterr(divide='ignore', invalid='ignore')
erat = xr.open_dataset(f'{erapath}' + 'era5.mo.2001-2005.JJA.remap.nc')['t2m'].values[0, :, :] - 273.15
aphrot = xr.open_dataset(f'{aphropath}' + 'APHRO.2001-2005.025.JJA.remap.nc')['tave'].values[0, :, :]

erabias = np.nanmean(mdt - erat)
aphrobias = np.nanmean(mdt - aphrot)

# plot
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

ar = 1.0  # initial aspect ratio for first trial
wi = 15.5  # height in inches
hi = wi * ar  # width in inches
ncol = 3  # edit here
nrow = 1
axs, cs, ct = np.empty(3, dtype='object'), np.empty(3, dtype='object'), np.empty(3, dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.035, 0.14, 0.995, 0.95
gs = gridspec.GridSpec(1, 3, left=left, bottom=bottom, right=right, top=top, wspace=0.12, hspace=0.1)
for i in range(3):
    axs[i] = fig.add_subplot(gs[i], projection=rot_pole_crs)
    axs[i] = plotcosmo(axs[i])

# plot model
levels = MaxNLocator(nbins=20).tick_values(0, 40)
cmap = cmc.lajolla
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

cs[0] = axs[0].pcolormesh(rlon, rlat, mdt, cmap=cmap, clim=(0, 40), norm=norm, shading="auto")
# ct[0] = axs[0].contour(rlon, rlat, mdpr, levels=np.linspace(5, 20, 4, endpoint=True), colors='maroon', linewidths=1)
axs[0].text(0, 1.02, 'COSMO', ha='left', va='bottom', transform=axs[0].transAxes, fontsize=14)

# cax = fig.add_axes([axs[0].get_position().x0, axs[0].get_position().y0 - 0.35, axs[0].get_position().width, 0.05])
# cbar = fig.colorbar(cs[0], cax=cax, orientation='horizontal', extend='both')
# cbar.ax.tick_params(labelsize=13)
# cbar.ax.set_xlabel('$^{o}C$', fontsize=13, labelpad=-0.01)
#
# # plot difference
# levels = MaxNLocator(nbins=25).tick_values(-8, 8)
# cmap = custom_div_cmap(29, cmc.vik)
# norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

cs[1] = axs[1].pcolormesh(rlon, rlat, erat, cmap=cmap, clim=(0, 40), norm=norm, shading="auto")
# ct[1] = axs[1].contour(rlon, rlat, erapr, levels=np.linspace(-15, 15, 7, endpoint=True), colors='maroon',
#                        linewidths=1)
axs[1].text(0, 1.02, 'ERA5', ha='left', va='bottom', transform=axs[1].transAxes, fontsize=14)

cs[2] = axs[2].pcolormesh(rlon, rlat, aphrot, cmap=cmap, clim=(0, 40), norm=norm, shading="auto")
# ct[2] = axs[2].contour(rlon, rlat, crupr, levels=np.linspace(-15, 15, 7, endpoint=True), colors='maroon',
#                        linewidths=1)
axs[2].text(0, 1.02, 'APHRO', ha='left', va='bottom', transform=axs[2].transAxes, fontsize=14)

cax = fig.add_axes([(axs[0].get_position().x0+axs[0].get_position().x1)/2, axs[1].get_position().y0 - 0.35,
                    (axs[2].get_position().x0+axs[2].get_position().x1)/2 - (axs[0].get_position().x0+axs[0].get_position().x1)/2, 0.05])
cbar = fig.colorbar(cs[2], cax=cax, orientation='horizontal', extend='both')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('$^{o}C$', fontsize=13, labelpad=-0.01)

# add bias
n = str(round(erabias, 2))
t = axs[1].text(0.99, 0.98, f'bias = {n}', horizontalalignment='right', verticalalignment='top',
                transform=axs[1].transAxes, fontsize=14, zorder=4)
rect = plt.Rectangle((0.74, 0.91), width=0.255, height=0.08,
                         transform=axs[1].transAxes, zorder=3,
                         fill=True, facecolor="white", alpha=0.7, clip_on=False)
axs[1].add_patch(rect)

n = str(round(aphrobias, 2))
t = axs[2].text(0.99, 0.98, f'bias = {n}', horizontalalignment='right', verticalalignment='top',
                transform=axs[2].transAxes, fontsize=14, zorder=4)
rect = plt.Rectangle((0.74, 0.91), width=0.255, height=0.08,
                         transform=axs[2].transAxes, zorder=3,
                         fill=True, facecolor="white", alpha=0.7, clip_on=False)
axs[2].add_patch(rect)

fig.suptitle('Validation - Summer Surface Temperature (JJA)', fontsize=16, fontweight='bold')

xmin, xmax = axs[1].get_xbound()
ymin, ymax = axs[1].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol + 0.09
fig.set_figheight(wi * y2x_ratio)

fig.show()
plotpath = "/project/pr133/rxiang/figure/EAS11/validation/"
fig.savefig(plotpath + 'tmp_jja.png', dpi=500)
plt.close(fig)


