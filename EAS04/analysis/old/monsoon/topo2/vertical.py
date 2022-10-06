# -------------------------------------------------------------------------------
# modules
#
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo04, pole04, colorbar
import cartopy.crs as ccrs
from numpy import inf
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap, custom_seq_cmap, drywet
from pyproj import Transformer
import scipy.ndimage as ndimage


# -------------------------------------------------------------------------------
# read data
#
sims = ['ctrl', 'topo2']
wind = []
mdpath = "/project/pr133/rxiang/data/cosmo/"
fname_w = '01-05.W.50000.smr.yhourmean.nc'

for s in range(len(sims)):
    sim = sims[s]
    data = xr.open_dataset(f'{mdpath}EAS04_{sim}/diurnal/W/{fname_w}')
    W = np.nanmean(data['W'].values[:, 0, :, :], axis=0)
    wind.append(W)

lon = data['lon'].values[:, :]
lat = data['lat'].values[:, :]

ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_BECCY_4.4km_merit_unmod_topo.nc')
hsurf_ctrl = ds['HSURF'].values[:, :]
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_BECCY_4.4km_merit_env_topo_adj.nc')
hsurf_topo2 = ds['HSURF'].values[:, :]
lat_ = ds["lat"].values
lon_ = ds["lon"].values
ds.close()

# -------------------------------------------------------------------------------
# plot
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole04()

ar = 1.0  # initial aspect ratio for first trial
wi = 8.4  # height in inches
hi = wi * ar  # width in inches
ncol = 2  # edit here
nrow = 2
axs, cs, ct = np.empty(4, dtype='object'), np.empty(4, dtype='object'), np.empty(4, dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.07, 0.11, 0.87, 0.92
gs = gridspec.GridSpec(2, 2, left=left, bottom=bottom, right=right, top=top, wspace=0.2, hspace=0.13)
axs[0] = fig.add_subplot(gs[0], projection=rot_pole_crs)
axs[1] = fig.add_subplot(gs[2], projection=rot_pole_crs)
axs[2] = fig.add_subplot(gs[3], projection=rot_pole_crs)

# plot topography
axs[3] = fig.add_subplot(gs[1], projection=rot_pole_crs)

levels = MaxNLocator(nbins=20).tick_values(0, 0.03)
cmap = cmc.davos_r
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

sims = ['Control', 'Envelope topography']

for i in range(2):
    sim = sims[i]
    axs[i] = plotcosmo04(axs[i])
    cs[i] = axs[i].pcolormesh(rlon, rlat, wind[i], cmap=cmap, norm=norm, shading="auto")
    # ct[i] = axs[i].contour(rlon, rlat, all_smr_sms[i], levels=[5, 10, 15, 20, 25], colors='maroon', linewidths=1)
    axs[i].text(0, 1.02, f'{sim}', ha='left', va='bottom', transform=axs[i].transAxes, fontsize=14)

    # clabel = axs[i].clabel(ct[i], [5, 10, 15, 20, 25], inline=True, fontsize=13, use_clabeltext=True)
    # for l in clabel:
    #     l.set_rotation(0)
    # [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

cax = fig.add_axes([axs[1].get_position().x0, axs[1].get_position().y0 - 0.055, axs[1].get_position().width, 0.02])
cbar = fig.colorbar(cs[1], cax=cax, orientation='horizontal', extend='max')
cbar.ax.set_xticks([0, 0.01, 0.02, 0.03])
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('m/s', fontsize=13, labelpad=-0.01)

# plot difference
levels = MaxNLocator(nbins=15).tick_values(-5, 5)
cmap = drywet(25, cmc.vik_r)
cmap = custom_div_cmap(25, cmc.vik_r)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

axs[2] = plotcosmo04(axs[2])
cs[2] = axs[2].pcolormesh(rlon, rlat, wind[1] - wind[0], cmap=cmap, clim=(-0.03, 0.03), shading="auto")
# ct[2] = axs[2].contour(rlon, rlat, all_smr_sms[1] - all_smr_sms[0], levels=[-10, -5, -2, 2, 5, 10], colors='maroon',
#                        linewidths=1)
axs[2].text(0, 1.02, 'Envelope topography - Control', ha='left', va='bottom', transform=axs[2].transAxes, fontsize=14)

# clabel = axs[2].clabel(ct[2], [-10, -5, -2., 2, 5, 10], inline=True, use_clabeltext=True, fontsize=13)
# for l in clabel:
#     l.set_rotation(0)
# [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.7)) for txt in clabel]

cax = fig.add_axes([axs[2].get_position().x0, axs[2].get_position().y0 - 0.055, axs[2].get_position().width, 0.02])
cbar = fig.colorbar(cs[2], cax=cax, orientation='horizontal', extend='both')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('m/s', fontsize=13, labelpad=-0.01)

# plot topography
levels = np.arange(0, 501, 10.0)
ticks = np.arange(0, 501, 100.0)
cmap = custom_seq_cmap(70, cmc.broc)
norm = BoundaryNorm(levels, ncolors=cmap.N, extend="max")

axs[3] = plotcosmo04(axs[3])
cs[3] = axs[3].pcolormesh(lon_, lat_,  hsurf_topo2 - hsurf_ctrl, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading="auto")
# ct[3] = axs[3].contour(lon_, lat_, hsurf_ctrl - hsurf_topo2, colors='maroon', linewidths=1)
axs[3].text(0, 1.02, 'Envelope topography - Control', ha='left', va='bottom', transform=axs[3].transAxes, fontsize=14)

cax = fig.add_axes([axs[3].get_position().x1 + 0.02, axs[3].get_position().y0, 0.02, axs[3].get_position().height])
cbar = fig.colorbar(cs[3], cax=cax, orientation='vertical', extend='both', ticks=[0, 100, 200, 300, 400, 500])
cbar.ax.tick_params(labelsize=13)
axs[3].text(1.05, -0.1, 'm', ha='left', va='bottom', transform=axs[3].transAxes, fontsize=13)
# cbar.ax.set_xlabel('m', fontsize=13, labelpad=-0.01, loc='left')

fig.suptitle('Vertical velocity at 500 hPa', fontsize=16, fontweight='bold')

xmin, xmax = axs[1].get_xbound()
ymin, ymax = axs[1].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol
fig.set_figheight(wi * y2x_ratio)

fig.show()
plotpath = "/project/pr133/rxiang/figure/EAS04/analysis/monsoon/topo2/"
fig.savefig(plotpath + 'vertical.png', dpi=500)
plt.close(fig)






