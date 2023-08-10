###############################################################################
# Load module
###############################################################################
import cartopy.crs as ccrs
import cmcrameri.cm as cmc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm

###############################################################################
# Data
###############################################################################
ds = xr.open_dataset('/project/pr133/rxiang/data/aerosol/aod_MACv2.nc')
PI = np.nanmean(ds.variables['dust'][...], axis=0)
# ds = xr.open_dataset('/project/pr133/rxiang/data/aerosol/PMIP4_DUST_Albani_aot_PI.nc')
# PI = np.sum(np.nanmean(ds.variables['aot'][...], axis=0), axis=0)
lon1 = ds['lon'].values[:]
lat1 = ds['lat'].values[:]
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/aerosol/PMIP4_DUST_Albani_aot_LGM.nc')
LGM = np.sum(np.nanmean(ds.variables['aot'][...], axis=0), axis=0)
# LGM = np.nanmean(ds.variables['aot'][...], axis=0)[3, ...]
lon2 = ds['lon'].values[:]
lat2 = ds['lat'].values[:]
ds.close()

###############################################################################
#%% Plot
###############################################################################
wi = 8  # height in inches #15
hi = 3  # width in inches #10
ncol = 2  # edit here
nrow = 1

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.03, 0.2, 0.98, 0.82
gs = gridspec.GridSpec(nrows=1, ncols=2, left=left, bottom=bottom, right=right, top=top, wspace=0.015, hspace=0.05)

cmap = cmc.lapaz_r
levels = np.linspace(0, 0.3, 19, endpoint=True)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
tick = np.linspace(0, 0.3, 4, endpoint=True)

axs, cs = np.empty(shape=(1, 2), dtype='object'), np.empty(shape=(1, 2), dtype='object')
axs[0, 0] = fig.add_subplot(gs[0], projection=ccrs.Robinson(central_longitude=180, globe=None))
axs[0, 0].coastlines(zorder=3)
axs[0, 0].gridlines()
cs[0, 0] = axs[0, 0].pcolormesh(lon1, lat1, PI, shading="auto", cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
axs[0, 0].set_title("(a) PI", fontsize=13, loc='left')

axs[0, 1] = fig.add_subplot(gs[1], projection=ccrs.Robinson(central_longitude=180, globe=None))
axs[0, 1].coastlines(zorder=3)
axs[0, 1].gridlines()
cs[0, 1] = axs[0, 1].pcolormesh(lon2, lat2, LGM, shading="auto", cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
axs[0, 1].set_title("(b) LGM", fontsize=13, loc='left')

cax = fig.add_axes([axs[0, 0].get_position().x0+0.25, axs[0, 1].get_position().y0 - 0.1, axs[0, 1].get_position().width, 0.05])
cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='horizontal', extend='max', ticks=tick)
cbar.ax.tick_params(labelsize=13)

fig.suptitle('Aerosol Optical Depth due to dust', fontsize=14)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper2/setup/"
fig.savefig(plotpath + 'dust.png', dpi=500, transparent=True)
