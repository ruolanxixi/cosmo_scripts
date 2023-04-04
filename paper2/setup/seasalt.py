###########################################
#%% load module
###########################################
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
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_, wind
from pyproj import Transformer
import scipy.ndimage as ndimage
import matplotlib

###########################################
#%% load data
###########################################
path = '/project/pr133/rxiang/data/aerosol'
sims = ['MPI-ESM-1-2-HAM', 'CNRM-CM6-1', 'EC-Earth3-AerChem', 'GISS-E2-1-G', 'IPSL-CM6A-LR-INCA','NorESM2-LM',
        'CNRM-ESM2-1', 'GFDL-CM4', 'IPSL-CM6A-LR', 'MIROC6', 'MRI-ESM2-0', 'NorESM2-MM']

data = {}

for sim in sims:
    f = f'aod_AeroCom_{sim}.nc'
    data[sim] = {}
    ds = np.nanmean(xr.open_dataset(f'{path}/{f}')['dust'].values[...], axis=0)
    # ds = xr.open_dataset(f'{path}/{f}')['dust'].values[6, :, :]
    data[sim]['dust'] = ds
    data[sim]['lon'] = xr.open_dataset(f'{path}/{f}')['lon'].values[:]
    data[sim]['lat'] = xr.open_dataset(f'{path}/{f}')['lat'].values[:]

###########################################
#%% plot
###########################################
wi = 16  # height in inches #15
hi = 14  # width in inches #10
ncol = 3  # edit here
nrow = 4
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), np.empty(
    shape=(nrow, ncol), dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.03, 0.1, 0.98, 0.95
gs = gridspec.GridSpec(nrows=4, ncols=3, left=left, bottom=bottom, right=right, top=top, wspace=0.015, hspace=0.05)

cmap = cmc.lapaz_r
levels = np.linspace(0, 0.14, 21, endpoint=True)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
tick = np.linspace(0, 0.14, 3, endpoint=True)

cmap = cmc.lapaz_r
levels = np.linspace(0, 0.5, 21, endpoint=True)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
tick = np.linspace(0, 0.5, 6, endpoint=True)

for i in range(len(sims)):
    sim = sims[i]
    ii = i // ncol
    jj = i % ncol
    axs[ii, jj] = fig.add_subplot(gs[ii, jj], projection=ccrs.Robinson(central_longitude=180, globe=None))
    axs[ii, jj].coastlines(zorder=3)
    axs[ii, jj].stock_img()
    axs[ii, jj].gridlines()
    cs[ii, jj] = axs[ii, jj].pcolormesh(data[sim]['lon'], data[sim]['lat'], data[sim]['dust'], norm=norm, cmap=cmap,
                                        shading="auto", transform=ccrs.PlateCarree())
    axs[ii, jj].set_title(f'{sim}', fontweight='bold', pad=6, fontsize=13, loc='center')

cax = fig.add_axes([axs[3, 1].get_position().x0, axs[3, 1].get_position().y0 - 0.06, axs[3, 1].get_position().width, 0.02])
cbar = fig.colorbar(cs[3, 1], cax=cax, orientation='horizontal', extend='max', ticks=tick)
cbar.ax.tick_params(labelsize=13)

plt.show()
