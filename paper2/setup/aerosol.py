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
sims = ['PI', 'PD']
fname = {'PD': 'aod_AeroCom1.nc', 'PI': 'aod_MACv2.nc'}
vars = ['black_carbon', 'dust', 'organic', 'sulfate', 'sea_salt']
labels = {'black_carbon': 'black carbon', 'dust': 'dust', 'organic': 'organic', 'sulfate': 'sulfate', 'sea_salt': 'sea salt'}

data = {}

for sim in sims:
    f = fname[sim]
    data[sim] = {}
    for var in vars:
        ds = np.nanmean(xr.open_dataset(f'{path}/{f}')[var].values[...], axis=0)
        data[sim][var] = ds
    data[sim]['lon'] = xr.open_dataset(f'{path}/{f}')['lon'].values[:]
    data[sim]['lat'] = xr.open_dataset(f'{path}/{f}')['lat'].values[:]

###########################################
#%% plot
###########################################
wi = 16  # height in inches #15
hi = 4  # width in inches #10
ncol = 5  # edit here
nrow = 2
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), np.empty(
    shape=(nrow, ncol), dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.03, 0.14, 0.98, 0.95
gs = gridspec.GridSpec(nrows=2, ncols=5, left=left, bottom=bottom, right=right, top=top, wspace=0.015, hspace=0.05)

# cmap1 = cmc.lapaz_r
# levels1 = np.linspace(0, 0.05, 21, endpoint=True)
# norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
# tick1 = np.linspace(0, 0.05, 6, endpoint=True)
# cmap2 = cmc.lapaz_r
# levels2 = np.linspace(0, 0.5, 21, endpoint=True)
# norm2 = BoundaryNorm(levels2, ncolors=cmap1.N, clip=True)
# tick2 = np.linspace(0, 0.5, 6, endpoint=True)
# cmap3 = cmc.lapaz_r
# levels3 = np.linspace(0, 0.18, 19, endpoint=True)
# norm3 = BoundaryNorm(levels3, ncolors=cmap1.N, clip=True)
# tick3 = np.linspace(0, 0.18, 4, endpoint=True)
# cmap4 = cmc.lapaz_r
# levels4 = np.linspace(0, 0.4, 21, endpoint=True)
# norm4 = BoundaryNorm(levels4, ncolors=cmap1.N, clip=True)
# tick4 = np.linspace(0, 0.4, 5, endpoint=True)
cmap = cmc.lapaz_r
levels = np.linspace(0, 0.3, 20, endpoint=True)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
tick = np.linspace(0, 0.3, 4, endpoint=True)

# cmaps = [cmap1, cmap2, cmap3, cmap4, cmap5]
# norms = [norm1, norm2, norm3, norm4, norm5]
# ticks = [tick1, tick2, tick3, tick4, tick5]

for i in range(len(sims)):
    sim = sims[i]
    for j in range(len(vars)):
        var = vars[j]
        # cmap = cmaps[j]
        # norm = norms[j]
        axs[i, j] = fig.add_subplot(gs[i, j], projection=ccrs.Robinson(central_longitude=180, globe=None))
        axs[i, j].coastlines(zorder=3)
        axs[i, j].stock_img()
        axs[i, j].gridlines()
        cs[i, j] = axs[i, j].pcolormesh(data[sim]['lon'], data[sim]['lat'], data[sim][var], norm=norm, cmap=cmap,
                                        shading="auto", transform=ccrs.PlateCarree())
    axs[i, 0].text(-0.09, 0.5, f'{sim}', ha='left', va='center', rotation='vertical',
                   transform=axs[i, 0].transAxes, fontsize=13, fontweight='bold')

for j in range(len(vars)):
    var = vars[j]
    label = labels[var]
    # tick = ticks[j]
    axs[0, j].set_title(f'{label}', fontweight='bold', pad=6, fontsize=13, loc='center')
    cax = fig.add_axes([axs[1, j].get_position().x0+0.02, axs[1, j].get_position().y0 - 0.06, axs[1, j].get_position().width*0.8, 0.03])
    cbar = fig.colorbar(cs[1, j], cax=cax, orientation='horizontal', extend='max', ticks=tick)
    cbar.ax.tick_params(labelsize=13)

plt.show()
# plotpath = "/project/pr133/rxiang/figure/echam5/"
# fig.savefig(plotpath + 'temp2' + f'{mon}.png', dpi=500)
