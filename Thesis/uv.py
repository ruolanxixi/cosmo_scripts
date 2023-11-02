# annual mean climate-change signal of mean surface precipitation
# Load modules
import xarray as xr
from pyproj import CRS, Transformer
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib
import cmcrameri.cm as cmc
import numpy.ma as ma
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_
from plotcosmomap import plotcosmo_notick, pole, plotcosmo_notick_nogrid, plotcosmo_notick_lgm
import matplotlib.colors as colors
import scipy.ndimage as ndimage
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from metpy.units import units
import metpy.calc as mpcalc
import pandas as pd

mpl.style.use("classic")
font = {'size': 14}
matplotlib.rc('font', **font)
plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'dimgrey'

# --------------------------------------------------------------------
# -- Data
# COSMO
data = {}
dir = "/project/pr133/rxiang/data/era5/uv/"
pres = ['300', '500']
for i in range(2):
    pre = pres[i]
    ds = xr.open_mfdataset(f'{dir}' + f'era5.uv.1985-2014.ymonmean.nc')
    ds = ds.sel(time=ds['time.season'] == 'DJF')
    u = ds['u'].values[:, i, :, :]
    v = ds['v'].values[:, i, :, :]
    data[pre] = {"u": np.nanmean(u, axis=0),
                 "v": np.nanmean(v, axis=0),
                 "ws": np.nanmean(np.sqrt(u**2+v**2), axis=0)}

lat = xr.open_dataset(f'{dir}' + 'era5.uv10.1985-2014.ymonmean.nc')['latitude'].values[...]
lon = xr.open_dataset(f'{dir}' + 'era5.uv10.1985-2014.ymonmean.nc')['longitude'].values[...]
# %% ---------------------------------------------------------------------
# -- Plot
pres = ['500', '300']
labels = ['500 hPa', '300 hPa']
lb = ['a', 'b']

fig = plt.figure(figsize=(11, 3.4))
gs = gridspec.GridSpec(1, 2, left=0.05, bottom=0.04, right=0.91,
                        top=0.94, hspace=0.05, wspace=0.05,
                        width_ratios=[1, 1])
ncol = 2  # edit here
nrow = 1

axs, cs, ct, topo, q = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object')

longitude_offset = 130
proj = ccrs.PlateCarree(central_longitude=longitude_offset)

for i in range(ncol):
    axs[0, i] = fig.add_subplot(gs[0, i], projection=proj)
    axs[0, i].set_extent([60, 200, -0.1, 80.1], ccrs.PlateCarree())  # for extended 12km domain
    axs[0, i].add_feature(cfeature.COASTLINE)
    axs[0, i].add_feature(cfeature.BORDERS, linestyle=':')
    axs[0, i].add_feature(cfeature.LAKES, alpha=0.5)

    gl = axs[0, i].gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=0.7,
                      color='gray', alpha=0.7, linestyle='--', crs=ccrs.PlateCarree())
    gl.xlocator = mticker.FixedLocator([60, 80, 100, 120, 140, 160, 180, -160])
    gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60, 70, 80])


# --
levels = MaxNLocator(nbins=30).tick_values(0, 30)
cmap = cmc.roma_r
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

for i in range(ncol):
    sim = pres[i]
    cs[0, i] = axs[0, i].streamplot(lon-longitude_offset, lat, data[sim]['u'], data[sim]['v'], color=data[sim]['ws'],
                                    density=1.3, cmap=cmap, norm=norm, arrowstyle='->', transform=proj)

for i in range(ncol):
    label = lb[i]
    t = axs[0, i].text(0, 1.02, f'({label})', ha='left', va='bottom',
                       transform=axs[0, i].transAxes, fontsize=14)
    # t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

# --
for i in range(nrow):
    cax = fig.add_axes(
        [axs[i, 1].get_position().x1 + 0.015, axs[i, 1].get_position().y0, 0.015, axs[i, 1].get_position().height])
    cbar = fig.colorbar(cs[i, 1].lines, cax=cax, orientation='vertical', extend='max', ticks=[0, 5, 10, 15, 20, 25, 30])
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.minorticks_off()
    cbar.set_label('[m s$^{-1}$]', fontsize=14)

# --
for j in range(ncol):
    label = labels[j]
    axs[0, j].set_title(f'{label}', pad=8.5, fontsize=15, loc='center')

for i in range(nrow):
    axs[i, 0].text(-0.008, 1, '80°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.008, 0.75, '60°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.008, 0.5, '40°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.008, 0.25, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.008, 0, '0°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    # axs[i, 0].text(-0.008, 0, '20°S', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)

for j in range(ncol):
    # axs[nrow - 1, j].text(0.11, -0.02, '40°E', ha='center', va='top', transform=axs[nrow - 1, j].transAxes, fontsize=14)
    axs[nrow - 1, j].text(0.1429, -0.02, '80°E', ha='center', va='top', transform=axs[nrow - 1, j].transAxes,
                          fontsize=14)
    axs[nrow - 1, j].text(0.4286, -0.02, '120°E', ha='center', va='top', transform=axs[nrow - 1, j].transAxes,
                          fontsize=14)
    axs[nrow - 1, j].text(0.7143, -0.02, '160°E', ha='center', va='top', transform=axs[nrow - 1, j].transAxes,
                          fontsize=14)
    axs[nrow - 1, j].text(1, -0.02, '160°W', ha='center', va='top', transform=axs[nrow - 1, j].transAxes,
                          fontsize=14)

plt.show()
plotpath = "/project/pr133/rxiang/figure/thesis/"
fig.savefig(plotpath + 'uv_DJF.png', dpi=500)
plt.close(fig)
