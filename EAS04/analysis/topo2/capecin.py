# -------------------------------------------------------------------------------
# modules
#
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo04sm_notick, pole04, colorbar
import cartopy.crs as ccrs
from numpy import inf
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_, wind, hotcold, conv, custom_seq_cmap
from pyproj import Transformer
import scipy.ndimage as ndimage
import matplotlib

# %% -------------------------------------------------------------------------------
# read data
sims = ['CTRL04', 'TENV04']
seasons = "JJA"

# --- edit here
ctrlpath = "/project/pr133/rxiang/data/cosmo/EAS04_ctrl/monsoon"
topo2path = "/project/pr133/rxiang/data/cosmo/EAS04_topo2/monsoon"
paths = [ctrlpath, topo2path]
data = {}
vars = ['CAPE_ML', 'CIN_ML']
signs = [1, -1]

[pole_lat04, pole_lon04, lat04, lon04, rlat04, rlon04, rot_pole_crs04] = pole04()

for i in range(len(sims)):
    sim = sims[i]
    path = paths[i]
    sign = signs[i]
    data[sim] = {}
    for j in range(len(vars)):
        var = vars[j]
        f = xr.open_dataset(f'{path}/{var}/smr/01-05.{var}.smr.nc')
        ds = f[var].values[:, :] * sign
        data[sim][var] = np.nanmean(ds, axis=0)

data['diff'] = {}
for j in range(len(vars)):
    var = vars[j]
    data['diff'][var] = data['TENV04'][var] - data['CTRL04'][var]

# %% -------------------------------------------------------------------------------
# plot
ar = 1.0  # initial aspect ratio for first trial
wi = 9.5  # height in inches #15
hi = 5.5  # width in inches #10
ncol = 3  # edit here
nrow = 2
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), np.empty(
    shape=(nrow, ncol), dtype='object')

fig = plt.figure(figsize=(wi, hi))
gs1 = gridspec.GridSpec(2, 2, left=0.06, bottom=0.024, right=0.575,
                        top=0.97, hspace=0.1, wspace=0.1, width_ratios=[1, 1], height_ratios=[1, 1])
gs2 = gridspec.GridSpec(2, 1, left=0.665, bottom=0.024, right=0.91,
                        top=0.97, hspace=0.01, wspace=0.1, height_ratios=[1, 1])

level1 = MaxNLocator(nbins=20).tick_values(0, 600)
cmap1 = custom_seq_cmap(21, cmc.roma_r, 0, 0)
norm1 = BoundaryNorm(level1, ncolors=cmap1.N, clip=True)

level2 = MaxNLocator(nbins=20).tick_values(-50, 0)
cmap2 = custom_seq_cmap(21, cmc.roma, 0, 1)
norm2 = BoundaryNorm(level2, ncolors=cmap1.N, clip=True)

norms = [norm1, norm2]
cmaps = [cmap1, cmap2]
for i in range(2):
    sim = sims[i]
    for j in range(2):
        var = vars[j]
        norm = norms[j]
        cmap = cmaps[j]
        axs[j, i] = fig.add_subplot(gs1[j, i], projection=rot_pole_crs04)
        axs[j, i] = plotcosmo04sm_notick(axs[j, i])
        cs[j, i] = axs[j, i].pcolormesh(rlon04, rlat04, data[sim][var], norm=norm, cmap=cmap,
                                         shading="auto", transform=rot_pole_crs04)

level1 = MaxNLocator(nbins=20).tick_values(-400, 200)
cmap1 = custom_div_cmap(21, cmc.vik)
norm1 = matplotlib.colors.Normalize(vmin=-400, vmax=200)

level2 = MaxNLocator(nbins=20).tick_values(-10, 10)
cmap2 = custom_div_cmap(21, cmc.vik)
norm2 = matplotlib.colors.Normalize(vmin=-10, vmax=10)

norms = [norm1, norm2]
cmaps = [cmap1, cmap2]
for j in range(2):
    var = vars[j]
    norm = norms[j]
    cmap = cmaps[j]
    axs[j, 2] = fig.add_subplot(gs2[j, 0], projection=rot_pole_crs04)
    axs[j, 2] = plotcosmo04sm_notick(axs[j, 2])
    cs[j, 2] = axs[j, 2].pcolormesh(rlon04, rlat04, data['diff'][var], norm=norm, cmap=cmap,
                                    shading="auto", transform=rot_pole_crs04)

extends = ['max', 'min']
for i in range(nrow):
    ext = extends[i]
    cax = fig.add_axes(
        [axs[i, 1].get_position().x1 + 0.01, axs[i, 1].get_position().y0, 0.015, axs[i, 1].get_position().height])
    cbar = fig.colorbar(cs[i, 1], cax=cax, orientation='vertical', extend=ext)
    cbar.ax.tick_params(labelsize=13)

for i in range(nrow):
    cax = fig.add_axes(
        [axs[i, 2].get_position().x1 + 0.01, axs[i, 2].get_position().y0, 0.015, axs[i, 2].get_position().height])
    cbar = fig.colorbar(cs[i, 2], cax=cax, orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=13)


for i in range(nrow):
    axs[i, 0].text(-0.01, 0.91, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.45, '25°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)

for j in range(ncol):
    axs[1, j].text(0.04, -0.02, '95°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=14)
    axs[1, j].text(0.46, -0.02, '100°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=14)
    axs[1, j].text(0.88, -0.02, '105°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=14)

lb = [['a', 'b', 'c'], ['d', 'e', 'f']]
for i in range(nrow):
    for j in range(ncol):
        label = lb[i][j]
        t = axs[i, j].text(0.01, 0.987, f'({label})', ha='left', va='top',
                           transform=axs[i, j].transAxes, fontsize=14)
        t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

titles = ['CTRL04', 'TENV04', 'TENV04-CTRL04']
for j in range(ncol):
    title = titles[j]
    axs[0, j].set_title(f'{title}', pad=5, fontsize=14, loc='center')

plt.show()
