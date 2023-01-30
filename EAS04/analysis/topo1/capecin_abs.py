# -------------------------------------------------------------------------------
# modules
#
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo04_notick, pole04, colorbar
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
sims = ['CTRL04', 'TRED04']
seasons = "JJA"

# --- edit here
ctrlpath = "/project/pr133/rxiang/data/cosmo/EAS04_ctrl/monsoon"
topo1path = "/project/pr133/rxiang/data/cosmo/EAS04_topo1/monsoon"
paths = [ctrlpath, topo1path]
data = {}
vars = ['CAPE_ML', 'CIN_ML']
signs = [1, -1]

[pole_lat04, pole_lon04, lat04, lon04, rlat04, rlon04, rot_pole_crs04] = pole04()

for i in range(len(sims)):
    sim = sims[i]
    path = paths[i]
    data[sim] = {}
    for j in range(len(vars)):
        var = vars[j]
        sign = signs[j]
        f = xr.open_dataset(f'{path}/{var}/smr/01-05.{var}.smr.nc')
        ds = f[var].values[:, :] * sign
        data[sim][var] = np.nanmean(ds, axis=0)

data['diff'] = {}
for j in range(len(vars)):
    var = vars[j]
    sign = signs[j]
    data['diff'][var] = data['TRED04'][var] - data['CTRL04'][var]

# load topo
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_BECCY_4.4km_merit_unmod_topo.nc')
hsurf_ctrl = ds['HSURF'].values[:, :]
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_BECCY_4.4km_merit_reduced_topo_adj.nc')
hsurf_topo1 = ds['HSURF'].values[:, :]
hsurf_diff = ndimage.gaussian_filter(hsurf_ctrl - hsurf_topo1, sigma=5, order=0)
hsurf_ctrl = ndimage.gaussian_filter(hsurf_ctrl, sigma=3, order=0)
lat_ = ds["lat"].values
lon_ = ds["lon"].values
ds.close()

# %% -------------------------------------------------------------------------------
# plot
ar = 1.0  # initial aspect ratio for first trial
wi = 9.5  # height in inches #15
hi = 4.8  # width in inches #10
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
                        top=0.97, hspace=0.01, wspace=0.1, width_ratios=[1, 1], height_ratios=[1, 1])
gs2 = gridspec.GridSpec(2, 1, left=0.665, bottom=0.024, right=0.91,
                        top=0.97, hspace=0.01, wspace=0.1, height_ratios=[1, 1])

level1 = MaxNLocator(nbins=24).tick_values(0, 600)
cmap1 = custom_seq_cmap(24, cmc.roma_r, 0, 0)
norm1 = BoundaryNorm(level1, ncolors=cmap1.N, clip=True)

level2 = MaxNLocator(nbins=24).tick_values(-60, 0)
cmap2 = custom_seq_cmap(24, cmc.roma, 0, 1)
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
        axs[j, i] = plotcosmo04_notick(axs[j, i])
        cs[j, i] = axs[j, i].pcolormesh(rlon04, rlat04, data[sim][var], norm=norm, cmap=cmap,
                                         shading="auto", transform=rot_pole_crs04)

level1 = MaxNLocator(nbins=20).tick_values(-100, 600)
cmap1 = custom_div_cmap(21, cmc.vik)
norm1 = matplotlib.colors.TwoSlopeNorm(vmin=-100, vcenter=0., vmax=600)

level2 = MaxNLocator(nbins=20).tick_values(-20, 20)
cmap2 = custom_div_cmap(21, cmc.vik)
norm2 = matplotlib.colors.Normalize(vmin=-20, vmax=20)

norms = [norm1, norm2]
cmaps = [cmap1, cmap2]
for j in range(2):
    var = vars[j]
    norm = norms[j]
    cmap = cmaps[j]
    axs[j, 2] = fig.add_subplot(gs2[j, 0], projection=rot_pole_crs04)
    axs[j, 2] = plotcosmo04_notick(axs[j, 2])
    cs[j, 2] = axs[j, 2].pcolormesh(rlon04, rlat04, data['diff'][var], norm=norm, cmap=cmap,
                                    shading="auto", transform=rot_pole_crs04)
    topo[j, 2] = axs[j, 2].contour(lon_, lat_, hsurf_diff, levels=[500], colors='darkgreen', linewidths=1,
                                   transform=ccrs.PlateCarree())

tick1 = np.linspace(0, 600, 7, endpoint=True)
tick2 = np.linspace(-60, 0, 7, endpoint=True)
ticks = [tick1, tick2]
extends = ['max', 'min']
for i in range(nrow):
    ext = extends[i]
    tick = ticks[i]
    cax = fig.add_axes(
        [axs[i, 1].get_position().x1 + 0.01, axs[i, 1].get_position().y0, 0.015, axs[i, 1].get_position().height])
    cbar = fig.colorbar(cs[i, 1], cax=cax, orientation='vertical', extend=ext, ticks=tick)
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.minorticks_off()

tick1 = [-100, -50, 0, 300, 600]
tick2 = np.linspace(-20, 20, 5, endpoint=True)
ticks = [tick1, tick2]
for i in range(nrow):
    tick = ticks[i]
    cax = fig.add_axes(
        [axs[i, 2].get_position().x1 + 0.01, axs[i, 2].get_position().y0, 0.015, axs[i, 2].get_position().height])
    cbar = fig.colorbar(cs[i, 2], cax=cax, orientation='vertical', extend='both', ticks=tick)
    cbar.ax.tick_params(labelsize=13)


for i in range(nrow):
    axs[i, 0].text(-0.01, 0.83, '35°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.57, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.31, '25°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.05, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)

for j in range(ncol):
    axs[1, j].text(0.04, -0.02, '90°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=14)
    axs[1, j].text(0.46, -0.02, '100°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=14)
    axs[1, j].text(0.88, -0.02, '110°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=14)

lb = [['a', 'b', 'c'], ['d', 'e', 'f']]
for i in range(nrow):
    for j in range(ncol):
        label = lb[i][j]
        t = axs[i, j].text(0.01, 0.987, f'({label})', ha='left', va='top',
                           transform=axs[i, j].transAxes, fontsize=14)
        t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

titles = ['CTRL04', 'TRED04', 'TRED04-CTRL04']
for j in range(ncol):
    title = titles[j]
    axs[0, j].set_title(f'{title}', pad=5, fontsize=14, loc='center')

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/results/extreme/"
fig.savefig(plotpath + 'capecin1.png', dpi=500)
plt.close(fig)
