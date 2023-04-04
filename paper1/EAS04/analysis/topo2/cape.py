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
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_, wind, hotcold, conv
from pyproj import Transformer
import scipy.ndimage as ndimage
import matplotlib

# %% -------------------------------------------------------------------------------
# read data
sims = ['CTRL04', 'TRED04']
seasons = "JJA"

# --- edit here
ctrlpath = "/project/pr133/rxiang/data/cosmo/EAS04_ctrl/monsoon/CAPE_ML/smr"
topo2path = "/project/pr133/rxiang/data/cosmo/EAS04_topo2/monsoon/CAPE_ML/smr"
paths = [ctrlpath, topo2path]
data = {}
vars = ['CAPE_ML']

[pole_lat04, pole_lon04, lat04, lon04, rlat04, rlon04, rot_pole_crs04] = pole04()

lb = [['a', 'b', 'c']]

for i in range(len(sims)):
    sim = sims[i]
    path = paths[i]
    data[sim] = {}
    f = xr.open_dataset(f'{path}/01-05.CAPE_ML.smr.nc')
    ds = f["CAPE_ML"].values[:, :]
    data[sim]["CAPE"] = np.nanmean(ds, axis=0)

data['diff'] = {}
data['diff']["CAPE"] = data['TRED04']["CAPE"] - data['CTRL04']["CAPE"]

# %% -------------------------------------------------------------------------------
# plot
ar = 1.0  # initial aspect ratio for first trial
wi = 9.5  # height in inches #15
hi = 3  # width in inches #10
ncol = 3  # edit here
nrow = 1
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), np.empty(
    shape=(nrow, ncol), dtype='object')

fig = plt.figure(figsize=(wi, hi))
gs1 = gridspec.GridSpec(1, 2, left=0.06, bottom=0.024, right=0.575,
                        top=0.97, hspace=0.1, wspace=0.1, width_ratios=[1, 1])
gs2 = gridspec.GridSpec(1, 1, left=0.665, bottom=0.024, right=0.91,
                        top=0.97, hspace=0.01, wspace=0.1)

level1 = MaxNLocator(nbins=20).tick_values(0, 2000)
cmap1 = cmc.roma_r
norm1 = BoundaryNorm(level1, ncolors=cmap1.N, clip=True)

level2 = MaxNLocator(nbins=20).tick_values(-400, 200)
cmap2 = custom_div_cmap(21, cmc.vik)
norm2 = matplotlib.colors.Normalize(vmin=-400, vmax=200)

for j in range(2):
    sim = sims[j]
    axs[0, j] = fig.add_subplot(gs1[0, j], projection=rot_pole_crs04)
    axs[0, j] = plotcosmo04sm_notick(axs[0, j])
    cs[0, j] = axs[0, j].pcolormesh(rlon04, rlat04, data[sim]["CAPE"], norm=norm1, cmap=cmap1,
                                    shading="auto", transform=rot_pole_crs04)

axs[0, 2] = fig.add_subplot(gs2[0, 0], projection=rot_pole_crs04)
axs[0, 2] = plotcosmo04sm_notick(axs[0, 2])
cs[0, 2] = axs[0, 2].pcolormesh(rlon04, rlat04, data['diff']["CAPE"], norm=norm2, cmap=cmap2,
                                shading="auto", transform=rot_pole_crs04)

for i in range(nrow):
    cax = fig.add_axes(
        [axs[i, 1].get_position().x1 + 0.01, axs[i, 1].get_position().y0, 0.015, axs[i, 1].get_position().height])
    cbar = fig.colorbar(cs[i, 1], cax=cax, orientation='vertical', extend='max')
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
    axs[0, j].text(0.04, -0.02, '95°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=14)
    axs[0, j].text(0.46, -0.02, '100°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=14)
    axs[0, j].text(0.88, -0.02, '105°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=14)

plt.show()
