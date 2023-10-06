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
data['cosmo'] = {}
dir = "/project/pr133/rxiang/data/cosmo/"
data['cosmo']['ctrl'] = np.nanmean(
    xr.open_dataset(f'{dir}' + 'EAS11_ctrl/monsoon/T_2M/' + '01-05.T_2M.cpm.nc')['T_2M'].values[...],
    axis=0) - 273.15
data['cosmo']['lgm'] = np.nanmean(
    xr.open_dataset(dir + 'EAS11_lgm/monsoon/T_2M/' + '01-05.T_2M.cpm.nc')['T_2M'].values[...], axis=0) - 273.15
data['cosmo']['change'] = data['cosmo']['lgm'] - data['cosmo']['ctrl']

# ECHAM5
data['echam'] = {}
dir = "/project/pr133/rxiang/data/pgw/deltas/native/day/ECHAM5/"
data['echam']['ctrl'] = np.nanmean(xr.open_dataset(f'{dir}' + 'ts_piControl.nc')['ts'].values[...], axis=0) - 273.15
data['echam']['lgm'] = np.nanmean(xr.open_dataset(f'{dir}' + 'ts_lgm.nc')['ts'].values[...], axis=0) - 273.15
data['echam']['change'] = data['echam']['lgm'] - data['echam']['ctrl']
lat1 = xr.open_dataset(f'{dir}' + 'ts_lgm.nc')['lat'].values[...]
lon1 = xr.open_dataset(f'{dir}' + 'ts_lgm.nc')['lon'].values[...]

# PMIP
data['pmip'] = {}
dir = "/project/pr133/rxiang/data/pmip/var/ts/"
data['pmip']['ctrl'] = np.nanmean(
    xr.open_dataset(f'{dir}' + 'ts_Amon_PMIP4_piControl.nc')['ts'].values[...],
    axis=0) - 273.15
data['pmip']['lgm'] = np.nanmean(
    xr.open_dataset(f'{dir}' + 'ts_Amon_PMIP4_lgm.nc')['ts'].values[...], axis=0) - 273.15
data['pmip']['change'] = data['pmip']['lgm'] - data['pmip']['ctrl']
lat2 = xr.open_dataset(f'{dir}' + 'ts_Amon_PMIP4_piControl.nc')['lat'].values[...]
lon2 = xr.open_dataset(f'{dir}' + 'ts_Amon_PMIP4_piControl.nc')['lon'].values[...]

# proxy
df = pd.read_excel('/project/pr133/rxiang/script/mapping_cosmo/paper2/lgm/' + "tmp.xlsx", index_col=0)
df_filtered = df[df["Reference"] != 'Liu (1988)']
annt = df_filtered["TANN"].values

# %% ---------------------------------------------------------------------
# -- Plot
labels = ['CTRL | PI', 'PGW | LGM', 'PGW | LGM - CTRL | PI']
lefts = ['COSMO', 'ECHAM5', 'PMIP4']
lb = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']]
models = ['cosmo', 'echam', 'pmip']
sims = ['ctrl', 'lgm', 'change']

[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

fig = plt.figure(figsize=(11, 5.7))
gs1 = gridspec.GridSpec(3, 2, left=0.075, bottom=0.03, right=0.598,
                        top=0.96, hspace=0.02, wspace=0.05,
                        width_ratios=[1, 1], height_ratios=[1, 1, 1])
gs2 = gridspec.GridSpec(3, 1, left=0.675, bottom=0.03, right=0.93,
                        top=0.96, hspace=0.02, wspace=0.05)
ncol = 3  # edit here
nrow = 3

axs, cs, ct, topo, q = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object')

for i in range(nrow):
    axs[i, 0] = fig.add_subplot(gs1[i, 0], projection=rot_pole_crs)
    axs[i, 0] = plotcosmo_notick(axs[i, 0])
    axs[i, 1] = fig.add_subplot(gs1[i, 1], projection=rot_pole_crs)
    axs[i, 1] = plotcosmo_notick_lgm(axs[i, 1], diff=False)
    axs[i, 2] = fig.add_subplot(gs2[i, 0], projection=rot_pole_crs)
    axs[i, 2] = plotcosmo_notick_lgm(axs[i, 2], diff=False)

# --
levels1 = MaxNLocator(nbins=40).tick_values(-20, 20)
cmap1 = cmc.roma_r
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
levels2 = MaxNLocator(nbins=20).tick_values(-10, 0)
cmap2 = cmc.davos
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)
# --
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
lats = [rlat, lat1, lat2]
lons = [rlon, lon1, lon2]
transforms = [rot_pole_crs, ccrs.PlateCarree(), ccrs.PlateCarree()]
# --

for i in range(nrow):
    model = models[i]
    lat_ = lats[i]
    lon_ = lons[i]
    transform_ = transforms[i]
    for j in range(ncol):
        sim = sims[j]
        cmap = cmaps[j]
        norm = norms[j]
        cs[i, j] = axs[i, j].pcolormesh(lon_, lat_, data[model][sim], cmap=cmap, norm=norm, shading="auto",
                                        transform=transform_)

# add proxy data
lon = df_filtered["Longitude"].values
lat = df_filtered["Latitude"].values
axs[0, 2].scatter(lon, lat, c=annt, s=20, linewidths=0.5, cmap=cmap, norm=norm, transform=transform_)
axs[1, 2].scatter(lon, lat, c=annt, s=20, linewidths=0.5, cmap=cmap, norm=norm, transform=transform_)

# for i in range(nrow):
#     for j in range(ncol):
#         label = lb[i][j]
#         t = axs[i, j].text(0.01, 0.985, f'({label})', ha='left', va='top',
#                            transform=axs[i, j].transAxes, fontsize=14)
#         t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

# --
for i in range(nrow):
    cax = fig.add_axes(
        [axs[i, 1].get_position().x1 + 0.01, axs[i, 1].get_position().y0, 0.015, axs[i, 1].get_position().height])
    cbar = fig.colorbar(cs[i, 1], cax=cax, orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.minorticks_off()
    cax = fig.add_axes(
        [axs[i, 2].get_position().x1 + 0.01, axs[i, 2].get_position().y0, 0.015, axs[i, 2].get_position().height])
    cbar = fig.colorbar(cs[i, 2], cax=cax, orientation='vertical', extend='both', ticks=[0, -2, -4, -6, -8, -10])
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.minorticks_off()
# --
for j in range(ncol):
    label = labels[j]
    axs[0, j].set_title(f'{label}', pad=7, fontsize=14, loc='center')
# --
for i in range(nrow):
    left = lefts[i]
    axs[i, 0].text(-0.2, 0.5, f'{left}', ha='right', va='center',
                   transform=axs[i, 0].transAxes, fontsize=14, rotation=90)

for i in range(nrow):
    axs[i, 0].text(-0.008, 0.95, '50°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.77, '40°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.59, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.41, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.23, '10°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.05, '0°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)

for j in range(ncol):
    axs[nrow - 1, j].text(0.12, -0.02, '80°E', ha='center', va='top', transform=axs[nrow - 1, j].transAxes, fontsize=13)
    axs[nrow - 1, j].text(0.32, -0.02, '100°E', ha='center', va='top', transform=axs[nrow - 1, j].transAxes,
                          fontsize=13)
    axs[nrow - 1, j].text(0.52, -0.02, '120°E', ha='center', va='top', transform=axs[nrow - 1, j].transAxes,
                          fontsize=13)
    axs[nrow - 1, j].text(0.72, -0.02, '140°E', ha='center', va='top', transform=axs[nrow - 1, j].transAxes,
                          fontsize=13)
    axs[nrow - 1, j].text(0.92, -0.02, '160°E', ha='center', va='top', transform=axs[nrow - 1, j].transAxes,
                          fontsize=13)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper2/results/gm/"
fig.savefig(plotpath + 'tmp.png', dpi=500)
plt.close(fig)
