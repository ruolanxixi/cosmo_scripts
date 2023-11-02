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
data['COSMO'] = {}
dir = "/project/pr133/rxiang/data/cosmo/"
for i in ("ctrl", "lgm"):
    data['COSMO'][i] = {}
    for j in ("DJF", "JJA"):
        ds = xr.open_mfdataset(f'{dir}' + f'EAS11_{i}/szn/U_10M/' + f'01-05.U_10M.{j}.nc')
        u = ds['U_10M'].values[...]
        ds = xr.open_mfdataset(f'{dir}' + f'EAS11_{i}/szn/V_10M/' + f'01-05.V_10M.{j}.nc')
        v = ds['V_10M'].values[...]
        data['COSMO'][i][j] = {"u": np.nanmean(u, axis=0),
                               "v": np.nanmean(v, axis=0),
                               "ws": np.nanmean(np.sqrt(u**2+v**2), axis=0)}

data['COSMO']['diff'] = {}
for j in ("DJF", "JJA"):
    data['COSMO']['diff'][j] = {"u": data['COSMO']['lgm'][j]['u'] - data['COSMO']['ctrl'][j]['u'],
                                "v": data['COSMO']['lgm'][j]['v'] - data['COSMO']['ctrl'][j]['v'],
                                "ws": data['COSMO']['lgm'][j]['ws'] - data['COSMO']['ctrl'][j]['ws']}

# ECHAM5
data['ECHAM5'] = {}
dir = "/project/pr133/rxiang/data/pgw/deltas/native/day/ECHAM5/"
for i in ("ctrl", "lgm"):
    data['ECHAM5'][i] = {}
    sim = i
    if i == "ctrl":
        sim = "piControl"
    for j in ("DJF", "JJA"):
        ds = xr.open_mfdataset(f'{dir}' + f'u10_{sim}.nc')
        ds = ds.sel(time=ds['time.season'] == j)
        u = ds['u10'].values[...]
        ds = xr.open_mfdataset(f'{dir}' + f'v10_{sim}.nc')
        ds = ds.sel(time=ds['time.season'] == j)
        v = ds['v10'].values[...]
        data['ECHAM5'][i][j] = {"u": np.nanmean(u, axis=0),
                                "v": np.nanmean(v, axis=0),
                                "ws": np.nanmean(np.sqrt(u**2+v**2), axis=0)}

data['ECHAM5']['diff'] = {}
for j in ("DJF", "JJA"):
    data['ECHAM5']['diff'][j] = {"u": data['ECHAM5']['lgm'][j]['u'] - data['ECHAM5']['ctrl'][j]['u'],
                                 "v": data['ECHAM5']['lgm'][j]['v'] - data['ECHAM5']['ctrl'][j]['v'],
                                 "ws": data['ECHAM5']['lgm'][j]['ws'] - data['ECHAM5']['ctrl'][j]['ws']}

lat1 = xr.open_dataset(f'{dir}' + 'pr_lgm.nc')['lat'].values[...]
lon1 = xr.open_dataset(f'{dir}' + 'pr_lgm.nc')['lon'].values[...]
# %% ---------------------------------------------------------------------
# -- Plot
labels = ['CTRL | PI', 'PGW | LGM', 'PGW | LGM - CTRL | PI']
lefts = ['COSMO', 'ECHAM5', 'COSMO', 'ECHAM5']
lb = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i'], ['j', 'k', 'l']]
sims = ['ctrl', 'lgm', 'diff']
mons = ["DJF", "JJA"]
models = ['COSMO', 'ECHAM5']
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

fig = plt.figure(figsize=(11, 7.5))
gs = gridspec.GridSpec(4, 4, left=0.1, bottom=0.04, right=0.94,
                        top=0.94, hspace=0.05, wspace=0.05,
                        width_ratios=[1, 1, 0.15, 1], height_ratios=[1, 1, 1, 1])
ncol = 3  # edit here
nrow = 4

axs, cs, ct, topo, q = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object')

for i in range(nrow):
    axs[i, 0] = fig.add_subplot(gs[i, 0], projection=rot_pole_crs)
    axs[i, 0] = plotcosmo_notick(axs[i, 0])
    axs[i, 1] = fig.add_subplot(gs[i, 1], projection=rot_pole_crs)
    axs[i, 1] = plotcosmo_notick_lgm(axs[i, 1], diff=False)
    axs[i, 2] = fig.add_subplot(gs[i, 3], projection=rot_pole_crs)
    axs[i, 2] = plotcosmo_notick_lgm(axs[i, 2], diff=True)

# --
levels1 = MaxNLocator(nbins=20).tick_values(0, 10)
cmap1 = cmc.roma_r
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
levels2 = MaxNLocator(nbins=15).tick_values(-2, 2)
cmap2 = custom_div_cmap(25, cmc.vik)
# cmap2 = cmc.vik
norm2 = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2.)
# --
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
# --
lats = [rlat, lat1]
lons = [rlon, lon1]
transforms = [rot_pole_crs, ccrs.PlateCarree()]

for i in range(2):
    model = models[i]
    lat_ = lats[i]
    lon_ = lons[i]
    transform_ = transforms[i]
    for j in range(ncol):
        sim = sims[j]
        cmap = cmaps[j]
        norm = norms[j]
        cs[i, j] = axs[i, j].streamplot(lon_, lat_, data[model][sim]['DJF']['u'], data[model][sim]['DJF']['v'], color=data[model][sim]['DJF']['ws'],
                                        density=1.1, cmap=cmap, norm=norm, arrowstyle='->', transform=transform_)

for i in range(2):
    model = models[i]
    lat_ = lats[i]
    lon_ = lons[i]
    transform_ = transforms[i]
    for j in range(ncol):
        sim = sims[j]
        cmap = cmaps[j]
        norm = norms[j]
        cs[i+2, j] = axs[i+2, j].streamplot(lon_, lat_, data[model][sim]['JJA']['u'], data[model][sim]['JJA']['v'], color=data[model][sim]['JJA']['ws'],
                                        density=1.1, cmap=cmap, norm=norm, arrowstyle='->', transform=transform_)

for i in range(nrow):
    for j in range(ncol):
        label = lb[i][j]
        t = axs[i, j].text(0.01, 0.985, f'({label})', ha='left', va='top',
                           transform=axs[i, j].transAxes, fontsize=14)
        t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

# --
for i in range(nrow):
    cax = fig.add_axes(
        [axs[i, 1].get_position().x1 + 0.01, axs[i, 1].get_position().y0, 0.015, axs[i, 1].get_position().height])
    cbar = fig.colorbar(cs[i, 1].lines, cax=cax, orientation='vertical', extend='max', ticks=[0, 2, 4, 6, 8, 10])
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.minorticks_off()
    cax = fig.add_axes(
        [axs[i, 2].get_position().x1 + 0.01, axs[i, 2].get_position().y0, 0.015, axs[i, 2].get_position().height])
    cbar = fig.colorbar(cs[i, 2].lines, cax=cax, orientation='vertical', extend='both', ticks=[-2, -1, 0, 1, 2])
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

axs[0, 0].text(axs[0, 0].get_position().x0 - 0.07, (axs[0, 0].get_position().y0+axs[1, 0].get_position().y1)/2, 'DJF', ha='right', va='center', fontsize=14, rotation=90, transform=fig.transFigure)
axs[2, 0].text(axs[2, 0].get_position().x0 - 0.07, (axs[2, 0].get_position().y0+axs[3, 0].get_position().y1)/2, 'JJA', ha='right', va='center', fontsize=14, rotation=90, transform=fig.transFigure)

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
plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + 'uv10_season_compare.png', dpi=500)
plt.close(fig)
