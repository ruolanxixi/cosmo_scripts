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

mpl.style.use("classic")
font = {'size': 14}
matplotlib.rc('font', **font)
plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'dimgrey'

###############################################################################
# Data
###############################################################################
sims = ['ctrl', 'lgm']
path = "/project/pr133/rxiang/data/cosmo/"

data = {}
labels = ['PD', 'LGM', 'LGM - PD']
lb = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i'], ['j', 'k', 'l']]

g = 9.80665

vars = ['FI700', 'T700', 'FI500', 'T500']
# load data
for s in range(len(sims)):
    sim = sims[s]
    data[sim] = {}
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/FI/' + f'01-05.FI.70000.smr.yearmean.nc')
    smr = ds['FI'].values[:, 0, :, :] / g
    data[sim]['FI700'] = smr
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/T/' + f'01-05.T.70000.smr.yearmean.nc')
    smr = ds['T'][:, 0, :, :]
    data[sim]['T700'] = smr
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/FI/' + f'01-05.FI.50000.smr.yearmean.nc')
    smr = ds['FI'].values[:, 0, :, :] / g
    data[sim]['FI500'] = smr
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/T/' + f'01-05.T.50000.smr.yearmean.nc')
    smr = ds['T'][:, 0, :, :]
    data[sim]['T500'] = smr

# compute difference
data['diff'] = {}
for v in range(len(vars)):
    var = vars[v]
    data['diff'][var] = data['lgm'][var] - data['ctrl'][var]

###############################################################################
# %% Plot
###############################################################################
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()
rlon_, rlat_ = np.meshgrid(rlon, rlat)
sims = ['ctrl', 'lgm', 'diff']
fig = plt.figure(figsize=(11, 3))
gs1 = gridspec.GridSpec(1, 2, left=0.05, bottom=0.03, right=0.585,
                        top=0.96, hspace=0.05, wspace=0.05,
                        width_ratios=[1, 1])
gs2 = gridspec.GridSpec(1, 1, left=0.664, bottom=0.03, right=0.925,
                        top=0.96, hspace=0.05, wspace=0.05)
ncol = 3  # edit here
nrow = 1

axs, cs, ct, topo, q = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object')


axs[0, 0] = fig.add_subplot(gs1[0, 0], projection=rot_pole_crs)
axs[0, 0] = plotcosmo_notick(axs[0, 0])
axs[0, 1] = fig.add_subplot(gs1[0, 1], projection=rot_pole_crs)
axs[0, 1] = plotcosmo_notick_lgm(axs[0, 1], diff=False)
axs[0, 2] = fig.add_subplot(gs2[0, 0], projection=rot_pole_crs)
axs[0, 2] = plotcosmo_notick_lgm(axs[0, 2], diff=True)

levels1 = MaxNLocator(nbins=20).tick_values(12, 22)
cmap1 = cmc.roma_r
# cmap1 = custom_white_cmap(100, cmc.roma_r)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
levels2 = MaxNLocator(nbins=20).tick_values(-10, -10)
# cmap2 = drywet(20, cmc.vik_r)
cmap2 = cmc.davos_r
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)
# norm2 = colors.TwoSlopeNorm(vmin=-100., vcenter=0., vmax=100.)
# --
levels1 = np.linspace(-3000, -2400, 21, endpoint=True)
levels2 = [-60, -40, -20, 20, 40, 60]
# --
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
# --
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    level = levels[j]
    cs[0, j] = axs[0, j].pcolormesh(rlon, rlat, np.nanmean(data[sim]['T700'] - data[sim]['T500'], axis=0), cmap=cmap, norm=norm, shading="auto")
    ct[0, j] = axs[0, j].contour(rlon, rlat, np.nanmean(data[sim]['FI700'] - data[sim]['FI500'], axis=0), levels=level,
                                 colors='k', linewidths=.8)
    clabel = axs[0, j].clabel(ct[0, j], levels=level, inline=True, fontsize=10,
                              use_clabeltext=True)

cax = fig.add_axes(
    [axs[0, 1].get_position().x1 + 0.01, axs[0, 1].get_position().y0, 0.015, axs[0, 1].get_position().height])
cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical', extend='both', ticks=[12, 14, 16, 18, 20, 22])
cbar.ax.minorticks_off()
cbar.ax.tick_params(labelsize=13)
cax = fig.add_axes(
    [axs[0, 2].get_position().x1 + 0.01, axs[0, 2].get_position().y0, 0.015, axs[0, 2].get_position().height])
cbar = fig.colorbar(cs[0, 2], cax=cax, orientation='vertical', extend='max', ticks=[-10, -5, 0, 5, 10])
cbar.ax.minorticks_off()
cbar.ax.tick_params(labelsize=13)
# --
labels = ['PD', 'LGM', 'LGM - PD']
for j in range(ncol):
    label = labels[j]
    axs[0, j].set_title(f'{label}', pad=7, fontsize=14, loc='center')
# --
for i in range(nrow):
    axs[i, 0].text(-0.008, 0.95, '50°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.77, '40°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.59, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.41, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.23, '10°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.05, '0°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)

for j in range(ncol):
    axs[0, j].text(0.12, -0.02, '80°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=13)
    axs[0, j].text(0.32, -0.02, '100°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=13)
    axs[0, j].text(0.52, -0.02, '120°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=13)
    axs[0, j].text(0.72, -0.02, '140°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=13)
    axs[0, j].text(0.92, -0.02, '160°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=13)

# plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
# fig.savefig(plotpath + 'nsnow_large.png', dpi=500, transparent='True')
plt.show()
plt.close()
