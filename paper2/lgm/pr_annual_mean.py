# fsnow = MAX[0.01; MIN(1, Wsnow/0.015]

# -------------------------------------------------------------------------------
# modules
#
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
from mycolor import custom_white_cmap, drywet
from plotcosmomap import plotcosmo_notick, pole, pole04, plotcosmo04_notick, plotcosmo_notick_lgm, plotcosmo04_notick_lgm
import matplotlib.colors as colors
import scipy.ndimage as ndimage
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from metpy.units import units
import metpy.calc as mpcalc

###############################################################################
#%% Data
###############################################################################
sims = ['ctrl', 'lgm']
path = "/project/pr133/rxiang/data/cosmo/"

data = {}
for s in range(len(sims)):
    sim = sims[s]
    data[sim] = {}
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/TOT_PREC/' + '01-05.TOT_PREC.cpm.nc')
    tmp = np.mean(ds['TOT_PREC'].values[...], axis=0)
    data[sim]['TOT_PREC'] = tmp

data['diff'] = {}
data['diff']['TOT_PREC'] = data['lgm']['TOT_PREC'] - data['ctrl']['TOT_PREC']

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
axs[0, 2] = plotcosmo_notick_lgm(axs[0, 2], diff=False)

levels1 = MaxNLocator(nbins=20).tick_values(0, 20)
cmap1 = cmc.davos_r
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
levels2 = MaxNLocator(nbins=11).tick_values(-10, 10)
cmap2 = drywet(25, cmc.vik_r)
norm2 = colors.TwoSlopeNorm(vmin=-6., vcenter=0., vmax=6.)
# norm2 = colors.TwoSlopeNorm(vmin=-10., vcenter=0., vmax=10.)
# --
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
# --
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    cs[0, j] = axs[0, j].pcolormesh(rlon, rlat, data[sim]['TOT_PREC'], cmap=cmap, norm=norm, shading="auto")

cax = fig.add_axes(
    [axs[0, 1].get_position().x1 + 0.01, axs[0, 1].get_position().y0, 0.015, axs[0, 1].get_position().height])
cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical', extend='max', ticks=[0, 5, 10, 15, 20])
cbar.ax.minorticks_off()
cbar.ax.tick_params(labelsize=13)
cax = fig.add_axes(
    [axs[0, 2].get_position().x1 + 0.01, axs[0, 2].get_position().y0, 0.015, axs[0, 2].get_position().height])
cbar = fig.colorbar(cs[0, 2], cax=cax, orientation='vertical', extend='both', ticks=[-6, -3, 0, 3, 6])
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

plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + 'pr.png', dpi=500, transparent='True')
plt.show()
plt.close()

# # %%
# [pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()
# rlon_, rlat_ = np.meshgrid(rlon, rlat)
# sims = ['ctrl', 'lgm', 'diff']
# fig = plt.figure(figsize=(9.5, 3))
# gs1 = gridspec.GridSpec(1, 2, left=0.05, bottom=0.03, right=0.585,
#                         top=0.96, hspace=0.05, wspace=0.05,
#                         width_ratios=[1, 1])
# gs2 = gridspec.GridSpec(1, 1, left=0.664, bottom=0.03, right=0.925,
#                         top=0.96, hspace=0.05, wspace=0.05)
# ncol = 3  # edit here
# nrow = 1
# 
# axs, cs, ct, topo, q = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
#                        np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
#                        np.empty(shape=(nrow, ncol), dtype='object')
# 
# axs[0, 0] = fig.add_subplot(gs1[0, 0], projection=rot_pole_crs)
# axs[0, 0] = plotcosmo04_notick(axs[0, 0])
# axs[0, 1] = fig.add_subplot(gs1[0, 1], projection=rot_pole_crs)
# axs[0, 1] = plotcosmo04_notick_lgm(axs[0, 1], diff=False)
# axs[0, 2] = fig.add_subplot(gs2[0, 0], projection=rot_pole_crs)
# axs[0, 2] = plotcosmo04_notick_lgm(axs[0, 2], diff=True)
# 
# # --
# for j in range(ncol):
#     sim = sims[j]
#     cmap = cmaps[j]
#     norm = norms[j]
#     cs[0, j] = axs[0, j].pcolormesh(rlon, rlat, data[sim]['TOT_PREC'], cmap=cmap, norm=norm, shading="auto")
# 
# cax = fig.add_axes(
#     [axs[0, 1].get_position().x1 + 0.01, axs[0, 1].get_position().y0, 0.015, axs[0, 1].get_position().height])
# cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical', extend='both', ticks=[-20, -10, 0, 10, 20])
# cbar.ax.minorticks_off()
# cbar.ax.tick_params(labelsize=13)
# cax = fig.add_axes(
#     [axs[0, 2].get_position().x1 + 0.01, axs[0, 2].get_position().y0, 0.015, axs[0, 2].get_position().height])
# cbar = fig.colorbar(cs[0, 2], cax=cax, orientation='vertical', extend='min', ticks=[-10, -8, -6, -4, -2, 0])
# cbar.ax.minorticks_off()
# cbar.ax.tick_params(labelsize=13)
# # --
# labels = ['PD', 'LGM', 'LGM - PD']
# for j in range(ncol):
#     label = labels[j]
#     axs[0, j].set_title(f'{label}', pad=7, fontsize=14, loc='center')
# # --
# for i in range(nrow):
#     axs[i, 0].text(-0.01, 0.83, '35°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
#     axs[i, 0].text(-0.01, 0.57, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
#     axs[i, 0].text(-0.01, 0.31, '25°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
#     axs[i, 0].text(-0.01, 0.05, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
# 
# for j in range(ncol):
#     axs[0, j].text(0.06, -0.02, '90°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=13)
#     axs[0, j].text(0.46, -0.02, '100°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=13)
#     axs[0, j].text(0.86, -0.02, '110°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=13)
# 
# plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
# fig.savefig(plotpath + 't2m_local.png', dpi=500, transparent='True')
# plt.show()





