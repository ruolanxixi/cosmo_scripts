###############################################################################
# Modules
###############################################################################
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib
import cartopy.feature as feature
import matplotlib.ticker as mticker
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import drywet
from plotcosmomap import plotcosmo_notick, pole, plotcosmo_notick_lgm
import matplotlib.colors as colors
import scipy.ndimage as ndimage
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from mycolor import custom_div_cmap
import metpy.calc as mpcalc
from metpy.units import units

mpl.style.use("classic")
font = {'size': 13}
matplotlib.rc('font', **font)
plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'dimgrey'

###############################################################################
# %% Plot
###############################################################################
fig = plt.figure(figsize=(10.2, 9.5))
gs = gridspec.GridSpec(3, 2, left=0.07, bottom=0.03, right=0.99,
                       top=0.96, hspace=0.05, wspace=0.05,
                       width_ratios=[1, 1], height_ratios=[1, 1, 1])
path = '/project/pr133/rxiang/data/pmip/'

models = ('AWI-ESM-1-1-LR', 'CESM2-WACCM-FV2', 'INM-CM4-8', 'MIROC-ES2L', 'MPI-ESM1-2-LR')
levels = MaxNLocator(nbins=11).tick_values(-4, 4)
cmap = drywet(25, cmc.vik_r)
norm = colors.TwoSlopeNorm(vmin=-4., vcenter=0., vmax=4.)
for i in range(5):
    model = models[i]
    ax = fig.add_subplot(gs[i], projection=ccrs.PlateCarree())
    ax.add_feature(feature.BORDERS, linestyle="-", linewidth=0.6)
    ax.add_feature(feature.COASTLINE, linestyle="-", linewidth=0.6)
    #ax.set_aspect("auto")
    ax.set_extent([65, 173, 0, 60], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.7, linestyle='--')
    gl.xlocator = mticker.FixedLocator([60, 80, 100, 120, 140, 160, 180])
    gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])
    ds = xr.open_dataset(f'{path}' + f'lgm/{model}/pr/' + f'pr_Amon_{model}_lgm.nc')
    pr_lgm = np.nanmean(ds['pr'].values[...], axis=0) * 84000
    ds = xr.open_dataset(f'{path}' + f'piControl/{model}/pr/' + f'pr_Amon_{model}_piControl.nc')
    pr_pi = np.nanmean(ds['pr'].values[...], axis=0) * 84000
    lat = ds["lat"].values
    lon = ds["lon"].values
    cs = plt.pcolormesh(lon, lat, pr_lgm - pr_pi, cmap=cmap, norm=norm)
    plt.title(f'{model}', fontsize=14, y=1.007)
    if i == 0 or i == 2:
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = False
    if i == 4:
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
    if i == 3:
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = False
        
cax = fig.add_axes([ax.get_position().x1+0.02, ax.get_position().y0, 0.015, ax.get_position().height])
cbar = fig.colorbar(cs, cax=cax, orientation='vertical', extend='both', ticks=[-4, -2, 0, 2, 4],
                    label="[mm/day]")
cbar.ax.tick_params(labelsize=13)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + 'pmip_pr.png', dpi=500, transparent='true')
plt.close(fig)

# %%
fig = plt.figure(figsize=(10.2, 9.5))
gs = gridspec.GridSpec(3, 2, left=0.07, bottom=0.03, right=0.99,
                       top=0.96, hspace=0.05, wspace=0.05,
                       width_ratios=[1, 1], height_ratios=[1, 1, 1])
path = '/project/pr133/rxiang/data/pmip/'

models = ('AWI-ESM-1-1-LR', 'CESM2-WACCM-FV2', 'INM-CM4-8', 'MIROC-ES2L', 'MPI-ESM1-2-LR')
levels = MaxNLocator(nbins=11).tick_values(-10, 10)
cmap = cmc.vik
norm = colors.TwoSlopeNorm(vmin=-10., vcenter=0., vmax=10.)
for i in range(5):
    model = models[i]
    ax = fig.add_subplot(gs[i], projection=ccrs.PlateCarree())
    ax.add_feature(feature.BORDERS, linestyle="-", linewidth=0.6)
    ax.add_feature(feature.COASTLINE, linestyle="-", linewidth=0.6)
    # ax.set_aspect("auto")
    ax.set_extent([65, 173, 0, 60], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.7, linestyle='--')
    gl.xlocator = mticker.FixedLocator([60, 80, 100, 120, 140, 160, 180])
    gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])
    ds = xr.open_dataset(f'{path}' + f'lgm/{model}/ts/' + f'ts_Amon_{model}_lgm.nc')
    ts_lgm = np.nanmean(ds['ts'].values[...], axis=0)
    ds = xr.open_dataset(f'{path}' + f'piControl/{model}/ts/' + f'ts_Amon_{model}_piControl.nc')
    ts_pi = np.nanmean(ds['ts'].values[...], axis=0)
    lat = ds["lat"].values
    lon = ds["lon"].values
    cs = plt.pcolormesh(lon, lat, ts_lgm - ts_pi, cmap=cmap, norm=norm)
    plt.title(f'{model}', fontsize=14, y=1.007)
    if i == 0 or i == 2:
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = False
    if i == 4:
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
    if i == 3:
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = False

cax = fig.add_axes([ax.get_position().x1 + 0.02, ax.get_position().y0, 0.015, ax.get_position().height])
cbar = fig.colorbar(cs, cax=cax, orientation='vertical', extend='both', ticks=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10],
                    label="[$^o$C]")
cbar.ax.tick_params(labelsize=13)
# cbar.xlabel("[$^o$C]", fontsize=11)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + 'pmip_ts.png', dpi=500, transparent='true')
plt.close(fig)
