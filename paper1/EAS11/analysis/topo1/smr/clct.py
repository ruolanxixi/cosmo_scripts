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
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import drywet
from plotcosmomap import plotcosmo_notick, pole, plotcosmo_notick_nogrid
import matplotlib.colors as colors
import scipy.ndimage as ndimage
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from mycolor import custom_div_cmap
import metpy.calc as mpcalc
from metpy.units import units

mpl.style.use("classic")
font = {'size': 14}
matplotlib.rc('font', **font)
plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'dimgrey'

###############################################################################
# Function
###############################################################################
def compute_pvalue(ctrl, topo):
    ctrl = np.array(ctrl)
    topo = np.array(topo)
    p = np.zeros((int(ctrl.shape[1]), int(ctrl.shape[2]))) # make sure the shape tuple contains only integers
    for i in range(ctrl.shape[1]):
        for j in range(ctrl.shape[2]):
            ii, jj = mannwhitneyu(ctrl[:, i, j], topo[:, i, j], alternative='two-sided')
            p[i, j] = jj
    p_values = multipletests(p.flatten(), alpha=0.05, method='fdr_bh')[1].reshape((int(ctrl.shape[1]), int(ctrl.shape[2]))) # make sure the shape tuple contains only integers
    return p, p_values

###############################################################################
# Data
###############################################################################
sims = ['ctrl', 'topo1']
path = "/scratch/snx3000/rxiang/data/cosmo/"

ds = xr.open_dataset(f'{path}' + 'EAS11_ctrl/monsoon/CLCT/' + f'01-05.CLCT.smr.yearmean.nc')
ctrl = ds['CLCT'].values[...]*100
ds = xr.open_dataset(f'{path}' + 'EAS11_topo1/monsoon/CLCT/' + f'01-05.CLCT.smr.yearmean.nc')
topo1 = ds['CLCT'].values[...]*100
diff = topo1 - ctrl

p1, corr_p1 = compute_pvalue(ctrl, topo1)
mask1 = np.full_like(p1, fill_value=np.nan)
mask1[p1 > 0.05] = 1
np.save('/project/pr133/rxiang/data/cosmo/sgnfctt/topo1_CLCT.npy', mask1)

###############################################################################
# %% Plot
###############################################################################
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

fig = plt.figure(figsize=(11, 2.5))
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
axs[0, 1] = plotcosmo_notick(axs[0, 1])
axs[0, 2] = fig.add_subplot(gs2[0, 0], projection=rot_pole_crs)
axs[0, 2] = plotcosmo_notick_nogrid(axs[0, 2])

cmap1 = cmc.roma
levels1 = MaxNLocator(nbins=20).tick_values(0, 100)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
levels2 = MaxNLocator(nbins=15).tick_values(-15, 12)
cmap2 = custom_div_cmap(25, cmc.vik)
norm2 = colors.TwoSlopeNorm(vmin=-15, vcenter=0., vmax=12)

cs[0, 0] = axs[0, 0].pcolormesh(rlon, rlat, np.nanmean(ctrl, axis=0), cmap=cmap1, norm=norm1)
cs[0, 1] = axs[0, 1].pcolormesh(rlon, rlat, np.nanmean(topo1, axis=0), cmap=cmap1, norm=norm1)
cs[0, 2] = axs[0, 2].pcolormesh(rlon, rlat, np.nanmean(diff, axis=0), cmap=cmap2, norm=norm2)
ha = axs[0, 2].contourf(rlon, rlat, mask1, levels=1, colors='none', hatches=['////'], rasterized=True, zorder=101)

cax = fig.add_axes([axs[0, 1].get_position().x1+0.01, axs[0, 1].get_position().y0, 0.015, axs[0, 1].get_position().height])
cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical', extend='both')
cbar.ax.tick_params(labelsize=13)
cax = fig.add_axes([axs[0, 2].get_position().x1+0.01, axs[0, 2].get_position().y0, 0.015, axs[0, 2].get_position().height])
cbar = fig.colorbar(cs[0, 2], cax=cax, orientation='vertical', extend='both')
cbar.ax.tick_params(labelsize=13)

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


plt.show()
