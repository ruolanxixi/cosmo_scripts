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
dir = "/project/pr133/rxiang/data/cosmo/"
for i in ("ctrl", "lgm"):
    data[i] = {}
    for j in ("DJF", 'MAM', 'JJA', 'SON'):
        ds = xr.open_mfdataset(f'{dir}' + f'EAS11_{i}/szn/IVT/' + f'01-05.IVT.{j}.nc')
        iuq = ds['IUQ'].values[:, :, :]
        ivq = ds['IVQ'].values[:, :, :]
        data[i][j] = {"iuq": iuq,
                      "ivq": ivq,
                      "tqf": np.sqrt(iuq ** 2 + ivq ** 2)}

data['diff'] = {}
for j in ("DJF", 'MAM', 'JJA', 'SON'):
    data['diff'][j] = {"iuq": data['lgm'][j]['iuq'] - data['ctrl'][j]['iuq'],
                       "ivq": data['lgm'][j]['ivq'] - data['ctrl'][j]['ivq'],
                       "tqf": data['lgm'][j]['tqf'] - data['ctrl'][j]['tqf']}

# %% ---------------------------------------------------------------------
# -- Plot
labels = ['CTRL', 'PGW', 'PGW - CTRL']
lefts = ["DJF", 'MAM', 'JJA', 'SON']
lb = [['a', 'b', 'c'], ['d', 'e', 'f']]
sims = ['ctrl', 'lgm', 'diff']
mons = ["DJF", 'MAM', 'JJA', 'SON']
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

fig = plt.figure(figsize=(11, 7.5))
gs1 = gridspec.GridSpec(4, 2, left=0.075, bottom=0.05, right=0.598,
                        top=0.94, hspace=0.02, wspace=0.05,
                        width_ratios=[1, 1], height_ratios=[1, 1, 1, 1])
gs2 = gridspec.GridSpec(4, 1, left=0.675, bottom=0.05, right=0.93,
                        top=0.94, hspace=0.02, wspace=0.05, height_ratios=[1, 1, 1, 1])
ncol = 3  # edit here
nrow = 4

axs, cs, ct, topo, q = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object')

for i in range(nrow):
    axs[i, 0] = fig.add_subplot(gs1[i, 0], projection=rot_pole_crs)
    axs[i, 0] = plotcosmo_notick(axs[i, 0])
    axs[i, 1] = fig.add_subplot(gs1[i, 1], projection=rot_pole_crs)
    axs[i, 1] = plotcosmo_notick_lgm(axs[i, 1], diff=False)
    axs[i, 2] = fig.add_subplot(gs2[i, 0], projection=rot_pole_crs)
    axs[i, 2] = plotcosmo_notick_lgm(axs[i, 2], diff=True)

# --
levels1 = np.linspace(200, 500, 13, endpoint=True)
# cmap1 = plt.cm.get_cmap("Spectral")
cmap1 = cmc.roma
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
levels2 = MaxNLocator(nbins=21).tick_values(-40, 40)
cmap2 = drywet(25, cmc.vik_r)
norm2 = colors.TwoSlopeNorm(vmin=-40., vcenter=0., vmax=40.)
# --
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
scales = [8000, 8000, 8000]
# --

for i in range(nrow):
    mon = mons[i]
    for j in range(ncol):
        sim = sims[j]
        cmap = cmaps[j]
        norm = norms[j]
        scale = scales[j]
        cs[i, j] = axs[i, j].quiver(rlon[::15], rlat[::15], data[sim][mon]['iuq'][0, ::15, ::15],
                                    data[sim][mon]['ivq'][0, ::15, ::15], data[sim][mon]['tqf'][0, ::15, ::15],
                                    cmap=cmap, norm=norm, scale=scale, headaxislength=3.5, headwidth=5, minshaft=0)

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
    cbar = fig.colorbar(cs[i, 1], cax=cax, orientation='vertical', extend='both', ticks = [200, 250, 300, 350, 400, 450, 500])
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.minorticks_off()
    cax = fig.add_axes(
        [axs[i, 2].get_position().x1 + 0.01, axs[i, 2].get_position().y0, 0.015, axs[i, 2].get_position().height])
    cbar = fig.colorbar(cs[i, 2], cax=cax, orientation='vertical', extend='both', ticks=np.linspace(-40, 40, 5, endpoint=True))
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
fig.savefig(plotpath + 'ivt.png', dpi=500)
plt.close(fig)
