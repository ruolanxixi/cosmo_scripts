# -------------------------------------------------------------------------------
# modules
#
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import pole04, colorbar, plotcosmo04_notick, pole
import cartopy.crs as ccrs
from numpy import inf
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_
from mycolor import wind as windmap
from pyproj import Transformer
import scipy.ndimage as ndimage
import matplotlib
from matplotlib.patches import Rectangle
import numpy.ma as ma
import matplotlib.patches as patches

font = {'size': 13}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
sims = ['CTRL04', 'CTRL11', 'IMERG']
seasons = "JJA"
cpmpath = "/project/pr133/rxiang/data/cosmo/EAS04_ctrl/indices"
lsmpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/indices"
imergpath = "/scratch/snx3000/rxiang/IMERG/indices"
paths = [cpmpath, lsmpath, imergpath]
data = {}
vars = ['mean', 'wet_day_freq', 'perc_95.00', 'perc_99.00']

[pole_lat04, pole_lon04, lat04, lon04, rlat04, rlon04, rot_pole_crs04] = pole04()
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

lb = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i'], ['j', 'k', 'l']]

for i in range(len(sims)):
    sim = sims[i]
    path = paths[i]
    data[sim] = {}
    f = xr.open_dataset(f'{path}/2001-2005_JJA_all_day_perc.nc')
    for j in range(len(vars)):
        var = vars[j]
        data[sim][var] = {}
        ds = f[var].values[:, :]
        data[sim][var]["value"] = ds

models = ['EAS04', 'EAS11']
for i in range(2):
    model = models[i]
    sim = sims[i]
    path = f"/project/pr133/rxiang/data/cosmo/{model}_ctrl/remap/TOT_PREC/"
    f = xr.open_dataset(f'{path}/2001-2005_JJA_all_day_perc.remap.imerg_full.nc')
    fslice = f.sel(**{"lat": slice(22, 40),
                      "lon": slice(89, 113)})
    fim = xr.open_dataset(f'{imergpath}/2001-2005_JJA_all_day_perc.nc')
    fimslice = fim.sel(**{"lat": slice(22, 40),
                      "lon": slice(89, 113)})
    for j in range(len(vars)):
        var = vars[j]
        ds = fslice[var].values[:, :]
        dsim = fimslice[var].values[:, :]
        data[sim][var]["R"] = ma.corrcoef(ma.masked_invalid(dsim.flatten()), ma.masked_invalid(ds.flatten()))[0, 1]
        data[sim][var]["BIAS"] = np.nanmean(ds - dsim)

data['CTRL04']['lon'] = rlon04
data['CTRL04']['lat'] = rlat04
data['CTRL04']['proj'] = rot_pole_crs04
data['CTRL11']['lon'] = rlon
data['CTRL11']['lat'] = rlat
data['CTRL11']['proj'] = rot_pole_crs
data['IMERG']['lon'] = f['lon'].values[...]
data['IMERG']['lat'] = f['lat'].values[...]
data['IMERG']['proj'] = ccrs.PlateCarree()

# %%
ar = 1.0  # initial aspect ratio for first trial
wi = 9  # height in inches #15
hi = 9.3  # width in inches #10
ncol = 3  # edit here
nrow = 4
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), np.empty(
    shape=(nrow, ncol), dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.07, 0.02, 0.90, 0.98
gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.1, hspace=0.01)

level1 = MaxNLocator(nbins=20).tick_values(0, 20)
cmap1 = cmc.davos_r
norm1 = BoundaryNorm(level1, ncolors=cmap1.N, clip=True)

level2 = MaxNLocator(nbins=18).tick_values(0, 0.9)
cmap2 = cmc.roma_r
norm2 = BoundaryNorm(level2, ncolors=cmap2.N, clip=True)

level3 = MaxNLocator(nbins=18).tick_values(0, 90)
cmap3 = cmc.davos_r
norm3 = BoundaryNorm(level3, ncolors=cmap2.N, clip=True)

level4 = MaxNLocator(nbins=18).tick_values(0, 90)
cmap4 = cmc.davos_r
norm4 = BoundaryNorm(level4, ncolors=cmap2.N, clip=True)

cmaps = [cmap1, cmap2, cmap3, cmap4]
norms = [norm1, norm2, norm3, norm4]
levels = [level1, level2, level3, level4]

for i in range(len(vars)):
    var = vars[i]
    cmap, norm, level = cmaps[i], norms[i], levels[i]
    for j in range(ncol):
        sim = sims[j]
        axs[i, j] = fig.add_subplot(gs[i, j], projection=rot_pole_crs04)
        axs[i, j] = plotcosmo04_notick(axs[i, j])
        cs[i, j] = axs[i, j].pcolormesh(data[sim]['lon'], data[sim]['lat'], data[sim][var]["value"],
                                        cmap=cmap, norm=norm, shading="auto", transform=data[sim]['proj'])

for i in range(nrow):
    axs[i, 0].text(-0.01, 0.83, '35°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.57, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.31, '25°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.05, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)

for j in range(ncol):
    axs[3, j].text(0.04, -0.02, '90°E', ha='center', va='top', transform=axs[3, j].transAxes, fontsize=14)
    axs[3, j].text(0.46, -0.02, '100°E', ha='center', va='top', transform=axs[3, j].transAxes, fontsize=14)
    axs[3, j].text(0.86, -0.02, '110°E', ha='center', va='top', transform=axs[3, j].transAxes, fontsize=14)

for i in range(nrow):
    for j in range(ncol):
        label = lb[i][j]
        t = axs[i, j].text(0.01, 0.985, f'({label})', ha='left', va='top',
                           transform=axs[i, j].transAxes, fontsize=14)
        t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

for j in range(ncol):
    title = sims[j]
    axs[0, j].set_title(f'{title}', pad=5, fontsize=14, loc='center')

for i in range(len(vars)):
    var = vars[i]
    rect = patches.Rectangle((0.25, 0.75), 0.8, 0.24, linewidth=1, edgecolor='none', facecolor='white', alpha=0.5,
                             transform=axs[i, 2].transAxes)
    axs[i, 2].add_patch(rect)
    t1 = axs[i, 2].text(0.78, 0.95, "BIAS", fontsize=9, horizontalalignment='center',
                        verticalalignment='center', transform=axs[i, 2].transAxes)
    t2 = axs[i, 2].text(0.92, 0.95, "R", fontsize=9, horizontalalignment='center',
                        verticalalignment='center', transform=axs[i, 2].transAxes)
    t3 = axs[i, 2].text(0.48, 0.87, f"CTRL04 - IMERG", fontsize=9, horizontalalignment='center',
                        verticalalignment='center', transform=axs[i, 2].transAxes)
    t4 = axs[i, 2].text(0.48, 0.79, f"CTRL11 - IMERG", fontsize=9, horizontalalignment='center',
                        verticalalignment='center', transform=axs[i, 2].transAxes)
    # Create a Rectangle patch
    # Add the patch to the Axes
    txt = data['CTRL04'][var]['BIAS']
    t5 = axs[i, 2].text(0.78, 0.87, '%0.2f' % txt, fontsize=9, horizontalalignment='center',
                       verticalalignment='center', transform=axs[i, 2].transAxes)
    txt = data['CTRL11'][var]['BIAS']
    t6 = axs[i, 2].text(0.78, 0.79, '%0.2f' % txt, fontsize=9, horizontalalignment='center',
                       verticalalignment='center', transform=axs[i, 2].transAxes)
    txt = data['CTRL04'][var]['R']
    t7 = axs[i, 2].text(0.92, 0.87, '%0.2f' % txt, fontsize=9, horizontalalignment='center',
                       verticalalignment='center', transform=axs[i, 2].transAxes)
    txt = data['CTRL11'][var]['R']
    t8 = axs[i, 2].text(0.92, 0.79, '%0.2f' % txt, fontsize=9, horizontalalignment='center',
                       verticalalignment='center', transform=axs[i, 2].transAxes)

extends = ['max', 'neither', 'neither', 'neither']
for i in range(nrow):
    extend = extends[i]
    cax = fig.add_axes(
        [axs[i, 2].get_position().x1 + 0.01, axs[i, 2].get_position().y0, 0.015, axs[i, 2].get_position().height])
    cbar = fig.colorbar(cs[i, 2], cax=cax, orientation='vertical', extend='max')
    cbar.ax.tick_params(labelsize=13)

plt.show()
