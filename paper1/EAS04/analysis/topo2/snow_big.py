# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import pole04, colorbar, plotcosmo04_notick, pole
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap
import numpy.ma as ma
import matplotlib.patches as patches
import scipy.ndimage as ndimage

font = {'size': 13}

def prcp(numcolors):
    colvals = [[255, 255, 255, 255],
               [254, 217, 118, 255], # [254, 178, 76, 255],
               [255, 237, 160, 255],
               [237, 250, 194, 255],
               [205, 255, 205, 255],
               [153, 240, 178, 255],
               [83, 189, 159, 255],
               [50, 166, 150, 255],
               [50, 150, 180, 255],
               [5, 112, 176, 255],
               [5, 80, 140, 255],
               [10, 31, 150, 255],
               [44, 2, 70, 255],
               [106, 44, 90, 255]]
               # [168, 65, 91, 255]]
    rgb = []
    for i in range(len(colvals)):
        z = [x / 255 for x in colvals[i]]
        rgb.append(z)

    cmap = LinearSegmentedColormap.from_list('', rgb, numcolors)

    return cmap

# -------------------------------------------------------------------------------
# read data
sims = ['CTRL04', 'TENV04']

# --- edit here
ctrlpath = "/scratch/snx3000/rxiang/data/cosmo/EAS04_ctrl"
topo2path = "/scratch/snx3000/rxiang/data/cosmo/EAS04_topo2"
paths = [ctrlpath, topo2path]
data = {}
vars = ['SNOW_CON', 'SNOW_GSP']

[pole_lat04, pole_lon04, lat04, lon04, rlat04, rlon04, rot_pole_crs04] = pole04()

lb = [['a', 'b', 'c'], ['d', 'e', 'f']]

for i in range(len(sims)):
    sim = sims[i]
    path = paths[i]
    data[sim] = {}
    for j in range(len(vars)):
        var = vars[j]
        f = xr.open_dataset(f'{path}/1h/{var}/01-05_{var}_ts.nc')
        ds = f[var].values[0, :, :]
        data[sim][var] = ds
    data[sim]['SNOW'] = data[sim]['SNOW_CON'] + data[sim]['SNOW_GSP']

    f = xr.open_dataset(f'{path}/24h/W_SNOW/01-05_W_SNOW_mt_glc.nc')
    ds = f['W_SNOW'].values[0, :, :]
    data[sim]['GLACIAL'] = ds

vars = ['SNOW', 'GLACIAL']
data['diff'] = {}
for j in range(len(vars)):
    var = vars[j]
    data['diff'][var] = data['TENV04'][var] - data['CTRL04'][var]

# load topo
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_BECCY_4.4km_merit_unmod_topo.nc')
hsurf_ctrl = ds['HSURF'].values[:, :]
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_BECCY_4.4km_merit_env_topo_adj.nc')
hsurf_topo2 = ds['HSURF'].values[:, :]
hsurf_diff = ndimage.gaussian_filter(hsurf_topo2 - hsurf_ctrl, sigma=9, order=0)
hsurf_ctrl = ndimage.gaussian_filter(hsurf_ctrl, sigma=3, order=0)
lat_ = ds["lat"].values
lon_ = ds["lon"].values
ds.close()

# %%
ar = 1.0  # initial aspect ratio for first trial
wi = 8.5  # height in inches #15
hi = 4.5  # width in inches #10
ncol = 3  # edit here
nrow = 2
axs, cs, ct, topo, q, qk, topo2 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), np.empty(
    shape=(nrow, ncol), dtype='object')

fig = plt.figure(figsize=(wi, hi))
gs1 = gridspec.GridSpec(2, 2, left=0.06, bottom=0.024, right=0.545,
                        top=0.97, hspace=0.07, wspace=0.07, width_ratios=[1, 1], height_ratios=[1, 1])
gs2 = gridspec.GridSpec(2, 1, left=0.655, bottom=0.024, right=0.89,
                        top=0.97, hspace=0.07, wspace=0.07, height_ratios=[1, 1])

level1 = MaxNLocator(nbins=20).tick_values(0, 300)
cmap1 = prcp(20)
norm1 = BoundaryNorm(level1, ncolors=cmap1.N, clip=True)
tick1 = np.linspace(0, 300, 7, endpoint=True)

level2 = MaxNLocator(nbins=2).tick_values(0, 1)
cmap2 = cmc.davos_r
norm2 = BoundaryNorm(level2, ncolors=cmap2.N, clip=True)
tick2 = np.linspace(0, 1, 2, endpoint=True)

cmaps1 = [cmap1, cmap2]
norms1 = [norm1, norm2]
levels1 = [level1, level2]
ticks1 = [tick1, tick2]

level1 = MaxNLocator(nbins=20).tick_values(-100, 100)
cmap1 = drywet(21, cmc.vik_r)
norm1 = matplotlib.colors.Normalize(vmin=-100, vmax=100)
tick1 = np.linspace(-100, 100, 5, endpoint=True)

level2 = MaxNLocator(nbins=3).tick_values(-1, 1)
cmap2 = drywet(21, cmc.vik_r)
norm2 = matplotlib.colors.Normalize(vmin=-1, vmax=1)
tick2 = np.linspace(-1, 1, 3, endpoint=True)

cmaps2 = [cmap1, cmap2]
norms2 = [norm1, norm2]
levels2 = [level1, level2]
ticks2 = [tick1, tick2]

for i in range(len(vars)):
    var = vars[i]
    cmap, norm, level = cmaps1[i], norms1[i], levels1[i]
    for j in range(2):
        sim = sims[j]
        axs[i, j] = fig.add_subplot(gs1[i, j], projection=rot_pole_crs04)
        axs[i, j] = plotcosmo04_notick(axs[i, j])
        cs[i, j] = axs[i, j].pcolormesh(rlon04, rlat04, data[sim][var],
                                        cmap=cmap, norm=norm, shading="auto", transform=rot_pole_crs04)
    cmap, norm, level = cmaps2[i], norms2[i], levels2[i]
    for j in range(2):
        axs[i, 2] = fig.add_subplot(gs2[i, 0], projection=rot_pole_crs04)
        axs[i, 2] = plotcosmo04_notick(axs[i, 2])
        cs[i, 2] = axs[i, 2].pcolormesh(rlon04, rlat04, data['diff'][var],
                                        cmap=cmap, norm=norm, shading="auto", transform=rot_pole_crs04)
        topo[i, 2] = axs[i, 2].contour(lon_, lat_, hsurf_diff, levels=[100], colors='darkgreen', linewidths=1,
                                       transform=ccrs.PlateCarree())

for i in range(nrow):
    axs[i, 0].text(-0.01, 0.83, '35°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.01, 0.57, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.01, 0.31, '25°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.01, 0.05, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)

for j in range(ncol):
    axs[0, j].text(0.06, -0.02, '90°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=13)
    axs[0, j].text(0.46, -0.02, '100°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=13)
    axs[0, j].text(0.86, -0.02, '110°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=13)

for i in range(nrow):
    for j in range(ncol):
        label = lb[i][j]
        t = axs[i, j].text(0.01, 0.987, f'({label})', ha='left', va='top',
                           transform=axs[i, j].transAxes, fontsize=13)
        t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

titles = ['CTRL04', 'TENV04', 'TENV04-CTRL04']
for j in range(ncol):
    title = titles[j]
    axs[0, j].set_title(f'{title}', pad=5, fontsize=14, loc='center')

xlabels = ['mm', '']
for i in range(nrow):
    tick = ticks1[i]
    xlabel = xlabels[i]
    cax = fig.add_axes(
        [axs[i, 1].get_position().x1 + 0.01, axs[i, 1].get_position().y0, 0.015, axs[i, 1].get_position().height])
    cbar = fig.colorbar(cs[i, 1], cax=cax, orientation='vertical', extend='max', ticks=tick)
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.minorticks_off()
    axs[i, 1].text(1.35, 0.5, f'{xlabel}', ha='left', va='center', transform=axs[i, 1].transAxes, fontsize=13, rotation="vertical")
    # cbar.ax.set_xlabel(f'{xlabel}', fontsize=13)

xlabels = ['mm', '']
for i in range(nrow):
    tick = ticks2[i]
    xlabel = xlabels[i]
    cax = fig.add_axes(
        [axs[i, 2].get_position().x1 + 0.01, axs[i, 2].get_position().y0, 0.015, axs[i, 2].get_position().height])
    cbar = fig.colorbar(cs[i, 2], cax=cax, orientation='vertical', extend='both', ticks=tick)
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.minorticks_off()
    axs[i, 2].text(1.35, 0.5, f'{xlabel}', ha='left', va='center', transform=axs[i, 2].transAxes, fontsize=13, rotation="vertical")

plt.show()

plotpath = "/project/pr133/rxiang/figure/paper1/results/TENV/"
fig.savefig(plotpath + 'snow_big.png', dpi=500)

