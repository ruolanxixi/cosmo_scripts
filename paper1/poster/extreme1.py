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
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_, custom_seq_cmap
import numpy.ma as ma
import matplotlib.patches as patches
import scipy.ndimage as ndimage

font = {'size': 13}
matplotlib.rc('font', **font)


# precipitation color map
def prcp(numcolors):
    colvals = [[254, 217, 118, 255], # [254, 178, 76, 255],
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
sims = ['CTRL04', 'TRED04']
seasons = "JJA"

# --- edit here
ctrlpath = "/project/pr133/rxiang/data/cosmo/EAS04_ctrl/indices"
topo1path = "/project/pr133/rxiang/data/cosmo/EAS04_topo1/indices"
paths = [ctrlpath, topo1path]
data = {}
vars1 = ['mean', 'perc_99.00']
vars2 = ['perc_99.90']
vars = ['mean', 'perc_99.00', 'perc_99.90', 'CAPE']

[pole_lat04, pole_lon04, lat04, lon04, rlat04, rlon04, rot_pole_crs04] = pole04()

for i in range(len(sims)):
    sim = sims[i]
    path = paths[i]
    data[sim] = {}
    f = xr.open_dataset(f'{path}/day/2001-2005_smr_all_day_perc.nc')
    for j in range(len(vars1)):
        var = vars1[j]
        data[sim][var] = {}
        ds = f[var].values[:, :]
        data[sim][var]["value"] = ds
    f = xr.open_dataset(f'{path}/hr/2001-2005_smr_all_day_perc.nc')
    for j in range(len(vars2)):
        var = vars2[j]
        data[sim][var] = {}
        ds = f[var].values[:, :]
        data[sim][var]["value"] = ds

ctrlpath = "/project/pr133/rxiang/data/cosmo/EAS04_ctrl/monsoon"
topo1path = "/project/pr133/rxiang/data/cosmo/EAS04_topo1/monsoon"
paths = [ctrlpath, topo1path]
for i in range(len(sims)):
    sim = sims[i]
    path = paths[i]
    data[sim]['CAPE'] = {}
    f = xr.open_dataset(f'{path}/CAPE_ML/smr/01-05.CAPE_ML.smr.nc')
    ds = f['CAPE_ML'].values[:, :]
    data[sim]['CAPE']["value"] = np.nanmean(ds, axis=0)

np.seterr(divide='ignore', invalid='ignore')
data['diff'] = {}
for j in range(len(vars)):
    var = vars[j]
    data['diff'][var] = {}
    data['diff'][var]["value"] = (data['TRED04'][var]["value"] - data['CTRL04'][var]["value"]) / data['CTRL04'][var]["value"] * 100

data['diff']['CAPE']["value"] = data['TRED04']['CAPE']["value"] - data['CTRL04']['CAPE']["value"]
# load topo
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_BECCY_4.4km_merit_unmod_topo.nc')
hsurf_ctrl = ds['HSURF'].values[:, :]
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_BECCY_4.4km_merit_reduced_topo_adj.nc')
hsurf_topo1 = ds['HSURF'].values[:, :]
hsurf_diff = ndimage.gaussian_filter(hsurf_ctrl - hsurf_topo1, sigma=5, order=0)
hsurf_ctrl = ndimage.gaussian_filter(hsurf_ctrl, sigma=3, order=0)
lat_ = ds["lat"].values
lon_ = ds["lon"].values
ds.close()


# %%
ar = 1.0  # initial aspect ratio for first trial
wi = 10  # height in inches #15
hi = 3  # width in inches #10
ncol = 4  # edit here
nrow = 1
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), np.empty(
    shape=(nrow, ncol), dtype='object')

fig = plt.figure(figsize=(wi, hi))
gs = gridspec.GridSpec(1, 4, left=0.06, bottom=0.242, right=0.99,
                        top=0.972, hspace=0.1, wspace=0.1, width_ratios=[1, 1, 1, 1])

level1 = MaxNLocator(nbins=20).tick_values(-60, 60)
cmap1 = drywet(21, cmc.vik_r)
norm1 = matplotlib.colors.Normalize(vmin=-60, vmax=60)
tick1 = np.linspace(-60, 60, 5, endpoint=True)

level2 = MaxNLocator(nbins=20).tick_values(-60, 60)
cmap2 = drywet(21, cmc.vik_r)
norm2 = matplotlib.colors.Normalize(vmin=-60, vmax=60)
tick2 = np.linspace(-60, 60, 5, endpoint=True)

level3 = MaxNLocator(nbins=20).tick_values(-60, 60)
cmap3 = drywet(21, cmc.vik_r)
norm3 = matplotlib.colors.Normalize(vmin=-60, vmax=60)
tick3 = np.linspace(-60, 60, 5, endpoint=True)

level4 = MaxNLocator(nbins=20).tick_values(-100, 600)
cmap4 = custom_div_cmap(21, cmc.vik)
norm4 = matplotlib.colors.TwoSlopeNorm(vmin=-100, vcenter=0., vmax=600)
tick4 = [-100, -50, 0, 300, 600]

cmaps = [cmap1, cmap2, cmap3, cmap4]
norms = [norm1, norm2, norm3, norm4]
levels = [level1, level2, level3, level4]
ticks = [tick1, tick2, tick3, tick4]

for i in range(len(vars)):
    var = vars[i]
    cmap, norm, level = cmaps[i], norms[i], levels[i]
    for j in range(2):
        axs[0, i] = fig.add_subplot(gs[0, i], projection=rot_pole_crs04)
        axs[0, i] = plotcosmo04_notick(axs[0, i])
        cs[0, i] = axs[0, i].pcolormesh(rlon04, rlat04, data['diff'][var]["value"],
                                        cmap=cmap, norm=norm, shading="auto", transform=rot_pole_crs04)
        topo[0, i] = axs[0, i].contour(lon_, lat_, hsurf_diff, levels=[500], colors='darkgreen', linewidths=1,
                                       transform=ccrs.PlateCarree())

for i in range(nrow):
    axs[i, 0].text(-0.01, 0.83, '35°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.57, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.31, '25°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.05, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)

for j in range(ncol):
    axs[0, j].text(0.04, -0.02, '90°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=14)
    axs[0, j].text(0.46, -0.02, '100°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=14)
    axs[0, j].text(0.88, -0.02, '110°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=14)

titles = ['(a) mean precipitation', '(b) p99D', '(c) p99.9H', '(d) CAPE']
for j in range(ncol):
    title = titles[j]
    axs[0, j].set_title(f'{title}', pad=5, fontsize=14, loc='left')

cax = fig.add_axes(
    [axs[0, 1].get_position().x0, axs[0, 1].get_position().y0-0.13, axs[0, 1].get_position().width, 0.04])
cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='horizontal', extend='both', ticks=tick1)
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('%', fontsize=13, labelpad=-0.1)

cax = fig.add_axes(
    [axs[0, 3].get_position().x0, axs[0, 3].get_position().y0-0.13, axs[0, 3].get_position().width, 0.04])
cbar = fig.colorbar(cs[0, 3], cax=cax, orientation='horizontal', extend='both', ticks=tick4)
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('J kg$^{-1}$', fontsize=13, labelpad=-0.1)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/results/"
fig.savefig(plotpath + 'results2.png', dpi=500, transparent=True)
plt.close(fig)


