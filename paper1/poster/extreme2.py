# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import pole04, colorbar, plotcosmo04sm_notick, pole
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
topo2path = "/project/pr133/rxiang/data/cosmo/EAS04_topo2/indices"
paths = [ctrlpath, topo2path]
data = {}
vars1 = ['mean', 'perc_99.00']
vars2 = ['perc_99.90']
vars = ['AEVAP_S', 'RUNOFF_S', 'RUNOFF_G', 'T_2M']

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
topo2path = "/project/pr133/rxiang/data/cosmo/EAS04_topo2/monsoon"
paths = [ctrlpath, topo2path]
for i in range(len(sims)):
    sim = sims[i]
    path = paths[i]
    for j in range(len(vars)):
        var = vars[j]
        data[sim][var] = {}
        f = xr.open_dataset(f'{path}/{var}/smr/01-05.{var}.smr.nc')
        ds = f[var].values[:, :]
        data[sim][var]["value"] = np.nanmean(ds, axis=0)
    data[sim]['RUNOFF'] = {}
    data[sim]['RUNOFF']["value"] = data[sim]['RUNOFF_S']["value"] + data[sim]['RUNOFF_G']["value"]

vars = ['mean', 'T_2M', 'AEVAP_S', 'RUNOFF']
np.seterr(divide='ignore', invalid='ignore')
data['diff'] = {}
for j in range(len(vars)):
    var = vars[j]
    data['diff'][var] = {}
    data['diff'][var]["value"] = data['TRED04'][var]["value"] - data['CTRL04'][var]["value"]

# data['diff']['mean']["value"] = (data['TRED04']['mean']["value"] - data['CTRL04']['mean']["value"]) / data['CTRL04']['mean']["value"] * 100
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
wi = 10  # height in inches #15
hi = 3.2  # width in inches #10
ncol = 4  # edit here
nrow = 1
axs, cs, ct, topo, q, qk, topo2 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), np.empty(
    shape=(nrow, ncol), dtype='object')

fig = plt.figure(figsize=(wi, hi))
gs = gridspec.GridSpec(1, 4, left=0.055, bottom=0.22, right=0.99,
                        top=0.968, hspace=0.1, wspace=0.1, width_ratios=[1, 1, 1, 1])

level1 = MaxNLocator(nbins=20).tick_values(-10, 10)
cmap1 = drywet(21, cmc.vik_r)
norm1 = matplotlib.colors.Normalize(vmin=-10, vmax=10)
tick1 = np.linspace(-10, 10, 5, endpoint=True)

level2 = MaxNLocator(nbins=20).tick_values(-5, 5)
cmap2 = custom_div_cmap(21, cmc.vik)
norm2 = matplotlib.colors.Normalize(vmin=-5, vmax=5)
tick2 = np.linspace(-4, 4, 5, endpoint=True)

level3 = MaxNLocator(nbins=20).tick_values(-0.1, 0.1)
cmap3 = drywet(21, cmc.vik_r)
norm3 = matplotlib.colors.Normalize(vmin=-0.1, vmax=0.1)
tick3 = [-0.1, -0.05, 0, 0.05, 0.1]

level4 = MaxNLocator(nbins=20).tick_values(-10, 10)
cmap4 = drywet(21, cmc.vik_r)
norm4 = matplotlib.colors.TwoSlopeNorm(vmin=-10, vcenter=0., vmax=10)
tick4 = np.linspace(-10, 10, 5, endpoint=True)

cmaps = [cmap1, cmap2, cmap3, cmap4]
norms = [norm1, norm2, norm3, norm4]
levels = [level1, level2, level3, level4]
ticks = [tick1, tick2, tick3, tick4]

for i in range(len(vars)):
    var = vars[i]
    cmap, norm, level = cmaps[i], norms[i], levels[i]
    for j in range(2):
        axs[0, i] = fig.add_subplot(gs[0, i], projection=rot_pole_crs04)
        axs[0, i] = plotcosmo04sm_notick(axs[0, i])
        cs[0, i] = axs[0, i].pcolormesh(rlon04, rlat04, data['diff'][var]["value"],
                                        cmap=cmap, norm=norm, shading="auto", transform=rot_pole_crs04)
        topo[0, i] = axs[0, i].contour(lon_, lat_, hsurf_diff, levels=[100], colors='darkgreen', linewidths=1,
                                       transform=ccrs.PlateCarree())

for i in range(nrow):
    axs[i, 0].text(-0.01, 0.91, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.45, '25°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)

for j in range(ncol):
    axs[0, j].text(0.04, -0.02, '95°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=14)
    axs[0, j].text(0.46, -0.02, '100°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=14)
    axs[0, j].text(0.88, -0.02, '105°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=14)

titles = ['(a) precipitation', '(b) 2m temperature', '(c) evaporation', '(d) runoff']
for j in range(ncol):
    title = titles[j]
    axs[0, j].set_title(f'{title}', pad=5, fontsize=14, loc='left')

xlabels = ['mm d$^{-1}$', '$^{o}$C', 'mm d$^{-1}$', 'mm d$^{-1}$']
for i in range(len(vars)):
    tick = ticks[i]
    label = xlabels[i]
    cax = fig.add_axes(
        [axs[0, i].get_position().x0, axs[0, i].get_position().y0-0.13, axs[0, i].get_position().width, 0.04])
    cbar = fig.colorbar(cs[0, i], cax=cax, orientation='horizontal', extend='both', ticks=tick)
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.set_xlabel(f'{label}', fontsize=13, labelpad=-0.1)
    if i == 2:
        cbar.ax.set_xticklabels([-0.1, -0.05, 0, 0.05, 0.1])

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/results/"
fig.savefig(plotpath + 'results3.png', dpi=500, transparent=True)
plt.close(fig)


