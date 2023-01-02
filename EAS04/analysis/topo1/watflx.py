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
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_, hotcold
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

lb = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']]

# -------------------------------------------------------------------------------
# read data
#
# -------------------------------------------------------------------------------
# read data
sims = ['ctrl', 'topo1']
seasons = "JJA"
vars = ['mean', 'VIMD', 'IUQ', 'IVQ']

# --- edit here
[pole_lat04, pole_lon04, lat04, lon04, rlat04, rlon04, rot_pole_crs04] = pole04()

data = {}

for i in range(len(sims)):
    sim = sims[i]
    data[sim] = {}
    path = f'/project/pr133/rxiang/data/cosmo/EAS04_{sim}/indices'
    f = xr.open_dataset(f'{path}/day/2001-2005_smr_all_day_perc.nc')
    data[sim]['mean'] = {}
    ds = f['mean'].values[...]
    data[sim]['mean']["value"] = ds
    path = f'/project/pr133/rxiang/data/cosmo/EAS04_{sim}/monsoon/IVT/smr'
    f = xr.open_dataset(f'{path}/01-05.IVT.smr.nc')
    data[sim]['VIMD'] = {}
    ds = ndimage.gaussian_filter(np.nanmean(f['VIMD'].values[...], axis=0)*86400, sigma=4, order=0)
    data[sim]['VIMD']["value"] = ds
    data[sim]['IUQ'] = {}
    data[sim]['IVQ'] = {}
    ds = np.nanmean(f['IUQ'].values[...], axis=0)
    data[sim]['IUQ']["value"] = ds
    ds = np.nanmean(f['IVQ'].values[...], axis=0)
    data[sim]['IVQ']["value"] = ds

data['diff'] = {}
for j in range(len(vars)):
    var = vars[j]
    data['diff'][var] = {}
    data['diff'][var]["value"] = data['topo1'][var]["value"] - data['ctrl'][var]["value"]

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

# %% -------------------------------------------------------------------------------
# plot
#
ar = 1.0  # initial aspect ratio for first trial
wi = 9.5  # height in inches #15
hi = 7.2  # width in inches #10
ncol = 3  # edit here
nrow = 3
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), np.empty(
    shape=(nrow, ncol), dtype='object')

fig = plt.figure(figsize=(wi, hi))
gs1 = gridspec.GridSpec(3, 2, left=0.06, bottom=0.024, right=0.58,
                        top=0.97, hspace=0.1, wspace=0.1, width_ratios=[1, 1], height_ratios=[1, 1, 1])
gs2 = gridspec.GridSpec(3, 1, left=0.66, bottom=0.024, right=0.91,
                        top=0.97, hspace=0.1, wspace=0.1, height_ratios=[1, 1, 1])

level1 = MaxNLocator(nbins=20).tick_values(0, 20)
cmap1 = prcp(20)
norm1 = BoundaryNorm(level1, ncolors=cmap1.N, clip=True)
tick1 = np.linspace(0, 20, 5, endpoint=True)

level2 = MaxNLocator(nbins=15).tick_values(-30, 30)
cmap2 = hotcold(15)
norm2 = matplotlib.colors.Normalize(vmin=-30, vmax=30)
tick2 = np.linspace(-24, 24, 5, endpoint=True)

level3 = MaxNLocator(nbins=15).tick_values(-5, 5)
cmap3 = hotcold(15)
norm3 = BoundaryNorm(level3, ncolors=cmap3.N, clip=True)
tick3 = np.linspace(-4, 4, 5, endpoint=True)

cmaps1 = [cmap1, cmap2, cmap3]
norms1 = [norm1, norm2, norm3]
levels1 = [level1, level2, level3]
ticks1 = [tick1, tick2, tick3]

level1 = MaxNLocator(nbins=20).tick_values(-10, 10)
cmap1 = drywet(21, cmc.vik_r)
norm1 = matplotlib.colors.Normalize(vmin=-10, vmax=10)
tick1 = np.linspace(-10, 10, 5, endpoint=True)

level2 = MaxNLocator(nbins=15).tick_values(-30, 30)
cmap2 = hotcold(15)
norm2 = matplotlib.colors.Normalize(vmin=-30, vmax=30)
tick2 = np.linspace(-24, 24, 5, endpoint=True)

level3 = MaxNLocator(nbins=20).tick_values(-5, 5)
cmap3 = hotcold(15)
norm3 = matplotlib.colors.Normalize(vmin=-5, vmax=5)
tick3 = np.linspace(-4, 4, 5, endpoint=True)

cmaps2 = [cmap1, cmap2, cmap3]
norms2 = [norm1, norm2, norm3]
levels2 = [level1, level2, level3]
ticks2 = [tick1, tick2, tick3]

vars = ['mean', 'VIMD', 'VIMD']
for i in range(len(vars)):
    var = vars[i]
    cmap, norm, level = cmaps1[i], norms1[i], levels1[i]
    for j in range(2):
        sim = sims[j]
        axs[i, j] = fig.add_subplot(gs1[i, j], projection=rot_pole_crs04)
        axs[i, j] = plotcosmo04_notick(axs[i, j])
        cs[i, j] = axs[i, j].pcolormesh(rlon04, rlat04, data[sim][var]["value"],
                                        cmap=cmap, norm=norm, shading="auto", transform=rot_pole_crs04)
    cmap, norm, level = cmaps2[i], norms2[i], levels2[i]

    axs[i, 2] = fig.add_subplot(gs2[i, 0], projection=rot_pole_crs04)
    axs[i, 2] = plotcosmo04_notick(axs[i, 2])
    cs[i, 2] = axs[i, 2].pcolormesh(rlon04, rlat04, data['diff'][var]["value"],
                                    cmap=cmap, norm=norm, shading="auto", transform=rot_pole_crs04)
    topo[i, 2] = axs[i, 2].contour(lon_, lat_, hsurf_diff, levels=[500], colors='darkgreen', linewidths=1,
                                   transform=ccrs.PlateCarree())

for j in range(2):
    sim = sims[j]
    q[1, j] = axs[1, j].quiver(rlon04[::20], rlat04[::20], data[sim]['IUQ']["value"][::20, ::20],
                               data[sim]['IVQ']["value"][::20, ::20], color='black', scale=3000, headaxislength=3.5, headwidth=5, minshaft=0)
q[1, 2] = axs[1, 2].quiver(rlon04[::20], rlat04[::20], data['diff']['IUQ']["value"][::20, ::20],
                               data['diff']['IVQ']["value"][::20, ::20], color='black', scale=1000, headaxislength=3.5, headwidth=5, minshaft=0)
qk[1, 1] = axs[1, 1].quiverkey(q[1, 1], 0.83, 1.06, 200, r'$200$', labelpos='E', transform=axs[1, 1].transAxes,
                      fontproperties={'size': 12})
qk[1, 2] = axs[1, 2].quiverkey(q[1, 2], 0.87, 1.06, 50, r'$50$', labelpos='E', transform=axs[1, 2].transAxes,
                      fontproperties={'size': 12})

for i in range(nrow):
    axs[i, 0].text(-0.01, 0.83, '35°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.57, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.31, '25°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.05, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)

for j in range(ncol):
    axs[2, j].text(0.04, -0.02, '90°E', ha='center', va='top', transform=axs[2, j].transAxes, fontsize=14)
    axs[2, j].text(0.46, -0.02, '100°E', ha='center', va='top', transform=axs[2, j].transAxes, fontsize=14)
    axs[2, j].text(0.86, -0.02, '110°E', ha='center', va='top', transform=axs[2, j].transAxes, fontsize=14)

for i in range(nrow):
    for j in range(ncol):
        label = lb[i][j]
        t = axs[i, j].text(0.01, 0.987, f'({label})', ha='left', va='top',
                           transform=axs[i, j].transAxes, fontsize=14)
        t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

titles = ['CTRL04', 'TRED04', 'TRED04-CTRL04']
for j in range(ncol):
    title = titles[j]
    axs[0, j].set_title(f'{title}', pad=5, fontsize=14, loc='center')

extends = ['max', 'neither', 'neither', 'neither']
for i in range(nrow):
    extend = extends[i]
    tick = ticks1[i]
    cax = fig.add_axes(
        [axs[i, 1].get_position().x1 + 0.01, axs[i, 1].get_position().y0, 0.015, axs[i, 1].get_position().height])
    cbar = fig.colorbar(cs[i, 1], cax=cax, orientation='vertical', extend='both', ticks=tick)
    cbar.ax.tick_params(labelsize=13)

for i in range(nrow):
    extend = extends[i]
    tick = ticks2[i]
    cax = fig.add_axes(
        [axs[i, 2].get_position().x1 + 0.01, axs[i, 2].get_position().y0, 0.015, axs[i, 2].get_position().height])
    cbar = fig.colorbar(cs[i, 2], cax=cax, orientation='vertical', extend='both', ticks=tick)
    cbar.ax.tick_params(labelsize=13)

plt.show()





