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
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_
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
vars1 = ['mean', 'perc_95.00', 'perc_99.00']
vars2 = ['perc_99.90']
vars = ['mean', 'perc_95.00', 'perc_99.00', 'perc_99.90']

[pole_lat04, pole_lon04, lat04, lon04, rlat04, rlon04, rot_pole_crs04] = pole04()

lb = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i'], ['j', 'k', 'l']]

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

np.seterr(divide='ignore', invalid='ignore')
data['diff'] = {}
for j in range(len(vars)):
    var = vars[j]
    data['diff'][var] = {}
    data['diff'][var]["value"] = (data['TRED04'][var]["value"] - data['CTRL04'][var]["value"]) / data['CTRL04'][var]["value"] * 100

ctrlpath = "/project/pr133/rxiang/data/cosmo/EAS04_ctrl/monsoon/IVT/smr"
topo1path = "/project/pr133/rxiang/data/cosmo/EAS04_topo1/monsoon/IVT/smr"
paths = [ctrlpath, topo1path]
vars = ['IUQ', 'IVQ']
for i in range(len(sims)):
    sim = sims[i]
    path = paths[i]
    data[sim] = {}
    f = xr.open_dataset(f'{path}/01-05.IVT.smr.nc')
    for j in range(len(vars)):
        var = vars[j]
        data[sim][var] = {}
        ds = np.nanmean(f[var].values[...], axis=0)
        data[sim][var]["value"] = ds

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

# models = ['EAS04', 'EAS11']
# for i in range(2):
#     model = models[i]
#     sim = sims[i]
#     path = f"/project/pr133/rxiang/data/cosmo/{model}_ctrl/remap/TOT_PREC/"
#     f = xr.open_dataset(f'{path}/2001-2005_JJA_all_day_perc.remap.imerg_full.nc')
#     fslice = f.sel(**{"lat": slice(22, 40),
#                       "lon": slice(89, 113)})
#     fim = xr.open_dataset(f'{imergpath}/2001-2005_JJA_all_day_perc.nc')
#     fimslice = fim.sel(**{"lat": slice(22, 40),
#                       "lon": slice(89, 113)})
#     for j in range(len(vars)):
#         var = vars[j]
#         ds = fslice[var].values[:, :]
#         dsim = fimslice[var].values[:, :]
#         data[sim][var]["R"] = ma.corrcoef(ma.masked_invalid(dsim.flatten()), ma.masked_invalid(ds.flatten()))[0, 1]
#         data[sim][var]["BIAS"] = np.nanmean(ds - dsim)

# %%
ar = 1.0  # initial aspect ratio for first trial
wi = 9.5  # height in inches #15
hi = 9.2  # width in inches #10
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
gs1 = gridspec.GridSpec(4, 2, left=0.06, bottom=0.024, right=0.58,
                        top=0.97, hspace=0.1, wspace=0.1, width_ratios=[1, 1], height_ratios=[1, 1, 1, 1])
gs2 = gridspec.GridSpec(4, 1, left=0.66, bottom=0.024, right=0.91,
                        top=0.97, hspace=0.1, wspace=0.1, height_ratios=[1, 1, 1, 1])

level1 = MaxNLocator(nbins=20).tick_values(0, 20)
cmap1 = prcp(20)
norm1 = BoundaryNorm(level1, ncolors=cmap1.N, clip=True)
tick1 = np.linspace(0, 20, 5, endpoint=True)

level2 = MaxNLocator(nbins=20).tick_values(0, 80)
cmap2 = prcp(20)
norm2 = BoundaryNorm(level2, ncolors=cmap2.N, clip=True)
tick2 = np.linspace(0, 80, 5, endpoint=True)

level3 = MaxNLocator(nbins=20).tick_values(0, 100)
cmap3 = prcp(20)
norm3 = BoundaryNorm(level3, ncolors=cmap3.N, clip=True)
tick3 = np.linspace(0, 100, 6, endpoint=True)

level4 = MaxNLocator(nbins=20).tick_values(0, 40)
cmap4 = prcp(20)
norm4 = BoundaryNorm(level4, ncolors=cmap4.N, clip=True)
tick4 = np.linspace(0, 40, 5, endpoint=True)

cmaps1 = [cmap1, cmap2, cmap3, cmap4]
norms1 = [norm1, norm2, norm3, norm4]
levels1 = [level1, level2, level3, level4]
ticks1 = [tick1, tick2, tick3, tick4]

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

level4 = MaxNLocator(nbins=20).tick_values(-60, 60)
cmap4 = drywet(21, cmc.vik_r)
norm4 = matplotlib.colors.Normalize(vmin=-60, vmax=60)
tick4 = np.linspace(-60, 60, 5, endpoint=True)

cmaps2 = [cmap1, cmap2, cmap3, cmap4]
norms2 = [norm1, norm2, norm3, norm4]
levels2 = [level1, level2, level3, level4]
ticks2 = [tick1, tick2, tick3, tick4]


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
    for j in range(2):
        axs[i, 2] = fig.add_subplot(gs2[i, 0], projection=rot_pole_crs04)
        axs[i, 2] = plotcosmo04_notick(axs[i, 2])
        cs[i, 2] = axs[i, 2].pcolormesh(rlon04, rlat04, data['diff'][var]["value"],
                                        cmap=cmap, norm=norm, shading="auto", transform=rot_pole_crs04)
        topo[i, 2] = axs[i, 2].contour(lon_, lat_, hsurf_diff, levels=[500], colors='darkgreen', linewidths=1,
                                       transform=ccrs.PlateCarree())

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
        t = axs[i, j].text(0.01, 0.987, f'({label})', ha='left', va='top',
                           transform=axs[i, j].transAxes, fontsize=14)
        t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

titles = ['CTRL04', 'TRED04', 'TRED04-CTRL04']
for j in range(ncol):
    title = titles[j]
    axs[0, j].set_title(f'{title}', pad=5, fontsize=14, loc='center')

# for i in range(len(vars)):
#     var = vars[i]
#     rect = patches.Rectangle((0.25, 0.75), 0.8, 0.24, linewidth=1, edgecolor='none', facecolor='white', alpha=0.5,
#                              transform=axs[i, 2].transAxes)
#     axs[i, 2].add_patch(rect)
#     t1 = axs[i, 2].text(0.78, 0.95, "BIAS", fontsize=9, horizontalalignment='center',
#                         verticalalignment='center', transform=axs[i, 2].transAxes)
#     t2 = axs[i, 2].text(0.92, 0.95, "R", fontsize=9, horizontalalignment='center',
#                         verticalalignment='center', transform=axs[i, 2].transAxes)
#     t3 = axs[i, 2].text(0.48, 0.87, f"CTRL04 - IMERG", fontsize=9, horizontalalignment='center',
#                         verticalalignment='center', transform=axs[i, 2].transAxes)
#     t4 = axs[i, 2].text(0.48, 0.79, f"CTRL11 - IMERG", fontsize=9, horizontalalignment='center',
#                         verticalalignment='center', transform=axs[i, 2].transAxes)
    # Create a Rectangle patch
    # Add the patch to the Axes
    # txt = data['CTRL04'][var]['BIAS']
    # t5 = axs[i, 2].text(0.78, 0.87, '%0.2f' % txt, fontsize=9, horizontalalignment='center',
    #                    verticalalignment='center', transform=axs[i, 2].transAxes)
    # txt = data['CTRL11'][var]['BIAS']
    # t6 = axs[i, 2].text(0.78, 0.79, '%0.2f' % txt, fontsize=9, horizontalalignment='center',
    #                    verticalalignment='center', transform=axs[i, 2].transAxes)
    # txt = data['CTRL04'][var]['R']
    # t7 = axs[i, 2].text(0.92, 0.87, '%0.2f' % txt, fontsize=9, horizontalalignment='center',
    #                    verticalalignment='center', transform=axs[i, 2].transAxes)
    # txt = data['CTRL11'][var]['R']
    # t8 = axs[i, 2].text(0.92, 0.79, '%0.2f' % txt, fontsize=9, horizontalalignment='center',
    #                    verticalalignment='center', transform=axs[i, 2].transAxes)

extends = ['max', 'neither', 'neither', 'neither']
for i in range(nrow):
    extend = extends[i]
    tick = ticks1[i]
    cax = fig.add_axes(
        [axs[i, 1].get_position().x1 + 0.01, axs[i, 1].get_position().y0, 0.015, axs[i, 1].get_position().height])
    cbar = fig.colorbar(cs[i, 1], cax=cax, orientation='vertical', extend='max', ticks=tick)
    cbar.ax.tick_params(labelsize=13)

for i in range(nrow):
    extend = extends[i]
    tick = ticks2[i]
    cax = fig.add_axes(
        [axs[i, 2].get_position().x1 + 0.01, axs[i, 2].get_position().y0, 0.015, axs[i, 2].get_position().height])
    cbar = fig.colorbar(cs[i, 2], cax=cax, orientation='vertical', extend='both', ticks=tick)
    cbar.ax.tick_params(labelsize=13)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/results/extreme/"
fig.savefig(plotpath + 'extreme1.png', dpi=500)
plt.close(fig)


