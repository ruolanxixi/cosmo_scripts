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
import matplotlib.ticker as mticker
from matplotlib import lines

# matplotlib.style.use("classic")
font = {'size': 13}
# matplotlib.rc('font', **font)
# matplotlib.rcParams['font.family'] = 'dejavusans'
# matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

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
sims = ['PD', 'LGM']
seasons = "JJA"

# --- edit here
ctrlpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/indices"
lgmpath = "/project/pr133/rxiang/data/cosmo/EAS11_lgm/indices"
paths = [ctrlpath, lgmpath]
data = {}
vars1 = ['mean', 'perc_95.00', 'perc_99.00']
vars = ['IVT', 'mean', 'perc_95.00', 'perc_99.00']

[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

lb = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i'], ['j', 'k', 'l'], ['m', 'n', 'o']]

for i in range(len(sims)):
    sim = sims[i]
    path = paths[i]
    data[sim] = {}
    f = xr.open_dataset(f'{path}/day/2001-2005_smr_all_day_perc.nc')
    for j in range(len(vars1)):
        var = vars1[j]
        ds = f[var].values[:, :]
        data[sim][var] = ds

ctrlpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/monsoon"
topo1path = "/project/pr133/rxiang/data/cosmo/EAS11_lgm/monsoon"
paths = [ctrlpath, topo1path]
vars3 = ['IVQ', 'IUQ']

for i in range(len(sims)):
    sim = sims[i]
    path = paths[i]
    for j in range(len(vars3)):
        var = vars3[j]
        file = xr.open_dataset(f'{path}/IVT/01-05.IVT.smr.yearmean.nc')
        ds = file[var].values[:, :, :]
        data[sim][var] = np.nanmean(ds, axis=0)
    data[sim]['IVT'] = np.nanmean(np.sqrt(file['IVQ'].values[:, :, :]**2+file['IUQ'].values[:, :, :]**2), axis=0)


vars4 = ['mean', 'IVT', 'perc_95.00', 'perc_99.00', 'IVQ', 'IUQ']
np.seterr(divide='ignore', invalid='ignore')
data['diff'] = {}
for j in range(len(vars4)):
    var = vars4[j]
    if var in ["IVQ", "IUQ", "IVT"]:
        data['diff'][var] = data['LGM'][var] - data['PD'][var]
    else:
        data['diff'][var] = (data['LGM'][var] - data['PD'][var]) / data['PD'][var] * 100

vars = ['IVT', 'mean', 'perc_95.00']

# %%
ar = 1.0  # initial aspect ratio for first trial
wi = 8.5  # height in inches #15
hi = 6  # width in inches #10
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
gs1 = gridspec.GridSpec(3, 2, left=0.06, bottom=0.024, right=0.545,
                        top=0.97, hspace=0.07, wspace=0.07, width_ratios=[1, 1], height_ratios=[1, 1, 1])
gs2 = gridspec.GridSpec(3, 1, left=0.655, bottom=0.024, right=0.89,
                        top=0.97, hspace=0.07, wspace=0.07, height_ratios=[1, 1, 1])

level1 = MaxNLocator(nbins=20).tick_values(0, 20)
cmap1 = prcp(20)
norm1 = BoundaryNorm(level1, ncolors=cmap1.N, clip=True)
tick1 = np.linspace(0, 20, 5, endpoint=True)

level2 = MaxNLocator(nbins=30).tick_values(0, 300)
cmap2 = cmc.davos_r
norm2 = BoundaryNorm(level2, ncolors=cmap2.N, clip=True)
tick2 = np.linspace(0, 300, 7, endpoint=True)

level3 = MaxNLocator(nbins=20).tick_values(0, 100)
cmap3 = prcp(20)
norm3 = BoundaryNorm(level3, ncolors=cmap3.N, clip=True)
tick3 = np.linspace(0, 100, 6, endpoint=True)

level4 = MaxNLocator(nbins=20).tick_values(0, 200)
cmap4 = prcp(20)
norm4 = BoundaryNorm(level4, ncolors=cmap4.N, clip=True)
tick4 = np.linspace(0, 200, 5, endpoint=True)

cmaps1 = [cmap2, cmap1, cmap3, cmap4]
norms1 = [norm2, norm1, norm3, norm4]
levels1 = [level2, level1, level3, level4]
ticks1 = [tick2, tick1, tick3, tick4]

level1 = MaxNLocator(nbins=20).tick_values(-60, 60)
cmap1 = drywet(21, cmc.vik_r)
norm1 = matplotlib.colors.Normalize(vmin=-60, vmax=60)
tick1 = np.linspace(-60, 60, 5, endpoint=True)

level2 = MaxNLocator(nbins=20).tick_values(-120, 120)
cmap2 = drywet(21, cmc.vik_r)
norm2 = matplotlib.colors.Normalize(vmin=-120, vmax=120)
tick2 = np.linspace(-120, 120, 5, endpoint=True)

level3 = MaxNLocator(nbins=20).tick_values(-60, 60)
cmap3 = drywet(21, cmc.vik_r)
norm3 = matplotlib.colors.Normalize(vmin=-60, vmax=60)
tick3 = np.linspace(-60, 60, 5, endpoint=True)

level4 = MaxNLocator(nbins=20).tick_values(-60, 60)
cmap4 = drywet(21, cmc.vik_r)
norm4 = matplotlib.colors.Normalize(vmin=-60, vmax=60)
tick4 = np.linspace(-60, 60, 5, endpoint=True)

cmaps2 = [cmap2, cmap1, cmap3, cmap4]
norms2 = [norm2, norm1, norm3, norm4]
levels2 = [level2, level1, level3, level4]
ticks2 = [tick2, tick1, tick3, tick4]


for i in range(len(vars)):
    var = vars[i]
    cmap, norm, level = cmaps1[i], norms1[i], levels1[i]
    for j in range(2):
        sim = sims[j]
        axs[i, j] = fig.add_subplot(gs1[i, j], projection=rot_pole_crs)
        axs[i, j] = plotcosmo04_notick(axs[i, j])
        cs[i, j] = axs[i, j].pcolormesh(rlon, rlat, data[sim][var],
                                        cmap=cmap, norm=norm, shading="auto", transform=rot_pole_crs)
    cmap, norm, level = cmaps2[i], norms2[i], levels2[i]
    for j in range(2):
        axs[i, 2] = fig.add_subplot(gs2[i, 0], projection=rot_pole_crs)
        axs[i, 2] = plotcosmo04_notick(axs[i, 2])
        cs[i, 2] = axs[i, 2].pcolormesh(rlon, rlat, data['diff'][var],
                                        cmap=cmap, norm=norm, shading="auto", transform=rot_pole_crs)

for j in range(2):
    sim = sims[j]
    q[0, j] = axs[0, j].quiver(rlon[::10], rlat[::10], data[sim]['IUQ'][::10, ::10],
                            data[sim]['IVQ'][::10, ::10], color='black', scale=2000, headaxislength=3.5,
                            headwidth=5, minshaft=0)

q[0, 2] = axs[0, 2].quiver(rlon[::10], rlat[::10], data['diff']['IUQ'][::10, ::10],
                        data['diff']['IVQ'][::10, ::10], color='black', scale=1000,
                        headaxislength=3.5, headwidth=5, minshaft=0)

for i in range(nrow):
    axs[i, 0].text(-0.01, 0.83, '35°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.01, 0.57, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.01, 0.31, '25°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.01, 0.05, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)

for j in range(ncol):
    axs[2, j].text(0.06, -0.02, '90°E', ha='center', va='top', transform=axs[2, j].transAxes, fontsize=13)
    axs[2, j].text(0.46, -0.02, '100°E', ha='center', va='top', transform=axs[2, j].transAxes, fontsize=13)
    axs[2, j].text(0.86, -0.02, '110°E', ha='center', va='top', transform=axs[2, j].transAxes, fontsize=13)

for i in range(nrow):
    for j in range(ncol):
        label = lb[i][j]
        t = axs[i, j].text(0.01, 0.987, f'({label})', ha='left', va='top',
                           transform=axs[i, j].transAxes, fontsize=13)
        t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

titles = ['PD', 'LGM', 'PD-LGM']
for j in range(ncol):
    title = titles[j]
    axs[0, j].set_title(f'{title}', pad=5, fontsize=14, loc='center')

extends = ['max', 'neither', 'neither', 'neither']
xlabels = ['kg m$^{-1}$ s$^{-1}$', 'mm d$^{-1}$', 'mm d$^{-1}$', 'mm d$^{-1}$']
for i in range(nrow):
    extend = extends[i]
    tick = ticks1[i]
    xlabel = xlabels[i]
    cax = fig.add_axes(
        [axs[i, 1].get_position().x1 + 0.01, axs[i, 1].get_position().y0, 0.015, axs[i, 1].get_position().height])
    cbar = fig.colorbar(cs[i, 1], cax=cax, orientation='vertical', extend='max', ticks=tick)
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.minorticks_off()
    axs[i, 1].text(1.35, 0.5, f'{xlabel}', ha='left', va='center', transform=axs[i, 1].transAxes, fontsize=13, rotation="vertical")
    # cbar.ax.set_xlabel(f'{xlabel}', fontsize=13)

xlabels = ['kg m$^{-1}$ s$^{-1}$', '%', '%', '%']
for i in range(nrow):
    extend = extends[i]
    tick = ticks2[i]
    xlabel = xlabels[i]
    cax = fig.add_axes(
        [axs[i, 2].get_position().x1 + 0.01, axs[i, 2].get_position().y0, 0.015, axs[i, 2].get_position().height])
    cbar = fig.colorbar(cs[i, 2], cax=cax, orientation='vertical', extend='both', ticks=tick)
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.minorticks_off()
    axs[i, 2].text(1.35, 0.5, f'{xlabel}', ha='left', va='center', transform=axs[i, 2].transAxes, fontsize=13, rotation="vertical")

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + 'extreme.png', dpi=500)

plt.close(fig)


