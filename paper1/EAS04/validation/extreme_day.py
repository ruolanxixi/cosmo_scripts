# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma
import matplotlib.patches as patches
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

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

def plotcosmo04_notick(ax):
    ax.set_extent([89, 112.5, 22.2, 39], crs=ccrs.PlateCarree())  # for extended 12km domain
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)

    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([90, 100, 110, 120])
    gl.ylocator = mticker.FixedLocator([20, 25, 30, 35, 40])

    return ax

def pole():
    file = "/ruolan/CTRL11/day/01_TOT_PREC.nc"
    ds = xr.open_dataset(f'{file}')
    pole_lat = ds["rotated_pole"].grid_north_pole_latitude
    pole_lon = ds["rotated_pole"].grid_north_pole_longitude
    lat = ds["lat"].values
    lon = ds["lon"].values
    rlat = ds["rlat"].values
    rlon = ds["rlon"].values
    rot_pole_crs = ccrs.RotatedPole(pole_latitude=pole_lat, pole_longitude=pole_lon)

    return pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs

def pole04():
    file = "/ruolan/CTRL04/day/01_TOT_PREC.nc"
    ds = xr.open_dataset(f'{file}')
    pole_lat = ds["rotated_pole"].grid_north_pole_latitude
    pole_lon = ds["rotated_pole"].grid_north_pole_longitude
    lat = ds["lat"].values
    lon = ds["lon"].values
    rlat = ds["rlat"].values
    rlon = ds["rlon"].values
    rot_pole_crs = ccrs.RotatedPole(pole_latitude=pole_lat, pole_longitude=pole_lon)

    return pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs
# -------------------------------------------------------------------------------
# read data
sims = ['obs', 'CTRL04', 'CTRL11']
seasons = "JJA"

# --- edit here
obspath = "/scratch/snx3000/rxiang/obs/indices/"
# ---
cpmpath = "/project/pr133/rxiang/data/cosmo/EAS04_ctrl/indices/day"
lsmpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/indices/day"
paths = [obspath, cpmpath, lsmpath]
data = {}
vars = ['mean', 'wet_day_freq', 'intensity', 'perc_97.50']

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

for i in range(2):
    sim = sims[i+1]
    path = f"/ruolan/{sim}/"
    # remap simulations to observation
    f = xr.open_dataset(f'{path}/2001-2005_JJA_all_day_perc.remap.nc')
    fslice = f.sel(**{"lat": slice(22, 40),
                      "lon": slice(89, 113)})
    fim = xr.open_dataset(f'{obspath}/2001-2005_JJA_all_day_perc.nc')
    fimslice = fim.sel(**{"lat": slice(22, 40),
                      "lon": slice(89, 113)})
    for j in range(len(vars)):
        var = vars[j]
        ds = fslice[var].values[:, :]
        dsim = fimslice[var].values[:, :]
        data[sim][var]["R"] = ma.corrcoef(ma.masked_invalid(dsim.flatten()), ma.masked_invalid(ds.flatten()))[0, 1]
        data[sim][var]["BIAS"] = np.nanmean(ds - dsim)

# --- edit here
f = xr.open_dataset(f'{obspath}/2001-2005_JJA_all_day_perc.nc')
data['obs']['lon'] = f['lon'].values[...]
data['obs']['lat'] = f['lat'].values[...]
data['obs']['proj'] = ccrs.PlateCarree()
# ---
data['CTRL04']['lon'] = rlon04
data['CTRL04']['lat'] = rlat04
data['CTRL04']['proj'] = rot_pole_crs04
data['CTRL11']['lon'] = rlon
data['CTRL11']['lat'] = rlat
data['CTRL11']['proj'] = rot_pole_crs

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
cmap1 = prcp(20)
norm1 = BoundaryNorm(level1, ncolors=cmap1.N, clip=True)
tick1 = np.linspace(0, 20, 5, endpoint=True)

level2 = MaxNLocator(nbins=20).tick_values(0, 1)
cmap2 = prcp(20)
norm2 = BoundaryNorm(level2, ncolors=cmap2.N, clip=True)
tick2 = np.linspace(0, 1, 6, endpoint=True)

level3 = MaxNLocator(nbins=30).tick_values(0, 30)
cmap3 = prcp(30)
norm3 = BoundaryNorm(level3, ncolors=cmap3.N, clip=True)
tick3 = np.linspace(0, 30, 7, endpoint=True)

level4 = MaxNLocator(nbins=30).tick_values(0, 150)
cmap4 = prcp(30)
norm4 = BoundaryNorm(level4, ncolors=cmap4.N, clip=True)
tick4 = np.linspace(0, 150, 6, endpoint=True)

cmaps = [cmap1, cmap2, cmap3, cmap4]
norms = [norm1, norm2, norm3, norm4]
levels = [level1, level2, level3, level4]
ticks = [tick1, tick2, tick3, tick4]

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
    t3 = axs[i, 2].text(0.51, 0.87, f"CTRL04 - obs", fontsize=9, horizontalalignment='center',
                        verticalalignment='center', transform=axs[i, 2].transAxes)
    t4 = axs[i, 2].text(0.51, 0.79, f"CTRL11 - obs", fontsize=9, horizontalalignment='center',
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

extends = ['max', 'neither', 'max', 'max']
for i in range(nrow):
    ex = extends[i]
    tick = ticks[i]
    cax = fig.add_axes(
        [axs[i, 2].get_position().x1 + 0.01, axs[i, 2].get_position().y0, 0.015, axs[i, 2].get_position().height])
    cbar = fig.colorbar(cs[i, 2], cax=cax, orientation='vertical', extend=ex, ticks=tick)
    cbar.ax.tick_params(labelsize=13)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/validation/CPM/"
fig.savefig(plotpath + 'extreme1.png', dpi=500)
plt.close(fig)

