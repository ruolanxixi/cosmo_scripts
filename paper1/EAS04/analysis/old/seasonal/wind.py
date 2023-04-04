# -------------------------------------------------------------------------------
# modules
#
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo04, pole04, colorbar
import matplotlib
import cartopy.crs as ccrs
from numpy import inf
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap, drywet, custom_seq_cmap_
from pyproj import Transformer
import scipy.ndimage as ndimage

font = {'size': 14}
matplotlib.rc('font', **font)
# -------------------------------------------------------------------------------
# import data
#
seasons = ["DJF", "MAM", "JJA", "SON"]
sims = ['ctrl', 'topo1', 'diff']
mdvname = 'TOT_PREC'  # edit here
year = '2001-2005'
ctrlpath = "/project/pr133/rxiang/data/cosmo/EAS04_ctrl/szn/"
topo1path = "/project/pr133/rxiang/data/cosmo/EAS04_topo1/szn/"

# -------------------------------------------------------------------------------
# read model data
#
mdvname = 'TOT_PREC'
ctrldata = []
for seas in range(len(seasons)):
    season = seasons[seas]
    filename = f'{year}.{mdvname}.{season}.nc'
    data = xr.open_dataset(f'{ctrlpath}{mdvname}/{filename}')[mdvname].values[0, :, :]
    ctrldata.append(data)

topo1data = []
for seas in range(len(seasons)):
    season = seasons[seas]
    filename = f'{year}.{mdvname}.{season}.nc'
    data = xr.open_dataset(f'{topo1path}{mdvname}/{filename}')[mdvname].values[0, :, :]
    topo1data.append(data)

mdvname = 'U'
ctrludata = []
for seas in range(len(seasons)):
    season = seasons[seas]
    filename = f'{year}.{mdvname}.{season}.nc'
    data = xr.open_dataset(f'{ctrlpath}{mdvname}/{filename}')[mdvname].values[0, 8, :, :]
    ctrludata.append(data)

topo1udata = []
for seas in range(len(seasons)):
    season = seasons[seas]
    filename = f'{year}.{mdvname}.{season}.nc'
    data = xr.open_dataset(f'{topo1path}{mdvname}/{filename}')[mdvname].values[0, 8, :, :]
    topo1udata.append(data)

mdvname = 'V'
ctrlvdata = []
for seas in range(len(seasons)):
    season = seasons[seas]
    filename = f'{year}.{mdvname}.{season}.nc'
    data = xr.open_dataset(f'{ctrlpath}{mdvname}/{filename}')[mdvname].values[0, 8, :, :]
    ctrlvdata.append(data)

topo1vdata = []
for seas in range(len(seasons)):
    season = seasons[seas]
    filename = f'{year}.{mdvname}.{season}.nc'
    data = xr.open_dataset(f'{topo1path}{mdvname}/{filename}')[mdvname].values[0, 8, :, :]
    topo1vdata.append(data)
# -------------------------------------------------------------------------------
# plot
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole04()

ar = 1.0  # initial aspect ratio for first trial
wi = 12  # height in inches
hi = wi * ar  # width in inches
ncol = 3  # edit here
nrow = 4
axs, cs, q = np.empty(shape=(4, 3), dtype='object'), np.empty(shape=(4, 3), dtype='object'), np.empty(shape=(4, 3),
                                                                                                      dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.1, 0.09, 0.99, 0.96
gs = gridspec.GridSpec(4, 3, left=left, bottom=bottom, right=right, top=top, wspace=0.16, hspace=0.13)
for i in range(12):
    axs[i // 3, i % 3] = fig.add_subplot(gs[i], projection=rot_pole_crs)
    axs[i // 3, i % 3] = plotcosmo04(axs[i // 3, i % 3])
# -------------------------
# panel plot
levels = MaxNLocator(nbins=20).tick_values(0, 20)
cmap = cmc.davos_r
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
for i in range(nrow):
    cs[i % 4, i // 4] = axs[i % 4, i // 4].pcolormesh(rlon, rlat, ctrldata[i], cmap=cmap, norm=norm, shading="auto")
    q[i % 4, i // 4] = axs[i % 4, i // 4].quiver(rlon[::30], rlat[::30], ctrludata[i][::30, ::30],
                                                 ctrlvdata[i][::30, ::30], color='black', scale=50)

axs[0, 0].quiverkey(q[0, 0], 0.92, 1.14, 2, r'$2\ \frac{m}{s}$', labelpos='S', labelsep=0.08, transform=axs[0, 0].transAxes,
                    fontproperties={'size': 11})
for i in np.arange(nrow, 2 * nrow, 1):
    cs[i % 4, i // 4] = axs[i % 4, i // 4].pcolormesh(rlon, rlat, topo1data[i - 4], cmap=cmap, norm=norm,
                                                      shading="auto")
    q[i % 4, i // 4] = axs[i % 4, i // 4].quiver(rlon[::30], rlat[::30], topo1udata[i-4][::30, ::30],
                                                 topo1vdata[i-4][::30, ::30], color='black', scale=50)

levels = MaxNLocator(nbins=15).tick_values(-10, 10)
cmap = drywet(25, cmc.vik_r)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
for i in np.arange(2 * nrow, 3 * nrow, 1):
    cs[i % 4, i // 4] = axs[i % 4, i // 4].pcolormesh(rlon, rlat, topo1data[i - 8] - ctrldata[i - 8], cmap=cmap,
                                                      clim=(-10, 10), shading="auto")
    q[i % 4, i // 4] = axs[i % 4, i // 4].quiver(rlon[::30], rlat[::30], topo1udata[i - 8][::30, ::30] - ctrludata[i - 8][::30, ::30],
                                                 topo1vdata[i - 8][::30, ::30] - ctrlvdata[i - 8][::30, ::30], color='black', scale=50)

axs[0, 2].quiverkey(q[0, 2], 0.92, 1.14, 2, r'$2\ \frac{m}{s}$', labelpos='S', labelsep=0.08, transform=axs[0, 2].transAxes,
                    fontproperties={'size': 11})
# -------------------------
# add title
axs[0, 0].set_title("Control", pad=10, fontsize=14, fontweight='bold')
axs[0, 1].set_title("Reduced topography", pad=10, fontsize=14, fontweight='bold')
axs[0, 2].set_title("Difference", pad=10, fontsize=14, fontweight='bold')

# -------------------------
# add label
axs[0, 0].text(-0.28, 0.5, 'DJF', ha='right', va='center', rotation='vertical',
               transform=axs[0, 0].transAxes, fontsize=14, fontweight='bold')
axs[1, 0].text(-0.28, 0.5, 'MAM', ha='right', va='center', rotation='vertical',
               transform=axs[1, 0].transAxes, fontsize=14, fontweight='bold')
axs[2, 0].text(-0.28, 0.5, 'JJA', ha='right', va='center', rotation='vertical',
               transform=axs[2, 0].transAxes, fontsize=14, fontweight='bold')
axs[3, 0].text(-0.28, 0.5, 'SON', ha='right', va='center', rotation='vertical',
               transform=axs[3, 0].transAxes, fontsize=14, fontweight='bold')

# -------------------------
# adjust figure
xmin, xmax = axs[0, 0].get_xbound()
ymin, ymax = axs[0, 0].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol * 1 * 0.94
fig.set_figwidth(hi / y2x_ratio)

# -------------------------
# add colorbar
cax = fig.add_axes([axs[3, 0].get_position().x0, axs[3, 0].get_position().y0 - 0.045, axs[3, 1].get_position().x1 - axs[3, 0].get_position().x0, 0.015])
cbar = fig.colorbar(cs[3, 0], cax=cax, orientation='horizontal', extend='max')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('mm/day', fontsize=13, labelpad=-0.01)

cax = fig.add_axes([axs[3, 2].get_position().x0, axs[3, 2].get_position().y0 - 0.045, axs[3, 2].get_position().width, 0.015])
cbar = fig.colorbar(cs[3, 2], cax=cax, orientation='horizontal', extend='both', ticks=[-10, -5, 0, 5, 10])
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('mm/day', fontsize=13, labelpad=-0.01)

plt.show()
# -------------------------
# save figure
plotpath = "/project/pr133/rxiang/figure/EAS04/analysis/seasonal/topo1/"
fig.savefig(plotpath + 'pr+wind.png', dpi=300)
plt.close(fig)
