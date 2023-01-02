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
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from auxiliary import truncate_colormap, spat_agg_1d, spat_agg_2d

font = {'size': 13}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
sims = ['(a) CTRL', '(b) TRED', '(c) TENV']
path = '/users/rxiang/lmp/lib'
ctrl = xr.open_dataset(f'{path}/extpar_BECCY_4.4km_merit_unmod_topo.nc')['HSURF'].values[...]
topo1 = xr.open_dataset(f'{path}/extpar_BECCY_4.4km_merit_reduced_topo_adj.nc')['HSURF'].values[...]
topo2 = xr.open_dataset(f'{path}/extpar_BECCY_4.4km_merit_env_topo_adj.nc')['HSURF'].values[...]
ds = xr.open_dataset(f'{path}/extpar_BECCY_4.4km_merit_env_topo_adj.nc')
rlat = ds["rlat"].values
rlon = ds["rlon"].values
pole_lat = ds["rotated_pole"].grid_north_pole_latitude
pole_lon = ds["rotated_pole"].grid_north_pole_longitude
rot_pole_crs = ccrs.RotatedPole(pole_latitude=pole_lat, pole_longitude=pole_lon)

# %%
ar = 1.0  # initial aspect ratio for first trial
wi = 12  # height in inches #15
hi = 4  # width in inches #10
ncol = 3  # edit here
nrow = 1
axs, cs, ct, topo, q, qk= np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.05, 0.07, 0.90, 0.94
gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.1, hspace=0.01)

levels = np.arange(0., 6500.0, 500.0)
ticks = np.arange(0., 6500.0, 1000.0)
cmap = truncate_colormap(cmc.bukavu, 0.55, 1.0)
norm = BoundaryNorm(levels, ncolors=cmap.N, extend="max")

map_ext = [88, 114, 16, 40]

for i in range(ncol):
    sim = sims[i]
    axs[0, i] = fig.add_subplot(gs[i], projection=rot_pole_crs)
    axs[0, i].set_extent(map_ext, crs=ccrs.PlateCarree())
    axs[0, i].set_aspect("auto")
    gl = axs[0, i].gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([85, 90, 95, 100, 105, 110])
    axs[0, i].add_feature(cfeature.OCEAN, zorder=100, edgecolor='k')
    axs[0, i].add_feature(cfeature.COASTLINE)
    axs[0, i].add_feature(cfeature.BORDERS, linestyle=':')
    axs[0, i].add_feature(cfeature.RIVERS, alpha=0.5)


cs[0, 0] = axs[0, 0].pcolormesh(rlon, rlat, ctrl, cmap=cmap, norm=norm, shading="auto", transform=rot_pole_crs)
cs[0, 1] = axs[0, 1].pcolormesh(rlon, rlat, topo1, cmap=cmap, norm=norm, shading="auto", transform=rot_pole_crs)
cs[0, 2] = axs[0, 2].pcolormesh(rlon, rlat, topo2, cmap=cmap, norm=norm, shading="auto", transform=rot_pole_crs)

cax = fig.add_axes([axs[0, 2].get_position().x1+0.02, axs[0, 2].get_position().y0, 0.02, axs[0, 2].get_position().height])
cbar = fig.colorbar(cs[0, 2], cax=cax, orientation='vertical', extend='max', ticks=ticks)
cbar.ax.tick_params(labelsize=14)

for j in range(ncol):
    title = sims[j]
    axs[0, j].set_title(f'{title}', pad=5, fontsize=14, loc='left')

for i in range(nrow):
    axs[i, 0].text(-0.01, 0.86, '35°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.665, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.47, '25°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.275, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.08, '15°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)

for j in range(ncol):
    axs[0, j].text(0.04, -0.02, '90°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=14)
    axs[0, j].text(0.46, -0.02, '100°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=14)
    axs[0, j].text(0.86, -0.02, '110°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=14)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/topo/"
fig.savefig(plotpath + 'topo.png', dpi=500)
plt.close(fig)


