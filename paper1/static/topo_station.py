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
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib import lines

font = {'size': 13}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
path = '/users/rxiang/lmp/lib'
ctrl11 = xr.open_dataset(f'{path}/extpar_EAS_ext_12km_merit_unmod_topo.nc')['HSURF'].values[...]
ctrl04 = xr.open_dataset(f'{path}/extpar_BECCY_4.4km_merit_unmod_topo.nc')['HSURF'].values[...]

path = f'/project/pr133/rxiang/data/cosmo/EAS11_ctrl/monsoon/TWATFLXU/smr'
data = xr.open_dataset(f'{path}' + '/' + '01-05.TWATFLXU.smr.cpm.nc')
smr = data['TWATFLXU'].values[:, :, :]
u11 = np.nanmean(smr, axis=0)
path = f'/project/pr133/rxiang/data/cosmo/EAS11_ctrl/monsoon/TWATFLXV/smr'
data = xr.open_dataset(f'{path}' + '/' + '01-05.TWATFLXV.smr.cpm.nc')
smr = data['TWATFLXV'].values[:, :, :]
v11 = np.nanmean(smr, axis=0)

path = f'/project/pr133/rxiang/data/cosmo/EAS04_ctrl/monsoon/TWATFLXU/smr'
data = xr.open_dataset(f'{path}' + '/' + '01-05.TWATFLXU.smr.cpm.nc')
smr = data['TWATFLXU'].values[:, :, :]
u04 = np.nanmean(smr, axis=0)
path = f'/project/pr133/rxiang/data/cosmo/EAS04_ctrl/monsoon/TWATFLXV/smr'
data = xr.open_dataset(f'{path}' + '/' + '01-05.TWATFLXV.smr.cpm.nc')
smr = data['TWATFLXV'].values[:, :, :]
v04 = np.nanmean(smr, axis=0)

file = '/users/rxiang/Stations_info.txt'
df = pd.read_csv(f"{file}", sep="  ", header=None, names=["lat", "lon"])/100


# %%
path = '/users/rxiang/lmp/lib'
ds = xr.open_dataset(f'{path}/extpar_BECCY_4.4km_merit_env_topo_adj.nc')
rlat04 = ds["rlat"].values
rlon04 = ds["rlon"].values
ds = xr.open_dataset(f'{path}/extpar_EAS_ext_12km_merit_env_topo_adj.nc')
rlat11 = ds["rlat"].values
rlon11 = ds["rlon"].values
pole_lat = ds["rotated_pole"].grid_north_pole_latitude
pole_lon = ds["rotated_pole"].grid_north_pole_longitude
rot_pole_crs = ccrs.RotatedPole(pole_latitude=pole_lat, pole_longitude=pole_lon)

ar = 1.0  # initial aspect ratio for first trial
wi = 9.5  # height in inches #15
hi = 3.3  # width in inches #10
ncol = 2  # edit here
nrow = 1
axs, cs, ct, topo, q, qk= np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.06, 0.04, 0.89, 0.95
gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.155, hspace=0.2, width_ratios=[2, 1.084])

levels = np.arange(0., 6500.0, 500.0)
ticks = np.arange(0., 6500.0, 1000.0)
cmap = truncate_colormap(cmc.bukavu, 0.55, 1.0)
norm = BoundaryNorm(levels, ncolors=cmap.N, extend="max")

map_ext1 = [65, 173, 7, 61]
map_ext2 = [88, 114, 16, 40]
map_ext = [map_ext1, map_ext2]

for i in range(ncol):
    ext = map_ext[i]
    axs[0, i] = fig.add_subplot(gs[i], projection=rot_pole_crs)
    axs[0, i].set_extent(ext, crs=ccrs.PlateCarree())
    # axs[0, i].set_aspect("auto")

    axs[0, i].add_feature(cfeature.OCEAN, zorder=100)
    axs[0, i].add_feature(cfeature.COASTLINE, linewidth=2)
    axs[0, i].add_feature(cfeature.BORDERS, linestyle=':')
    axs[0, i].add_feature(cfeature.RIVERS, alpha=0.5)

gl = axs[0, 0].gridlines(draw_labels=False, linewidth=1,
                             color='grey', alpha=0.5, linestyle='--', zorder=101)
gl.xlocator = mticker.FixedLocator([60, 80, 100, 120, 140, 160, 180])
gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])
gl = axs[0, 1].gridlines(draw_labels=False, linewidth=1,
                             color='grey', alpha=0.5, linestyle='--', zorder=101)
gl.xlocator = mticker.FixedLocator([85, 90, 95, 100, 105, 110])
gl.ylocator = mticker.FixedLocator([15, 20, 25, 30, 35, 40])

file = "/users/rxiang/miniconda3/envs/ncl_stable/lib/ncarg/colormaps/OceanLakeLandSnow.rgb"
# Source of NCL rgb-file:
# https://www.ncl.ucar.edu/Document/Graphics/color_table_gallery.shtml
rgb = np.loadtxt(file, comments=("#", "ncolors"))
if rgb.max() > 1.0:
    rgb /= 255.0
print("Number of colors: " + str(rgb.shape[0]))

cmap_ncl = matplotlib.colors.LinearSegmentedColormap.from_list("OceanLakeLandSnow",
                                                        rgb, N=rgb.shape[0])

cmap_ter, min_v, max_v = cmap_ncl, 0.05, 1.00  # NCL
cmap_ter = matplotlib.colors.LinearSegmentedColormap.from_list(
    "trunc({n},{a:.2f},{b:.2f})".format(
        n=cmap_ter.name, a=min_v, b=max_v),
    cmap_ter(np.linspace(min_v, max_v, 100)))

cs[0, 0] = axs[0, 0].pcolormesh(rlon11, rlat11, ctrl11, cmap=cmap_ter, norm=norm, shading="auto", transform=rot_pole_crs)
cs[0, 1] = axs[0, 1].pcolormesh(rlon04, rlat04, ctrl04, cmap=cmap_ter, norm=norm, shading="auto", transform=rot_pole_crs)

# [pole_lat, pole_lon, lat, lon, rlat11, rlon11, rot_pole_crs] = pole()
# [pole_lat, pole_lon, lat, lon, rlat04, rlon04, rot_pole_crs] = pole04()
# q[0, 0] = axs[0, 0].quiver(rlon11[::40], rlat11[::40], u11[::40, ::40], v11[::40, ::40], color='black', scale=5000, zorder=102, headaxislength=3.5, headwidth=5, minshaft=0)
# q[0, 1] = axs[0, 1].quiver(rlon04[::40], rlat04[::40], u04[::40, ::40], v04[::40, ::40], color='black', scale=5000, zorder=102, headaxislength=3.5, headwidth=5, minshaft=0)
# qk[0, 1] = axs[0, 1].quiverkey(q[0, 1], 0.45, 1.04, 400, r'$400\ kg\ m^{-1}\ s^{-1}$', labelpos='E', transform=axs[0, 1].transAxes,
#                       fontproperties={'size': 12})

cax = fig.add_axes([axs[0, 1].get_position().x1+0.02, axs[0, 1].get_position().y0, 0.022, axs[0, 1].get_position().height])
cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical', extend='max', ticks=ticks)
cbar.ax.tick_params(labelsize=14)

axs[0, 1].text(-0.01, 0.86, '35°N', ha='right', va='center', transform=axs[0, 1].transAxes, fontsize=14)
axs[0, 1].text(-0.01, 0.665, '30°N', ha='right', va='center', transform=axs[0, 1].transAxes, fontsize=14)
axs[0, 1].text(-0.01, 0.47, '25°N', ha='right', va='center', transform=axs[0, 1].transAxes, fontsize=14)
axs[0, 1].text(-0.01, 0.275, '20°N', ha='right', va='center', transform=axs[0, 1].transAxes, fontsize=14)
axs[0, 1].text(-0.01, 0.08, '15°N', ha='right', va='center', transform=axs[0, 1].transAxes, fontsize=14)

axs[0, 1].text(0.04, -0.02, '90°E', ha='center', va='top', transform=axs[0, 1].transAxes, fontsize=14)
axs[0, 1].text(0.46, -0.02, '100°E', ha='center', va='top', transform=axs[0, 1].transAxes, fontsize=14)
axs[0, 1].text(0.86, -0.02, '110°E', ha='center', va='top', transform=axs[0, 1].transAxes, fontsize=14)

axs[0, 0].text(-0.008, 0.95, '50°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=14)
axs[0, 0].text(-0.008, 0.77, '40°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=14)
axs[0, 0].text(-0.008, 0.59, '30°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=14)
axs[0, 0].text(-0.008, 0.41, '20°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=14)
axs[0, 0].text(-0.008, 0.23, '10°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=14)
axs[0, 0].text(-0.008, 0.05, '0°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=14)

axs[0, 0].text(0.12, -0.02, '80°E', ha='center', va='top', transform=axs[0, 0].transAxes, fontsize=14)
axs[0, 0].text(0.32, -0.02, '100°E', ha='center', va='top', transform=axs[0, 0].transAxes, fontsize=14)
axs[0, 0].text(0.52, -0.02, '120°E', ha='center', va='top', transform=axs[0, 0].transAxes, fontsize=14)
axs[0, 0].text(0.72, -0.02, '140°E', ha='center', va='top', transform=axs[0, 0].transAxes, fontsize=14)
axs[0, 0].text(0.92, -0.02, '160°E', ha='center', va='top', transform=axs[0, 0].transAxes, fontsize=14)

axs[0, 0].text(0, 1.01, '(a) LSM: $\Delta$x = 12 km', ha='left', va='bottom', transform=axs[0, 0].transAxes, fontsize=14)
axs[0, 1].text(0, 1.01, '(b) CPM: $\Delta$x = 4.4 km', ha='left', va='bottom', transform=axs[0, 1].transAxes, fontsize=14)

axs[0, 1].scatter(x=df.lon, y=df.lat, edgecolors="k", marker='o', facecolors='none', s=10, transform=ccrs.PlateCarree(), zorder=105)

points = ((0.235, 0.16), (0.47, 0.16), (0.47, 0.58), (0.235, 0.58))
p0 = Polygon(points, edgecolor='k', facecolor='none', transform=axs[0, 0].transAxes, zorder=106, linestyle="-")
axs[0, 0].add_patch(p0)

points = ((0.1, 0.25), (0.955, 0.25), (0.955, 0.975), (0.1, 0.975))
p1 = Polygon(points, edgecolor='k', facecolor='none', transform=axs[0, 1].transAxes, zorder=106, linestyle="--")
axs[0, 1].add_patch(p1)

# points = ((0.3, 0.23), (0.734, 0.23), (0.734, 0.647), (0.3, 0.647))
# p2 = Polygon(points, edgecolor='k', facecolor='none', transform=axs[0, 1].transAxes, zorder=106, linestyle="--")
# axs[0, 1].add_patch(p2)

# points = ((0.5, 0.5), (1.1, 1))
# p3 = Polygon(points, edgecolor='k', facecolor='none', transform=axs[0, 0].transAxes, zorder=107, linestyle="-")
# axs[0, 0].add_patch(p3)

x_values = np.array([-1.629, 0])
y_values = np.array([0.58, 1])
lines1 = lines.Line2D(x_values, y_values, color='k', linestyle="-", linewidth=1, transform=axs[0, 1].transAxes, zorder=110)
fig.lines.append(lines1)

x_values = np.array([-1.629, 0])
y_values = np.array([0.157, 0])
lines2 = lines.Line2D(x_values, y_values, color='k', linestyle="-", linewidth=1, transform=axs[0, 1].transAxes, zorder=110)
fig.lines.append(lines2)

# points = ((88.4, 26), (113, 29.7))
# p3 = Polygon(points, edgecolor='k', facecolor='none', transform=ccrs.PlateCarree(), zorder=107, linewidth=1, linestyle="-")
# axs[0, 1].add_patch(p3)

points = ((90, 22), (113, 35))
p3 = Polygon(points, edgecolor='k', facecolor='none', transform=ccrs.PlateCarree(), zorder=107, linewidth=1, linestyle="-")
axs[0, 1].add_patch(p3)

# x1, y1 = 90, 22
# x2, y2 = transform(ccrs.PlateCarree(),rot_pole_crs,x1,y1)

# x_values = np.array([0.105, 0.955])
# y_values = np.array([0.485, 0.515])
# lines3 = lines.Line2D(x_values, y_values, color='k', linestyle="-", linewidth=1, transform=axs[0, 1].transAxes, zorder=110)
# fig.lines.append(lines3)

t = axs[0, 1].text(0.11, 0.965, '(c)', ha='left', va='top', transform=axs[0, 1].transAxes, fontsize=14)
t.set_bbox(dict(facecolor='white', alpha=0.65, pad=1, edgecolor='none'))
# t = axs[0, 1].text(0.31, 0.637, '(d)', ha='left', va='top', transform=axs[0, 1].transAxes, fontsize=14)
# t.set_bbox(dict(facecolor='white', alpha=0.65, pad=1, edgecolor='none'))


plt.show()
# plotpath = "/project/pr133/rxiang/figure/paper1/topo/"
# fig.savefig(plotpath + 'topo_station.png', dpi=500, transparent=True)
# plt.close(fig)


