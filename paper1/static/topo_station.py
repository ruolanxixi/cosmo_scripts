# -------------------------------------------------------------------------------
# modules
#
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
import matplotlib
import matplotlib.ticker as mticker
from auxiliary import truncate_colormap
import pandas as pd
from matplotlib import lines
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from shapely.geometry import Polygon, LineString
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.ops import transform
from shapely.ops import split
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cfeature
import fiona
from pyproj import CRS, Transformer
import utilities
from descartes import PolygonPatch

import os

# Set the PROJ_LIB environment variable to the correct path
os.environ['PROJ_LIB'] = '/project/pr133/rxiang/miniconda3/envs/rxiang/share/proj'

font = {'size': 13}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
path = '/users/rxiang/lmp/lib'
ctrl11 = xr.open_dataset(f'{path}/extpar_EAS_ext_12km_merit_unmod_topo.nc')['HSURF'].values[...]
ctrl04 = xr.open_dataset(f'{path}/extpar_BECCY_4.4km_merit_unmod_topo.nc')['HSURF'].values[...]

ds = xr.open_dataset("/project/pr133/rxiang/data/extpar/"
                     + "extpar_BECCY_4.4km_merit_unmod_topo.nc")
rlon = ds["rlon"].values
rlat = ds["rlat"].values
crs_rot_pole = ccrs.RotatedPole(
    pole_longitude=ds["rotated_pole"].grid_north_pole_longitude,
    pole_latitude=ds["rotated_pole"].grid_north_pole_latitude)
ds.close()

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
lon04 = data["rlon"].values
lat04 = data["rlat"].values
v04 = np.nanmean(smr, axis=0)

ds = xr.open_dataset("/project/pr133/rxiang/data/obs/pr/IMERG/day_old/"
                     + "IMERG.ydaymean.2001-2005.mon.nc4")
ds = ds.isel(lon=slice(250, 750), lat=slice(100, 500))
ds = ds.sel(time=(ds["time.month"] >= 5) & (ds["time.month"] <= 9))
prec = ds["pr"].values.mean(axis=0)  # [mm day-1]
lon = ds["lon"].values
lat = ds["lat"].values
ds.close()

file = '/users/rxiang/64stations/Stations_info.txt'
df = pd.read_csv(f"{file}", sep="  ", header=None, names=["lat", "lon"])/100
df = df.iloc[:-2]

# Define regions (polygons)
regions = {}
box = (-24.84, -6.67 - 0.9, -3.00, 12.57)  # (x_min, y_min, x_max, y_max)
print("Boundary with (right): %.2f" % (rlon[-1] - box[2]) + " deg")
print("Boundary with (top): %.2f" % (rlat[-1] - box[3]) + " deg")
regions["ET"] = utilities.grid.polygon_rectangular(box, spacing=0.01)
# Eastern Tibet
box = (-19.0, -6.67, -9.0, 6.0)
regions["HM"] = utilities.grid.polygon_rectangular(box, spacing=0.01)
# Hengduan Mountains

# %%
# Get borders of certain countries
countries = ("India", "Myanmar")
file_shp = shapereader.natural_earth("10m", "cultural", "admin_0_countries")
ds = fiona.open(file_shp)
geom_names = [i["properties"]["NAME"] for i in ds]
poly_count = unary_union([shape(ds[geom_names.index(i)]["geometry"])
                          for i in countries])  # merge all polygons
crs_count = CRS.from_string(ds.crs["init"])
ds.close()

# Transform country polygon to rotated latitude/longitude coordinates
project = Transformer.from_crs(crs_count, CRS.from_user_input(crs_rot_pole),
                               always_xy=True).transform
poly_count_rot = transform(project, poly_count)

# Intersect polygons
regions["HMU"] = regions["HM"].intersection(poly_count_rot)
# Hengduan Mountains Upstream
regions["HMC"] = regions["HM"].difference(regions["HMU"])
# Hengduan Mountains Centre
line = LineString([(-28.24, -3.6), (-2.28, -3.6)])
geom_split = split(regions["HMU"], line)
regions["HMUS"] = geom_split.geoms[0]  # South
regions["HMUN"] = geom_split.geoms[1]  # North

# Regional labels settings
labels = {"ET": {"pos": (86.5, 37.0), "color": "black"},
          "HM": {"pos": (94.8, 31.7), "color": "black"},
          "HMC": {"pos": (101.5, 32.5), "color": "red"},
          "HMUS": {"pos": (95.8, 21.8), "color": "red"},
          "HMUN": {"pos": (94.9, 26.2), "color": "red"}}


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
# %%
ar = 1.0  # initial aspect ratio for first trial
wi = 9.3  # height in inches #15
hi = 6.5  # width in inches #10
ncol = 2  # edit here
nrow = 2
axs, cs, ct, topo, q, qk= np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.06, 0.04, 0.89, 0.95
# gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, left=left, bottom=bottom, right=right, top=top,
#                        wspace=0.155, hspace=0.2, width_ratios=[2, 1.084])

gs = gridspec.GridSpec(2, 1, left=left, bottom=bottom, right=right, top=top,
                        wspace=0.155, hspace=0.2, height_ratios=[1, 1])
gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], width_ratios=[2, 1.084])
gs01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1])

levels = np.arange(0., 6500.0, 500.0)
ticks = np.arange(0., 6500.0, 1000.0)
cmap = truncate_colormap(cmc.bukavu, 0.55, 1.0)
# cmap = cmc.lapaz_r
norm = BoundaryNorm(levels, ncolors=cmap.N, extend="max")

map_ext1 = [65, 173, 7, 61]
map_ext2 = [88, 114, 16, 40]
map_ext = [map_ext1, map_ext2]

for i in range(ncol):
    ext = map_ext[i]
    axs[0, i] = fig.add_subplot(gs00[i], projection=rot_pole_crs)
    axs[0, i].set_extent(ext, crs=ccrs.PlateCarree())
    # axs[0, i].set_aspect("auto")

    axs[0, i].add_feature(cfeature.OCEAN, zorder=100)
    axs[0, i].add_feature(cfeature.COASTLINE, linewidth=2)
    axs[0, i].add_feature(cfeature.BORDERS, linestyle=':')
    axs[0, i].add_feature(cfeature.RIVERS, alpha=0.5)

axs[1, 0] = fig.add_subplot(gs01[0], projection=rot_pole_crs)
axs[1, 0].set_extent(ext, crs=ccrs.PlateCarree())
# axs[0, i].set_aspect("auto")

# axs[1, 0].add_feature(cfeature.OCEAN, zorder=100)
axs[1, 0].add_feature(cfeature.COASTLINE, linewidth=1)
axs[1, 0].add_feature(cfeature.BORDERS, linestyle=':')
# axs[1, 0].add_feature(cfeature.RIVERS, alpha=0.5)

gl = axs[0, 0].gridlines(draw_labels=False, linewidth=1,
                             color='grey', alpha=0.5, linestyle='--', zorder=101)
gl.xlocator = mticker.FixedLocator([60, 80, 100, 120, 140, 160, 180])
gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])
gl = axs[0, 1].gridlines(draw_labels=False, linewidth=1,
                             color='grey', alpha=0.5, linestyle='--', zorder=101)
gl.xlocator = mticker.FixedLocator([85, 90, 95, 100, 105, 110])
gl.ylocator = mticker.FixedLocator([15, 20, 25, 30, 35, 40])

file = "/project/pr133/rxiang/miniconda3/envs/ncl_stable/lib/ncarg/colormaps/OceanLakeLandSnow.rgb"
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

cs[0, 0] = axs[0, 0].pcolormesh(rlon11, rlat11, ctrl11, cmap=cmap, norm=norm, shading="auto", transform=rot_pole_crs)
cs[0, 1] = axs[0, 1].pcolormesh(rlon04, rlat04, ctrl04, cmap=cmap, norm=norm, shading="auto", transform=rot_pole_crs)

levels2 = np.arange(0.0, 21.0, 1.0)
ticks2 = np.arange(0.0, 22.0, 2.0)
cmap = cmc.lapaz_r
norm = matplotlib.colors.BoundaryNorm(levels2, ncolors=cmap.N, extend="max")
cs[1, 0] = axs[1, 0].pcolormesh(lon, lat, prec, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

# [pole_lat, pole_lon, lat, lon, rlat11, rlon11, rot_pole_crs] = pole()
# [pole_lat, pole_lon, lat, lon, rlat04, rlon04, rot_pole_crs] = pole04()
# q[0, 0] = axs[0, 0].quiver(rlon11[::40], rlat11[::40], u11[::40, ::40], v11[::40, ::40], color='black', scale=5000, zorder=102, headaxislength=3.5, headwidth=5, minshaft=0, transform=rot_pole_crs)
q[1, 0] = axs[1, 0].quiver(lon04[::40], lat04[::40], u04[::40, ::40], v04[::40, ::40], color='black', scale=3000, zorder=102, headaxislength=3.5, headwidth=5, minshaft=0, transform=rot_pole_crs)
qk[1, 0] = axs[1, 0].quiverkey(q[1, 0], 0.95, 1.1, 200, r'200', labelpos='S', transform=axs[1, 0].transAxes,
                      fontproperties={'size': 10}, zorder=103)

cax = fig.add_axes([axs[0, 1].get_position().x1+0.02, axs[0, 1].get_position().y0, 0.022, axs[0, 1].get_position().height])
cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical', extend='max', ticks=ticks)
cbar.ax.tick_params(labelsize=14)

cax = fig.add_axes([axs[1, 0].get_position().x1+0.02, axs[1, 0].get_position().y0, 0.022, axs[1, 0].get_position().height])
cbar = fig.colorbar(cs[1, 0], cax=cax, orientation='vertical', extend='max', ticks=ticks2)
cbar.ax.tick_params(labelsize=14)

axs[0, 1].text(-0.01, 0.86, '35°N', ha='right', va='center', transform=axs[0, 1].transAxes, fontsize=14)
axs[0, 1].text(-0.01, 0.665, '30°N', ha='right', va='center', transform=axs[0, 1].transAxes, fontsize=14)
axs[0, 1].text(-0.01, 0.47, '25°N', ha='right', va='center', transform=axs[0, 1].transAxes, fontsize=14)
axs[0, 1].text(-0.01, 0.275, '20°N', ha='right', va='center', transform=axs[0, 1].transAxes, fontsize=14)
axs[0, 1].text(-0.01, 0.08, '15°N', ha='right', va='center', transform=axs[0, 1].transAxes, fontsize=14)

axs[0, 1].text(0.04, -0.02, '90°E', ha='center', va='top', transform=axs[0, 1].transAxes, fontsize=14)
axs[0, 1].text(0.46, -0.02, '100°E', ha='center', va='top', transform=axs[0, 1].transAxes, fontsize=14)
axs[0, 1].text(0.86, -0.02, '110°E', ha='center', va='top', transform=axs[0, 1].transAxes, fontsize=14)

axs[1, 0].text(-0.01, 0.86, '35°N', ha='right', va='center', transform=axs[1, 0].transAxes, fontsize=14)
axs[1, 0].text(-0.01, 0.665, '30°N', ha='right', va='center', transform=axs[1, 0].transAxes, fontsize=14)
axs[1, 0].text(-0.01, 0.47, '25°N', ha='right', va='center', transform=axs[1, 0].transAxes, fontsize=14)
axs[1, 0].text(-0.01, 0.275, '20°N', ha='right', va='center', transform=axs[1, 0].transAxes, fontsize=14)
axs[1, 0].text(-0.01, 0.08, '15°N', ha='right', va='center', transform=axs[1, 0].transAxes, fontsize=14)

axs[1, 0].text(0.04, -0.02, '90°E', ha='center', va='top', transform=axs[1, 0].transAxes, fontsize=14)
axs[1, 0].text(0.46, -0.02, '100°E', ha='center', va='top', transform=axs[1, 0].transAxes, fontsize=14)
axs[1, 0].text(0.86, -0.02, '110°E', ha='center', va='top', transform=axs[1, 0].transAxes, fontsize=14)

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
axs[1, 0].text(0, 1.01, '(c) IMERG precipitation', ha='left', va='bottom', transform=axs[1, 0].transAxes, fontsize=14)

axs[0, 1].scatter(x=df.lon, y=df.lat, edgecolors="k", marker='o', facecolors='none', s=10, transform=ccrs.PlateCarree(), zorder=105)

points = ((0.235, 0.16), (0.47, 0.16), (0.47, 0.58), (0.235, 0.58))
p0 = Polygon(points)
axs[0, 0].add_patch(plt.Polygon(points, edgecolor='k', facecolor='none', transform=axs[0, 0].transAxes, zorder=106, linestyle="-"))

# points = ((0.1, 0.25), (0.955, 0.25), (0.955, 0.975), (0.1, 0.975))
# p1 = Polygon(points, edgecolor='k', facecolor='none', transform=axs[0, 1].transAxes, zorder=106, linestyle="--")
# axs[0, 1].add_patch(p1)

x_values = np.array([-1.69, 0])
y_values = np.array([0.58, 1])
lines1 = lines.Line2D(x_values, y_values, color='k', linestyle="-", linewidth=1, transform=axs[0, 1].transAxes, zorder=110)
fig.lines.append(lines1)

x_values = np.array([-1.69, 0])
y_values = np.array([0.157, 0])
lines2 = lines.Line2D(x_values, y_values, color='k', linestyle="-", linewidth=1, transform=axs[0, 1].transAxes, zorder=110)
fig.lines.append(lines2)

# points = ((88.4, 26), (113, 29.7))
# p3 = Polygon(points, edgecolor='k', facecolor='none', transform=ccrs.PlateCarree(), zorder=107, linewidth=1, linestyle="-")
# axs[0, 1].add_patch(p3)

# points = ((90, 22), (113, 35))
# p3 = Polygon(points)
# axs[0, 1].add_patch(plt.Polygon(points, edgecolor='k', facecolor='none', transform=ccrs.PlateCarree(), zorder=107, linewidth=1, linestyle="-"))
# plot cross section
start_rlat = -2.23
start_rlon = -24.8
end_rlat = 3.00
end_rlon = -2.96
# axs[0, 1].plot([-24.8, -2.96], [-0.03, 0.63], transform=rot_pole_crs)
# axs[0, 2].plot([-24.8, -2.96], [-0.03, 0.63], transform=rot_pole_crs)
axs[0, 1].plot([start_rlon, end_rlon], [start_rlat, end_rlat], transform=rot_pole_crs)
axs[1, 0].plot([start_rlon, end_rlon], [start_rlat, end_rlat], transform=rot_pole_crs)

poly_plot = PolygonPatch(regions["ET"], facecolor="none",
                         edgecolor="black", lw=1.5, ls="--",
                         transform=crs_rot_pole, zorder=106)
axs[0, 1].add_patch(poly_plot)
poly_plot = PolygonPatch(regions["ET"], facecolor="none",
                         edgecolor="black", lw=1.5, ls="--",
                         transform=crs_rot_pole, zorder=106)
axs[1, 0].add_patch(poly_plot)
poly_plot = PolygonPatch(regions["HM"], facecolor="none",
                         edgecolor="black", lw=1.5, ls="--",
                         transform=crs_rot_pole, zorder=106)
axs[0, 1].add_patch(poly_plot)
poly_plot = PolygonPatch(regions["HM"], facecolor="none",
                         edgecolor="black", lw=1.5, ls="--",
                         transform=crs_rot_pole, zorder=106)
axs[1, 0].add_patch(poly_plot)
for i in ["HMC", "HMUS", "HMUN"]:
    poly_plot = PolygonPatch(regions[i], facecolor="none",
                             edgecolor="orangered", lw=1.5, ls="-",
                             transform=crs_rot_pole, zorder=2)
    axs[1, 0].add_patch(poly_plot)
for i in ["ET", "HM"]:
    axs[0, 1].text(*labels[i]["pos"], i, color=labels[i]["color"], fontsize=10, transform=ccrs.PlateCarree(), zorder=200, weight='bold')
    axs[1, 0].text(*labels[i]["pos"], i, color=labels[i]["color"], fontsize=10, transform=ccrs.PlateCarree(), zorder=200, weight='bold')
for i in ["HMC", "HMUS", "HMUN"]:
    axs[1, 0].text(*labels[i]["pos"], i, color=labels[i]["color"], fontsize=7, transform=ccrs.PlateCarree(), zorder=200, weight='bold')

# x1, y1 = 90, 22
# x2, y2 = transform(ccrs.PlateCarree(),rot_pole_crs,x1,y1)

# x_values = np.array([0.105, 0.955])
# y_values = np.array([0.485, 0.515])
# lines3 = lines.Line2D(x_values, y_values, color='k', linestyle="-", linewidth=1, transform=axs[0, 1].transAxes, zorder=110)
# fig.lines.append(lines3)

# t = axs[0, 1].text(0.11, 0.965, '(c)', ha='left', va='top', transform=axs[0, 1].transAxes, fontsize=14)
# t.set_bbox(dict(facecolor='white', alpha=0.65, pad=1, edgecolor='none'))
# t = axs[0, 1].text(0.31, 0.637, '(d)', ha='left', va='top', transform=axs[0, 1].transAxes, fontsize=14)
# t.set_bbox(dict(facecolor='white', alpha=0.65, pad=1, edgecolor='none'))


plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/topo/"
fig.savefig(plotpath + 'topo3.png', dpi=500, transparent=True)
# plt.close(fig)


