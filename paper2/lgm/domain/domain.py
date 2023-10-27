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
import pickle

import os

# Set the PROJ_LIB environment variable to the correct path
os.environ['PROJ_LIB'] = '/project/pr133/rxiang/miniconda3/envs/rxiang/share/proj'

font = {'size': 15}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
path = '/users/rxiang/lmp/lib'
ctrl11 = xr.open_dataset(f'{path}/extpar_EAS_ext_12km_merit_lgm.nc')['HSURF'].values[...]
masked_ctrl11 = np.ma.masked_where(ctrl11 <= 0, ctrl11)
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
# --
path_shp = "/project/pr133/csteger/Data/Shapefiles/Pan-Tibetan_Highlands/" \
           + "Pan-Tibetan_Highlands_Liu_2022/Shapefile/"
# Tibetan Plateau outlines (Liu et al., 2022)
ds = fiona.open(path_shp + "Pan-Tibetan_Highlands_Liu_2022_L.shp")
poly_tb = shape(ds[0]["geometry"])  # shapely LineString

# TPSCE outlines
box = (66.0, 25.0, 106.0, 40.0)
poly_tpsce = utilities.grid.polygon_rectangular(box, spacing=0.01)

# Compute intersection region
poly_inters = Polygon(poly_tb).intersection(poly_tpsce)
crs_geo = ccrs.PlateCarree()

project = Transformer.from_crs(CRS.from_user_input(crs_geo),
                               CRS.from_user_input(crs_rot_pole),
                               always_xy=True).transform
poly_inters_rot = transform(project, poly_inters)

# Compute intersections between region polygons and area covered by
# 'HMA_SR_D' product
regions["HM_snow"] = regions["HM"].intersection(poly_inters_rot)
# %%
ar = 1.0  # initial aspect ratio for first trial
wi = 9  # height in inches #15
hi = 6  # width in inches #10
ncol = 2  # edit here
nrow = 2
axs, cs, ct, topo, q, qk= np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.07, 0.13, 0.99, 1
# gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, left=left, bottom=bottom, right=right, top=top,
#                        wspace=0.155, hspace=0.2, width_ratios=[2, 1.084])

gs = gridspec.GridSpec(1, 1, left=left, bottom=bottom, right=right, top=top,
                        wspace=0.155, hspace=0.2)

levels = np.arange(0., 6500.0, 500.0)
ticks = np.arange(0., 6500.0, 1000.0)
cmap = truncate_colormap(cmc.bukavu, 0.55, 1.0)
cmap.set_bad(color="skyblue")
# cmap = cmc.lapaz_r
norm = BoundaryNorm(levels, ncolors=cmap.N, extend="max")

map_ext1 = [65, 173, 7, 61]
map_ext2 = [88, 114, 16, 40]
map_ext = [map_ext1, map_ext2]


ext = map_ext1
axs[0, 0] = fig.add_subplot(gs[0], projection=rot_pole_crs)
axs[0, 0].set_extent(ext, crs=ccrs.PlateCarree())
    # axs[0, i].set_aspect("auto")
with open('/project/pr133/rxiang/data/extpar/lgm_contour.pkl', 'rb') as file:
    contours_rlatrlon = pickle.load(file)

for contour in contours_rlatrlon:
    axs[0, 0].plot(contour[:, 0], contour[:, 1], c='black', linewidth=1)

gl = axs[0, 0].gridlines(draw_labels=False, linewidth=1,
                             color='grey', alpha=0.5, linestyle='--', zorder=101)
gl.xlocator = mticker.FixedLocator([60, 80, 100, 120, 140, 160, 180])
gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])
# gl = axs[0, 1].gridlines(draw_labels=False, linewidth=1,
#                              color='grey', alpha=0.5, linestyle='--', zorder=101)
# gl.xlocator = mticker.FixedLocator([85, 90, 95, 100, 105, 110])
# gl.ylocator = mticker.FixedLocator([15, 20, 25, 30, 35, 40])

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

axs[0, 0].add_feature(cfeature.OCEAN, zorder=1)
cs[0, 0] = axs[0, 0].pcolormesh(rlon11, rlat11, masked_ctrl11, cmap=cmap, norm=norm, shading="auto", transform=rot_pole_crs)

axs[0, 0].add_feature(cfeature.BORDERS, linestyle=':')
axs[0, 0].add_feature(cfeature.RIVERS, alpha=0.5)
#
cax = fig.add_axes([axs[0, 0].get_position().x0, axs[0, 0].get_position().y0 - 0.08, axs[0, 0].get_position().width, 0.025])
cbar = fig.colorbar(cs[0, 0], cax=cax, orientation='horizontal', extend='max', ticks=ticks, label="[m]")
cbar.ax.tick_params(labelsize=14)

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
#
poly_plot = PolygonPatch(regions["HM"], facecolor="none",
                         edgecolor="black", lw=1.5, ls="--",
                         transform=crs_rot_pole, zorder=106)
axs[0, 0].add_patch(poly_plot)

poly_plot = PolygonPatch(regions["HM_snow"], facecolor="none",
                         edgecolor="blue", lw=2.5, ls="--",
                         transform=crs_rot_pole)
axs[0, 0].add_patch(poly_plot)

for i in ["HM"]:
    axs[0, 0].text(*labels[i]["pos"], i, color=labels[i]["color"], fontsize=10, transform=ccrs.PlateCarree(), zorder=200, weight='bold')

plt.show()
# plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
# fig.savefig(plotpath + 'sim_domain.png', dpi=700, transparent=True)
# plt.close(fig)

# %%
wi = 7  # height in inches #15
hi = 7  # width in inches #10
axs, cs, ct, topo, q, qk= np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.1, 0.15, 0.99, 1
# gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, left=left, bottom=bottom, right=right, top=top,
#                        wspace=0.155, hspace=0.2, width_ratios=[2, 1.084])

gs = gridspec.GridSpec(1, 1, left=left, bottom=bottom, right=right, top=top,
                        wspace=0.155, hspace=0.2)

map_ext2 = [88, 114, 20, 40]
axs[0, 0] = fig.add_subplot(gs[0], projection=rot_pole_crs)
axs[0, 0].set_extent(map_ext2, crs=ccrs.PlateCarree())
    # axs[0, i].set_aspect("auto")
with open('/project/pr133/rxiang/data/extpar/lgm_contour.pkl', 'rb') as file:
    contours_rlatrlon = pickle.load(file)

for contour in contours_rlatrlon:
    axs[0, 0].plot(contour[:, 0], contour[:, 1], c='black', linewidth=1)

gl.xlocator = mticker.FixedLocator([85, 90, 95, 100, 105, 110])
gl.ylocator = mticker.FixedLocator([20, 25, 30, 35, 40])
gl = axs[0, 0].gridlines(draw_labels=False, linewidth=1,
                             color='grey', alpha=0.5, linestyle='--', zorder=101)
gl.xlabels_top = False
gl.ylabels_right = False

axs[0, 0].add_feature(cfeature.OCEAN, zorder=1)
cs[0, 0] = axs[0, 0].pcolormesh(rlon11, rlat11, masked_ctrl11, cmap=cmap, norm=norm, shading="auto", transform=rot_pole_crs)

# axs[0, 0].add_feature(cfeature.COASTLINE, linestyle='--')
axs[0, 0].add_feature(cfeature.BORDERS, linestyle=':')
axs[0, 0].add_feature(cfeature.RIVERS, alpha=0.5)
#
# poly_plot = PolygonPatch(regions["ET"], facecolor="none",
#                          edgecolor="black", lw=1.5, ls="--",
#                          transform=crs_rot_pole, zorder=106)
# axs[0, 0].add_patch(poly_plot)
poly_plot = PolygonPatch(regions["HM"], facecolor="none",
                         edgecolor="black", lw=1.5, ls="--",
                         transform=crs_rot_pole, zorder=106)
axs[0, 0].add_patch(poly_plot)

for i in ["HM"]:
    axs[0, 0].text(*labels[i]["pos"], i, color=labels[i]["color"], fontsize=17, transform=ccrs.PlateCarree(), zorder=200, weight='bold')

axs[0, 0].text(-0.008, 0.83, '35°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=17)
axs[0, 0].text(-0.008, 0.60, '30°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=17)
axs[0, 0].text(-0.008, 0.37, '25°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=17)
axs[0, 0].text(-0.008, 0.14, '20°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=17)

axs[0, 0].text(0.05, -0.02, '90°E', ha='center', va='top', transform=axs[0, 0].transAxes, fontsize=17)
axs[0, 0].text(0.25, -0.02, '95°E', ha='center', va='top', transform=axs[0, 0].transAxes, fontsize=17)
axs[0, 0].text(0.45, -0.02, '100°E', ha='center', va='top', transform=axs[0, 0].transAxes, fontsize=17)
axs[0, 0].text(0.65, -0.02, '105°E', ha='center', va='top', transform=axs[0, 0].transAxes, fontsize=17)
axs[0, 0].text(0.85, -0.02, '110°E', ha='center', va='top', transform=axs[0, 0].transAxes, fontsize=17)

cax = fig.add_axes([axs[0, 0].get_position().x0, axs[0, 0].get_position().y0 - 0.08, axs[0, 0].get_position().width, 0.025])
cbar = fig.colorbar(cs[0, 0], cax=cax, orientation='horizontal', extend='max', ticks=ticks, label="[m]")
cbar.ax.tick_params(labelsize=15)
plt.show()
# plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
# fig.savefig(plotpath + 'sim_domain_HM.png', dpi=700, transparent=True)
