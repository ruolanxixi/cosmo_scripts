# Description: Plot nested COSMO domains
#
# Author: Christian R. Steger, October 2022

# Load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from shapely.geometry import Polygon
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection

from shapely.geometry import MultiLineString
import matplotlib.path as mpath
import matplotlib.patches as mpatches

mpl.style.use("classic")

# Change latex fonts
mpl.rcParams["mathtext.fontset"] = "custom"
# custom mathtext font (set default to Bitstream Vera Sans)
mpl.rcParams["mathtext.default"] = "rm"
mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"

font = {'size': 18}
mpl.rc('font', **font)

###############################################################################
# Load data (elevation, coordinates, rotated coordinate system)
###############################################################################

# Notes: load required input data either from EXTPAR NetCDF files or from
#        *c.nc files with constant fields that are created during the
#        COSMO run

# From EXTPAR file (select relevant subdomain)
files = {"coarse": {"path": "/users/rxiang/lmp/lib/"
                            + "extpar_EAS_ext_12km_merit_unmod_topo.nc",
                    "slice_rlon": slice(32, 1089),
                    "slice_rlat": slice(28, 637)},
         "fine": {"path": "/users/rxiang/lmp/lib/"
                          + "extpar_BECCY_4.4km_merit_unmod_topo.nc",
                  "slice_rlon": slice(30, 679),
                  "slice_rlat": slice(30, 679)}}

# # From COSMO output file (*c.nc)
# files = {"coarse": {"path": "/users/rxiang/lmp/lib/"
#                             + "lffd20210522000000c_lm_c.nc",
#                     "slice_rlon": slice(None, None),
#                     "slice_rlat": slice(None, None)},
#          "fine": {"path": "/users/rxiang/lmp/lib/"
#                           + "lffd20210522000000c_lm_f.nc",
#                   "slice_rlon": slice(None, None),
#                   "slice_rlat": slice(None, None)}}
# %%
# Load data
data = {}
for i in list(files.keys()):
    ds = xr.open_dataset(files[i]["path"])
    ds = ds.isel(rlon=files[i]["slice_rlon"], rlat=files[i]["slice_rlat"])
    data[i] = {"elev": ds["HSURF"].values.squeeze(),
               "lsm": ds["FR_LAND"].values.squeeze(),
               "rlon": ds["rlon"].values,
               "rlat": ds["rlat"].values,
               "lon": ds["lon"].values,
               "lat": ds["lat"].values,
               "rot_pole_lat": ds["rotated_pole"].grid_north_pole_latitude,
               "rot_pole_lon": ds["rotated_pole"].grid_north_pole_longitude}
    ds.close()

# %%
###############################################################################
# Domain specifications
###############################################################################
# %%
# COSMO domains
domains = {
    # # "SAS": {
    # #     "name_plot": "SAS",
    # #     "startlat_tot": 5.0,
    # #     "startlon_tot": 60.0,
    # #     "endlat_tot": 30.0,
    # #     "endlon_tot": 100.0,
    # #     "dlon": 0.11,
    # #     "dlat": 0.11,
    # },
    "EAS": {
        "name_plot": "EAS",
        "startlat_tot": 20.0,
        "startlon_tot": 100.0,
        "endlat_tot": 50.0,
        "endlon_tot": 145.0,
        "dlon": 0.11,
        "dlat": 0.11,
    },
    # "SEA": {
    #     "name_plot": "SEA",
    #     "startlat_tot": -10.0,
    #     "startlon_tot": 95.0,
    #     "endlat_tot": 20.0,
    #     "endlon_tot": 155.0,
    #     "dlon": 0.11,
    #     "dlat": 0.11,
    # },
    "TIB": {
        "name_plot": "SEA",
        "startlat_tot": 30.0,
        "startlon_tot": 75.0,
        "endlat_tot": 50.0,
        "endlon_tot": 100.0,
        "dlon": 0.11,
        "dlat": 0.11,
    }
}


###############################################################################
# Functions
###############################################################################


# Compute 1-dimensional coordinates for domain
def lon_lat_1d(domain, poly_res=0.01):
    lon = np.arange(domain["startlon_tot"], domain["endlon_tot"], domain["dlon"])
    lat = np.arange(domain["startlat_tot"], domain["endlat_tot"], domain["dlat"])

    return lon, lat


# Compute polygon from 1-dimensional coordinates
def coord2poly(x, y):
    poly_x = np.hstack((x,
                        np.repeat(x[-1], len(y))[1:],
                        x[::-1][1:],
                        np.repeat(x[0], len(y))[1:]))
    poly_y = np.hstack((np.repeat(y[0], len(x)),
                        y[1:],
                        np.repeat(y[-1], len(x))[1:],
                        y[::-1][1:]))

    return poly_x, poly_y


# %%
###############################################################################
# Map plot
###############################################################################

# Load NCL colormap (optional; comment out in case NCL colormap is not used)
file = "/users/rxiang/miniconda3/envs/ncl_stable/lib/ncarg/colormaps/OceanLakeLandSnow.rgb"
# Source of NCL rgb-file:
# https://www.ncl.ucar.edu/Document/Graphics/color_table_gallery.shtml
rgb = np.loadtxt(file, comments=("#", "ncolors"))
if rgb.max() > 1.0:
    rgb /= 255.0
print("Number of colors: " + str(rgb.shape[0]))
cmap_ncl = mpl.colors.LinearSegmentedColormap.from_list("OceanLakeLandSnow",
                                                        rgb, N=rgb.shape[0])

# Colormap for terrain
# cmap_ter, min_v, max_v = plt.get_cmap("terrain"), 0.25, 1.00  # matplotlib
# cmap_ter, min_v, max_v = cm.fes, 0.50, 1.00                   # crameri
cmap_ter, min_v, max_v = cmap_ncl, 0.05, 1.00  # NCL
cmap_ter = mpl.colors.LinearSegmentedColormap.from_list(
    "trunc({n},{a:.2f},{b:.2f})".format(
        n=cmap_ter.name, a=min_v, b=max_v),
    cmap_ter(np.linspace(min_v, max_v, 100)))
levels_ter = np.arange(0.0, 6000.0, 500.0)
norm_ter = mpl.colors.BoundaryNorm(levels_ter, ncolors=cmap_ter.N,
                                   extend="max")

# Color for sea/ocean (water)
cmap_sea = mpl.colors.ListedColormap(["lightskyblue"])
bounds_sea = [0.5, 1.5]
norm_sea = mpl.colors.BoundaryNorm(bounds_sea, cmap_sea.N)

# Domain labels
lab = {"coarse": {"txt": r'LSM: $\Delta$x = 12 km', "offset": (10, -2)},
       "fine": {"txt": r'CPM: $\Delta$x = 4.4 km', "offset": (10.5, -2)},
       "EAS": {"txt": r'EAS', "offset": (2, -3), "rotation": -5},
       "SAS": {"txt": r'SAS', "offset": (1.8, -4.2), "rotation": -20},
       "SEA": {"txt": r'SEA', "offset": (1.5, -3.6), "rotation": -6},
       "TIB": {"txt": r'TIB', "offset": (2.3, -3.6), "rotation": -17}}

# Inner domain (without boundary relaxation zone) (optional)
brz_w = {"coarse": 15, "fine": 80}
plot_brz = False

# Map plot
crs_map = ccrs.RotatedPole(pole_latitude=data["coarse"]["rot_pole_lat"],
                           pole_longitude=data["coarse"]["rot_pole_lon"])
fig = plt.figure(figsize=(17, 10))
gs = gridspec.GridSpec(3, 2, left=0.06, bottom=0.02, right=0.92,
                       top=0.98, hspace=0.0, wspace=0.04,
                       width_ratios=[1.0, 0.025],
                       height_ratios=[0.3, 1.0, 0.3])
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[:, 0], projection=crs_map)
for i in list(files.keys()):
    crs_rot = ccrs.RotatedPole(pole_latitude=data[i]["rot_pole_lat"],
                               pole_longitude=data[i]["rot_pole_lon"])
    # -------------------------------------------------------------------------
    land = cfeature.NaturalEarthFeature("physical", "land", scale="50m",
                                        edgecolor="black",
                                        facecolor="lightgray")
    ax.add_feature(land, zorder=1)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=2)
    # -------------------------------------------------------------------------
    rlon, rlat = data[i]["rlon"], data[i]["rlat"]
    data_plot = np.ones_like(data[i]["lsm"])
    plt.pcolormesh(rlon, rlat, data_plot, transform=crs_rot, shading="auto",
                   cmap=cmap_sea, norm=norm_sea, zorder=2, rasterized=True)
    data_plot = np.ma.masked_where(data[i]["lsm"] < 0.5, data[i]["elev"])
    plt.pcolormesh(rlon, rlat, data_plot, transform=crs_rot, shading="auto",
                   cmap=cmap_ter, norm=norm_ter, zorder=3, rasterized=True)
    # -------------------------------------------------------------------------
    dx_h = np.diff(rlon).mean() / 2.0
    x = [rlon[0] - dx_h, rlon[-1] + dx_h, rlon[-1] + dx_h, rlon[0] - dx_h]
    dy_h = np.diff(rlat).mean() / 2.0
    y = [rlat[0] - dy_h, rlat[0] - dy_h, rlat[-1] + dy_h, rlat[-1] + dy_h]
    poly = plt.Polygon(list(zip(x, y)), facecolor="none", edgecolor="black",
                       linewidth=2.5, zorder=4)
    ax.add_patch(poly)
    # -------------------------------------------------------------------------
    # Plot inner domain (without boundary relaxation zone)
    # -------------------------------------------------------------------------
    if plot_brz:
        dx_h = np.diff(rlon).mean() / 2.0
        x = [rlon[0 + brz_w[i]] - dx_h, rlon[-1 - brz_w[i]] + dx_h,
             rlon[-1 - brz_w[i]] + dx_h, rlon[0 + brz_w[i]] - dx_h]
        dy_h = np.diff(rlat).mean() / 2.0
        y = [rlat[0 + brz_w[i]] - dy_h, rlat[0 + brz_w[i]] - dy_h,
             rlat[-1 - brz_w[i]] + dy_h, rlat[-1 - brz_w[i]] + dy_h]
        poly = plt.Polygon(list(zip(x, y)), facecolor="none",
                           edgecolor="black", linestyle="--",
                           linewidth=1.0, zorder=4)
        ax.add_patch(poly)
    # -------------------------------------------------------------------------
    t = plt.text(rlon[0] + lab[i]["offset"][0],
                 rlat[-1] + lab[i]["offset"][1], lab[i]["txt"],
                 fontsize=16, fontweight="bold", horizontalalignment="center",
                 verticalalignment="center", transform=crs_rot, zorder=6)
    t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.5))
# -----------------------------------------------------------------------------
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.6, color="gray",
                  draw_labels=True, alpha=0.5, linestyle="--",
                  x_inline=False, y_inline=False, zorder=5)
gl_spac = 10  # grid line spacing for map plot [degree]
gl.xlocator = mticker.FixedLocator(range(-180, 180 + gl_spac, gl_spac))
gl.ylocator = mticker.FixedLocator(range(-90, 90 + gl_spac, gl_spac))
# gl.xlocator = mticker.FixedLocator([80, 100, 120, 140, 160])
# gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])

# add ticks manually
ax.text(-0.008, 0.93, '50°N', ha='right', va='center', transform=ax.transAxes)
ax.text(-0.008, 0.77, '40°N', ha='right', va='center', transform=ax.transAxes)
ax.text(-0.008, 0.61, '30°N', ha='right', va='center', transform=ax.transAxes)
ax.text(-0.008, 0.45, '20°N', ha='right', va='center', transform=ax.transAxes)
ax.text(-0.008, 0.29, '10°N', ha='right', va='center', transform=ax.transAxes)
ax.text(-0.008, 0.13, '0°N', ha='right', va='center', transform=ax.transAxes)

# ax.text(0.12, -0.02, '80°E', ha='center', va='top', transform=ax.transAxes)
# ax.text(0.32, -0.02, '100°E', ha='center', va='top', transform=ax.transAxes)
# ax.text(0.52, -0.02, '120°E', ha='center', va='top', transform=ax.transAxes)
# ax.text(0.72, -0.02, '140°E', ha='center', va='top', transform=ax.transAxes)
# ax.text(0.92, -0.02, '160°E', ha='center', va='top', transform=ax.transAxes)
gl.right_labels, gl.top_labels, gl.left_labels = False, False, False
# -----------------------------------------------------------------------------
# add subdomains
# -----------------------------------------------------------------------------
for i in list(domains.keys()):
    lon, lat = lon_lat_1d(domains[i], poly_res=0.02)
    poly_lon, poly_lat = coord2poly(lon, lat)
    # coords = crs_rot.transform_points(crs_rot, poly_rlon, poly_rlat)
    poly = plt.Polygon(list(zip(poly_lon, poly_lat)),
                       facecolor="none", edgecolor="black", alpha=1.0,
                       linewidth=1.5, zorder=4, transform=ccrs.PlateCarree())
    ax.add_patch(poly)
    poly_shp = Polygon(zip(poly_lon, poly_lat))
    poly = PolygonPatch(poly_shp, facecolor="none", edgecolor="gray",
                        alpha=1.0, linewidth=2, zorder=4,
                        transform=ccrs.PlateCarree())
    ax.add_patch(poly)
    x, y = poly_shp.exterior.coords.xy
    x_txt, y_txt = lon[0], lat[-1]
    if (i in lab.keys()):
        x_txt += lab[i]['offset'][0]
        y_txt += lab[i]['offset'][1]
        plt.text(x_txt, y_txt, lab[i]['txt'], rotation=lab[i]["rotation"], fontweight="bold", transform=ccrs.PlateCarree())

from matplotlib.patches import Polygon
lon = np.arange(60, 100, 0.11)
lon1 = np.arange(60, 95, 0.11)
lon2 = np.arange(95, 100, 0.11)
lat1 = np.arange(5, 20, 0.11)
lat2 = np.arange(20, 30, 0.11)
lat = np.arange(5, 30, 0.11)
poly_lon = np.hstack((lon[31:len(lon1)], np.repeat(95, len(lat1))[1:], lon[len(lon1):], np.repeat(100, len(lat2))[1:], lon[::-1][1:], np.repeat(lon[0], len(lat)-72)[1:]))
poly_lat = np.hstack((np.repeat(lat[0], len(lon1)-31), lat[0:len(lat1)-1], np.repeat(20, len(lon2))[1:], lat[len(lat1):][1:],  np.repeat(lat[-1], len(lon))[1:], lat[::-1][1:-72]))
polygon = Polygon(list(zip(poly_lon, poly_lat)), False, facecolor="none", edgecolor="grey", alpha=1.0, linewidth=2, zorder=4, transform=ccrs.PlateCarree())
ax.add_patch(polygon)
plt.text(lon[0]+1.8, lat[-1]-4.2, "SAS", rotation=-20, fontweight="bold", transform=ccrs.PlateCarree())

lon = np.arange(95, 155, 0.11)
lat = np.arange(-10, 20, 0.11)
poly_lon = np.hstack((np.repeat(lon[-1], len(lat)-68)[1:], lon[::-1][1:], np.repeat(lon[0], len(lat-112))[1:]))
poly_lat = np.hstack((lat[68:], np.repeat(lat[-1], len(lon))[1:], lat[::-1][1:-112]))
polygon = Polygon(list(zip(poly_lon, poly_lat)), False, facecolor="none", edgecolor="grey", alpha=1.0, linewidth=2, zorder=4, transform=ccrs.PlateCarree())
ax.add_patch(polygon)
plt.text(lon[-1]-6.5, lat[-1]-2.7, "SEA", rotation=13, fontweight="bold", transform=ccrs.PlateCarree())


# -----------------------------------------------------------------------------
ext_dom = 2.0  # increase map extent [degree]
ax.set_extent([data["coarse"]["rlon"][0] - ext_dom,
               data["coarse"]["rlon"][-1] + ext_dom,
               data["coarse"]["rlat"][0] - ext_dom,
               data["coarse"]["rlat"][-1] + ext_dom], crs=crs_map)

# -----------------------------------------------------------------------------
ax = plt.subplot(gs[1:2, 1])
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap_ter, norm=norm_ter,
                               ticks=levels_ter, orientation="vertical")
plt.ylabel("Elevation [m]", labelpad=8.0)
# -----------------------------------------------------------------------------

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/validation/LSM/"
fig.savefig(plotpath + 'domains.png', dpi=500)
plt.close(fig)
# ds = xr.open_dataset(files["coarse"]["path"])
# ds = ds.isel(rlon=files["coarse"]["slice_rlon"], rlat=files["coarse"]["slice_rlat"])
# ds = ds.where((ds.lon > 70) & (ds.lon < 90) & (ds.lat > 10) & (ds.lat < 27) & (ds.FR_LAND > 0.5))
# data_sub = {}
# data_sub['IN'] = {"elev": ds["HSURF"].values.squeeze(),
#                   "lsm": ds["FR_LAND"].values.squeeze(),
#                   "rlon": ds["rlon"].values,
#                   "rlat": ds["rlat"].values,
#                   "lon": ds["lon"].values,
#                   "lat": ds["lat"].values,
#                   "rot_pole_lat": ds["rotated_pole"].grid_north_pole_latitude,
#                   "rot_pole_lon": ds["rotated_pole"].grid_north_pole_longitude}
# rlon, rlat = data_sub['IN']["rlon"], data_sub['IN']["rlat"]
# # data_plot = np.ma.masked_where(data_sub['IN']["lsm"] < 0.5, data_sub['IN']["elev"])
# data_plot = data_sub["IN"]["elev"]
# data_pplot = copy.deepcopy(data_plot)
# for i in range(609):
#     for j in range(1057):
#         if (not np.isnan(data_plot[i][j])) and (not np.isnan(data_plot[i][j - 1])) and \
#                 (not np.isnan(data_plot[i][j + 1])) and (not np.isnan(data_plot[i + 1][j])) and \
#                 (not np.isnan(data_plot[i + 1][j - 1])) and (not np.isnan(data_plot[i + 1][j + 1])) and \
#                 (not np.isnan(data_plot[i - 1][j])) and (not np.isnan(data_plot[i - 1][j + 1])) and \
#                 (not np.isnan(data_plot[i - 1][j - 1])):
#             data_pplot[i][j] = np.nan
#
#         if (not np.isnan(data_pplot[i][j])):
#             data_pplot[i][j] = 5000
# plt.pcolormesh(rlon, rlat, data_pplot, transform=crs_rot, shading="auto",
#                cmap='binary', norm=norm_ter, zorder=3, rasterized=True)
