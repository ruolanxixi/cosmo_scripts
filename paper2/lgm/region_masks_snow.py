#%% Description: Compute Eastern Tibet / Hengduan Mountains regions masks for
#              different products for snow evaluation
#
# Author: Christian R. Steger, September 2023

# Load modules
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely.geometry import Polygon
from shapely.ops import transform
from shapely.geometry import shape
import cartopy.crs as ccrs
from descartes import PolygonPatch
from pyproj import CRS, Transformer
import utilities
import matplotlib.gridspec as gridspec
from netCDF4 import Dataset
import pickle
import fiona
from matplotlib.colors import BoundaryNorm
from auxiliary import truncate_colormap
import cmcrameri.cm as cmc
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

mpl.style.use("classic")
# %%
# Path to folders
path_masks = "/project/pr133/csteger/Data/Model/BECCY/region_masks/"
path_shp = "/project/pr133/csteger/Data/Shapefiles/Pan-Tibetan_Highlands/" \
           + "Pan-Tibetan_Highlands_Liu_2022/Shapefile/"

###############################################################################
# Define regions in rotated latitude/longitude coordinates
###############################################################################

# Get COSMO rotated coordinate system
ds = xr.open_dataset("/project/pr133/rxiang/data/extpar/"
                     + "extpar_BECCY_4.4km_merit_unmod_topo.nc")
crs_rot_pole = ccrs.RotatedPole(
    pole_longitude=ds["rotated_pole"].grid_north_pole_longitude,
    pole_latitude=ds["rotated_pole"].grid_north_pole_latitude)
topo = ds['HSURF'].values[...]
rlon = ds['rlon'].values[...]
rlat = ds['rlat'].values[...]
ds.close()

# Load relevant polygons (-> created with 'region_masks.py')
regions = {}
for i in ["ET", "HM"]:
    file_poly = path_masks + "region_polygons/" + i + "_rot_coord.poly"
    with open(file_poly, "rb") as file:
        regions[i] = pickle.load(file)

# Tibetan Plateau outlines (Liu et al., 2022)
ds = fiona.open(path_shp + "Pan-Tibetan_Highlands_Liu_2022_L.shp")
poly_tb = shape(ds[0]["geometry"])  # shapely LineString

# TPSCE outlines
box = (66.0, 25.0, 106.0, 40.0)
poly_tpsce = utilities.grid.polygon_rectangular(box, spacing=0.01)

# Compute intersection region
poly_inters = Polygon(poly_tb).intersection(poly_tpsce)
crs_geo = ccrs.PlateCarree()

labels = {"ET": {"pos": (86.5, 37.0), "color": "black"},
          "HM": {"pos": (94.8, 31.7), "color": "black"},
          "HMC": {"pos": (101.5, 32.5), "color": "red"},
          "HMUS": {"pos": (95.8, 21.8), "color": "red"},
          "HMUN": {"pos": (94.9, 26.2), "color": "red"}}

# Test plot
plt.figure(dpi=150)
ax = plt.axes()
plt.plot(*poly_tb.coords.xy, color="blue")
plt.plot(*poly_tpsce.exterior.coords.xy, color="red")
plt.plot(*poly_inters.exterior.coords.xy, color="black")
plt.show()

# %%
# Transform intersection polygon to rotated latitude/longitude coordinates
project = Transformer.from_crs(CRS.from_user_input(crs_geo),
                               CRS.from_user_input(crs_rot_pole),
                               always_xy=True).transform
poly_inters_rot = transform(project, poly_inters)

# Compute intersections between region polygons and area covered by
# 'HMA_SR_D' product
regions["ET_snow"] = regions["ET"].intersection(poly_inters_rot)
regions["HM_snow"] = regions["HM"].intersection(poly_inters_rot)

# Test plot
left, bottom, right, top = 0.08, 0.13, 0.99, 0.99
gs = gridspec.GridSpec(1, 1, left=left, bottom=bottom, right=right, top=top)
fig = plt.figure(figsize=(9, 8.3))
ax = fig.add_subplot(gs[0], projection=crs_rot_pole)

levels = np.arange(0., 6500.0, 500.0)
ticks = np.arange(0., 6500.0, 1000.0)
cmap = truncate_colormap(cmc.bukavu, 0.55, 1.0)
cmap.set_bad(color="skyblue")
# cmap = cmc.lapaz_r
norm = BoundaryNorm(levels, ncolors=cmap.N, extend="max")
cs = ax.pcolormesh(rlon, rlat, topo, cmap=cmap, norm=norm, shading="auto", transform=crs_rot_pole)
ax.add_feature(cfeature.RIVERS, alpha=0.5)
ax.add_feature(cfeature.OCEAN, zorder=1)

poly_plot = PolygonPatch(poly_inters_rot, facecolor="none",
                         edgecolor="black", lw=1.0, ls="-",
                         transform=crs_rot_pole)
ax.add_patch(poly_plot)
for i in ("ET", "HM"):
    poly_plot = PolygonPatch(regions[i], facecolor="none",
                             edgecolor="black", lw=1.0, ls="-",
                             transform=crs_rot_pole)
    ax.add_patch(poly_plot)
poly_plot = PolygonPatch(regions["ET_snow"], facecolor="none",
                         edgecolor="red", lw=2.5, ls="-",
                         transform=crs_rot_pole)
ax.add_patch(poly_plot)
poly_plot = PolygonPatch(regions["HM_snow"], facecolor="none",
                         edgecolor="blue", lw=2.5, ls="--",
                         transform=crs_rot_pole)
ax.add_patch(poly_plot)
ax.axis((-28.0, -2.0, -8.0, 13.0))

for i in ["ET", "HM"]:
    ax.text(*labels[i]["pos"], i, color=labels[i]["color"], fontsize=17, transform=ccrs.PlateCarree(), zorder=200, weight='bold')
    ax.text(*labels[i]["pos"], i, color=labels[i]["color"], fontsize=17, transform=ccrs.PlateCarree(), zorder=200, weight='bold')

cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.08, ax.get_position().width, 0.025])
cbar = fig.colorbar(cs, cax=cax, orientation='horizontal', extend='max', ticks=ticks, label="[m]")
cbar.ax.tick_params(labelsize=14)

gl = ax.gridlines(draw_labels=False, linewidth=1,
                             color='grey', alpha=0.9, linestyle='--', zorder=101)
gl.xlocator = mticker.FixedLocator([85, 90, 95, 100, 105, 110])
gl.ylocator = mticker.FixedLocator([20, 25, 30, 35, 40])

ax.text(-0.008, 0.85, '35°N', ha='right', va='center', transform=ax.transAxes, fontsize=17)
ax.text(-0.008, 0.61, '30°N', ha='right', va='center', transform=ax.transAxes, fontsize=17)
ax.text(-0.008, 0.36, '25°N', ha='right', va='center', transform=ax.transAxes, fontsize=17)
ax.text(-0.008, 0.12, '20°N', ha='right', va='center', transform=ax.transAxes, fontsize=17)

ax.text(0.10, -0.02, '90°E', ha='center', va='top', transform=ax.transAxes, fontsize=17)
ax.text(0.29, -0.02, '95°E', ha='center', va='top', transform=ax.transAxes, fontsize=17)
ax.text(0.48, -0.02, '100°E', ha='center', va='top', transform=ax.transAxes, fontsize=17)
ax.text(0.67, -0.02, '105°E', ha='center', va='top', transform=ax.transAxes, fontsize=17)
ax.text(0.86, -0.02, '110°E', ha='center', va='top', transform=ax.transAxes, fontsize=17)


plt.show()

plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + 'sim_domain_snow.png', dpi=500, transparent=True)

# # Save region polygons to disk
# for i in ["ET_snow", "HM_snow"]:
#     file = path_masks + "snow/region_polygons/" + i + "_rot_coord.poly"
#     with open(file, "wb") as file:
#         pickle.dump(regions[i], file, pickle.HIGHEST_PROTOCOL)

# ###############################################################################
# # Compute region masks for different products
# ###############################################################################
#
# # List of products (with file and geographic reference)
# crs_ims = ccrs.Stereographic(central_latitude=90.0, central_longitude=-80.0,
#                              false_easting=0.0, false_northing=0.0,
#                              true_scale_latitude=60.0, globe=None)
# products = {
#     # -------------------------------------------------------------------------
#     "ECHAM5": {"file": "/project/pr133/rxiang/data/echam5_raw/PI/input/"
#                        + "T159_jan_surf.nc",
#                "geo_ref": CRS.from_epsg(4326),
#                "atol": 1e-02},
#     "ERA5": {"file": "/project/pr133/csteger/Data/Observations/ERA5/Data_raw/"
#                      + "ERA5_snow_2001_cp.nc",
#              "geo_ref": CRS.from_epsg(4326),
#              "atol": 1e-04},
#     "IMS": {"file": "/project/pr133/csteger/Data/Observations/IMS/Processed/"
#                     + "24km/IMS_24km_sc_2001.nc",
#             "geo_ref": CRS.from_user_input(crs_ims),
#             "atol": 10.0},
#     "TPSCE": {"file": "/project/pr133/csteger/Data/Observations/TPSCE/"
#                       + "Processed/TPSCE_sc_2001.nc",
#               "geo_ref": CRS.from_epsg(4326),
#               "atol": 1e-04},
#     "ESA-CCI-AVHRR": {"file": "/project/pr133/csteger/Data/Observations/"
#                               + "ESA-CCI/Processed/BECCY/"
#                               + "2001-ESACCI-L3C_SNOW-SCFG-AVHRR_MERGED-fv2.0"
#                               + "_gap_filled.nc",
#                       "geo_ref": CRS.from_epsg(4326),
#                       "atol": 1e-04},
#     "ESA-CCI-MODIS": {"file": "/project/pr133/csteger/Data/Observations/"
#                               + "ESA-CCI/Processed/BECCY/"
#                               + "2001-ESACCI-L3C_SNOW-SCFG-MODIS_TERRA-fv2.0"
#                               + "_gap_filled.nc",
#                       "geo_ref": CRS.from_epsg(4326),
#                       "atol": 1e-04},
#     # -------------------------------------------------------------------------
#     "CTRL11": {"file": "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/mon/"
#                        + "TOT_PREC/2001-2005.TOT_PREC.nc",
#                "geo_ref": CRS.from_user_input(crs_rot_pole),
#                "atol": 1e-04},
#     # -------------------------------------------------------------------------
#     }
#
# # Full names of regions
# regions_fn = {"ET_snow": "Eastern Tibet (snow)",
#               "HM_snow": "Hengduan Mountains (snow)"}
#
# # Loop through products
# for i in products.keys():
#
#     # Load spatial coordinates
#     ds = xr.open_dataset(products[i]["file"])
#     if "rlon" in list(ds.coords):
#         x = ds["rlon"].values
#         y = ds["rlat"].values
#     elif "lon" in list(ds.coords):
#         x = ds["lon"].values
#         y = ds["lat"].values
#     elif "longitude" in list(ds.coords):
#         x = ds["longitude"].values
#         y = ds["latitude"].values
#     elif "x" in list(ds.coords):
#         x = ds["x"].values
#         y = ds["y"].values
#     else:
#         raise ValueError("Unknown spatial coordinates")
#
#     # Transform region polygons and compute (binary) masks
#     project = Transformer.from_crs(CRS.from_user_input(crs_rot_pole),
#                                    products[i]["geo_ref"],
#                                    always_xy=True).transform
#     x_edge, y_edge = np.meshgrid(
#         *utilities.grid.coord_edges(x, y, atol=products[i]["atol"]))
#     region_masks = {}
#     for j in ["ET_snow", "HM_snow"]:
#         region_trans = transform(project, regions[j])
#         area_frac = utilities.grid.polygon_inters_exact(
#             x_edge, y_edge, region_trans,
#             agg_cells=np.array([10, 5, 2]))
#         region_masks[j] = (area_frac > 0.5)
#
#     # Save to NetCDF file
#     ncfile = Dataset(filename=path_masks + "snow/" + i + "_region_masks.nc",
#                      mode="w", format="NETCDF4")
#     ncfile.createDimension(dimname="y", size=area_frac.shape[0])
#     ncfile.createDimension(dimname="x", size=area_frac.shape[1])
#     for j in region_masks.keys():
#         nc_data = ncfile.createVariable(varname=j, datatype="b",
#                                         dimensions=("y", "x"))
#         nc_data.units = "-"
#         nc_data.long_name = regions_fn[j]
#         nc_data[:] = region_masks[j]
#     ncfile.close()
#
#     print("File " + i + "_region_masks.nc created")
