# Description: Compare snow cover characteristics of COSMO/ECHAM5 simulation
#              with other data (ECHAM5, ERA5, TPSCE, IMS, ESA-CCI)
#
# Author: Christian Steger, September 2023

# Load modules
import os
import sys
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import colors as col
from cmcrameri import cm
import glob
from dask.diagnostics import ProgressBar
# import subprocess
# from utilities import remap
import matplotlib.ticker as mticker
import cmcrameri.cm as cmc
import numpy.ma as ma
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

mpl.style.use("classic")

# Paths to folders
path_temp = "/project/pr133/rxiang/data/BECCY_snow/"
path_cosmo = "/project/pr133/rxiang/data/cosmo/"
path_echam5 = "/project/pr133/rxiang/data/echam5_raw/"
path_era5 = "/project/pr133/csteger/Data/Observations/ERA5/Data_raw/"
path_tpsce = "/project/pr133/csteger/Data/Observations/TPSCE/Processed/"
path_ims = "/project/pr133/csteger/Data/Observations/IMS/Processed/24km/"
path_esa_cci = "/project/pr133/csteger/Data/Observations/ESA-CCI/" \
               + "Processed/BECCY/"


###############################################################################
# Functions
###############################################################################

# ECHAM5 snow cover fraction parameterisation (ECHAM5, Part 1, p. 44)
def scf_echam5(h_sn, sigma_z, epsilon=1e-08):
    """
    h_sn:    snow depth (snow water equivalent) [m]
    sigma_z: subgrid-scale standard deviation of height [m]
    epsilon: small number to avoid division by zero [-]
    scf:     snow cover fraction [-]
    """
    gamma_1 = 0.95
    gamma_2 = 0.15
    scf = gamma_1 * np.tanh(100.0 * h_sn) \
        * np.sqrt((1000.0 * h_sn)
                  / (1000.0 * h_sn + gamma_2 * sigma_z + epsilon))
    return scf


# Test plot
h_sn = np.linspace(0.0, 0.4, 200)
sigma_z = np.linspace(0.0, 700.0, 200)
h_sn, sigma_z = np.meshgrid(h_sn, sigma_z)
plt.figure(dpi=150)
plt.pcolormesh(h_sn, sigma_z, scf_echam5(h_sn, sigma_z))
plt.colorbar()
plt.show()

###############################################################################
# Remap reference data to COSMO grid (if required)
###############################################################################

# # Get target grid information
# ds = xr.open_dataset(path_cosmo + "EAS04_ctrl/24h/W_SNOW/01-05_W_SNOW_mt.nc")
# rlon_cent_in = ds["rlon"].values
# rlat_cent_in = ds["rlat"].values
# grid_mapping_name = ds["rotated_pole"].grid_mapping_name
# pole_longitude = ds["rotated_pole"].grid_north_pole_longitude
# pole_latitude = ds["rotated_pole"].grid_north_pole_latitude
# ds.close()
#
# # Write grid description file for source grid
# file_txt = path_temp + "grid_target_rot.txt"
# remap.grid_desc(file_txt, gridtype="projection",
#                 xsize=len(rlon_cent_in), ysize=len(rlat_cent_in),
#                 xfirst=rlon_cent_in[0], yfirst=rlat_cent_in[0],
#                 xinc=np.diff(rlon_cent_in).mean(),
#                 yinc=np.diff(rlat_cent_in).mean(),
#                 grid_mapping_name="rotated_latitude_longitude",
#                 grid_north_pole_longitude=pole_longitude,
#                 grid_north_pole_latitude=pole_latitude)
#
# # Remap TPSCE to COSMO-04
# method = "remapnn"  # "remapbil", "remapcon", "remapnn"
# # export REMAP_EXTRAPOLATE='off'
# cmd = "cdo " + method + "," + file_txt
# sf = path_tpsce + "TPSCE_sc_2005.nc"
# tf = path_temp + "TPSCE_sc_2005_" + method + ".nc"
# print(cmd + " " + sf + " " + tf)
# # subprocess.call(cmd + " " + sf + " " + tf, shell=True)

###############################################################################
# Load and plot snow cover duration (SCD)
###############################################################################

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

data_scd = {}

# TPSCE (-> binary snow coverage)
ds = xr.open_mfdataset(path_tpsce + "TPSCE_sc_200?.nc")
ds = ds.sel(time=slice("2001-01-01", "2005-12-31"))
ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
print(len(ds["time"]) / 5)
lon_tpsce = ds["lon"].values
lat_tpsce = ds["lat"].values
data_scd["TPSCE"] = {"scd": ds["sc"].values.sum(axis=0) / 5.0,
                     "x": ds["lon"].values, "y": ds["lat"].values,
                     "crs": ccrs.PlateCarree()}
ds.close()

# IMS (-> binary snow coverage)
ds = xr.open_mfdataset(path_ims + "IMS_24km_sc_200?.nc")
ds = ds.sel(time=slice("2001-01-01", "2005-12-31"))
ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
print(len(ds["time"]) / 5)
ds = ds.isel(y=slice(0, 500))
crs_ims = ccrs.Stereographic(central_latitude=90.0, central_longitude=-80.0,
                             false_easting=0.0, false_northing=0.0,
                             true_scale_latitude=60.0, globe=None)
data_scd["IMS"] = {"scd": ds["sc"].values.sum(axis=0) / 5.0,
                   "x": ds["x"].values, "y": ds["y"].values,
                   "crs": crs_ims}
ds.close()

# ERA5
ds = xr.open_mfdataset(path_era5 + "ERA5_snow_200?_cp.nc")
# ds = ds.isel(latitude=slice(100, 221), longitude=slice(350, 601))
ds = ds.sel(time=slice("2001-01-01", "2005-12-31"))
ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
print(len(ds["time"]) / 5)
# -----------------------------------------------------------------------------
dens_water = 1000.0  # density of water [kg m-3]
# rsn:  snow density [kg m-3]
# sd:  snow depth [m swe]
scf = np.minimum(1.0, (dens_water * ds["sd"] / ds["rsn"]) / 0.1)  # [-]
# Source:
# https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
# -> Computation of snow cover
sc = (scf >= 0.5).astype(np.float32)  # binary snow cover [-]
# -----------------------------------------------------------------------------
data_scd["ERA5"] = {"scd": sc.sum(axis=0) / 5.0,
                    "x": ds["longitude"].values, "y": ds["latitude"].values,
                    "crs": ccrs.PlateCarree()}
ds.close()

# COSMO 0.11 / 0.04 deg
cosmo_run = {"COSMO CTRL": path_cosmo + "EAS11_ctrl/24h/W_SNOW/0?_W_SNOW.nc",
             "cosmo11lgm": path_cosmo + "EAS11_lgm/24h/W_SNOW/0?_W_SNOW.nc"}
for i in cosmo_run.keys():
    ds = xr.open_mfdataset(cosmo_run[i])
    # ds = ds.isel(time=slice(0, -1))
    print(ds["time"][0].values, ds["time"][-1].values)
    _, index = np.unique(ds["time"], return_index=True)
    ds = ds.isel(time=index)
    print(np.all(np.diff(ds["time"]).astype("timedelta64[D]")
                 == np.timedelta64(1, "D")))
    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
    print(len(ds["time"]) / 5)
    # -------------------------------------------------------------------------
    delta_s = 0.015  # [m]
    scf = np.maximum(0.01, np.minimum(1.0, ds["W_SNOW"] / delta_s))
    # Source: COSMO documentation (version 6.0) - Part 2 - p. 121
    sc = (scf >= 0.5).astype(np.float32)  # binary snow cover [-]
    # -------------------------------------------------------------------------
    crs_cosmo = ccrs.RotatedPole(
        pole_latitude=ds["rotated_pole"].grid_north_pole_latitude,
        pole_longitude=ds["rotated_pole"].grid_north_pole_longitude)
    data_scd[i] = {"scd": sc.sum(axis=0) / 5.0,
                   "x": ds["rlon"].values, "y": ds["rlat"].values,
                   "crs": crs_cosmo}
    ds.close()

# # ECHAM5 (-> save snow variable in separate file)
# files = sorted(glob.glob(path_echam5
#                          + "PI/output_raw/e007_2_101[3-7]??.01.nc"))
# ds = xr.open_mfdataset(files, decode_times=False)
# ds["sn"].to_netcdf(path_temp + "ECHAM5 PI_sn_5years.nc")
# files = sorted(glob.glob(path_echam5
#                          + "LGM/output_raw/e009_101[4-8]??.01.nc"))
# ds = xr.open_mfdataset(files, decode_times=False)
# ds["sn"].to_netcdf(path_temp + "ECHAM5_LGM_sn_5years.nc")

# ECHAM5
for i in ("ECHAM5 PI", "echam5_lgm"):
    if i == "ECHAM5 PI":
        ds = xr.open_mfdataset(path_echam5 + "PI/input/T159_jan_surf.nc")
        oro_std = ds["OROSTD"].values  # [m]
        ds.close()
        ds = xr.open_dataset(path_temp + "ECHAM5_PI_sn_5years.nc")
        ds = ds.sel(time=~((ds["time.month"] == 2) & (ds["time.day"] == 29)))
        print(len(ds["time"]) / 5)
        sn = ds["sn"].values  # [m]
        ds.close()
    else:
        ds = xr.open_mfdataset(path_echam5
                               + "LGM/input/T159_jan_surf.lgm.veg.nc")
        oro_std = ds["OROSTD"].values  # [m]
        ds.close()
        ds = xr.open_dataset(path_temp + "ECHAM5_LGM_sn_5years.nc")
        ds = ds.sel(time=~((ds["time.month"] == 2) & (ds["time.day"] == 29)))
        print(len(ds["time"]) / 5)
        sn = ds["sn"].values  # [m]
        ds.close()
    sc = (scf_echam5(sn, oro_std) > 0.5).astype(np.float32)
    # binary snow cover [-]
    data_scd[i] = {"scd": sc.sum(axis=0) / 5.0,
                   "x": ds["lon"].values, "y": ds["lat"].values,
                   "crs": ccrs.PlateCarree()}
    ds.close()

# ESA-CCI (AVHRR; ~5 km resolution, MODIS: ~1 km resolution)
for i in ("AVHRR_MERGED", "MODIS_TERRA"):
    file_map = path_temp + i + "_map.npy"
    da = xr.open_mfdataset(path_esa_cci + "????-ESACCI-L3C_SNOW-SCFG-"
                           + i + "-fv2.0_gap_filled.nc",
                           chunks={"lat": 500, "lon": 500})["scfg"]
    # -------------------------------------------------------------------------
    # Time-consuming -> only compute once and subsequently reload from disk
    # -------------------------------------------------------------------------
    if not os.path.isfile(file_map):
        da = da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))
        print(len(da["time"]) / 5)
        with ProgressBar():
            num_value = np.isfinite(da).sum(dim="time").values
            scd = da.sum(dim="time", skipna=True).values
        scd_365 = (scd / num_value) * 365.0
        np.save(file_map, scd_365)
    else:
        scd_365 = np.load(file_map)
    # -------------------------------------------------------------------------
    ind = "ESA-CCI_" + i.split("_")[0].lower()
    data_scd[ind] = {"scd": scd_365,
                     "x": da["lon"].values,
                     "y": da["lat"].values,
                     "crs": ccrs.PlateCarree()}
    da.close()

#%%
# -----------------------------------------------------------------------------
# Plot data
# -----------------------------------------------------------------------------

# Average ESA-CCI products (AVHRR and MODIS)
if not np.all(np.array(data_scd["ESA-CCI_modis"]["scd"].shape) / 5.0
              == np.array(data_scd["ESA-CCI_avhrr"]["scd"].shape)):
    raise ValueError("ESA-CCI products have incompatible array sizes")
shp = data_scd["ESA-CCI_avhrr"]["scd"].shape
scd_modis_agg = np.empty(shp, dtype=np.float32)
for i in range(shp[0]):
    for j in range(shp[1]):
        slic = (slice(i * 5, ((i + 1) * 5)),
                slice(j * 5, ((j + 1) * 5)))
        scd_modis_agg[i, j] \
            = np.nanmean(data_scd["ESA-CCI_modis"]["scd"][slic])
data_scd["ESA-CCI"] = {"scd": (data_scd["ESA-CCI_avhrr"]["scd"]
                               + scd_modis_agg) / 2.0,
                       "x": data_scd["ESA-CCI_avhrr"]["x"],
                       "y": data_scd["ESA-CCI_avhrr"]["y"],
                       "crs": ccrs.PlateCarree()}

prod = ["COSMO CTRL", "ECHAM5 PI", "ERA5", "ESA-CCI", "IMS", "TPSCE"]
for i in prod:
    data_scd[i]["scd"] = np.ma.masked_where(data_scd[i]["scd"] < 1, data_scd[i]["scd"])


# Colormap
# levels = np.append(np.array([5.0]), np.arange(25.0, 275.0, 25.0))
# cmap = cm.lapaz_r
# # cmap = cm.batlowW_r
# norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="both")

levels = MaxNLocator(nbins=100).tick_values(0, 250)
cmap = cmc.roma_r
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# %%
# Asia
prod = ["COSMO CTRL", "ECHAM5 PI", "ERA5", "IMS"]  # ~present-day
index = ['a', 'b', 'c', 'd']
# prod = ["echam5_lgm", "cosmo11lgm"]  # LGM
fig = plt.figure(figsize=(9, 6), dpi=150)
gs = gridspec.GridSpec(3, 2, left=0.06, bottom=0.065, right=0.99, top=0.95,
                       hspace=0.18, wspace=0.05,
                       height_ratios=[1.0, 1.0, 0.06])
for ind_i, i in enumerate(prod):
    ax = plt.subplot(gs[ind_i], projection=data_scd["COSMO CTRL"]["crs"])
    ax.set_facecolor((0.76, 0.76, 0.76))
    cs = plt.pcolormesh(data_scd[i]["x"], data_scd[i]["y"], data_scd[i]["scd"],
                   transform=data_scd[i]["crs"], cmap=cmap, norm=norm)
    ax.add_feature(feature.BORDERS, linestyle="-", linewidth=0.6)
    ax.add_feature(feature.COASTLINE, linestyle="-", linewidth=0.6)
    # ax.set_aspect("auto")
    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.7, linestyle='--')
    gl.xlocator = mticker.FixedLocator([60, 80, 100, 120, 140, 160, 180])
    gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])
    ax.set_extent([-52.0, 34.0, -2.0, 40.0], crs=data_scd["COSMO CTRL"]["crs"])
    if i == "COSMO CTRL" or i == "ERA5":
        ax.text(-0.008, 0.94, '50°N', ha='right', va='center', transform=ax.transAxes, fontsize=12)
        ax.text(-0.008, 0.66, '40°N', ha='right', va='center', transform=ax.transAxes, fontsize=12)
        ax.text(-0.008, 0.38, '30°N', ha='right', va='center', transform=ax.transAxes, fontsize=12)
        ax.text(-0.008, 0.10, '20°N', ha='right', va='center', transform=ax.transAxes, fontsize=12)
    if i == "ERA5" or i == "IMS":
        ax.text(0.22, -0.02, '100°E', ha='center', va='top', transform=ax.transAxes, fontsize=12)
        ax.text(0.44, -0.02, '120°E', ha='center', va='top', transform=ax.transAxes, fontsize=12)
        ax.text(0.66, -0.02, '140°E', ha='center', va='top', transform=ax.transAxes, fontsize=12)
        ax.text(0.88, -0.02, '160°E', ha='center', va='top', transform=ax.transAxes, fontsize=12)

    id = index[ind_i]
    t = ax.text(0.007, 0.988, f'({id})', ha='left', va='top',
                       transform=ax.transAxes, fontsize=12)
    t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

    plt.title(i, fontsize=12, fontweight="bold", y=1.002)

# -----------------------------------------------------------------------------
# ax = plt.subplot(gs[-1, 0])
# cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, ticks=levels,
#                                orientation="horizontal")
# ax = plt.subplot(gs[-1, -1])
cax = fig.add_axes([ax.get_position().x0-0.238, ax.get_position().y0-0.06, ax.get_position().width, 0.02])
cb = fig.colorbar(cs, cax=cax, ticks=[0, 50, 100, 150, 200, 250], orientation="horizontal", extend='max')
cb.ax.tick_params(labelsize=11)
cb.ax.minorticks_off()
plt.xlabel("Snow cover duration [days]", fontsize=11)
# -----------------------------------------------------------------------------
plt.show()
plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + 'snow_asia.png', dpi=500, transparent='True')
plt.close(fig)

# %%
# Eastern Tibetan Plateau
# plt.rcParams['axes.facecolor'] = 'lightgrey'
prod = ["COSMO CTRL", "ECHAM5 PI", "ERA5",
        "ESA-CCI", "IMS", "TPSCE"]
index = ['a', 'b', 'c', 'd', 'e', 'f']
# prod = ["echam5_lgm", "cosmo11lgm"]  # LGM
fig = plt.figure(figsize=(10, 6.0), dpi=150)  # width, height
gs = gridspec.GridSpec(3, 3, left=0.048, bottom=0.06, right=0.988, top=0.95,
                       hspace=0.18, wspace=0.05,
                       height_ratios=[1.0, 1.0, 0.08])
for ind_i, i in enumerate(prod):
    ax = plt.subplot(gs[ind_i], projection=data_scd["COSMO CTRL"]["crs"])
    ax.set_facecolor((0.76, 0.76, 0.76))
    plt.pcolormesh(data_scd[i]["x"], data_scd[i]["y"], data_scd[i]["scd"],
                   transform=data_scd[i]["crs"], cmap=cmap, norm=norm)
    ax.add_feature(feature.BORDERS, linestyle="-", linewidth=0.6)
    ax.add_feature(feature.COASTLINE, linestyle="-", linewidth=0.6)
    # ax.set_aspect("auto")
    ax.set_extent([-26.5, -9.0, -2.0, 14.0], crs=data_scd["COSMO CTRL"]["crs"])
    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([85, 90, 95, 100, 105])
    gl.ylocator = mticker.FixedLocator([20, 25, 30, 35, 40])

    if i == "COSMO CTRL" or i == "ESA-CCI":
        ax.text(-0.01, 0.72, '35°N', ha='right', va='center', transform=ax.transAxes, fontsize=12)
        ax.text(-0.01, 0.40, '30°N', ha='right', va='center', transform=ax.transAxes, fontsize=12)
        ax.text(-0.01, 0.08, '25°N', ha='right', va='center', transform=ax.transAxes, fontsize=12)
    if i == "ESA-CCI" or i == "IMS" or i == "TPSCE":
        ax.text(0.15, -0.02, '90°E', ha='center', va='top', transform=ax.transAxes, fontsize=12)
        ax.text(0.40, -0.02, '95°E', ha='center', va='top', transform=ax.transAxes, fontsize=12)
        ax.text(0.70, -0.02, '100°E', ha='center', va='top', transform=ax.transAxes, fontsize=12)
        ax.text(0.95, -0.02, '105°E', ha='center', va='top', transform=ax.transAxes, fontsize=12)

    plt.title(i, fontsize=12, fontweight="bold", y=1.002)

    if i == "IMS":
        cax = fig.add_axes([ax.get_position().x0-0.075, ax.get_position().y0 - 0.06, ax.get_position().width*1.5, 0.02])
        cb = fig.colorbar(cs, cax=cax, ticks=[0, 50, 100, 150, 200, 250], orientation="horizontal", extend='max')
        cb.ax.tick_params(labelsize=11)
        cb.ax.minorticks_off()
        plt.xlabel("Snow cover duration [days]", fontsize=11)

    id = index[ind_i]
    t = ax.text(0.007, 0.988, f'({id})', ha='left', va='top',
                transform=ax.transAxes, fontsize=12)
    t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
plt.show()
plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + 'snow_etp.png', dpi=500, transparent='True')
plt.close(fig)
#%%
# %%
# Eastern Tibetan Plateau
plt.rcParams['axes.facecolor'] = 'lightgrey'
prod = ["COSMO CTRL", "ECHAM5 PI", "ERA5",
        "ESA-CCI", "IMS", "TPSCE"]
index = ['a', 'b', 'c', 'd', 'e', 'f']
# prod = ["echam5_lgm", "cosmo11lgm"]  # LGM
fig = plt.figure(figsize=(10, 4.4), dpi=150)  # width, height
gs = gridspec.GridSpec(3, 3, left=0.048, bottom=0.1, right=0.988, top=0.95,
                       hspace=0.18, wspace=0.05,
                       height_ratios=[1.0, 1.0, 0.08])
for ind_i, i in enumerate(prod):
    ax = plt.subplot(gs[ind_i], projection=data_scd["COSMO CTRL"]["crs"])
    ax.set_facecolor((0.76, 0.76, 0.76))
    cs = plt.pcolormesh(data_scd[i]["x"], data_scd[i]["y"], data_scd[i]["scd"],
                   transform=data_scd[i]["crs"], cmap=cmap, norm=norm)
    ax.add_feature(feature.BORDERS, linestyle="-", linewidth=0.6)
    ax.add_feature(feature.COASTLINE, linestyle="-", linewidth=0.6)
    # ax.set_aspect("auto")
    ax.set_extent([-52.0, 34.0, -2.0, 40.0], crs=data_scd["COSMO CTRL"]["crs"])
    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([60, 80, 100, 120, 140, 160, 180])
    gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])

    if i == "COSMO CTRL" or i == "ESA-CCI":
        ax.text(-0.008, 0.94, '50°N', ha='right', va='center', transform=ax.transAxes, fontsize=12)
        ax.text(-0.008, 0.66, '40°N', ha='right', va='center', transform=ax.transAxes, fontsize=12)
        ax.text(-0.008, 0.38, '30°N', ha='right', va='center', transform=ax.transAxes, fontsize=12)
        ax.text(-0.008, 0.10, '20°N', ha='right', va='center', transform=ax.transAxes, fontsize=12)
    if i == "ESA-CCI" or i == "IMS" or i == "TPSCE":
        ax.text(0.22, -0.02, '100°E', ha='center', va='top', transform=ax.transAxes, fontsize=12)
        ax.text(0.44, -0.02, '120°E', ha='center', va='top', transform=ax.transAxes, fontsize=12)
        ax.text(0.66, -0.02, '140°E', ha='center', va='top', transform=ax.transAxes, fontsize=12)
        ax.text(0.88, -0.02, '160°E', ha='center', va='top', transform=ax.transAxes, fontsize=12)

    plt.title(i, fontsize=12, fontweight="bold", y=1.002)

    if i == "IMS":
        cax = fig.add_axes([ax.get_position().x0-0.075, ax.get_position().y0 - 0.08, ax.get_position().width*1.5, 0.03])
        cb = fig.colorbar(cs, cax=cax, ticks=[0, 50, 100, 150, 200, 250], orientation="horizontal", extend='max')
        cb.ax.tick_params(labelsize=11)
        cb.ax.minorticks_off()
        plt.xlabel("Snow cover duration [days]", fontsize=11)

    id = index[ind_i]
    t = ax.text(0.007, 0.988, f'({id})', ha='left', va='top',
                transform=ax.transAxes, fontsize=12)
    t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
plt.show()
plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + 'snow_validation.png', dpi=500, transparent='True')
plt.close(fig)
# %%
###############################################################################
# Load and plot snow water equivalent (SWE)
###############################################################################


# Winter definition (November - June; 8 months)
def winter(month):
    return (month >= 11) | (month <= 6)


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

data_swe = {}

# ERA5
ds = xr.open_mfdataset(path_era5 + "ERA5_snow_200?_cp.nc")
# ds = ds.isel(latitude=slice(100, 221), longitude=slice(350, 601))
ds = ds.sel(time=slice("2001-01-01", "2005-12-31"))
# ------------ glacier mask (-> perennial snow accumulation) ------------------
mask_glacier = np.all(ds["sd"].values > 0.01, axis=0)
# -----------------------------------------------------------------------------
ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
print(len(ds["time"]) / 5)
ds = ds.sel(time=winter(ds["time.month"]))
swe = ds["sd"].mean(dim="time").values * 1000.0  # [mm]
data_swe["ERA5"] = {"swe": swe,
                    "mask_glacier": mask_glacier,
                    "x": ds["longitude"].values, "y": ds["latitude"].values,
                    "crs": ccrs.PlateCarree()}
ds.close()

# COSMO 0.11 / 0.04 deg
cosmo_run = {"COSMO CTRL": path_cosmo + "EAS11_ctrl/24h/W_SNOW/0?_W_SNOW.nc",
             "cosmo11lgm": path_cosmo + "EAS11_lgm/24h/W_SNOW/0?_W_SNOW.nc"}
for i in cosmo_run.keys():
    ds = xr.open_mfdataset(cosmo_run[i])
    print(ds["time"][0].values, ds["time"][-1].values)
    _, index = np.unique(ds["time"], return_index=True)
    ds = ds.isel(time=index)
    print(np.all(np.diff(ds["time"]).astype("timedelta64[D]")
                 == np.timedelta64(1, "D")))
    # ------------ glacier mask (-> perennial snow accumulation) --------------
    mask_glacier = np.all(ds["W_SNOW"].values > 0.01, axis=0)
    # -------------------------------------------------------------------------
    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
    print(len(ds["time"]) / 5)
    ds = ds.sel(time=winter(ds["time.month"]))
    swe = ds["W_SNOW"].values  # [m]
    swe = swe.mean(axis=0) * 1000.0  # [mm]
    crs_cosmo = ccrs.RotatedPole(
        pole_latitude=ds["rotated_pole"].grid_north_pole_latitude,
        pole_longitude=ds["rotated_pole"].grid_north_pole_longitude)
    data_swe[i] = {"swe": swe,
                   "mask_glacier": mask_glacier,
                   "x": ds["rlon"].values, "y": ds["rlat"].values,
                   "crs": crs_cosmo}
    ds.close()

# ECHAM5
for i in ("ECHAM5 PI", "echam5_lgm"):
    if i == "ECHAM5 PI":
        ds = xr.open_dataset(path_temp + "ECHAM5 PI_sn_5years.nc")
    else:
        ds = xr.open_dataset(path_temp + "ECHAM5_LGM_sn_5years.nc")
    # ------------ glacier mask (-> perennial snow accumulation) --------------
    mask_glacier = np.all(ds["sn"].values > 0.01, axis=0)
    # -------------------------------------------------------------------------
    ds = ds.sel(time=~((ds["time.month"] == 2) & (ds["time.day"] == 29)))
    print(len(ds["time"]) / 5)
    ds = ds.sel(time=winter(ds["time.month"]))
    swe = ds["sn"].mean(dim="time").values * 1000.0  # [mm]
    ds.close()
    data_swe[i] = {"swe": swe,
                   "mask_glacier": mask_glacier,
                   "x": ds["lon"].values, "y": ds["lat"].values,
                   "crs": ccrs.PlateCarree()}
    ds.close()
# Notes:
# - ECHAM5 simulations do not exhibit any 'glaciated' grid cells

# -----------------------------------------------------------------------------
# Plot data
# -----------------------------------------------------------------------------

# Colormap
levels = np.arange(0.0, 85.0, 5.0)
ticks = np.arange(0.0, 90.0, 10.0)
cmap = cm.oslo_r
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")
cmap_glacier = mpl.colors.ListedColormap(["darkorange"])
norm_glacier = mpl.colors.BoundaryNorm([0.5, 1.5], cmap_glacier.N)

# Eastern Tibetan Plateau
prod = ["ECHAM5 PI", "ERA5", "COSMO CTRL"]
# prod = ["echam5_lgm", "cosmo11lgm"]  # LGM
fig = plt.figure(figsize=(8, 6.0), dpi=150)  # width, height
gs = gridspec.GridSpec(3, 2, left=0.1, bottom=0.1, right=0.9, top=0.9,
                       hspace=0.16, wspace=0.05,
                       height_ratios=[1.0, 1.0, 0.08])
for ind_i, i in enumerate(prod):
    ax = plt.subplot(gs[ind_i], projection=data_swe["COSMO CTRL"]["crs"])
    ax.set_facecolor((0.76, 0.76, 0.76))
    plt.pcolormesh(data_swe[i]["x"], data_swe[i]["y"], data_swe[i]["swe"],
                   transform=data_swe[i]["crs"], cmap=cmap, norm=norm)
    data_plot = np.ma.masked_where(~data_swe[i]["mask_glacier"],
                                   data_swe[i]["mask_glacier"].astype(float))
    plt.pcolormesh(data_swe[i]["x"], data_swe[i]["y"], data_plot,
                   transform=data_swe[i]["crs"],
                   cmap=cmap_glacier, norm=norm_glacier)
    ax.add_feature(feature.BORDERS, linestyle="-", linewidth=0.6)
    ax.add_feature(feature.COASTLINE, linestyle="-", linewidth=0.6)
    ax.set_aspect("auto")
    ax.set_extent([-26.5, -9.0, -2.0, 14.0], crs=data_scd["COSMO CTRL"]["crs"])
    plt.title(i, fontsize=10, fontweight="bold", y=1.002)
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[-1, :1])
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, ticks=ticks,
                               orientation="horizontal")
cb.ax.tick_params(labelsize=10)
plt.xlabel("Snow water equivalent [mm]", fontsize=10)
# -----------------------------------------------------------------------------
plt.show()
