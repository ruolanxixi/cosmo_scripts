# Description: Compare snow cover characteristics of COSMO/ECHAM5 simulation
#              with other data (ECHAM5, ERA5, TPSCE, IMS, ESA-CCI)
#              -> seasonal cycles
#
# Author: Christian Steger, September 2023

# Load modules
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from scipy import ndimage
import datetime as dt
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.ticker import NullFormatter
import dask
from dask.diagnostics import ProgressBar

mpl.style.use("classic")

# Paths to folders
path_temp = "/scratch/snx3000/csteger/Temp/BECCY_snow/"
path_cosmo = "/project/pr133/rxiang/data/cosmo/"
path_echam5 = "/project/pr133/rxiang/data/echam5_raw/"
path_era5 = "/project/pr133/csteger/Data/Observations/ERA5/Data_raw/"
path_tpsce = "/project/pr133/csteger/Data/Observations/TPSCE/Processed/"
path_ims = "/project/pr133/csteger/Data/Observations/IMS/Processed/24km/"
path_esa_cci = "/project/pr133/csteger/Data/Observations/ESA-CCI/" \
               + "Processed/BECCY/"
path_masks = "/project/pr133/csteger/Data/Model/BECCY/region_masks/snow/"


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


# Moving average (running mean) with periodic boundaries
def moving_average(y, window=15):
    y_ma = ndimage.convolve1d(y, weights=(np.ones(window) / float(window)),
                              mode="wrap")
    return y_ma


###############################################################################
# Load and plot snow cover duration (SCD)
###############################################################################

# Select region
reg = "HM_snow"  # "ET_snow", "HM_snow"

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
dask.config.set({"array.slicing.split_large_chunks": True})

data_scf = {}

# TPSCE (-> binary snow coverage)
ds = xr.open_dataset(path_masks + "TPSCE_region_masks.nc")
ind_0, ind_1 = np.where(ds[reg].values.astype(bool))
ds.close()
ds = xr.open_mfdataset(path_tpsce + "TPSCE_sc_200?.nc")
ds = ds.sel(time=slice("2001-01-01", "2005-12-31"))
ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
print(len(ds["time"]) / 5)
month_day = xr.DataArray(ds.indexes["time"].strftime("%m-%d"),
                         coords=ds["time"].coords, name="month_day")
ds_clim = ds.groupby(month_day).mean("time")
data_scf["tpsce"] = ds_clim["sc"].values[:, ind_0, ind_1].mean(axis=1)
ds.close()

# IMS (-> binary snow coverage)
ds = xr.open_dataset(path_masks + "IMS_region_masks.nc")
ind_0, ind_1 = np.where(ds[reg].values.astype(bool))
ds.close()
ds = xr.open_mfdataset(path_ims + "IMS_24km_sc_200?.nc")
ds = ds.sel(time=slice("2001-01-01", "2005-12-31"))
ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
print(len(ds["time"]) / 5)
month_day = xr.DataArray(ds.indexes["time"].strftime("%m-%d"),
                         coords=ds["time"].coords, name="month_day")
ds_clim = ds.groupby(month_day).mean("time")
data_scf["ims"] = ds_clim["sc"].values[:, ind_0, ind_1].mean(axis=1)
ds.close()

# ERA5
ds = xr.open_dataset(path_masks + "ERA5_region_masks.nc")
ind_0, ind_1 = np.where(ds[reg].values.astype(bool))
ds.close()
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
sc = (scf >= 0.5)  # binary snow cover [-]
# -----------------------------------------------------------------------------
month_day = xr.DataArray(ds.indexes["time"].strftime("%m-%d"),
                         coords=ds["time"].coords, name="month_day")
ds_clim = sc.groupby(month_day).mean("time")
data_scf["era5"] = ds_clim.values[:, ind_0, ind_1].mean(axis=1)
ds.close()

# COSMO 0.11
ds = xr.open_dataset(path_masks + "CTRL11_region_masks.nc")
ind_0, ind_1 = np.where(ds[reg].values.astype(bool))
ds.close()
cosmo_run = {"cosmo11pd": path_cosmo + "EAS11_ctrl/24h/W_SNOW/0?_W_SNOW.nc",
             "cosmo11lgm": path_cosmo + "EAS11_lgm/24h/W_SNOW/0?_W_SNOW.nc"}
for i in cosmo_run.keys():
    ds = xr.open_mfdataset(cosmo_run[i])
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
    sc = (scf >= 0.5)  # binary snow cover [-]
    # -------------------------------------------------------------------------
    month_day = xr.DataArray(ds.indexes["time"].strftime("%m-%d"),
                             coords=ds["time"].coords, name="month_day")
    ds_clim = sc.groupby(month_day).mean("time")
    data_scf[i] = ds_clim.values[:, ind_0, ind_1].mean(axis=1)
    ds.close()

# ECHAM5
ds = xr.open_dataset(path_masks + "ECHAM5_region_masks.nc")
ind_0, ind_1 = np.where(ds[reg].values.astype(bool))
ds.close()
for i in ("echam5_pi", "echam5_lgm"):
    if i == "echam5_pi":
        ds = xr.open_mfdataset(path_echam5 + "PI/input/T159_jan_surf.nc")
        oro_std = ds["OROSTD"]  # [m]
        ds.close()
        ds = xr.open_dataset(path_temp + "ECHAM5_PI_sn_5years.nc")
        ds = ds.sel(time=~((ds["time.month"] == 2) & (ds["time.day"] == 29)))
        print(len(ds["time"]) / 5)
        sn = ds["sn"]  # [m]
        ds.close()
    else:
        ds = xr.open_mfdataset(path_echam5
                               + "LGM/input/T159_jan_surf.lgm.veg.nc")
        oro_std = ds["OROSTD"]  # [m]
        ds.close()
        ds = xr.open_dataset(path_temp + "ECHAM5_LGM_sn_5years.nc")
        ds = ds.sel(time=~((ds["time.month"] == 2) & (ds["time.day"] == 29)))
        print(len(ds["time"]) / 5)
        sn = ds["sn"]  # [m]
        ds.close()
    sn.values = (scf_echam5(sn.values, oro_std.values) > 0.5) \
        .astype(np.float32)
    # binary snow cover [-]
    month_day = xr.DataArray(ds.indexes["time"].strftime("%m-%d"),
                             coords=ds["time"].coords, name="month_day")
    ds_clim = sn.groupby(month_day).mean("time")
    data_scf[i] = ds_clim.values[:, ind_0, ind_1].mean(axis=1)
    ds.close()

# ESA-CCI (AVHRR; ~5 km resolution, MODIS: ~1 km resolution)
for i in ("AVHRR_MERGED", "MODIS_TERRA"):
    file_clim = path_temp + i + "_" + reg + "_clim.npy"
    # -------------------------------------------------------------------------
    # Time-consuming -> only compute once and subsequently reload from disk
    # -------------------------------------------------------------------------
    if not os.path.isfile(file_clim):
        ds = xr.open_dataset(path_masks + "ESA-CCI-" + i.split("_")[0]
                             + "_region_masks.nc")
        ind_0, ind_1 = np.where(ds[reg].values.astype(bool))
        ds.close()
        da = xr.open_mfdataset(path_esa_cci + "????-ESACCI-L3C_SNOW-SCFG-"
                               + i + "-fv2.0_gap_filled.nc",
                               chunks={"lat": 500, "lon": 500})["scfg"]
        da = da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))
        print(len(da["time"]) / 5)
        month_day = xr.DataArray(da.indexes["time"].strftime("%m-%d"),
                                 coords=da["time"].coords, name="month_day")
        with ProgressBar():
            sc_clim = da.groupby(month_day).mean("time", skipna=True).values
        scf = np.nanmean(sc_clim[:, ind_0, ind_1], axis=1)
        np.save(file_clim, scf)
    else:
        scf = np.load(file_clim)
    # ------------------------------------------------------------------------
    ind = "esa-cci_" + i.split("_")[0].lower()
    data_scf[ind] = scf
    ds.close()

# -----------------------------------------------------------------------------
# %% Plot data
# -----------------------------------------------------------------------------
font = {'size': 14}
# Average ESA-CCI products (AVHRR and MODIS)
data_scf["esa-cci"] = (data_scf["esa-cci_avhrr"]
                       + data_scf["esa-cci_modis"]) / 2.0

# Plot (~present-day)
colors = ("#762a83", "#2166ac", "#4393c3", "#b2182b", "#d6604d", "#f4a582")
labels = {"era5": "ERA5", "cosmo11pd": "COSMO", "echam5_pi": "ECHAM5",
          "tpsce": "TPSCE", "ims": "IMS", "esa-cci": "ESA-CCI"}
ta = np.asarray([dt.datetime(2001, 1, 1, 12)
                 + dt.timedelta(days=i) for i in range(365)])
fig = plt.figure(dpi=150)
ax = plt.axes()
# -----------------------------------------------------------------------------
data = np.empty((3, 365), dtype=np.float32)
for ind_i, i in enumerate(("tpsce", "ims", "esa-cci")):
    data[ind_i, :] = moving_average(data_scf[i] * 100.0, window=30)
plt.fill_between(x=ta, y1=data.min(axis=0), y2=data.max(axis=0), color="grey",
                 alpha=0.25)
# -----------------------------------------------------------------------------
for ind_i, i in enumerate(("era5", "cosmo11pd", "echam5_pi",
                           "tpsce", "ims", "esa-cci")):
    plt.plot(ta, moving_average(data_scf[i] * 100.0, window=30),
             color=colors[ind_i], label=labels[i], lw=1.8)
plt.legend(loc="upper center", frameon=False, fontsize=14, ncol=2)
ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_minor_locator(MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(NullFormatter())
ax.xaxis.set_minor_formatter(DateFormatter("%b"))
plt.xlim([dt.datetime(2001, 1, 1), dt.datetime(2002, 1, 1)])
plt.ylabel("Snow cover extent [%]", fontsize=14)
plt.title('HM Snow', fontsize=14, fontweight="bold")
ax.tick_params(axis='x', which='major', labelsize=14)
ax.tick_params(axis='x', which='minor', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
plt.show()
plotpath = "/project/pr133/rxiang/figure/paper2/results/gm/"
fig.savefig(plotpath + 'HM_Snow_PD.png', dpi=700, transparent=True)

# %%
# Plot (~present-day vs. LGM)
colors = ("navy", "slateblue")
labels = ("COSMO 0.11", "ECHAM5")
plt.figure(dpi=150)
ax = plt.axes()
# -----------------------------------------------------------------------------
for ind_i, i in enumerate(("cosmo11pd", "echam5_pi")):
    plt.plot(ta, moving_average(data_scf[i] * 100.0, window=30),
             color=colors[ind_i], ls="--", lw=1.8)
for ind_i, i in enumerate(("cosmo11lgm", "echam5_lgm")):
    plt.plot(ta, moving_average(data_scf[i] * 100.0, window=30),
             color=colors[ind_i], label=labels[ind_i], lw=1.8)
# -----------------------------------------------------------------------------
# delta_cosmo = moving_average(data_scf["cosmo11lgm"] - data_scf["cosmo11pd"],
#                              window=30)
# plt.plot(ta, delta_cosmo, color="navy", lw=1.8)
# delta_echam = moving_average(data_scf["echam5_lgm"] - data_scf["echam5_pi"],
#                              window=30)
# plt.plot(ta, delta_echam, color="slateblue", lw=1.8)
# -----------------------------------------------------------------------------
plt.legend(loc="upper center", frameon=False, fontsize=12, ncol=2)
ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_minor_locator(MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(NullFormatter())
ax.xaxis.set_minor_formatter(DateFormatter("%b"))
plt.xlim([dt.datetime(2001, 1, 1), dt.datetime(2002, 1, 1)])
plt.ylabel("Snow cover extent [%]")
plt.title(reg, fontsize=12, fontweight="bold")
plt.show()

###############################################################################
# %% Load and plot snow water equivalent (SWE)
###############################################################################

# Select region
reg = "HM_snow"  # "ET_snow", "HM_snow"

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

data_swe = {}

# ERA5
ds = xr.open_dataset(path_masks + "ERA5_region_masks.nc")
mask_reg = ds[reg].values.astype(bool)
ds.close()
ds = xr.open_mfdataset(path_era5 + "ERA5_snow_200?_cp.nc")
# ds = ds.isel(latitude=slice(100, 221), longitude=slice(350, 601))
ds = ds.sel(time=slice("2001-01-01", "2005-12-31"))
# ------------ glacier mask (-> perennial snow accumulation) ------------------
mask_glacier = np.all(ds["sd"].values > 0.01, axis=0)
# -----------------------------------------------------------------------------
ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
print(len(ds["time"]) / 5)
month_day = xr.DataArray(ds.indexes["time"].strftime("%m-%d"),
                         coords=ds["time"].coords, name="month_day")
ds_clim = ds.groupby(month_day).mean("time")
ind_0, ind_1 = np.where(mask_reg & ~mask_glacier)
data_swe["era5"] = ds_clim["sd"].values[:, ind_0, ind_1].mean(axis=1) * 1000.0
# [mm]
ds.close()

# COSMO 0.11
ds = xr.open_dataset(path_masks + "CTRL11_region_masks.nc")
mask_reg = ds[reg].values.astype(bool)
ds.close()
cosmo_run = {"cosmo11pd": path_cosmo + "EAS11_ctrl/24h/W_SNOW/0?_W_SNOW.nc",
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
    month_day = xr.DataArray(ds.indexes["time"].strftime("%m-%d"),
                             coords=ds["time"].coords, name="month_day")
    ds_clim = ds.groupby(month_day).mean("time")
    ind_0, ind_1 = np.where(mask_reg & ~mask_glacier)
    data_swe[i] = np.nanmean(ds_clim["W_SNOW"].values[:, ind_0, ind_1],
                             axis=1) * 1000.0
    # 'np.nanmean()' due to water grid cells
    ds.close()

# ECHAM5
ds = xr.open_dataset(path_masks + "ECHAM5_region_masks.nc")
mask_reg = ds[reg].values.astype(bool)
ds.close()
for i in ("echam5_pi", "echam5_lgm"):
    if i == "echam5_pi":
        ds = xr.open_dataset(path_temp + "ECHAM5_PI_sn_5years.nc")
        # ------------ glacier mask (-> perennial snow accumulation) ----------
        mask_glacier = np.all(ds["sn"].values > 0.01, axis=0)
        # ---------------------------------------------------------------------
        ds = ds.sel(time=~((ds["time.month"] == 2) & (ds["time.day"] == 29)))
        print(len(ds["time"]) / 5)
    else:
        ds = xr.open_dataset(path_temp + "ECHAM5_LGM_sn_5years.nc")
        # ------------ glacier mask (-> perennial snow accumulation) ----------
        mask_glacier = np.all(ds["sn"].values > 0.01, axis=0)
        # ---------------------------------------------------------------------
        ds = ds.sel(time=~((ds["time.month"] == 2) & (ds["time.day"] == 29)))
        print(len(ds["time"]) / 5)
    month_day = xr.DataArray(ds.indexes["time"].strftime("%m-%d"),
                             coords=ds["time"].coords, name="month_day")
    ds_clim = ds.groupby(month_day).mean("time")
    print(mask_glacier.sum())
    ind_0, ind_1 = np.where(mask_reg & ~mask_glacier)
    data_swe[i] = ds_clim["sn"].values[:, ind_0, ind_1].mean(axis=1) * 1000.0
    ds.close()

# -----------------------------------------------------------------------------
# Plot data
# -----------------------------------------------------------------------------

# Plot
colors = ("dodgerblue", "navy", "slateblue")
labels = {"era5": "ERA5", "cosmo11pd": "COSMO 0.11", "echam5_pi": "ECHAM5"}
plt.figure(dpi=150)
ax = plt.axes()
for ind_i, i in enumerate(("era5", "cosmo11pd", "echam5_pi")):
    plt.plot(ta, moving_average(data_swe[i], window=30),
             color=colors[ind_i], label=labels[i], lw=1.8)
plt.legend(loc="upper center", frameon=False, fontsize=12, ncol=2)
ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_minor_locator(MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(NullFormatter())
ax.xaxis.set_minor_formatter(DateFormatter("%b"))
plt.xlim([dt.datetime(2001, 1, 1), dt.datetime(2002, 1, 1)])
plt.ylabel("Snow water equivalent [mm]")
plt.title(reg, fontsize=12, fontweight="bold")
plt.show()
