# Description: Compute precipitation indices from CPM simulations. Statistics
#              are computed from yearly blocks and subsequently merged. For
#              percentiles, this is achieved by keeping the largest occurring
#              precipitation elements (and updating them during the iteration
#              through the yearly blocks). All specified percentiles can then
#              be computed from these maximal values in "one go".
#
# Literature referenced:
# - Ban et al. (2021): The first multi‐model ensemble of regional climate
#   simulations at kilometer‐scale resolution, part I: evaluation of
#   precipitation
# - Schär et al. (2016): Percentile indices for assessing changes in heavy
#   precipitation events
#
# Author: Christian R. Steger, November 2022

# Load modules
import numpy as np
import time
import xarray as xr
import calendar as cal
import glob
from itertools import compress
import numba as nb
import textwrap

# Paths to folders
# path_mod = "/scratch/snx3000/rxiang/IMERG/day/"
path_mod = "/project/pr133/rxiang/data/cosmo/EAS04_topo2/indices/"
path_out = "/scratch/snx3000/rxiang/IMERG/indices/"
path_out = "/project/pr133/rxiang/data/cosmo/EAS04_topo2/indices/"


###############################################################################
# Functions to update maximal precipitation values
###############################################################################

@nb.jit((nb.float32[:, :, :], nb.float32[:, :, :], nb.int64, nb.int64,
         nb.int64), nopython=True, parallel=True)
def update_max_values_all_day(prec_keep, prec, len_y, len_x, num_keep):
    """Update maximal precipitation values for all day percentile
    calculation"""
    for i in nb.prange(len_y):
        for j in range(len_x):
            mask = (prec[:, i, j] > prec_keep[i, j, 0])
            prec_keep[i, j, :] \
                = np.sort(np.append(prec_keep[i, j, :],
                                    prec[mask, i, j]))[-num_keep:]


@nb.jit((nb.float32[:, :, :], nb.float32[:, :, :], nb.int64, nb.int64,
         nb.int64, nb.float64), nopython=True, parallel=True)
def update_max_values_wet_day(prec_keep, prec, len_y, len_x, num_keep,
                              prec_thresh):
    """Update maximal precipitation values for wet day percentile
    calculation"""
    for i in nb.prange(len_y):
        for j in range(len_x):
            mask = (prec[:, i, j] > prec_keep[i, j, 0]) \
                   & (prec[:, i, j] >= prec_thresh)
            prec_keep[i, j, :] \
                = np.sort(np.append(prec_keep[i, j, :],
                                    prec[mask, i, j]))[-num_keep:]


###############################################################################
# Settings
###############################################################################

# Simulations (yearly NetCDF blocks)
# mod = "IMERG"  # ~1.3G per file
mod = "COSMO"  # ~8.2 GB per file
# mod = "CTRL04"

# Settings
intra_yr_per = "JJA"  # "year", "JJA", "SON", "DJF", "MAM"
temp_gran = "day"  # temporal granularity; "1hr" or "day"
ts_cons_perc = "all"  # "all", "wet"
# time steps considered for computing percentiles (Schär et al., 2016)
qs = np.array([90.0, 95.0, 99.0, 99.9])  # percentiles [0.0, 100.0]
years = np.arange(2001, 2006)  # years 1996 - 2005
# years = np.arange(1998, 2008)  # for HadREM3-RA-UM10.1
prec_thresh = {"1hr": 0.1, "day": 1.0}  # [mm]
# threshold for wet day/hour according to Ban et al., (2021)

###############################################################################
# Process data
###############################################################################
print((" Process model " + mod + " ").center(79, "-"))

# Check input settings
if temp_gran not in ("1hr", "day"):
    raise ValueError("Unknown selected temporal granularity")
if ts_cons_perc not in ("all", "wet"):
    raise ValueError("Unknown value for 'ts_cons_perc'")
if (qs.min() < 85.0) or (qs.max() > 100.0):
    raise ValueError("Allowed range for qs of [85.0, 100.0] is exceeded")
if intra_yr_per not in ("year", "JJA", "SON", "DJF", "MAM"):
    raise ValueError("Unknown value for 'intra_yr_per'")

# Adapt model input path to temporal granularity
path_mod_tg = path_mod.replace("temp_gran", temp_gran)

# Get relevant files
# files = glob.glob(path_mod_tg + "????.day.corr.nc")
files = glob.glob(path_mod_tg + "??_TOT_PREC.nc")
files.sort()
# mask_files = [(int(i[-11:-7]) >= years[0]) & (int(i[-11:-7]) <= years[-1])
#               for i in files]
# files = list(compress(files, mask_files))
if len(files) != 5:
    raise ValueError("Incorrect number of files")
# files = path_mod_tg + "2001-2005.day.corr.nc"
# %%
# Load dimension information
ds = xr.open_dataset(files[0], decode_times=False)
if "calendar" in list(ds["time"].attrs):
    mod_cal = ds["time"].calendar
elif "calender" in list(ds["time"].attrs):
    mod_cal = ds["time"].calender
else:
    raise ValueError("Calendar attribute not found")

if "rlon" in list(ds.coords):
    len_x = ds.coords["rlon"].size
    len_y = ds.coords["rlat"].size
    out_dim = ("rlat", "rlon")
elif "lon" in list(ds.coords):
    len_x = ds.coords["lon"].size
    len_y = ds.coords["lat"].size
    out_dim = ("lat", "lon")
else:
    raise ValueError("Unknown spatial coordinates")
ds.close()
# %%
# Compute total number of time steps
if mod_cal in ("standard", "gregorian", "proleptic_gregorian"):
    if intra_yr_per == "year":
        num_days = sum([int(cal.isleap(i)) + 365 for i in years])
    elif intra_yr_per in ("MAM", "JJA"):
        num_days = 92 * len(years)
    elif intra_yr_per == "SON":
        num_days = 91 * len(years)
    else:
        num_days = 62 * len(years) \
                   + sum([cal.monthrange(i, 2)[1] for i in years])
elif mod_cal == "360_day":
    if intra_yr_per == "year":
        num_days = len(years) * 360
    else:
        num_days = len(years) * 90
else:
    raise ValueError("Unknown calendar")
ts_per_day = {"1hr": 24, "day": 1}
num_ts_tot = num_days * ts_per_day[temp_gran]
print("Total number of time steps: " + str(num_ts_tot))
# %%
# -----------------------------------------------------------------------------
# Compute annual mean, rx1d, wet day frequency and intensity
# -----------------------------------------------------------------------------
print(" Compute annual mean, rx1d/h, wet day frequency and intensity "
      .center(79, "-"))

t_beg_tot = time.time()

# Allocate arrays
prec_mean_iter = np.empty((len(years), len_y, len_x), dtype=np.float32)
prec_rx1d_iter = np.empty((len(years), len_y, len_x), dtype=np.float32)
ts_above_thresh_py = np.empty((len(years), len_y, len_x), dtype=np.int32)
prec_int_iter = np.empty((len(years), len_y, len_x), dtype=np.float32)
num_keep = (np.ceil(num_ts_tot * (1.0 - qs.min() / 100)) + 1) \
    .astype(np.int32)
prec_keep = np.empty((len_y, len_x, num_keep), dtype=np.float32)
prec_keep.fill(-999.0)
print("Size of array 'prec_keep': %.1f" % (prec_keep.nbytes / (10 ** 9))
      + " GB")

# Conversion of input precipitation unit to [mm h-1] or [mm day-1]
con_fac = {"1hr": 3600.0, "day": 1.0}
out_unit = {"1hr": "mm h-1", "day": "mm day-1"}
var_name = {"IMERG": "precipitation_corr", "COSMO": "TOT_PREC"}
# %%
# Loop through years
num_ts_per_block = np.empty(len(years), dtype=np.int32)
var = var_name[mod]
for ind, year in enumerate(years):

    print((" Process year " + str(year) + " ").center(79, "-"))

    # Load data
    t_beg = time.time()
    file_in = files[ind]
    print(textwrap.fill("Load file " + file_in.split("/")[-1], width=79))
    if intra_yr_per == "year":
        # ---------------------------------------------------------------------
        # Open larger NetCDF files (ca. > 5 GB) in blocks. Importing the data
        # with xarray or netCDF4 in one go can cause the function to freeze.
        # ---------------------------------------------------------------------
        block_size = 5.0
        # maximal block size loaded during by one function call [GB]
        ds = xr.open_dataset(file_in)
        len_t = ds.coords["time"].size
        ds.close()
        prec = np.empty((len_t, len_y, len_x), dtype=np.float32)
        prec.fill(np.nan)
        num_blocks = int(np.ceil((prec.nbytes / (10 ** 9)) / block_size))
        lim = np.linspace(0, len_t, num_blocks + 1, dtype=np.int32)
        for i in range(num_blocks):
            ds = xr.open_dataset(file_in)
            slice_t = slice(lim[i], lim[i + 1])
            prec[slice_t, :, :] = ds[var][slice_t, :, :].values
            ds.close()
            print("Data blocks loaded: " + str(i + 1) + "/" + str(num_blocks))
        # ---------------------------------------------------------------------
    else:
        ds = xr.open_dataset(file_in)
        ds = ds.sel(time=ds["time.season"] == intra_yr_per)
        len_t = ds.coords["time"].size
        prec = ds[var].values
        ds.close()
    # prec *= con_fac[temp_gran]  # convert units to [mm h-1] or [mm day-1]
    print("Data opened (" + "%.1f" % (time.time() - t_beg) + " s)")
    num_ts_per_block[ind] = len_t

    # Check input data
    prec_min = prec.min()
    if prec_min < 0.0:
        print("Warning: negative precipitation value(s) detected")
        print("Minimum value: %.5f" % prec_min + " " + out_unit[temp_gran])
        if prec_min < -1.0:
            raise ValueError("Negative precipitation smaller than -1.0 "
                             + out_unit[temp_gran] + " found")
    # if np.isnan(prec_min):
    #     raise ValueError("NaN-value(s) detected")

    # Compute mean
    t_beg = time.time()
    prec_mean_iter[ind, :, :] = np.nanmean(prec, axis=0)
    print("Mean computed (" + "%.1f" % (time.time() - t_beg) + " s)")

    # Compute Rx1d/h (maximum daily/hourly precipitation)
    t_beg = time.time()
    prec_rx1d_iter[ind, :, :] = np.nanmax(prec, axis=0)
    print("Rx1d/h computed (" + "%.1f" % (time.time() - t_beg) + " s)")

    # Compute wet day frequency
    t_beg = time.time()
    mask_inc = (prec >= prec_thresh[temp_gran])
    ts_above_thresh_py[ind, :, :] = mask_inc.sum(axis=0)
    print("Wet day frequency computed (" + "%.1f" % (time.time() - t_beg)
          + " s)")

    # Compute percentiles
    t_beg = time.time()
    if ts_cons_perc == "all":
        update_max_values_all_day(prec_keep, prec, len_y, len_x, num_keep)
    else:
        update_max_values_wet_day(prec_keep, prec, len_y, len_x, num_keep,
                                  prec_thresh[temp_gran])
    print("Maximal values updated (" + "%.1f" % (time.time() - t_beg) + " s)")

    # Intensity
    t_beg = time.time()
    prec[~mask_inc] = 0.0
    num_ts_cons = ts_above_thresh_py[ind, :, :].astype(np.float32)
    mask_issue = (num_ts_cons == 0)
    if np.any(mask_issue):
        print("Warning: some grid cells never exceed the threshold value")
        print("Set affected grid cells to NaN")
        num_ts_cons[mask_issue] = np.nan
        # avoid division by 0.0 by setting the grid cells to NaN
    prec_int_iter[ind, :, :] = np.nanmean(prec, axis=0) \
        * (prec.shape[0] / num_ts_cons)
    print("Intensity computed (" + "%.1f" % (time.time() - t_beg) + " s)")

print((" Aggregate statistics over " + str(len(years)) + " years ")
      .center(79, "-"))
# %%
# Compute annual mean, Rx1d/h and wet day frequency for entire period
prec_mean = np.average(prec_mean_iter, weights=num_ts_per_block, axis=0) \
    .astype(np.float32)
prec_rx1d = prec_rx1d_iter.mean(axis=0)
if num_ts_tot != num_ts_per_block.sum():
    raise ValueError("Inconsistency in total number of time steps")
ts_above_thresh = ts_above_thresh_py.sum(axis=0).astype(np.int32)
prec_wtf = (ts_above_thresh / num_ts_tot).astype(np.float32)

# Compute intensity for entire period
mask_valid = (ts_above_thresh > 0)
if np.all(mask_valid):
    prec_int = np.average(prec_int_iter, weights=ts_above_thresh_py,
                          axis=0).astype(np.float32)
else:
    print("Mask problematic grid cells for intensity")
    ind_0_inval, ind_1_inval = np.where(~mask_valid)
    ts_above_thresh_py_mod = ts_above_thresh_py.copy()
    ts_above_thresh_py_mod[:, ind_0_inval, ind_1_inval] = 1
    prec_int = (np.average(prec_int_iter, weights=ts_above_thresh_py_mod,
                           axis=0)).astype(np.float32)
    prec_int[ind_0_inval, ind_1_inval] = np.nan

# Compute percentiles for entire period
prec_per = np.empty((len(qs), len_y, len_x), dtype=np.float32)
if ts_cons_perc == "all":
    print("Compute all day precipitation percentiles")
    x = np.linspace(0.0, 100.0, num_ts_tot, dtype=np.float32)
    if qs.min() < x[-num_keep]:
        raise ValueError("x-position for interpolation is out of range")
    for i in range(len_y):
        for j in range(len_x):
            prec_per[:, i, j] = np.interp(qs, x[-num_keep:],
                                          prec_keep[i, j, :])
else:
    print("Compute wet day precipitation percentiles")
    for i in range(len_y):
        for j in range(len_x):
            x = np.linspace(0.0, 100.0, ts_above_thresh[i, j],
                            dtype=np.float32)
            if ts_above_thresh[i, j] >= num_keep:
                if qs.min() < x[-num_keep]:
                    raise ValueError("x-position for interpolation is out of "
                                     + "range")
                prec_per[:, i, j] = np.interp(qs, x[-num_keep:],
                                              prec_keep[i, j, :])
            else:
                prec_per[:, i, j] = np.interp(qs, x, prec_keep[i, j, -len(x):])

# %%
# -----------------------------------------------------------------------------
# Save statistics to NetCDF file
# -----------------------------------------------------------------------------
print("Save statistics to NetCDF file".center(79, "-"))

# Processing information
proc_info = "Considered years: " + str(years[0]) + " - " + str(years[-1]) \
            + ", intra-annual period: " + str(intra_yr_per) + ", " \
            + "threshold for wet day: %.2f" % prec_thresh[temp_gran] + " mm"
fn_add = str(years[0]) + "-" + str(years[-1]) + "_" + str(intra_yr_per) \
         + "_" + ts_cons_perc + "_day_perc"

# Save
# file_out = files[0].split("/")[-1][:-3] + "_" + fn_add + ".nc"
file_out = fn_add + ".nc"
ds = xr.open_mfdataset(files[0])
ds = ds.drop_dims("time")
ds.attrs["processing_information"] = proc_info
ds["mean"] = (out_dim, prec_mean)
ds["mean"].attrs["units"] = out_unit[temp_gran]
ds["mean"].attrs["long_name"] = "intra-annual period maximum averaged " \
                                + "over years"
ds["max"] = (out_dim, prec_rx1d)
ds["max"].attrs["units"] = out_unit[temp_gran]
ds["max"].attrs["long_name"] = "intra-annual period mean averaged over years"
ds["wet_day_freq"] = (out_dim, prec_wtf)
ds["wet_day_freq"].attrs["units"] = "-"
ds["wet_day_freq"].attrs["long_name"] = "wet day frequency"
ds["intensity"] = (out_dim, prec_int)
ds["intensity"].attrs["units"] = out_unit[temp_gran]
ds["intensity"].attrs["long_name"] = "intensity"
for ind, q in enumerate(qs):
    name = "perc_%.2f" % q
    ds[name] = (out_dim, prec_per[ind, :, :])
    ds[name].attrs["units"] = out_unit[temp_gran]
    ds[name].attrs["long_name"] = "%.2f" % q + " " + ts_cons_perc \
                                  + "-day percentile"
ds.to_netcdf(path_out + file_out)

print("Total elapsed time: %.1f" % (time.time() - t_beg_tot) + " s")
