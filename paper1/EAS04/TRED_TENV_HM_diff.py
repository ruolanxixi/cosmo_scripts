# Description: Compute precipitation quantities for the Hengduan Mountains for
#              the control experiment as well as the reduced and envelope
#              topography experiment (and also difference).
#
# Author: Christian R. Steger, April 2023

# Load modules
import os
import numpy as np
import xarray as xr

# Path to folders
path_reg_masks = "/project/pr133/csteger/Data/Model/BECCY/region_masks/"

###############################################################################
# Load and process precipitation and temperature data
###############################################################################

# Load region masks
regions = ["ET", "HM", "HMU", "HMC", "HMUS", "HMUN"]
ds = xr.open_dataset(path_reg_masks + "CTRL04_region_masks.nc")
region_masks = {i: ds[i].values.astype(bool) for i in regions}
ds.close()

# Load precipiation
exp = ["ctrl", "topo1", "topo2"]
prec = {}
for i in exp:
    ds = xr.open_dataset("/project/pr133/rxiang/data/cosmo/EAS04_" + i
                         + "/mon/TOT_PREC/2001-2005.TOT_PREC.nc")
    ds = ds.sel(time=(ds["time.month"] >= 5) & (ds["time.month"] <= 9))
    prec[i] = ds["TOT_PREC"].mean(axis=0).values  # mm/day
    ds.close()

# Compute spatially integrated quantity over regions
for i in ["HM", "HMC", "HMU"]:
    print("--------------" + i + "--------------")
    print("ctrl: %.1f" % prec["ctrl"][region_masks[i]].mean() + " mm/day")
    for j in exp[1:]:
        diff_abs = (prec[j][region_masks[i]].mean()
                    - prec["ctrl"][region_masks[i]].mean())
        diff_rel = (prec[j][region_masks[i]].mean()
                    / prec["ctrl"][region_masks[i]].mean() - 1.0) * 100.0
        print(j + ": %.1f" % prec[j][region_masks[i]].mean()
              + " ({0:+.1f}".format(diff_abs) + ")" + " mm/day"
              + " ({0:+.1f}".format(diff_rel) + " %)")

# continue with other precipiation quantities...
# Load extreme
p99D = {}
for i in exp:
    ds = xr.open_dataset("/project/pr133/rxiang/data/cosmo/EAS04_" + i
                         + "/indices/day/2001-2005_smr_all_day_perc.nc")
    p99D[i] = ds["perc_99.00"].values  # mm/day
    ds.close()

# Compute spatially integrated quantity over regions
for i in ["HM", "HMC", "HMU"]:
    print("--------------" + i + "--------------")
    print("ctrl: %.1f" % p99D["ctrl"][region_masks[i]].mean() + " mm/day")
    for j in exp[1:]:
        diff_abs = (p99D[j][region_masks[i]].mean()
                    - p99D["ctrl"][region_masks[i]].mean())
        diff_rel = (p99D[j][region_masks[i]].mean()
                    / p99D["ctrl"][region_masks[i]].mean() - 1.0) * 100.0
        print(j + ": %.1f" % p99D[j][region_masks[i]].mean()
              + " ({0:+.1f}".format(diff_abs) + ")" + " mm/day"
              + " ({0:+.1f}".format(diff_rel) + " %)")

p999H = {}
for i in exp:
    ds = xr.open_dataset("/project/pr133/rxiang/data/cosmo/EAS04_" + i
                         + "/indices/hr/2001-2005_smr_all_day_perc.nc")
    p999H[i] = ds["perc_99.90"].values  # mm/day
    ds.close()

# Compute spatially integrated quantity over regions
for i in ["HM", "HMC", "HMU"]:
    print("--------------" + i + "--------------")
    print("ctrl: %.1f" % p999H["ctrl"][region_masks[i]].mean() + " mm/hr")
    for j in exp[1:]:
        diff_abs = (p999H[j][region_masks[i]].mean()
                    - p999H["ctrl"][region_masks[i]].mean())
        diff_rel = (p999H[j][region_masks[i]].mean()
                    / p999H["ctrl"][region_masks[i]].mean() - 1.0) * 100.0
        print(j + ": %.1f" % p999H[j][region_masks[i]].mean()
              + " ({0:+.1f}".format(diff_abs) + ")" + " mm/hr"
              + " ({0:+.1f}".format(diff_rel) + " %)")
        

R = {}
for i in exp:
    ds1 = xr.open_dataset("/project/pr133/rxiang/data/cosmo/EAS04_" + i
                          + "/monsoon/RUNOFF_G/smr/01-05.RUNOFF_G.smr.nc")
    ds2 = xr.open_dataset("/project/pr133/rxiang/data/cosmo/EAS04_" + i
                          + "/monsoon/RUNOFF_S/smr/01-05.RUNOFF_S.smr.nc")
    R[i] = np.nanmean(ds1["RUNOFF_G"].values, axis=0) + np.nanmean(ds2["RUNOFF_S"].values, axis=0)  # mm/day
    ds.close()

# Compute spatially integrated quantity over regions
for i in ["HM", "HMC", "HMU"]:
    print("--------------" + i + "--------------")
    print("ctrl: %.1f" % np.nanmean(R["ctrl"][region_masks[i]]) + " mm/day")
    for j in exp[1:]:
        diff_abs = (np.nanmean(R[j][region_masks[i]])
                    - np.nanmean(R["ctrl"][region_masks[i]]))
        diff_rel = (np.nanmean(R[j][region_masks[i]])
                    / np.nanmean(R["ctrl"][region_masks[i]]) - 1.0) * 100.0
        print(j + ": %.1f" % np.nanmean(R[j][region_masks[i]])
              + " ({0:+.1f}".format(diff_abs) + ")" + " mm/day"
              + " ({0:+.1f}".format(diff_rel) + " %)")
