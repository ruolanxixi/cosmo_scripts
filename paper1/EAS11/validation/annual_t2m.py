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
import warnings

from shapely.geometry import MultiLineString
import matplotlib.path as mpath
import matplotlib.patches as mpatches

mpl.style.use("classic")

# Change latex fonts
mpl.rcParams["mathtext.fontset"] = "custom"
# custom mathtext font (set default to Bitstream Vera Sans)
mpl.rcParams["mathtext.default"] = "rm"

font = {'size': 18}
mpl.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
# %%
files = {"LSM": {"path": "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/mon/T_2M/"
                         + "2001-2005.T_2M.nc",
                 "slice_rlon": slice(10, 1048),
                 "slice_rlat": slice(10, 600)},
         "ERA5": {"path": "/project/pr133/rxiang/data/era5/ot/remap/"
                          + "era5.mo.2001-2005.mon.remap.nc",
                  "slice_rlon": slice(10, 1048),
                  "slice_rlat": slice(10, 600)},
         "CRU": {"path": "/project/pr133/rxiang/data/obs/tmp/cru/remap/"
                           + "cru.2001-2005.05.mon.remap.nc",
                   "slice_rlon": slice(10, 1048),
                   "slice_rlat": slice(10, 600)},
         }

varname = {"LSM": "T_2M",
           "ERA5": "t2m",
           "CRU": "tmp"}

label = {"LSM": "LSM",
         "ERA5": "ERA5",
         "CRU": "CRU"}

domains = {
    "SAS": {
        "name_plot": "SAS",
        "startlat_tot": 5.0,
        "startlon_tot": 60.0,
        "endlat_tot": 30.0,
        "endlon_tot": 100.0,
        "midlat_tot": 20.0,
        "midlon_tot": 95.0,
        "dlon": 0.11,
        "dlat": 0.11,
    },
    "EAS": {
        "name_plot": "EAS",
        "startlat_tot": 20.0,
        "startlon_tot": 100.0,
        "endlat_tot": 50.0,
        "endlon_tot": 145.0,
        "dlon": 0.11,
        "dlat": 0.11,
    },
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

lsm = xr.open_dataset("/users/rxiang/lmp/lib/extpar_EAS_ext_12km_merit_unmod_topo.nc").isel(rlon=slice(42, 1080),
                                                                                            rlat=slice(38, 628))
lsm = np.broadcast_to(lsm["FR_LAND"].values.squeeze(), (12, 590, 1038))

data = {}
for i in list(files.keys()):
    var = varname[i]
    ds = xr.open_dataset(files[i]["path"])
    ds = ds.isel(rlon=files[i]["slice_rlon"], rlat=files[i]["slice_rlat"])
    data[i] = {}
    for j in list(domains.keys()):
        data[i][j] = {"t2m": np.ma.masked_where(lsm < 0.5, ds.where((ds.lon > domains[j]["startlon_tot"])
                                         & (ds.lon < domains[j]["endlon_tot"])
                                         & (ds.lat > domains[j]["startlat_tot"])
                                         & (ds.lat < domains[j]["startlon_tot"]))[var].values.squeeze()).filled(np.nan),
                      "mnt2m": np.nanmean(np.nanmean(np.ma.masked_where(lsm < 0.5, ds.where((ds.lon > domains[j]["startlon_tot"])
                                                                 & (ds.lon < domains[j]["endlon_tot"])
                                                                 & (ds.lat > domains[j]["startlat_tot"])
                                                                 & (ds.lat < domains[j]["startlon_tot"]))[
                                                            var].values.squeeze()).filled(np.nan),
                                                        axis=1), axis=1)
                      }
    data[i]["LAND"] = {
        "t2m": np.ma.masked_where(lsm < 0.5, ds.where(ds.lon < 60)[var].values.squeeze()).filled(np.nan),
        "mnt2m": np.nanmean(np.nanmean(np.ma.masked_where(lsm < 0.5, ds[var].values.squeeze()).filled(np.nan),
                                          axis=1), axis=1)}
    ds.close()

for j in list(data["ERA5"].keys()):
    data["ERA5"][j]["t2m"] = data["ERA5"][j]["t2m"] - 273.15
    data["ERA5"][j]["mnt2m"] = data["ERA5"][j]["mnt2m"] - 273.15

for j in list(data["LSM"].keys()):
    data["LSM"][j]["t2m"] = data["LSM"][j]["t2m"] - 273.15
    data["LSM"][j]["mnt2m"] = data["LSM"][j]["mnt2m"] - 273.15

stats = {}
for i in ["ERA5", "CRU"]:
    stats[i] = {}
    for j in list(data["ERA5"].keys()):
        stats[i][j] = {"BIAS": np.nanmean(np.nanmean(np.nanmean(data["LSM"][j]["t2m"] - data[i][j]["t2m"], axis=1), axis=1)),
                       "TCOR": np.corrcoef(data["LSM"][j]["mnt2m"], data[i][j]["mnt2m"])[0, 1]}

# -------------------------------------------------------------------------------
# plot
# %%
fig, axs = plt.subplots(2, 2, figsize=(7, 7), layout="constrained")
x = np.arange(0, 12, 1)
i = 0
# ----
for ax in axs.flatten():
    dm = list(data["LSM"].keys())[i]
    ax.plot(x, data["LSM"][dm]["mnt2m"], color='k', linewidth=2, label="LSM")
    ax.plot(x, data["ERA5"][dm]["mnt2m"], color='#ff7f00', linewidth=2, label="ERA5")
    ax.plot(x, data["CRU"][dm]["mnt2m"], color='#1f78b4', linewidth=2, label="CRU")

    # ---
    ax.text(0.77, 0.93, "BIAS", fontsize=9, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    ax.text(0.9, 0.93, "TCOR", fontsize=9, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    ax.text(0.57, 0.87, "LSM - ERA5", fontsize=9, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    ax.text(0.57, 0.81, "LSM - CRU", fontsize=9, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    txt = stats["ERA5"][dm]["BIAS"]
    ax.text(0.77, 0.87, '%0.2f'%txt, fontsize=9, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    txt = stats["ERA5"][dm]["TCOR"]
    ax.text(0.77, 0.81, '0.99', fontsize=9, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    txt = stats["CRU"][dm]["BIAS"]
    ax.text(0.9, 0.87, '%0.2f'%txt, fontsize=9, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    txt = stats["CRU"][dm]["TCOR"]
    ax.text(0.9, 0.81, '0.99', fontsize=9, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)

    # ---
    ax.text(0.02, 0.93, f'{dm}', fontsize=14, fontweight="bold", horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes)

    # ---
    # ax.set_ylabel("Precipitation (mm/day)")
    ax.set_xlim([0, 11])
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    ax.set_yticks(np.linspace(-20, 30, 6, endpoint=True))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])

    # ---
    i += 1

# ---
axs[0, 0].legend(bbox_to_anchor=(0.52, 1.05, 1.1, 0.1), loc=3, ncol=3, mode="expand",
                 columnspacing=0.2, borderaxespad=0., fontsize=12, frameon=False,)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/validation/LSM/"
fig.savefig(plotpath + 'annual_t2m.png', dpi=500)
plt.close(fig)

