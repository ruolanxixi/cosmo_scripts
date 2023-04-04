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

# mpl.style.use("classic")

# Change latex fonts
# mpl.rcParams["mathtext.fontset"] = "custom"
# custom mathtext font (set default to Bitstream Vera Sans)
# mpl.rcParams["mathtext.default"] = "rm"

font = {'size': 16}
mpl.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
# %%
files = {"LSM": {"path": "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/mon/TOT_PREC/"
                         + "2001-2005.TOT_PREC.nc",
                 "lon": "lon",
                 "lat": "lat"},
         "CPM": {"path": "/project/pr133/rxiang/data/cosmo/EAS04_ctrl/mon/TOT_PREC/"
                         + "2001-2005.TOT_PREC.nc",
                 "lon": "lon",
                 "lat": "lat"},
         "IMERG": {"path": "/project/pr133/rxiang/data/obs/pr/IMERG/v06/"
                           + "IMERG.2001-2005.nc",
                   "lon": "longitude",
                   "lat": "latitude"},
         "IMERG-BA": {"path": "/project/pr133/rxiang/data/obs/pr/IMERG/v06/"
                              + "IMERG.2001-2005.corr.nc",
                      "lon": "lon",
                      "lat": "lat"},
         "APHRO": {"path": "/project/pr133/rxiang/data/obs/pr/APHRO/"
                           + "APHRO_2001-2005_ymonmean.nc",
                   "lon": "lon",
                   "lat": "lat"},
         "APHRO-BA": {"path": "/project/pr133/rxiang/data/obs/pr/APHRO/"
                              + "APHRO_2001-2005_corr.nc",
                      "lon": "lon",
                      "lat": "lat"},
         "ERA5": {"path": "/project/pr133/rxiang/data/era5/pr/mo/"
                          + "era5.mo.2001-2005.mon.nc",
                  "lon": "lon",
                  "lat": "lat"}
         }

varname = {"LSM": "TOT_PREC",
           "CPM": "TOT_PREC",
           "ERA5": "tp",
           "IMERG": "precipitation",
           "IMERG-BA": "precipitation_corr",
           "APHRO": "precip",
           "APHRO-BA": "precipitation_corr"}

label = {"LSM": "CTRL11",
         "CPM": "CTRL04",
         "ERA5": "ERA5",
         "IMERG-BA": "IMERG-BA",
         "APHRO": "APHRO",
         "APHRO-BA": "APHRO-BA",
         "IMERG": "IMERG"}

# define domain
startlat_tot = 23.0
startlon_tot = 88.0
endlat_tot = 37.5
endlon_tot = 112.5

precip = {}
for i in list(files.keys()):
    var = varname[i]
    ds = xr.open_dataset(files[i]["path"])
    lon = files[i]["lon"]
    lat = files[i]["lat"]
    if i == 'ERA5':
        precip[i] = np.nanmean(np.nanmean(ds.where((ds.longitude > startlon_tot)
                                                   & (ds.longitude < endlon_tot)
                                                   & (ds.latitude > startlat_tot)
                                                   & (ds.latitude < endlat_tot))[var].values.squeeze(),
                                          axis=1), axis=1)
    else:
        precip[i] = np.nanmean(np.nanmean(ds.where((ds.lon > startlon_tot)
                                                   & (ds.lon < endlon_tot)
                                                   & (ds.lat > startlat_tot)
                                                   & (ds.lat < endlat_tot))[var].values.squeeze(),
                                          axis=1), axis=1)
    ds.close()

precip["ERA5"] = precip["ERA5"] * 1000

# -------------------------------------------------------------------------------
# plot
# %%
import seaborn as sns

colors = sns.color_palette("Paired", 10).as_hex()
del colors[4:6]
fig, axs = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
x = np.arange(0, 12, 1)
i = 0
# ----
for ds in list(precip.keys()):
    axs[0].plot(x, precip[ds], color=colors[i], linewidth=2, label=label[ds])
    i += 1

    # ---
    axs[0].set_title("(a) Precipitation [mm d$^{-1}$]", loc='left', fontsize=16)
    axs[0].set_xlim([0, 11])
    axs[0].set_ylim([0, 10])
    axs[0].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    axs[0].set_yticks([0, 2, 4, 6, 8, 10])
    axs[0].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])

# ---
# axs.legend(bbox_to_anchor=(0.52, 1.05, 1.1, 0.1), loc=3, ncol=3, mode="expand",
#             columnspacing=0.2, borderaxespad=0., fontsize=12, frameon=False,)

axs[0].legend(fontsize=12, frameon=False)

# plot
# %%

files = {"LSM": {"path": "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/mon/T_2M/"
                         + "2001-2005.T_2M.nc",
                 "lon": "lon",
                 "lat": "lat"},
         "CPM": {"path": "/project/pr133/rxiang/data/cosmo/EAS04_ctrl/mon/T_2M/"
                         + "2001-2005.T_2M.nc",
                 "lon": "lon",
                 "lat": "lat"},
         "CRU": {"path": "/project/pr133/rxiang/data/obs/tmp/cru/mo/"
                           + "cru.2001-2005.05.mon.nc",
                   "lon": "longitude",
                   "lat": "latitude"},
         "APHRO": {"path": "/project/pr133/rxiang/data/obs/tmp/APHRO/day/"
                           + "APHRO.2001-2005.025.mon.nc",
                   "lon": "lon",
                   "lat": "lat"},
         "ERA5": {"path": "/project/pr133/rxiang/data/era5/ot/mo/"
                          + "era5.mo.2001-2005.mon.nc",
                  "lon": "lon",
                  "lat": "lat"}
         }

varname = {"LSM": "T_2M",
           "CPM": "T_2M",
           "ERA5": "t2m",
           "CRU": "tmp",
           "APHRO": "tave"}

label = {"LSM": "CTRL11",
         "CPM": "CTRL04",
         "ERA5": "ERA5",
         "CRU": "CRU",
         "APHRO": "APHRO"}

t2m = {}
for i in list(files.keys()):
    var = varname[i]
    ds = xr.open_dataset(files[i]["path"])
    lon = files[i]["lon"]
    lat = files[i]["lat"]
    if i == 'ERA5':
        t2m[i] = np.nanmean(np.nanmean(ds.where((ds.longitude > startlon_tot)
                                                   & (ds.longitude < endlon_tot)
                                                   & (ds.latitude > startlat_tot)
                                                   & (ds.latitude < endlat_tot))[var].values.squeeze(),
                                          axis=1), axis=1) - 273.15
    else:
        t2m[i] = np.nanmean(np.nanmean(ds.where((ds.lon > startlon_tot)
                                                   & (ds.lon < endlon_tot)
                                                   & (ds.lat > startlat_tot)
                                                   & (ds.lat < endlat_tot))[var].values.squeeze(),
                                          axis=1), axis=1) - 273.15
    ds.close()

t2m["CRU"] = t2m["CRU"] + 273.15
t2m["APHRO"] = t2m["APHRO"] + 273.15

del colors[2]
del colors[3]
i = 0
# ----
for ds in list(t2m.keys()):
    axs[1].plot(x, t2m[ds], color=colors[i], linewidth=2, label=label[ds])
    i += 1

    # ---
    axs[1].set_title("(b) 2m Temperature [$^o$C]", loc='left', fontsize=16)
    axs[1].set_xlim([0, 11])
    axs[1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    axs[1].set_yticks([-5, 0, 5, 10, 15, 20, 25])
    axs[1].set_ylim([-5, 25])
    axs[1].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])

axs[1].legend(fontsize=12, frameon=False)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/validation/CPM/"
fig.savefig(plotpath + 'annual.png', dpi=500, transparent=True)
plt.close(fig)
