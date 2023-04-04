# -----------------------------------------------------------------------------
# import module
# -----------------------------------------------------------------------------
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.colors as colors
from copy import copy
from plotcosmomap import plotcosmo, pole, colorbar
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from cmcrameri import cm
from auxiliary import truncate_colormap, spat_agg_1d, spat_agg_2d
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import pvlib.atmosphere as pla
import matplotlib.tri as tri
from scipy import interpolate

# -----------------------------------------------------------------------------
# Load topography data
# -----------------------------------------------------------------------------
path = "/project/pr94/rxiang/data/extpar/"
file1 = 'extpar_12km_1118x670_MERIT_raw_remap.nc'
file2 = 'extpar_EAS_ext_12km_merit_adj_remap.nc'

ds = xr.open_dataset(path + file1)
ds = ds.sel(lon=slice(95, 105), lat=slice(0, 65))
elev_ctrl = ds["HSURF"].values[:, :]
elev_ctrl = np.nanmean(elev_ctrl, axis=1)
lat = ds["lat"].values
lon = ds["lon"].values
ds.close()

ds = xr.open_dataset(path + file2)
ds = ds.sel(lon=slice(95, 105), lat=slice(0, 65))
elev_topo1 = ds["HSURF"].values[:, :]
elev_topo1 = np.nanmean(elev_topo1, axis=1)
ds.close()

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
season = 'JJA'
mdvnames = ['SOHR_SUM', 'THHR_SUM', 'TTTUR_SUM', 'TCONVLH_SUM', 'TMPHYS_SUM']  # edit here
year = '2001-2005'
ctrlpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/zonmean/"
topo1path = "/project/pr133/rxiang/data/cosmo/EAS11_topo1/zonmean/"
pressure = [100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 20000, 10000]

def read_data(mdvname):
    da_ctrl, da_topo1, da_diff = np.arange(1,1001,1), np.arange(1,1001,1), np.arange(1,1001,1)
    for i in range(len(pressure)):
        p = pressure[i]
        filename = f'{year}.{mdvname}.{p}.{season}.95-105zm.nc'
        lat = xr.open_dataset(f'{ctrlpath}{mdvname}/{filename}')["lat"].values
        lon = xr.open_dataset(f'{ctrlpath}{mdvname}/{filename}')["lon"].values
        data_ctrl = xr.open_dataset(f'{ctrlpath}{mdvname}/{filename}')[mdvname].values[0, 0, :, 0]
        data_topo1 = xr.open_dataset(f'{topo1path}{mdvname}/{filename}')[mdvname].values[0, 0, :, 0]
        data_diff = data_topo1 - data_ctrl
        da_ctrl = np.vstack([da_ctrl, data_ctrl])
        da_topo1 = np.vstack([da_topo1, data_topo1])
        da_diff = np.vstack([da_diff, data_diff])
    da_ctrl, da_topo1, da_diff = da_ctrl[1:, :], da_topo1[1:, :], da_diff[1:, :]

    return da_ctrl, da_topo1, da_diff

[SOHR_ctrl, SOHR_topo1, SOHR_diff] = read_data('SOHR_SUM')
[THHR_ctrl, THHR_topo1, THHR_diff] = read_data('THHR_SUM')
[TTTUR_ctrl, TTTUR_topo1, TTTUR_diff] = read_data('TTTUR_SUM')
[TCONVLH_ctrl, TCONVLH_topo1, TCONVLH_diff] = read_data('TCONVLH_SUM')
[TMPHYS_ctrl, TMPHYS_topo1, TMPHYS_diff] = read_data('TMPHYS_SUM')

DIAB_ctrl = SOHR_ctrl + THHR_ctrl + TTTUR_ctrl + TCONVLH_ctrl + TMPHYS_ctrl
DIAB_topo1 = SOHR_topo1 + THHR_topo1 + TTTUR_topo1 + TCONVLH_topo1 + TMPHYS_topo1
DIAB_diff = SOHR_diff + THHR_diff + TTTUR_diff + TCONVLH_diff + TMPHYS_diff

datalat = xr.open_dataset(f'{ctrlpath}/SOHR_SUM/{year}.SOHR_SUM.10000.{season}.95-105zm.nc')["lat"].values
datalon = xr.open_dataset(f'{ctrlpath}/SOHR_SUM/{year}.SOHR_SUM.10000.{season}.95-105zm.nc')["lon"].values
# -----------------------------------------------------------------------------
# Convert elevation to hPa
# -----------------------------------------------------------------------------
z = np.arange(0, 14000, 1)
p = pla.alt2pres(z)
# -----------------------------------------------------------------------------
# Cross-section plot
# -----------------------------------------------------------------------------
gs = gridspec.GridSpec(1, 3)
gs.update(left=0.03, right=0.99, top=0.98, bottom=0.05, hspace=0.1, wspace=0.20)
fig = plt.figure(figsize=(18, 4.5), constrained_layout=True)

ax0 = plt.subplot(gs[0])
plt.fill_between(lat, 0.0, elev_ctrl, color="lightgray")
plt.plot(lat, elev_ctrl, color="black", lw=1.0)
plt.yticks(range(0, 14000, 2000))
plt.ylabel("Elevation [m]")
plt.axis([0, 65, 0.0, 14000.0])
labels = [0, 2, 4, 6, 8, 10, 12]
ax0.set_yticklabels(labels)

ax1 = plt.subplot(gs[1])
plt.fill_between(lat, 0.0, elev_topo1, color="lightgray")
plt.plot(lat, elev_topo1, color="black", lw=1.0)
plt.yticks(range(0, 14000, 2000))
plt.ylabel("Elevation [m]")
plt.axis([0, 65, 0.0, 14000.0])
labels = [0, 2, 4, 6, 8, 10, 12]
ax1.set_yticklabels(labels)

ax2 = plt.subplot(gs[2])
plt.fill_between(lat, 0.0, elev_ctrl - elev_topo1, color="lightgray")
plt.plot(lat, elev_ctrl - elev_topo1, color="black", lw=1.0)
plt.yticks(range(0, 14000, 2000))
plt.ylabel("Elevation [km]")
plt.axis([0, 65, 0.0, 14000.0])
labels = [0, 2, 4, 6, 8, 10, 12]
ax2.set_yticklabels(labels)

fig.show()

p = np.array(pressure)
height = pla.pres2alt(p)
DIAB_ctrl_int = np.empty(shape=(1000, 1000))

for i in range(len(datalat)):
    y = np.linspace(112, 15797, 1000, endpoint=True)
    DIAB_ctrl_int[:, i] = np.interp(y, height, DIAB_ctrl[:, i])

x = datalat
x = np.concatenate((x, np.tile(x,999)))
y = np.linspace(112, 15797, 1000, endpoint=True)
y = np.concatenate((y, np.tile(y,999)))
triang = tri.Triangulation(x, y)
# y = np.linspace(112, 15797, 1000, endpoint=True)
# triang = tri.Triangulation(datalat, y)
interpolator = tri.LinearTriInterpolator(triang, DIAB_ctrl_int)
Xi, Yi = np.meshgrid(datalat, y)
zi = interpolator(Xi, Yi)