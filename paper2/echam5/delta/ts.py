###########################################
#%% load module
###########################################
import cartopy.crs as ccrs
import cmcrameri.cm as cmc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm
from mycolor import drywet
import matplotlib.colors as colors

###########################################
#%% load data
###########################################
path = '/project/pr133/rxiang/data/pgw/deltas/regridded/day/ECHAM5/'
#
# ts = np.nanmean(xr.open_dataset(f'{path}/ts_lgm.nc')['ts'].values[...] - 273.15, axis=0)
# tas = np.nanmean(xr.open_dataset(f'{path}/tas_lgm.nc')['tas'].values[...] - 273.15, axis=0)
# tos = np.nanmean(xr.open_dataset(f'{path}/tos_lgm.nc')['tos'].values[...] - 273.15, axis=0)
# lat = xr.open_dataset(f'{path}/ts_lgm.nc')['lat'].values[:]
# lon = xr.open_dataset(f'{path}/ts_lgm.nc')['lon'].values[:]

path1 = '/scratch/snx3000/rxiang/lmp_EAS11_lgm/wd/00090100_EAS11_lgm/int2lm_in_ctrl/'
path2 = '/scratch/snx3000/rxiang/lmp_EAS11_lgm/wd/00090100_EAS11_lgm/int2lm_in/'

before = xr.open_dataset(f'{path1}/cas20000911000000.nc')['T_SKIN'].values[0, ...] - 273.15
after = xr.open_dataset(f'{path2}/cas20000911000000.nc')['T_SKIN'].values[0, ...] - 273.15

lat = xr.open_dataset(f'{path1}/cas20000901000000.nc')['lat'].values[:]
lon = xr.open_dataset(f'{path1}/cas20000901000000.nc')['lon'].values[:]
#
# dts = xr.open_dataset(f'{path}/ts_delta.nc')['ts'].values[8, ...]
# lat = xr.open_dataset(f'{path}/ts_delta.nc')['lat'].values[:]
# lon = xr.open_dataset(f'{path}/ts_delta.nc')['lon'].values[:]

###########################################
#%% plot
###########################################
fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([60, 173, 0, 65], crs=ccrs.PlateCarree())
cmap = drywet(25, cmc.vik_r)
norm = colors.TwoSlopeNorm(vmin=-10., vcenter=0., vmax=10.)
q = ax.pcolormesh(lon, lat, before-after, cmap=cmap, clim=(-10, 10), shading="auto", transform=ccrs.PlateCarree())
# q = ax.pcolormesh(lon, lat, -dts, cmap=cmap, clim=(-10, 10), shading="auto", transform=ccrs.PlateCarree())

plt.show()

