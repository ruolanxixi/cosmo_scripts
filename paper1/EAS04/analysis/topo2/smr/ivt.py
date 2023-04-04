# -------------------------------------------------------------------------------
# modules
#
import numpy as np
import xarray as xr
import geocat.comp
import sys
import metpy.calc as mpcalc
import os

# -------------------------------------------------------------------------------
# read data
mdpath = "/scratch/snx3000/rxiang/test"

ptop = 100 * 100
g = 9.8

q = xr.open_mfdataset(f'{mdpath}/01_QV.nc', chunks={'time': '500MB'}).QV  # kg/kg
u = xr.open_mfdataset(f'{mdpath}/01_U.nc', chunks={'time': '500MB'}).U
v = xr.open_mfdataset(f'{mdpath}/01_V.nc', chunks={'time': '500MB'}).V
plev = xr.open_mfdataset(f'{mdpath}/01_QV.nc', chunks={'time': '500MB'}).pressure  # Pa
psfc = xr.open_mfdataset(f'{mdpath}/01_PS.nc', chunks={'time': '500MB'}).PS  # Pa

# compute the height between the layer
dp = geocat.comp.dpres_plevel(plev, psfc, ptop)  # pa
# Layer Mass Weighting
dpg = dp / g  # kg/m2
# del plev, psfc, ptop, dp

# 水汽通量
uq = u * q
vq = v * q

uq_dpg = uq * dpg.values  # m/s kg/kg kg/m2 -> kg / s / m
vq_dpg = vq * dpg.values

# 散度
lons = q.lon.values
lats = q.lat.values
dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)
# del u, q, v, lons, lats

uq_dpg = uq * dpg.values  # m/s kg/kg kg/m2 -> g/s/m
vq_dpg = vq * dpg.values

duvq = mpcalc.divergence(uq/g, vq/g, dx=dx[np.newaxis, np.newaxis, :, :], dy=dy[np.newaxis, np.newaxis, :, :], x_dim=-1, y_dim=-2)  # s-1 kg/kg
duvq_dpg = duvq * dp.values  # s-1 kg/kg kg/m2 -> kg/m2/s
# del uq, vq, dpg, dx, dy, duvq

#%%
# 整层水汽通量
IUQ = uq_dpg.sum(axis=1, dtype=np.float32)  # kg/s/m
IVQ = vq_dpg.sum(axis=1, dtype=np.float32)
IUQ.name = 'IUQ'
IVQ.name = 'IVQ'
IUQ.attrs['long_name'] = 'Zonal integrated water vapour flux'
IVQ.attrs['long_name'] = 'Meridional integrated water vapour flux'
IUQ.attrs["units"] = "kg m-1 s-1"
IVQ.attrs["units"] = "kg m-1 s-1"

# del uq_dpg, vq_dpg

# 整层水汽散度
VIMD = duvq_dpg.sum(axis=1, dtype=np.float32)
VIMD.name = 'VIMD'
VIMD.attrs['long_name'] = 'Vertically integrated moisture divergence'
VIMD.attrs["units"] = "kg m-2 s-1"

file = f'{mdpath}/01_PS.nc'
ds = xr.open_dataset(file)
ds["IUQ"] = IUQ
ds["IVQ"] = IVQ
ds["VIMD"] = VIMD

# save file to netcdf
file = f'{mdpath}/' + 'IVT.nc'
if os.path.exists(file):
    os.remove(file)
else:
    print("Can not delete the file as it doesn't exists")

ds.to_netcdf(file, format="NETCDF4",
             encoding={"time": {"_FillValue": None, 'units': "seconds since 2000-09-01T00:00:00Z", "dtype": 'double'}})
