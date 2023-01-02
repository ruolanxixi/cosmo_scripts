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
rgas = 287.058            # J/(kg-K) => m2/(s2 K)

q = xr.open_mfdataset(f'{mdpath}/01_QV.nc', chunks={'time': '500MB'}).QV  # kg/kg
w = xr.open_mfdataset(f'{mdpath}/01_W.nc', chunks={'time': '500MB'}).W
t = xr.open_mfdataset(f'{mdpath}/01_T.nc', chunks={'time': '500MB'}).T
plev = xr.open_mfdataset(f'{mdpath}/01_QV.nc', chunks={'time': '500MB'}).pressure  # Pa
psfc = xr.open_mfdataset(f'{mdpath}/01_PS.nc', chunks={'time': '500MB'}).PS  # Pa


# compute the height between the layer
dp = geocat.comp.dpres_plevel(plev, psfc, ptop)  # pa

# Layer Mass Weighting
dpg = dp / g  # kg/m2

# del plev, psfc, ptop, dp

# omega
rho = plev/(rgas*t)        # density => kg/m3
omega = -w*rho*g           # Pa/s
mfc = omega*q*dp


#%%
# 整层水汽通量
VMFC= mfc.sum(axis=1, dtype=np.float32)  # kg/s/m
VMFC.name = 'VMFC'
VMFC.attrs['long_name'] = 'Vertically integrated vertical moisture flux divergence'
IUQ.attrs["units"] = "kg m-1 s-1"

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
