# -------------------------------------------------------------------------------
# modules
#
import matplotlib
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import Polygon
import geocat.comp
from metpy.units import units
import os

# -------------------------------------------------------------------------------
# read data
sims = ['ctrl']
mdpath = "/project/pr133/rxiang/data/cosmo"
labels = ['CTRL', 'TENV']

ptop = 100 * 100
g = 9.8


# %%
for yr in ['01', '02', '03', '04', '05']:
    for s in range(len(sims)):
        sim = sims[s]
        q = xr.open_dataset(f'{mdpath}/EAS04_{sim}/3h3D/QV/{yr}_QV.nc').QV  # kg/kg
        u = xr.open_dataset(f'{mdpath}/EAS04_{sim}/3h3D/U/{yr}_U.nc').U
        v = xr.open_dataset(f'{mdpath}/EAS04_{sim}/3h3D/V/{yr}_V.nc').V
        plev = xr.open_dataset(f'{mdpath}/EAS04_{sim}/3h3D/QV/{yr}_QV.nc').pressure  # Pa
        psfc = xr.open_dataset(f'{mdpath}/EAS04_{sim}/3h/PS/{yr}_PS.nc').PS  # Pa

        # compute the height between the layer
        dp = geocat.comp.dpres_plevel(plev, psfc, ptop)  # pa
        # Layer Mass Weighting
        dpg = dp / g
        dpg.attrs["units"] = "kg/m2"

        # 水汽通量
        uq = u * q
        vq = v * q
        uq.attrs["units"] = "(" + u.units + ")(" + q.units + ")"  # m/s g/kg
        vq.attrs["units"] = "(" + v.units + ")(" + q.units + ")"

        uq_dpg = uq * dpg.data  # m/s kg/kg kg/m2 -> g / s / m
        vq_dpg = vq * dpg.data

        # 整层水汽通量
        IUQ = uq_dpg.sum(axis=1, dtype=np.float32)  # kg/s/m
        IVQ = vq_dpg.sum(axis=1, dtype=np.float32)
        IUQ.name = 'IUQ'
        IVQ.name = 'IVQ'
        IUQ.attrs['long_name'] = 'Zonal integrated water vapour flux'
        IVQ.attrs['long_name'] = 'Meridional integrated water vapour flux'
        IUQ.attrs["units"] = "kg m-1 s-1"
        IVQ.attrs["units"] = "kg m-1 s-1"

        file = f'{mdpath}/EAS04_{sim}/3h/PS/{yr}_PS.nc'
        ds = xr.open_dataset(file)
        ds["IUQ"] = IUQ
        ds["IVQ"] = IVQ

        ds.to_netcdf(f'{mdpath}/EAS04_{sim}/3h/IVT/' + f'{yr}_IVT.nc', format="NETCDF4",
                     encoding={"time": {"_FillValue": None, 'units': "seconds since 2000-09-01T00:00:00Z", "dtype": 'double'}})
