import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader  # 读取 .shp
import metpy.calc as mpcalc
import geocat.comp
from metpy.units import units
import cfgrib

# 读取省界
# shpName = 'Province_9.shp'
# reader = Reader(shpName)
# provinces = cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='k', facecolor='none')

# 需要的层次
levels = [1000.,  975.,  950.,  925.,  900.,  850.,  800.,  750.,  700.,  650.,
          600.,  550.,  500.,  450.,  400.,  350.,  300.,  250.,  200.]

# 常数
ptop = 200*100 # 垂直积分层顶，单位Pa
g = 9.8

# 读取文件
files = '/users/rxiang/' # 这里改成自己的数据路径
file = 'pl_2014070506.grib'
file_sl = "sl_2014070506.grib"
ds = cfgrib.open_datasets(files+file)
ds_sl = cfgrib.open_datasets(files+file_sl)
# 读取变量
u_levels = ds[0].u.sel(isobaricInhPa=levels)
v_levels = ds[0].v.sel(isobaricInhPa=levels)
q_levels = ds[0].q.sel(isobaricInhPa=levels) # kg/kg
psfc = ds_sl[4].sp # 为了在垂直积分时考虑地形，计算时要用到地表气压，单位Pa

plev = u_levels.coords["isobaricInhPa"] * 100 # 单位变为Pa
plev.attrs['units'] = 'Pa'
q_levels = q_levels * 1000 # kg/kg -> g/kg
q_levels.attrs['units'] = 'g/kg'

# 垂直积分的关键，计算各层次厚度，效果类似ncl中的dpres_plevel
dp = geocat.comp.dpres_plevel(plev, psfc, ptop)
# Layer Mass Weighting
dpg = dp / g
dpg.attrs["units"] = "kg/m2"

# %%
# 水汽通量
uq = u_levels * q_levels
vq = v_levels * q_levels
uq.attrs["units"] = "("+u_levels.units+")("+q_levels.units+")"
vq.attrs["units"] = "("+v_levels.units+")("+q_levels.units+")"

uq_dpg = uq * dpg.data
vq_dpg = vq * dpg.data

# 整层水汽通量
iuq = uq_dpg.sum(axis=0)
ivq = vq_dpg.sum(axis=0)
iuq.attrs["units"] = "[m/s][g/kg]"
ivq.attrs["units"] = "[m/s][g/kg]"

# 计算散度要用到dx，dy
lons = u_levels.coords['longitude'][:]
lats = v_levels.coords['latitude'][:]
dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)
# 水汽通量散度
duvq = mpcalc.divergence(uq/g, vq/g, dx=dx[np.newaxis,:,:], dy=dy[np.newaxis,:,:])
duvq.attrs["units"] = "g/(kg-s)"
duvq_dpg = duvq * dpg.data
# 整层水汽通量散度
iduvq = duvq_dpg.sum(axis=0)
iduvq = iduvq * units('kg/m^2')
iduvq.attrs["units"] = "g/(m2-s)"
