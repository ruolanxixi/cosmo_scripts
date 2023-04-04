# -------------------------------------------------------------------------------
# modules
#
import matplotlib
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import Polygon
import metpy.calc as mpcalc
import geocat.comp
from metpy.units import units

font = {'size': 14}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
# %%
sims = ['ctrl', 'topo2']
mdpath = "/project/pr133/rxiang/data/cosmo"
watflx = {}
labels = ['CTRL', 'TENV']

for s in range(len(sims)):
    sim = sims[s]
    # COSMO 12 km
    watflx[sim] = {}
    data = xr.open_dataset(f'{mdpath}/EAS04_{sim}/monsoon/IVT/smr/01-05.IVT.smr.nc')
    qu = np.nanmean(data['IUQ'].values[:, 165:390, 248:472], axis=0)
    qv = np.nanmean(data['IVQ'].values[:, 165:390, 248:472], axis=0)
    data = xr.open_dataset(f'{mdpath}/EAS04_{sim}/monsoon/AEVAP_S/smr/01-05.AEVAP_S.smr.nc')
    evp = np.nanmean(data['AEVAP_S'].values[:, 165:390, 248:472], axis=0) / 3600
    data = xr.open_dataset(f'{mdpath}/EAS04_{sim}/monsoon/RUNOFF_G/smr/01-05.RUNOFF_G.smr.nc')
    rog = np.nanmean(data['RUNOFF_G'].values[:, 165:390, 248:472], axis=0) / 86400
    data = xr.open_dataset(f'{mdpath}/EAS04_{sim}/monsoon/RUNOFF_S/smr/01-05.RUNOFF_S.smr.nc')
    ros = np.nanmean(data['RUNOFF_S'].values[:, 165:390, 248:472], axis=0) / 86400
    data = xr.open_dataset(f'{mdpath}/EAS04_{sim}/monsoon/TOT_PREC/smr/01-05.TOT_PREC.smr.nc')
    pr = np.nanmean(data['TOT_PREC'].values[:, 165:390, 248:472], axis=0) / 86400
    lon = data['lon'].values[165:390, 248:472]
    watflx[sim]['qu'] = qu
    watflx[sim]['qv'] = qv
    watflx[sim]['evp'] = evp
    watflx[sim]['runoff'] = rog + ros
    watflx[sim]['pr'] = pr

lon = data['lon'].values[165:390, 248:472]
lat = data['lat'].values[165:390, 248:472]

env_cen = (26.50, 100.80)
env_rad = 500.0 * 1000.0
crs_wgs84 = CRS.from_epsg(4326)
crs_aeqd = CRS.from_dict({"proj": "aeqd", "lat_0": env_cen[0],
                          "lon_0": env_cen[1], "datum": "WGS84", "units": "m"})

transformer = Transformer.from_crs(crs_wgs84, crs_aeqd, always_xy=True)
x, y = transformer.transform(lon, lat)
dist = np.sqrt(x ** 2 + y ** 2)

args = (x[0, :], x[1:, -1], x[-1, -2::-1], x[-2::-1, 0])
x_ = np.concatenate(args)
args = (y[0, :], y[1:, -1], y[-1, -2::-1], y[-2::-1, 0])
y_ = np.concatenate(args)
coords = np.array(list(zip(x_, y_)))
polygon = Polygon(coords)
A = polygon.area


# %%
def get_flux(x1, y1, x2, y2, u, v, x1_, y1_, x2_, y2_):
    dx = x1 - x2
    dy = y1 - y2
    len = np.sqrt((2 * dy) ** 2 + (2 * dx) ** 2)
    dx_ = 2 * dy / len
    dy_ = -2 * dx / len
    n = [dx_, dy_]
    q = [u, v]
    dis = np.sqrt((x1_ - x2_) ** 2 + (y1_ - y2_) ** 2) / 2
    f = np.dot(n, q) * dis
    return f


for s in range(len(sims)):
    sim = sims[s]
    flux = np.zeros(lat.shape)
    for i in range(223):
        i += 1
        flux[i, 0] = get_flux(lon[i + 1, 0], lat[i + 1, 0], lon[i - 1, 0], lat[i - 1, 0],
                              watflx[sim]['qu'][i, 0], watflx[sim]['qv'][i, 0],
                              x[i + 1, 0], y[i + 1, 0], x[i - 1, 0], y[i - 1, 0])
        flux[i, 223] = get_flux(lon[i - 1, 223], lat[i - 1, 223], lon[i + 1, 223], lat[i + 1, 223],
                                watflx[sim]['qu'][i, 223], watflx[sim]['qv'][i, 223],
                                x[i - 1, 223], y[i - 1, 223], x[i + 1, 223], y[i + 1, 223], )
    for j in range(222):
        j += 1
        flux[0, j] = get_flux(lon[0, j - 1], lat[0, j - 1], lon[0, j + 1], lat[0, j + 1],
                              watflx[sim]['qu'][0, j], watflx[sim]['qv'][0, j],
                              x[0, j - 1], y[0, j - 1], x[0, j + 1], y[0, j + 1])
        flux[224, j] = get_flux(lon[223, j + 1], lat[223, j + 1], lon[223, j - 1], lat[223, j - 1],
                                watflx[sim]['qu'][224, j], watflx[sim]['qv'][224, j],
                                x[223, j + 1], y[223, j + 1], x[223, j - 1], y[223, j - 1])

    fluxin = np.nansum(flux[flux > 0])
    fluxout = - np.nansum(flux[flux < 0])

    watflx[sim]['Fin'] = fluxin * 10**(-7)
    watflx[sim]['Fout'] = fluxout * 10**(-7)

    # watflx[sim]['Fin'] = np.nansum(flux)

    watflx[sim]['E'] = - np.nanmean(watflx[sim]['evp']) * A * 10**(-7)
    watflx[sim]['R'] = np.nanmean(watflx[sim]['runoff']) * A * 10**(-7)
    watflx[sim]['P'] = np.nanmean(watflx[sim]['pr']) * A * 10**(-7)

    watflx[sim]['r'] = watflx[sim]['E']/(watflx[sim]['E']+2*watflx[sim]['Fin'])
