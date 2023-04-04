# -------------------------------------------------------------------------------
# modules
#
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import math
from plotcosmomap import plotcosmo04, pole04, colorbar
import cartopy.crs as ccrs
from numpy import inf
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_
from pyproj import CRS, Transformer
import scipy.ndimage as ndimage
import matplotlib
import copy

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
    data = xr.open_dataset(f'{mdpath}/EAS04_{sim}/monsoon/TWATFLXU/smr/01-05.TWATFLXU.smr.nc')
    qu = np.nanmean(data['TWATFLXU'].values[:, :, :], axis=0)
    data = xr.open_dataset(f'{mdpath}/EAS04_{sim}/monsoon/TWATFLXV/smr/01-05.TWATFLXV.smr.nc')
    qv = np.nanmean(data['TWATFLXV'].values[:, :, :], axis=0)
    data = xr.open_dataset(f'{mdpath}/EAS04_{sim}/monsoon/AEVAP_S/smr/01-05.AEVAP_S.smr.nc')
    evp = np.nanmean(data['AEVAP_S'].values[:, :, :], axis=0) * 24 / 86400
    data = xr.open_dataset(f'{mdpath}/EAS04_{sim}/monsoon/RUNOFF_G/smr/01-05.RUNOFF_G.smr.nc')
    rog = np.nanmean(data['RUNOFF_G'].values[:, :, :], axis=0) / 86400
    data = xr.open_dataset(f'{mdpath}/EAS04_{sim}/monsoon/RUNOFF_S/smr/01-05.RUNOFF_S.smr.nc')
    ros = np.nanmean(data['RUNOFF_S'].values[:, :, :], axis=0) / 86400
    data = xr.open_dataset(f'{mdpath}/EAS04_{sim}/monsoon/TOT_PREC/smr/01-05.TOT_PREC.smr.nc')
    pr = np.nanmean(data['TOT_PREC'].values[:, :, :], axis=0) / 86400
    watflx[sim]['qu'] = qu
    watflx[sim]['qv'] = qv
    watflx[sim]['evp'] = evp
    watflx[sim]['runoff'] = rog + ros
    watflx[sim]['pr'] = pr

env_cen = (26.50, 100.80)
env_rad = 500.0 * 1000.0
crs_wgs84 = CRS.from_epsg(4326)
crs_aeqd = CRS.from_dict({"proj": "aeqd", "lat_0": env_cen[0],
                          "lon_0": env_cen[1], "datum": "WGS84", "units": "m"})

[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole04()

transformer = Transformer.from_crs(crs_wgs84, crs_aeqd, always_xy=True)
x, y = transformer.transform(lon, lat)
dist = np.sqrt(x ** 2 + y ** 2)
mask = dist <= env_rad

A = np.pi * env_rad**2
# 635 is num of grid in the boundary
dx = 2 * np.pi * env_rad / 635

def get_normals(x, y, norm, u, v):
    dx = - x / norm
    dy = - y / norm
    f = dx * u + dy * v
    return f


for s in range(len(sims)):
    sim = sims[s]
    flux = get_normals(x, y, dist, watflx[sim]['qu'], watflx[sim]['qv'])
    mask_flux = flux * mask
    mask_flux[mask_flux == 0] = np.nan
    flux_bd = copy.deepcopy(mask_flux)
    df = pd.DataFrame(mask_flux)

    for i in range(650):
        for j in range(650):
            var = df.iloc[i - 1:i + 2, j - 1: j + 2]
            if var.isna().sum().sum() <= 1:
                flux_bd[i][j] = np.nan


    fluxin = np.nansum(flux_bd[:, 0:361])
    fluxout = np.nansum(flux_bd[:, 361:])

    watflx[sim]['Fin'] = fluxin * dx
    watflx[sim]['Fout'] = - fluxout * dx

    mask_evp = watflx[sim]['evp'] * mask
    mask_evp[mask_evp == 0] = np.nan
    watflx[sim]['E'] = np.nansum(mask_evp)

    mask_runoff = watflx[sim]['runoff'] * mask
    mask_runoff[mask_runoff == 0] = np.nan
    watflx[sim]['R'] = np.nansum(mask_runoff)

    mask_pr = watflx[sim]['pr'] * mask
    mask_pr[mask_pr == 0] = np.nan
    watflx[sim]['P'] = np.nansum(mask_pr)

