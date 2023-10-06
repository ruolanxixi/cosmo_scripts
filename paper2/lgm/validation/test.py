# %%
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy as ctp
import xesmf as xe
import metpy.calc as mpcalc
from metpy.interpolate import cross_section
from metpy.units import units
from scipy import interpolate
from haversine import haversine
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pvlib.atmosphere as pva
from mycolor import prcp, custom_div_cmap
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as colors
from auxiliary import truncate_colormap
import scipy.ndimage as ndimage

def rotate_points(lon_pole, lat_pole, rlon, rlat):
    crs_geo = ccrs.PlateCarree()
    crs_rot_pole = ccrs.RotatedPole(pole_longitude=lon_pole,
                                    pole_latitude=lat_pole)
    lon, lat = crs_geo.transform_point(rlon, rlat, crs_rot_pole)
    return lon, lat

def rotate_points_to(lon_pole, lat_pole, lon, lat):
    crs_geo = ccrs.PlateCarree()
    crs_rot_pole = ccrs.RotatedPole(pole_longitude=lon_pole,
                                    pole_latitude=lat_pole)
    rlon, rlat = crs_rot_pole.transform_point(lon, lat, crs_geo)
    return rlon, rlat

# %%
font = {'size': 13}
matplotlib.rc('font', **font)

ds1 = xr.open_dataset('/project/pr133/rxiang/data/pmip/var/ta/ta_Amon_PMIP4_lgm_timmean.nc')
ds2 = xr.open_dataset('/project/pr133/rxiang/data/pmip/var/hur/hur_Amon_PMIP4_lgm_timmean.nc')

# create regional latlon grid and regridder
target_grid = xe.util.grid_2d(lon0_b=75, lon1_b=165, d_lon=0.11, lat0_b=0., lat1_b=50., d_lat=0.11)
regridder = xe.Regridder(ds1, target_grid, 'bilinear')
t = regridder(ds1.ta) - 273.15
t.name = 'T'
u = regridder(ds2.hur)
u.name = 'RH'

ds = xr.merge([t, u])
ds = ds.squeeze('time')
ds = ds.metpy.assign_crs(grid_mapping_name='latitude_longitude', earth_radius=6371229.0)

# create cross section
lon_pole, lat_pole = -63.7, 61
start_rlat = -2.23
start_rlon = -24.8
end_rlat = 3.00
end_rlon = -2.96
start_lon, start_lat = rotate_points(lon_pole, lat_pole, start_rlon, start_rlat)
end_lon, end_lat = rotate_points(lon_pole, lat_pole, end_rlon, end_rlat)
start_lon, start_lat = 75, 15
end_lon, end_lat = 165, 15
distance = haversine((start_lat, start_lon), (end_lat, end_lon))
start = (start_lat, start_lon)
end = (end_lat, end_lon)
ds['y'] = ds['lat'].values[:, 0]
ds['x'] = ds['lon'].values[0, :]

p = ds['plev'].values
height = pva.pres2alt(p)
ds = ds.assign_coords(height=('plev', height))
ds = ds.swap_dims({'plev': 'height'})
#%%
cross = cross_section(ds, start, end, steps=int(distance/2.2)+1).set_coords(('lat', 'lon'))
cross['T'] = cross['T'] * units.kelvin

# %%
fig = plt.figure(figsize=(10, 10.3))
axs = fig.add_subplot()
pres = np.array([100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000,
    20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100])
alts = [alt for alt in pva.pres2alt(pres) if alt <= 14000]
axs.set_yticks([2000, 4000, 6000, 8000, 10000, 12000])
axs.set_ylim(100, pva.pres2alt(15000))
axs.set_yticklabels([2, 4, 6, 8, 10, 12], fontsize=15)
axs.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
axp = axs.twinx()
# axp.set_ylim(100, pva.pres2alt(15000))
axp.set_yticks(alts[:])
axp.set_yticklabels(list(map(int, pres[:len(alts)] / 100)), fontsize=15)
axp.set_ylabel('Pressure (hPa)', fontsize=15, labelpad=13, rotation=270)

axs.set_xlim(80, 160)
axs.set_xticks([80, 100, 120, 140, 160])
axs.set_xticklabels([u'80\N{DEGREE SIGN}E', u'100\N{DEGREE SIGN}E', u'120\N{DEGREE SIGN}E', u'140\N{DEGREE SIGN}E', u'160\N{DEGREE SIGN}E'], fontsize=15)
axs.xaxis.set_label_coords(1.06, -0.018)

level = np.linspace(-80, 20, 40, endpoint=True)
cmap = cmc.roma
axs.contourf(cross['lon'], cross['height'], cross['T'], levels=level, cmap=cmap, norm=BoundaryNorm(level, ncolors=cmap.N, clip=True), extend='max')
plt.show()



