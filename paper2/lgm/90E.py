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
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_

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

# %% -------------------------------------------------------------------------------
nrow = 2
ncol = 2
axs, q, ct = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
               np.empty(shape=(nrow, ncol), dtype='object')
fig = plt.figure(figsize=(18, 10.3))
gs = gridspec.GridSpec(2, 2, left=0.04, bottom=0.03, right=0.887,
                       top=0.97, hspace=0.07, wspace=0.09, width_ratios=[1, 1], height_ratios=[1, 1])
level1 = np.arange(280, 400, 5)
level2 = np.arange(250, 380, 5)
levels= [level1, level2]
# %%
font = {'size': 13}
matplotlib.rc('font', **font)
# create regional latlon grid and regridder
ds = xr.open_dataset('/scratch/snx3000/rxiang/data/cosmo/EAS11_ctrl/monsoon/U/01-05.U.jan.timmean.lonlat.nc')
target_grid = xe.util.grid_2d(lon0_b=75, lon1_b=165, d_lon=0.11, lat0_b=0., lat1_b=50., d_lat=0.11)
regridder = xe.Regridder(ds, target_grid, 'bilinear')
# # COSMO
data = {}
dir = "/project/pr133/rxiang/data/cosmo/"
sims = ["ctrl", "lgm"]
mons = ["jan", 'jul']
for i in range(2):
    sim = sims[i]
    data[sim] = {}
    for j in range(2):
        mon = mons[j]
        level = levels[j]
        ds1 = xr.open_dataset(f'/scratch/snx3000/rxiang/data/cosmo/EAS11_{sim}/monsoon/V/01-05.V.{mon}.timmean.lonlat.nc')
        ds1 = ds1.isel(pressure=slice(None, None, -1))
        data_var = ds1['V']
        zonal_avg = data_var.sel(lon=slice(88, 92)).mean(dim='lon', skipna=True).values
        updated_data_var = data_var.copy(deep=True)
        updated_data_var.loc[dict(lon=slice(88, 92))] = zonal_avg[:, :, :, np.newaxis]
        ds1['V'] = updated_data_var
        # ds2 = xr.open_dataset('/scratch/snx3000/rxiang/data/cosmo/EAS11_ctrl/monsoon/V/01-05.V.jan.nc')
        # ds2 = ds2.isel(pressure=slice(None, None, -1))
        # data_var = ds2['V']
        # zonal_avg = data_var.sel(lon=slice(89, 91)).mean(dim='lon', skipna=True).values
        # updated_data_var = data_var.copy(deep=True)
        # updated_data_var.loc[dict(lon=slice(89, 91))] = zonal_avg[:, :, np.newaxis, :]
        # ds2['V'] = updated_data_var
        ds3 = xr.open_dataset(f'/scratch/snx3000/rxiang/data/cosmo/EAS11_{sim}/monsoon/W/01-05.W.{mon}.timmean.lonlat.nc')
        ds3 = ds3.isel(pressure=slice(None, None, -1))
        data_var = ds3['W'] * 500
        zonal_avg = data_var.sel(lon=slice(88, 92)).mean(dim='lon', skipna=True).values
        updated_data_var = data_var.copy(deep=True)
        updated_data_var.loc[dict(lon=slice(88, 92))] = zonal_avg[:, :, :, np.newaxis]
        ds3['W'] = updated_data_var
        ds4 = xr.open_dataset(f'/scratch/snx3000/rxiang/data/cosmo/EAS11_{sim}/monsoon/T/01-05.T.{mon}.timmean.lonlat.nc')
        ds4 = ds4.isel(pressure=slice(None, None, -1))
        data_var = ds4['T']
        zonal_avg = data_var.sel(lon=slice(88, 92)).mean(dim='lon', skipna=True).values
        updated_data_var = data_var.copy(deep=True)
        updated_data_var.loc[dict(lon=slice(88, 92))] = zonal_avg[:, :, :, np.newaxis]
        ds4['T'] = updated_data_var

        if sim == "ctrl":
            f = xr.open_dataset(f'/project/pr133/rxiang/data/extpar/extpar_ctrl_lonlat.nc')
        else:
            f = xr.open_dataset(f'/project/pr133/rxiang/data/extpar/extpar_lgm_lonlat.nc')

        time = ds1['time'].values
        ds4 = ds4.assign_coords(time=('time', time))

        v = regridder(ds1.V)
        v.name = 'V'
        w = regridder(ds3.W)
        w.name = 'W'
        t = regridder(ds4.T)
        t.name = 'T'
        hsurf = regridder(f.HSURF)
        hsurf.name = 'HSURF'


        ds_cosmo = xr.merge([v, w, t])
        ds_cosmo = ds_cosmo.squeeze('time')
        ds_cosmo = xr.merge([ds_cosmo, hsurf])
        ds_cosmo = ds_cosmo.metpy.assign_crs(grid_mapping_name='latitude_longitude', earth_radius=6371229.0)

        # create cross section
        lon_pole, lat_pole = -63.7, 61
        start_lon, start_lat = 90, 60
        end_lon, end_lat = 90, 5
        distance = haversine((start_lat, start_lon), (end_lat, end_lon))
        start = (start_lat, start_lon)
        end = (end_lat, end_lon)
        ds_cosmo['y'] = ds_cosmo['lat'].values[:, 0]
        ds_cosmo['x'] = ds_cosmo['lon'].values[0, :]
        
        p = ds_cosmo['pressure'].values
        height = pva.pres2alt(p)
        ds_cosmo = ds_cosmo.assign_coords(height=('pressure', height))
        ds_cosmo = ds_cosmo.swap_dims({'pressure': 'height'})
        
        cross = cross_section(ds_cosmo, start, end, steps=int(distance/2.2)+1).set_coords(('lat', 'lon'))
        cross['V'] = cross['V'] * units('m/s')
        cross['W'] = cross['W'] * units('m/s')
        cross['omega'] = mpcalc.vertical_velocity_pressure(cross['W'], cross['pressure'] * units.Pa, cross['T'] * units.kelvin) * -1 * 10
        cross['theta'] = mpcalc.potential_temperature(cross['pressure'] * units.Pa, cross['T'] * units.kelvin)

        # ----------------------------------------------------------------------------------
        # plot
        axs[i, j] = fig.add_subplot(gs[i, j])
        pres = np.array([100000, 85000, 70000, 60000, 50000, 40000, 30000, 20000, 15000])
        alts = [alt for alt in pva.pres2alt(pres) if alt <= 14000]
        axs[i, j].set_yticks([2000, 4000, 6000, 8000, 10000, 12000])
        axs[i, j].set_ylim(100, pva.pres2alt(15000))
        axs[i, j].set_yticklabels([2, 4, 6, 8, 10, 12], fontsize=15)
        axs[i, j].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        axs[i, j].set_xlim(5, 50)

        if j == 1:
            axp = axs[i, j].twinx()

            axp.set_ylim(100, pva.pres2alt(15000))
            axp.set_yticks(alts[:])
            axp.set_yticklabels(list(map(int, pres[:len(alts)] / 100)), fontsize=15)
            axp.set_ylabel('Pressure (hPa)', fontsize=15, labelpad=13, rotation=270)

        if j == 0:
            axs[i, j].set_ylabel('Height (km)', fontsize=15, labelpad=2, rotation=90)

        if i == nrow - 1:
            axs[i, j].set_xticks([10, 20, 30, 40, 50])
            axs[i, j].set_xticklabels(
                [u'10\N{DEGREE SIGN}N', u'20\N{DEGREE SIGN}N', u'30\N{DEGREE SIGN}N', u'40\N{DEGREE SIGN}N',
                 u'50\N{DEGREE SIGN}N'], fontsize=15)
            axs[i, j].xaxis.set_label_coords(1.06, -0.018)
        else:
            axs[i, j].set_xticks([])

        ct[i, j] = axs[i, j].contour(cross['lat'], cross['height'], cross['theta'], levels=level, colors='black')
        q[i, j] = axs[i, j].quiver(cross['lat'][::100], cross['height'], cross['V'][:, ::100], cross['W'][:, ::100], scale=100)

        axs[i, j].plot(cross['lat'], cross['HSURF'], color='black', linewidth=0.01)
        axs[i, j].fill_between(cross['lat'], cross['HSURF'], color='black')


plt.show()
