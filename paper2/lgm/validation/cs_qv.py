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

# %%
font = {'size': 13}
matplotlib.rc('font', **font)

# # # COSMO
# ds1 = xr.open_dataset('/scratch/snx3000/rxiang/data/cosmo/EAS11_ctrl/monsoon/QV/01-05.QV.timmean.nc')
# ds1 = ds1.isel(pressure=slice(None, None, -1))
# data_var = ds1['QV']
# meridional_avg = data_var.sel(lat=slice(15, 25)).mean(dim='lat', skipna=True).values
# updated_data_var = data_var.copy(deep=True)
# updated_data_var.loc[dict(lat=slice(15, 25))] = meridional_avg[:, :, np.newaxis, :]
# ds1['QV'] = updated_data_var
# ds2 = xr.open_dataset('/scratch/snx3000/rxiang/data/cosmo/EAS11_lgm/monsoon/QV/01-05.QV.timmean.nc')
# ds2 = ds2.isel(pressure=slice(None, None, -1))
# data_var = ds2['QV']
# meridional_avg = data_var.sel(lat=slice(15, 25)).mean(dim='lat', skipna=True).values
# updated_data_var = data_var.copy(deep=True)
# updated_data_var.loc[dict(lat=slice(15, 25))] = meridional_avg[:, :, np.newaxis, :]
# ds2['QV'] = updated_data_var
#
# time = ds1['time'].values
# ds2 = ds2.assign_coords(time=('time', time))
#
# # create regional latlon grid and regridder
# target_grid = xe.util.grid_2d(lon0_b=75, lon1_b=165, d_lon=0.11, lat0_b=0., lat1_b=50., d_lat=0.11)
# regridder = xe.Regridder(ds1, target_grid, 'bilinear')
# cosmo1 = regridder(ds1.QV)
# cosmo1.name = 'QV1'
# cosmo2 = regridder(ds2.QV)
# cosmo2.name = 'QV2'
#
# ds_cosmo = xr.merge([cosmo1, cosmo2])
# ds_cosmo = ds_cosmo.squeeze('time')
# ds_cosmo = ds_cosmo.metpy.assign_crs(grid_mapping_name='latitude_longitude', earth_radius=6371229.0)
#
# # create cross section
# lon_pole, lat_pole = -63.7, 61
# start_lon, start_lat = 75, 15
# end_lon, end_lat = 165, 15
# distance = haversine((start_lat, start_lon), (end_lat, end_lon))
# start = (start_lat, start_lon)
# end = (end_lat, end_lon)
# ds_cosmo['y'] = ds_cosmo['lat'].values[:, 0]
# ds_cosmo['x'] = ds_cosmo['lon'].values[0, :]
#
# p = ds_cosmo['pressure'].values
# height = pva.pres2alt(p)
# ds_cosmo = ds_cosmo.assign_coords(height=('pressure', height))
# ds_cosmo = ds_cosmo.swap_dims({'pressure': 'height'})
#
# cross_cosmo = cross_section(ds_cosmo, start, end, steps=int(distance/2.2)+1).set_coords(('lat', 'lon'))
# cross_cosmo['QV1'] = cross_cosmo['QV1'] * units.kelvin
# cross_cosmo['QV2'] = cross_cosmo['QV2'] * units.kelvin
#
# # %%
# # ECHAM5
# # --
# ds1 = xr.open_dataset('/project/pr133/rxiang/data/pgw/deltas/native/day/ECHAM5/hus_piControl_timmean.nc')
# data_var = ds1['hus']
# meridional_avg = data_var.sel(lat=slice(25, 15)).mean(dim='lat', skipna=True).values
# updated_data_var = data_var.copy(deep=True)
# updated_data_var.loc[dict(lat=slice(25, 15))] = meridional_avg[:, :, np.newaxis, :]
# ds1['hus'] = updated_data_var
# ds2 = xr.open_dataset('/project/pr133/rxiang/data/pgw/deltas/native/day/ECHAM5/hus_lgm_timmean.nc')
# data_var = ds2['hus']
# meridional_avg = data_var.sel(lat=slice(25, 15)).mean(dim='lat', skipna=True).values
# updated_data_var = data_var.copy(deep=True)
# updated_data_var.loc[dict(lat=slice(25, 15))] = meridional_avg[:, :, np.newaxis, :]
# ds2['hus'] = updated_data_var
#
# time = ds1['time'].values
# ds2 = ds2.assign_coords(time=('time', time))
#
# # create regional latlon grid and regridder
# target_grid = xe.util.grid_2d(lon0_b=75, lon1_b=165, d_lon=0.11, lat0_b=0., lat1_b=50., d_lat=0.11)
# regridder = xe.Regridder(ds1, target_grid, 'bilinear')
# echam1 = regridder(ds1.hus)
# echam1.name = 'QV1'
# echam2 = regridder(ds2.hus)
# echam2.name = 'QV2'
#
# ds_echam = xr.merge([echam1, echam2])
# ds_echam = ds_echam.squeeze('time')
# ds_echam = ds_echam.metpy.assign_crs(grid_mapping_name='latitude_longitude', earth_radius=6371229.0)
#
# # create cross section
# lon_pole, lat_pole = -63.7, 61
# start_lon, start_lat = 75, 15
# end_lon, end_lat = 165, 15
# distance = haversine((start_lat, start_lon), (end_lat, end_lon))
# start = (start_lat, start_lon)
# end = (end_lat, end_lon)
# ds_echam['y'] = ds_echam['lat'].values[:, 0]
# ds_echam['x'] = ds_echam['lon'].values[0, :]
#
# p = ds_echam['lev'].values
# height = pva.pres2alt(p)
# ds_echam = ds_echam.assign_coords(height=('lev', height))
# ds_echam = ds_echam.swap_dims({'lev': 'height'})
#
# cross_echam = cross_section(ds_echam, start, end, steps=int(distance/2.2)+1).set_coords(('lat', 'lon'))
# cross_echam['QV1'] = cross_echam['QV1'] * units.kelvin
# cross_echam['QV2'] = cross_echam['QV2'] * units.kelvin

# %%
# PMIP
# --
ds1 = xr.open_dataset('/project/pr133/rxiang/data/pmip/var/hus/hus_Amon_PMIP4_piControl_timmean.nc')
data_var = ds1['hus']
meridional_avg = data_var.sel(lat=slice(15, 25)).mean(dim='lat', skipna=True).values
updated_data_var = data_var.copy(deep=True)
updated_data_var.loc[dict(lat=slice(15, 25))] = meridional_avg[:, :, np.newaxis, :]
ds1['hus'] = updated_data_var
ds2 = xr.open_dataset('/project/pr133/rxiang/data/pmip/var/hus/hus_Amon_PMIP4_lgm_timmean.nc')
data_var = ds2['hus']
meridional_avg = data_var.sel(lat=slice(15, 25)).mean(dim='lat', skipna=True).values
updated_data_var = data_var.copy(deep=True)
updated_data_var.loc[dict(lat=slice(15, 25))] = meridional_avg[:, :, np.newaxis, :]
ds2['hus'] = updated_data_var
time = ds1['time'].values
ds2 = ds2.assign_coords(time=('time', time))

# create regional latlon grid and regridder
target_grid = xe.util.grid_2d(lon0_b=75, lon1_b=165, d_lon=0.11, lat0_b=0., lat1_b=50., d_lat=0.11)
regridder = xe.Regridder(ds1, target_grid, 'bilinear')
pmip1 = regridder(ds1.hus)
pmip1.name = 'QV1'
pmip2 = regridder(ds2.hus)
pmip2.name = 'QV2'

ds_pmip = xr.merge([pmip1, pmip2])
ds_pmip = ds_pmip.squeeze('time')
ds_pmip = ds_pmip.metpy.assign_crs(grid_mapping_name='latitude_longitude', earth_radius=6371229.0)
# %%
# create cross section
lon_pole, lat_pole = -63.7, 61
start_lon, start_lat = 75, 15
end_lon, end_lat = 165, 15
distance = haversine((start_lat, start_lon), (end_lat, end_lon))
start = (start_lat, start_lon)
end = (end_lat, end_lon)
ds_pmip['y'] = ds_pmip['lat'].values[:, 0]
ds_pmip['x'] = ds_pmip['lon'].values[0, :]

p = ds_pmip['plev'].values
height = pva.pres2alt(p)
ds_pmip = ds_pmip.assign_coords(height=('plev', height))
ds_pmip = ds_pmip.swap_dims({'plev': 'height'})

cross_pmip = cross_section(ds_pmip, start, end, steps=int(distance/2.2)+1).set_coords(('lat', 'lon'))
cross_pmip['QV1'] = cross_pmip['QV1'] * units.kelvin
cross_pmip['QV2'] = cross_pmip['QV2'] * units.kelvin
# %%
fig = plt.figure(figsize=(9, 10.3))
gs = gridspec.GridSpec(3, 2, left=0.12, bottom=0.1, right=0.89,
                        top=0.96, hspace=0.1, wspace=0.14,
                        width_ratios=[1, 1], height_ratios=[1, 1, 1])
ncol = 2  # edit here
nrow = 3

axs, ctf, ct = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
               np.empty(shape=(nrow, ncol), dtype='object')

for j in range(2):
    for i in range(3):
        axs[i, j] = fig.add_subplot(gs[i, j])

        pres = np.array([100000, 85000, 70000, 60000, 50000, 40000, 30000, 20000, 15000])
        alts = [alt for alt in pva.pres2alt(pres) if alt <= 14000]
        axs[i, j].set_yticks([2000, 4000, 6000, 8000, 10000, 12000])
        axs[i, j].set_ylim(100, pva.pres2alt(15000))
        axs[i, j].set_yticklabels([2, 4, 6, 8, 10, 12], fontsize=15)
        axs[i, j].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        axs[i, j].set_xlim(75, 160)

        if j == 1:
            axp = axs[i, j].twinx()

            axp.set_ylim(100, pva.pres2alt(15000))
            axp.set_yticks(alts[:])
            axp.set_yticklabels(list(map(int, pres[:len(alts)] / 100)), fontsize=15)
            axp.set_ylabel('Pressure (hPa)', fontsize=15, labelpad=13, rotation=270)

        if j == 0:
            axs[i, j].set_ylabel('Height (km)', fontsize=15, labelpad=2, rotation=90)

        if i == nrow - 1:
            axs[i, j].set_xticks([80, 100, 120, 140, 160])
            axs[i, j].set_xticklabels([u'80\N{DEGREE SIGN}E', u'100\N{DEGREE SIGN}E', u'120\N{DEGREE SIGN}E', u'140\N{DEGREE SIGN}E', u'160\N{DEGREE SIGN}E'], fontsize=15)
            axs[i, j].xaxis.set_label_coords(1.06, -0.018)
        else:
            axs[i, j].set_xticks([])

labels = ['CQVRL | PI', 'change']
lefts = ['COSMO', 'ECHAM5', 'PMIP4']
for j in range(ncol):
    label = labels[j]
    axs[0, j].set_title(f'{label}', pad=7, fontsize=16, loc='center')
# --
for i in range(nrow):
    left = lefts[i]
    axs[i, 0].text(-0.23, 0.5, f'{left}', ha='right', va='center',
                   transform=axs[i, 0].transAxes, fontsize=16, rotation=90)

level1 = np.linspace(0, 10, 20, endpoint=True)
cmap1 = cmc.davos_r
level2 = MaxNLocator(nbins=17).tick_values(-2, 2)
cmap2 = drywet(25, cmc.vik_r)
norm2 = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2.)

# ctf[1, 0] = axs[1, 0].contourf(cross_cosmo['lon'], cross_cosmo['height'], cross_cosmo['QV1'], levels=level1, cmap=cmap1, norm=BoundaryNorm(level1, ncolors=cmap1.N, clip=True), extend='both')
# ctf[1, 1] = axs[1, 1].contourf(cross_cosmo['lon'], cross_cosmo['height'], cross_cosmo['QV2'] - cross_cosmo['QV1'], levels=level2, cmap=cmap2, norm=BoundaryNorm(level2, ncolors=cmap1.N, clip=True), extend='both')

# ctf[1, 0] = axs[1, 0].contourf(cross_echam['lon'], cross_echam['height'], cross_echam['QV1'], levels=level1, cmap=cmap1, norm=BoundaryNorm(level1, ncolors=cmap1.N, clip=True), extend='both')
# ctf[1, 1] = axs[1, 1].contourf(cross_echam['lon'], cross_echam['height'], cross_echam['QV2'] - cross_echam['QV1'], levels=level2, cmap=cmap2, norm=BoundaryNorm(level2, ncolors=cmap1.N, clip=True), extend='both')

ctf[2, 0] = axs[2, 0].contourf(cross_pmip['lon'], cross_pmip['height'], cross_pmip['QV1'] * 1000, levels=level1, cmap=cmap1, norm=BoundaryNorm(level1, ncolors=cmap1.N, clip=True), extend='both')
ctf[2, 1] = axs[2, 1].contourf(cross_pmip['lon'], cross_pmip['height'], cross_pmip['QV2'] * 1000 - cross_pmip['QV1'] * 1000, levels=level2, cmap=cmap2, norm=norm2, extend='both')

ticks = [[0, 2, 4, 6, 8, 10], [-2, -1, 0, 1, 2]]
for i in range(ncol):
    tick = ticks[i]
    cax = fig.add_axes([axs[2, i].get_position().x0, axs[2, i].get_position().y0-0.05, axs[2, i].get_position().width, 0.017])
    cbar = fig.colorbar(ctf[2, i], cax=cax, orientation='horizontal', extend='both', ticks=tick)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.minorticks_off()

plt.show()



