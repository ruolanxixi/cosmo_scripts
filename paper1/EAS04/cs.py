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

font = {'size': 13}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
def rotate_points(lon_pole, lat_pole, rlon, rlat):
    crs_geo = ccrs.PlateCarree()
    crs_rot_pole = ccrs.RotatedPole(pole_longitude=lon_pole,
                                    pole_latitude=lat_pole)
    lon, lat = crs_geo.transform_point(rlon, rlat, crs_rot_pole)
    return lon, lat
# -------------------------------------------------------------------------------
nrow = 3
ncol = 3
axs, ctf, ct = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
               np.empty(shape=(nrow, ncol), dtype='object')
fig = plt.figure(figsize=(18, 10.3))
gs = gridspec.GridSpec(3, 3, left=0.04, bottom=0.03, right=0.887,
                       top=0.97, hspace=0.07, wspace=0.09, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])

color = ['#9e0142','#d53e4f','#f46d43','#fdae61','#fee08b','#ffffbf','#e6f598','#abdda4','#66c2a5','#3288bd','#5e4fa2']
color.reverse()
cmapp = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', color, N=25)

level1 = np.linspace(0, 20, 21, endpoint=True)
level2 = np.linspace(-5, 15, 21, endpoint=True)
level3 = np.linspace(338, 360, 23, endpoint=True)
levels = [level1, level2, level3]
cmaps = [prcp(20), custom_div_cmap(41, cmc.vik), cmapp]
extends = ['max', 'both', 'both']

lb = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']]
#%%
# -------------------------------------------------------------------------------
sims = ['ctrl', 'topo1', 'topo2']
for j in range(len(sims)):
    sim = sims[j]
    ds1 = xr.open_dataset(f'/project/pr133/rxiang/data/cosmo/EAS04_{sim}/monsoon/QV/smr/01-05.QV.smr.timmean.nc')
    ds1 = ds1.isel(pressure=slice(None, None, -1))
    ds2 = xr.open_dataset(f'/project/pr133/rxiang/data/cosmo/EAS04_{sim}/monsoon/T/smr/01-05.T.smr.timmean.nc')
    ds2 = ds2.isel(pressure=slice(None, None, -1))
    ds3 = xr.open_dataset(f'/project/pr133/rxiang/data/cosmo/EAS04_{sim}/monsoon/W/smr/01-05.W.smr.timmean.nc')
    ds3 = ds3.isel(pressure=slice(None, None, -1))
    ds4 = xr.open_dataset(f'/project/pr133/rxiang/data/cosmo/EAS04_{sim}/monsoon/uv/01-05.uv.smr.timmean.nc')
    ds4 = ds4.isel(pressure=slice(None, None, -1))
    ds5 = xr.open_dataset(
        f'/project/pr133/rxiang/data/cosmo/EAS04_{sim}/monsoon/DIAB_SUM/smr/01-05.DIAB_SUM.smr.timmean.nc')
    ds5 = ds5.isel(pressure=slice(None, None, -1))

    f = xr.open_dataset(f'/project/pr133/rxiang/data/extpar/extpar_cross_{sim}.nc')

    # create regional latlon grid and regridder
    target_grid = xe.util.grid_2d(lon0_b=88, lon1_b=115, d_lon=0.04, lat0_b=15., lat1_b=40., d_lat=0.04)
    regridder = xe.Regridder(ds1, target_grid, 'bilinear')

    # regrid fields
    qv = regridder(ds1.QV)*1000
    t = regridder(ds2.T)
    w = regridder(ds3.W)
    u = regridder(ds4.ugeo)
    v = regridder(ds4.vgeo)
    diab = regridder(ds5.DIAB_SUM)
    hsurf = regridder(f.HSURF)
    qv.name = 'QV'
    t.name = 'T'
    w.name = 'W'
    hsurf.name = 'HSURF'
    u.name = 'U'
    v.name = 'V'
    diab.name = 'DIAB'

    # create new dataset
    ds = xr.merge([qv, t, w, hsurf, u, v])
    ds = ds.squeeze('time')
    ds = ds.metpy.assign_crs(grid_mapping_name='latitude_longitude', earth_radius=6371229.0)

    ds5 = xr.merge([diab])
    ds5 = ds5.squeeze('time')
    ds5 = ds5.metpy.assign_crs(grid_mapping_name='latitude_longitude', earth_radius=6371229.0)

    # create cross section
    lon_pole, lat_pole = -63.7, 61
    # start_rlat = 0.00
    # start_rlon = -23.8
    # end_rlat = 1.30
    # end_rlon = -2.96
    start_rlat = -2.23
    start_rlon = -24.8
    end_rlat = 3.00
    end_rlon = -2.96
    start_lon, start_lat = rotate_points(lon_pole, lat_pole, start_rlon, start_rlat)
    end_lon, end_lat = rotate_points(lon_pole, lat_pole, end_rlon, end_rlat)
    distance = haversine((start_lat, start_lon), (end_lat, end_lon))
    start = (start_lat, start_lon)
    end = (end_lat, end_lon)
    ds['y'] = ds['lat'].values[:, 0]
    ds['x'] = ds['lon'].values[0, :]

    ds5['y'] = ds5['lat'].values[:, 0]
    ds5['x'] = ds5['lon'].values[0, :]

    p = ds['pressure'].values
    height = pva.pres2alt(p)
    ds = ds.assign_coords(height=('pressure', height))
    ds = ds.swap_dims({'pressure': 'height'})
    # ds = ds.drop_vars('pressure')

    p = ds5['pressure'].values
    height = pva.pres2alt(p)
    ds5 = ds5.assign_coords(height=('pressure', height))
    ds5 = ds5.swap_dims({'pressure': 'height'})
    # ds5 = ds5.drop_vars('pressure')

    cross = cross_section(ds, start, end, steps=int(distance/2.2)+1).set_coords(('lat', 'lon'))
    cross2 = cross_section(ds5, start, end, steps=int(distance/2.2)+1).set_coords(('lat', 'lon'))

    cross['dewpoint'] = mpcalc.dewpoint_from_specific_humidity(cross['pressure'] * units.Pa, cross['T'] * units.kelvin, cross['QV'] * units('g/kg'))
    cross['mr'] = mpcalc.mixing_ratio_from_specific_humidity(cross['QV'] * units('g/kg'))
    cross['rho'] = mpcalc.density(cross['pressure'] * units.Pa, cross['T'] * units.kelvin, cross['mr'])
    cross['ept'] = mpcalc.equivalent_potential_temperature(cross['pressure'] * units.Pa, cross['T'] * units.kelvin, cross['dewpoint'])
    cross['ws'] = mpcalc.wind_speed(cross['U'] * units('m/s'), cross['V'] * units('m/s'))
    cross['omega'] = mpcalc.vertical_velocity_pressure(cross['W'] * units('m/s'), cross['pressure'] * units.Pa, cross['T'] * units.kelvin) * -1
    cross['t_wind'], cross['n_wind'] = mpcalc.cross_section_components(cross['U'] * units('m/s'), cross['V'] * units('m/s'))

    for i in range(3):
        axs[i, j] = fig.add_subplot(gs[i, j])

        pres = np.array([100000, 85000, 70000, 60000, 50000, 40000, 30000, 20000, 15000])
        alts = [alt for alt in pva.pres2alt(pres) if alt <= 14000]
        axs[i, j].set_yticks([2000, 4000, 6000, 8000, 10000, 12000])
        axs[i, j].set_ylim(100, pva.pres2alt(15000))
        axs[i, j].set_yticklabels([2, 4, 6, 8, 10, 12], fontsize=15)
        axs[i, j].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        if j == 2:
            axp = axs[i, j].twinx()

            axp.set_ylim(100, pva.pres2alt(15000))
            axp.set_yticks(alts[:])
            axp.set_yticklabels(list(map(int, pres[:len(alts)] / 100)), fontsize=15)
            axp.set_ylabel('Pressure (hPa)', fontsize=15, labelpad=13, rotation=270)

        if j == 0:
            axs[i, j].set_ylabel('Height (km)', fontsize=15, labelpad=17, rotation=270)

        if i == nrow - 1:
            axs[i, j].set_xticks([90, 95, 100, 105, 110])
            axs[i, j].set_xticklabels(
                [u'90\N{DEGREE SIGN}E', u'95\N{DEGREE SIGN}E', u'100\N{DEGREE SIGN}E', u'105\N{DEGREE SIGN}E',
                 u'110\N{DEGREE SIGN}E'])
            axs[i, j].xaxis.set_label_coords(1.06, -0.018)
        else:
            axs[i, j].set_xticks([])

    # --------------------
    # add profile
    ctf[0, j] = axs[0, j].contourf(cross['lon'], cross['height'], cross['QV'], levels=level1, cmap=cmaps[0], norm=BoundaryNorm(level1, ncolors=cmaps[0].N, clip=True), extend='max')
    ctf[1, j] = axs[1, j].contourf(cross2['lon'], cross2['height'], cross2['DIAB'], levels=level2, cmap=cmaps[1], norm=colors.TwoSlopeNorm(vmin=-5., vcenter=0., vmax=15), extend='both')
    ctf[2, j] = axs[2, j].contourf(cross['lon'], cross['height'], cross['ept'], levels=level3, cmap=cmaps[2], norm=BoundaryNorm(level3, ncolors=cmaps[2].N, clip=True), extend='both')

    ct[0, j] = axs[0, j].contour(cross['lon'], cross['height'], ndimage.gaussian_filter1d(cross['t_wind'], sigma=3, order=0, axis=1),
                                 levels=[-7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15], colors='black', linewidths=1.2)
    clabel = axs[0, j].clabel(ct[0, j], inline=True, fontsize=13, levels=[-7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15], use_clabeltext=True)
    # clabel2 = axs[0, j].clabel(ct[0, j], inline=True, fontsize=13,
    #                            manual=[(91, 12000), (91, 7000), (92, 7000), (93, 6000), (98, 7000), (105, 5000), (104, 8500), (107, 8000), (110, 9200)], use_clabeltext=True)
    clabel2 = axs[0, j].clabel(ct[0, j], inline=True, fontsize=13,
                               manual=[(90, 12000), (91, 11000), (93, 10000), (94, 9000), (95, 7000), (105, 5000),
                                       (104, 12000), (107, 8500)], use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)
    for l in clabel2:
        l.set_rotation(0)

    ct[1, j] = axs[1, j].contour(cross['lon'], cross['height'], ndimage.gaussian_filter1d(cross['omega'], sigma=7, order=0, axis=1), colors='black',
                                 levels=[.1, .2, .3], linewidths=1.2)
    clabel = axs[1, j].clabel(ct[1, j], inline=True, fontsize=13, levels=[.1, .2, .3], use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)

    for i in range(3):
        axs[i, j].plot(cross['lon'], cross['HSURF'], color='black', linewidth=0.01)
        axs[i, j].fill_between(cross['lon'], cross['HSURF'], color='black')


# %% Define the CRS and inset axes
# data_crs = ds['QV'].metpy.cartopy_crs
ax_inset = fig.add_axes([0.04, 0.87, 0.1, 0.1], projection=ccrs.PlateCarree())  # 0.787
ax_inset.set_extent([88, 114, 20, 35], crs=ccrs.PlateCarree())
path = '/users/rxiang/lmp/lib'
ds = xr.open_dataset(f'{path}/extpar_BECCY_4.4km_merit_unmod_topo.nc')
ctrl04 = ds['HSURF'].values[...]
rlat04 = ds["rlat"].values
rlon04 = ds["rlon"].values
rot_pole_crs = ccrs.RotatedPole(pole_latitude=61, pole_longitude=-63.7)
ax_inset.pcolormesh(rlon04, rlat04, ctrl04, cmap=truncate_colormap(cmc.bukavu, 0.55, 1.0), norm=BoundaryNorm(np.arange(0., 6500.0, 500.0), ncolors=truncate_colormap(cmc.bukavu, 0.55, 1.0).N, extend="max"), shading="auto", transform=rot_pole_crs)
ax_inset.text(0.01, 0.98, '(a)', ha='left', va='top',
                           transform=ax_inset.transAxes, fontsize=15, zorder=200)

# Plot the path of the cross section
endpoints = ccrs.PlateCarree().transform_points(ccrs.Geodetic(),
                                      *np.vstack([start, end]).transpose()[::-1])
ax_inset.scatter(endpoints[:, 0], endpoints[:, 1], c='k', zorder=2)
ax_inset.plot(cross['x'], cross['y'], c='k', zorder=2)

# Add geographic features
ax_inset.add_feature(cfeature.COASTLINE, linewidth=2)
ax_inset.add_feature(cfeature.BORDERS, linestyle=':')
# ax_inset.add_feature(cfeature.RIVERS, alpha=0.5)
ax_inset.add_feature(cfeature.OCEAN, zorder=100)

# ax_inset.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='k', alpha=0.2, zorder=0)
# %%
level1 = np.linspace(0, 20, 11, endpoint=True)
level2 = [-4, -2, 0, 2, 4, 6, 8, 10, 12, 14]
level3 = np.linspace(340, 358, 7, endpoint=True)
levels = [level1, level2, level3]
extends = ['max', 'both', "both"]
xlabels = ['g kg$^{-1}$', 'K day$^{-1}$', 'K']
for i in range(nrow):
    ex = extends[i]
    label = xlabels[i]
    level = levels[i]
    cax = fig.add_axes(
        [axs[i, 2].get_position().x1 + 0.055, axs[i, 2].get_position().y0, 0.013, axs[i, 2].get_position().height])
    cbar = fig.colorbar(ctf[i, 2], cax=cax, orientation='vertical', extend=ex)
    cbar.set_ticks(level)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.minorticks_off()
    axs[i, 2].text(1.35, 0.5, f'{label}', ha='left', va='center', transform=axs[i, 2].transAxes, fontsize=14,
                   rotation=270)

titles = ['CTRL04', 'TRED04', 'TENV04']
for j in range(ncol):
    title = titles[j]
    axs[0, j].set_title(f'{title}', pad=5, fontsize=15, loc='center')

for i in range(nrow):
    for j in range(ncol):
        label = lb[i][j]
        t = axs[i, j].text(0.01, 0.987, f'({label})', ha='left', va='top',
                           transform=axs[i, j].transAxes, fontsize=15, zorder=200)
        t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

plotpath = "/project/pr133/rxiang/figure/paper1/results/"
fig.savefig(plotpath + 'cs.png', dpi=500)
plt.show()
