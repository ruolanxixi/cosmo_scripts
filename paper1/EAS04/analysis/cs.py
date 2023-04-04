import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.ma as ma
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib.gridspec as gridspec

import wrf
from math import radians, cos, sin, asin, sqrt
import pvlib.atmosphere as pva
import cartopy.crs as ccrs
import cmcrameri.cm as cmc
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import scipy.ndimage as ndimage
import matplotlib.colors as colors

import metpy
from pyproj import Geod

from mycolor import prcp, custom_div_cmap


# -------------------------------------------------------------------------------
# define functions
def distances_xy(lon_, lat_):
    """Calculate the distances in the x and y directions along a cross-section.
    cross : `xarray.DataArray`
        The input DataArray of a cross-section from which to obtain geometeric distances in
        the x and y directions.
    x, y : tuple of `xarray.DataArray`
        A tuple of the x and y distances as DataArrays
    """

    g = Geod(ellps="WGS84")

    forward_az, _, distance = g.inv(lon_[0][0].values * np.ones_like(lon_[0]),
                                    lat_[0][0].values * np.ones_like(lat_[0]),
                                    lon_[0].values,
                                    lat_[0].values)

    x = distance * np.sin(np.deg2rad(forward_az))
    y = distance * np.cos(np.deg2rad(forward_az))

    return x, y


def get_id(var, value):
    id = np.argwhere(var == value)

    return id[0][0]


def get_closed_id(var, value):
    cid = np.argsort(np.abs(ma.getdata(var - value)))[0]

    return cid


def vector_units(lon_, lat_):
    x, y = distances_xy(lon_, lat_)

    dx_di = np.gradient(x)
    dy_di = np.gradient(y)

    tangent_vector_mag = np.hypot(dx_di, dy_di)
    unit_tangent_vector = np.vstack([dx_di / tangent_vector_mag, dy_di / tangent_vector_mag])
    unit_normal_vector = np.vstack([-dy_di / tangent_vector_mag, dx_di / tangent_vector_mag])

    return unit_tangent_vector, unit_normal_vector


def vector_components(data_x, data_y, lon_, lat_):
    unit_tang, unit_norm = vector_units(lon_, lat_)

    # Take the dot products
    component_tang = data_x * unit_tang[0] + data_y * unit_tang[1]
    component_norm = data_x * unit_norm[0] + data_y * unit_norm[1]

    return component_tang, component_norm


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    input: lon1, lat1, lon2, lat2 (float) -> coordinates of point 1 and point 2
    return: distance between point 1 and point 2 in km (float)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers.
    r = 6371
    return c * r


def rotate_points(lon_pole, lat_pole, rlon, rlat):
    crs_geo = ccrs.PlateCarree()
    crs_rot_pole = ccrs.RotatedPole(pole_longitude=lon_pole,
                                    pole_latitude=lat_pole)
    lon, lat = crs_geo.transform_point(rlon, rlat, crs_rot_pole)
    return lon, lat


# -------------------------------------------------------------------------------
# read data
sims = ['ctrl', 'topo1', 'topo2']
dt, dt['ept'], dt['qv'], dt['rhov'], dt['rhow'], dt['diab'] = {}, {}, {}, {}, {}, {}
for i in range(len(sims)):
    sim = sims[i]
    ds1 = xr.open_dataset(f'/project/pr133/rxiang/data/cosmo/EAS04_{sim}/monsoon/QV/smr/01-05.QV.smr.timmean.nc')
    ds2 = xr.open_dataset(f'/project/pr133/rxiang/data/cosmo/EAS04_{sim}/monsoon/T/smr/01-05.T.smr.timmean.nc')
    ds3 = xr.open_dataset(f'/project/pr133/rxiang/data/cosmo/EAS04_{sim}/monsoon/W/smr/01-05.W.smr.timmean.nc')
    ds4 = xr.open_dataset(f'/project/pr133/rxiang/data/cosmo/EAS04_{sim}/monsoon/U/smr/01-05.U.smr.timmean.nc')
    ds5 = xr.open_dataset(f'/project/pr133/rxiang/data/cosmo/EAS04_{sim}/monsoon/V/smr/01-05.V.smr.timmean.nc')
    ds6 = xr.open_dataset(
        f'/project/pr133/rxiang/data/cosmo/EAS04_{sim}/monsoon/DIAB_SUM/smr/01-05.DIAB_SUM.smr.timmean.nc')
    ds7 = xr.open_dataset('/store/c2sm/pr04/rxiang/data_lmp/01010100_EAS11_ctrl/lm_fine/24h3D/TMPHYS_SUM.nc')

    pre = ds1['pressure'].values[::-1]
    temp = ds2['T'].values[0, ::-1, ...]
    qv = ds1['QV'].values[0, ::-1, ...]
    w = ds3['W'].values[0, ::-1, ...]
    u = ds4['U'].values[0, ::-1, ...]
    v = ds5['V'].values[0, ::-1, ...]
    diab = ds6['DIAB_SUM'].values[0, ::-1, ...]

    pressure = np.repeat(np.repeat(pre[..., np.newaxis], 650, axis=1)[..., np.newaxis], 650, axis=2)
    dewpoint = metpy.calc.dewpoint_from_specific_humidity(pressure * units.Pa, temp * units.kelvin, qv * units('kg/kg'))
    mr = metpy.calc.mixing_ratio_from_specific_humidity(qv * units('kg/kg'))
    rho = metpy.calc.density(pressure * units.Pa, temp * units.kelvin, mr)
    epts = metpy.calc.equivalent_potential_temperature(pressure * units.Pa, temp * units.kelvin, dewpoint)
    data = np.array([x.to_base_units().magnitude for x in epts])
    density = np.array([x.to_base_units().magnitude for x in rho])
    rhow = density * w
    omg = metpy.calc.vertical_velocity_pressure(w * units('m/s'), pressure * units.Pa, temp * units.kelvin)
    omega = np.array([x.to_base_units().magnitude for x in omg])*-1
    # rhow = ndimage.gaussian_filter(rhow, sigma=2, order=0)
    rhou = density * u
    # rhov = ndimage.gaussian_filter(rhov, sigma=2, order=0)
    dt['ept'][sim] = data
    dt['qv'][sim] = qv * 1000
    dt['rhow'][sim] = omega
    dt['rhov'][sim] = u
    dt['diab'][sim] = diab

rlon = np.round(ds1['rlon'].values[...], 2)
rlat = np.round(ds1['rlat'].values[...], 2)
lon = ds1['lon'].values[...]
lat = ds1['lat'].values[...]

vcoord = ds1['pressure'].values[...]
vcoord = pva.pres2alt(vcoord)
vcoord_ = wrf.destagger(vcoord, 0)
vcoord2 = ds7['pressure'].values[...]
vcoord2 = pva.pres2alt(vcoord2)

vars = ['qv', 'diab', 'ept']
dt['qv']['topo1'] = dt['qv']['topo1'] - dt['qv']['ctrl']
dt['qv']['topo2'] = dt['qv']['topo2'] - dt['qv']['ctrl']
dt['diab']['topo2'] = dt['diab']['topo2'] - dt['diab']['ctrl']
# -------------------------------------------------------------------------------
# %% plot
font = {'size': 15}
matplotlib.rc('font', **font)

rlon_start = -25.2
rlon_end = -2.88
rlat_start = -0.07
rlat_end = 0.69
zmax = 14
lat_pole = 61
lon_pole = -63.7

# initialize figure
nrow = 3
ncol = 3
axs, ctf, ct = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
               np.empty(shape=(nrow, ncol), dtype='object')
fig = plt.figure(figsize=(18, 10.3))
gs = gridspec.GridSpec(3, 3, left=0.04, bottom=0.03, right=0.887,
                        top=0.97, hspace=0.07, wspace=0.09, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])

color = ['#9e0142','#d53e4f','#f46d43','#fdae61','#fee08b','#ffffbf','#e6f598','#abdda4','#66c2a5','#3288bd','#5e4fa2']
color.reverse()
# desat_colors = []
# for c in color:
#     # Calculate the desaturated color by taking the average of the RGB values
#     desat_color = tuple([(c[i] + 1) / 2 for i in range(3)])
#     desat_colors.append(desat_color)
cmapp = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', color, N=25)

# level1 = np.linspace(0, 20, 21, endpoint=True)
level1 = np.linspace(-1, 1, 21, endpoint=True)
level2 = np.linspace(338, 360, 23, endpoint=True)
level3 = np.linspace(-5, 15, 21, endpoint=True)
levels = [level1, level3, level2]
cmaps = [custom_div_cmap(41, cmc.vik), custom_div_cmap(41, cmc.vik), cmapp]
# cmaps = [prcp(20), custom_div_cmap(41, cmc.vik), cmapp]
# extends = ['max', 'both', 'both']
extends = ['both', 'both', 'both']

# matplotlib.colormaps['twilight']

lb = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']]

for i in range(nrow):
    var = vars[i]
    level = levels[i]
    cmap = cmaps[i]
    ex = extends[i]
    if i == 1:
        vc = vcoord2
    else:
        vc = vcoord
    for j in range(ncol):
        axs[i, j] = fig.add_subplot(gs[i, j])
        sim = sims[j]

        if j == 2:
            axp = axs[i, j].twinx()

        # ax.set_ylabel('Height (km)', fontsize=15, labelpad=1.5)
        # ax.tick_params(axis='both', which='major', labelsize=13)

            axs[i, j].set_ylim(0.1, pva.pres2alt(15000) / 1000)
            axp.set_ylim(0.1, pva.pres2alt(15000) / 1000)

            axs[i, j].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

            pres = np.array([100000, 85000, 70000, 60000, 50000, 40000, 30000, 20000, 15000])
            alts = [alt for alt in pva.pres2alt(pres) / 1000 if alt <= zmax]

            axp.set_yticks(alts[:])
            axp.set_yticklabels(list(map(int, pres[:len(alts)] / 100)), fontsize=15)

            # axs[i, j].set_yticks([])
            axp.set_ylabel('Pressure (hPa)', fontsize=15, labelpad=13, rotation=270)

        else:
            axs[i, j].set_ylim(0.1, pva.pres2alt(15000) / 1000)
            axs[i, j].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        
        if j == 0:
            axs[i, j].set_ylabel('Height (km)', fontsize=15, labelpad=17, rotation=270)


        # add profile
        rlon_start_id = get_closed_id(rlon, rlon_start)
        rlat_start_id = get_closed_id(rlat, rlat_start)

        rlon_end_id = get_closed_id(rlon, rlon_end)
        rlat_end_id = get_closed_id(rlat, rlat_end)

        start = (rlon_start_id, rlat_start_id)
        end = (rlon_end_id, rlat_end_id)

        lon_start, lat_start = rotate_points(lon_pole, lat_pole, rlon_start, rlat_start)
        lon_end, lat_end = rotate_points(lon_pole, lat_pole, rlon_end, rlat_end)

        # return the x, y points for a line within a two-dimensional grid (the cross section)
        distance = haversine(lon_start, lat_start, lon_start, lat_end)
        xyline = wrf.xy(dt[var][sim], start_point=start, end_point=end)

        vert_cross = wrf.interp2dxy(dt[var][sim], xyline)
        vert_cross = np.ma.array(vert_cross, mask=np.isnan(vert_cross))

        # if i == 2:
        #     vert_cross = ndimage.gaussian_filter(vert_cross, sigma=1, order=0)

        # levels = np.linspace(340, 364, 25, endpoint=True)
        # cmap = matplotlib.colormaps['rainbow']
        if i == 1:
            norm = colors.TwoSlopeNorm(vmin=-5., vcenter=0., vmax=15)
        else:
            norm = BoundaryNorm(level, ncolors=cmap.N, clip=True)

        ctf[i, j] = axs[i, j].contourf(np.arange(vert_cross.shape[1]), vc[::-1] / 1000, vert_cross,
                                       extend=ex, cmap=cmap, levels=level, norm=norm)
        if i == 0:
            ct[i, j] = axs[i, j].contour(np.arange(vert_cross.shape[1]), vc[::-1] / 1000, vert_cross,
                                         levels=[8], colors='white',
                                         linewidths=1.5, zorder=9)
            vert_cross = wrf.interp2dxy(dt['rhov'][sim], xyline)
            vert_cross = np.ma.array(vert_cross, mask=np.isnan(vert_cross))
            vert_cross = ndimage.gaussian_filter1d(vert_cross, sigma=3, order=0, axis=1)
            ct[i, j] = axs[i, j].contour(np.arange(vert_cross.shape[1]), vc[::-1] / 1000, vert_cross,
                                         levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], colors='black', linewidths=1.2)
            clabel = axs[i, j].clabel(ct[i, j], inline=True, levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                             fontsize=13, use_clabeltext=True)
            clabel2 = axs[i, j].clabel(ct[i, j], inline=True, fontsize=13,
                                       manual=[(15, 11), (100, 11), (95, 10), (95, 8), (150, 8)], use_clabeltext=True)
            for l in clabel:
                l.set_rotation(0)
            for l in clabel2:
                l.set_rotation(0)

        if i == 1:
            vc = vcoord
            vert_cross = wrf.interp2dxy(dt['rhow'][sim], xyline)
            vert_cross = np.ma.array(vert_cross, mask=np.isnan(vert_cross))
            vert_cross = ndimage.gaussian_filter1d(vert_cross, sigma=4, order=0, axis=1)
            ct[i, j] = axs[i, j].contour(np.arange(vert_cross.shape[1]), vc[::-1] / 1000, vert_cross,
                                         levels=[.1, .2, .3], colors='black', linewidths=1.2)
            clabel = axs[i, j].clabel(ct[i, j], inline=True, levels=[.1, .2, .3],
                             fontsize=13, use_clabeltext=True)
            for l in clabel:
                l.set_rotation(0)
            vc = vcoord2

        # add terrain
        f = xr.open_dataset(f'/project/pr133/rxiang/data/extpar/extpar_cross_{sim}.nc')
        hsurf = f['HSURF'].values[...]
        hsurf = np.repeat(hsurf[np.newaxis, ...], 57, axis=0)

        terrain = wrf.interp2dxy(hsurf, xyline)[1]
        axs[i, j].plot(np.arange(terrain.shape[0]), terrain / 1000, color='black', linewidth=0.01)
        axs[i, j].fill_between(np.arange(terrain.shape[0]), terrain / 1000, color='black', zorder=10)

        #
        lon_cross = np.repeat(lon[np.newaxis, ...], 10, axis=0)
        lon_value = wrf.interp2dxy(lon_cross, xyline)[1]

        lons = [90, 95, 100, 105, 110]

        index = []
        for l in range(len(lons)):
            ii = lons[l]
            id = get_closed_id(lon_value, ii)
            index = np.append(index, id)

        if i == nrow-1:
            axs[i, j].set_xticks(index)
            axs[i, j].set_xticklabels([u'90\N{DEGREE SIGN}E', u'95\N{DEGREE SIGN}E', u'100\N{DEGREE SIGN}E', u'105\N{DEGREE SIGN}E', u'110\N{DEGREE SIGN}E'])
            axs[i, j].xaxis.set_label_coords(1.06, -0.018)
        else:
            axs[i, j].set_xticks([])

clabel2 = axs[1, 0].clabel(ct[1, 0], inline=True, fontsize=13, manual=[(300, 8)], use_clabeltext=True)
for l in clabel2:
    l.set_rotation(0)

clabel2 = axs[0, 0].clabel(ct[0, 0], inline=True, fontsize=13, manual=[(150, 12.5), (210, 8), (250, 8)], use_clabeltext=True)  # , (400, 8), (500, 9)
for l in clabel2:
    l.set_rotation(0)

clabel2 = axs[0, 1].clabel(ct[0, 1], inline=True, fontsize=13, manual=[(10, 12), (50, 12), (95, 9), (400, 8), (460, 8)], use_clabeltext=True)
for l in clabel2:
    l.set_rotation(0)

clabel2 = axs[0, 2].clabel(ct[0, 2], inline=True, fontsize=13, manual=[(50, 12), (70, 11), (200, 8), (300, 8), (350, 8), (550, 8), (600, 8)], use_clabeltext=True)
for l in clabel2:
    l.set_rotation(0)

level1 = np.linspace(0, 20, 11, endpoint=True)
level2 = np.linspace(340, 358, 7, endpoint=True)
level3 = [-4, -2, 0, 2, 4, 6, 8, 10, 12, 14]
levels = [level1, level3, level2]
extends = ['max', 'both', "both"]
xlabels = ['g kg$^{-1}$', 'K day$^{-1}$', 'K']
for i in range(nrow):
    ex = extends[i]
    label = xlabels[i]
    level = levels[i]
    cax = fig.add_axes([axs[i, 2].get_position().x1 + 0.055, axs[i, 2].get_position().y0, 0.013, axs[i, 2].get_position().height])
    cbar = fig.colorbar(ctf[i, 2], cax=cax, orientation='vertical', extend=ex)
    cbar.set_ticks(level)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.minorticks_off()
    axs[i, 2].text(1.35, 0.5, f'{label}', ha='left', va='center', transform=axs[i, 2].transAxes, fontsize=14, rotation=270)

titles = ['CTRL04', 'TRED04', 'TENV04']
for j in range(ncol):
    title = titles[j]
    axs[0, j].set_title(f'{title}', pad=5, fontsize=15, loc='center')

for i in range(nrow):
    for j in range(ncol):
        label = lb[i][j]
        t = axs[i, j].text(0.01, 0.987, f'({label})', ha='left', va='top',
                           transform=axs[i, j].transAxes, fontsize=15)
        t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

plt.show()
# plotpath = "/project/pr133/rxiang/figure/paper1/results/"
# fig.savefig(plotpath + 'cs.png', dpi=500)

plt.close(fig)
