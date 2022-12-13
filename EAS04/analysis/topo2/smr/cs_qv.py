import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.ma as ma
import xarray as xr
import metpy.calc as mpcalc

import wrf
from netCDF4 import Dataset
from math import radians, cos, sin, asin, sqrt, ceil, floor
from collections import UserDict
import pvlib.atmosphere as pva
import cartopy.crs as ccrs
import cmcrameri.cm as cmc
from mycolor import custom_div_cmap
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator, FixedLocator, MultipleLocator
from matplotlib.colors import BoundaryNorm

import metpy
from pyproj import Geod


class Plot_Cross(UserDict):
    """ Plot cross-section in lonlat coordinates"""

    def __init__(self, rlon_start, rlon_end, rlat_start, rlat_end, zmax=14, lat_pole=61, lon_pole=-63.7):

        self.rlon_start = rlon_start
        self.rlon_end = rlon_end

        self.rlat_start = rlat_start
        self.rlat_end = rlat_end

        self.lat_pole = lat_pole
        self.lon_pole = lon_pole

        # initialize figure
        self.fig, self.axes = plt.subplots(1, 1, figsize=(5, 3.5), constrained_layout=True)
        self.ax = plt.subplot(1, 1, 1)
        self.axp = self.ax.twinx()

        # self.ax.set_ylabel('Height (km)', fontsize=12, labelpad=1.5)
        self.ax.tick_params(axis='both', which='major', labelsize=12)

        self.ax.set_ylim(0.1, pva.pres2alt(15000) / 1000)
        self.axp.set_ylim(0.1, pva.pres2alt(15000) / 1000)

        self.ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        pres = np.array([100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 20000, 15000])
        alts = [alt for alt in pva.pres2alt(pres) / 1000 if alt <= zmax]

        self.axp.set_yticks(alts[:])
        self.axp.set_yticklabels(list(map(int, pres[:len(alts)] / 100)), fontsize=12)
        # self.axp.set_ylabel('Pressure (hPa)', rotation=270, fontsize=12, labelpad=5.7)

    def distances_xy(self):
        """Calculate the distances in the x and y directions along a cross-section.
        cross : `xarray.DataArray`
            The input DataArray of a cross-section from which to obtain geometeric distances in
            the x and y directions.
        x, y : tuple of `xarray.DataArray`
            A tuple of the x and y distances as DataArrays
        """

        g = Geod(ellps="WGS84")

        forward_az, _, distance = g.inv(self.lon_[0][0].values * np.ones_like(self.lon_[0]),
                                        self.lat_[0][0].values * np.ones_like(self.lat_[0]),
                                        self.lon_[0].values,
                                        self.lat_[0].values)

        x = distance * np.sin(np.deg2rad(forward_az))
        y = distance * np.cos(np.deg2rad(forward_az))

        return x, y

    def get_id(self, var, value):

        id = np.argwhere(var == value)

        return id[0][0]

    def get_closed_id(self, var, value):

        cid = np.argsort(np.abs(ma.getdata(var - value)))[0]

        return cid

    def vector_units(self):

        x, y = self.distances_xy()

        dx_di = np.gradient(x)
        dy_di = np.gradient(y)

        tangent_vector_mag = np.hypot(dx_di, dy_di)
        unit_tangent_vector = np.vstack([dx_di / tangent_vector_mag, dy_di / tangent_vector_mag])
        unit_normal_vector = np.vstack([-dy_di / tangent_vector_mag, dx_di / tangent_vector_mag])

        return unit_tangent_vector, unit_normal_vector

    def vector_components(self, data_x, data_y):

        unit_tang, unit_norm = self.vector_units()

        # Take the dot products
        component_tang = data_x * unit_tang[0] + data_y * unit_tang[1]
        component_norm = data_x * unit_norm[0] + data_y * unit_norm[1]

        return component_tang, component_norm

    def haversine(self, lon1, lat1, lon2, lat2):
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

    def set_xticks_distance(self, xlen, distance, step):

        self.ax.set_xlim(0, xlen)
        self.ax.set_xticks(np.arange(0, xlen / distance * floor(distance / step) * step + xlen / distance * step,
                                     xlen / distance * step))
        self.ax.set_xticklabels(list(map(str, np.arange(0, (floor(distance / step) + 1) * step, step))), fontsize=12)
        self.ax.set_xlabel('Distance (km)', fontsize=14, labelpad=4.5)

    def add_terrain(self, xyline, level=57):
        # FIXME
        f1 = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_cross_topo2.nc')
        f2 = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_cross_topo2.nc')
        hsurf1 = f1['HSURF'].values[...]
        hsurf2 = f2['HSURF'].values[...]

        hsurf1 = np.repeat(hsurf1[np.newaxis,...], level, axis=0)
        hsurf2 = np.repeat(hsurf2[np.newaxis,...], level, axis=0)

        terrain2 = wrf.interp2dxy(hsurf2, xyline)[1]
        self.ax.plot(np.arange(terrain2.shape[0]), terrain2 / 1000, color='black', linewidth=0.01)
        self.ax.fill_between(np.arange(terrain2.shape[0]), terrain2 / 1000, color='black')

        # terrain1 = wrf.interp2dxy(hsurf1, xyline)[1]
        # self.ax.plot(np.arange(terrain1.shape[0]), terrain1 / 1000, color='lightgrey',
        #              linewidth=0.01)
        # self.ax.fill_between(np.arange(terrain1.shape[0]), terrain1 / 1000, color='black',
        #                      alpha=0.5)
        # self.ax.fill_between(np.arange(self.lat_start, self.lat_end + 0.04, 0.04), terrain2 / 1000, color='black')
        #
        # terrain1 = wrf.interp2dxy(hsurf1, xyline)[1]
        # self.ax.plot(np.arange(self.lat_start, self.lat_end + 0.04, 0.04), terrain1 / 1000, color='lightgrey',
        #              linewidth=0.01)
        # self.ax.fill_between(np.arange(self.lat_start, self.lat_end + 0.04, 0.04), terrain1 / 1000, color='black',
        #                      alpha=0.5)

    def rotate_points(self, lon_pole, lat_pole, rlon, rlat):
        crs_geo = ccrs.PlateCarree()
        crs_rot_pole = ccrs.RotatedPole(pole_longitude=lon_pole,
                                        pole_latitude=lat_pole)
        lon, lat = crs_geo.transform_point(rlon, rlat, crs_rot_pole)
        return lon, lat


    def add_profile(self, var, varname, w_factor=None, colorbar=True):

        rlon_start_id = self.get_id(rlon, self.rlon_start)
        rlat_start_id = self.get_id(rlat, self.rlat_start)

        rlon_end_id = self.get_id(rlon, self.rlon_end)
        rlat_end_id = self.get_id(rlat, self.rlat_end)

        start = (rlon_start_id, rlat_start_id)
        end = (rlon_end_id, rlat_end_id)

        lon_start, lat_start = self.rotate_points(self.lon_pole, self.lat_pole, self.rlon_start, self.rlat_start)
        lon_end, lat_end = self.rotate_points(self.lon_pole, self.lat_pole, self.rlon_end, self.rlat_end)

        # return the x, y points for a line within a two-dimensional grid (the cross section)
        self.distance = self.haversine(lon_start, lat_start, lon_start, lat_end)
        self.xyline = wrf.xy(DIAB, start_point=start, end_point=end)

        # Return a cross section for a three-dimensional field
        # self.lon_ = wrf.interp2dxy(np.repeat(lon[np.newaxis, ...], 57, axis=0), self.xyline)
        # self.lat_ = wrf.interp2dxy(np.repeat(lat[np.newaxis, ...], 57, axis=0), self.xyline)

        # vert_cross = wrf.interp2dxy(var, self.xyline)
        # vert_cross = np.ma.array(vert_cross, mask=np.isnan(vert_cross))

        vert_cross = wrf.interp2dxy(var, self.xyline)
        vert_cross = np.ma.array(vert_cross, mask=np.isnan(vert_cross))

        # vert_cross_tang, vert_cross_norm = self.vector_components(vert_cross_U, vert_cross_V)

        cmap = custom_div_cmap(21, cmc.vik)
        # levels = [-3, -2, -1, -0.5, 0.5, 1, 2, 3]
        # # levels = MaxNLocator(nbins=14).tick_values(-7, 7)
        levels = np.linspace(0, 20, 21, endpoint=True)
        cmap = cmc.davos_r
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        # divnorm = colors.TwoSlopeNorm(vmin=-3., vcenter=0., vmax=3)
        # ctf = self.ax.contourf(np.arange(self.lat_start, self.lat_end + 0.04, 0.04 * 5), vcoord[::-1] / 1000,
        #                        vert_cross[:, ::5], extend='both',
        #                        cmap=cmap, levels=levels, norm=divnorm)

        ctf = self.ax.contourf(np.arange(vert_cross.shape[1]), vcoord[::-1] / 1000, vert_cross,
                               extend='max', cmap=cmap, levels=levels, norm=norm)

        # X, Y = np.meshgrid(np.arange(self.lat_start, self.lat_end + 0.04, 0.04 * 20), vcoord2[::-1] / 1000)

        # q = self.ax.quiver(X, Y, vert_cross_V[..., ::20],
        #                    vert_cross_W[..., ::20] / 100 * 86400, scale=400)

        # self.ax.quiverkey(q, 0.85, 1.02, 100, r'$100\ hPa\ day^{-1}$', labelpos='E', transform=self.ax.transAxes,
        #                  labelsep=0.03,
        #                  fontproperties={'size': 12})

        self.add_terrain(self.xyline)

        self.ax.set_xlim(0, 240)

        # self.ax.set_xlim(0, 600)
        # self.ax.set_xticks(np.linspace(0, 600, 7, endpoint=True))
        # self.ax.set_xticklabels([u'85\N{DEGREE SIGN}E', u'90\N{DEGREE SIGN}E', u'95\N{DEGREE SIGN}E', u'100\N{DEGREE SIGN}E',
        #                          u'105\N{DEGREE SIGN}E', u'110\N{DEGREE SIGN}E', u'115\N{DEGREE SIGN}E'])
        # self.ax.xaxis.set_label_coords(1.06, -0.018)
        #
        cbar = plt.colorbar(ctf, orientation='vertical', pad=0.01)
        # # cbar.ax.set_xlabel('K day$^{-1}$', fontsize=11)
        cbar.set_ticks(np.linspace(0, 20, 11))
        # cbar.set_ticklabels([-7, -5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5, 7])
        cbar.ax.tick_params(labelsize=12)
        #
        # # self.ax.set_title('Anomalies in summer total diabatic heating', fontweight='bold', pad=18, fontsize=12)
        # self.ax.text(0, 1.02, 'Reduced topography - Control', ha='left', va='center', transform=self.ax.transAxes,
        #             fontsize=14)
        # self.ax.text(1, 1.02, '2001-2005 JJA', ha='right', va='center', transform=self.ax.transAxes, fontsize=10)

    def save_fig(self, imagename=None, format='png', dpi=550, transparent=True):

        if imagename is not None:
            if format != 'svg':  # 'png','pdf', ...
                self.fig.savefig(imagename + '.' + format, dpi=dpi, transparent=transparent)
            else:
                self.fig.savefig(imagename + '.' + format)
        else:
            plt.show()


if __name__ == '__main__':
    data_d1 = xr.open_dataset(
        '/project/pr133/rxiang/data/cosmo/EAS04_topo1/szn/QV/2001-2005.QV.JJA.nc')
    data_d2 = xr.open_dataset(
        '/project/pr133/rxiang/data/cosmo/EAS04_topo2/szn/QV/2001-2005.QV.JJA.nc')
    data_c = xr.open_dataset('/store/c2sm/pr04/rxiang/data_lmp/01010100_EAS11_topo2/lm_fine/24h3D/TMPHYS_SUM.nc')
    data_c2 = xr.open_dataset('/store/c2sm/pr04/rxiang/data_lmp/01010100_EAS11_topo2/lm_fine/3h3D/W.nc')


    # DIAB = (data_d1['QV'].values[0, ::-1, ...] - data_d2['QV'].values[0, ::-1, ...])*1000
    DIAB = data_d2['QV'].values[0, ::-1, ...] * 1000

    rlon = np.round(data_d1['rlon'].values[...], 2)
    rlat = np.round(data_d1['rlat'].values[...], 2)

    lon = data_d1['lon'].values[...]
    lat = data_d1['lat'].values[...]

    vcoord = data_c2['pressure'].values[...]
    vcoord = pva.pres2alt(vcoord)
    vcoord2 = data_c2['pressure'].values[...]
    vcoord2 = pva.pres2alt(vcoord2)
    vcoord_ = wrf.destagger(vcoord2, 0)

    # data = Plot_Cross(rlon_start=-26.64, rlon_end=-2.4, rlat_start=-1.15, rlat_end=1.61, zmax=14)
    data = Plot_Cross(rlon_start=-19.04, rlon_end=-9.08, rlat_start=0.33, rlat_end=0.33, zmax=14)
    data.add_profile(DIAB, DIAB)

    data.save_fig('/project/pr133/rxiang/figure/paper1/results/TENV/qv_topo2_sm.png', format='png', dpi=550,
                   transparent=True)
    plt.show()
    # data.save_fig()
