import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import wrf
from netCDF4 import Dataset
from math import radians, cos, sin, asin, sqrt, ceil, floor
from collections import UserDict
import pvlib.atmosphere as pva
import cartopy.crs as ccrs
import cmcrameri.cm as cmc

import metpy
from pyproj import Geod

class Plot_Cross(UserDict):
    """ Plot cross-section in rotated coordinates"""

    def __init__(self, rlon_start, rlon_end, rlat_start, rlat_end, zmax=16, lat_pole=61, lon_pole=-63.7):

        self.rlon_start = rlon_start
        self.rlon_end = rlon_end

        self.rlat_start = rlat_start
        self.rlat_end = rlat_end

        self.lat_pole = lat_pole
        self.lon_pole = lon_pole

        # initialize figure
        self.fig, self.axes = plt.subplots(1, 1, figsize=(5, 6), constrained_layout=True)
        self.ax = plt.subplot(1, 1, 1)

        self.ax.set_ylabel('Height (km)', fontsize=13, labelpad=1.5)
        self.ax.tick_params(axis='both', which='major', labelsize=12)
        self.axp = self.ax.twinx()
        self.ax.set_ylim(0, zmax)
        self.axp.set_ylim(0, zmax)

        self.ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        pres = np.array([100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 20000, 10000])
        alts = [alt for alt in pva.pres2alt(pres) / 1000 if alt <= zmax]

        self.axp.set_yticks(alts[:])
        self.axp.set_yticklabels(list(map(int, pres[:len(alts)] / 100)), fontsize=12)
        self.axp.set_ylabel('Pressure (hPa)', rotation=270, fontsize=13, labelpad=5.5)

    def get_id(self, var, value):

        id = np.argwhere(var == value)

        return id[0][0]

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

    def set_xticks_distance(self, xlen, distance, step):

        self.ax.set_xlim(0, xlen)
        self.ax.set_xticks(np.arange(0, xlen / distance * floor(distance / step) * step + xlen / distance * step,
                                     xlen / distance * step))
        self.ax.set_xticklabels(list(map(str, np.arange(0, (floor(distance / step) + 1) * step, step))), fontsize=12)
        self.ax.set_xlabel('Distance (km)', fontsize=14, labelpad=4.5)

    def add_terrain(self, xyline, level=57):
        # FIXME
        f = Dataset('/scratch/snx3000/rxiang/lmp_EAS11_ctrl/wd/00090100_EAS11_ctrl/lm_coarse/lffd20000901000000c.nc')
        hsurf = f.variables['HSURF'][0, ...]

        hsurf = np.repeat(hsurf[np.newaxis, ...], level, axis=0)
        terrain = wrf.interp2dxy(hsurf, xyline)[1]
        self.ax.plot(np.arange(terrain.shape[0]), terrain / 1000, color='black', linewidth=2)

    def add_profile(self, var, varname, w_factor=None, colorbar=True):

        rlon_start_id = self.get_id(rlon, self.rlon_start)
        rlat_start_id = self.get_id(rlat, self.rlat_start)

        rlon_end_id = self.get_id(rlon, self.rlon_end)
        rlat_end_id = self.get_id(rlat, self.rlat_end)

        start = (rlon_start_id, rlat_start_id)
        end = (rlon_end_id, rlat_end_id)

        crs_geo = ccrs.PlateCarree()
        crs_rot_pole = ccrs.RotatedPole(pole_longitude=self.lon_pole, pole_latitude=self.lat_pole)

        lon_start, lat_start = crs_geo.transform_point(self.rlon_start, self.rlat_start, crs_rot_pole)
        lon_end, lat_end = crs_geo.transform_point(self.rlon_end, self.rlat_end, crs_rot_pole)

        # great circle distance between two points on the earth (specified in decimal degrees)
        self.distance = self.haversine(lon_start, lat_start, lon_start, lat_end)

        # return the x, y points for a line within a two-dimensional grid (the cross section)
        self.xyline = wrf.xy(DIAB, start_point=start, end_point=end)

        # Return a cross section for a three-dimensional field
        self.lon_ = wrf.interp2dxy(np.repeat(lon[np.newaxis, ...], 57, axis=0), self.xyline)
        self.lat_ = wrf.interp2dxy(np.repeat(lat[np.newaxis, ...], 57, axis=0), self.xyline)

        self.add_terrain(self.xyline)

        vert_cross = wrf.interp2dxy(var, self.xyline)
        vert_cross = np.ma.array(vert_cross, mask=np.isnan(vert_cross))

        ctf = self.ax.contourf(np.arange(vert_cross.shape[1]), vcoord[::-1] / 1000, vert_cross, extend='max', cmap=cmc.vik)

        # ct = self.ax.contour(np.arange(vert_cross_Te.shape[1]), vcoord_[::-1] / 1000, vert_cross_Te, linewidths=0.6, colors='#868E96', alpha=0.8)
        # self.ax.clabel(ct, var_consts("thetae")['clabel'], inline=1, fmt='%d')

        if w_factor is not None:
            vert_cross_W = wrf.interp2dxy(W, self.xyline)
            vert_cross_W = np.ma.array(vert_cross_W, mask=np.isnan(vert_cross)) * w_factor

            vert_cross_U = wrf.interp2dxy(U, self.xyline)
            vert_cross_U = np.ma.array(vert_cross_U, mask=np.isnan(vert_cross))

            vert_cross_V = wrf.interp2dxy(V, self.xyline)
            vert_cross_V = np.ma.array(vert_cross_V, mask=np.isnan(vert_cross))

            vert_cross_tang, vert_cross_norm = self.vector_components(vert_cross_U, vert_cross_V)

            self.ax.quiver(np.arange(0, vert_cross_U.shape[1], 5), vcoord_[::-1] / 1000, vert_cross_tang[..., ::5],
                           vert_cross_W[..., ::5])

        # cpt = self.ax.contour(np.arange(vert_cross_T.shape[1]), vcoord_[::-1] / 1000, vert_cross_T,
        #                       np.arange(230, 390, 2.5),
        #                       linewidths=0.6, cmap=MPL_RED())
        # self.ax.clabel(cpt, np.arange(230, 390, 5), inline=1, fmt='%d')

        # self.set_xticks_distance(vert_cross.shape[1], self.distance, 50)


        # if colorbar:
        #     cbar = plt.colorbar(ctf, orientation='horizontal', pad=0.125)
        #     cbar.ax.set_xlabel(var_consts(varname)['long_name'] + ' (' + var_consts(varname)['units'] + ')',
        #                        fontsize=14)

        # add time labels
        # plt.text(0.731, 0.934, str(date)[-4:] + ' UTC',
        #          bbox=dict(facecolor='white', edgecolor='black', lw=0.7, pad=2.5), fontsize=14.75,
        #          transform=self.ax.transAxes)



    def save_fig(self, imagename=None, format='png', dpi=550, transparent=True):

        if imagename is not None:
            if format != 'svg':  # 'png','pdf', ...
                self.fig.savefig(imagename + '.' + format, dpi=dpi, transparent=transparent)
            else:
                self.fig.savefig(imagename + '.' + format)
        else:
            plt.show()


if __name__ == '__main__':

    data_d = Dataset('/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn/DIAB_SUM/2001-2005.DIAB_SUM.JJA.nc')
    data_c = Dataset('/store/c2sm/pr04/rxiang/data_lmp/01010100_EAS11_ctrl/lm_coarse/3h3D/FI.nc')

    # W = wrf.destagger(data_w.variables['W'][0, ...], -3)
    # U = data_w.variables['U'][0, ...]
    # V = data_w.variables['V'][0, ...]

    DIAB = data_d.variables['DIAB_SUM'][0, ...]

    rlon = np.round(data_d.variables['rlon'][...], 2)
    rlat = np.round(data_d.variables['rlat'][...], 2)

    lon = data_d.variables['lon'][...]
    lat = data_d.variables['lat'][...]

    vcoord = data_c.variables['pressure'][...]
    vcoord = pva.pres2alt(vcoord)
    vcoord_ = wrf.destagger(vcoord, 0)

    # brunt = brunt_vaisala_frequency(h=vcoord_, pt=pt)

    # FIXME -1.96, 2, -1, 2.96

    data = Plot_Cross(rlon_start=-18.78, rlon_end=-10.64, rlat_start=-3.27, rlat_end=2.45, zmax=16)
    data.add_profile(DIAB, DIAB)

    plt.show()

    # dbz = data_t.variables['DBZ'][0, ...]
    # data.add_profile(dbz, varname='DBZ', colorbar=False)

    # data.save_fig('/users/rcui/graphs/cross_' + varname + '_' + str(date), format='svg', dpi=550, transparent=True)
    # data.save_fig()
