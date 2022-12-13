import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.ma as ma

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

import metpy
from pyproj import Geod


class Plot_Cross(UserDict):
    """ Plot cross-section in lonlat coordinates"""

    def __init__(self, lon_start, lon_end, lat_start, lat_end, zmax=14):

        self.lon_start = lon_start
        self.lon_end = lon_end

        self.lat_start = lat_start
        self.lat_end = lat_end

        # initialize figure
        self.fig, self.axes = plt.subplots(1, 1, figsize=(5, 6), constrained_layout=True)
        self.ax = plt.subplot(1, 1, 1)

        self.ax.set_ylabel('Height (km)', fontsize=12, labelpad=1.5)
        self.ax.tick_params(axis='both', which='major', labelsize=12)
        self.axp = self.ax.twinx()
        self.ax.set_ylim(0, pva.pres2alt(15000) / 1000)
        self.axp.set_ylim(0, pva.pres2alt(15000) / 1000)

        self.ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        pres = np.array([100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 20000, 15000])
        alts = [alt for alt in pva.pres2alt(pres) / 1000 if alt <= zmax]

        self.axp.set_yticks(alts[:])
        self.axp.set_yticklabels(list(map(int, pres[:len(alts)] / 100)), fontsize=12)
        self.axp.set_ylabel('Pressure (hPa)', rotation=270, fontsize=12, labelpad=5.7)

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

    def add_terrain(self, xyline, level=57):
        # FIXME
        f2 = Dataset('/project/pr133/rxiang/data/cross/terrain.ctrl.zonmean.nc')
        f1 = Dataset('/project/pr133/rxiang/data/cross/terrain.topo2.zonmean.nc')
        hsurf1 = f1.variables['HSURF'][0, ...]
        hsurf2 = f2.variables['HSURF'][0, ...]

        hsurf1 = np.repeat(hsurf1[np.newaxis, ...], level, axis=0)
        hsurf2 = np.repeat(hsurf2[np.newaxis, ...], level, axis=0)

        terrain2 = wrf.interp2dxy(hsurf2, xyline)[1]
        self.ax.plot(np.arange(self.lat_start, self.lat_end + 0.11, 0.11), terrain2 / 1000, color='black', linewidth=0.01)
        self.ax.fill_between(np.arange(self.lat_start, self.lat_end + 0.11, 0.11), terrain2 / 1000, color='black')

        terrain1 = wrf.interp2dxy(hsurf1, xyline)[1]
        self.ax.plot(np.arange(self.lat_start, self.lat_end + 0.11, 0.11), terrain1 / 1000, color='lightgrey', linewidth=0.01)
        self.ax.fill_between(np.arange(self.lat_start, self.lat_end + 0.11, 0.11), terrain1 / 1000, color='black', alpha=0.5)

    def add_profile(self, var, varname, w_factor=None, colorbar=True):

        lon_start_id = self.get_id(lon, self.lon_start)
        lat_start_id = self.get_id(lat, self.lat_start)

        lon_end_id = self.get_id(lon, self.lon_end)
        lat_end_id = self.get_id(lat, self.lat_end)

        start = (lon_start_id, lat_start_id)
        end = (lon_end_id, lat_end_id)

        # return the x, y points for a line within a two-dimensional grid (the cross section)
        self.xyline = wrf.xy(DIAB, start_point=start, end_point=end)

        # Return a cross section for a three-dimensional field
        # self.lon_ = wrf.interp2dxy(np.repeat(lon[np.newaxis, ...], 57, axis=0), self.xyline)
        # self.lat_ = wrf.interp2dxy(np.repeat(lat[np.newaxis, ...], 57, axis=0), self.xyline)

        # vert_cross = wrf.interp2dxy(var, self.xyline)
        # vert_cross = np.ma.array(vert_cross, mask=np.isnan(vert_cross))

        vert_cross = var[:, lat_start_id:lat_end_id + 1, 0]

        # vert_cross_W = W[:, lat_start_id:lat_end_id + 1, 0]
        # vert_cross_W = np.ma.array(vert_cross_W, mask=np.isnan(vert_cross_W))
        #
        # vert_cross_U = U[:, lat_start_id:lat_end_id + 1, 0]
        # vert_cross_U = np.ma.array(vert_cross_U, mask=np.isnan(vert_cross_U))
        #
        # vert_cross_V = V[:, lat_start_id:lat_end_id + 1, 0]
        # vert_cross_V = np.ma.array(vert_cross_V, mask=np.isnan(vert_cross_V))

        cmap = custom_div_cmap(21, cmc.vik)
        levels = [-7, -5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5, 7]
        # levels = MaxNLocator(nbins=14).tick_values(-7, 7)
        divnorm = colors.TwoSlopeNorm(vmin=-7., vcenter=0., vmax=7)

        ctf = self.ax.contourf(np.arange(self.lat_start, self.lat_end + 0.11, 0.11), vcoord[::-1] / 1000, vert_cross,
                               extend='both',
                               cmap=cmap, levels=levels, norm=divnorm)
        X, Y = np.meshgrid(np.arange(self.lat_start, self.lat_end + 0.04, 0.04 * 20), vcoord[::-1] / 1000)
        # q = self.ax.quiver(X, Y, vert_cross_V[..., ::20],
        #                    vert_cross_W[..., ::20] * 100, scale=20)

        # self.ax.quiverkey(q, 0.85, 1.02, 1, r'$1\ m\ s^{-1}$', labelpos='E', transform=self.ax.transAxes,
        #                   labelsep=0.03,
        #                   fontproperties={'size': 10})

        self.add_terrain(self.xyline)

        # xticks = np.array([self.get_closed_id(lat, 10), self.get_closed_id(lat, 20), self.get_closed_id(lat, 30),
        #                    self.get_closed_id(lat, 40), self.get_closed_id(lat, 50)])
        # labels = ('10°N', '20°N', '30°N', '40°N', '50°N')
        # self.ax.set_xticks(ticks=xticks, labels=labels)
        # self.ax.xaxis.set_label_coords(1.06, -0.018)

        self.ax.set_xlim(10, 50)
        self.ax.set_xticks(np.linspace(10, 50, 5, endpoint=True))
        self.ax.set_xticklabels(['10°N', '20°N', '30°N', '40°N', '50°N'])

        # self.ax.set_xticks(np.linspace(self.lat_start, self.lat_end, 5, endpoint=True))
        # self.ax.set_xticklabels([u'20\N{DEGREE SIGN}N', u'25\N{DEGREE SIGN}N', u'30\N{DEGREE SIGN}N',
        #                          u'35\N{DEGREE SIGN}N', u'40\N{DEGREE SIGN}N'])
        # self.ax.xaxis.set_label_coords(1.06, -0.018)

        cbar = plt.colorbar(ctf, orientation='horizontal', pad=0.01)
        # cbar.ax.set_xlabel('K day$^{-1}$', fontsize=11)
        cbar.set_ticks([-7, -5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5, 7])
        cbar.set_ticklabels([-7, -5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5, 7])
        cbar.ax.tick_params(labelsize=10)

        # self.ax.set_title('Anomalies in summer total diabatic heating', fontweight='bold', pad=18, fontsize=12)
        self.ax.text(0, 1.025, '(c) TENV11 - CTRL11', ha='left', va='center', transform=self.ax.transAxes,
                fontsize=14)
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
    data_d1 = Dataset('/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn/DIAB_SUM/2001-2005.DIAB_SUM.JJA.zonmean.nc')
    data_d2 = Dataset('/project/pr133/rxiang/data/cosmo/EAS11_topo2/szn/DIAB_SUM/2001-2005.DIAB_SUM.JJA.zonmean.nc')
    data_c = Dataset('/store/c2sm/pr04/rxiang/data_lmp/01010100_EAS11_topo2/lm_coarse/3h3D/FI.nc')

    # data_u1 = Dataset(
    #     '/project/pr133/rxiang/data/cosmo/EAS11_topo2/szn/U/2001-2005.U.JJA.zonmean.nc')
    # data_u2 = Dataset(
    #     '/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn/U/2001-2005.U.JJA.zonmean.nc')
    # 
    # data_v1 = Dataset(
    #     '/project/pr133/rxiang/data/cosmo/EAS11_topo2/szn/V/2001-2005.V.JJA.zonmean.nc')
    # data_v2 = Dataset(
    #     '/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn/V/2001-2005.V.JJA.zonmean.nc')
    # 
    # data_w1 = Dataset(
    #     '/project/pr133/rxiang/data/cosmo/EAS11_topo2/szn/W/2001-2005.W.JJA.zonmean.nc')
    # data_w2 = Dataset(
    #     '/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn/W/2001-2005.W.JJA.zonmean.nc')
    # 
    # W = - data_w1.variables['W'][0, ::-1, ...] + data_w2.variables['W'][0, ::-1, ...]
    # U = - data_u1.variables['U'][0, ::-1, ...] + data_u2.variables['U'][0, ::-1, ...]
    # V = - data_v1.variables['V'][0, ::-1, ...] + data_v2.variables['V'][0, ::-1, ...]

    DIAB = - data_d1.variables['DIAB_SUM'][0, ::-1, ...] + data_d2.variables['DIAB_SUM'][0, ::-1, ...]

    lon = data_d1.variables['lon'][...]
    lat = data_d1.variables['lat'][...]

    vcoord = data_c.variables['pressure'][...]
    vcoord = pva.pres2alt(vcoord)
    vcoord_ = wrf.destagger(vcoord, 0)

    # brunt = brunt_vaisala_frequency(h=vcoord_, pt=pt)

    data = Plot_Cross(lon_start=0, lon_end=0, lat_start=10.07, lat_end=50, zmax=14)
    data.add_profile(DIAB, DIAB)

    # dbz = data_t.variables['DBZ'][0, ...]
    # data.add_profile(dbz, varname='DBZ', colorbar=False)

    data.save_fig('/project/pr133/rxiang/figure/paper1/results/TENV/diabatic_diff.png', format='png', dpi=550, transparent=True)
    plt.show()
    # data.save_fig()
