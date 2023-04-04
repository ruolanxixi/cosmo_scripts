import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.ma as ma
import xarray as xr

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

    def __init__(self, lon_start, lon_end, lat_start, lat_end, zmax=16):

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
        self.ax.set_ylim(0, pva.pres2alt(10000) / 1000)
        self.axp.set_ylim(0, pva.pres2alt(10000) / 1000)

        self.ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        pres = np.array([100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 20000, 10000])
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

    def add_terrain(self, xyline, level=57):
        # FIXME
        f = xr.open_dataset('/project/pr133/rxiang/data/cross/terrain.ctrl.zonmean.04.nc')
        hsurf = f['HSURF'].values[...]

        hsurf = np.repeat(hsurf[np.newaxis, ...], level, axis=0)

        terrain = wrf.interp2dxy(hsurf, xyline)[1]

        self.ax.plot(np.arange(self.lat_start, self.lat_end + 0.04, 0.04), terrain / 1000, color='black', linewidth=0.01)
        self.ax.fill_between(np.arange(self.lat_start, self.lat_end + 0.04, 0.04), terrain / 1000, color='black')

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

        vert_cross_W = W[:, lat_start_id:lat_end_id + 1, 0]
        vert_cross_W = np.ma.array(vert_cross_W, mask=np.isnan(vert_cross_W))

        vert_cross_U = U[:, lat_start_id:lat_end_id + 1, 0]
        vert_cross_U = np.ma.array(vert_cross_U, mask=np.isnan(vert_cross_U))

        vert_cross_V = V[:, lat_start_id:lat_end_id + 1, 0]
        vert_cross_V = np.ma.array(vert_cross_V, mask=np.isnan(vert_cross_V))

        cmap = custom_div_cmap(41, cmc.vik)
        levels = [-2, -1.5, -1, -0.5, 0.5, 1, 3, 6, 9, 12, 15]
        # levels = MaxNLocator(nbins=14).tick_values(-7, 7)


        divnorm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=15)
        z = vcoord[::-1] / 1000
        ctf = self.ax.contourf(np.arange(self.lat_start, self.lat_end + 0.04, 0.04*5), vcoord[::-1] / 1000,
                               vert_cross[:, ::5], extend='both',
                               cmap=cmap, levels=levels, norm=divnorm)

        X, Y = np.meshgrid(np.arange(self.lat_start, self.lat_end + 0.04, 0.04 * 20), vcoord2[::-1] / 1000)

        q = self.ax.quiver(X, Y, vert_cross_V[..., ::20],
                       vert_cross_W[..., ::20] * 500, scale=90)

        self.ax.quiverkey(q, 0.83, 1.02, 10, r'$10\ m\ s^{-1}$', labelpos='E', transform=self.ax.transAxes, labelsep=0.03,
                      fontproperties={'size': 10})

        self.add_terrain(self.xyline)

        self.ax.set_xlim(self.lat_start, self.lat_end)
        self.ax.set_xticks(np.linspace(self.lat_start, self.lat_end, 5, endpoint=True))
        self.ax.set_xticklabels([u'20\N{DEGREE SIGN}N', u'25\N{DEGREE SIGN}N', u'30\N{DEGREE SIGN}N',
                                u'35\N{DEGREE SIGN}N', u'40\N{DEGREE SIGN}N'])
        self.ax.xaxis.set_label_coords(1.06, -0.018)

        cbar = plt.colorbar(ctf, orientation='horizontal', pad=0.01)
        cbar.ax.set_xlabel('K day$^{-1}$', fontsize=11)
        cbar.set_ticks([-2, -1.5, -1, -0.5, 0.5, 1, 3, 6, 9, 12, 15])
        cbar.set_ticklabels([-2, -1.5, -1, -0.5, 0.5, 1, 3, 6, 9, 12, 15])
        cbar.ax.tick_params(labelsize=10)

        self.ax.set_title('Summer total diabatic heating (95°E - 105°E)', fontweight='bold', pad=18, fontsize=12)
        self.ax.text(0, 1.02, 'Control', ha='left', va='center', transform=self.ax.transAxes,
                fontsize=10)
        # self.ax.text(1, 1.02, 'averaged within 95°E - 105°E', ha='right', va='center', transform=self.ax.transAxes, fontsize=10)

    def save_fig(self, imagename=None, format='png', dpi=550, transparent=True):

        if imagename is not None:
            if format != 'svg':  # 'png','pdf', ...
                self.fig.savefig(imagename + '.' + format, dpi=dpi, transparent=transparent)
            else:
                self.fig.savefig(imagename + '.' + format)
        else:
            plt.show()


if __name__ == '__main__':
    data_d = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS04_ctrl/szn/DIAB_SUM/2001-2005.DIAB_SUM.JJA.zonmean.nc')
    data_c = xr.open_dataset('/store/c2sm/pr04/rxiang/data_lmp/01010100_EAS11_ctrl/lm_fine/24h3D/TMPHYS_SUM.nc')
    data_c2 = xr.open_dataset('/store/c2sm/pr04/rxiang/data_lmp/01010100_EAS11_ctrl/lm_fine/3h3D/W.nc')

    data_u1 = xr.open_dataset(
        '/project/pr133/rxiang/data/cosmo/EAS04_ctrl/szn/U/2001-2005.U.JJA.zonmean.nc')
    data_v1 = xr.open_dataset(
        '/project/pr133/rxiang/data/cosmo/EAS04_ctrl/szn/V/2001-2005.V.JJA.zonmean.nc')
    data_w1 = xr.open_dataset(
        '/project/pr133/rxiang/data/cosmo/EAS04_ctrl/szn/W/2001-2005.W.JJA.zonmean.nc')

    W = data_w1['W'].values[0, :, :]
    U = data_u1['U'].values[0, :, :]
    V = data_v1['V'].values[0, :, :]

    DIAB = data_d['DIAB_SUM'].values[0, ...]

    lon = data_d['lon'].values[...]
    lat = data_d['lat'].values[...]

    vcoord = data_c['pressure'].values[...]
    vcoord = pva.pres2alt(vcoord)
    vcoord2 = data_c2['pressure'].values[...]
    vcoord2 = pva.pres2alt(vcoord2)
    vcoord_ = wrf.destagger(vcoord, 0)

    # brunt = brunt_vaisala_frequency(h=vcoord_, pt=pt)

    data = Plot_Cross(lon_start=0, lon_end=0, lat_start=20, lat_end=40, zmax=16)
    data.add_profile(DIAB, DIAB)

    # dbz = data_t.variables['DBZ'][0, ...]
    # data.add_profile(dbz, varname='DBZ', colorbar=False)

    data.save_fig('/project/pr133/rxiang/figure/EAS04/analysis/diabatic/ctrl', format='png', dpi=550, transparent=True)
    plt.show()
    # data.save_fig()
