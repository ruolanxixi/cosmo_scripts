# clean up for cross-section plot
# Author: Ruolan Xiang

# --------------------------------------------------
# Imports
# --------------------------------------------------
import sys
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy as ctp
import cartopy.crs as ccrs
import pickle
import glob
import metpy.calc as mpcalc
import matplotlib as mpl
import xesmf as xe

mpl.rcParams['figure.dpi'] = 300
from cosmoFunctions import uvrot2uv_3D_py
from metpy.interpolate import cross_section
from metpy.units import units
import matplotlib.gridspec as gridspec
from cosmoFunctions import haversine
import pvlib.atmosphere as pva

# --------------------------------------------------
# Define
# --------------------------------------------------
lon_cs = 100
ext = [lon_cs-20, lon_cs+20, 0, 55]
seasons = ('DJF', 'JJA')

# --------------------------------------------------
# Load data
# --------------------------------------------------
sims = ('ctrl', 'lgm')

data = {}
for season in seasons:
    data[season] = {}
    for sim in sims:
        ds0 = xr.open_dataset(f'/project/pr133/rxiang/data/extpar/laf_{sim}.nc')
        ds1 = xr.open_dataset(f'/scratch/snx3000/rxiang/data/cosmo/EAS11_{sim}/szn/U/01-05.U.{season}.nc')
        ds1 = ds1.isel(pressure=slice(None, None, -1))
        ds2 = xr.open_dataset(f'/scratch/snx3000/rxiang/data/cosmo/EAS11_{sim}/szn/V/01-05.V.{season}.nc')
        ds2 = ds2.isel(pressure=slice(None, None, -1))
        ds3 = xr.open_dataset(f'/scratch/snx3000/rxiang/data/cosmo/EAS11_{sim}/szn/W/01-05.W.{season}.nc')
        ds3 = ds3.isel(pressure=slice(None, None, -1))
        ds4 = xr.open_dataset(f'/scratch/snx3000/rxiang/data/cosmo/EAS11_{sim}/szn/T/01-05.T.{season}.nc')
        ds4 = ds4.isel(pressure=slice(None, None, -1))
        ds5 = xr.open_dataset(f'/scratch/snx3000/rxiang/data/cosmo/EAS11_{sim}/szn/TOT_PREC/01-05.TOT_PREC.{season}.nc')

        time = ds1['time'].values
        ds4 = ds4.assign_coords(time=('time', time))
        ds5 = ds5.assign_coords(time=('time', time))
        ds0 = ds0.assign_coords(time=('time', time))

        # unrotate horizontal velocity field
        pollat = ds0["rotated_pole"].grid_north_pole_latitude
        pollon = ds0["rotated_pole"].grid_north_pole_longitude
        rot_pole_crs = ccrs.RotatedPole(pole_latitude=pollat, pole_longitude=pollon)
        u_geo, v_geo = uvrot2uv_3D_py(ds1.U.values, ds2.V.values,
                                      ds1.lat.values, ds1.lon.values,
                                      pollat, pollon)
        ds1.U.values = u_geo
        ds2.V.values = v_geo

        # create regional latlon grid and regridder
        target_grid = xe.util.grid_2d(lon0_b=75.0, lon1_b=165.0, d_lon=0.11, lat0_b=0., lat1_b=50., d_lat=0.11)
        regridder = xe.Regridder(ds1, target_grid, 'bilinear')

        # regrid fields
        u = regridder(ds1.U)
        u.name = 'U'
        v = regridder(ds2.V)
        v.name = 'V'
        w = regridder(ds3.W)
        w.name = 'W'
        t = regridder(ds4.T)
        t.name = 'T'
        pr = regridder(ds5.TOT_PREC)
        pr.name = 'TOT_PREC'
        hsurf = regridder(ds0.HSURF)
        hsurf.name = 'HSURF'

        del ds2, ds3, ds4

        # create new dataset
        ds = xr.merge([u, v, w, t, hsurf, pr])
        ds = ds.squeeze('time')
        # ds = xr.merge([ds, hsurf])  # cannot select a dimension to squeeze out which has length greater than one

        # add height values
        p = ds1['pressure'].values
        height = pva.pres2alt(p)
        ds = ds.assign_coords(height=('pressure', height))
        ds = ds.swap_dims({'pressure': 'height'})

        ds = ds.metpy.assign_crs(grid_mapping_name='latitude_longitude', earth_radius=6371229.0)
        # --------------------------------------------------
        #  Creat cross section
        # --------------------------------------------------
        ds_cs = ds.metpy.parse_cf()

        start_lat = 5.0
        start_lon = lon_cs
        end_lat = 50.0
        end_lon = lon_cs
        distance = haversine((start_lat, start_lon), (end_lat, end_lon))
        start = (start_lat, start_lon)
        end = (end_lat, end_lon)
        ds_cs['y'] = ds_cs['lat'].values[:, 0]
        ds_cs['x'] = ds_cs['lon'].values[0, :]

        cross = cross_section(ds_cs, start, end, steps=int(distance / 12) + 1).set_coords(('lat', 'lon'))
        #
        # update unit
        cross['U'] = cross['U'] * units('m/s')
        cross['V'] = cross['V'] * units('m/s')
        cross['W'] = cross['W'] * units('m/s')

        # Tangential and normal horizontal wind
        cross['t_wind'], cross['n_wind'] = mpcalc.cross_section_components(cross['U'], cross['V'])
        cross['t_wind'] = cross['t_wind'].metpy.convert_units('knots')
        cross['n_wind'] = cross['n_wind'].metpy.convert_units('knots')
        #
        # compute potential temperature
        cross['theta'] = mpcalc.potential_temperature(cross['pressure'] * units.Pa, cross['T'] * units.kelvin)

        data[season][sim] = cross
    data[season]['diff'] = data[season]['lgm'] - data[season]['ctrl']
# --------------------------------------------------
# %% Plot
# --------------------------------------------------
# position of cross section
fig, axs = plt.subplots(1, 1, subplot_kw={'projection': rot_pole_crs})

coastline = ctp.feature.NaturalEarthFeature('physical', 'coastline', '10m',
                                            edgecolor='black', facecolor='none')
borders = ctp.feature.NaturalEarthFeature('cultural',
                                          'admin_0_boundary_lines_land', '10m', edgecolor='grey',
                                          facecolor='none')
axs.set_extent([65, 173, 0, 61], crs=ccrs.PlateCarree())
axs.add_feature(coastline)
axs.add_feature(borders)
axs.plot([start_lon, end_lon], [start_lat, end_lat],
         color='xkcd:greenish', linewidth=1, marker='o', markersize=3,
         transform=ccrs.PlateCarree())

vals = ds0['HSURF'][0, :, :]
axs.pcolormesh(vals['rlon'], vals['rlat'], vals, transform=rot_pole_crs, cmap=plt.get_cmap('Greys'))

plt.show()
plt.close(fig)

# %% data['ctrl'] section
fig = plt.figure(figsize=(22, 12))
gs = gridspec.GridSpec(7, 3, left=0.05, bottom=0.03, right=0.88, top=0.96,
                       hspace=0.05, wspace=0.0,
                       height_ratios=[5, 0.5, 2, 0.5, 5, 0.5, 2],
                       width_ratios=[6, 0.6, 6])

# -----------------------------------------------------------------
# plot DJF
# -----------------------------------------------------------------
# plot CTRL
ax1 = plt.subplot(gs[0, 0])
pres = np.array([100000, 85000, 70000, 60000, 50000, 40000, 30000, 20000, 15000])
alts = [alt for alt in pva.pres2alt(pres) if alt <= 14000]
ax1.set_yticks([2000, 4000, 6000, 8000, 10000, 12000])
ax1.set_ylim(1, pva.pres2alt(15000))
ax1.set_yticklabels([2, 4, 6, 8, 10, 12], fontsize=19)
ax1.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
ax1.set_xlim(5, 50)

axp1 = ax1.twinx()

axp1.set_ylim(1, pva.pres2alt(15000))
axp1.set_yticks([])
axp1.set_yticklabels([])
# axp1.set_ylabel('Pressure (hPa)', fontsize=19, labelpad=2, rotation=90)

ax1.set_ylabel('Height (km)', fontsize=19, labelpad=2, rotation=90)

ax1.set_xticks([10, 20, 30, 40, 50])
ax1.set_xticklabels(
        [u'10\N{DEGREE SIGN}N', u'20\N{DEGREE SIGN}N', u'30\N{DEGREE SIGN}N', u'40\N{DEGREE SIGN}N',
         u'50\N{DEGREE SIGN}N'], fontsize=19)
ax1.xaxis.set_label_coords(1.06, -0.018)

level = np.arange(250, 380, 5)
ct = ax1.contour(data['DJF']['ctrl']['y'], data['DJF']['ctrl']['height'], data['DJF']['ctrl']['theta'], levels=level, colors='black', linewidths=1)
ct.clabel(ct.levels[1::2], fontsize=16, colors='xkcd:charcoal', inline=1, inline_spacing=8, fmt='%i', rightside_up=True, use_clabeltext=True)
w_contourf = ax1.contourf(data['DJF']['ctrl']['y'], data['DJF']['ctrl']['height'], data['DJF']['ctrl']['W'],
                          cmap='RdBu_r', extend='both', levels=np.arange(-0.05, 0.055, 0.005))

# Plot winds using the axes interface directly, with some custom indexing to make the barbs less crowded
ax1.barbs(data['DJF']['ctrl']['y'][::15000], data['DJF']['ctrl']['height'],
          data['DJF']['ctrl']['t_wind'][:, ::15000],
          data['DJF']['ctrl']['W'][:, ::15000], color='green',
          pivot='middle', length=6,
          sizes={'spacing': 0.2,
                 'height': 0.6,
                 'width': 0.6,
                 'emptybarb': 0.2},
          linewidth=1.7
          )

ax1.plot(data['DJF']['ctrl']['y'], data['DJF']['ctrl']['HSURF'], color='black', linewidth=0.01)
ax1.fill_between(data['DJF']['ctrl']['y'], data['DJF']['ctrl']['HSURF'], color='black', zorder=100)

ax1.text(-0.01, 1, '(a)', ha='right', va='bottom', transform=ax1.transAxes, fontsize=19)
ax1.set_title('DJF', pad=12, fontsize=19)

axb1 = plt.subplot(gs[2, 0])
axb1.bar(data['DJF']['ctrl']['y'][::5000], data['DJF']['ctrl']['TOT_PREC'][::5000], color='black', width=.5)
axb1.spines['top'].set_visible(False)
axb1.spines['bottom'].set_visible(True)
axb1.spines['left'].set_visible(True)
axb1.spines['right'].set_visible(False)
axb1.tick_params(axis='both', which='major', bottom=False, labelbottom=False, labelsize=19, zorder=10)
axb1.set_ylabel('Precipitation\n' + r'[mm$\,$day$^{-1}$]', size=19, labelpad=2, zorder=10)
axb1.patch.set_alpha(0.)
axb1.set_xlim(5, 50)

axb1.text(-0.01, 1.1, '(b)', ha='right', va='bottom', transform=axb1.transAxes, fontsize=19)

# -------------------------
# add cross section map
ax_inset = fig.add_axes([0.011, 0.83, 0.13, 0.13], projection=ccrs.PlateCarree())  # 0.787

coastline = ctp.feature.NaturalEarthFeature('physical', 'coastline', '10m',
                                            edgecolor='black', facecolor='none')
borders = ctp.feature.NaturalEarthFeature('cultural',
                                          'admin_0_boundary_lines_land', '10m', edgecolor='grey',
                                          facecolor='none')
ax_inset.set_extent(ext, crs=ccrs.PlateCarree())
ax_inset.add_feature(coastline)
ax_inset.add_feature(borders)
ax_inset.plot([start_lon, end_lon], [start_lat, end_lat],
         color='xkcd:greenish', linewidth=1, marker='o', markersize=3,
         transform=ccrs.PlateCarree())

vals = ds0['HSURF'][0, :, :]
ax_inset.pcolormesh(vals['rlon'], vals['rlat'], vals, transform=rot_pole_crs, cmap=plt.get_cmap('Greys'))

# -------------------------
# plot difference
ax2 = plt.subplot(gs[4, 0])
pres = np.array([100000, 85000, 70000, 60000, 50000, 40000, 30000, 20000, 15000])
alts = [alt for alt in pva.pres2alt(pres) if alt <= 14000]
ax2.set_yticks([2000, 4000, 6000, 8000, 10000, 12000])
ax2.set_ylim(10, pva.pres2alt(15000))
ax2.set_yticklabels([2, 4, 6, 8, 10, 12], fontsize=19)
ax2.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
ax2.set_xlim(5, 50)

axp2 = ax2.twinx()

axp2.set_ylim(10, pva.pres2alt(15000))
axp2.set_yticks([])
axp2.set_yticklabels([])
# axp2.set_ylabel('Pressure (hPa)', fontsize=19, labelpad=2, rotation=90)

ax2.set_ylabel('Height (km)', fontsize=19, labelpad=2, rotation=90)

ax2.set_xticks([10, 20, 30, 40, 50])
ax2.set_xticklabels(
        [u'10\N{DEGREE SIGN}N', u'20\N{DEGREE SIGN}N', u'30\N{DEGREE SIGN}N', u'40\N{DEGREE SIGN}N',
         u'50\N{DEGREE SIGN}N'], fontsize=19)
ax2.xaxis.set_label_coords(1.06, -0.018)

level = np.arange(-10, 10, 1)
ct = ax2.contour(data['DJF']['diff']['y'], data['DJF']['diff']['height'], data['DJF']['diff']['theta'], levels=level, colors='black', linewidths=1)
for line in ct.collections:
    line.set_linestyle('-')
ct.clabel(ct.levels, fontsize=16, colors='xkcd:charcoal', inline=1, inline_spacing=8, fmt='%i', rightside_up=True, use_clabeltext=True)
w_contourf = ax2.contourf(data['DJF']['diff']['y'], data['DJF']['diff']['height'], data['DJF']['diff']['W'],
                          cmap='RdBu_r', extend='both', levels=np.arange(-0.05, 0.055, 0.005))

# Plot winds using the axes interface directly, with some custom indexing to make the barbs less crowded
ax2.barbs(data['DJF']['diff']['y'][::15000], data['DJF']['diff']['height'],
          data['DJF']['diff']['t_wind'][:, ::15000],
          data['DJF']['diff']['W'][:, ::15000], color='green',
          pivot='middle', length=6,
          sizes={'spacing': 0.2,
                 'height': 0.6,
                 'width': 0.6,
                 'emptybarb': 0.2},
          linewidth=1.7
          )

ax2.plot(data['DJF']['lgm']['y'], data['DJF']['lgm']['HSURF'], color='black', linewidth=0.01)
ax2.plot(data['DJF']['lgm']['y'], data['DJF']['ctrl']['HSURF'], color='black', linewidth=0.01)
ax2.fill_between(data['DJF']['lgm']['y'], data['DJF']['lgm']['HSURF'], color='grey', zorder=100)
ax2.fill_between(data['DJF']['lgm']['y'], data['DJF']['ctrl']['HSURF'], color='black', zorder=100)

ax2.text(-0.01, 1, '(c)', ha='right', va='bottom', transform=ax2.transAxes, fontsize=19)

axb2 = plt.subplot(gs[6, 0])
axb2.bar(data['DJF']['diff']['y'][::5000], data['DJF']['diff']['TOT_PREC'][::5000], color='black', width=.5)
axb2.spines['top'].set_visible(False)
axb2.spines['bottom'].set_visible(True)
axb2.spines['left'].set_visible(True)
axb2.spines['right'].set_visible(False)
axb2.spines['bottom'].set_position('zero')
axb2.tick_params(axis='both', which='major', bottom=False, labelbottom=False, labelsize=19, zorder=10)
axb2.set_ylabel('Precipitation\n' + r'[mm$\,$day$^{-1}$]', size=19, labelpad=2, zorder=10)
axb2.patch.set_alpha(0.)
axb2.set_xlim(5, 50)

axb2.text(-0.01, 1.1, '(d)', ha='right', va='bottom', transform=axb2.transAxes, fontsize=19)

# -----------------------------------------------------------------
# plot JJA
# -----------------------------------------------------------------
# plot CTRL
ax1 = plt.subplot(gs[0, 2])
pres = np.array([100000, 85000, 70000, 60000, 50000, 40000, 30000, 20000, 15000])
alts = [alt for alt in pva.pres2alt(pres) if alt <= 14000]
ax1.set_yticks([2000, 4000, 6000, 8000, 10000, 12000])
ax1.set_ylim(1, pva.pres2alt(15000))
ax1.set_yticklabels([2, 4, 6, 8, 10, 12], fontsize=19)
ax1.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
ax1.set_xlim(5, 50)

axp1 = ax1.twinx()

axp1.set_ylim(1, pva.pres2alt(15000))
axp1.set_yticks(alts[:])
axp1.set_yticklabels(list(map(int, pres[:len(alts)] / 100)), fontsize=19)
axp1.set_ylabel('Pressure (hPa)', fontsize=19, labelpad=2, rotation=90)

# ax1.set_ylabel('Height (km)', fontsize=19, labelpad=2, rotation=90)

ax1.set_xticks([10, 20, 30, 40, 50])
ax1.set_xticklabels(
        [u'10\N{DEGREE SIGN}N', u'20\N{DEGREE SIGN}N', u'30\N{DEGREE SIGN}N', u'40\N{DEGREE SIGN}N',
         u'50\N{DEGREE SIGN}N'], fontsize=19)
ax1.xaxis.set_label_coords(1.06, -0.018)

level = np.arange(250, 380, 5)
ct = ax1.contour(data['JJA']['ctrl']['y'], data['JJA']['ctrl']['height'], data['JJA']['ctrl']['theta'], levels=level, colors='black', linewidths=1)
ct.clabel(ct.levels[1::2], fontsize=16, colors='xkcd:charcoal', inline=1, inline_spacing=8, fmt='%i', rightside_up=True, use_clabeltext=True)
w_contourf = ax1.contourf(data['JJA']['ctrl']['y'], data['JJA']['ctrl']['height'], data['JJA']['ctrl']['W'],
                          cmap='RdBu_r', extend='both', levels=np.arange(-0.05, 0.055, 0.005))

# add colorbar
cax = fig.add_axes([ax1.get_position().x1+0.055, ax1.get_position().y0, 0.012, ax1.get_position().height])
cbar = fig.colorbar(w_contourf, cax=cax, orientation='vertical', extend='max', ticks=np.arange(-0.05, 0.055, 0.01))
cbar.ax.tick_params(labelsize=16)
cbar.ax.minorticks_off()
cbar.set_label('w [m$\,$s$^{-1}$]', fontsize=19)

# Plot winds using the axes interface directly, with some custom indexing to make the barbs less crowded
ax1.barbs(data['JJA']['ctrl']['y'][::15000], data['JJA']['ctrl']['height'],
          data['JJA']['ctrl']['t_wind'][:, ::15000],
          data['JJA']['ctrl']['W'][:, ::15000], color='green',
          pivot='middle', length=6,
          sizes={'spacing': 0.2,
                 'height': 0.6,
                 'width': 0.6,
                 'emptybarb': 0.2},
          linewidth=1.7
          )

ax1.plot(data['JJA']['ctrl']['y'], data['JJA']['ctrl']['HSURF'], color='black', linewidth=0.01)
ax1.fill_between(data['JJA']['ctrl']['y'], data['JJA']['ctrl']['HSURF'], color='black', zorder=100)

ax1.text(-0.01, 1, '(e)', ha='right', va='bottom', transform=ax1.transAxes, fontsize=19)
ax1.set_title('JJA', pad=12, fontsize=19)

axb1 = plt.subplot(gs[2, 2])
axb1.bar(data['JJA']['ctrl']['y'][::5000], data['JJA']['ctrl']['TOT_PREC'][::5000], color='black', width=.5)
axb1.spines['top'].set_visible(False)
axb1.spines['bottom'].set_visible(True)
axb1.spines['left'].set_visible(True)
axb1.spines['right'].set_visible(False)
axb1.tick_params(axis='both', which='major', bottom=False, labelbottom=False, labelsize=19, zorder=10)
# axb1.set_ylabel('Precipitation\n' + r'[mm$\,$day$^{-1}$]', size=19, labelpad=2, zorder=10)
axb1.patch.set_alpha(0.)
axb1.set_xlim(5, 50)

axb1.text(-0.01, 1.1, '(f)', ha='right', va='bottom', transform=axb1.transAxes, fontsize=19)

# -------------------------
# plot difference
ax2 = plt.subplot(gs[4, 2])
pres = np.array([100000, 85000, 70000, 60000, 50000, 40000, 30000, 20000, 15000])
alts = [alt for alt in pva.pres2alt(pres) if alt <= 14000]
ax2.set_yticks([2000, 4000, 6000, 8000, 10000, 12000])
ax2.set_ylim(1, pva.pres2alt(15000))
ax2.set_yticklabels([2, 4, 6, 8, 10, 12], fontsize=19)
ax2.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
ax2.set_xlim(5, 50)

axp2 = ax2.twinx()

axp2.set_ylim(10, pva.pres2alt(15000))
axp2.set_yticks(alts[:])
axp2.set_yticklabels(list(map(int, pres[:len(alts)] / 100)), fontsize=19)
axp2.set_ylabel('Pressure (hPa)', fontsize=19, labelpad=2, rotation=90)

# ax2.set_ylabel('Height (km)', fontsize=19, labelpad=2, rotation=90)

ax2.set_xticks([10, 20, 30, 40, 50])
ax2.set_xticklabels(
        [u'10\N{DEGREE SIGN}N', u'20\N{DEGREE SIGN}N', u'30\N{DEGREE SIGN}N', u'40\N{DEGREE SIGN}N',
         u'50\N{DEGREE SIGN}N'], fontsize=19)
ax2.xaxis.set_label_coords(1.06, -0.018)

level = np.arange(-10, 10, 1)
ct = ax2.contour(data['JJA']['diff']['y'], data['JJA']['diff']['height'], data['JJA']['diff']['theta'], levels=level, colors='black', linewidths=1)
for line in ct.collections:
    line.set_linestyle('-')
ct.clabel(ct.levels, fontsize=16, colors='xkcd:charcoal', inline=1, inline_spacing=8, fmt='%i', rightside_up=True, use_clabeltext=True)
w_contourf = ax2.contourf(data['JJA']['diff']['y'], data['JJA']['diff']['height'], data['JJA']['diff']['W'],
                          cmap='RdBu_r', extend='both', levels=np.arange(-0.05, 0.055, 0.005))

cax = fig.add_axes([ax2.get_position().x1+0.055, ax2.get_position().y0, 0.012, ax2.get_position().height])
cbar = fig.colorbar(w_contourf, cax=cax, orientation='vertical', extend='max', ticks=np.arange(-0.05, 0.055, 0.01))
cbar.ax.tick_params(labelsize=16)
cbar.ax.minorticks_off()
cbar.set_label('w [m$\,$s$^{-1}$]', fontsize=19)

# Plot winds using the axes interface directly, with some custom indexing to make the barbs less crowded
ax2.barbs(data['JJA']['diff']['y'][::15000], data['JJA']['diff']['height'],
          data['JJA']['diff']['t_wind'][:, ::15000],
          data['JJA']['diff']['W'][:, ::15000], color='green',
          pivot='middle', length=6,
          sizes={'spacing': 0.2,
                 'height': 0.6,
                 'width': 0.6,
                 'emptybarb': 0.2},
          linewidth=1.7
          )

ax2.plot(data['JJA']['lgm']['y'], data['JJA']['lgm']['HSURF'], color='black', linewidth=0.01)
ax2.plot(data['JJA']['lgm']['y'], data['JJA']['ctrl']['HSURF'], color='black', linewidth=0.01)
ax2.fill_between(data['JJA']['lgm']['y'], data['JJA']['lgm']['HSURF'], color='grey', zorder=100)
ax2.fill_between(data['JJA']['lgm']['y'], data['JJA']['ctrl']['HSURF'], color='black', zorder=100)

ax2.text(-0.01, 1, '(g)', ha='right', va='bottom', transform=ax2.transAxes, fontsize=19)

axb2 = plt.subplot(gs[6, 2])
axb2.bar(data['JJA']['diff']['y'][::5000], data['JJA']['diff']['TOT_PREC'][::5000], color='black', width=.5)
axb2.spines['top'].set_visible(False)
axb2.spines['bottom'].set_visible(True)
axb2.spines['left'].set_visible(True)
axb2.spines['right'].set_visible(False)
axb2.spines['bottom'].set_position('zero')
axb2.tick_params(axis='both', which='major', bottom=False, labelbottom=False, labelsize=19, zorder=10)
# axb2.set_ylabel('Precipitation\n' + r'[mm$\,$day$^{-1}$]', size=19, labelpad=2, zorder=10)
axb2.patch.set_alpha(0.)
axb2.set_xlim(5, 50)

axb2.text(-0.01, 1.1, '(h)', ha='right', va='bottom', transform=axb2.transAxes, fontsize=19)

plt.show()

plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + f'{lon_cs}E.png', dpi=500, transparent='True')
