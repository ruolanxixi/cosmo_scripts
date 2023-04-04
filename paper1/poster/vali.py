# -------------------------------------------------------------------------------
# modules
#
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo_notick, pole, colorbar
import cartopy.crs as ccrs
from numpy import inf
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_
from mycolor import wind as windmap
from pyproj import Transformer
import scipy.ndimage as ndimage
import matplotlib
from matplotlib.patches import Rectangle
import numpy.ma as ma

font = {'size': 13}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
sims = ['CTRL11', 'ERA5', 'DIFF']
seasons = ['DJF', 'MAM', 'JJA', 'SON']
mdpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn"
rmpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/remap"
erapath = "/project/pr133/rxiang/data/era5/ot/szn"
imergpath = "/project/pr133/rxiang/data/obs/pr/IMERG/szn"
crupath = "/project/pr133/rxiang/data/obs/tmp/cru/szn"
dt = {}
# labels = [['CTRL11', 'IMERG', 'CTRL11 - IMERG'], ['CTRL11', 'CRU', 'CTRL11 - CRU'],
#           ['CTRL11', 'ERA5', 'CTRL11 - ERA5']]
labels = [['(a) CTRL11', '(b) IMERG', '(c) CTRL11 - IMERG']]
vars = ['v850', 'u850', 'ws850', 'u500', 'v500', 'ws500', 'q850']
dt['CTRL11'], dt['ERA5'], dt['CTRL11_ERA5'], dt['CTRL11_IMERG'], dt['CTRL11_CRU'], dt['DIFF'], \
dt['IMERG'], dt['DIFF_IMERG'], dt['CRU'], dt['DIFF_CRU'] = \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()
lb = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']]

for s in range(len(seasons)):
    season = seasons[s]
    # COSMO 12 km
    dt['CTRL11'][season] = {}
    data = xr.open_dataset(f'{mdpath}/U/2001-2005.U.85000.{season}.nc')
    u = data['U'].values[0, 0, :, :]
    data = xr.open_dataset(f'{mdpath}/V/2001-2005.V.85000.{season}.nc')
    v = data['V'].values[0, 0, :, :]
    ws = np.sqrt(u ** 2 + v ** 2)
    dt['CTRL11'][season]['v850'] = v
    dt['CTRL11'][season]['u850'] = u
    dt['CTRL11'][season]['ws850'] = ws
    data = xr.open_dataset(f'{mdpath}/U/2001-2005.U.50000.{season}.nc')
    u = data['U'].values[0, 0, :, :]
    data = xr.open_dataset(f'{mdpath}/V/2001-2005.V.50000.{season}.nc')
    v = data['V'].values[0, 0, :, :]
    data = xr.open_dataset(f'{mdpath}/QV/2001-2005.QV.85000.{season}.nc')
    q = data['QV'].values[0, 0, :, :]
    ws = np.sqrt(u ** 2 + v ** 2)
    dt['CTRL11'][season]['v500'] = v
    dt['CTRL11'][season]['u500'] = u
    dt['CTRL11'][season]['ws500'] = ws
    dt['CTRL11'][season]['q850'] = q * 1000
    data = xr.open_dataset(f'{mdpath}/TOT_PREC/2001-2005.TOT_PREC.{season}.nc')
    pr = data['TOT_PREC'].values[0, :, :]
    dt['CTRL11'][season]['pr'] = pr
    data = xr.open_dataset(f'{mdpath}/T_2M/2001-2005.T_2M.{season}.nc')
    tmp = data['T_2M'].values[0, :, :] - 273.15
    dt['CTRL11'][season]['tmp'] = tmp
    dt['CTRL11']['lon'] = rlon
    dt['CTRL11']['lat'] = rlat
    dt['CTRL11']['proj'] = rot_pole_crs
    # COSMO 12km remap
    dt['CTRL11_ERA5'][season] = {}
    data = xr.open_dataset(f'{rmpath}/U/2001-2005.U.85000.{season}.remap.era5.nc')
    u = data['U'].values[0, 0, :, :]
    data = xr.open_dataset(f'{rmpath}/V/2001-2005.V.85000.{season}.remap.era5.nc')
    v = data['V'].values[0, 0, :, :]
    ws = np.sqrt(u ** 2 + v ** 2)
    dt['CTRL11_ERA5'][season]['v850'] = v
    dt['CTRL11_ERA5'][season]['u850'] = u
    dt['CTRL11_ERA5'][season]['ws850'] = ws
    data = xr.open_dataset(f'{rmpath}/U/2001-2005.U.50000.{season}.remap.era5.nc')
    u = data['U'].values[0, 0, :, :]
    data = xr.open_dataset(f'{rmpath}/V/2001-2005.V.50000.{season}.remap.era5.nc')
    v = data['V'].values[0, 0, :, :]
    data = xr.open_dataset(f'{rmpath}/QV/2001-2005.QV.85000.{season}.remap.era5.nc')
    q = data['QV'].values[0, 0, :, :]
    ws = np.sqrt(u ** 2 + v ** 2)
    dt['CTRL11_ERA5'][season]['v500'] = v
    dt['CTRL11_ERA5'][season]['u500'] = u
    dt['CTRL11_ERA5'][season]['ws500'] = ws
    dt['CTRL11_ERA5'][season]['q850'] = q * 1000
    data = xr.open_dataset(f'{rmpath}/TOT_PREC/2001-2005.TOT_PREC.{season}.remap.imerg.nc')
    pr = data['TOT_PREC'].values[0, :, :]
    dt['CTRL11_IMERG'][season] = {}
    dt['CTRL11_IMERG'][season]['pr'] = pr
    data = xr.open_dataset(f'{rmpath}/T_2M/2001-2005.T_2M.{season}.remap.cru.nc')
    tmp = data['T_2M'].values[0, :, :] - 273.15
    dt['CTRL11_CRU'][season] = {}
    dt['CTRL11_CRU'][season]['tmp'] = tmp
    # ERA5
    dt['ERA5'][season] = {}
    data = xr.open_dataset(f'{erapath}/era5.mo.2001-2005.p.{season}.nc')
    u = data['u'].values[0, 2, :, :]
    v = data['v'].values[0, 2, :, :]
    ws = np.sqrt(u ** 2 + v ** 2)
    dt['ERA5'][season]['v850'] = v
    dt['ERA5'][season]['u850'] = u
    dt['ERA5'][season]['ws850'] = ws
    u = data['u'].values[0, 1, :, :]
    v = data['v'].values[0, 1, :, :]
    ws = np.sqrt(u ** 2 + v ** 2)
    q = data['q'].values[0, 2, :, :]
    dt['ERA5'][season]['v500'] = v
    dt['ERA5'][season]['u500'] = u
    dt['ERA5'][season]['ws500'] = ws
    dt['ERA5'][season]['q850'] = q * 1000
    dt['ERA5']['lon'] = data['longitude'].values[...]
    dt['ERA5']['lat'] = data['latitude'].values[...]
    dt['ERA5']['proj'] = ccrs.PlateCarree()
    # IMERG
    dt['IMERG'][season] = {}
    data = xr.open_dataset(f'{imergpath}/IMERG.2001-2005.corr.{season}.nc')
    pr = data['precipitation_corr'].values[0, :, :]
    dt['IMERG'][season]['pr'] = pr
    dt['IMERG']['lon'] = data['lon'].values[...]
    dt['IMERG']['lat'] = data['lat'].values[...]
    dt['IMERG']['proj'] = ccrs.PlateCarree()
    # CRU
    dt['CRU'][season] = {}
    data = xr.open_dataset(f'{crupath}/cru.2001-2005.05.{season}.nc')
    tmp = data['tmp'].values[0, :, :]
    dt['CRU'][season]['tmp'] = tmp
    dt['CRU']['lon'] = data['lon'].values[...]
    dt['CRU']['lat'] = data['lat'].values[...]
    dt['CRU']['proj'] = ccrs.PlateCarree()

# compute difference
for s in range(len(seasons)):
    season = seasons[s]
    dt['DIFF'][season] = {}
    dt['DIFF_IMERG'][season] = {}
    dt['DIFF_CRU'][season] = {}
    for v in range(len(vars)):
        var = vars[v]
        dt['DIFF'][season][var] = dt['CTRL11_ERA5'][season][var] - dt['ERA5'][season][var]
    dt['DIFF'][season]['R'] = \
        ma.corrcoef(ma.masked_invalid(dt['CTRL11_ERA5'][season]['q850'].flatten()),
                    ma.masked_invalid(dt['ERA5'][season]['q850'].flatten()))[0, 1]
    dt['DIFF'][season]['BIAS'] = np.nanmean(dt['CTRL11_ERA5'][season]['q850'] - dt['ERA5'][season]['q850'])
    dt['DIFF_IMERG'][season]['pr'] = dt['CTRL11_IMERG'][season]['pr'] - dt['IMERG'][season]['pr']
    dt['DIFF_CRU'][season]['tmp'] = dt['CTRL11_CRU'][season]['tmp'] - dt['CRU'][season]['tmp']
    dt['DIFF_IMERG'][season]['R'] = \
        ma.corrcoef(ma.masked_invalid(dt['CTRL11_IMERG'][season]['pr'].flatten()),
                    ma.masked_invalid(dt['IMERG'][season]['pr'].flatten()))[0, 1]
    dt['DIFF_IMERG'][season]['BIAS'] = np.nanmean(dt['CTRL11_IMERG'][season]['pr'] - dt['IMERG'][season]['pr'])
    dt['DIFF_CRU'][season]['R'] = \
    ma.corrcoef(ma.masked_invalid(dt['CTRL11_CRU'][season]['tmp'].flatten()), ma.masked_invalid(dt['CRU'][season]['tmp'].flatten()))[0, 1]
    dt['DIFF_CRU'][season]['BIAS'] = np.nanmean(dt['CTRL11_CRU'][season]['tmp'] - dt['CRU'][season]['tmp'])

dt['DIFF']['lon'], dt['DIFF_IMERG']['lon'], dt['DIFF_CRU']['lon'] = dt['ERA5']['lon'], dt['IMERG']['lon'], dt['CRU'][
    'lon']
dt['DIFF']['lat'], dt['DIFF_IMERG']['lat'], dt['DIFF_CRU']['lat'] = dt['ERA5']['lat'], dt['IMERG']['lat'], dt['CRU'][
    'lat']
dt['DIFF']['proj'], dt['DIFF_IMERG']['proj'], dt['DIFF_CRU']['proj'] = dt['ERA5']['proj'], dt['IMERG']['proj'], \
                                                                       dt['CRU']['proj']

###############################################################################
# %% Plot
###############################################################################
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()
rlon_, rlat_ = np.meshgrid(rlon, rlat)
sims = ['CTRL11', 'ERA5', 'DIFF']
fig = plt.figure(figsize=(11, 2.1))
gs1 = gridspec.GridSpec(1, 2, left=0.05, bottom=0.03, right=0.6,
                        top=0.97, hspace=0.1, wspace=0.08,
                        width_ratios=[1, 1])
gs2 = gridspec.GridSpec(1, 1, left=0.66, bottom=0.03, right=0.925,
                        top=0.97, hspace=0.1, wspace=0.08)
ncol = 3  # edit here
nrow = 1

axs, cs, ct, qk, q = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                     np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                     np.empty(shape=(nrow, ncol), dtype='object')

for i in range(nrow):
    for j in range(ncol - 1):
        axs[i, j] = fig.add_subplot(gs1[i, j], projection=rot_pole_crs)
        axs[i, j] = plotcosmo_notick(axs[i, j])
    axs[i, 2] = fig.add_subplot(gs2[i, 0], projection=rot_pole_crs)
    axs[i, 2] = plotcosmo_notick(axs[i, 2])

# plot uqv 850
# levels1 = MaxNLocator(nbins=16).tick_values(0, 16)
# cmap1 = plt.cm.get_cmap('YlGnBu')
# norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
#
# levels2 = MaxNLocator(nbins=21).tick_values(-4, 4)
# cmap2 = drywet(21, cmc.vik_r)
# norm2 = matplotlib.colors.Normalize(vmin=-4, vmax=4)
#
# cmaps = [cmap1, cmap1, cmap2]
# norms = [norm1, norm1, norm2]
# scales = [200, 200, 150]
# steplons = [np.arange(0, 1058, 1)[::40], np.arange(0, 1440, 1)[::30], np.arange(0, 1440, 1)[::30]]
# steplats = [np.arange(0, 610, 1)[::40], np.arange(0, 369, 1)[::12], np.arange(0, 369, 1)[::12]]
#
# for j in range(3):
#     sim = sims[j]
#     cmap = cmaps[j]
#     norm = norms[j]
#     scale = scales[j]
#     steplon = steplons[j]
#     steplat = steplats[j]
#     cs[2, j] = axs[2, j].pcolormesh(dt[sim]['lon'], dt[sim]['lat'], dt[sim]['JJA']['q850'], cmap=cmap, norm=norm,
#                                     shading="auto", transform=dt[sim]['proj'])
#     q[2, j] = axs[2, j].quiver(dt[sim]['lon'][steplon], dt[sim]['lat'][steplat],
#                                dt[sim]['JJA']['u850'][steplat, :][:, steplon],
#                                dt[sim]['JJA']['v850'][steplat, :][:, steplon],
#                                color='black', scale=scale, transform=dt[sim]['proj'])
#
# qk[2, 1] = axs[2, 1].quiverkey(q[2, 1], 0.88, 1.06, 10, r'$10$', labelpos='E', transform=axs[2, 1].transAxes,
#                                fontproperties={'size': 13})
# qk[2, 2] = axs[2, 2].quiverkey(q[2, 2], 0.88, 1.06, 10, r'$10$', labelpos='E', transform=axs[2, 2].transAxes,
#                                fontproperties={'size': 13})
#
# txt = dt['DIFF']['JJA']['R']
# t = axs[2, 2].text(0.99, 0.92, 'R=%0.2f' % txt, fontsize=13, horizontalalignment='right',
#                verticalalignment='center', transform=axs[2, 2].transAxes)
# t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))
# txt = dt['DIFF']['JJA']['BIAS']
# t = axs[2, 2].text(0.99, 0.81, 'BIAS=%0.2f' % txt, fontsize=13, horizontalalignment='right',
#                verticalalignment='center', transform=axs[2, 2].transAxes)
# t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))
#
# cax = fig.add_axes(
#     [axs[2, 1].get_position().x1 + 0.01, axs[2, 1].get_position().y0, 0.015, axs[2, 1].get_position().height])
# cbar = fig.colorbar(cs[2, 1], cax=cax, orientation='vertical', extend='max')
# cbar.ax.tick_params(labelsize=13)
# cax = fig.add_axes(
#     [axs[2, 2].get_position().x1 + 0.01, axs[2, 2].get_position().y0, 0.015, axs[2, 2].get_position().height])
# cbar = fig.colorbar(cs[2, 2], cax=cax, orientation='vertical', extend='both')
# cbar.ax.tick_params(labelsize=13)

# plot precipitation
levels1 = MaxNLocator(nbins=20).tick_values(0, 20)
cmap1 = cmap = cmc.davos_r
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

levels2 = MaxNLocator(nbins=21).tick_values(-15, 15)
cmap2 = drywet(21, cmc.vik_r)
norm2 = matplotlib.colors.Normalize(vmin=-15, vmax=15)

cmaps = [cmap1, cmap1, cmap2]
norms = [norm1, norm1, norm2]
sims = ['CTRL11', 'IMERG', 'DIFF_IMERG']

for j in range(3):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    cs[0, j] = axs[0, j].pcolormesh(dt[sim]['lon'], dt[sim]['lat'], dt[sim]['JJA']['pr'], cmap=cmap, norm=norm,
                                    shading="auto", transform=dt[sim]['proj'])
txt = dt['DIFF_IMERG']['JJA']['R']
t = axs[0, 2].text(0.99, 0.92, 'R=%0.2f' % txt, fontsize=13, horizontalalignment='right',
               verticalalignment='center', transform=axs[0, 2].transAxes)
t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))
txt = dt['DIFF_IMERG']['JJA']['BIAS']
t = axs[0, 2].text(0.99, 0.81, 'BIAS=%0.2f' % txt, fontsize=13, horizontalalignment='right',
               verticalalignment='center', transform=axs[0, 2].transAxes)
t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

cax = fig.add_axes(
    [axs[0, 1].get_position().x1 + 0.01, axs[0, 1].get_position().y0, 0.015, axs[0, 1].get_position().height])
cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical', extend='max', ticks=np.linspace(0, 20, 5, endpoint=True))
cbar.ax.tick_params(labelsize=13)
cax = fig.add_axes(
    [axs[0, 2].get_position().x1 + 0.01, axs[0, 2].get_position().y0, 0.015, axs[0, 2].get_position().height])
cbar = fig.colorbar(cs[0, 2], cax=cax, orientation='vertical', extend='both', ticks=np.linspace(-15, 15, 7, endpoint=True))
cbar.ax.tick_params(labelsize=13)

# plot tmp
# levels1 = MaxNLocator(nbins=18).tick_values(0, 36)
# cmap1 = cmc.roma_r
# norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
#
# levels2 = MaxNLocator(nbins=27).tick_values(-9, 9)
# cmap2 = cmap = custom_div_cmap(27, cmc.vik)
# norm2 = matplotlib.colors.Normalize(vmin=-9, vmax=9)
#
# cmaps = [cmap1, cmap1, cmap2]
# norms = [norm1, norm1, norm2]
# sims = ['CTRL11', 'CRU', 'DIFF_CRU']
#
# for j in range(3):
#     sim = sims[j]
#     cmap = cmaps[j]
#     norm = norms[j]
#     cs[1, j] = axs[1, j].pcolormesh(dt[sim]['lon'], dt[sim]['lat'], dt[sim]['JJA']['tmp'], cmap=cmap, norm=norm,
#                                     shading="auto", transform=dt[sim]['proj'])
# txt = dt['DIFF_CRU']['JJA']['R']
# t = axs[1, 2].text(0.99, 0.92, 'R=%0.2f' % txt, fontsize=13, horizontalalignment='right',
#                verticalalignment='center', transform=axs[1, 2].transAxes)
# t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))
# txt = dt['DIFF_CRU']['JJA']['BIAS']
# t = axs[1, 2].text(0.99, 0.81, 'BIAS=%0.2f' % txt, fontsize=13, horizontalalignment='right',
#                verticalalignment='center', transform=axs[1, 2].transAxes)
# t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))
#
# cax = fig.add_axes(
#     [axs[1, 1].get_position().x1 + 0.01, axs[1, 1].get_position().y0, 0.015, axs[1, 1].get_position().height])
# cbar = fig.colorbar(cs[1, 1], cax=cax, orientation='vertical', extend='both', ticks=np.linspace(0, 36, 7, endpoint=True))
# cbar.ax.tick_params(labelsize=13)
# cax = fig.add_axes(
#     [axs[1, 2].get_position().x1 + 0.01, axs[1, 2].get_position().y0, 0.015, axs[1, 2].get_position().height])
# cbar = fig.colorbar(cs[1, 2], cax=cax, orientation='vertical', extend='both', ticks=np.linspace(-9, 9, 7, endpoint=True))
# cbar.ax.tick_params(labelsize=13)
# ---
# for i in range(nrow):
#     for j in range(ncol):
#         label = lb[i][j]
#         t = axs[i, j].text(0.01, 0.985, f'({label})', ha='left', va='top',
#                            transform=axs[i, j].transAxes, fontsize=14)
#         t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

for i in range(nrow):
    for j in range(ncol):
        title = labels[i][j]
        axs[i, j].set_title(f'{title}', pad=5, fontsize=14, loc='left')

# axs[0, 1].text(0.98, 1.01, 'mm d$^{-1}$', ha='left', va='bottom', transform=axs[0, 1].transAxes, fontsize=13)
axs[0, 2].text(1, 1.01, 'mm d$^{-1}$', ha='left', va='bottom', transform=axs[0, 2].transAxes, fontsize=13)

for i in range(nrow):
    axs[i, 0].text(-0.008, 0.95, '50°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.77, '40°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.59, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.41, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.23, '10°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.05, '0°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)

for j in range(ncol):
    axs[0, j].text(0.12, -0.02, '80°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=13)
    axs[0, j].text(0.32, -0.02, '100°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=13)
    axs[0, j].text(0.52, -0.02, '120°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=13)
    axs[0, j].text(0.72, -0.02, '140°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=13)
    axs[0, j].text(0.92, -0.02, '160°E', ha='center', va='top', transform=axs[0, j].transAxes, fontsize=13)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/results/"
fig.savefig(plotpath + 'vali1.png', dpi=500, transparent=True)
plt.close(fig)
