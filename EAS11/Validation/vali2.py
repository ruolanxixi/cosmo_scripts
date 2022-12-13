# -------------------------------------------------------------------------------
# modules
#
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo, pole, colorbar
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

font = {'size': 13}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
sims = ['LSM', 'ERA5', 'DIFF']
seasons = ['DJF', 'MAM', 'JJA', 'SON']
mdpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn"
rmpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/remap"
erapath = "/project/pr133/rxiang/data/era5/ot/szn"
wind = {}
labels = ['LSM', 'ERA5', 'LSM - ERA5']
vars = ['v850', 'u850', 'ws850', 'u500', 'v500', 'ws500', 'q850']
wind['LSM'], wind['ERA5'], wind['LSM_ERA5'], wind['DIFF'] = {}, {}, {}, {}
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()
lb_rows = ['a', 'b']

for s in range(len(seasons)):
    season = seasons[s]
    # COSMO 12 km
    wind['LSM'][season] = {}
    data = xr.open_dataset(f'{mdpath}/U/2001-2005.U.85000.{season}.nc')
    u = data['U'].values[0, 0, :, :]
    data = xr.open_dataset(f'{mdpath}/V/2001-2005.V.85000.{season}.nc')
    v = data['V'].values[0, 0, :, :]
    ws = np.sqrt(u ** 2 + v ** 2)
    wind['LSM'][season]['v850'] = v
    wind['LSM'][season]['u850'] = u
    wind['LSM'][season]['ws850'] = ws
    data = xr.open_dataset(f'{mdpath}/U/2001-2005.U.50000.{season}.nc')
    u = data['U'].values[0, 0, :, :]
    data = xr.open_dataset(f'{mdpath}/V/2001-2005.V.50000.{season}.nc')
    v = data['V'].values[0, 0, :, :]
    data = xr.open_dataset(f'{mdpath}/QV/2001-2005.QV.85000.{season}.nc')
    q = data['QV'].values[0, 0, :, :]
    ws = np.sqrt(u ** 2 + v ** 2)
    wind['LSM'][season]['v500'] = v
    wind['LSM'][season]['u500'] = u
    wind['LSM'][season]['ws500'] = ws
    wind['LSM'][season]['q850'] = q * 1000
    wind['LSM']['lon'] = rlon
    wind['LSM']['lat'] = rlat
    wind['LSM']['proj'] = rot_pole_crs
    # COSMO 12km remap
    wind['LSM_ERA5'][season] = {}
    data = xr.open_dataset(f'{rmpath}/U/2001-2005.U.85000.{season}.remap.era5.nc')
    u = data['U'].values[0, 0, :, :]
    data = xr.open_dataset(f'{rmpath}/V/2001-2005.V.85000.{season}.remap.era5.nc')
    v = data['V'].values[0, 0, :, :]
    ws = np.sqrt(u ** 2 + v ** 2)
    wind['LSM_ERA5'][season]['v850'] = v
    wind['LSM_ERA5'][season]['u850'] = u
    wind['LSM_ERA5'][season]['ws850'] = ws
    data = xr.open_dataset(f'{rmpath}/U/2001-2005.U.50000.{season}.remap.era5.nc')
    u = data['U'].values[0, 0, :, :]
    data = xr.open_dataset(f'{rmpath}/V/2001-2005.V.50000.{season}.remap.era5.nc')
    v = data['V'].values[0, 0, :, :]
    data = xr.open_dataset(f'{rmpath}/QV/2001-2005.QV.85000.{season}.remap.era5.nc')
    q = data['QV'].values[0, 0, :, :]
    ws = np.sqrt(u ** 2 + v ** 2)
    wind['LSM_ERA5'][season]['v500'] = v
    wind['LSM_ERA5'][season]['u500'] = u
    wind['LSM_ERA5'][season]['ws500'] = ws
    wind['LSM_ERA5'][season]['q850'] = q * 1000
    # ERA5
    wind['ERA5'][season] = {}
    data = xr.open_dataset(f'{erapath}/era5.mo.2001-2005.p.{season}.nc')
    u = data['u'].values[0, 2, :, :]
    v = data['v'].values[0, 2, :, :]
    ws = np.sqrt(u ** 2 + v ** 2)
    wind['ERA5'][season]['v850'] = v
    wind['ERA5'][season]['u850'] = u
    wind['ERA5'][season]['ws850'] = ws
    u = data['u'].values[0, 1, :, :]
    v = data['v'].values[0, 1, :, :]
    ws = np.sqrt(u ** 2 + v ** 2)
    q = data['q'].values[0, 2, :, :]
    wind['ERA5'][season]['v500'] = v
    wind['ERA5'][season]['u500'] = u
    wind['ERA5'][season]['ws500'] = ws
    wind['ERA5'][season]['q850'] = q * 1000
    wind['ERA5']['lon'] = data['longitude'].values[...]
    wind['ERA5']['lat'] = data['latitude'].values[...]
    wind['ERA5']['proj'] = ccrs.PlateCarree()

# compute difference
for s in range(len(seasons)):
    season = seasons[s]
    wind['DIFF'][season] = {}
    for v in range(len(vars)):
        var = vars[v]
        wind['DIFF'][season][var] = wind['LSM_ERA5'][season][var] - wind['ERA5'][season][var]
wind['DIFF']['lon'] = wind['ERA5']['lon']
wind['DIFF']['lat'] = wind['ERA5']['lat']
wind['DIFF']['proj'] = wind['ERA5']['proj']

###############################################################################
# %% Plot
###############################################################################
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()
rlon_, rlat_ = np.meshgrid(rlon, rlat)
sims = ['LSM', 'ERA5', 'DIFF']
fig = plt.figure(figsize=(12.5, 4.3))
gs1 = gridspec.GridSpec(2, 2, left=0.05, bottom=0.03, right=0.575,
                        top=0.95, hspace=0.15, wspace=0.18,
                        width_ratios=[1, 1], height_ratios=[1, 1])
gs2 = gridspec.GridSpec(2, 1, left=0.682, bottom=0.03, right=0.925,
                        top=0.95, hspace=0.15, wspace=0.18)
ncol = 3  # edit here
nrow = 2

axs, cs, ct, qk, q = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                     np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                     np.empty(shape=(nrow, ncol), dtype='object')

for i in range(nrow):
    label = lb_rows[i]
    for j in range(ncol - 1):
        axs[i, j] = fig.add_subplot(gs1[i, j], projection=rot_pole_crs)
        axs[i, j] = plotcosmo(axs[i, j])
    axs[i, 2] = fig.add_subplot(gs2[i, 0], projection=rot_pole_crs)
    axs[i, 2] = plotcosmo(axs[i, 2])
    axs[i, 0].text(-0.12, 1.03, f'({label})', ha='right', va='bottom', transform=axs[i, 0].transAxes, fontsize=14)

# plot wind 500
levels1 = MaxNLocator(nbins=20).tick_values(0, 20)
cmap1 = windmap(20, cmc.batlowW_r)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

levels2 = MaxNLocator(nbins=21).tick_values(-10, 10)
cmap2 = custom_div_cmap(21, cmc.broc_r)
norm2 = matplotlib.colors.Normalize(vmin=-10, vmax=10)

cmaps = [cmap1, cmap1, cmap2]
norms = [norm1, norm1, norm2]
scales = [200, 200, 150]
steplons = [np.arange(0, 1058, 1)[::40], np.arange(0, 1440, 1)[::30], np.arange(0, 1440, 1)[::30]]
steplats = [np.arange(0, 610, 1)[::40], np.arange(0, 369, 1)[::12], np.arange(0, 369, 1)[::12]]

for j in range(3):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    steplon = steplons[j]
    steplat = steplats[j]
    scale = scales[j]
    cs[0, j] = axs[0, j].pcolormesh(wind[sim]['lon'], wind[sim]['lat'], wind[sim]['JJA']['ws500'], cmap=cmap, norm=norm,
                                    shading="auto", transform=wind[sim]['proj'])
    q[0, j] = axs[0, j].quiver(wind[sim]['lon'][steplon], wind[sim]['lat'][steplat],
                               wind[sim]['JJA']['u500'][steplat, :][:, steplon],
                               wind[sim]['JJA']['v500'][steplat, :][:, steplon], color='black', scale=scale,
                               transform=wind[sim]['proj'])

qk[0, 1] = axs[0, 1].quiverkey(q[0, 1], 0.88, 1.06, 10, r'$10$', labelpos='E', transform=axs[0, 1].transAxes,
                               fontproperties={'size': 13})
qk[0, 2] = axs[0, 2].quiverkey(q[0, 2], 0.88, 1.06, 10, r'$10$', labelpos='E', transform=axs[0, 2].transAxes,
                               fontproperties={'size': 13})

cax = fig.add_axes(
    [axs[0, 1].get_position().x1 + 0.01, axs[0, 1].get_position().y0, 0.015, axs[0, 1].get_position().height])
cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical', extend='max')
cbar.ax.tick_params(labelsize=13)
cax = fig.add_axes(
    [axs[0, 2].get_position().x1 + 0.01, axs[0, 2].get_position().y0, 0.015, axs[0, 2].get_position().height])
cbar = fig.colorbar(cs[0, 2], cax=cax, orientation='vertical', extend='both')
cbar.ax.tick_params(labelsize=13)

for j in range(ncol):
    label = labels[j]
    axs[0, j].set_title(f'{label}', fontweight='bold', pad=7, fontsize=14, loc='center')

# plot uqv 850
levels1 = MaxNLocator(nbins=16).tick_values(0, 16)
cmap1 = plt.cm.get_cmap('YlGnBu')
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

levels2 = MaxNLocator(nbins=21).tick_values(-4, 4)
cmap2 = drywet(21, cmc.vik_r)
norm2 = matplotlib.colors.Normalize(vmin=-4, vmax=4)

cmaps = [cmap1, cmap1, cmap2]
norms = [norm1, norm1, norm2]
scales = [200, 200, 150]

for j in range(3):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    scale = scales[j]
    steplon = steplons[j]
    steplat = steplats[j]
    cs[1, j] = axs[1, j].pcolormesh(wind[sim]['lon'], wind[sim]['lat'], wind[sim]['JJA']['q850'], cmap=cmap, norm=norm,
                                    shading="auto", transform=wind[sim]['proj'])
    q[1, j] = axs[1, j].quiver(wind[sim]['lon'][steplon], wind[sim]['lat'][steplat],
                               wind[sim]['JJA']['u850'][steplat, :][:, steplon],
                               wind[sim]['JJA']['v850'][steplat, :][:, steplon],
                               color='black', scale=scale, transform=wind[sim]['proj'])

qk[1, 1] = axs[1, 1].quiverkey(q[1, 1], 0.88, 1.06, 10, r'$10$', labelpos='E', transform=axs[1, 1].transAxes,
                               fontproperties={'size': 13})
qk[1, 2] = axs[1, 2].quiverkey(q[1, 2], 0.88, 1.06, 10, r'$10$', labelpos='E', transform=axs[1, 2].transAxes,
                               fontproperties={'size': 13})

cax = fig.add_axes(
    [axs[1, 1].get_position().x1 + 0.01, axs[1, 1].get_position().y0, 0.015, axs[1, 1].get_position().height])
cbar = fig.colorbar(cs[1, 1], cax=cax, orientation='vertical', extend='max')
cbar.ax.tick_params(labelsize=13)
cax = fig.add_axes(
    [axs[1, 2].get_position().x1 + 0.01, axs[1, 2].get_position().y0, 0.015, axs[1, 2].get_position().height])
cbar = fig.colorbar(cs[1, 2], cax=cax, orientation='vertical', extend='both')
cbar.ax.tick_params(labelsize=13)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/validation/LSM/"
fig.savefig(plotpath + 'vali1.png', dpi=500)
plt.close(fig)
