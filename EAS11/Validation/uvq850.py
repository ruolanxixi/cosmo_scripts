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

font = {'size': 14}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
# %%
sims = ['LSM', 'ERA5', 'DIFF']
seasons = ['DJF', 'MAM', 'JJA', 'SON']
mdpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn"
erapath = "/project/pr133/rxiang/data/era5/ot/remap"
wind = {}
labels = ['LSM', 'ERA5', 'LSM - ERA5']

wind['LSM'], wind['ERA5'], wind['DIFF'] = {}, {}, {}
for s in range(len(seasons)):
    season = seasons[s]
    # COSMO 12 km
    wind['LSM'][season] = {}
    data = xr.open_dataset(f'{mdpath}/U/2001-2005.U.85000.{season}.nc')
    u = data['U'].values[0, 0, :, :]
    data = xr.open_dataset(f'{mdpath}/V/2001-2005.V.85000.{season}.nc')
    v = data['V'].values[0, 0, :, :]
    data = xr.open_dataset(f'{mdpath}/QV/2001-2005.QV.85000.{season}.nc')
    q = data['QV'].values[0, 0, :, :]
    ws = np.sqrt(u ** 2 + v ** 2)
    wind['LSM'][season]['v'] = v
    wind['LSM'][season]['u'] = u
    wind['LSM'][season]['ws'] = ws
    wind['LSM'][season]['q'] = q
    # ERA5
    wind['ERA5'][season] = {}
    data = xr.open_dataset(f'{erapath}/era5.mo.2001-2005.p.{season}.remap.nc')
    u = data['u'].values[0, 2, :, :]
    v = data['v'].values[0, 2, :, :]
    ws = np.sqrt(u ** 2 + v ** 2)
    q = data['q'].values[0, 2, :, :]
    wind['ERA5'][season]['v'] = v
    wind['ERA5'][season]['u'] = u
    wind['ERA5'][season]['ws'] = ws
    wind['ERA5'][season]['q'] = q
    # DIFF
    wind['DIFF'][season] = {}
    wind['DIFF'][season]['v'] = wind['LSM'][season]['v'] - wind['ERA5'][season]['v']
    wind['DIFF'][season]['u'] = wind['LSM'][season]['u'] - wind['ERA5'][season]['u']
    wind['DIFF'][season]['ws'] = wind['LSM'][season]['ws'] - wind['ERA5'][season]['ws']
    wind['DIFF'][season]['q'] = wind['LSM'][season]['q'] - wind['ERA5'][season]['q']

# plot
# %%
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

ar = 1.0  # initial aspect ratio for first trial
wi = 13  # height in inches #15
hi = 10  # width in inches #10
ncol = 3  # edit here
nrow = 4
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), np.empty(
    shape=(nrow, ncol), dtype='object')

levels1 = MaxNLocator(nbins=16).tick_values(0, 0.016)
cmap1 = plt.cm.get_cmap('YlGnBu')
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

levels2 = MaxNLocator(nbins=21).tick_values(-0.004, 0.004)
cmap2 = custom_div_cmap(21, cmc.broc_r)
norm2 = matplotlib.colors.Normalize(vmin=-0.004, vmax=0.004)

cmaps = [cmap1, cmap1, cmap2]
norms = [norm1, norm1, norm2]
scales = [200, 200, 150]

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.05, 0.1, 0.99, 0.97
gs = gridspec.GridSpec(nrows=4, ncols=3, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.025, hspace=0.15)

for i in range(len(seasons)):
    season = seasons[i]
    for j in range(3):
        sim = sims[j]
        cmap = cmaps[j]
        norm = norms[j]
        scale = scales[j]
        axs[i, j] = fig.add_subplot(gs[i, j], projection=rot_pole_crs)
        axs[i, j] = plotcosmo(axs[i, j])
        cs[i, j] = axs[i, j].pcolormesh(rlon, rlat, wind[sim][season]['q'], cmap=cmap, norm=norm, shading="auto")
        q[i, j] = axs[i, j].quiver(rlon[::40], rlat[::40], wind[sim][season]['u'][::40, ::40],
                                   wind[sim][season]['v'][::40, ::40], color='black', scale=scale)

qk[0, 1] = axs[0, 1].quiverkey(q[0, 1], 0.75, 1.06, 10, r'$10\ m\ s^{-1}$', labelpos='E', transform=axs[0, 1].transAxes,
                               fontproperties={'size': 14})
qk[0, 2] = axs[0, 2].quiverkey(q[0, 2], 0.75, 1.06, 10, r'$10\ m\ s^{-1}$', labelpos='E', transform=axs[0, 2].transAxes,
                               fontproperties={'size': 14})

x = (axs[3, 0].get_position().x0 + axs[3, 0].get_position().x1)/2
dis = (axs[3, 1].get_position().x0 + axs[3, 1].get_position().x1)/2 - (axs[3, 0].get_position().x0 + axs[3, 0].get_position().x1)/2
cax = fig.add_axes(
    [x, axs[3, 0].get_position().y0 - 0.04, dis, 0.015])
cbar = fig.colorbar(cs[3, 1], cax=cax, orientation='horizontal', extend='max',
                    ticks=np.linspace(0, 0.016, 5, endpoint=True))
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_xlabel('kg kg$^{-1}$', fontsize=14)

cax = fig.add_axes(
    [axs[3, 2].get_position().x0, axs[3, 2].get_position().y0 - 0.04, axs[3, 2].get_position().width, 0.015])
cbar = fig.colorbar(cs[3, 2], cax=cax, orientation='horizontal', extend='both',
                    ticks=np.linspace(-0.004, 0.004, 5, endpoint=True))
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_xlabel('kg kg$^{-1}$', fontsize=14)

for j in range(3):
    label = labels[j]
    axs[0, j].set_title(f'{label}', fontweight='bold', pad=6, fontsize=14, loc='left')

for i in range(4):
    season = seasons[i]
    axs[i, 0].text(-0.17, 0.5, f'{season}', ha='right', va='center', rotation='vertical',
                   transform=axs[i, 0].transAxes, fontsize=14, fontweight='bold')

fig.show()
plotpath = "/project/pr133/rxiang/figure/paper1/validation/LSM/"
fig.savefig(plotpath + 'uvq850.png', dpi=500)
plt.close(fig)






