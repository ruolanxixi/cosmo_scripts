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
from pyproj import Transformer
import scipy.ndimage as ndimage
import matplotlib
import numpy.ma as ma

font = {'size': 14}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
# %%
sims = ['LSM', 'ERA5', 'APHRODITE', 'CRU']
seasons = ['DJF', 'MAM', 'JJA', 'SON']
mdpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn"
erapath = "/project/pr133/rxiang/data/era5/ot/remap"
aphroditepath = "/project/pr133/rxiang/data/obs/tmp/APHRO/remap/"
crupath = "/project/pr133/rxiang/data/obs/tmp/cru/remap/"

t2m = {}
labels = ['LSM', 'LSM - ERA5', 'LSM - APHRODITE', 'LSM - CRU']

t2m['LSM'], t2m['ERA5'], t2m['APHRODITE'], t2m['CRU'] = {}, {}, {}, {}
for s in range(len(seasons)):
    season = seasons[s]
    # COSMO 12 km
    t2m['LSM'][season] = {}
    data = xr.open_dataset(f'{mdpath}/T_2M/2001-2005.T_2M.{season}.nc')
    tmp = data['T_2M'].values[0, :, :] - 273.15
    t2m['LSM'][season]['t2m'] = tmp
    # ERA5
    t2m['ERA5'][season] = {}
    data = xr.open_dataset(f'{erapath}/era5.mo.2001-2005.{season}.remap.nc')
    tmp = data['t2m'].values[0, :, :] - 273.15
    t2m['ERA5'][season]['R'] = np.corrcoef(t2m['LSM'][season]['t2m'].flatten(), tmp.flatten())[0, 1]
    t2m['ERA5'][season]['BIAS'] = np.nanmean(t2m['LSM'][season]['t2m'] - tmp)
    t2m['ERA5'][season]['t2m'] = t2m['LSM'][season]['t2m'] - tmp
    # APHRODITE
    t2m['APHRODITE'][season] = {}
    data = xr.open_dataset(f'{aphroditepath}/APHRO.2001-2005.025.{season}.remap.nc')
    tmp = data['tave'].values[0, :, :]
    t2m['APHRODITE'][season]['R'] = \
    ma.corrcoef(ma.masked_invalid(t2m['LSM'][season]['t2m'].flatten()), ma.masked_invalid(tmp.flatten()))[0, 1]
    t2m['APHRODITE'][season]['BIAS'] = np.nanmean(t2m['LSM'][season]['t2m'] - tmp)
    t2m['APHRODITE'][season]['t2m'] = t2m['LSM'][season]['t2m'] - tmp
    # CRU
    t2m['CRU'][season] = {}
    data = xr.open_dataset(f'{crupath}/cru.2001-2005.05.{season}.remap.nc')
    tmp = data['tmp'].values[0, :, :]
    t2m['CRU'][season]['R'] = \
    ma.corrcoef(ma.masked_invalid(t2m['LSM'][season]['t2m'].flatten()), ma.masked_invalid(tmp.flatten()))[0, 1]
    t2m['CRU'][season]['BIAS'] = np.nanmean(t2m['LSM'][season]['t2m'] - tmp)
    t2m['CRU'][season]['t2m'] = t2m['LSM'][season]['t2m'] - tmp

# plot
# %%
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

ar = 1.0  # initial aspect ratio for first trial
wi = 17  # height in inches #15
hi = 10  # width in inches #10
ncol = 4  # edit here
nrow = 4
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), np.empty(
    shape=(nrow, ncol), dtype='object')

levels1 = MaxNLocator(nbins=35).tick_values(-35, 35)
cmap1 = cmc.roma_r
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

levels2 = MaxNLocator(nbins=27).tick_values(-9, 9)
cmap2 = cmap = custom_div_cmap(27, cmc.vik)
norm2 = matplotlib.colors.Normalize(vmin=-9, vmax=9)

cmaps = [cmap1, cmap2, cmap2, cmap2]
norms = [norm1, norm2, norm2, norm2]

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.05, 0.1, 0.99, 0.97
gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.025, hspace=0.15)

for i in range(len(seasons)):
    season = seasons[i]
    for j in range(ncol):
        sim = sims[j]
        cmap = cmaps[j]
        norm = norms[j]
        axs[i, j] = fig.add_subplot(gs[i, j], projection=rot_pole_crs)
        axs[i, j] = plotcosmo(axs[i, j])
        cs[i, j] = axs[i, j].pcolormesh(rlon, rlat, t2m[sim][season]['t2m'], cmap=cmap, norm=norm, shading="auto")

for i in range(len(seasons)):
    season = seasons[i]
    for j in range(3):
        sim = sims[j+1]
        txt = t2m[sim][season]['R']
        axs[i, j+1].text(0.98, 0.92, 'R=%0.2f'%txt, fontsize=13, horizontalalignment='right',
            verticalalignment='center', transform=axs[i, j+1].transAxes)
        txt = t2m[sim][season]['BIAS']
        axs[i, j+1].text(0.98, 0.81, 'BIAS=%0.2f' % txt, fontsize=13, horizontalalignment='right',
                       verticalalignment='center', transform=axs[i, j+1].transAxes)

x = axs[3, 0].get_position().x0
dis = axs[3, 0].get_position().width
cax = fig.add_axes(
    [x, axs[3, 0].get_position().y0 - 0.04, dis, 0.015])
cbar = fig.colorbar(cs[3, 0], cax=cax, orientation='horizontal', extend='both',
                    ticks=np.linspace(-30, 30, 5, endpoint=True))
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_xlabel('$^{o}$C', fontsize=14)

x = (axs[3, 1].get_position().x0 + axs[3, 1].get_position().x1)/2
dis = (axs[3, 3].get_position().x0 + axs[3, 3].get_position().x1)/2 - (axs[3, 1].get_position().x0 + axs[3, 1].get_position().x1)/2
cax = fig.add_axes(
    [x, axs[3, 2].get_position().y0 - 0.04, dis, 0.015])
cbar = fig.colorbar(cs[3, 2], cax=cax, orientation='horizontal', extend='both',
                    ticks=np.linspace(-9, 9, 7, endpoint=True))
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_xlabel('$^{o}$C', fontsize=14)

for j in range(ncol):
    label = labels[j]
    axs[0, j].set_title(f'{label}', fontweight='bold', pad=6, fontsize=14, loc='left')

for i in range(nrow):
    season = seasons[i]
    axs[i, 0].text(-0.17, 0.5, f'{season}', ha='right', va='center', rotation='vertical',
                   transform=axs[i, 0].transAxes, fontsize=14, fontweight='bold')

fig.show()
plotpath = "/project/pr133/rxiang/figure/paper1/validation/LSM/"
fig.savefig(plotpath + 't2m.png', dpi=500)
plt.close(fig)






