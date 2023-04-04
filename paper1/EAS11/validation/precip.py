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
from pyproj import Transformer
import scipy.ndimage as ndimage
import matplotlib
import numpy.ma as ma

font = {'size': 14}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
# %%
sims = ['LSM', 'ERA5', 'IMERG', 'CRU']
seasons = ['DJF', 'MAM', 'JJA', 'SON']
mdpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn"
erapath = "/project/pr133/rxiang/data/era5/pr/remap"
imergpath = "/project/pr133/rxiang/data/obs/pr/IMERG/remap/"
crupath = "/project/pr133/rxiang/data/obs/pr/cru/remap/"

precip = {}
labels = ['CTRL11', 'CTRL11 - ERA5', 'CTRL11 - IMERG', 'CTRL11 - CRU']

precip['LSM'], precip['ERA5'], precip['IMERG'], precip['CRU'] = {}, {}, {}, {}
for s in range(len(seasons)):
    season = seasons[s]
    # COSMO 12 km
    precip['LSM'][season] = {}
    data = xr.open_dataset(f'{mdpath}/TOT_PREC/2001-2005.TOT_PREC.{season}.nc')
    pr = data['TOT_PREC'].values[0, :, :]
    precip['LSM'][season]['precip'] = pr
    # ERA5
    precip['ERA5'][season] = {}
    data = xr.open_dataset(f'{erapath}/era5.mo.2001-2005.{season}.remap.nc')
    pr = data['tp'].values[0, :, :] * 1000
    precip['ERA5'][season]['R'] = np.corrcoef(precip['LSM'][season]['precip'].flatten(), pr.flatten())[0, 1]
    precip['ERA5'][season]['BIAS'] = np.nanmean(precip['LSM'][season]['precip'] - pr)
    precip['ERA5'][season]['precip'] = precip['LSM'][season]['precip'] - pr
    # IMERG
    precip['IMERG'][season] = {}
    data = xr.open_dataset(f'{imergpath}/IMERG.2001-2005.{season}.remap.nc')
    pr = data['precipitation'].values[0, :, :]
    precip['IMERG'][season]['R'] = ma.corrcoef(ma.masked_invalid(precip['LSM'][season]['precip'].flatten()), ma.masked_invalid(pr.flatten()))[0, 1]
    precip['IMERG'][season]['BIAS'] = np.nanmean(precip['LSM'][season]['precip'] - pr)
    precip['IMERG'][season]['precip'] = precip['LSM'][season]['precip'] - pr
    # CRU
    precip['CRU'][season] = {}
    data = xr.open_dataset(f'{crupath}/cru.2001-2005.05.{season}.remap.nc')
    pr = data['pre'].values[0, :, :]
    precip['CRU'][season]['R'] = ma.corrcoef(ma.masked_invalid(precip['LSM'][season]['precip'].flatten()), ma.masked_invalid(pr.flatten()))[0, 1]
    precip['CRU'][season]['BIAS'] = np.nanmean(precip['LSM'][season]['precip'] - pr)
    precip['CRU'][season]['precip'] = precip['LSM'][season]['precip'] - pr

# plot
# %%
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

ar = 1.0  # initial aspect ratio for first trial
wi = 12.7  # height in inches #15
hi = 8  # width in inches #10
ncol = 4  # edit here
nrow = 4
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), np.empty(
    shape=(nrow, ncol), dtype='object')

levels1 = MaxNLocator(nbins=20).tick_values(0, 20)
cmap1 = cmap = cmc.davos_r
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

levels2 = MaxNLocator(nbins=21).tick_values(-15, 15)
cmap2 = drywet(21, cmc.vik_r)
norm2 = matplotlib.colors.Normalize(vmin=-15, vmax=15)

cmaps = [cmap1, cmap2, cmap2, cmap2]
norms = [norm1, norm2, norm2, norm2]

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.06, 0.11, 0.99, 0.97
gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.025, hspace=0.025)

for i in range(len(seasons)):
    season = seasons[i]
    for j in range(ncol):
        sim = sims[j]
        cmap = cmaps[j]
        norm = norms[j]
        axs[i, j] = fig.add_subplot(gs[i, j], projection=rot_pole_crs)
        axs[i, j] = plotcosmo_notick(axs[i, j])
        cs[i, j] = axs[i, j].pcolormesh(rlon, rlat, precip[sim][season]['precip'], cmap=cmap, norm=norm, shading="auto")

for i in range(len(seasons)):
    season = seasons[i]
    for j in range(3):
        sim = sims[j+1]
        txt = precip[sim][season]['R']
        axs[i, j+1].text(0.98, 0.92, 'R=%0.2f'%txt, fontsize=13, horizontalalignment='right',
            verticalalignment='center', transform=axs[i, j+1].transAxes)
        txt = precip[sim][season]['BIAS']
        axs[i, j+1].text(0.98, 0.81, 'BIAS=%0.2f' % txt, fontsize=13, horizontalalignment='right',
                       verticalalignment='center', transform=axs[i, j+1].transAxes)
        

x = axs[3, 0].get_position().x0
dis = axs[3, 0].get_position().width
cax = fig.add_axes(
    [x, axs[3, 0].get_position().y0 - 0.045, dis, 0.015])
cbar = fig.colorbar(cs[3, 0], cax=cax, orientation='horizontal', extend='max',
                    ticks=np.linspace(0, 20, 5, endpoint=True))
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_xlabel('mm day$^{-1}$', fontsize=14, labelpad=-0.02)

x = (axs[3, 1].get_position().x0 + axs[3, 1].get_position().x1)/2
dis = (axs[3, 3].get_position().x0 + axs[3, 3].get_position().x1)/2 - (axs[3, 1].get_position().x0 + axs[3, 1].get_position().x1)/2
cax = fig.add_axes(
    [x, axs[3, 2].get_position().y0 - 0.045, dis, 0.015])
cbar = fig.colorbar(cs[3, 2], cax=cax, orientation='horizontal', extend='both',
                    ticks=np.linspace(-15, 15, 7, endpoint=True))
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_xlabel('mm day$^{-1}$', fontsize=14, labelpad=-0.02)

for j in range(ncol):
    label = labels[j]
    axs[0, j].set_title(f'{label}', pad=6, fontsize=14, loc='center')

for i in range(nrow):
    season = seasons[i]
    axs[i, 0].text(-0.17, 0.5, f'{season}', ha='right', va='center', rotation='vertical',
                   transform=axs[i, 0].transAxes, fontsize=14)

for i in range(nrow):
    axs[i, 0].text(-0.008, 0.95, '50°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.77, '40°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.59, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.41, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.23, '10°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.05, '0°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)

for j in range(ncol):
    axs[3, j].text(0.12, -0.02, '80°E', ha='center', va='top', transform=axs[3, j].transAxes, fontsize=13)
    axs[3, j].text(0.32, -0.02, '100°E', ha='center', va='top', transform=axs[3, j].transAxes, fontsize=13)
    axs[3, j].text(0.52, -0.02, '120°E', ha='center', va='top', transform=axs[3, j].transAxes, fontsize=13)
    axs[3, j].text(0.72, -0.02, '140°E', ha='center', va='top', transform=axs[3, j].transAxes, fontsize=13)
    axs[3, j].text(0.92, -0.02, '160°E', ha='center', va='top', transform=axs[3, j].transAxes, fontsize=13)

fig.show()
plotpath = "/project/pr133/rxiang/figure/paper1/validation/LSM/"
fig.savefig(plotpath + 'precip.png', dpi=500, transparent=True)
plt.close(fig)






