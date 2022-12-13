# -------------------------------------------------------------------------------
# modules
#
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo, pole, colorbar, plotcosmo04, pole04
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
import matplotlib.patches as patches

font = {'size': 14}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
# %%
sims = ['CPM', 'LSM', 'IMERG', 'ERA5']
seasons = ['DJF', 'MAM', 'JJA', 'SON']
mdpath04 = "/scratch/snx3000/rxiang/data/cosmo/EAS04_ctrl/freq"
mdpath = "/scratch/snx3000/rxiang/data/cosmo/EAS11_ctrl/freq"
erapath = "/project/pr133/rxiang/data/era5/pr"
imergpath = "/scratch/snx3000/rxiang/IMERG"

precip = {}
labels = ['CPM', 'LSM', 'IMERG', 'ERA5']

[pole_lat04, pole_lon04, lat04, lon04, rlat04, rlon04, rot_pole_crs04] = pole04()
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

precip['CPM'], precip['LSM'], precip['ERA5'], precip['IMERG'] = {}, {}, {}, {}
for s in range(len(seasons)):
    season = seasons[s]
    # COSMO 4.4 km
    precip['CPM'][season] = {}
    data = xr.open_dataset(f'{mdpath04}/2001-2005.TOT_PREC.hr.{season}.freq.nc')
    pr = data['TOT_PREC'].values[0, :, :] * 4
    precip['CPM'][season]['precip'] = pr
    precip['CPM']['lon'] = rlon04
    precip['CPM']['lat'] = rlat04
    precip['CPM']['proj'] = rot_pole_crs04
    # COSMO 12 km
    precip['LSM'][season] = {}
    data = xr.open_dataset(f'{mdpath}/2001-2005.TOT_PREC.hr.{season}.freq.nc')
    pr = data['TOT_PREC'].values[0, :, :] * 4
    precip['LSM'][season]['precip'] = pr
    precip['LSM']['lon'] = rlon
    precip['LSM']['lat'] = rlat
    precip['LSM']['proj'] = rot_pole_crs
    # ERA5
    precip['ERA5'][season] = {}
    data = xr.open_dataset(f'{erapath}/freq/2001-2005.hr.{season}.freq.nc')
    pr = data['tp'].values[0, :, :] * 4
    precip['ERA5'][season]['precip'] = pr
    precip['ERA5']['lon'] = data['longitude'].values[...]
    precip['ERA5']['lat'] = data['latitude'].values[...]
    precip['ERA5']['proj'] = ccrs.PlateCarree()
    # --
    data = xr.open_dataset(f'{erapath}/remap/2001-2005.hr.{season}.freq.remap.04.nc')
    pr = data['tp'].values[0, :, :]* 4
    precip['CPM'][season]['R'], precip['CPM'][season]['BIAS'] = {}, {}
    precip['LSM'][season]['R'], precip['LSM'][season]['BIAS'] = {}, {}
    precip['CPM'][season]['R']['ERA5'] = \
        ma.corrcoef(ma.masked_invalid(precip['CPM'][season]['precip'][87:-10, 160:-10].flatten()), ma.masked_invalid(pr[87:-10, 160:-10].flatten()))[0, 1]
    precip['CPM'][season]['BIAS']['ERA5'] = np.nanmean(
        precip['CPM'][season]['precip'][87:-10, 160:-10] - pr[87:-10, 160:-10])
    data = xr.open_dataset(f'{erapath}/remap/2001-2005.hr.{season}.freq.remap.nc')
    pr = data['tp'].values[0, :, :]* 4
    precip['LSM'][season]['R']['ERA5'] = \
        ma.corrcoef(ma.masked_invalid(precip['LSM'][season]['precip'][271:471, 167:346].flatten()), ma.masked_invalid(pr[271:471, 167:346].flatten()))[0, 1]
    precip['LSM'][season]['BIAS']['ERA5'] = np.nanmean(precip['LSM'][season]['precip'][271:471, 167:346] - pr[271:471, 167:346]) # 241 112
    # ERA5
    precip['IMERG'][season] = {}
    data = xr.open_dataset(f'{imergpath}/freq/2001-2005.hr.{season}.freq.nc')
    pr = data['precip'].values[0, :, :]* 4
    precip['IMERG'][season]['precip'] = pr
    precip['IMERG']['lon'] = data['lon'].values[...]
    precip['IMERG']['lat'] = data['lat'].values[...]
    precip['IMERG']['proj'] = ccrs.PlateCarree()
    # --
    data = xr.open_dataset(f'{imergpath}/remap/2001-2005.hr.{season}.freq.remap.04.nc')
    pr = data['precip'].values[0, :, :]* 4
    precip['CPM'][season]['R']['IMERG'] = \
        ma.corrcoef(ma.masked_invalid(precip['CPM'][season]['precip'][87:-10, 160:-10].flatten()), ma.masked_invalid(pr[87:-10, 160:-10].flatten()))[0, 1]
    precip['CPM'][season]['BIAS']['IMERG'] = np.nanmean(
        precip['CPM'][season]['precip'][87:-10, 160:-10] - pr[87:-10, 160:-10])
    data = xr.open_dataset(f'{imergpath}/remap/2001-2005.hr.{season}.freq.remap.nc')
    pr = data['precip'].values[0, :, :]* 4
    precip['LSM'][season]['R']['IMERG'] = \
        ma.corrcoef(ma.masked_invalid(precip['LSM'][season]['precip'][271:471, 167:346].flatten()), ma.masked_invalid(pr[271:471, 167:346].flatten()))[0, 1]
    precip['LSM'][season]['BIAS']['IMERG'] = np.nanmean(
        precip['LSM'][season]['precip'][271:471, 167:346] - pr[271:471, 167:346])  # 241 112

    # # CRU
    # precip['CRU'][season] = {}
    # data = xr.open_dataset(f'{erapath}/freq/2001-2005.hr.{season}.freq.nc')
    # pr = data['tp'].values[0, :, :]
    # precip['CRU']['lon'] = data['lon'].values[...]
    # precip['CRU']['lat'] = data['lat'].values[...]
    # precip['CRU'][season]['precip'] = pr
    # precip['CRU']['proj'] = ccrs.PlateCarree()
    # # --
    # data = xr.open_dataset(f'{erapath}/remap/2001-2005.hr.{season}.freq.remap.04.nc')
    # pr = data['tp'].values[0, :, :]
    # precip['CPM'][season]['R']['CRU'] = \
    #     ma.corrcoef(ma.masked_invalid(precip['CPM'][season]['precip'][10:-10, 10:-10].flatten()), ma.masked_invalid(pr[10:-10, 10:-10].flatten()))[0, 1]
    # data = xr.open_dataset(f'{erapath}/remap/2001-2005.hr.{season}.freq.remap.nc')
    # pr = data['tp'].values[0, :, :]
    # precip['LSM'][season]['R']['CRU'] = \
    #     ma.corrcoef(ma.masked_invalid(precip['LSM'][season]['precip'][241:471, 112:346].flatten()), ma.masked_invalid(pr[241:471, 112:346].flatten()))[0, 1]

# lon.flatten()[min(range(len(lon.flatten())), key=lambda i: abs((lon.flatten()[i])**2+(lat.flatten()[i])**2)-((lon04[10][10])**2+(lat04[10][10])**2)))]

# plot
# %%
ar = 1.0  # initial aspect ratio for first trial
wi = 11.5  # height in inches #15
hi = 9.6  # width in inches #10
ncol = 4  # edit here
nrow = 4
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), np.empty(
    shape=(nrow, ncol), dtype='object')

levels = MaxNLocator(nbins=20).tick_values(0, 40)
cmap = cmc.roma_r
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.05, 0.1, 0.998, 0.97
gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.025, hspace=0.15)

for i in range(len(seasons)):
    season = seasons[i]
    for j in range(ncol):
        sim = sims[j]
        axs[i, j] = fig.add_subplot(gs[i, j], projection=rot_pole_crs04)
        axs[i, j] = plotcosmo04(axs[i, j])
        cs[i, j] = axs[i, j].pcolormesh(precip[sim]['lon'], precip[sim]['lat'], precip[sim][season]['precip'], cmap=cmap, norm=norm, shading="auto", transform=precip[sim]['proj'])

# ---
for i in range(len(seasons)):
    season = seasons[i]
    for j in range(2):
        sim = sims[j+2]
        rect = patches.Rectangle((0.26, 0.75), 0.72, 0.24, linewidth=1, edgecolor='none', facecolor='white', alpha=0.5,
                                 transform=axs[i, j + 2].transAxes)
        # Add the patch to the Axes
        axs[i, j + 2].add_patch(rect)
        axs[i, j + 2].text(0.73, 0.95, "BIAS", fontsize=9, horizontalalignment='center',
                           verticalalignment='center', transform=axs[i, j + 2].transAxes)
        axs[i, j + 2].text(0.9, 0.95, "R", fontsize=9, horizontalalignment='center',
                           verticalalignment='center', transform=axs[i, j + 2].transAxes)
        axs[i, j + 2].text(0.46, 0.87, f"CPM - {sim}", fontsize=9, horizontalalignment='center',
                           verticalalignment='center', transform=axs[i, j + 2].transAxes)
        axs[i, j + 2].text(0.46, 0.79, f"LSM - {sim}", fontsize=9, horizontalalignment='center',
                           verticalalignment='center', transform=axs[i, j + 2].transAxes)
        txt = precip['CPM'][season]['BIAS'][sim]
        axs[i, j + 2].text(0.73, 0.87, '%0.2f' % txt, fontsize=9, horizontalalignment='center',
                           verticalalignment='center', transform=axs[i, j + 2].transAxes)
        txt = precip['LSM'][season]['BIAS'][sim]
        axs[i, j + 2].text(0.73, 0.79, '%0.2f' % txt, fontsize=9, horizontalalignment='center',
                           verticalalignment='center', transform=axs[i, j + 2].transAxes)
        txt = precip['CPM'][season]['R'][sim]
        axs[i, j + 2].text(0.9, 0.87, '%0.2f' % txt, fontsize=9, horizontalalignment='center',
                           verticalalignment='center', transform=axs[i, j + 2].transAxes)
        txt = precip['LSM'][season]['R'][sim]
        axs[i, j + 2].text(0.9, 0.79, '%0.2f' % txt, fontsize=9, horizontalalignment='center',
                           verticalalignment='center', transform=axs[i, j + 2].transAxes)

x = axs[3, 1].get_position().x0
dis = axs[3, 2].get_position().x1 - axs[3, 1].get_position().x0
cax = fig.add_axes(
    [x, axs[3, 1].get_position().y0 - 0.04, dis, 0.015])
cbar = fig.colorbar(cs[3, 1], cax=cax, orientation='horizontal', extend='max',
                    ticks=np.linspace(0, 40, 5, endpoint=True))
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_xlabel('%', fontsize=14)

for j in range(ncol):
    label = labels[j]
    axs[0, j].set_title(f'{label}', fontweight='bold', pad=6, fontsize=14, loc='left')

for i in range(nrow):
    season = seasons[i]
    axs[i, 0].text(-0.27, 0.5, f'{season}', ha='right', va='center', rotation='vertical',
                   transform=axs[i, 0].transAxes, fontsize=14, fontweight='bold')

fig.show()
plotpath = "/project/pr133/rxiang/figure/paper1/validation/CPM/"
fig.savefig(plotpath + 'precip_freq_hr.png', dpi=500)
plt.close(fig)






