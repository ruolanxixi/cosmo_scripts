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
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_, wind
from pyproj import Transformer
import scipy.ndimage as ndimage
import matplotlib
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib.colors as colors

font = {'size': 11}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# define function
def calc(dew, p, t):
    t0 = 273.16
    Rdry = 287.0597
    Rvap = 461.5250
    if dew > 273.15:
        a1, a3, a4 = 611.21, 17.502, 32.19
    else:
        a1, a3, a4 = 611.21, 22.587, -0.7
    e = a1*np.exp(a3*(dew-t0)/(dew-a4))
    if t > 273.15:
        a1, a3, a4 = 611.21, 17.502, 32.19
    else:
        a1, a3, a4 = 611.21, 22.587, -0.7
    es = a1*np.exp(a3*(t-t0)/(t-a4))
    q = (Rdry/Rvap)*e/(p-(1-Rdry/Rvap)*e)*1000
    rh = e/es*100
    return q, rh

# read data
# %%
sims = ['PD', 'ERA5']
mdpath = "/scratch/snx3000/rxiang/echam5"
erapath = "/project/pr133/rxiang/data/era5/ot/remap"
tmp = {}
fname = {'01': 'jan.nc', '07': 'jul.nc'}
labels = {'PD': 'ECHAM5', 'ERA5': 'ERA5'}
month = {'01': 'JAN', '07': 'JUL'}

tmp['PD'], tmp['ERA5'] = {}, {}
tmp['PD']['label'] = labels['PD']
tmp['ERA5']['label'] = labels['ERA5']
for mon in ['01', '07']:
    tmp['PD'][mon] = {}
    name = fname[mon]
    data = xr.open_dataset(f'{mdpath}/PD/analysis/temp2/mon/{name}')
    t2m = data['temp2'].values[0, :, :] - 273.16
    dew2 = data['dew2'].values[0, :, :] - 273.16
    tmp['PD'][mon]['t2m'], tmp['PD'][mon]['dew2'] = t2m, dew2

#%%
data = xr.open_dataset(f'{erapath}/era5.mo.1970-1995.mon.remap.echam5.nc')
tmp['ERA5']['01'], tmp['ERA5']['07'] = {}, {}
dew2 = data['d2m'].values[0, :, :] - 273.16
t2m = data['t2m'].values[0, :, :] - 273.16

tmp['ERA5']['01']['t2m'], tmp['ERA5']['01']['dew2'] = t2m, dew2

dew2 = data['d2m'].values[6, :, :] - 273.16
t2m = data['t2m'].values[6, :, :] - 273.16

tmp['ERA5']['07']['t2m'], tmp['ERA5']['07']['dew2'] = t2m, dew2
# %%
lat = xr.open_dataset(f'{mdpath}/LGM/analysis/temp2/mon/{name}')['lat'].values[:]
lon = xr.open_dataset(f'{mdpath}/LGM/analysis/temp2/mon/{name}')['lon'].values[:]
lat_, lon_ = np.meshgrid(lon, lat)

# -------------------------------------------------------------------------------
# plot
# %%
ar = 1.0  # initial aspect ratio for first trial
wi = 8.7  # height in inches #15
hi = 4.3  # width in inches #10
ncol = 2  # edit here
nrow = 2
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), np.empty(
    shape=(nrow, ncol), dtype='object')

cmap1 = cmc.roma_r
levels1 = np.linspace(-40, 40, 20, endpoint=True)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

cmap2 = custom_div_cmap(23, cmc.vik)
levels2 = np.linspace(-10, 10, 10, endpoint=True)
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

for mon in ['01', '07']:
    m = month[mon]
    fig = plt.figure(figsize=(wi, hi))
    left, bottom, right, top = 0.05, 0.01, 0.9, 0.94
    gs = gridspec.GridSpec(nrows=2, ncols=2, left=left, bottom=bottom, right=right, top=top,
                           wspace=0.01, hspace=0.15)

    for i in range(2):
        sim = sims[i]
        label = tmp[sim]['label']
        axs[0, i] = fig.add_subplot(gs[0, i], projection=ccrs.Robinson(central_longitude=180, globe=None))
        axs[0, i].coastlines(zorder=3)
        axs[0, i].stock_img()
        axs[0, i].gridlines()
        cs[0, i] = axs[0, i].pcolormesh(lon, lat, tmp[sim][mon]['dew2'], cmap=cmap1, norm=norm1, shading="auto",
                                        transform=ccrs.PlateCarree())
        axs[0, i].set_title(f'{label}', fontweight='bold', pad=6, fontsize=13, loc='center')

    for i in range(1):
        sim = sims[i + 1]
        axs[1, i + 1] = fig.add_subplot(gs[1, i + 1], projection=ccrs.Robinson(central_longitude=180, globe=None))
        axs[1, i + 1].coastlines(zorder=3)
        axs[1, i + 1].stock_img()
        axs[1, i + 1].gridlines()
        cs[1, i + 1] = axs[1, i + 1].pcolormesh(lon, lat, tmp[sim][mon]['dew2'] - tmp['PD'][mon]['dew2'], cmap=cmap2,
                                                clim=(-10,  10), shading="auto", transform=ccrs.PlateCarree())

    cax = fig.add_axes(
        [axs[0, 1].get_position().x1 + 0.01, axs[0, 1].get_position().y0, 0.02, axs[0, 1].get_position().height])
    cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical', extend='both', ticks=np.linspace(-40, 40, 5, endpoint=True))
    cbar.ax.tick_params(labelsize=13)

    cax = fig.add_axes(
        [axs[1, 1].get_position().x1 + 0.01, axs[1, 1].get_position().y0, 0.02, axs[1, 1].get_position().height])
    cbar = fig.colorbar(cs[1, 1], cax=cax, orientation='vertical', extend='both',
                        ticks=np.linspace(-10, 10, 5, endpoint=True))
    cbar.ax.tick_params(labelsize=13)

    axs[0, 0].text(-0.05, 0.5, '2m dew point temp', ha='center', va='center', rotation='vertical',
                   transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')
    axs[1, 1].text(-0.05, 0.5, 'Differences', ha='center', va='center', rotation='vertical',
                   transform=axs[1, 1].transAxes, fontsize=13, fontweight='bold')
    axs[0, 1].text(1.05, 1.08, '[$^{o}$C]', ha='center', va='center', rotation='horizontal',
                   transform=axs[0, 1].transAxes, fontsize=12)
    axs[0, 0].text(-0.09, 1.075, f'PD-{m}', ha='left', va='center', rotation='horizontal',
                   transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')

    fig.show()
    plotpath = "/project/pr133/rxiang/figure/echam5/"
    fig.savefig(plotpath + 'dew2.' + f'{mon}.era5.png', dpi=500)
    plt.close(fig)

#%%

for mon in ['01', '07']:
    m = month[mon]
    fig = plt.figure(figsize=(wi, hi))
    left, bottom, right, top = 0.05, 0.01, 0.9, 0.94
    gs = gridspec.GridSpec(nrows=2, ncols=2, left=left, bottom=bottom, right=right, top=top,
                           wspace=0.01, hspace=0.15)

    for i in range(2):
        sim = sims[i]
        label = tmp[sim]['label']
        axs[0, i] = fig.add_subplot(gs[0, i], projection=ccrs.Robinson(central_longitude=180, globe=None))
        axs[0, i].coastlines(zorder=3)
        axs[0, i].stock_img()
        axs[0, i].gridlines()
        cs[0, i] = axs[0, i].pcolormesh(lon, lat, tmp[sim][mon]['t2m'], cmap=cmap1, norm=norm1, shading="auto",
                                        transform=ccrs.PlateCarree())
        axs[0, i].set_title(f'{label}', fontweight='bold', pad=6, fontsize=13, loc='center')

    for i in range(1):
        sim = sims[i + 1]
        axs[1, i + 1] = fig.add_subplot(gs[1, i + 1], projection=ccrs.Robinson(central_longitude=180, globe=None))
        axs[1, i + 1].coastlines(zorder=3)
        axs[1, i + 1].stock_img()
        axs[1, i + 1].gridlines()
        cs[1, i + 1] = axs[1, i + 1].pcolormesh(lon, lat, tmp[sim][mon]['t2m'] - tmp['PD'][mon]['t2m'], cmap=cmap2,
                                                clim=(-10,  10), shading="auto", transform=ccrs.PlateCarree())

    cax = fig.add_axes(
        [axs[0, 1].get_position().x1 + 0.01, axs[0, 1].get_position().y0, 0.02, axs[0, 1].get_position().height])
    cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical', extend='both', ticks=np.linspace(-40, 40, 5, endpoint=True))
    cbar.ax.tick_params(labelsize=13)

    cax = fig.add_axes(
        [axs[1, 1].get_position().x1 + 0.01, axs[1, 1].get_position().y0, 0.02, axs[1, 1].get_position().height])
    cbar = fig.colorbar(cs[1, 1], cax=cax, orientation='vertical', extend='both',
                        ticks=np.linspace(-10, 10, 5, endpoint=True))
    cbar.ax.tick_params(labelsize=13)

    axs[0, 0].text(-0.05, 0.5, '2m temp', ha='center', va='center', rotation='vertical',
                   transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')
    axs[1, 1].text(-0.05, 0.5, 'Differences', ha='center', va='center', rotation='vertical',
                   transform=axs[1, 1].transAxes, fontsize=13, fontweight='bold')
    axs[0, 1].text(1.05, 1.08, '[$^{o}$C]', ha='center', va='center', rotation='horizontal',
                   transform=axs[0, 1].transAxes, fontsize=12)
    axs[0, 0].text(-0.09, 1.075, f'PD-{m}', ha='left', va='center', rotation='horizontal',
                   transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')

    fig.show()
    plotpath = "/project/pr133/rxiang/figure/echam5/"
    fig.savefig(plotpath + 'temp2.' + f'{mon}.era5.png', dpi=500)
    plt.close(fig)




