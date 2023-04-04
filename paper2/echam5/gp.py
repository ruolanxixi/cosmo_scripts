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

font = {'size': 11}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
# %%
sims = ['PD', 'PI', 'LGM', 'PLIO']
mdpath = "/project/pr133/rxiang/data/echam5"
gp500, gp200 = {}, {}
fname500 = 'gp_50000.smr.nc'
fname200 = 'gp_20000.smr.nc'
labels = {'PD': 'Present-day', 'PI': 'Pre-industrial', 'LGM': 'Last glacial bothimum', 'PLIO': 'Mid-Pliocene'}

for s in range(len(sims)):
    sim = sims[s]
    data_geopoth = xr.open_dataset(f'{mdpath}/{sim}/analysis/gp/monsoon/smr/{fname500}')['geopoth'].values[:, 0, :, :]
    geopoth = np.nanmean(data_geopoth, axis=0)
    gp500[sim] = {}
    gp500[sim]['geopoth'] = geopoth
    gp_sms = ndimage.gaussian_filter(geopoth, sigma=10, order=0)
    gp500[sim]['geopothsms'] = gp_sms
    gp500[sim]['label'] = labels[sim]
    del data_geopoth, geopoth, gp_sms

for s in range(len(sims)):
    sim = sims[s]
    data_geopoth = xr.open_dataset(f'{mdpath}/{sim}/analysis/gp/monsoon/smr/{fname200}')['geopoth'].values[:, 0, :, :]
    geopoth = np.nanmean(data_geopoth, axis=0)
    gp200[sim] = {}
    gp200[sim]['geopoth'] = geopoth
    gp_sms = ndimage.gaussian_filter(geopoth, sigma=10, order=0)
    gp200[sim]['geopothsms'] = gp_sms
    gp200[sim]['label'] = labels[sim]
    del data_geopoth, geopoth, gp_sms

lat1 = xr.open_dataset(f'{mdpath}/PD/analysis/gp/monsoon/smr/{fname500}')['lat'].values[:]
lon1 = xr.open_dataset(f'{mdpath}/PD/analysis/gp/monsoon/smr/{fname500}')['lon'].values[:]
lat1_, lon1_ = np.meshgrid(lon1, lat1)
lat2 = xr.open_dataset(f'{mdpath}/LGM/analysis/gp/monsoon/smr/{fname500}')['lat'].values[:]
lon2 = xr.open_dataset(f'{mdpath}/LGM/analysis/gp/monsoon/smr/{fname500}')['lon'].values[:]
lat2_, lon2_ = np.meshgrid(lon2, lat2)

# -------------------------------------------------------------------------------
# plot
# %%
ar = 1.0  # initial aspect ratio for first trial
wi = 17  # height in inches #15
hi = 6  # width in inches #10
ncol = 4  # edit here
nrow = 2
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                           np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'),\
                           np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object')

fig = plt.figure(figsize=(wi, hi))

left, bottom, right, top = 0.04, 0.61, 0.99, 0.95
gs1 = gridspec.GridSpec(nrows=1, ncols=4, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.02, hspace=0.15)
left, bottom, right, top = 0.04, 0.14, 0.99, 0.48
gs2 = gridspec.GridSpec(nrows=1, ncols=4, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.02, hspace=0.15)

levels1 = np.linspace(5400, 6000, 13, endpoint=True)
cmap1 = cmc.roma_r
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

levels2 = np.linspace(11500, 12700, 13, endpoint=True)
cmap2 = cmc.roma_r
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

axs[0, 0] = fig.add_subplot(gs1[0], projection=ccrs.Robinson(central_longitude=180, globe=None))
axs[1, 0] = fig.add_subplot(gs2[0], projection=ccrs.Robinson(central_longitude=180, globe=None))
for ax in [axs[0, 0], axs[1, 0]]:
    ax.coastlines(zorder=3)
    ax.stock_img()
    ax.gridlines()

cs[0, 0] = axs[0, 0].pcolormesh(lon1, lat1, gp500['PD']['geopoth'], cmap=cmap1, norm=norm1, shading="auto", transform=ccrs.PlateCarree())
ct[0, 0] = axs[0, 0].contour(lon1, lat1, gp500['PD']['geopothsms'], levels=np.linspace(5400, 6000, 13, endpoint=True),
                             colors='maroon', linewidths=1, transform=ccrs.PlateCarree())
# clabel = axs[0, 0].clabel(ct[0, 0], np.linspace(5480, 5900, 15, endpoint=True), inline=True, fontsize=13, use_clabeltext=True)
# for l in clabel:
#     l.set_rotation(0)
# [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

label = gp500['PD']['label']
axs[0, 0].set_title(f'{label}', fontweight='bold', pad=7, fontsize=13, loc='center')

cs[1, 0] = axs[1, 0].pcolormesh(lon1, lat1, gp200['PD']['geopoth'], cmap=cmap2, norm=norm2, shading="auto", transform=ccrs.PlateCarree())
ct[1, 0] = axs[1, 0].contour(lon1, lat1, gp200['PD']['geopothsms'], levels=np.linspace(11500, 12700, 13, endpoint=True),
                             colors='maroon', linewidths=1, transform=ccrs.PlateCarree())
# clabel = axs[1, 0].clabel(ct[1, 0], levels=np.linspace(11700, 12600, 10, endpoint=True), inline=True, fontsize=13, use_clabeltext=True)
# for l in clabel:
#     l.set_rotation(0)
# [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

for j in range(3):
    sim = sims[j+1]
    axs[0, j+1] = fig.add_subplot(gs1[j+1], projection=ccrs.Robinson(central_longitude=180, globe=None))
    axs[1, j+1] = fig.add_subplot(gs2[j+1], projection=ccrs.Robinson(central_longitude=180, globe=None))
    for ax in [axs[0, j+1], axs[1, j+1]]:
        ax.coastlines(zorder=3)
        ax.stock_img()
        ax.gridlines()

    cs[0, j+1] = axs[0, j+1].pcolormesh(lon2, lat2, gp500[sim]['geopoth'], cmap=cmap1, norm=norm1, shading="auto", transform=ccrs.PlateCarree())
    ct[0, j+1] = axs[0, j+1].contour(lon2, lat2, gp500[sim]['geopothsms'], levels=np.linspace(5400, 6000, 13, endpoint=True),
                                 colors='maroon', linewidths=1, transform=ccrs.PlateCarree())
    # clabel = axs[0, j+1].clabel(ct[0, j+1], levels=np.linspace(5480, 5900, 15, endpoint=True), inline=True, fontsize=13, use_clabeltext=True)
    # for l in clabel:
    #     l.set_rotation(0)
    # [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]
    
    label = gp500[sim]['label']
    axs[0, j+1].set_title(f'{label}', fontweight='bold', pad=7, fontsize=13, loc='center')

    cs[1, j+1] = axs[1, j+1].pcolormesh(lon2, lat2, gp200[sim]['geopoth'], cmap=cmap2, norm=norm2, shading="auto", transform=ccrs.PlateCarree())
    ct[1, j+1] = axs[1, j+1].contour(lon2, lat2, gp200[sim]['geopothsms'], levels=np.linspace(11500, 12700, 13, endpoint=True),
                                 colors='maroon', linewidths=1, transform=ccrs.PlateCarree())
    # clabel = axs[1, j+1].clabel(ct[1, j+1], levels=np.linspace(11700, 12600, 10, endpoint=True), inline=True, fontsize=13, use_clabeltext=True)
    # for l in clabel:
    #     l.set_rotation(0)
    # [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]
    
cax = fig.add_axes([axs[0, 1].get_position().x0, axs[0, 1].get_position().y0 - .05, axs[0, 2].get_position().x1 - axs[0, 1].get_position().x0, 0.02])
cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='horizontal', extend='both', ticks=np.linspace(5400, 6000, 13, endpoint=True))
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('gpm', fontsize=13)

cax = fig.add_axes([axs[1, 1].get_position().x0, axs[1, 1].get_position().y0 - .05, axs[1, 2].get_position().x1 - axs[1, 1].get_position().x0, 0.02])
cbar = fig.colorbar(cs[1, 1], cax=cax, orientation='horizontal', extend='both', ticks=np.linspace(11500, 12700, 13, endpoint=True))
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('gpm', fontsize=13)

axs[0, 0].text(-0.1, 0.5, '500 hPa gp', ha='center', va='center', rotation='vertical',
               transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')
axs[1, 0].text(-0.1, 0.5, '200 hPa gp', ha='center', va='center', rotation='vertical',
               transform=axs[1, 0].transAxes, fontsize=13, fontweight='bold')

fig.show()
plotpath = "/project/pr133/rxiang/figure/echam5/"
fig.savefig(plotpath + 'gp.png', dpi=500)
plt.close(fig)






