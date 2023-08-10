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
import matplotlib.path as mpath

font = {'size': 11}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
#
sims = ['PD', 'PI', 'LGM', 'PLIO']
mdpath = "/project/pr133/rxiang/data/echam"
wind850, wind200 = {}, {}
fname850 = 'wind_85000.smr.nc'
fname200 = 'wind_20000.smr.nc'
labels = {'PD': 'Present-day', 'PI': 'Pre-industrial', 'LGM': 'Last glacial maximum', 'PLIO': 'Mid-Pliocene'}

for s in range(len(sims)):
    sim = sims[s]
    data_v = xr.open_dataset(f'{mdpath}/{sim}/analysis/wind/monsoon/smr/{fname850}')['v'].values[:, 0, :, :]
    v = np.nanmean(data_v, axis=0)
    wind850[sim] = {}
    wind850[sim]['v'] = v
    data_u = xr.open_dataset(f'{mdpath}/{sim}/analysis/wind/monsoon/smr/{fname850}')['u'].values[:, 0, :, :]
    u = np.nanmean(data_u, axis=0)
    wind850[sim]['u'] = u
    ws = np.sqrt(u**2+v**2)
    ws_sms = ndimage.gaussian_filter(np.sqrt(u**2+v**2), sigma=10, order=0)
    wind850[sim]['ws'] = ws_sms
    wind850[sim]['wssms'] = ws_sms
    wind850[sim]['label'] = labels[sim]
    del data_v, data_u, u, v, ws, ws_sms

for s in range(len(sims)):
    sim = sims[s]
    data_v = xr.open_dataset(f'{mdpath}/{sim}/analysis/wind/monsoon/smr/{fname200}')['v'].values[:, 0, :, :]
    v = np.nanmean(data_v, axis=0)
    wind200[sim] = {}
    wind200[sim]['v'] = v
    data_u = xr.open_dataset(f'{mdpath}/{sim}/analysis/wind/monsoon/smr/{fname200}')['u'].values[:, 0, :, :]
    u = np.nanmean(data_u, axis=0)
    wind200[sim]['u'] = u
    ws = np.sqrt(u ** 2 + v ** 2)
    ws_sms = ndimage.gaussian_filter(np.sqrt(u**2+v**2), sigma=10, order=0)
    wind200[sim]['ws'] = ws
    wind200[sim]['wssms'] = ws_sms
    del data_v, data_u, u, v, ws

lat1 = xr.open_dataset(f'{mdpath}/PD/analysis/wind/monsoon/smr/{fname850}')['lat'].values[:]
lon1 = xr.open_dataset(f'{mdpath}/PD/analysis/wind/monsoon/smr/{fname850}')['lon'].values[:]
lat1_, lon1_ = np.meshgrid(lon1, lat1)
lat2 = xr.open_dataset(f'{mdpath}/LGM/analysis/wind/monsoon/smr/{fname850}')['lat'].values[:]
lon2 = xr.open_dataset(f'{mdpath}/LGM/analysis/wind/monsoon/smr/{fname850}')['lon'].values[:]
lat2_, lon2_ = np.meshgrid(lon2, lat2)

# -------------------------------------------------------------------------------
# plot
# %%
ar = 1.0  # initial aspect ratio for first trial
wi = 12  # height in inches #15
hi = 7  # width in inches #10
ncol = 4  # edit here
nrow = 2
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                           np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'),\
                           np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object')

fig = plt.figure(figsize=(wi, hi))

left, bottom, right, top = 0.06, 0.60, 0.99, 0.95
gs1 = gridspec.GridSpec(nrows=1, ncols=4, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.02, hspace=0.15)
left, bottom, right, top = 0.06, 0.12, 0.99, 0.48
gs2 = gridspec.GridSpec(nrows=1, ncols=4, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.02, hspace=0.15)

levels1 = MaxNLocator(nbins=14).tick_values(0, 14)
cmap1 = wind(14, cmc.batlowW_r)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

levels2 = MaxNLocator(nbins=40).tick_values(0, 40)
cmap2 = wind(40, cmc.batlowW_r)
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

axs[0, 0] = fig.add_subplot(gs1[0], projection=ccrs.NorthPolarStereo(central_longitude=120.0))
axs[0, 0].set_extent([-180, 180, 0, 90], ccrs.PlateCarree())
axs[0, 0].set_boundary(circle, transform=axs[0, 0].transAxes)
axs[1, 0] = fig.add_subplot(gs2[0], projection=ccrs.NorthPolarStereo(central_longitude=120.0))
axs[1, 0].set_extent([-180, 180, 0, 90], ccrs.PlateCarree())
axs[1, 0].set_boundary(circle, transform=axs[1, 0].transAxes)
for ax in [axs[0, 0], axs[1, 0]]:
    ax.coastlines(zorder=3)
    ax.stock_img()
    ax.gridlines()

cs[0, 0] = axs[0, 0].pcolormesh(lon1, lat1, wind850['PD']['ws'], cmap=cmap1, norm=norm1, shading="auto", transform=ccrs.PlateCarree())
ct[0, 0] = axs[0, 0].contour(lon1, lat1, wind850['PD']['wssms'], levels=np.linspace(2, 14, 5, endpoint=True),
                             colors='maroon', linewidths=1, transform=ccrs.PlateCarree())
clabel = axs[0, 0].clabel(ct[0, 0], [2, 5, 8, 11, 14], inline=True, fontsize=13, use_clabeltext=True)
for l in clabel:
    l.set_rotation(0)
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]
q[0, 0] = axs[0, 0].quiver(lon1[::3], lat1[::3], wind850['PD']['u'][::3, ::3],
                           wind850['PD']['v'][::3, ::3], color='black', scale=150, transform=ccrs.PlateCarree())

label = wind850['PD']['label']
axs[0, 0].set_title(f'{label}', fontweight='bold', pad=7, fontsize=13, loc='center')

cs[1, 0] = axs[1, 0].pcolormesh(lon1, lat1, wind200['PD']['ws'], cmap=cmap2, norm=norm2, shading="auto", transform=ccrs.PlateCarree())
ct[1, 0] = axs[1, 0].contour(lon1, lat1, wind200['PD']['wssms'], levels=np.linspace(5, 40, 8, endpoint=True),
                             colors='maroon', linewidths=1, transform=ccrs.PlateCarree())
clabel = axs[1, 0].clabel(ct[1, 0], levels=np.linspace(5, 40, 8, endpoint=True), inline=True, fontsize=13, use_clabeltext=True)
for l in clabel:
    l.set_rotation(0)
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]
q[1, 0] = axs[1, 0].quiver(lon1[::3], lat1[::3], wind200['PD']['u'][::3, ::3],
                           wind200['PD']['v'][::3, ::3], color='black', scale=300, transform=ccrs.PlateCarree())

for j in range(3):
    sim = sims[j+1]
    axs[0, j+1] = fig.add_subplot(gs1[j+1], projection=ccrs.NorthPolarStereo(central_longitude=120.0))
    axs[0, j+1].set_extent([-180, 180, 0, 90], ccrs.PlateCarree())
    axs[0, j+1].set_boundary(circle, transform=axs[0, j+1].transAxes)
    axs[1, j+1] = fig.add_subplot(gs2[j+1], projection=ccrs.NorthPolarStereo(central_longitude=120.0))
    axs[1, j+1].set_extent([-180, 180, 0, 90], ccrs.PlateCarree())
    axs[1, j+1].set_boundary(circle, transform=axs[1, j+1].transAxes)
    for ax in [axs[0, j+1], axs[1, j+1]]:
        ax.coastlines(zorder=3)
        ax.stock_img()
        ax.gridlines()

    cs[0, j+1] = axs[0, j+1].pcolormesh(lon2, lat2, wind850[sim]['ws'], cmap=cmap1, norm=norm1, shading="auto", transform=ccrs.PlateCarree())
    ct[0, j+1] = axs[0, j+1].contour(lon2, lat2, wind850[sim]['wssms'], levels=np.linspace(2, 14, 5, endpoint=True),
                                 colors='maroon', linewidths=1, transform=ccrs.PlateCarree())
    clabel = axs[0, j+1].clabel(ct[0, j+1], [2, 5, 8, 11, 14], inline=True, fontsize=13, use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]
    q[0, j+1] = axs[0, j+1].quiver(lon2[::6], lat2[::6], wind850[sim]['u'][::6, ::6],
                               wind850[sim]['v'][::6, ::6], color='black', scale=150, transform=ccrs.PlateCarree())

    label = wind850[sim]['label']
    axs[0, j+1].set_title(f'{label}', fontweight='bold', pad=7, fontsize=13, loc='center')

    cs[1, j+1] = axs[1, j+1].pcolormesh(lon2, lat2, wind200[sim]['ws'], cmap=cmap2, norm=norm2, shading="auto", transform=ccrs.PlateCarree())
    ct[1, j+1] = axs[1, j+1].contour(lon2, lat2, wind200[sim]['wssms'], levels=np.linspace(5, 40, 8, endpoint=True),
                                 colors='maroon', linewidths=1, transform=ccrs.PlateCarree())
    clabel = axs[1, j+1].clabel(ct[1, j+1], levels=np.linspace(5, 40, 8, endpoint=True), inline=True, fontsize=13, use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]
    q[1, j+1] = axs[1, j+1].quiver(lon2[::6], lat2[::6], wind200[sim]['u'][::6, ::6],
                               wind200[sim]['v'][::6, ::6], color='black', scale=300, transform=ccrs.PlateCarree())

qk[0, 3] = axs[0, 3].quiverkey(q[0, 3], 0.9, 1.06, 10, r'$10\ \frac{m}{s}$', labelpos='E', transform=axs[0, 3].transAxes,
                      fontproperties={'size': 12})
qk[1, 3] = axs[1, 3].quiverkey(q[1, 3], 0.9, 1.06, 20, r'$20\ \frac{m}{s}$', labelpos='E', transform=axs[1, 3].transAxes,
                      fontproperties={'size': 12})

cax = fig.add_axes([axs[0, 1].get_position().x0, axs[0, 1].get_position().y0 - .05, axs[0, 2].get_position().x1 - axs[0, 1].get_position().x0, 0.02])
cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='horizontal', extend='max', ticks=np.linspace(0, 20, 11, endpoint=True))
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('m s$^{-1}$', fontsize=13)

cax = fig.add_axes([axs[1, 1].get_position().x0, axs[1, 1].get_position().y0 - .05, axs[1, 2].get_position().x1 - axs[1, 1].get_position().x0, 0.02])
cbar = fig.colorbar(cs[1, 1], cax=cax, orientation='horizontal', extend='max', ticks=np.linspace(0, 40, 11, endpoint=True))
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('m s$^{-1}$', fontsize=13)

axs[0, 0].text(-0.15, 0.5, '850 hPa wind', ha='center', va='center', rotation='vertical',
               transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')
axs[1, 0].text(-0.15, 0.5, '200 hPa wind', ha='center', va='center', rotation='vertical',
               transform=axs[1, 0].transAxes, fontsize=13, fontweight='bold')

fig.show()
# plotpath = "/project/pr133/rxiang/figure/analysis/EAS11/topo1/smr/"
# fig.savefig(plotpath + 'wind850.png', dpi=500)
# plt.close(fig)






