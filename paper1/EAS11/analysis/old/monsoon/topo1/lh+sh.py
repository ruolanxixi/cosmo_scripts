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


# -------------------------------------------------------------------------------
# read data
#
sims = ['ctrl', 'topo1']
all_cpm, all_cpm_sms = [], []
# ALHFL_S  ASHFL_S
for s in range(len(sims)):
    sim = sims[s]
    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/ALHFL_S'
    data = Dataset(f'{path}' + '/' + '01-05.ALHFL_S.cpm.nc')
    cpm = - data.variables['ALHFL_S'][...]
    cpm = np.nanmean(cpm[24:30], axis=0)
    cpm_sms = ndimage.gaussian_filter(cpm, sigma=10, order=0)
    all_cpm.append(cpm)
    all_cpm_sms.append(cpm_sms)

ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_EAS_ext_12km_merit_unmod_topo.nc')
hsurf_ctrl = ds['HSURF'].values[:, :]
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/old/extpar_EAS_ext_12km_merit_adj.nc')
hsurf_topo1 = ds['HSURF'].values[:, :]
lat_ = ds["lat"].values
lon_ = ds["lon"].values
ds.close()

# -------------------------------------------------------------------------------
# plot
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

ar = 1.0  # initial aspect ratio for first trial
wi = 12  # height in inches
hi = wi * ar  # width in inches
ncol = 2  # edit here
nrow = 2
axs, cs, ct = np.empty(4, dtype='object'), np.empty(4, dtype='object'), np.empty(4, dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.05, 0.1, 0.9, 0.93
gs = gridspec.GridSpec(2, 2, left=left, bottom=bottom, right=right, top=top, wspace=0.12, hspace=0.09)
axs[0] = fig.add_subplot(gs[0], projection=rot_pole_crs)
axs[1] = fig.add_subplot(gs[2], projection=rot_pole_crs)
axs[2] = fig.add_subplot(gs[3], projection=rot_pole_crs)

# plot topography
axs[3] = fig.add_subplot(gs[1], projection=rot_pole_crs)

levels = MaxNLocator(nbins=22).tick_values(0, 200)
cmap = cmc.roma_r
# cmap = cbr_wet(15)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

sims = ['Control', 'Reduced topography']

for i in range(2):
    sim = sims[i]
    axs[i] = plotcosmo(axs[i])
    cs[i] = axs[i].pcolormesh(rlon, rlat, all_cpm[i], cmap=cmap, norm=norm, shading="auto")
    ct[i] = axs[i].contour(rlon, rlat, all_cpm_sms[i], levels=np.linspace(0, 200, 11, endpoint=True), colors='maroon', linewidths=1)
    axs[i].text(0., 1.02, f'{sim}', ha='left', va='bottom', transform=axs[i].transAxes, fontsize=14)

    # trans = Transformer.from_proj(ccrs.PlateCarree(), rot_pole_crs, always_xy=True)
    # x = np.array([90, 102, 152, 140, 105, 147, 70])
    # y = np.array([35, 33, 29, 21, 23, 10, 9])
    # loc_lon, loc_lat = trans.transform(x, y)
    # manual_locations = [i for i in zip(loc_lon, loc_lat)]
    clabel = axs[i].clabel(ct[i], np.linspace(0, 200, 11, endpoint=True), inline=True, fontsize=13, use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

cax = fig.add_axes([axs[1].get_position().x0, axs[1].get_position().y0 - .125, axs[1].get_position().width, 0.02])
cbar = fig.colorbar(cs[1], cax=cax, orientation='horizontal', extend='max')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('$W/m^2$', fontsize=13, labelpad=-0.01)

# plot difference
levels = MaxNLocator(nbins=15).tick_values(-30, 30)
cmap = custom_div_cmap(25, cmc.vik)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

axs[2] = plotcosmo(axs[2])
cs[2] = axs[2].pcolormesh(rlon, rlat, all_cpm[1] - all_cpm[0], cmap=cmap, clim=(-40, 40), shading="auto")
ct[2] = axs[2].contour(rlon, rlat, all_cpm_sms[1] - all_cpm_sms[0], levels=np.linspace(-20, 20, 5, endpoint=True), colors='maroon',
                       linewidths=1)
axs[2].text(0, 1.02, 'Reduced topography - Control', ha='left', va='bottom', transform=axs[2].transAxes, fontsize=14)

# trans = Transformer.from_proj(ccrs.PlateCarree(), rot_pole_crs, always_xy=True)
# x = np.array([90, 90, 119, 113, 140, 138, 160, 170, 90])
# y = np.array([35, 27, 28, 22, 29, 13, 15, 42, 57])
# loc_lon, loc_lat = trans.transform(x, y)
# manual_locations = [i for i in zip(loc_lon, loc_lat)]
# clabel = axs[2].clabel(ct[2], [-2., 0, 2], inline=True, fontsize=13, manual=manual_locations)
clabel = axs[2].clabel(ct[2], [-20, -10., 0, 10, 20], inline=True, use_clabeltext=True, fontsize=13)
for l in clabel:
    l.set_rotation(0)
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.7)) for txt in clabel]

cax = fig.add_axes([axs[2].get_position().x0, axs[2].get_position().y0 - .125, axs[2].get_position().width, 0.02])
cbar = fig.colorbar(cs[2], cax=cax, orientation='horizontal', extend='both')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('$W/m^2$', fontsize=13, labelpad=-0.01)

# plot topography
levels = np.arange(-3500.0, 0, 50.0)
ticks = np.arange(-3500.0, 0, 500.0)
cmap = custom_seq_cmap_(70, cmc.lisbon)
norm = BoundaryNorm(levels, ncolors=cmap.N, extend="max")

axs[3] = plotcosmo(axs[3])
cs[3] = axs[3].pcolormesh(lon_, lat_,  hsurf_topo1 - hsurf_ctrl, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading="auto")
# ct[3] = axs[3].contour(lon_, lat_, hsurf_ctrl - hsurf_topo1, colors='maroon', linewidths=1)
axs[3].text(0, 1.02, 'Reduced topography - Control', ha='left', va='bottom', transform=axs[3].transAxes, fontsize=14)

cax = fig.add_axes([axs[3].get_position().x1 + 0.015, axs[3].get_position().y0-0.07, 0.012, axs[3].get_position().height*1.6])
cbar = fig.colorbar(cs[3], cax=cax, orientation='vertical', extend='min', ticks=ticks)
cbar.ax.tick_params(labelsize=13)
axs[3].text(1.035, -0.1, 'm', ha='left', va='bottom', transform=axs[3].transAxes, fontsize=13)
# cbar.ax.set_xlabel('m', fontsize=13, labelpad=-0.01, loc='left')

fig.suptitle('Latent Heat Flux (May)', fontsize=16, fontweight='bold')

xmin, xmax = axs[1].get_xbound()
ymin, ymax = axs[1].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol + 0.065
fig.set_figheight(wi * y2x_ratio)

fig.show()
plotpath = "/project/pr133/rxiang/figure/EAS11/analysis/monsoon/topo1/"
fig.savefig(plotpath + 'lh_may.png', dpi=500)
plt.close(fig)

# Sensible heat
sims = ['ctrl', 'topo1']
all_cpm, all_cpm_sms = [], []
# ALHFL_S  ASHFL_S
for s in range(len(sims)):
    sim = sims[s]
    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/ASHFL_S'
    data = Dataset(f'{path}' + '/' + '01-05.ASHFL_S.cpm.nc')
    cpm = - data.variables['ASHFL_S'][...]
    cpm = np.nanmean(cpm[24:30], axis=0)
    cpm_sms = ndimage.gaussian_filter(cpm, sigma=10, order=0)
    all_cpm.append(cpm)
    all_cpm_sms.append(cpm_sms)

ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_EAS_ext_12km_merit_unmod_topo.nc')
hsurf_ctrl = ds['HSURF'].values[:, :]
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/old/extpar_EAS_ext_12km_merit_adj.nc')
hsurf_topo1 = ds['HSURF'].values[:, :]
lat_ = ds["lat"].values
lon_ = ds["lon"].values
ds.close()

# -------------------------------------------------------------------------------
# plot
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

ar = 1.0  # initial aspect ratio for first trial
wi = 12  # height in inches
hi = wi * ar  # width in inches
ncol = 2  # edit here
nrow = 2
axs, cs, ct = np.empty(4, dtype='object'), np.empty(4, dtype='object'), np.empty(4, dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.05, 0.1, 0.9, 0.93
gs = gridspec.GridSpec(2, 2, left=left, bottom=bottom, right=right, top=top, wspace=0.12, hspace=0.09)
axs[0] = fig.add_subplot(gs[0], projection=rot_pole_crs)
axs[1] = fig.add_subplot(gs[2], projection=rot_pole_crs)
axs[2] = fig.add_subplot(gs[3], projection=rot_pole_crs)

# plot topography
axs[3] = fig.add_subplot(gs[1], projection=rot_pole_crs)

levels = MaxNLocator(nbins=22).tick_values(-20, 160)
cmap = cmc.roma_r
# cmap = cbr_wet(15)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

sims = ['Control', 'Reduced topography']

for i in range(2):
    sim = sims[i]
    axs[i] = plotcosmo(axs[i])
    cs[i] = axs[i].pcolormesh(rlon, rlat, all_cpm[i], cmap=cmap, norm=norm, shading="auto")
    ct[i] = axs[i].contour(rlon, rlat, all_cpm_sms[i], levels=np.linspace(-20, 160, 10, endpoint=True), colors='maroon', linewidths=1)
    axs[i].text(0., 1.02, f'{sim}', ha='left', va='bottom', transform=axs[i].transAxes, fontsize=14)

    # trans = Transformer.from_proj(ccrs.PlateCarree(), rot_pole_crs, always_xy=True)
    # x = np.array([90, 102, 152, 140, 105, 147, 70])
    # y = np.array([35, 33, 29, 21, 23, 10, 9])
    # loc_lon, loc_lat = trans.transform(x, y)
    # manual_locations = [i for i in zip(loc_lon, loc_lat)]
    clabel = axs[i].clabel(ct[i], np.linspace(-20, 160, 10, endpoint=True), inline=True, fontsize=13, use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

cax = fig.add_axes([axs[1].get_position().x0, axs[1].get_position().y0 - .125, axs[1].get_position().width, 0.02])
cbar = fig.colorbar(cs[1], cax=cax, orientation='horizontal', extend='max')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('$W/m^2$', fontsize=13, labelpad=-0.01)

# plot difference
levels = MaxNLocator(nbins=15).tick_values(-50, 50)
cmap = custom_div_cmap(25, cmc.vik)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

axs[2] = plotcosmo(axs[2])
cs[2] = axs[2].pcolormesh(rlon, rlat, all_cpm[1] - all_cpm[0], cmap=cmap, clim=(-30, 30), shading="auto")
ct[2] = axs[2].contour(rlon, rlat, all_cpm_sms[1] - all_cpm_sms[0], levels=np.linspace(-20, 20, 5, endpoint=True), colors='maroon',
                       linewidths=1)
axs[2].text(0, 1.02, 'Reduced topography - Control', ha='left', va='bottom', transform=axs[2].transAxes, fontsize=14)

# trans = Transformer.from_proj(ccrs.PlateCarree(), rot_pole_crs, always_xy=True)
# x = np.array([90, 90, 119, 113, 140, 138, 160, 170, 90])
# y = np.array([35, 27, 28, 22, 29, 13, 15, 42, 57])
# loc_lon, loc_lat = trans.transform(x, y)
# manual_locations = [i for i in zip(loc_lon, loc_lat)]
# clabel = axs[2].clabel(ct[2], [-2., 0, 2], inline=True, fontsize=13, manual=manual_locations)
clabel = axs[2].clabel(ct[2], [-20, -10., 0, 10, 20], inline=True, use_clabeltext=True, fontsize=13)
for l in clabel:
    l.set_rotation(0)
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.7)) for txt in clabel]

cax = fig.add_axes([axs[2].get_position().x0, axs[2].get_position().y0 - .125, axs[2].get_position().width, 0.02])
cbar = fig.colorbar(cs[2], cax=cax, orientation='horizontal', extend='both')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('$W/m^2$', fontsize=13, labelpad=-0.01)

# plot topography
levels = np.arange(-3500.0, 0, 50.0)
ticks = np.arange(-3500.0, 0, 500.0)
cmap = custom_seq_cmap_(70, cmc.lisbon)
norm = BoundaryNorm(levels, ncolors=cmap.N, extend="max")

axs[3] = plotcosmo(axs[3])
cs[3] = axs[3].pcolormesh(lon_, lat_,  hsurf_topo1 - hsurf_ctrl, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading="auto")
# ct[3] = axs[3].contour(lon_, lat_, hsurf_ctrl - hsurf_topo1, colors='maroon', linewidths=1)
axs[3].text(0, 1.02, 'Reduced topography - Control', ha='left', va='bottom', transform=axs[3].transAxes, fontsize=14)

cax = fig.add_axes([axs[3].get_position().x1 + 0.015, axs[3].get_position().y0-0.07, 0.012, axs[3].get_position().height*1.6])
cbar = fig.colorbar(cs[3], cax=cax, orientation='vertical', extend='min', ticks=ticks)
cbar.ax.tick_params(labelsize=13)
axs[3].text(1.035, -0.1, 'm', ha='left', va='bottom', transform=axs[3].transAxes, fontsize=13)
# cbar.ax.set_xlabel('m', fontsize=13, labelpad=-0.01, loc='left')

fig.suptitle('Sensible Heat Flux (May)', fontsize=16, fontweight='bold')

xmin, xmax = axs[1].get_xbound()
ymin, ymax = axs[1].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol + 0.065
fig.set_figheight(wi * y2x_ratio)

fig.show()
plotpath = "/project/pr133/rxiang/figure/EAS11/analysis/monsoon/topo1/"
fig.savefig(plotpath + 'sh_may.png', dpi=500)
plt.close(fig)




