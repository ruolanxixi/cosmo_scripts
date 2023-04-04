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
sims = ['ctrl', 'topo2']
all_smr, all_smr_sms = [], []
all_rg, all_rg_sms = [], []
all_rt, all_rt_sms = [], []

for s in range(len(sims)):
    sim = sims[s]
    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/TOT_PREC/smr'
    data = Dataset(f'{path}' + '/' + '01-05.TOT_PREC.smr.nc')
    smr = data.variables['TOT_PREC'][...]
    smr = np.nanmean(smr, axis=0)
    smr_sms = ndimage.gaussian_filter(smr, sigma=10, order=0)
    all_smr.append(smr)
    all_smr_sms.append(smr_sms)
    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/TOT_PREC/'
    data = Dataset(f'{path}' + '/' + '01-05.TOT_PREC.cpm.nc')
    cpm = data.variables['TOT_PREC'][...]
    rg = np.nanmax(cpm, axis=0) - np.nanmin(cpm, axis=0)
    rg_sms = ndimage.gaussian_filter(rg, sigma=10, order=0)
    all_rg.append(rg)
    all_rg_sms.append(rg_sms)
    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/TOT_PREC/smr'
    data = Dataset(f'{path}' + '/' + '01-05.TOT_PREC.smr.nc')
    smr = data.variables['TOT_PREC'][...]
    smr = np.nansum(smr, axis=0)
    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/day/TOT_PREC'
    data = Dataset(f'{path}' + '/' + '01-05.TOT_PREC.nc')
    yr = data.variables['TOT_PREC'][...]
    yr = np.nansum(yr, axis=0)
    rt = smr / yr * 100
    rt_sms = ndimage.gaussian_filter(rt, sigma=10, order=0)
    all_rt.append(rt)
    all_rt_sms.append(rt_sms)

ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_EAS_ext_12km_merit_unmod_topo.nc')
hsurf_ctrl = ds['HSURF'].values[:, :]
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/old/extpar_EAS_ext_12km_merit_adj.nc')
hsurf_topo2 = ds['HSURF'].values[:, :]
lat_ = ds["lat"].values
lon_ = ds["lon"].values
ds.close()

# -------------------------------------------------------------------------------
# plot
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

ar = 1.0  # initial aspect ratio for first trial
wi = 15  # height in inches
hi = 10  # width in inches
ncol = 3  # edit here
nrow = 3
axs, cs, ct = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.06, 0.44, 0.99, 0.94
gs1 = gridspec.GridSpec(nrows=2, ncols=3, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.1, hspace=0.15)
left, bottom, right, top = 0.06, 0.1, 0.99, 0.33
gs2 = gridspec.GridSpec(nrows=1, ncols=3, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.1, hspace=0.15)
for i in range(2):
    for j in range(3):
        axs[i, j] = fig.add_subplot(gs1[i, j], projection=rot_pole_crs)
        axs[i, j] = plotcosmo(axs[i, j])

for j in range(3):
    axs[2, j] = fig.add_subplot(gs2[j], projection=rot_pole_crs)
    axs[2, j] = plotcosmo(axs[2, j])

# total smr
levels = MaxNLocator(nbins=10).tick_values(0, 20)
cmap = cmc.davos_r
# cmap = cbr_wet(15)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

sims = ['Control', 'Envelope topography']

for i in range(2):
    sim = sims[i]
    cs[i, 0] = axs[i, 0].pcolormesh(rlon, rlat, all_smr[i], cmap=cmap, norm=norm, shading="auto")
    ct[i, 0] = axs[i, 0].contour(rlon, rlat, all_smr_sms[i], levels=np.linspace(2, 14, 5, endpoint=True), colors='maroon', linewidths=1)
    clabel = axs[i, 0].clabel(ct[i, 0], [2, 5, 8, 11, 14], inline=True, fontsize=13, use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

cax = fig.add_axes([axs[1, 0].get_position().x0, axs[1, 0].get_position().y0 - 0.05, axs[1, 0].get_position().width, 0.02])
cbar = fig.colorbar(cs[1, 0], cax=cax, orientation='horizontal', extend='max')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('mm/day', fontsize=13, labelpad=-0.01)

# plot difference
levels = MaxNLocator(nbins=15).tick_values(-5, 5)
cmap = drywet(25, cmc.vik_r)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

axs[2, 0] = plotcosmo(axs[2, 0])
cs[2, 0] = axs[2, 0].pcolormesh(rlon, rlat, all_smr[1] - all_smr[0], cmap=cmap, clim=(-5, 5), shading="auto")
ct[2, 0] = axs[2, 0].contour(rlon, rlat, all_smr_sms[1] - all_smr_sms[0], levels=np.linspace(-2, 2, 3, endpoint=True), colors='maroon',
                       linewidths=1)
clabel = axs[2, 0].clabel(ct[2, 0], [-2., 0, 2], inline=True, use_clabeltext=True, fontsize=13)
for l in clabel:
    l.set_rotation(0)
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.7)) for txt in clabel]

cax = fig.add_axes([axs[2, 0].get_position().x0, axs[2, 0].get_position().y0 - .05, axs[2, 0].get_position().width, 0.02])
cbar = fig.colorbar(cs[2, 0], cax=cax, orientation='horizontal', extend='both')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('mm/day', fontsize=13, labelpad=-0.01)

# monsoon annual rage
levels = MaxNLocator(nbins=15).tick_values(0, 35)
cmap = cmc.davos_r
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

sims = ['Control', 'Envelope topography']

for i in range(2):
    sim = sims[i]
    axs[i, 1] = plotcosmo(axs[i, 1])
    cs[i, 1] = axs[i, 1].pcolormesh(rlon, rlat, all_rg[i], cmap=cmap, norm=norm, shading="auto")
    ct[i, 1] = axs[i, 1].contour(rlon, rlat, all_rg_sms[i], levels=np.linspace(5, 35, 4, endpoint=True), colors='maroon',
                           linewidths=1)

    clabel = axs[i, 1].clabel(ct[i, 1], [5., 15., 25., 35.], inline=True, fontsize=13, use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

cax = fig.add_axes([axs[1, 1].get_position().x0, axs[1, 1].get_position().y0 - .05, axs[1, 1].get_position().width, 0.02])
cbar = fig.colorbar(cs[1, 1], cax=cax, orientation='horizontal', extend='max')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('mm/day', fontsize=13, labelpad=-0.01)

# plot difference
levels = MaxNLocator(nbins=23).tick_values(-20, 20)
cmap = drywet(25, cmc.vik_r)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

axs[2, 1] = plotcosmo(axs[2, 1])
cs[2, 1] = axs[2, 1].pcolormesh(rlon, rlat, all_rg[1] - all_rg[0], cmap=cmap, clim=(-15, 15), shading="auto")
ct[2, 1] = axs[2, 1].contour(rlon, rlat, all_rg_sms[1] - all_rg_sms[0], levels=np.linspace(-5, 5, 2, endpoint=True), colors='maroon',
                      linewidths=1)
clabel = axs[2, 1].clabel(ct[2, 1], [-5., 5], inline=True, fontsize=13, use_clabeltext=True)
for l in clabel:
    l.set_rotation(0)
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.7)) for txt in clabel]

cax = fig.add_axes([axs[2, 1].get_position().x0, axs[2, 1].get_position().y0 - .05, axs[2, 1].get_position().width, 0.02])
cbar = fig.colorbar(cs[2, 1], cax=cax, orientation='horizontal', extend='both')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('mm/day', fontsize=13, labelpad=-0.01)

levels = MaxNLocator(nbins=20).tick_values(0, 100)
cmap = cmc.davos_r
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

sims = ['Control', 'Envelope topography']

for i in range(2):
    sim = sims[i]
    axs[i, 2] = plotcosmo(axs[i, 2])
    cs[i, 2] = axs[i, 2].pcolormesh(rlon, rlat, all_rt[i], cmap=cmap, norm=norm, shading="auto")
    ct[i, 2] = axs[i, 2].contour(rlon, rlat, all_rt_sms[i], levels=np.linspace(55, 85, 3, endpoint=True), colors='maroon',
                           linewidths=1)
    clabel = axs[i, 2].clabel(ct[i, 2], [55., 70., 85.], inline=True, fontsize=13, use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

cax = fig.add_axes([axs[1, 2].get_position().x0, axs[1, 2].get_position().y0 - .05, axs[1, 2].get_position().width, 0.02])
cbar = fig.colorbar(cs[1, 2], cax=cax, orientation='horizontal', extend='max')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('%', fontsize=13, labelpad=-0.01)

# plot difference
levels = MaxNLocator(nbins=23).tick_values(-20, 20)
cmap = drywet(25, cmc.vik_r)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

axs[2, 2] = plotcosmo(axs[2, 2])
cs[2, 2] = axs[2, 2].pcolormesh(rlon, rlat, all_rt[1] - all_rt[0], cmap=cmap, clim=(-20, 20), shading="auto")
ct[2, 2] = axs[2, 2].contour(rlon, rlat, all_rt_sms[1] - all_rt_sms[0], levels=np.linspace(-20, 20, 5, endpoint=True), colors='maroon',
                      linewidths=1)
clabel = axs[2, 2].clabel(ct[2, 2], [-20., -10, 0, 10, 20], inline=True, fontsize=13, use_clabeltext=True)
for l in clabel:
    l.set_rotation(0)
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.7)) for txt in clabel]

cax = fig.add_axes([axs[2, 2].get_position().x0, axs[2, 2].get_position().y0 - .05, axs[2, 2].get_position().width, 0.02])
cbar = fig.colorbar(cs[2, 2], cax=cax, orientation='horizontal', extend='both')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('%', fontsize=13, labelpad=-0.01)

axs[0, 0].set_title("Total rainfall May to Sep", fontweight='bold', pad=10, fontsize=14)
axs[0, 1].set_title("Annual range", fontweight='bold', pad=10, fontsize=14)
axs[0, 2].set_title("Rainfall ratio", fontweight='bold', pad=10, fontsize=14)

axs[0, 0].text(-0.16, 0.55, 'Control', ha='center', va='center', rotation='vertical',
               transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')
axs[1, 0].text(-0.16, 0.55, 'Envelope Topography', ha='center', va='center', rotation='vertical',
               transform=axs[1, 0].transAxes, fontsize=13, fontweight='bold')
axs[2, 0].text(-0.16, 0.55, 'Envelope - Ctrl', ha='center', va='center', rotation='vertical',
               transform=axs[2, 0].transAxes, fontsize=13, fontweight='bold')

# fig.suptitle('Total Rainfall May to Sep', fontsize=16, fontweight='bold')

# xmin, xmax = axs[1, 2].get_xbound()
# ymin, ymax = axs[1, 2].get_ybound()
# y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol + 0.065
# fig.set_figheight(wi * y2x_ratio)

fig.show()
plotpath = "/project/pr133/rxiang/figure/EAS11/analysis/monsoon/topo2/"
fig.savefig(plotpath + 'pr.png', dpi=500)
plt.close(fig)






