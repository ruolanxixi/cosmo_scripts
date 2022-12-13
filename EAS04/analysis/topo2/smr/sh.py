# -------------------------------------------------------------------------------
# modules
#
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo04, pole04, colorbar
import cartopy.crs as ccrs
from numpy import inf
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_
from pyproj import Transformer
import scipy.ndimage as ndimage
import matplotlib.colors as colors


# -------------------------------------------------------------------------------
# read data
#
sims = ['ctrl', 'topo2']
all_sh, all_sh_sms = [], []

for s in range(len(sims)):
    sim = sims[s]
    path = f'/project/pr133/rxiang/data/cosmo/EAS04_{sim}/monsoon/ASHFL_S/smr'
    data = Dataset(f'{path}' + '/' + '01-05.ASHFL_S.smr.cpm.nc')
    smr = - data.variables['ASHFL_S'][...]
    smr = np.nanmean(smr, axis=0)
    smr_sms = ndimage.gaussian_filter(smr, sigma=10, order=0)
    all_sh.append(smr)
    all_sh_sms.append(smr_sms)

ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_EAS_ext_12km_merit_unmod_topo.nc')
hsurf_ctrl = ds['HSURF'].values[:, :]
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_EAS_ext_12km_merit_env_topo_adj.nc')
hsurf_topo2 = ds['HSURF'].values[:, :]
hsurf_diff = ndimage.gaussian_filter(hsurf_topo2 - hsurf_ctrl, sigma=5, order=0)
lat_ = ds["lat"].values
lon_ = ds["lon"].values
ds.close()

# -------------------------------------------------------------------------------
# plot
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole04()

ar = 1.0  # initial aspect ratio for first trial
wi = 3.5  # height in inches #15
hi = 10  # width in inches #10
ncol = 1  # edit here
nrow = 3
axs, cs, ct, topo, q, qk = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                           np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'),\
                           np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object')

fig = plt.figure(figsize=(wi, hi))

left, bottom, right, top = 0.06, 0.454, 1.08, 0.965
gs1 = gridspec.GridSpec(nrows=2, ncols=1, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.08, hspace=0.15)
left, bottom, right, top = 0.06, 0.105, 1.08, 0.343
gs2 = gridspec.GridSpec(nrows=1, ncols=1, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.08, hspace=0.15)

levels = np.arange(-3500.0, 0, 50.0)
ticks = np.arange(-3500.0, 0, 500.0)
cmap = custom_seq_cmap_(70, cmc.lisbon)
norm = BoundaryNorm(levels, ncolors=cmap.N, extend="max")

for i in range(2):
    for j in range(1):
        axs[i, j] = fig.add_subplot(gs1[i, j], projection=rot_pole_crs)
        axs[i, j] = plotcosmo04(axs[i, j])
        topo[i, j] = axs[i, j].contour(lon_, lat_, hsurf_diff, levels=[100], colors='darkgreen', linewidths=1, transform=ccrs.PlateCarree())

for j in range(1):
    axs[2, j] = fig.add_subplot(gs2[j], projection=rot_pole_crs)
    axs[2, j] = plotcosmo04(axs[2, j])
    topo[2, j] = axs[2, j].contour(lon_, lat_, hsurf_diff, levels=[100], colors='darkgreen', linewidths=1,
                                   transform=ccrs.PlateCarree())

levels1 = np.linspace(0, 80, 17, endpoint=True)
cmap1 = cmc.roma_r
# cmap = cbr_wet(15)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

levels2 = MaxNLocator(nbins=15).tick_values(-50, 50)
cmap2 = custom_div_cmap(25, cmc.vik)
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

# total smr
sims = ['Control', 'Reduced topography']

for i in range(2):
    sim = sims[i]
    cs[i, 0] = axs[i, 0].pcolormesh(rlon, rlat, all_sh[i], cmap=cmap1, norm=norm1, shading="auto")
    ct[i, 0] = axs[i, 0].contour(rlon, rlat, all_sh_sms[i], levels=np.linspace(0, 80, 5, endpoint=True), colors='k', linewidths=.8)
    clabel = axs[i, 0].clabel(ct[i, 0], levels=np.linspace(0, 80, 5, endpoint=True), inline=True, fontsize=11, use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

# plot difference
axs[2, 0] = plotcosmo04(axs[2, 0])
cs[2, 0] = axs[2, 0].pcolormesh(rlon, rlat, all_sh[1] - all_sh[0], cmap=cmap2, clim=(-15, 15), shading="auto")
# ct[2, 0] = axs[2, 0].contour(rlon, rlat, all_sh_sms[1] - all_sh_sms[0], [-5, 5], colors='k',
#                        linewidths=.8)
# clabel = axs[2, 0].clabel(ct[2, 0], [-5, 5], inline=True, use_clabeltext=True, fontsize=11)
# for l in clabel:
#     l.set_rotation(0)
# [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.7)) for txt in clabel]

cax = fig.add_axes([axs[1, 0].get_position().x0, axs[1, 0].get_position().y0 - .05, axs[1, 0].get_position().width, 0.02])
cbar = fig.colorbar(cs[1, 0], cax=cax, orientation='horizontal', extend='both', ticks=[0, 20, 40, 60, 80])
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('$W/m^2$', fontsize=13)

cax = fig.add_axes([axs[2, 0].get_position().x0, axs[2, 0].get_position().y0 - .05, axs[2, 0].get_position().width, 0.02])
cbar = fig.colorbar(cs[2, 0], cax=cax, orientation='horizontal', extend='both', ticks=[-12, -6, 0, 6, 12])
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('$W/m^2$', fontsize=13)

axs[0, 0].set_title("Surface sensible heat", fontweight='bold', pad=7, fontsize=13, loc='left')

# axs[0, 0].text(0, 1.01, '@ 500 hPa', ha='left', va='bottom',
#                transform=axs[0, 0].transAxes, fontsize=11)
axs[0, 0].text(-0.23, 0.5, 'CTRL', ha='right', va='center', rotation='vertical',
               transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')
axs[1, 0].text(-0.23, 0.5, 'TENV', ha='right', va='center', rotation='vertical',
               transform=axs[1, 0].transAxes, fontsize=13, fontweight='bold')
axs[2, 0].text(-0.23, 0.5, 'TENV - CTRL', ha='right', va='center', rotation='vertical',
               transform=axs[2, 0].transAxes, fontsize=13, fontweight='bold')

# fig.suptitle('Total Rainfall May to Sep', fontsize=16, fontweight='bold')

# xmin, xmax = axs[1, 2].get_xbound()
# ymin, ymax = axs[1, 2].get_ybound()
# y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol + 0.065
# fig.set_figheight(wi * y2x_ratio)

fig.show()
plotpath = "/project/pr133/rxiang/figure/analysis/EAS04/topo2/smr/"
fig.savefig(plotpath + 'sh.png', dpi=500)
plt.close(fig)






