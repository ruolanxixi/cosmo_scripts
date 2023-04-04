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
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_, tmpmap, tmpdiff
from pyproj import Transformer
import scipy.ndimage as ndimage
import matplotlib.colors as colors


# -------------------------------------------------------------------------------
# read data
#
sims = ['ctrl', 'topo1']
all_t2m_wtr, all_t2m_wtr_sms = [], []
all_slp_wtr, all_slp_wtr_sms = [], []

for s in range(len(sims)):
    sim = sims[s]

    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/T_2M/wtr'
    data = Dataset(f'{path}' + '/' + '01-05.T_2M.wtr.cpm.nc')
    wtr = data.variables['T_2M'][:, :, :] - 273.15
    t2m_wtr = np.nanmean(wtr, axis=0)
    t2m_wtr_sms = ndimage.gaussian_filter(t2m_wtr, sigma=10, order=0)
    all_t2m_wtr.append(t2m_wtr)
    all_t2m_wtr_sms.append(t2m_wtr_sms)

    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/PMSL/wtr'
    data = Dataset(f'{path}' + '/' + '01-05.PMSL.wtr.cpm.nc')
    wtr = data.variables['PMSL'][:, :, :]/100
    slp_wtr = np.nanmean(wtr, axis=0)
    slp_wtr_sms = ndimage.gaussian_filter(slp_wtr, sigma=10, order=0)
    all_slp_wtr.append(slp_wtr)
    all_slp_wtr_sms.append(slp_wtr_sms)

ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_EAS_ext_12km_merit_unmod_topo.nc')
hsurf_ctrl = ds['HSURF'].values[:, :]
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/old/extpar_EAS_ext_12km_merit_adj.nc')
hsurf_topo1 = ds['HSURF'].values[:, :]
hsurf_diff = ndimage.gaussian_filter(hsurf_ctrl - hsurf_topo1, sigma=3, order=0)
lat_ = ds["lat"].values
lon_ = ds["lon"].values
ds.close()

# -------------------------------------------------------------------------------
# plot
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

ar = 1.0  # initial aspect ratio for first trial
wi = 5.1  # height in inches #15
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
        axs[i, j] = plotcosmo(axs[i, j])
        topo[i, j] = axs[i, j].contour(lon_, lat_, hsurf_diff, levels=[500], colors='darkgreen', linewidths=1, transform=ccrs.PlateCarree())

for j in range(1):
    axs[2, j] = fig.add_subplot(gs2[j], projection=rot_pole_crs)
    axs[2, j] = plotcosmo(axs[2, j])
    topo[2, j] = axs[2, j].contour(lon_, lat_, hsurf_diff, levels=[500], colors='darkgreen', linewidths=1,
                                   transform=ccrs.PlateCarree())

levels1 = MaxNLocator(nbins=30).tick_values(-30, 30)
cmap1 = cmc.roma_r
cmap1 = tmpmap(35, cmc.roma_r, 5, 30)
cmap1 = cmc.roma_r
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

levels2 = MaxNLocator(nbins=29).tick_values(-4, 10)
cmap2 = custom_div_cmap(23, cmc.vik)
cmap2 = tmpdiff(30, cmc.vik, 8, 21)
# divnorm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=10.)
divnorm = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)


# total wtr
sims = ['Control', 'Reduced topography']

for i in range(2):
    sim = sims[i]
    cs[i, 0] = axs[i, 0].pcolormesh(rlon, rlat, all_t2m_wtr[i], cmap=cmap1, norm=norm1, shading="auto")
    ct[i, 0] = axs[i, 0].contour(rlon, rlat, all_slp_wtr_sms[i], levels=np.linspace(1000, 1030, 7, endpoint=True), colors='k', linewidths=.8)
    clabel = axs[i, 0].clabel(ct[i, 0], levels=np.linspace(1000, 1030, 7, endpoint=True), inline=True, fontsize=11, use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

# plot difference
axs[2, 0] = plotcosmo(axs[2, 0])
cs[2, 0] = axs[2, 0].pcolormesh(rlon, rlat, all_t2m_wtr[1] - all_t2m_wtr[0], cmap=cmap2, clim=(-4, 10), shading="auto")
ct[2, 0] = axs[2, 0].contour(rlon, rlat, all_slp_wtr_sms[1] - all_slp_wtr_sms[0], [-2, -1, 1, 2], colors='k',
                       linewidths=.8)
clabel = axs[2, 0].clabel(ct[2, 0], [-2, -1, 1, 2], inline=True, use_clabeltext=True, fontsize=11)
for l in clabel:
    l.set_rotation(0)
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.7)) for txt in clabel]

cax = fig.add_axes([axs[1, 0].get_position().x0, axs[1, 0].get_position().y0 - .05, axs[1, 0].get_position().width, 0.02])
cbar = fig.colorbar(cs[1, 0], cax=cax, orientation='horizontal', extend='both', ticks=np.linspace(-30, 30, 7, endpoint=True))
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('$^{o}C$', fontsize=13)

cax = fig.add_axes([axs[2, 0].get_position().x0, axs[2, 0].get_position().y0 - .05, axs[2, 0].get_position().width, 0.02])
cbar = fig.colorbar(cs[2, 0], cax=cax, orientation='horizontal', extend='both', ticks=np.linspace(-4, 10, 8, endpoint=True))
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('$^{o}C$', fontsize=13)

axs[0, 0].set_title("Surface temperature & SLP", fontweight='bold', pad=7, fontsize=13, loc='left')

# axs[0, 0].text(0, 1.01, '@ 500 hPa', ha='left', va='bottom',
  #             transform=axs[0, 0].transAxes, fontsize=11)
axs[0, 0].text(-0.15, 0.5, 'CTRL', ha='center', va='center', rotation='vertical',
               transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')
axs[1, 0].text(-0.15, 0.5, 'TRED', ha='center', va='center', rotation='vertical',
               transform=axs[1, 0].transAxes, fontsize=13, fontweight='bold')
axs[2, 0].text(-0.15, 0.5, 'TRED - CTRL', ha='center', va='center', rotation='vertical',
               transform=axs[2, 0].transAxes, fontsize=13, fontweight='bold')

# fig.suptitle('Total Rainfall Nov to Mar', fontsize=16, fontweight='bold')

# xmin, xmax = axs[1, 2].get_xbound()
# ymin, ymax = axs[1, 2].get_ybound()
# y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol + 0.065
# fig.set_figheight(wi * y2x_ratio)

fig.show()
plotpath = "/project/pr133/rxiang/figure/analysis/EAS11/topo1/wtr/"
fig.savefig(plotpath + 't2m.png', dpi=500)
plt.close(fig)






