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


# -------------------------------------------------------------------------------
# read data
#
sims = ['ctrl', 'topo1']
all_u_wtr, all_v_wtr= [], []
all_ws_wtr, all_ws_wtr_sms = [], []
all_u_wtr, all_v_wtr = [], []
all_ws_wtr, all_ws_wtr_sms = [], []

for s in range(len(sims)):
    sim = sims[s]

    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/TWATFLXU/wtr'
    data = Dataset(f'{path}' + '/' + '01-05.TWATFLXU.wtr.cpm.nc')
    wtr = data.variables['TWATFLXU'][:, :, :]
    u_wtr = np.nanmean(wtr, axis=0)
    all_u_wtr.append(u_wtr)
    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/TWATFLXV/wtr'
    data = Dataset(f'{path}' + '/' + '01-05.TWATFLXV.wtr.cpm.nc')
    wtr = data.variables['TWATFLXV'][:, :, :]
    v_wtr = np.nanmean(wtr, axis=0)
    all_v_wtr.append(v_wtr)

    ws_wtr = np.sqrt(u_wtr**2+v_wtr**2)
    all_ws_wtr.append(ws_wtr)
    ws_sms = ndimage.gaussian_filter(np.sqrt(u_wtr**2+v_wtr**2), sigma=10, order=0)
    all_ws_wtr_sms.append(ws_sms)

ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_EAS_ext_12km_merit_unmod_topo.nc')
hsurf_ctrl = ds['HSURF'].values[:, :]
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/old/extpar_EAS_ext_12km_merit_adj.nc')
hsurf_topo1 = ds['HSURF'].values[:, :]
hsurf_diff = ndimage.gaussian_filter(hsurf_ctrl - hsurf_topo1, sigma=3, order=0)
hsurf_ctrl = ndimage.gaussian_filter(hsurf_ctrl, sigma=3, order=0)
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
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                           np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'),\
                           np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object')

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
        topo[i, j] = axs[i, j].contour(lon_, lat_, hsurf_ctrl, levels=[3000], colors='darkgreen', linestyles='dashed',
                                       linewidths=1,
                                       transform=ccrs.PlateCarree())

for j in range(1):
    axs[2, j] = fig.add_subplot(gs2[j], projection=rot_pole_crs)
    axs[2, j] = plotcosmo(axs[2, j])
    topo[2, j] = axs[2, j].contour(lon_, lat_, hsurf_diff, levels=[500], colors='darkgreen', linewidths=1,
                                   transform=ccrs.PlateCarree())
    topo[2, j] = axs[2, j].contour(lon_, lat_, hsurf_ctrl, levels=[3000], colors='darkgreen', linestyles='dashed',
                                   linewidths=1,
                                   transform=ccrs.PlateCarree())

levels1 = MaxNLocator(nbins=25).tick_values(0, 600)
cmap1 = cmc.davos_r
# cmap = cbr_wet(15)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

levels2 = MaxNLocator(nbins=15).tick_values(-2, 2)
cmap2 = drywet(25, cmc.vik_r)
cmap2 = custom_div_cmap(25, cmc.broc_r)
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

# total wtr
sims = ['Control', 'Reduced topography']

for i in range(2):
    sim = sims[i]
    cs[i, 0] = axs[i, 0].pcolormesh(rlon, rlat, all_ws_wtr[i], cmap=cmap1, norm=norm1, shading="auto")
    ct[i, 0] = axs[i, 0].contour(rlon, rlat, all_ws_wtr_sms[i], levels=np.linspace(100, 600, 6, endpoint=True), colors='maroon', linewidths=1)
    clabel = axs[i, 0].clabel(ct[i, 0], levels=np.linspace(100, 600, 6, endpoint=True), inline=True, fontsize=13, use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]
    q[i, 0] = axs[i, 0].quiver(rlon[::40], rlat[::40], all_u_wtr[0][::40, ::40],
                  all_v_wtr[0][::40, ::40], color='black', scale=5000)

qk[0, 0] = axs[0, 0].quiverkey(q[0, 0], 0.895, 1.06, 200, r'$200$', labelpos='E', transform=axs[0, 0].transAxes,
                      fontproperties={'size': 12})
# plot difference
axs[2, 0] = plotcosmo(axs[2, 0])
cs[2, 0] = axs[2, 0].pcolormesh(rlon, rlat, all_ws_wtr[1] - all_ws_wtr[0], cmap=cmap2, clim=(-100, 100), shading="auto")
ct[2, 0] = axs[2, 0].contour(rlon, rlat, all_ws_wtr_sms[1] - all_ws_wtr_sms[0],  levels=[-50, -20, 20, 50], colors='maroon',
                       linewidths=1)
clabel = axs[2, 0].clabel(ct[2, 0], [-50, -20, 20, 50], inline=True, use_clabeltext=True, fontsize=13)
for l in clabel:
    l.set_rotation(0)
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.7)) for txt in clabel]
q[2, 0] = axs[2, 0].quiver(rlon[::40], rlat[::40], all_u_wtr[1][::40, ::40]-all_u_wtr[0][::40, ::40],
                  all_v_wtr[1][::40, ::40]-all_v_wtr[0][::40, ::40], color='black', scale=1000)
qk[2, 0] = axs[2, 0].quiverkey(q[2, 0], 0.92, 1.06, 50, r'$50$', labelpos='E', transform=axs[2, 0].transAxes,
                      fontproperties={'size': 12})

cax = fig.add_axes([axs[1, 0].get_position().x0, axs[1, 0].get_position().y0 - .05, axs[1, 0].get_position().width, 0.02])
cbar = fig.colorbar(cs[1, 0], cax=cax, orientation='horizontal', extend='max', ticks=np.linspace(0, 600, 7, endpoint=True))
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('kg m$^{-1}$ s$^{-1}$', fontsize=13)

cax = fig.add_axes([axs[2, 0].get_position().x0, axs[2, 0].get_position().y0 - .05, axs[2, 0].get_position().width, 0.02])
cbar = fig.colorbar(cs[2, 0], cax=cax, orientation='horizontal', extend='both')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('kg m$^{-1}$ s$^{-1}$', fontsize=13)

axs[0, 0].set_title("Total water flux", fontweight='bold', pad=7, fontsize=13, loc='left')

axs[0, 0].text(-0.15, 0.5, 'CTRL', ha='center', va='center', rotation='vertical',
               transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')
axs[1, 0].text(-0.15, 0.5, 'TRED', ha='center', va='center', rotation='vertical',
               transform=axs[1, 0].transAxes, fontsize=13, fontweight='bold')
axs[2, 0].text(-0.15, 0.5, 'TRED - CTRL', ha='center', va='center', rotation='vertical',
               transform=axs[2, 0].transAxes, fontsize=13, fontweight='bold')

# fig.suptitle('Total Rainfall May to Sep', fontsize=16, fontweight='bold')

# xmin, xmax = axs[1, 2].get_xbound()
# ymin, ymax = axs[1, 2].get_ybound()
# y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol + 0.065
# fig.set_figheight(wi * y2x_ratio)

fig.show()
plotpath = "/project/pr133/rxiang/figure/analysis/EAS11/topo1/wtr/"
fig.savefig(plotpath + 'watflx.png', dpi=500)
plt.close(fig)






