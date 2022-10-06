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
all_u_smr, all_v_smr= [], []
all_ws_smr, all_ws_smr_sms = [], []
all_u_wtr, all_v_wtr = [], []
all_ws_wtr, all_ws_wtr_sms = [], []

for s in range(len(sims)):
    sim = sims[s]

    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/U/smr'
    data = Dataset(f'{path}' + '/' + '01-05.U.85000.smr.cpm.nc')
    smr = data.variables['U'][:, 0, :, :]
    u_smr = np.nanmean(smr, axis=0)
    all_u_smr.append(u_smr)
    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/V/smr'
    data = Dataset(f'{path}' + '/' + '01-05.V.85000.smr.cpm.nc')
    smr = data.variables['V'][:, 0, :, :]
    v_smr = np.nanmean(smr, axis=0)
    all_v_smr.append(v_smr)

    ws_smr = np.sqrt(u_smr**2+v_smr**2)
    all_ws_smr.append(ws_smr)
    ws_sms = ndimage.gaussian_filter(np.sqrt(u_smr**2+v_smr**2), sigma=10, order=0)
    all_ws_smr_sms.append(ws_sms)

    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/U/wtr'
    data = Dataset(f'{path}' + '/' + '01-05.U.85000.wtr.cpm.nc')
    wtr = data.variables['U'][:, 0, :, :]
    u_wtr = np.nanmean(wtr, axis=0)
    all_u_wtr.append(u_wtr)
    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/V/wtr'
    data = Dataset(f'{path}' + '/' + '01-05.V.85000.wtr.cpm.nc')
    wtr = data.variables['V'][:, 0, :, :]
    v_wtr = np.nanmean(wtr, axis=0)
    all_v_wtr.append(v_wtr)

    ws_wtr = np.sqrt(u_wtr ** 2 + v_wtr ** 2)
    all_ws_wtr.append(ws_wtr)
    ws_sms = ndimage.gaussian_filter(np.sqrt(u_wtr ** 2 + v_wtr ** 2), sigma=10, order=0)
    all_ws_wtr_sms.append(ws_sms)

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
wi = 11  # height in inches #15
hi = 10  # width in inches #10
ncol = 2  # edit here
nrow = 3
axs, cs, ct, topo, q, qk = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                           np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'),\
                           np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object')

fig = plt.figure(figsize=(wi, hi))

left, bottom, right, top = 0.06, 0.44, 0.99, 0.94
gs1 = gridspec.GridSpec(nrows=2, ncols=2, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.1, hspace=0.15)
left, bottom, right, top = 0.06, 0.1, 0.99, 0.33
gs2 = gridspec.GridSpec(nrows=1, ncols=2, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.1, hspace=0.15)

levels = np.arange(-3500.0, 0, 50.0)
ticks = np.arange(-3500.0, 0, 500.0)
cmap = custom_seq_cmap_(70, cmc.lisbon)
norm = BoundaryNorm(levels, ncolors=cmap.N, extend="max")

for i in range(2):
    for j in range(2):
        axs[i, j] = fig.add_subplot(gs1[i, j], projection=rot_pole_crs)
        axs[i, j] = plotcosmo(axs[i, j])
        topo[i, j] = axs[i, j].contour(lon_, lat_, hsurf_diff, levels=[500], colors='darkgreen', linewidths=1, transform=ccrs.PlateCarree())

for j in range(2):
    axs[2, j] = fig.add_subplot(gs2[j], projection=rot_pole_crs)
    axs[2, j] = plotcosmo(axs[2, j])
    topo[2, j] = axs[2, j].contour(lon_, lat_, hsurf_diff, levels=[500], colors='darkgreen', linewidths=1,
                                   transform=ccrs.PlateCarree())

levels1 = MaxNLocator(nbins=20).tick_values(0, 20)
cmap1 = cmc.davos_r
# cmap = cbr_wet(15)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

levels2 = MaxNLocator(nbins=15).tick_values(-2, 2)
cmap2 = drywet(25, cmc.vik_r)
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

# total smr
sims = ['Control', 'Reduced topography']

for i in range(2):
    sim = sims[i]
    cs[i, 0] = axs[i, 0].pcolormesh(rlon, rlat, all_ws_smr[i], cmap=cmap1, norm=norm1, shading="auto")
    ct[i, 0] = axs[i, 0].contour(rlon, rlat, all_ws_smr_sms[i], levels=np.linspace(2, 14, 5, endpoint=True), colors='maroon', linewidths=1)
    clabel = axs[i, 0].clabel(ct[i, 0], [2, 5, 8, 11, 14], inline=True, fontsize=13, use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]
    q[i, 0] = axs[i, 0].quiver(rlon[::40], rlat[::40], all_u_smr[0][::40, ::40],
                  all_v_smr[0][::40, ::40], color='black', scale=150)

qk[0, 0] = axs[0, 0].quiverkey(q[0, 0], 0.92, 0.03, 2, r'$2\ \frac{m}{s}$', labelpos='E', transform=axs[0, 0].transAxes,
                      fontproperties={'size': 9})
# plot difference
axs[2, 0] = plotcosmo(axs[2, 0])
cs[2, 0] = axs[2, 0].pcolormesh(rlon, rlat, all_ws_smr[1] - all_ws_smr[0], cmap=cmap2, clim=(-3, 3), shading="auto")
ct[2, 0] = axs[2, 0].contour(rlon, rlat, all_ws_smr_sms[1] - all_ws_smr_sms[0],  levels=[-2, -1, 1, 2], colors='maroon',
                       linewidths=1)
clabel = axs[2, 0].clabel(ct[2, 0], [-2, -1, 1, 2], inline=True, use_clabeltext=True, fontsize=13)
for l in clabel:
    l.set_rotation(0)
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.7)) for txt in clabel]
q[2, 0] = axs[2, 0].quiver(rlon[::40], rlat[::40], all_u_smr[1][::40, ::40]-all_u_smr[0][::40, ::40],
                  all_v_smr[1][::40, ::40]-all_v_smr[0][::40, ::40], color='black', scale=30)
qk[2, 0] = axs[2, 0].quiverkey(q[2, 0], 0.92, 0.03, 1, r'$1\ \frac{m}{s}$', labelpos='E', transform=axs[2, 0].transAxes,
                      fontproperties={'size': 9})

for i in range(2):
    sim = sims[i]
    cs[i, 1] = axs[i, 1].pcolormesh(rlon, rlat, all_ws_wtr[i], cmap=cmap1, norm=norm1, shading="auto")
    ct[i, 1] = axs[i, 1].contour(rlon, rlat, all_ws_wtr_sms[i], levels=np.linspace(2, 14, 5, endpoint=True), colors='maroon', linewidths=1)
    clabel = axs[i, 1].clabel(ct[i, 1], [2, 5, 8, 11, 14], inline=True, fontsize=13, use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]
    q[i, 1] = axs[i, 1].quiver(rlon[::40], rlat[::40], all_u_wtr[0][::40, ::40],
                  all_v_wtr[0][::40, ::40], color='black', scale=150)

qk[0, 1] = axs[0, 1].quiverkey(q[0, 1], 0.92, 0.03, 2, r'$2\ \frac{m}{s}$', labelpos='E', transform=axs[0, 1].transAxes,
                      fontproperties={'size': 9})

cax = fig.add_axes([(axs[1, 0].get_position().x0 + axs[0, 0].get_position().x1)/2, axs[1, 0].get_position().y0 - .05,
                    ((axs[1, 1].get_position().x0 + axs[1, 1].get_position().x1)/2 -
                     (axs[1, 0].get_position().x0 + axs[1, 0].get_position().x1)/2), 0.02])
cbar = fig.colorbar(cs[1, 1], cax=cax, orientation='horizontal', extend='max', ticks=np.linspace(0, 20, 11, endpoint=True))
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('m/s', fontsize=13, labelpad=-0.01)

# plot difference
axs[2, 1] = plotcosmo(axs[2, 1])
cs[2, 1] = axs[2, 1].pcolormesh(rlon, rlat, all_ws_wtr[1] - all_ws_wtr[0], cmap=cmap2, clim=(-3, 3), shading="auto")
ct[2, 1] = axs[2, 1].contour(rlon, rlat, all_ws_wtr_sms[1] - all_ws_wtr_sms[0],  levels=[-2, -1, 1, 2], colors='maroon',
                       linewidths=1)
clabel = axs[2, 1].clabel(ct[2, 1], [-2, -1, 1, 2], inline=True, use_clabeltext=True, fontsize=13)
for l in clabel:
    l.set_rotation(0)
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.7)) for txt in clabel]
q[2, 1] = axs[2, 1].quiver(rlon[::40], rlat[::40], all_u_wtr[1][::40, ::40]-all_u_wtr[0][::40, ::40],
                  all_v_wtr[1][::40, ::40]-all_v_wtr[0][::40, ::40], color='black', scale=30)
qk[2, 1] = axs[2, 1].quiverkey(q[2, 1], 0.92, 0.03, 1, r'$1\ \frac{m}{s}$', labelpos='E', transform=axs[2, 1].transAxes,
                      fontproperties={'size': 9})

cax = fig.add_axes([(axs[2, 0].get_position().x0 + axs[2, 0].get_position().x1)/2, axs[2, 0].get_position().y0 - .05,
                    ((axs[2, 1].get_position().x0 + axs[2, 1].get_position().x1)/2) -
                    (axs[2, 0].get_position().x0 + axs[2, 0].get_position().x1)/2, 0.02])
cbar = fig.colorbar(cs[2, 1], cax=cax, orientation='horizontal', extend='both', ticks=np.linspace(-5, 5, 11, endpoint=True))
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('m/s', fontsize=13, labelpad=-0.01)

axs[0, 0].set_title("May to Sep", fontweight='bold', pad=10, fontsize=14)
axs[0, 1].set_title("Nov to Mar", fontweight='bold', pad=10, fontsize=14)

axs[0, 0].text(-0.16, 0.55, 'Control', ha='center', va='center', rotation='vertical',
               transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')
axs[1, 0].text(-0.16, 0.55, 'Reduced Topography', ha='center', va='center', rotation='vertical',
               transform=axs[1, 0].transAxes, fontsize=13, fontweight='bold')
axs[2, 0].text(-0.16, 0.55, 'Reduced - Ctrl', ha='center', va='center', rotation='vertical',
               transform=axs[2, 0].transAxes, fontsize=13, fontweight='bold')

# fig.suptitle('Total Rainfall May to Sep', fontsize=16, fontweight='bold')

# xmin, xmax = axs[1, 2].get_xbound()
# ymin, ymax = axs[1, 2].get_ybound()
# y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol + 0.065
# fig.set_figheight(wi * y2x_ratio)

fig.show()
plotpath = "/project/pr133/rxiang/figure/analysis/EAS11/topo1/"
fig.savefig(plotpath + 'wind.png', dpi=500)
plt.close(fig)






