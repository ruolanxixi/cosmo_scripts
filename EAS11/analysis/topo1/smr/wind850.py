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

levels1 = MaxNLocator(nbins=14).tick_values(0, 14)
cmap1 = wind(14, cmc.batlowW_r)
# cmap = cbr_wet(15)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

levels2 = MaxNLocator(nbins=15).tick_values(-2, 2)
cmap2 = drywet(25, cmc.vik_r)
cmap2 = custom_div_cmap(25, cmc.broc_r)
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

qk[0, 0] = axs[0, 0].quiverkey(q[0, 0], 0.9, 1.06, 5, r'$5\ \frac{m}{s}$', labelpos='E', transform=axs[0, 0].transAxes,
                      fontproperties={'size': 12})
# plot difference
axs[2, 0] = plotcosmo(axs[2, 0])
cs[2, 0] = axs[2, 0].pcolormesh(rlon, rlat, all_ws_smr[1] - all_ws_smr[0], cmap=cmap2, clim=(-2, 2), shading="auto")
ct[2, 0] = axs[2, 0].contour(rlon, rlat, all_ws_smr_sms[1] - all_ws_smr_sms[0],  levels=[-2, -1, 1, 2], colors='maroon',
                       linewidths=1)
clabel = axs[2, 0].clabel(ct[2, 0], [-2, -1, 1, 2], inline=True, use_clabeltext=True, fontsize=13)
for l in clabel:
    l.set_rotation(0)
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.7)) for txt in clabel]
q[2, 0] = axs[2, 0].quiver(rlon[::40], rlat[::40], all_u_smr[1][::40, ::40]-all_u_smr[0][::40, ::40],
                  all_v_smr[1][::40, ::40]-all_v_smr[0][::40, ::40], color='black', scale=30)
qk[2, 0] = axs[2, 0].quiverkey(q[2, 0], 0.9, 1.06, 1, r'$1\ \frac{m}{s}$', labelpos='E', transform=axs[2, 0].transAxes,
                      fontproperties={'size': 12})

cax = fig.add_axes([axs[1, 0].get_position().x0, axs[1, 0].get_position().y0 - .05, axs[1, 0].get_position().width, 0.02])
cbar = fig.colorbar(cs[1, 0], cax=cax, orientation='horizontal', extend='max', ticks=np.linspace(0, 20, 11, endpoint=True))
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('m/s', fontsize=13)

cax = fig.add_axes([axs[2, 0].get_position().x0, axs[2, 0].get_position().y0 - .05, axs[2, 0].get_position().width, 0.02])
cbar = fig.colorbar(cs[2, 0], cax=cax, orientation='horizontal', extend='both', ticks=np.linspace(-5, 5, 11, endpoint=True))
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('m/s', fontsize=13)

axs[0, 0].set_title("800 hPa Wind", fontweight='bold', pad=7, fontsize=13, loc='left')

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
plotpath = "/project/pr133/rxiang/figure/analysis/EAS11/topo1/smr/"
fig.savefig(plotpath + 'wind850.png', dpi=500)
plt.close(fig)






