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
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_, wind, hotcold, conv
from pyproj import Transformer
import scipy.ndimage as ndimage


# -------------------------------------------------------------------------------
# read data
#
sims = ['ctrl', 'topo1']
all_u_smr, all_v_smr= [], []
all_ws_smr, all_ws_smr_sms = [], []
all_vimd_smr, all_vimd_smr_sms = [], []

for s in range(len(sims)):
    sim = sims[s]

    path = f'/project/pr133/rxiang/data/cosmo/EAS04_{sim}/monsoon/IVT/smr'
    data = Dataset(f'{path}' + '/' + '01-05.IVT.smr.nc')
    smr = data.variables['IUQ'][:, :, :]
    u_smr = np.nanmean(smr, axis=0)
    all_u_smr.append(u_smr)
    smr = data.variables['IVQ'][:, :, :]
    v_smr = np.nanmean(smr, axis=0)
    all_v_smr.append(v_smr)
    smr = data.variables['VIMD'][:, :, :]
    smr = np.nanmean(smr, axis=0) * 10 ** 4
    all_vimd_smr.append(smr)
    vimd_sms = ndimage.gaussian_filter(smr, sigma=4, order=0)
    all_vimd_smr_sms.append(vimd_sms)

    ws_smr = np.sqrt(u_smr**2+v_smr**2)
    all_ws_smr.append(ws_smr)
    ws_sms = ndimage.gaussian_filter(np.sqrt(u_smr**2+v_smr**2), sigma=10, order=0)
    all_ws_smr_sms.append(ws_sms)

ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_BECCY_4.4km_merit_unmod_topo.nc')
hsurf_ctrl = ds['HSURF'].values[:, :]
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_BECCY_4.4km_merit_reduced_topo_adj.nc')
hsurf_topo1 = ds['HSURF'].values[:, :]
hsurf_diff = ndimage.gaussian_filter(hsurf_ctrl - hsurf_topo1, sigma=5, order=0)
hsurf_ctrl = ndimage.gaussian_filter(hsurf_ctrl, sigma=3, order=0)
lat_ = ds["lat"].values
lon_ = ds["lon"].values
ds.close()

# %% -------------------------------------------------------------------------------
# plot
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole04()

ar = 1.0  # initial aspect ratio for first trial
wi = 3.5  # height in inches #15
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
        axs[i, j] = plotcosmo04(axs[i, j])
        topo[i, j] = axs[i, j].contour(lon_, lat_, hsurf_diff, levels=[500], colors='darkgreen', linewidths=1, transform=ccrs.PlateCarree())

for j in range(1):
    axs[2, j] = fig.add_subplot(gs2[j], projection=rot_pole_crs)
    axs[2, j] = plotcosmo04(axs[2, j])
    topo[2, j] = axs[2, j].contour(lon_, lat_, hsurf_diff, levels=[500], colors='darkgreen', linewidths=1,
                                   transform=ccrs.PlateCarree())


levels1 = MaxNLocator(nbins=15).tick_values(-5, 5)
cmap1 = cmc.roma_r
cmap1 = drywet(21, cmc.vik_r)
cmap1 = custom_div_cmap(25, cmc.roma_r)
cmap1 = hotcold(15)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

levels2 = MaxNLocator(nbins=15).tick_values(-100, 100)
cmap2 = drywet(25, cmc.vik_r)
cmap2 = hotcold(15)
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

# total smr
sims = ['Control', 'Reduced topography']

for i in range(2):
    sim = sims[i]
    cs[i, 0] = axs[i, 0].pcolormesh(rlon, rlat, all_vimd_smr_sms[i], cmap=cmap1, clim=(-5, 5), shading="auto")
    # ct[i, 0] = axs[i, 0].contour(rlon, rlat, all_ws_smr_sms[i], levels=np.linspace(100, 400, 4, endpoint=True), colors='maroon', linewidths=1)
    # clabel = axs[i, 0].clabel(ct[i, 0], levels=np.linspace(100, 400, 4, endpoint=True), inline=True, fontsize=13, use_clabeltext=True)
    # for l in clabel:
    #     l.set_rotation(0)
    # [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]
    q[i, 0] = axs[i, 0].quiver(rlon[::30], rlat[::30], all_u_smr[0][::30, ::30],
                  all_v_smr[0][::30, ::30], color='black', scale=3000)

qk[0, 0] = axs[0, 0].quiverkey(q[0, 0], 0.83, 1.06, 200, r'$200$', labelpos='E', transform=axs[0, 0].transAxes,
                      fontproperties={'size': 12})
# plot difference
axs[2, 0] = plotcosmo04(axs[2, 0])
cs[2, 0] = axs[2, 0].pcolormesh(rlon, rlat, all_vimd_smr_sms[1] - all_vimd_smr_sms[0], cmap=cmap2, clim=(-5, 5), shading="auto")
# ct[2, 0] = axs[2, 0].contour(rlon, rlat, all_ws_smr_sms[1] - all_ws_smr_sms[0],  levels=np.linspace(-5, 5, 5, endpoint=True), colors='maroon',
#                        linewidths=1)
# clabel = axs[2, 0].clabel(ct[2, 0], np.linspace(-100, 100, 5, endpoint=True), inline=True, use_clabeltext=True, fontsize=13)
# for l in clabel:
#     l.set_rotation(0)
# [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.7)) for txt in clabel]
q[2, 0] = axs[2, 0].quiver(rlon[::30], rlat[::30], all_u_smr[1][::30, ::30]-all_u_smr[0][::30, ::30],
                  all_v_smr[1][::30, ::30]-all_v_smr[0][::30, ::30], color='black', scale=500)
qk[2, 0] = axs[2, 0].quiverkey(q[2, 0], 0.87, 1.06, 50, r'$30$', labelpos='E', transform=axs[2, 0].transAxes,
                      fontproperties={'size': 12})

cax = fig.add_axes([axs[1, 0].get_position().x0, axs[1, 0].get_position().y0 - .05, axs[1, 0].get_position().width, 0.02])
cbar = fig.colorbar(cs[1, 0], cax=cax, orientation='horizontal', extend='both')
cbar.ax.set_xlabel('kg m$^{-1}$ s$^{-1}$', fontsize=13)

cax = fig.add_axes([axs[2, 0].get_position().x0, axs[2, 0].get_position().y0 - .05, axs[2, 0].get_position().width, 0.02])
cbar = fig.colorbar(cs[2, 0], cax=cax, orientation='horizontal', extend='both', ticks=np.linspace(-10, 10, 5, endpoint=True))
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('kg m$^{-1}$ s$^{-1}$', fontsize=13)

# axs[0, 0].set_title("Integrated water vapour flux", fontweight='bold', pad=7, fontsize=13, loc='left')

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
# plotpath = "/project/pr133/rxiang/figure/analysis/EAS04/topo2/smr/"
# fig.savefig(plotpath + 'watflx.png', dpi=500)
# plt.close(fig)






