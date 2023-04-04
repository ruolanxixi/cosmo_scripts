# -------------------------------------------------------------------------------
# modules
#
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo04_notick, pole04, colorbar
import cartopy.crs as ccrs
from numpy import inf
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_, wind, hotcold, conv
from pyproj import Transformer
import scipy.ndimage as ndimage
import matplotlib


# -------------------------------------------------------------------------------
# read data
#
sims = ['ctrl', 'topo1']
all_u_smr, all_v_smr= [], []
all_ws_smr, all_ws_smr_sms = [], []
all_vimd_smr, all_vimd_smr_sms = [], []
all_vmfc_smr, all_vmfc_smr_sms = [], []

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
    smr = np.nanmean(smr, axis=0) * 86400
    all_vimd_smr.append(smr)
    vimd_sms = ndimage.gaussian_filter(smr, sigma=2, order=0)
    all_vimd_smr_sms.append(vimd_sms)

    ws_smr = np.sqrt(u_smr**2+v_smr**2)
    all_ws_smr.append(ws_smr)
    ws_sms = ndimage.gaussian_filter(np.sqrt(u_smr**2+v_smr**2), sigma=10, order=0)
    all_ws_smr_sms.append(ws_sms)

    path = f'/project/pr133/rxiang/data/cosmo/EAS04_{sim}/monsoon/VMFC/smr'
    data = Dataset(f'{path}' + '/' + '01-05.VMFC.smr.nc')
    smr = data.variables['VMFC'][:, :, :]
    smr = np.nanmean(smr, axis=0) * 20000
    all_vmfc_smr.append(smr)
    vmfc_sms = ndimage.gaussian_filter(smr, sigma=2, order=0)
    all_vmfc_smr_sms.append(vmfc_sms)

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
wi = 9.5  # height in inches #15
hi = 4.8  # width in inches #10
ncol = 3  # edit here
nrow = 2
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                           np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'),\
                           np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object')

fig = plt.figure(figsize=(wi, hi))

fig = plt.figure(figsize=(wi, hi))
gs1 = gridspec.GridSpec(2, 2, left=0.06, bottom=0.024, right=0.58,
                        top=0.97, hspace=0.01, wspace=0.1, width_ratios=[1, 1], height_ratios=[1, 1])
gs2 = gridspec.GridSpec(2, 1, left=0.663, bottom=0.024, right=0.91,
                        top=0.97, hspace=0.01, wspace=0.1, height_ratios=[1, 1])


levels1 = MaxNLocator(nbins=20).tick_values(-20, 20)
cmap1 = hotcold(21)
norm1 = matplotlib.colors.Normalize(vmin=-20, vmax=20)

levels2 = MaxNLocator(nbins=20).tick_values(-20, 20)
cmap2 = hotcold(21)
norm2 = matplotlib.colors.Normalize(vmin=-20, vmax=20)

# total smr
sims = ['Control', 'Reduced topography']

for i in range(2):
    for j in range(2):
        sim = sims[j]
        axs[i, j] = fig.add_subplot(gs1[i, j], projection=rot_pole_crs)
        axs[i, j] = plotcosmo04_notick(axs[i, j])
    axs[i, 2] = fig.add_subplot(gs2[i, 0], projection=rot_pole_crs)
    axs[i, 2] = plotcosmo04_notick(axs[i, 2])
    topo[i, 2] = axs[i, 2].contour(lon_, lat_, hsurf_diff, levels=[500], colors='darkgreen', linewidths=1,
                                   transform=ccrs.PlateCarree())

np.seterr(divide='ignore', invalid='ignore')
for j in range(2):
    cs[0, j] = axs[0, j].pcolormesh(rlon, rlat, all_vimd_smr_sms[j], cmap=cmap1, norm=norm1, shading="auto")
    # q[0, j] = axs[0, j].quiver(rlon[::30], rlat[::30], all_u_smr[0][::30, ::30],
    #              all_v_smr[0][::30, ::30], color='black', scale=3000)
    cs[1, j] = axs[1, j].pcolormesh(rlon, rlat, all_vmfc_smr_sms[j], cmap=cmap1, norm=norm1, shading="auto")

# qk[0, 1] = axs[0, 1].quiverkey(q[0, 1], 1, 1.06, 200, r'$200$', labelpos='E', transform=axs[0, 1].transAxes,
#                       fontproperties={'size': 12})
# plot difference
cs[0, 2] = axs[0, 2].pcolormesh(rlon, rlat, all_vimd_smr_sms[1] - all_vimd_smr_sms[0], cmap=cmap2, norm=norm2, shading="auto")
# q[0, 2] = axs[0, 2].quiver(rlon[::30], rlat[::30], all_u_smr[1][::30, ::30]-all_u_smr[0][::30, ::30],
#                   all_v_smr[1][::30, ::30]-all_v_smr[0][::30, ::30], color='black', scale=500)
# qk[0, 2] = axs[0, 2].quiverkey(q[0, 2], 1, 1.06, 50, r'$50$', labelpos='E', transform=axs[0, 2].transAxes,
#                       fontproperties={'size': 12})

cs[1, 2] = axs[1, 2].pcolormesh(rlon, rlat, all_vmfc_smr_sms[1] - all_vmfc_smr_sms[0], cmap=cmap2, norm=norm2, shading="auto")

extends = ['max', 'neither', 'neither', 'neither']
tick = np.linspace(-20, 20, 5, endpoint=True)
for i in range(nrow):
    extend = extends[i]
    cax = fig.add_axes(
        [axs[i, 1].get_position().x1 + 0.01, axs[i, 1].get_position().y0, 0.015, axs[i, 1].get_position().height])
    cbar = fig.colorbar(cs[i, 1], cax=cax, orientation='vertical', extend='both', ticks=tick)
    cbar.ax.tick_params(labelsize=13)

tick = np.linspace(-20, 20, 5, endpoint=True)
for i in range(nrow):
    extend = extends[i]
    cax = fig.add_axes(
        [axs[i, 2].get_position().x1 + 0.01, axs[i, 2].get_position().y0, 0.015, axs[i, 2].get_position().height])
    cbar = fig.colorbar(cs[i, 2], cax=cax, orientation='vertical', extend='both', ticks=tick)
    cbar.ax.tick_params(labelsize=13)

for i in range(nrow):
    axs[i, 0].text(-0.01, 0.83, '35°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.57, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.31, '25°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)
    axs[i, 0].text(-0.01, 0.05, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=14)

for j in range(ncol):
    axs[1, j].text(0.04, -0.02, '90°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=14)
    axs[1, j].text(0.46, -0.02, '100°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=14)
    axs[1, j].text(0.88, -0.02, '110°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=14)

lb = [['a', 'b', 'c'], ['d', 'e', 'f']]
for i in range(nrow):
    for j in range(ncol):
        label = lb[i][j]
        t = axs[i, j].text(0.01, 0.987, f'({label})', ha='left', va='top',
                           transform=axs[i, j].transAxes, fontsize=14)
        t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))
        
titles = ['CTRL04', 'TRED04', 'TRED04-CTRL04']
for j in range(ncol):
    title = titles[j]
    axs[0, j].set_title(f'{title}', pad=5, fontsize=14, loc='center')

fig.show()
plotpath = "/project/pr133/rxiang/figure/paper1/results/extreme/"
fig.savefig(plotpath + 'extreme2.png', dpi=500)
plt.close(fig)






