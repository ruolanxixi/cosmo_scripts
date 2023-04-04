# Description: plot the summer climatology: precipitation, water vapor flux, wind at 850 and at 200,
# Load modules
import xarray as xr
from pyproj import CRS, Transformer
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib
import cmcrameri.cm as cmc
import numpy.ma as ma
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_
from plotcosmomap import plotcosmo_notick, pole
import matplotlib.colors as colors
import scipy.ndimage as ndimage

font = {'size': 14}
matplotlib.rc('font', **font)
###############################################################################
# Data
###############################################################################
sims = ['ctrl', 'topo1']
path = "/project/pr133/rxiang/data/cosmo/"

data = {}
labels = ['CTRL11', 'TRED11', 'TRED11 - CTRL11']
lb_rows = ['a', 'b', 'c', 'd', 'e']

g = 9.80665

vars = ['TOT_PREC', 'IUQ', 'IVQ', 'TQF', 'FI', 'U200', 'V200', 'WS200', 'U850', 'V850', 'WS850', 'FI200', 'T200']
# load data
for s in range(len(sims)):
    sim = sims[s]
    data[sim] = {}
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/TOT_PREC/smr/' + f'01-05.TOT_PREC.smr.cpm.nc')
    smr = ds.variables['TOT_PREC'][...]
    smr = np.nanmean(smr, axis=0)
    data[sim]['TOT_PREC'] = smr
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/IVT/smr/' + f'01-05.IVT.smr.cpm.nc')
    iuq = ds.variables['IUQ'][:, :, :]
    smr = np.nanmean(iuq, axis=0)
    data[sim]['IUQ'] = smr
    ivq = ds.variables['IVQ'][:, :, :]
    smr = np.nanmean(ivq, axis=0)
    data[sim]['IVQ'] = smr
    # smr = ds.variables['VIMD'][:, :, :]
    # smr = np.nanmean(smr, axis=0)
    # data[sim]['VIMD'] = smr * 100000
    data[sim]['TQF'] = np.nanmean(np.sqrt(iuq ** 2 + ivq ** 2), axis=0)
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/FI/smr/' + f'01-05.FI.50000.smr.cpm.nc')
    smr = ds.variables['FI'][:, 0, :, :]/g
    smr = np.nanmean(smr, axis=0)
    data[sim]['FI'] = smr
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/U/smr/' + f'01-05.U.20000.smr.cpm.nc')
    u = ds.variables['U'][:, 0, :, :]
    smr = np.nanmean(u, axis=0)
    data[sim]['U200'] = smr
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/V/smr/' + f'01-05.V.20000.smr.cpm.nc')
    v = ds.variables['V'][:, 0, :, :]
    smr = np.nanmean(v, axis=0)
    data[sim]['V200'] = smr
    data[sim]['WS200'] = np.nanmean(np.sqrt(u ** 2 + v ** 2), axis=0)
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/U/smr/' + f'01-05.U.85000.smr.cpm.nc')
    u = ds.variables['U'][:, 0, :, :]
    smr = np.nanmean(u, axis=0)
    data[sim]['U850'] = smr
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/V/smr/' + f'01-05.V.85000.smr.cpm.nc')
    v = ds.variables['V'][:, 0, :, :]
    smr = np.nanmean(v, axis=0)
    data[sim]['V850'] = smr
    data[sim]['WS850'] = np.nanmean(np.sqrt(u ** 2 + v ** 2), axis=0)
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/FI/smr/' + f'01-05.FI.20000.smr.cpm.nc')
    smr = ds.variables['FI'][:, 0, :, :] / g
    smr = np.nanmean(smr, axis=0)
    data[sim]['FI200'] = smr
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/T/smr/' + f'01-05.T.20000.smr.cpm.nc')
    smr = ds.variables['T'][:, 0, :, :]
    smr = np.nanmean(smr, axis=0)
    data[sim]['T200'] = smr

# compute difference
data['diff'] = {}
for v in range(len(vars)):
    var = vars[v]
    data['diff'][var] = data['topo1'][var] - data['ctrl'][var]

# load topo
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

###############################################################################
# %% Plot
###############################################################################
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()
rlon_, rlat_ = np.meshgrid(rlon, rlat)
fig = plt.figure(figsize=(10, 2.95))
gs = gridspec.GridSpec(1, 3, left=0.05, bottom=0.17, right=0.99,
                       top=0.99, hspace=0.28, wspace=0.05,
                       width_ratios=[1, 1, 1])
ncol = 3  # edit here
nrow = 1

axs, cs, ct, topo, q = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                            np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object')

lb = ['(a) precipitation', '(b) IVT', '(c) 500-hPa GPH']
for i in range(ncol):
    label = lb[i]
    axs[0, i] = fig.add_subplot(gs[0, i], projection=rot_pole_crs)
    axs[0, i] = plotcosmo_notick(axs[0, i])
    t = axs[0, i].text(0, 1.03, f'{label}', ha='left', va='bottom',
                       transform=axs[0, i].transAxes, fontsize=14)
    # t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))
    topo[0, i] = axs[0, i].contour(lon_, lat_, hsurf_diff, levels=[500], colors='darkgreen', linewidths=1,
                                   transform=ccrs.PlateCarree())


# --- plot precipitation
levels = MaxNLocator(nbins=11).tick_values(-5, 5)
cmap = drywet(25, cmc.vik_r)
norm = colors.TwoSlopeNorm(vmin=-5., vcenter=0., vmax=5.)
# --
cs[0, 0] = axs[0, 0].pcolormesh(rlon, rlat, data['diff']['TOT_PREC'], cmap=cmap, norm=norm, shading="auto")
# --
cax = fig.add_axes([axs[0, 0].get_position().x0, axs[0, 0].get_position().y0-0.13, axs[0, 0].get_position().width, 0.04])
cbar = fig.colorbar(cs[0, 0], cax=cax, orientation='horizontal', extend='both', ticks=[-4, -2, 0, 2, 4])
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('mm d$^{-1}$', fontsize=13, labelpad=-0.1)

# --- plot water vapor flux stream
levels = MaxNLocator(nbins=21).tick_values(-20, 20)
cmap = drywet(25, cmc.vik_r)
norm = colors.TwoSlopeNorm(vmin=-20., vcenter=0., vmax=20.)
scale = 2000
# --
q[0, 1] = axs[0, 1].quiver(rlon[::15], rlat[::15], data['diff']['IUQ'][::15, ::15], data['diff']['IVQ'][::15, ::15],
                               data['diff']['TQF'][::15, ::15], cmap=cmap, norm=norm, scale=scale, headaxislength=3.5, headwidth=5, minshaft=0)
# --
cax = fig.add_axes([axs[0, 1].get_position().x0, axs[0, 1].get_position().y0-0.13, axs[0, 1].get_position().width, 0.04])
cbar = fig.colorbar(q[0, 1], cax=cax, orientation='horizontal', extend='both')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('kg m$^{-1}$ s$^{-1}$', fontsize=13, labelpad=-0.1)

# --- plot geopotential height 500
cmap = custom_div_cmap(23, cmc.vik)
norm = colors.TwoSlopeNorm(vmin=-24., vcenter=0., vmax=12.)
scale = 2000
# --
cs[0, 2] = axs[0, 2].pcolormesh(rlon, rlat, data['diff']['FI'], cmap=cmap, norm=norm, shading="auto")

# --
cax = fig.add_axes([axs[0, 2].get_position().x0, axs[0, 2].get_position().y0-0.13, axs[0, 2].get_position().width, 0.04])
cbar = fig.colorbar(cs[0, 2], cax=cax, orientation='horizontal', extend='both', ticks=[-24, -18, -12, -6, 0, 3, 6, 9, 12])
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('gpm', fontsize=13, labelpad=-0.1)

# # --- plot wind 850
# levels = MaxNLocator(nbins=15).tick_values(-0.8, 0.8)
# cmap = custom_div_cmap(25, cmc.vik)
# norm = colors.TwoSlopeNorm(vmin=-0.8, vcenter=0., vmax=0.8)
# # --
# q[1, 1] = axs[1, 1].streamplot(rlon, rlat, data['diff']['U850'], data['diff']['V850'], color=data['diff']['WS850'],
#                                    density=1, cmap=cmap, norm=norm)
# # --
# cax = fig.add_axes([axs[1, 1].get_position().x0, axs[1, 1].get_position().y0-0.07, axs[1, 1].get_position().width, 0.025])
# cbar = fig.colorbar(q[1, 1].lines, cax=cax, orientation='horizontal', extend='both', ticks=[-0.8, -0.4, 0, 0.4, 0.8])
# cbar.ax.tick_params(labelsize=13)
# cbar.ax.set_xlabel('m s$^{-1}$', fontsize=13, labelpad=-0.1)

axs[0, 0].text(-0.008, 0.95, '50°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=13)
axs[0, 0].text(-0.008, 0.77, '40°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=13)
axs[0, 0].text(-0.008, 0.59, '30°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=13)
axs[0, 0].text(-0.008, 0.41, '20°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=13)
axs[0, 0].text(-0.008, 0.23, '10°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=13)
axs[0, 0].text(-0.008, 0.05, '0°N', ha='right', va='center', transform=axs[0, 0].transAxes, fontsize=13)

for i in range(3):
    axs[0, i].text(0.12, -0.02, '80°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)
    axs[0, i].text(0.32, -0.02, '100°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)
    axs[0, i].text(0.52, -0.02, '120°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)
    axs[0, i].text(0.72, -0.02, '140°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)
    axs[0, i].text(0.92, -0.02, '160°E', ha='center', va='top', transform=axs[0, i].transAxes, fontsize=13)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/results/"
fig.savefig(plotpath + 'results1.png', dpi=500, transparent=True)
plt.close(fig)







