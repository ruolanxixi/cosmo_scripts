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
from plotcosmomap import plotcosmo, pole
import matplotlib.colors as colors
import scipy.ndimage as ndimage

mpl.style.use("classic")
font = {'size': 14}
matplotlib.rc('font', **font)
matplotlib.rcParams['savefig.dpi'] = 300
###############################################################################
# Data
###############################################################################
sims = ['ctrl', 'topo1']
path = "/project/pr133/rxiang/data/cosmo/"

data = {}
labels = ['CTRL11', 'TRED11', 'TRED11 - CTRL11']
lb_rows = ['a', 'b', 'c', 'd', 'e']

g = 9.80665

vars = ['TOT_PREC', 'IUQ', 'IVQ', 'TQF', 'FI', 'U200', 'V200', 'WS200', 'U850', 'V850', 'WS850', 'T200']
yrs = ['01', '02', '03', '04', '05']
# load data
for s in range(len(sims)):
    sim = sims[s]
    data[sim] = {}
    for y in range(len(yrs)):
        yr = yrs[y]
        data[sim][yr] = {}
        ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/szn/TOT_PREC/' + f'{yr}_TOT_PREC_JJA.nc')
        szn = ds.variables['TOT_PREC'][...]
        szn = np.nanmean(szn, axis=0)
        data[sim][yr]['TOT_PREC'] = szn
        ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/szn/IVT/' + f'{yr}_IVT_JJA.nc')
        iuq = ds.variables['IUQ'][:, :, :]
        szn = np.nanmean(iuq, axis=0)
        data[sim][yr]['IUQ'] = szn
        ivq = ds.variables['IVQ'][:, :, :]
        szn = np.nanmean(ivq, axis=0)
        data[sim][yr]['IVQ'] = szn
        # szn = ds.variables['VIMD'][:, :, :]
        # szn = np.nanmean(szn, axis=0)
        # data[sim][yr]['VIMD'] = szn * 100000
        data[sim][yr]['TQF'] = np.nanmean(np.sqrt(iuq ** 2 + ivq ** 2), axis=0)
        ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/szn/FI/' + f'{yr}_FI_50000_JJA.nc')
        szn = ds.variables['FI'][:, 0, :, :]/g
        szn = np.nanmean(szn, axis=0)
        data[sim][yr]['FI'] = szn
        ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/szn/U/' + f'{yr}_U_20000_JJA.nc')
        u = ds.variables['U'][:, 0, :, :]
        szn = np.nanmean(u, axis=0)
        data[sim][yr]['U200'] = szn
        ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/szn/V/' + f'{yr}_V_20000_JJA.nc')
        v = ds.variables['V'][:, 0, :, :]
        szn = np.nanmean(v, axis=0)
        data[sim][yr]['V200'] = szn
        data[sim][yr]['WS200'] = np.nanmean(np.sqrt(u ** 2 + v ** 2), axis=0)
        ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/szn/U/' + f'{yr}_U_85000_JJA.nc')
        u = ds.variables['U'][:, 0, :, :]
        szn = np.nanmean(u, axis=0)
        data[sim][yr]['U850'] = szn
        ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/szn/V/' + f'{yr}_V_85000_JJA.nc')
        v = ds.variables['V'][:, 0, :, :]
        szn = np.nanmean(v, axis=0)
        data[sim][yr]['V850'] = szn
        data[sim][yr]['WS850'] = np.nanmean(np.sqrt(u ** 2 + v ** 2), axis=0)
        # ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/szn/FI/' + f'{yr}_FI_20000_JJA.nc')
        # szn = ds.variables['FI'][:, 0, :, :] / g
        # szn = np.nanmean(szn, axis=0)
        # data[sim][yr]['FI200'] = szn
        ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/szn/T/' + f'{yr}_T_20000_JJA.nc')
        szn = ds.variables['T'][:, 0, :, :]
        szn = np.nanmean(szn, axis=0)
        data[sim][yr]['T200'] = szn

# compute difference
data['diff'] = {}
for y in range(len(yrs)):
    yr = yrs[y]
    data['diff'][yr] = {}
    for v in range(len(vars)):
        var = vars[v]
        data['diff'][yr][var] = data['topo1'][yr][var] - data['ctrl'][yr][var]

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
fig = plt.figure(figsize=(19, 10))
gs = gridspec.GridSpec(5, 5, left=0.05, bottom=0.03, right=0.95,
                       top=0.95, hspace=0.15, wspace=0.18,
                       width_ratios=[1, 1, 1, 1, 1], height_ratios=[1, 1, 1, 1, 1])
ncol = 5  # edit here
nrow = 5

axs, cs, ct, topo, q = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                            np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object')

for i in range(nrow):
    label = lb_rows[i]
    for j in range(ncol):
        axs[i, j] = fig.add_subplot(gs[i, j], projection=rot_pole_crs)
        axs[i, j] = plotcosmo(axs[i, j])
        axs[i, 0].text(-0.13, 1.03, f'({label})', ha='right', va='bottom', transform=axs[i, 0].transAxes, fontsize=14)
        topo[i, j] = axs[i, j].contour(lon_, lat_, hsurf_diff, levels=[500], colors='darkgreen', linewidths=1,
                                       transform=ccrs.PlateCarree())


# --- plot precipitation
levels = MaxNLocator(nbins=11).tick_values(-5, 5)
cmap = drywet(25, cmc.vik_r)
norm = colors.TwoSlopeNorm(vmin=-5., vcenter=0., vmax=5.)
# --
for j in range(ncol):
    yr = yrs[j]
    cs[0, j] = axs[0, j].pcolormesh(rlon, rlat, data['diff'][yr]['TOT_PREC'], cmap=cmap, norm=norm, shading="auto")
# --
cax = fig.add_axes([axs[0, 4].get_position().x1+0.01, axs[0, 4].get_position().y0, 0.015, axs[0, 4].get_position().height])
cbar = fig.colorbar(cs[0, 4], cax=cax, orientation='vertical', extend='both', ticks=[-4, -2, 0, 2, 4])
cbar.ax.tick_params(labelsize=13)
# --
for j in range(ncol):
    label = yrs[j]
    axs[0, j].set_title(f'{label}', fontweight='bold', pad=7, fontsize=14, loc='center')

# --- plot water vapor flux stream
levels = MaxNLocator(nbins=21).tick_values(-20, 20)
cmap = drywet(25, cmc.vik_r)
norm = colors.TwoSlopeNorm(vmin=-20., vcenter=0., vmax=20.)
scale = 2000
# --
for j in range(ncol):
    yr = yrs[j]
    # cs[1, j] = axs[1, j].pcolormesh(rlon, rlat, data[sim][yr]['TQF'], cmap=cmap, norm=norm, shading="auto")
    q[1, j] = axs[1, j].quiver(rlon[::15], rlat[::15], data['diff'][yr]['IUQ'][::15, ::15], data['diff'][yr]['IVQ'][::15, ::15],
                               data['diff'][yr]['TQF'][::15, ::15], cmap=cmap, norm=norm, scale=scale, headaxislength=3.5, headwidth=5, minshaft=0)
# --
cax = fig.add_axes([axs[1, 4].get_position().x1+0.01, axs[1, 4].get_position().y0, 0.015, axs[1, 4].get_position().height])
cbar = fig.colorbar(q[1, 4], cax=cax, orientation='vertical', extend='both')
cbar.ax.tick_params(labelsize=13)

# --- plot geopotential height 500
cmap = custom_div_cmap(23, cmc.vik)
norm = colors.TwoSlopeNorm(vmin=-36., vcenter=0., vmax=24.)
scales = 2000
# --
for j in range(ncol):
    yr = yrs[j]
    cs[2, j] = axs[2, j].pcolormesh(rlon, rlat, data['diff'][yr]['FI'], cmap=cmap, norm=norm, shading="auto")

# --
cax = fig.add_axes([axs[2, 4].get_position().x1+0.01, axs[2, 4].get_position().y0, 0.015, axs[2, 4].get_position().height])
cbar = fig.colorbar(cs[2, 4], cax=cax, orientation='vertical', extend='both', ticks=[-36, -27, -18, -9, 0, 6, 12, 18, 24])
cbar.ax.tick_params(labelsize=13)

# --- plot geopotential height & T 200
# levels = MaxNLocator(nbins=15).tick_values(-1, 1)
# cmap = custom_div_cmap(25, cmc.vik)
# norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0., vmax=1)
# level = np.linspace(11700, 12600, 10, endpoint=True)
# # --
#
# # --
# for j in range(ncol):
#     yr = yrs[j]
#     cs[2, j] = axs[2, j].pcolormesh(rlon, rlat, data['diff'][yr]['T200'], cmap=cmap, norm=norm, shading="auto")
#     # ct[2, j] = axs[2, j].contour(rlon, rlat, data[sim]['diff']['FI200'], levels=level,
#     #                              colors='k', linewidths=.8)
#     # clabel = axs[2, j].clabel(ct[2, j], levels=level, inline=True, fontsize=11,
#     #                           use_clabeltext=True)
#     # for l in clabel:
#     #     l.set_rotation(0)
#     # [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

# --
# cax = fig.add_axes([axs[2, 4].get_position().x1+0.01, axs[2, 4].get_position().y0, 0.015, axs[2, 4].get_position().height])
# cbar = fig.colorbar(cs[2, 4], cax=cax, orientation='vertical', extend='both', ticks=[-1, -0.5, 0, 0.5, 1])
# cbar.ax.tick_params(labelsize=13)

# --- plot wind 200
levels = MaxNLocator(nbins=15).tick_values(-2, 2)
cmap = custom_div_cmap(25, cmc.vik)
norm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2.)
# --
for j in range(ncol):
    yr = yrs[j]
    # cs[1, j] = axs[1, j].pcolormesh(rlon, rlat, data[sim][yr]['TQF'], cmap=cmap, norm=norm, shading="auto")
    q[3, j] = axs[3, j].streamplot(rlon, rlat, data['diff'][yr]['U200'], data['diff'][yr]['V200'], color=data['diff'][yr]['WS200'],
                                   density=1, cmap=cmap, norm=norm)
# --
cax = fig.add_axes([axs[3, 4].get_position().x1+0.01, axs[3, 4].get_position().y0, 0.015, axs[3, 4].get_position().height])
cbar = fig.colorbar(q[3, 4].lines, cax=cax, orientation='vertical', extend='both', ticks=[-2, -1, 0, 1, 2])
cbar.ax.tick_params(labelsize=13)

# --- plot wind 850
levels = MaxNLocator(nbins=15).tick_values(-0.8, 0.8)
cmap = custom_div_cmap(25, cmc.vik)
norm = colors.TwoSlopeNorm(vmin=-0.8, vcenter=0., vmax=0.8)
# --
for j in range(ncol):
    yr = yrs[j]
    # cs[1, j] = axs[1, j].pcolormesh(rlon, rlat, data[sim][yr]['TQF'], cmap=cmap, norm=norm, shading="auto")
    q[4, j] = axs[4, j].streamplot(rlon, rlat, data['diff'][yr]['U850'], data['diff'][yr]['V850'], color=data['diff'][yr]['WS850'],
                                   density=1, cmap=cmap, norm=norm)
# --
cax = fig.add_axes([axs[4, 4].get_position().x1+0.01, axs[4, 4].get_position().y0, 0.015, axs[4, 4].get_position().height])
cbar = fig.colorbar(q[4, 4].lines, cax=cax, orientation='vertical', extend='both', ticks=[-0.8, -0.4, 0, 0.4, 0.8])
cbar.ax.tick_params(labelsize=13)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/results/TRED/"
fig.savefig(plotpath + 'confidence.png', dpi=500)
# plt.close(fig)







