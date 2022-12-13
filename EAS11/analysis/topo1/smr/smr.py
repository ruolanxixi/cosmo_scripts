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
sims = ['ctrl', 'topo1', 'diff']
fig = plt.figure(figsize=(12.5, 9.8))
gs1 = gridspec.GridSpec(5, 2, left=0.05, bottom=0.03, right=0.575,
                       top=0.95, hspace=0.15, wspace=0.18,
                       width_ratios=[1, 1], height_ratios=[1, 1, 1, 1, 1])
gs2 = gridspec.GridSpec(5, 1, left=0.682, bottom=0.03, right=0.925,
                       top=0.95, hspace=0.15, wspace=0.18)
ncol = 3  # edit here
nrow = 5

axs, cs, ct, topo, q = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                            np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object')

for i in range(nrow):
    label = lb_rows[i]
    for j in range(ncol-1):
        axs[i, j] = fig.add_subplot(gs1[i, j], projection=rot_pole_crs)
        axs[i, j] = plotcosmo(axs[i, j])
    axs[i, 2] = fig.add_subplot(gs2[i, 0], projection=rot_pole_crs)
    axs[i, 2] = plotcosmo(axs[i, 2])
    axs[i, 0].text(-0.13, 1.03, f'({label})', ha='right', va='bottom', transform=axs[i, 0].transAxes, fontsize=14)

# plot topo_diff
for i in range(nrow):
    axs[i, 2] = plotcosmo(axs[i, 2])
    topo[i, 2] = axs[i, 2].contour(lon_, lat_, hsurf_diff, levels=[500], colors='darkgreen', linewidths=1,
                                   transform=ccrs.PlateCarree())

# --- plot precipitation
levels1 = MaxNLocator(nbins=20).tick_values(0, 20)
cmap1 = cmc.davos_r
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
levels2 = MaxNLocator(nbins=11).tick_values(-5, 5)
cmap2 = drywet(25, cmc.vik_r)
norm2 = colors.TwoSlopeNorm(vmin=-5., vcenter=0., vmax=5.)
# --
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
# --
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    cs[0, j] = axs[0, j].pcolormesh(rlon, rlat, data[sim]['TOT_PREC'], cmap=cmap, norm=norm, shading="auto")
# --
cax = fig.add_axes([axs[0, 1].get_position().x1+0.01, axs[0, 1].get_position().y0, 0.015, axs[0, 1].get_position().height])
cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical', extend='max')
cbar.ax.tick_params(labelsize=13)
cax = fig.add_axes([axs[0, 2].get_position().x1+0.01, axs[0, 2].get_position().y0, 0.015, axs[0, 2].get_position().height])
cbar = fig.colorbar(cs[0, 2], cax=cax, orientation='vertical', extend='both', ticks=[-4, -2, 0, 2, 4])
cbar.ax.tick_params(labelsize=13)
# --
for j in range(ncol):
    label = labels[j]
    axs[0, j].set_title(f'{label}', fontweight='bold', pad=7, fontsize=14, loc='center')

# --- plot water vapor flux stream
levels1 = np.linspace(200, 500, 13, endpoint=True)
# cmap1 = plt.cm.get_cmap("Spectral")
cmap1 = cmc.roma
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
levels2 = MaxNLocator(nbins=21).tick_values(-20, 20)
cmap2 = drywet(25, cmc.vik_r)
norm2 = colors.TwoSlopeNorm(vmin=-20., vcenter=0., vmax=20.)
# --
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
scales = [8000, 8000, 2000]
# --
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    scale = scales[j]
    # cs[1, j] = axs[1, j].pcolormesh(rlon, rlat, data[sim]['TQF'], cmap=cmap, norm=norm, shading="auto")
    q[1, j] = axs[1, j].quiver(rlon[::15], rlat[::15], data[sim]['IUQ'][::15, ::15], data[sim]['IVQ'][::15, ::15],
                               data[sim]['TQF'][::15, ::15], cmap=cmap, norm=norm, scale=scale, headaxislength=3.5, headwidth=5, minshaft=0)
# --
cax = fig.add_axes([axs[1, 1].get_position().x1+0.01, axs[1, 1].get_position().y0, 0.015, axs[1, 1].get_position().height])
cbar = fig.colorbar(q[1, 1], cax=cax, orientation='vertical', extend='both', ticks=[200, 250, 300, 350, 400, 450, 500])
cbar.ax.tick_params(labelsize=13)
cax = fig.add_axes([axs[1, 2].get_position().x1+0.01, axs[1, 2].get_position().y0, 0.015, axs[1, 2].get_position().height])
cbar = fig.colorbar(q[1, 2], cax=cax, orientation='vertical', extend='both')
cbar.ax.tick_params(labelsize=13)

# --- plot geopotential height 500
# levels1 = np.linspace(5480, 5900, 15, endpoint=True)
# # cmap1 = plt.cm.get_cmap("Spectral")
# cmap1 = cmc.roma_r
# norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
# cmap2 = custom_div_cmap(23, cmc.vik)
# norm2 = colors.TwoSlopeNorm(vmin=-24., vcenter=0., vmax=12.)
# # --
# norms = [norm1, norm1, norm2]
# cmaps = [cmap1, cmap1, cmap2]
# scales = [8000, 8000, 2000]
# # --
# for j in range(ncol):
#     sim = sims[j]
#     cmap = cmaps[j]
#     norm = norms[j]
#     scale = scales[j]
#     cs[2, j] = axs[2, j].pcolormesh(rlon, rlat, data[sim]['FI'], cmap=cmap, norm=norm, shading="auto")
#
# # --
# cax = fig.add_axes([axs[2, 1].get_position().x1+0.01, axs[2, 1].get_position().y0, 0.015, axs[2, 1].get_position().height])
# cbar = fig.colorbar(cs[2, 1], cax=cax, orientation='vertical', extend='both', ticks=np.linspace(5510, 5870, 7, endpoint=True))
# cbar.ax.tick_params(labelsize=13)
# cax = fig.add_axes([axs[2, 2].get_position().x1+0.01, axs[2, 2].get_position().y0, 0.015, axs[2, 2].get_position().height])
# cbar = fig.colorbar(cs[2, 2], cax=cax, orientation='vertical', extend='both', ticks=[-24, -18, -12, -6, 0, 3, 6, 9, 12])
# cbar.ax.tick_params(labelsize=13)

# --- plot geopotential height & T 200
levels1 = MaxNLocator(nbins=20).tick_values(216, 226)
# cmap1 = plt.cm.get_cmap("Spectral")
cmap1 = cmc.roma_r
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
levels2 = MaxNLocator(nbins=15).tick_values(-1, 1)
cmap2 = custom_div_cmap(25, cmc.vik)
norm2 = colors.TwoSlopeNorm(vmin=-1, vcenter=0., vmax=1)

level1 = np.linspace(11700, 12600, 10, endpoint=True)
level2 = [-30, -20, -10, 5, 10, 15]
# --
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
levels = [level1, level1, level2]
# --
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    level = levels[j]
    cs[2, j] = axs[2, j].pcolormesh(rlon, rlat, data[sim]['T200'], cmap=cmap, norm=norm, shading="auto")
    ct[2, j] = axs[2, j].contour(rlon, rlat, data[sim]['FI200'], levels=level,
                                 colors='k', linewidths=.8)
    clabel = axs[2, j].clabel(ct[2, j], levels=level, inline=True, fontsize=11,
                              use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

# --
cax = fig.add_axes([axs[2, 1].get_position().x1+0.01, axs[2, 1].get_position().y0, 0.015, axs[2, 1].get_position().height])
cbar = fig.colorbar(cs[2, 1], cax=cax, orientation='vertical', extend='both', ticks=np.linspace(216, 226, 6, endpoint=True))
cbar.ax.tick_params(labelsize=13)
cax = fig.add_axes([axs[2, 2].get_position().x1+0.01, axs[2, 2].get_position().y0, 0.015, axs[2, 2].get_position().height])
cbar = fig.colorbar(cs[2, 2], cax=cax, orientation='vertical', extend='both', ticks=[-1, -0.5, 0, 0.5, 1])
cbar.ax.tick_params(labelsize=13)

# --- plot wind 200
levels1 = MaxNLocator(nbins=15).tick_values(5, 20)
# cmap1 = plt.cm.get_cmap("Spectral")
cmap1 = cmc.roma_r
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
levels2 = MaxNLocator(nbins=15).tick_values(-2, 2)
cmap2 = custom_div_cmap(25, cmc.vik)
norm2 = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2.)
# --
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
# --
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    scale = scales[j]
    # cs[1, j] = axs[1, j].pcolormesh(rlon, rlat, data[sim]['TQF'], cmap=cmap, norm=norm, shading="auto")
    q[3, j] = axs[3, j].streamplot(rlon, rlat, data[sim]['U200'], data[sim]['V200'], color=data[sim]['WS200'],
                                   density=1, cmap=cmap, norm=norm)
# --
cax = fig.add_axes([axs[3, 1].get_position().x1+0.01, axs[3, 1].get_position().y0, 0.015, axs[3, 1].get_position().height])
cbar = fig.colorbar(q[3, 1].lines, cax=cax, orientation='vertical', extend='both', ticks=[6, 10, 14, 18])
cbar.ax.tick_params(labelsize=13)
cax = fig.add_axes([axs[3, 2].get_position().x1+0.01, axs[3, 2].get_position().y0, 0.015, axs[3, 2].get_position().height])
cbar = fig.colorbar(q[3, 2].lines, cax=cax, orientation='vertical', extend='both', ticks=[-2, -1, 0, 1, 2])
cbar.ax.tick_params(labelsize=13)

# --- plot wind 850
levels1 = MaxNLocator(nbins=20).tick_values(2, 12)
# cmap1 = plt.cm.get_cmap("Spectral")
cmap1 = cmc.roma_r
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
levels2 = MaxNLocator(nbins=15).tick_values(-0.8, 0.8)
cmap2 = custom_div_cmap(25, cmc.vik)
norm2 = colors.TwoSlopeNorm(vmin=-0.8, vcenter=0., vmax=0.8)
# --
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
# --
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    scale = scales[j]
    # cs[1, j] = axs[1, j].pcolormesh(rlon, rlat, data[sim]['TQF'], cmap=cmap, norm=norm, shading="auto")
    q[4, j] = axs[4, j].streamplot(rlon, rlat, data[sim]['U850'], data[sim]['V850'], color=data[sim]['WS850'],
                                   density=1, cmap=cmap, norm=norm)
# --
cax = fig.add_axes([axs[4, 1].get_position().x1+0.01, axs[4, 1].get_position().y0, 0.015, axs[4, 1].get_position().height])
cbar = fig.colorbar(q[4, 1].lines, cax=cax, orientation='vertical', extend='both', ticks=[2, 4, 6, 8, 10, 12])
cbar.ax.tick_params(labelsize=13)
cax = fig.add_axes([axs[4, 2].get_position().x1+0.01, axs[4, 2].get_position().y0, 0.015, axs[4, 2].get_position().height])
cbar = fig.colorbar(q[4, 2].lines, cax=cax, orientation='vertical', extend='both', ticks=[-0.8, -0.4, 0, 0.4, 0.8])
cbar.ax.tick_params(labelsize=13)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/results/TRED/"
fig.savefig(plotpath + 'smr.png', dpi=500)
plt.close(fig)







