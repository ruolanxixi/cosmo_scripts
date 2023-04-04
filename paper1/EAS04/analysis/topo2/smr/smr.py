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
from plotcosmomap import plotcosmo04sm, pole04
import matplotlib.colors as colors
import scipy.ndimage as ndimage

mpl.style.use("classic")
font = {'size': 14}
matplotlib.rc('font', **font)
matplotlib.rcParams['savefig.dpi'] = 300
###############################################################################
# Data
###############################################################################
sims = ['ctrl', 'topo2']
path = "/project/pr133/rxiang/data/cosmo/"

data = {}
labels = ['CTRL04', 'TENV04', 'TENV04 - CTRL04']
lb_rows = ['a', 'b', 'c', 'd', 'e']

g = 9.80665

vars = ['TOT_PREC', 'IUQ', 'IVQ', 'TQF']
# load data
for s in range(len(sims)):
    sim = sims[s]
    data[sim] = {}
    ds = xr.open_dataset(f'{path}' + f'EAS04_{sim}/monsoon/TOT_PREC/smr/' + f'01-05.TOT_PREC.smr.cpm.nc')
    smr = ds.variables['TOT_PREC'][...]
    smr = np.nanmean(smr, axis=0)
    data[sim]['TOT_PREC'] = smr
    ds = xr.open_dataset(f'{path}' + f'EAS04_{sim}/monsoon/IVT/smr/' + f'01-05.IVT.smr.cpm.nc')
    iuq = ds.variables['IUQ'][:, :, :]
    smr = np.nanmean(iuq, axis=0)
    data[sim]['IUQ'] = smr
    ivq = ds.variables['IVQ'][:, :, :]
    smr = np.nanmean(ivq, axis=0)
    data[sim]['IVQ'] = smr
    data[sim]['TQF'] = np.nanmean(np.sqrt(iuq ** 2 + ivq ** 2), axis=0)

# compute difference
data['diff'] = {}
for v in range(len(vars)):
    var = vars[v]
    data['diff'][var] = data['topo2'][var] - data['ctrl'][var]

# load topo
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_EAS_ext_12km_merit_unmod_topo.nc')
hsurf_ctrl = ds['HSURF'].values[:, :]
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_EAS_ext_12km_merit_env_topo_adj.nc')
hsurf_topo2 = ds['HSURF'].values[:, :]
hsurf_diff = ndimage.gaussian_filter(hsurf_topo2 - hsurf_ctrl, sigma=5, order=0)
hsurf_ctrl = ndimage.gaussian_filter(hsurf_ctrl, sigma=3, order=0)
lat_ = ds["lat"].values
lon_ = ds["lon"].values
ds.close()

###############################################################################
# %% Plot
###############################################################################
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole04()
rlon_, rlat_ = np.meshgrid(rlon, rlat)
sims = ['ctrl', 'topo2', 'diff']
fig = plt.figure(figsize=(11.1, 5.3))
gs1 = gridspec.GridSpec(2, 2, left=0.04, bottom=0.05, right=0.585,
                       top=0.94, hspace=0.15, wspace=0.13,
                       width_ratios=[1, 1], height_ratios=[1, 1])
gs2 = gridspec.GridSpec(2, 1, left=0.695, bottom=0.05, right=0.925,
                       top=0.94, hspace=0.15, wspace=0.13)
ncol = 3  # edit here
nrow = 2

axs, cs, ct, topo, q = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                            np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object')

for i in range(nrow):
    label = lb_rows[i]
    for j in range(ncol-1):
        axs[i, j] = fig.add_subplot(gs1[i, j], projection=rot_pole_crs)
        axs[i, j] = plotcosmo04sm(axs[i, j])
    axs[i, 2] = fig.add_subplot(gs2[i, 0], projection=rot_pole_crs)
    axs[i, 2] = plotcosmo04sm(axs[i, 2])
    axs[i, 0].text(-0.06, 0.97, f'({label})', ha='right', va='bottom', transform=axs[i, 0].transAxes, fontsize=14)

# plot topo_diff
for i in range(nrow):
    axs[i, 2] = plotcosmo04sm(axs[i, 2])
    topo[i, 2] = axs[i, 2].contour(lon_, lat_, hsurf_diff, levels=[100], colors='darkgreen', linewidths=1,
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
cbar.ax.tick_params(labelsize=14)
cax = fig.add_axes([axs[0, 2].get_position().x1+0.01, axs[0, 2].get_position().y0, 0.015, axs[0, 2].get_position().height])
cbar = fig.colorbar(cs[0, 2], cax=cax, orientation='vertical', extend='both', ticks=[-4, -2, 0, 2, 4])
cbar.ax.tick_params(labelsize=14)
# --
for j in range(ncol):
    label = labels[j]
    axs[0, j].set_title(f'{label}', fontweight='bold', pad=7, fontsize=14, loc='center')

# --- plot water vapor flux stream
levels1 = MaxNLocator(nbins=25).tick_values(0, 500)
cmap1 = cmc.davos_r
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

levels2 = MaxNLocator(nbins=21).tick_values(-50, 50)
cmap2 = custom_div_cmap(25, cmc.broc_r)
norm2 = colors.TwoSlopeNorm(vmin=-50., vcenter=0., vmax=50.)
# --
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
scales = [2000, 2000, 300]
# --
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    scale = scales[j]
    cs[1, j] = axs[1, j].pcolormesh(rlon, rlat, data[sim]['TQF'], cmap=cmap, norm=norm, shading="auto")
    q[1, j] = axs[1, j].quiver(rlon[::30], rlat[::30], data[sim]['IUQ'][::30, ::30], data[sim]['IVQ'][::30, ::30],
                        color='k', scale=scale, headaxislength=3.5, headwidth=5, minshaft=0)
    # q[1, j] = axs[1, j].streamplot(rlon, rlat, data[sim]['IUQ'], data[sim]['IVQ'], color='k', density=0.7, linewidth=0.7)
# --
cax = fig.add_axes([axs[1, 1].get_position().x1+0.01, axs[1, 1].get_position().y0, 0.015, axs[1, 1].get_position().height])
cbar = fig.colorbar(cs[1, 1], cax=cax, orientation='vertical', extend='max', ticks=[0, 100, 200, 300, 400, 500])
cbar.ax.tick_params(labelsize=14)
cax = fig.add_axes([axs[1, 2].get_position().x1+0.01, axs[1, 2].get_position().y0, 0.015, axs[1, 2].get_position().height])
cbar = fig.colorbar(cs[1, 2], cax=cax, orientation='vertical', extend='both', ticks=[-40, -20, 0, 20, 40])
cbar.ax.tick_params(labelsize=14)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/results/TENV/"
fig.savefig(plotpath + 'smr04sm.png', dpi=500)
plt.close(fig)







