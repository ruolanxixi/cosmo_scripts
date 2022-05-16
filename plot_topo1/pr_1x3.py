# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.colors as colors
from copy import copy
from plotcosmomap import plotcosmo, pole, colorbar
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import cmcrameri.cm as cmc
from auxiliary import read_topo
import matplotlib.gridspec as gridspec
from pyproj import Transformer
from mycolor import custom_div_cmap
from numpy import inf

# -------------------------------------------------------------------------------
# import data
#
season = 'JJA'
mdvnames = ['TOT_PREC', 'U', 'V', 'TWATFLXU', 'TWATFLXV', 'W', 'T', 'FI']  # edit here
year = '2001-2005'
ctrlpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn/"
topo1path = "/project/pr133/rxiang/data/cosmo/EAS11_topo1/szn/"
sims = ['ctrl', 'ctrl']
g = 9.80665

# -------------------------------------------------------------------------------
# read data
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()


def read_data(mdvname):
    filename = f'{year}.{mdvname}.{season}.nc'
    if mdvname in ('U', 'V', 'T', 'W', 'FI'):
        data_ctrl = xr.open_dataset(f'{ctrlpath}{mdvname}/{filename}')[mdvname].values[0, 0, :, :]
        data_topo1 = xr.open_dataset(f'{topo1path}{mdvname}/{filename}')[mdvname].values[0, 0, :, :]
        data_diff = data_topo1 - data_ctrl
    else:
        data_ctrl = xr.open_dataset(f'{ctrlpath}{mdvname}/{filename}')[mdvname].values[0, :, :]
        data_topo1 = xr.open_dataset(f'{topo1path}{mdvname}/{filename}')[mdvname].values[0, :, :]
        if mdvname == 'TOT_PREC':
            np.seterr(divide='ignore', invalid='ignore')
            data_diff = (data_topo1 - data_ctrl) / data_topo1 * 100
            data_diff[np.isnan(data_diff)] = 0
            data_diff[data_diff == -inf] = -100
            np.seterr(divide='warn', invalid='warn')
        else:
            data_diff = data_topo1 - data_ctrl
    data = np.dstack((data_ctrl, data_topo1, data_diff))
    da = xr.DataArray(data=data,
                      coords={"rlat": rlat,
                              "rlon": rlon,
                              "sim": ["ctrl", "topo1", "diff"]},
                      dims=["rlat", "rlon", "sim"])

    return da


# -------------------------------------------------------------------------------
# plot
# figure setup
gs = gridspec.GridSpec(1, 3)
gs.update(left=0.03, right=0.99, top=0.98, bottom=0.05, hspace=0.1, wspace=0.12)

fig = plt.figure(figsize=(15, 4.5), constrained_layout=True)
axs = np.empty(shape=(1, 3), dtype='object')
cs = np.empty(shape=(1, 3), dtype='object')
q = np.empty(shape=(1, 3), dtype='object')

axs[0, 0] = plt.subplot(gs[0], projection=rot_pole_crs)
axs[0, 0] = plotcosmo(axs[0, 0])
axs[0, 1] = plt.subplot(gs[1], projection=rot_pole_crs)
axs[0, 1] = plotcosmo(axs[0, 1])
axs[0, 2] = plt.subplot(gs[2], projection=rot_pole_crs)
axs[0, 2] = plotcosmo(axs[0, 2])

# plot data
da = read_data("TOT_PREC")
da_u = read_data("U")
da_v = read_data("V")

levels = MaxNLocator(nbins=15).tick_values(0, 20)
cmap = cmc.davos_r
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

cs[0, 0] = axs[0, 0].pcolormesh(rlon, rlat, da.sel(sim='ctrl').values[:, :], cmap=cmap, norm=norm)
cs[0, 1] = axs[0, 1].pcolormesh(rlon, rlat, da.sel(sim='topo1').values[:, :], cmap=cmap, norm=norm)
cmap = custom_div_cmap(27, cmc.vik_r)
cs[0, 2] = axs[0, 2].pcolormesh(rlon, rlat, da.sel(sim='diff').values[:, :], cmap=cmap, clim=(-120, 120),
                                shading="auto")

q[0, 0] = axs[0, 0].quiver(rlon[::30], rlat[::30], da_u.sel(sim='ctrl').values[::30, ::30],
                           da_v.sel(sim='ctrl').values[::30, ::30], color='black', scale=150)
axs[0, 0].quiverkey(q[0, 0], 0.92, 1.12, 10, r'$10\ m\ s^{-1}$', labelpos='S', transform=axs[0, 0].transAxes,
                     fontproperties={'size': 11})
q[0, 1] = axs[0, 1].quiver(rlon[::30], rlat[::30], da_u.sel(sim='topo1').values[::30, ::30],
                           da_v.sel(sim='topo1').values[::30, ::30], color='black', scale=150)
axs[0, 1].quiverkey(q[0, 1], 0.92, 1.12, 10, r'$10\ m\ s^{-1}$', labelpos='S', transform=axs[0, 1].transAxes,
                     fontproperties={'size': 11})
q[0, 2] = axs[0, 2].quiver(rlon[::30], rlat[::30], da_u.sel(sim='diff').values[::30, ::30],
                           da_v.sel(sim='diff').values[::30, ::30], color='black', scale=50)
axs[0, 2].quiverkey(q[0, 2], 0.94, 1.12, 2, r'$2\ m\ s^{-1}$', labelpos='S', transform=axs[0, 2].transAxes,
                     fontproperties={'size': 11})

axs[0, 0].set_title('Control', fontweight='bold', pad=12, fontsize=13)
axs[0, 1].set_title('Reduced elevation', fontweight='bold', pad=12, fontsize=13)
axs[0, 2].set_title('Reduced elevation - Control', fontweight='bold', pad=12, fontsize=13)

cax1 = fig.add_axes(
    [axs[0, 0].get_position().x0, axs[0, 0].get_position().y0 - 0.1, axs[0, 0].get_position().width * 2.12, 0.03])
cb1 = fig.colorbar(cs[0, 0], cax=cax1, orientation='horizontal', extend='max')
cb1.set_label('mm/day', fontsize=11)
cb1.ax.tick_params(labelsize=11)
cax2 = fig.add_axes(
    [axs[0, 2].get_position().x0, axs[0, 2].get_position().y0 - .1, axs[0, 2].get_position().width, 0.03])
cb2 = fig.colorbar(cs[0, 2], cax=cax2, orientation='horizontal', extend='max')
cb2.set_label('%', fontsize=11)
cb2.ax.tick_params(labelsize=11)

plt.show()
plotpath = "/project/pr133/rxiang/figure/topo1/"
fig.savefig(plotpath + 'pr_1x3.png', dpi=300)
plt.close(fig)
