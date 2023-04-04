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
# from auxiliary import read_topo
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
    data_diff = - data_diff
    data = np.dstack((data_ctrl, data_topo1, data_diff))
    da = xr.DataArray(data=data,
                      coords={"rlat": rlat,
                              "rlon": rlon,
                              "sim": ["ctrl", "topo1", "diff"]},
                      dims=["rlat", "rlon", "sim"])

    return da


da = read_data("T")
da_f = read_data("FI")

# [topo_ctrl, topo_topo1, topo_lat, topo_lon] = read_topo()
# topo_ctrl[topo_ctrl <= 5000] = 'nan'
# topo_topo1[topo_topo1 <= 5000] = 'nan'

# -------------------------------------------------------------------------------
# plot
# figure setup
# gs = gridspec.GridSpec(1, 3)
# gs.update(left=0.03, right=0.99, top=0.98, bottom=0.05, hspace=0.1, wspace=0.12)
#
# fig = plt.figure(figsize=(15, 4.5), constrained_layout=True)
# axs = np.empty(shape=(1, 3), dtype='object')
# cs = np.empty(shape=(1, 3), dtype='object')
# ct = np.empty(shape=(1, 3), dtype='object')
#
# axs[0, 0] = plt.subplot(gs[0], projection=rot_pole_crs)
# axs[0, 0] = plotcosmo(axs[0, 0])
# axs[0, 1] = plt.subplot(gs[1], projection=rot_pole_crs)
# axs[0, 1] = plotcosmo(axs[0, 1])
# axs[0, 2] = plt.subplot(gs[2], projection=rot_pole_crs)
# axs[0, 2] = plotcosmo(axs[0, 2])
#
# # plot data
# levels = MaxNLocator(nbins=24).tick_values(-20, 3)
# cmap = cmc.roma_r
# norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# cmap_topo = colors.ListedColormap(['grey'])
#
# cs[0, 0] = axs[0, 0].pcolormesh(rlon, rlat, da.sel(sim='ctrl').values[:, :] - 273.15, cmap=cmap, norm=norm)
# cax1 = fig.add_axes(
#     [axs[0, 0].get_position().x0, axs[0, 0].get_position().y0 - 0.1, axs[0, 0].get_position().width * 2.12, 0.03])
# cb1 = fig.colorbar(cs[0, 0], cax=cax1, orientation='horizontal', extend='max')
# cs[0, 0] = axs[0, 0].pcolormesh(topo_lon, topo_lat, topo_ctrl, transform=ccrs.PlateCarree(), cmap=cmap_topo)
# ct[0, 0] = axs[0, 0].contour(rlon, rlat, da_f.sel(sim='ctrl').values[:, :] / g, levels=np.linspace(5600, 5900, 13),
#                              colors='k',
#                              linewidths=.7)
# cs[0, 1] = axs[0, 1].pcolormesh(rlon, rlat, da.sel(sim='topo1').values[:, :] - 273.15, cmap=cmap, norm=norm)
# cs[0, 1] = axs[0, 1].pcolormesh(topo_lon, topo_lat, topo_topo1, transform=ccrs.PlateCarree(), cmap=cmap_topo)
# ct[0, 1] = axs[0, 1].contour(rlon, rlat, da_f.sel(sim='topo1').values[:, :] / g, levels=np.linspace(5600, 5900, 13),
#                              colors='k',
#                              linewidths=.7)
#
# cmap = custom_div_cmap(27, cmc.vik)
# cs[0, 2] = axs[0, 2].pcolormesh(rlon, rlat, da.sel(sim='diff').values[:, :], cmap=cmap, clim=(-1.2, 1.2),
#                                 shading="auto")
#
# trans = Transformer.from_proj(ccrs.PlateCarree(), rot_pole_crs, always_xy=True)
# x = np.array([120, 147, 130, 108, 125, 105, 135, 110, 110, 118, 130, 160])
# y = np.array([60, 53, 50, 50, 45, 46, 40, 40, 35, 32, 29, 18])
# loc_lon, loc_lat = trans.transform(x, y)
# manual_locations = [i for i in zip(loc_lon, loc_lat)]
# axs[0, 0].clabel(ct[0, 0], ct[0, 0].levels[::1], inline=True, fontsize=8, manual=manual_locations)
# axs[0, 1].clabel(ct[0, 1], ct[0, 1].levels[::1], inline=True, fontsize=8, manual=manual_locations)
#
# axs[0, 0].set_title('Control', fontweight='bold', pad=12, fontsize=13)
# axs[0, 1].set_title('Reduced elevation', fontweight='bold', pad=12, fontsize=13)
# axs[0, 2].set_title('Reduced elevation - Control', fontweight='bold', pad=12, fontsize=13)
#
#
# cb1.set_label('$^{o}C$', fontsize=11)
# cb1.ax.tick_params(labelsize=11)
# cax2 = fig.add_axes(
#     [axs[0, 2].get_position().x0, axs[0, 2].get_position().y0 - .1, axs[0, 2].get_position().width, 0.03])
# cb2 = fig.colorbar(cs[0, 2], cax=cax2, orientation='horizontal', extend='max')
# cb2.set_label('$^{o}C$', fontsize=11)
# cb2.ax.tick_params(labelsize=11)
#
# plt.show()
# plotpath = "/project/pr133/rxiang/figure/topo1/"
# fig.savefig(plotpath + 'tmp_1x3.png', dpi=300)
# plt.close(fig)

fig = plt.figure(figsize=(6, 4.5))
left, bottom, right, top = 0.09, 0.13, 0.99, 0.95
gs = gridspec.GridSpec(1, 1, left=left, bottom=bottom, right=right, top=top, wspace=0.12, hspace=0.1)
ax = fig.add_subplot(gs[0], projection=rot_pole_crs)
ax=plotcosmo(ax)
cmap = custom_div_cmap(27, cmc.vik)
cs = ax.pcolormesh(rlon, rlat, -da.sel(sim='diff').values[:, :], cmap=cmap, clim=(-1.2, 1.2), shading="auto")
ct = ax.contour(rlon, rlat, -da_f.sel(sim='diff').values[:, :] / g, levels=np.linspace(-12, 24, 13, endpoint=True),
                             colors='maroon',
                             linewidths=.8)
ax.clabel(ct, ct.levels[::1], inline=True, fontsize=8)
ax.text(0.21, 1.05, 'Reduced topography - Control', ha='center', va='center', transform=ax.transAxes, fontsize=11)
ax.text(0.90, 1.05, '2001-2005 JJA', ha='center', va='center', transform=ax.transAxes, fontsize=11)
ax.set_title('Anomalies in Temperature and Geopotential height at 500 hPa', fontweight='bold', pad=24, fontsize=11)
cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.1, ax.get_position().width, 0.03])
cb = fig.colorbar(cs, cax=cax, orientation='horizontal', extend='both')
cb.ax.tick_params(labelsize=11)
cb.set_label('$^{o}C$', fontsize=11)
# adjust figure
fig.show()
# save figure
plotpath = "/project/pr133/rxiang/figure/EAS11/analysis/JJA/topo1/"
fig.savefig(plotpath + 'tmp_diff.png', dpi=300)
plt.close(fig)
