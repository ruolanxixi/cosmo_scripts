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
from cmcrameri import cm
from auxiliary import truncate_colormap, spat_agg_1d, spat_agg_2d
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

# -------------------------------------------------------------------------------
# import data
#
path = "/project/pr94/rxiang/data/extpar/"
file1 = 'extpar_12km_1118x670_MERIT_raw.nc'
file2 = 'extpar_EAS_ext_12km_merit_adj.nc'

[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

ds = xr.open_dataset(path + file1)
elev_ctrl = ds["HSURF"].values[:, :]
lat = ds["lat"].values
lon = ds["lon"].values
ds.close()

ds = xr.open_dataset(path + file2)
elev_topo1 = ds["HSURF"].values[:, :]
ds.close()

elev_ctrl[elev_ctrl == 0] = np.nan
elev_topo1[elev_topo1 == 0] = np.nan
elev_diff = elev_ctrl - elev_topo1
elev_diff = np.ma.masked_where(elev_diff < 1, elev_diff)

ticks = np.arange(0., 6500.0, 250.0)
cmap1 = truncate_colormap(cm.bukavu, 0.55, 1.0)
levels1 = np.arange(0., 6500.0, 250.0)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, extend="max")

levels2 = np.arange(0., 3000.0, 250.0)
ticks = np.arange(0., 3000.0, 500.0)
cmap2 = cm.lajolla
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, extend="max")
# -------------------------------------------------------------------------------
# plot domain
#
# fig, ax = plt.subplots(subplot_kw={'projection': rot_pole_crs})
# ax = plotcosmo(ax)
# cs = ax.pcolormesh(lon, lat, elev_ctrl, transform=ccrs.PlateCarree(), cmap=cmap1, norm=norm1)
# cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.1, ax.get_position().width, 0.027])
# cb = fig.colorbar(cs, cax=cax, orientation='horizontal', extend='max')
# cb.set_label('m', fontsize=11)
# # adjust figure
# fig.show()
# # save figure
# plotpath = "/project/pr133/rxiang/figure/topo/"
# fig.savefig(plotpath + 'topo_ctrl.png', dpi=300)
# plt.close(fig)

# # plot compare
# gs1 = gridspec.GridSpec(1, 2)
# gs1.update(left=0.05, right=0.99, top=0.96, bottom=0.55, hspace=0.1, wspace=0.12)
# gs2 = gridspec.GridSpec(1, 2)
# gs2.update(left=0.05, right=0.99, top=0.50, bottom=0.05, hspace=0.1, wspace=0.12)
#
# fig = plt.figure(figsize=(13, 9.8), constrained_layout=True)
# axs = np.empty(shape=(2, 2), dtype='object')
# cs = np.empty(shape=(2, 2), dtype='object')
#
# axs[0, 0] = plt.subplot(gs1[0], projection=rot_pole_crs)
# axs[0, 0] = plotcosmo(axs[0, 0])
# axs[0, 1] = plt.subplot(gs1[1], projection=rot_pole_crs)
# axs[0, 1] = plotcosmo(axs[0, 1])
# axs[1, 0] = plt.subplot(gs2[0], projection=rot_pole_crs)
# axs[1, 0] = plotcosmo(axs[1, 0])
#
# cs[0, 0] = axs[0, 0].pcolormesh(lon, lat, elev_ctrl, transform=ccrs.PlateCarree(), cmap=cmap1, norm=norm1)
# cs[0, 1] = axs[0, 1].pcolormesh(lon, lat, elev_topo1, transform=ccrs.PlateCarree(), cmap=cmap1, norm=norm1)
# cs[1, 0] = axs[1, 0].pcolormesh(lon, lat, elev_diff, transform=ccrs.PlateCarree(), cmap=cmap2, norm=norm2)
#
# axs[0, 0].set_title('Control', fontweight='bold', pad=12, fontsize=13)
# axs[0, 1].set_title('Reduced elevation', fontweight='bold', pad=12, fontsize=13)
# axs[1, 0].set_title('Control - Reduced elevation', fontweight='bold', pad=12, fontsize=13)
#
# cax1 = fig.add_axes([axs[0, 0].get_position().x0, axs[0, 0].get_position().y0 - 0.06, axs[0, 0].get_position().width * 2.12, 0.02])
# cb1 = fig.colorbar(cs[0, 0], cax=cax1, orientation='horizontal', extend='max')
# cb1.set_label('m', fontsize=12)
# cax2 = fig.add_axes([axs[1, 0].get_position().x0, axs[1, 0].get_position().y0 - 0.06, axs[1, 0].get_position().width, 0.02])
# cb2 = fig.colorbar(cs[1, 0], cax=cax2, orientation='horizontal', extend='max')
# cb2.set_label('m', fontsize=12)
#
# plt.show()
# plotpath = "/project/pr133/rxiang/figure/topo/"
# fig.savefig(plotpath + 'topo_compare.png', dpi=300)
# plt.close(fig)

# plot compare
gs1 = gridspec.GridSpec(1, 2)
gs1.update(left=0.05, right=0.99, top=0.96, bottom=0.55, hspace=0.1, wspace=0.12)
gs2 = gridspec.GridSpec(1, 2)
gs2.update(left=0.05, right=0.99, top=0.50, bottom=0.05, hspace=0.1, wspace=0.12)

fig = plt.figure(figsize=(10, 12), constrained_layout=True)
axs = np.empty(shape=(2, 2), dtype='object')
cs = np.empty(shape=(2, 2), dtype='object')

axs[0, 0] = plt.subplot(gs1[0], projection=rot_pole_crs)
axs[0, 1] = plt.subplot(gs1[1], projection=rot_pole_crs)
axs[1, 0] = plt.subplot(gs2[0], projection=rot_pole_crs)

for ax in (axs[0, 0], axs[0, 1], axs[1, 0]):
    ax.set_extent([88, 118, 14, 40], crs=ccrs.PlateCarree())  # for extended 12km domain
    # ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN, edgecolor='k')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([90, 100, 110, 120])
    gl.ylocator = mticker.FixedLocator([15, 20, 25, 30, 35, 40])

    # add ticks manually
    ax.text(-0.05, 0.88, '35??N', ha='center', va='center', transform=ax.transAxes, fontsize=11)
    ax.text(-0.05, 0.70, '30??N', ha='center', va='center', transform=ax.transAxes, fontsize=11)
    ax.text(-0.05, 0.52, '25??N', ha='center', va='center', transform=ax.transAxes, fontsize=11)
    ax.text(-0.05, 0.34, '20??N', ha='center', va='center', transform=ax.transAxes, fontsize=11)
    ax.text(-0.05, 0.16, '15??N', ha='center', va='center', transform=ax.transAxes, fontsize=11)

    ax.text(0.04, -0.05, '90??E', ha='center', va='center', transform=ax.transAxes, fontsize=11)
    ax.text(0.38, -0.05, '100??E', ha='center', va='center', transform=ax.transAxes, fontsize=11)
    ax.text(0.72, -0.05, '110??E', ha='center', va='center', transform=ax.transAxes, fontsize=11)

cs[0, 0] = axs[0, 0].pcolormesh(lon, lat, elev_ctrl, transform=ccrs.PlateCarree(), cmap=cmap1, norm=norm1)
cs[0, 1] = axs[0, 1].pcolormesh(lon, lat, elev_topo1, transform=ccrs.PlateCarree(), cmap=cmap1, norm=norm1)
cs[1, 0] = axs[1, 0].pcolormesh(lon, lat, elev_diff, transform=ccrs.PlateCarree(), cmap=cmap2, norm=norm2)

axs[0, 0].set_title('Modern topography', fontweight='bold', pad=12, fontsize=13)
axs[0, 1].set_title('Reduced topography', fontweight='bold', pad=12, fontsize=13)
axs[1, 0].set_title('Reduction in topography', fontweight='bold', pad=12, fontsize=13)

cax1 = fig.add_axes([axs[0, 0].get_position().x0, axs[0, 0].get_position().y0 - 0.06, axs[0, 0].get_position().width * 2.12, 0.02])
cb1 = fig.colorbar(cs[0, 0], cax=cax1, orientation='horizontal', extend='max')
cb1.set_label('m', fontsize=12)
cax2 = fig.add_axes([axs[1, 0].get_position().x0, axs[1, 0].get_position().y0 - 0.06, axs[1, 0].get_position().width, 0.02])
cb2 = fig.colorbar(cs[1, 0], cax=cax2, orientation='horizontal', extend='max')
cb2.set_label('m', fontsize=12)

plt.show()
plotpath = "/project/pr133/rxiang/figure/topo/"
fig.savefig(plotpath + 'topo_compare_small.png', dpi=300)
plt.close(fig)
