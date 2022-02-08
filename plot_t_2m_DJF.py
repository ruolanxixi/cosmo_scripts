# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.colors as colors
from copy import copy
import matplotlib.gridspec as gridspec
from matplotlib import ticker

# -------------------------------------------------------------------------------
# import data
#
path1 = "/Users/kaktus/Documents/ETH/BECCY/myscripts/data/ctrl/"
path2 = "/Users/kaktus/Documents/ETH/BECCY/myscripts/data/topo1/"

file1 = '01_T_2M_DJF_cut.nc'

ds = xr.open_dataset(path1 + file1)
tmp_ctrl = ds["T_2M"].values[0, :, :] - 273.15
lat = ds["lat"].values
lon = ds["lon"].values
ds.close()

ds = xr.open_dataset(path2 + file1)
tmp_topo1 = ds["T_2M"].values[0, :, :] - 273.15
ds.close()

tmp_diff = tmp_ctrl - tmp_topo1

# -------------------------------------------------------------------------------
# plot
#
ar = 1.0  # initial aspect ratio for first trial
wi = 15  # width in inches
hi = wi * ar  # height in inches
ncol = 3
nrow = 1

gs = gridspec.GridSpec(nrow, ncol)
fig = plt.figure(figsize=(wi, hi))

projection = ccrs.LambertConformal(central_longitude=115, central_latitude=28.5)
projection = ccrs.PlateCarree()

axs0 = plt.subplot(gs[0], projection=projection)
axs1 = plt.subplot(gs[1], projection=projection)
axs2 = plt.subplot(gs[2], projection=projection)

cs0 = axs0.contourf(lon, lat, tmp_ctrl, transform=ccrs.PlateCarree(), levels=np.linspace(-25, 35, 25), cmap='YlOrRd', vmin=-15, vmax=35, extend='both')
# axs0.contour(lon, lat, tmp_ctrl, cs0.levels[::2], colors='k', linewidths=.3)
cs1 = axs1.contourf(lon, lat, tmp_topo1, transform=ccrs.PlateCarree(), levels=np.linspace(-25, 35, 25), cmap='YlOrRd', vmin=-15, vmax=35, extend='both')
# axs1.contour(lon, lat, tmp_topo1, cs1.levels[::2], colors='k', linewidths=.3)
divnorm=colors.TwoSlopeNorm(vmin=-8., vcenter=0., vmax=4)
cs2 = axs2.contourf(lon, lat, tmp_diff, transform=ccrs.PlateCarree(), levels=np.linspace(-8, 4, 13), cmap='RdYlBu_r', norm=divnorm, extend='both')
# axs2.contour(lon, lat, tmp_diff, cs2.levels[::3], colors='k', linewidths=.3)

axs0.set_title("Control", fontweight='bold')
axs1.set_title("Reduced topography 1", fontweight='bold')
axs2.set_title("Difference", fontweight='bold')

axs0.text(0.03, 0.96, 'a', ha='center', va='center', transform=axs0.transAxes,
          fontsize=12, family='sans-serif')
axs1.text(0.03, 0.96, 'b', ha='center', va='center', transform=axs1.transAxes,
          fontsize=12, family='sans-serif')
axs2.text(0.03, 0.96, 'c', ha='center', va='center', transform=axs2.transAxes,
          fontsize=12, family='sans-serif')

axs = [axs0, axs1, axs2]
gl = []
for ax in axs:
    # ax.set_extent([80, 150, 7, 50], crs=ccrs.PlateCarree())
    ax.set_extent([78, 150, 7, 55], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

gl0 = axs[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=1, color='grey', alpha=0.5, linestyle='--')
gl1 = axs[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=1, color='grey', alpha=0.5, linestyle='--')
gl2 = axs[2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=1, color='grey', alpha=0.5, linestyle='--')
gl = [gl0, gl1, gl2]

gl0.right_labels = False
gl0.top_labels = False
gl1.right_labels = False
gl1.top_labels = False
gl2.right_labels = False
gl2.top_labels = False

plt.tight_layout()
plt.subplots_adjust(left=0.03, bottom=None, right=None, top=0.90, wspace=0.1, hspace=0)

cb1 = fig.colorbar(cs1, orientation='horizontal', shrink=0.9, aspect=70, ax=axs[0:2], pad=0.07)
cb2 = fig.colorbar(cs2, orientation='horizontal', shrink=0.9, aspect=30, ax=axs[2], pad=0.07)

cb1.set_label('$^{o}C$')
# cb1.set_ticks([-18, -6, -3, -1, 0, 1, 3])
cb2.set_label('$^{o}C$')

xmin, xmax = axs[0].get_xbound()
ymin, ymax = axs[0].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol * 1.4

fig.set_figheight(wi * y2x_ratio)
plt.show()

fig.savefig('figure_tmp_2m_DJF.png', dpi=300)
