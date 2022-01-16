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
path1 = "/Users/rxiang/Desktop/plt/ctrl/"
path2 = "/Users/rxiang/Desktop/plt/topo1/"
file1 = '01_T_mergetime_50000_JJA.nc'

ds = xr.open_dataset(path1 + file1)
tmp_ctrl = ds["T"].values[0, 0, :, :] - 273.15
lat = ds["lat"].values
lon = ds["lon"].values
ds.close()

ds = xr.open_dataset(path2 + file1)
tmp_topo1 = ds["T"].values[0, 0, :, :] - 273.15
ds.close()

tmp_diff = tmp_ctrl - tmp_topo1

# -------------------------------------------------------------------------------
# plot
#
ar = 0.5  # initial aspect ratio for first trial
wi = 15  # width in inches
hi = wi * ar  # height in inches

gs = gridspec.GridSpec(1, 3)
fig = plt.figure(figsize=(wi, hi))

projection = ccrs.LambertConformal(central_longitude=115, central_latitude=28.5)
projection = ccrs.PlateCarree()

axs0 = plt.subplot(gs[0], projection=projection)
axs1 = plt.subplot(gs[1], projection=projection)
axs2 = plt.subplot(gs[2], projection=projection)

cs0 = axs0.contourf(lon, lat, tmp_ctrl, transform=ccrs.PlateCarree(), levels=np.linspace(-10, 3.0, 14), cmap='YlOrRd', vmin=-10, vmax=3, extend='both')
axs0.contour(lon, lat, tmp_ctrl, cs0.levels, colors='k', linewidths=.3)
cs1 = axs1.contourf(lon, lat, tmp_topo1, transform=ccrs.PlateCarree(), levels=np.linspace(-10, 3.0, 14), cmap='YlOrRd', vmin=-10, vmax=3, extend='both')
axs1.contour(lon, lat, tmp_topo1, cs1.levels, colors='k', linewidths=.3)
cs2 = axs2.contourf(lon, lat, tmp_diff, transform=ccrs.PlateCarree(), levels=np.linspace(-1, 1, 11), cmap='RdYlBu_r', vmin=-1, vmax=1, extend='both')
axs2.contour(lon, lat, tmp_diff, cs2.levels, colors='k', linewidths=.3)

axs0.set_title("Control", fontweight='bold')
axs1.set_title("Reduced topography 1", fontweight='bold')
axs2.set_title("Difference", fontweight='bold')

axs0.text(0.05, 0.95, 'a', ha='center', va='center', transform=axs0.transAxes,
          fontsize=12, family='sans-serif')
axs1.text(0.05, 0.95, 'b', ha='center', va='center', transform=axs1.transAxes,
          fontsize=12, family='sans-serif')
axs2.text(0.05, 0.95, 'c', ha='center', va='center', transform=axs2.transAxes,
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
gl1.right_labels = False
gl2.right_labels = False

plt.tight_layout()
plt.subplots_adjust(left=0.03, bottom=None, right=None, top=None, wspace=0.1, hspace=0)

cb1 = fig.colorbar(cs1, orientation='horizontal', shrink=0.9, aspect=70, ax=axs[0:2], pad=0.03)
cb2 = fig.colorbar(cs2, orientation='horizontal', shrink=0.9, aspect=30, ax=axs[2], pad=0.03)

cb1.set_label('[$^{o}C$]')
# cb1.set_ticks([-18, -6, -3, -1, 0, 1, 3])
cb2.set_label('[$^{o}C$]')

xmin, xmax = axs[0].get_xbound()
ymin, ymax = axs[0].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin)

fig.set_figheight(wi * y2x_ratio)
plt.show()

fig.savefig('figure_tmp500.png', dpi=300)