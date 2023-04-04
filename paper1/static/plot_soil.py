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
import matplotlib.gridspec as gridspec

# -------------------------------------------------------------------------------
# import data
#
path = "/Users/kaktus/Documents/ETH/BECCY/myscripts/data/soil/"
file1 = 'W_SO_COSMO-crCLIM_EAS11_V57_day_20020101.nc'
file2 = 'W_SO_COSMO-crCLIM_EAS11_V57_day_20100101.nc'

ds = xr.open_dataset(path + file1)
soil_ctrl = ds["W_SO"].values[0, 0, :, :]
lat = ds["lat"].values
lon = ds["lon"].values
ds.close()

ds = xr.open_dataset(path + file2)
soil_topo1 = ds["W_SO"].values[0, 0, :, :]
ds.close()

soil_diff = soil_ctrl - soil_topo1
# soil_diff = np.ma.masked_where(soil_diff < 1, soil_diff)

# -------------------------------------------------------------------------------
# plot
#
ar = 0.5  # initial aspect ratio for first trial
wi = 15  # width in inches
hi = wi * ar  # height in inches
ncol = 3
nrow = 1

color1 = plt.get_cmap('terrain')(np.linspace(0.22, 1, 256))
all_colors = np.vstack(color1)
cmap1 = colors.LinearSegmentedColormap.from_list('terrain', all_colors)

color1 = plt.get_cmap('terrain')(np.linspace(0.22, 0.9, 256))
all_colors = np.vstack(color1)
cmap2 = colors.LinearSegmentedColormap.from_list('terrain', all_colors)

palette = copy(cmap2)
palette.set_under('white', 0)
palette.set_bad(color='white')

gs = gridspec.GridSpec(nrow, ncol)
fig = plt.figure(figsize=(wi, hi))

projection = ccrs.LambertConformal(central_longitude=115, central_latitude=28.5)
projection = ccrs.PlateCarree()

axs0 = plt.subplot(gs[0], projection=projection)
axs1 = plt.subplot(gs[1], projection=projection)
axs2 = plt.subplot(gs[2], projection=projection)

cs0 = axs0.contourf(lon, lat, soil_ctrl, transform=ccrs.PlateCarree(), levels=14, cmap='YlGnBu', vmin=0, vmax=4)
axs0.contour(lon, lat, soil_ctrl, cs0.levels, colors='k', linewidths=.3)
cs1 = axs1.contourf(lon, lat, soil_topo1, transform=ccrs.PlateCarree(), levels=14, cmap='YlGnBu', vmin=0, vmax=4)
axs1.contour(lon, lat, soil_topo1, cs1.levels, colors='k', linewidths=.3)
divnorm=colors.TwoSlopeNorm(vmin=-1., vcenter=0., vmax=1)
cs2 = axs2.contourf(lon, lat, soil_diff, transform=ccrs.PlateCarree(), levels=np.linspace(-1, 1, 11), cmap='BrBG', norm=divnorm, extend='both')
# axs2.contour(lon, lat, soil_diff, cs2.levels, colors='k', linewidths=.3)

axs0.set_title("2002-01-01", fontweight='bold', pad=10)
axs1.set_title("2010-01-01", fontweight='bold', pad=10)
axs2.set_title("Difference", fontweight='bold', pad=10)

axs0.text(0.03, 0.96, 'a', ha='center', va='center', transform=axs0.transAxes,
          fontsize=12, family='sans-serif')
axs1.text(0.03, 0.96, 'b', ha='center', va='center', transform=axs1.transAxes,
          fontsize=12, family='sans-serif')
axs2.text(0.03, 0.96, 'c', ha='center', va='center', transform=axs2.transAxes,
          fontsize=12, family='sans-serif')

axs = [axs0, axs1, axs2]
gl = []
for ax in axs:
    ax.set_extent([78, 150, 7, 55], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k')
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

cb1.set_label('m')
cb2.set_label('m')

xmin, xmax = axs[0].get_xbound()
ymin, ymax = axs[0].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol * 1.4

fig.set_figheight(wi * y2x_ratio)
plt.show()

fig.savefig('figure_soil.png', dpi=300)
