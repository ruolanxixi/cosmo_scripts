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

# -------------------------------------------------------------------------------
# import data
#
path1 = "/Users/kaktus/Documents/ETH/BECCY/myscripts/data/ctrl/"
path2 = "/Users/kaktus/Documents/ETH/BECCY/myscripts/data/topo1/"
file1 = '01_TOT_PREC_JJA_cut.nc'
file2 = '01_U_85000_JJA_cut.nc'
file3 = '01_V_85000_JJA_cut.nc'

ds = xr.open_dataset(path1 + file1)
prec_ctrl = ds["TOT_PREC"].values[0, :, :]
lat = ds["lat"].values
lon = ds["lon"].values
ds.close()

ds = xr.open_dataset(path2 + file1)
prec_topo1 = ds["TOT_PREC"].values[0, :, :]
ds.close()

prec_diff = prec_ctrl - prec_topo1

ds = xr.open_dataset(path1 + file2)
u_ctrl = ds["U"].values[0, 0, :, :]
ds.close()

ds = xr.open_dataset(path2 + file2)
u_topo1 = ds["U"].values[0, 0, :, :]
ds.close()

u_diff = u_ctrl - u_topo1

ds = xr.open_dataset(path1 + file3)
v_ctrl = ds["V"].values[0, 0, :, :]
ds.close()

ds = xr.open_dataset(path2 + file3)
v_topo1 = ds["V"].values[0, 0, :, :]
ds.close()

v_diff = v_ctrl - v_topo1
# -------------------------------------------------------------------------------
# plot
#
ar = 1.0  # initial aspect ratio for first trial
wi = 15  # width in inches
hi = wi * ar  # height in inches

gs = gridspec.GridSpec(1, 3)
fig = plt.figure(figsize=(wi, hi))

projection = ccrs.LambertConformal(central_longitude=115, central_latitude=28.5)
projection = ccrs.PlateCarree()
axs0 = plt.subplot(gs[0], projection=projection)
axs1 = plt.subplot(gs[1], projection=projection)
axs2 = plt.subplot(gs[2], projection=projection)

cs0 = axs0.contourf(lon, lat, prec_ctrl, transform=ccrs.PlateCarree(), levels=np.linspace(0, 16, 11), cmap='YlGnBu', vmin=0, vmax=20, extend='max')
# axs0.contour(lon, lat, prec_ctrl, cs0.levels, colors='k', linewidths=.3)
cs1 = axs1.contourf(lon, lat, prec_topo1, transform=ccrs.PlateCarree(), levels=np.linspace(0, 16, 11), cmap='YlGnBu', vmin=0, vmax=20, extend='max')
# axs1.contour(lon, lat, prec_topo1, cs1.levels, colors='k', linewidths=.3)
cs2 = axs2.contourf(lon, lat, prec_diff, transform=ccrs.PlateCarree(), levels=np.linspace(-5, 5, 11), cmap='RdYlBu', vmin=-5, vmax=5)
# axs2.contour(lon, lat, prec_diff, cs2.levels, colors='k', linewidths=.3)

# add wind component
q0 = axs0.quiver(lon[::15,::15], lat[::15,::15], u_ctrl[::15,::15], v_ctrl[::15,::15], transform=projection, color='red', scale=130)
axs0.quiverkey(q0, 0.9, 1.05, 10, r'$10 m/s$', labelpos='E', transform=axs0.transAxes)
q1 = axs1.quiver(lon[::15,::15], lat[::15,::15], u_topo1[::15,::15], v_topo1[::15,::15], transform=projection, color='red', scale=130)
axs1.quiverkey(q1, 0.9, 1.05, 10, r'$10 m/s$', labelpos='E', transform=axs1.transAxes)
q2 = axs2.quiver(lon[::15,::15], lat[::15,::15], u_diff[::15,::15], v_diff[::15,::15], transform=projection, color='k', scale=60)
axs2.quiverkey(q2, 0.9, 1.05, 5, r'$5 m/s$', labelpos='E', transform=axs2.transAxes)

axs0.set_title("Control", fontweight='bold', pad=10)
axs1.set_title("Reduced topography 1", fontweight='bold', pad=10)
axs2.set_title("Difference", fontweight='bold', pad=10)

axs0.text(0.05, 0.95, 'a', ha='center', va='center', transform=axs0.transAxes,
          fontsize=12, family='sans-serif')
axs1.text(0.05, 0.95, 'b', ha='center', va='center', transform=axs1.transAxes,
          fontsize=12, family='sans-serif')
axs2.text(0.05, 0.95, 'c', ha='center', va='center', transform=axs2.transAxes,
          fontsize=12, family='sans-serif')

axs = [axs0, axs1, axs2]
gl = []
for ax in axs:
    ax.set_extent([78, 150, 7, 55], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

gl0 = axs[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='grey', alpha=0.5, linestyle='--')
gl1 = axs[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='grey', alpha=0.5, linestyle='--')
gl2 = axs[2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='grey', alpha=0.5, linestyle='--')

gl0.right_labels = False
gl0.top_labels = False
gl1.right_labels = False
gl1.top_labels = False
gl2.right_labels = False
gl2.top_labels = False

plt.tight_layout()
plt.subplots_adjust(left=0.03, bottom=None, right=None, top=None, wspace=0.1, hspace=0)

cb1 = fig.colorbar(cs1, orientation='horizontal', shrink=0.9, aspect=70, ax=axs[0:2], pad=0.03)
cb2 = fig.colorbar(cs2, orientation='horizontal', shrink=0.9, aspect=30, ax=axs[2], pad=0.03)

cb1.set_label('[mm/day]')
# cb1.set_ticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000])
cb2.set_label('[mm/day]')

xmin, xmax = axs[0].get_xbound()
ymin, ymax = axs[0].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin)

fig.set_figheight(wi * y2x_ratio)
plt.show()

#fig.savefig('figure_prec_wind.png', dpi=300)
