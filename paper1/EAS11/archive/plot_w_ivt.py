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
import pylab
from mycolor import cmap1

# -------------------------------------------------------------------------------
# import data
#
path1 = "/Users/kaktus/Documents/ETH/BECCY/myscripts/data/ctrl/"
path2 = "/Users/kaktus/Documents/ETH/BECCY/myscripts/data/topo1/"

file1 = '01_W_50000_JJA_cut.nc'
file2 = '01_IVT_U_JJA_cut.nc'
file3 = '01_IVT_V_JJA_cut.nc'
file4 = '01_TQV_JJA_cut.nc'

ds = xr.open_dataset(path1 + file1)
w_ctrl = ds["W"].values[0, 0, :, :]*100
lat = ds["lat"].values
lon = ds["lon"].values
ds.close()

ds = xr.open_dataset(path2 + file1)
w_topo1 = ds["W"].values[0, 0, :, :]*100
ds.close()


np.seterr(divide='ignore', invalid='ignore')
w_diff = w_ctrl - w_topo1
np.seterr(divide='warn', invalid='warn')
w_diff[np.isnan(w_diff)] = 0


ds = xr.open_dataset(path1 + file2)
u_ctrl = ds["qvu"].values[0, :, :]
ds.close()

ds = xr.open_dataset(path2 + file2)
u_topo1 = ds["qvu"].values[0, :, :]
ds.close()

u_diff = u_ctrl - u_topo1

ds = xr.open_dataset(path1 + file3)
v_ctrl = ds["qvv"].values[0, :, :]
ds.close()

ds = xr.open_dataset(path2 + file3)
v_topo1 = ds["qvv"].values[0, :, :]
ds.close()

v_diff = v_ctrl - v_topo1

ds = xr.open_dataset(path1 + file4)
tqv_ctrl = ds["TQV"].values[0, :, :]
ds.close()

ds = xr.open_dataset(path2 + file4)
tqv_topo1 = ds["TQV"].values[0, :, :]
ds.close()
# -------------------------------------------------------------------------------
# plot
#
# pylab.rcParams['xtick.major.pad']='8'
# pylab.rcParams['ytick.major.pad']='8'

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

divnorm=colors.TwoSlopeNorm(vmin=-1., vcenter=0., vmax=3)
cs0 = axs0.contourf(lon, lat, w_ctrl, transform=ccrs.PlateCarree(), levels=np.linspace(-1, 3, 25), cmap='RdYlBu', norm=divnorm, extend='both')
# css0 = axs0.contour(lon, lat, tqv_ctrl, levels=np.linspace(0, 300, 10), colors='k', linewidths=.7)
# manual_locations = [(146, 53), (130, 50), (125, 43), (105, 47), (115, 40), (147, 40), (110, 35), (110, 27), (135, 22)]
# axs0.clabel(css0, css0.levels[::1], inline=True, fontsize=8)
cs1 = axs1.contourf(lon, lat, w_topo1, transform=ccrs.PlateCarree(), levels=np.linspace(-1, 3, 25), cmap='RdYlBu', norm=divnorm, extend='both')
# css1 = axs1.contour(lon, lat, tqv_topo1, levels=np.linspace(0, 300, 10), colors='k', linewidths=.7)
# manual_locations = [(135, 50), (124, 48), (120, 45), (110, 49), (105, 47), (113, 35), (115, 30), (135, 27), (140, 20)]
# axs1.clabel(css1, css1.levels[::1], inline=True, fontsize=8)
divnorm=colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2)
cs2 = axs2.contourf(lon, lat, w_diff, transform=ccrs.PlateCarree(), levels=np.linspace(-2, 2, 21), cmap='RdYlBu', norm=divnorm, extend='both')
# axs2.contour(lon, lat, w_diff, cs2.levels, colors='k', linewidths=.1)

# add wind component
# q0 = axs0.quiver(lon[::15,::15], lat[::15,::15], u_ctrl[::15,::15], v_ctrl[::15,::15], transform=projection, color='red', scale=100000)
# axs0.quiverkey(q0, 0.9, 1.05, 10000, r'$10000 m/s$', labelpos='E', transform=axs0.transAxes)
# q1 = axs1.quiver(lon[::15,::15], lat[::15,::15], u_topo1[::15,::15], v_topo1[::15,::15], transform=projection, color='red', scale=100000)
# axs1.quiverkey(q1, 0.9, 1.05, 10000, r'$10000 m/s$', labelpos='E', transform=axs1.transAxes)
# q2 = axs2.quiver(lon[::15,::15], lat[::15,::15], u_diff[::15,::15], v_diff[::15,::15], transform=projection, color='k', scale=50000)
# axs2.quiverkey(q2, 0.9, 1.05, 500, r'$500 m/s$', labelpos='E', transform=axs2.transAxes)

axs0.set_title("Control", fontweight='bold', pad=10)
axs1.set_title("Reduced topography 1", fontweight='bold', pad=10)
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
plt.subplots_adjust(left=0.03, bottom=None, right=None, top=0.90, wspace=0.1, hspace=0)

cb1 = fig.colorbar(cs1, orientation='horizontal', shrink=0.9, aspect=70, ax=axs[0:2], pad=0.07)
cb2 = fig.colorbar(cs2, orientation='horizontal', shrink=0.9, aspect=30, ax=axs[2], pad=0.07)

cb1.set_label('$100^{-1} m s^{-1}$')
# cb1.set_ticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000])
cb2.set_label('$100^{-1} m s^{-1}$')

xmin, xmax = axs[0].get_xbound()
ymin, ymax = axs[0].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol * 1.4

fig.set_figheight(wi * y2x_ratio)
plt.show()

fig.savefig('figure_w_ivt.png', dpi=300)
