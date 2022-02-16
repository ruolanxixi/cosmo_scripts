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
path1 = "/Users/kaktus/Documents/ETH/BECCY/myscripts/data/ctrl_ex/"
path2 = "/Users/kaktus/Documents/ETH/BECCY/myscripts/data/MERIT/"
path3 = "/Users/kaktus/Documents/ETH/BECCY/myscripts/data/ERA5/"

file1 = '01_T_2M_DJF.nc'
file2 = '01_T_2M_MAM.nc'
file3 = '01_T_2M_JJA.nc'
file4 = '01_T_2M_SON.nc'

ds = xr.open_dataset(path1 + file1)
tmp_ctrl_DJF = ds["T_2M"].values[0, :, :] - 273.15
lat = ds["lat"].values
lon = ds["lon"].values
pole_lat = ds["rotated_pole"].grid_north_pole_latitude
pole_lon = ds["rotated_pole"].grid_north_pole_longitude
rlat = ds["rlat"].values
rlon = ds["rlon"].values
ds.close()

ds = xr.open_dataset(path1 + file2)
tmp_ctrl_MAM = ds["T_2M"].values[0, :, :] - 273.15
ds.close()

ds = xr.open_dataset(path1 + file3)
tmp_ctrl_JJA = ds["T_2M"].values[0, :, :] - 273.15
ds.close()

ds = xr.open_dataset(path1 + file4)
tmp_ctrl_SON = ds["T_2M"].values[0, :, :] - 273.15
ds.close()

ds = xr.open_dataset(path2 + file1)
tmp_merit_DJF = ds["T_2M"].values[0, :, :] - 273.15
ds.close()

ds = xr.open_dataset(path2 + file2)
tmp_merit_MAM = ds["T_2M"].values[0, :, :] - 273.15
ds.close()

ds = xr.open_dataset(path2 + file3)
tmp_merit_JJA = ds["T_2M"].values[0, :, :] - 273.15
ds.close()

ds = xr.open_dataset(path2 + file4)
tmp_merit_SON = ds["T_2M"].values[0, :, :] - 273.15
ds.close()

ds = xr.open_dataset(path3 + "era5_2001_DJF.nc")
tmp_era_DJF = ds["t2m"].values[0, :, :] - 273.15
ds.close()

ds = xr.open_dataset(path3 + "era5_2001_MAM.nc")
tmp_era_MAM = ds["t2m"].values[0, :, :] - 273.15
ds.close()

ds = xr.open_dataset(path3 + "era5_2001_JJA.nc")
tmp_era_JJA = ds["t2m"].values[0, :, :] - 273.15
ds.close()

ds = xr.open_dataset(path3 + "era5_2001_SON.nc")
tmp_era_SON = ds["t2m"].values[0, :, :] - 273.15
ds.close()

tmp_diff_DJF = tmp_ctrl_DJF - tmp_merit_DJF
tmp_diff_MAM = tmp_ctrl_MAM - tmp_merit_MAM
tmp_diff_JJA = tmp_ctrl_JJA - tmp_merit_JJA
tmp_diff_SON = tmp_ctrl_SON - tmp_merit_SON

tmp_merit_era_DJF = tmp_merit_DJF - tmp_era_DJF
tmp_merit_era_MAM = tmp_merit_MAM - tmp_era_MAM
tmp_merit_era_JJA = tmp_merit_JJA - tmp_era_JJA
tmp_merit_era_SON = tmp_merit_SON - tmp_era_SON

tmp_ctrl_era_DJF = tmp_ctrl_DJF - tmp_era_DJF
tmp_ctrl_era_MAM = tmp_ctrl_MAM - tmp_era_MAM
tmp_ctrl_era_JJA = tmp_ctrl_JJA - tmp_era_JJA
tmp_ctrl_era_SON = tmp_ctrl_SON - tmp_era_SON
# -------------------------------------------------------------------------------
# plot
#
# pylab.rcParams['xtick.major.pad']='8'
# pylab.rcParams['ytick.major.pad']='8'

ar = 1.0  # initial aspect ratio for first trial
wi = 15  # width in inches
hi = wi * ar  # height in inches
ncol = 3
nrow = 4

gs = gridspec.GridSpec(nrow, ncol)
fig = plt.figure(figsize=(wi, hi))

proj = ccrs.PlateCarree()
rot_pole_crs = ccrs.RotatedPole(pole_latitude=pole_lat, pole_longitude=pole_lon)
axs0 = plt.subplot(gs[0], projection=rot_pole_crs)
axs1 = plt.subplot(gs[1], projection=rot_pole_crs)
axs2 = plt.subplot(gs[2], projection=rot_pole_crs)
axs3 = plt.subplot(gs[3], projection=rot_pole_crs)
axs4 = plt.subplot(gs[4], projection=rot_pole_crs)
axs5 = plt.subplot(gs[5], projection=rot_pole_crs)
axs6 = plt.subplot(gs[6], projection=rot_pole_crs)
axs7 = plt.subplot(gs[7], projection=rot_pole_crs)
axs8 = plt.subplot(gs[8], projection=rot_pole_crs)
axs9 = plt.subplot(gs[9], projection=rot_pole_crs)
axs10 = plt.subplot(gs[10], projection=rot_pole_crs)
axs11 = plt.subplot(gs[11], projection=rot_pole_crs)

divnorm = colors.TwoSlopeNorm(vmin=-20., vcenter=0., vmax=30)
cs0 = axs0.pcolormesh(rlon, rlat, tmp_ctrl_DJF, cmap='RdYlBu_r', norm=divnorm, shading="auto")
cs1 = axs1.pcolormesh(rlon, rlat, tmp_merit_DJF, cmap='RdYlBu_r', norm=divnorm, shading="auto")
cs2 = axs2.pcolormesh(rlon, rlat, tmp_diff_DJF, cmap='RdYlBu_r', vmin=-5, vmax=5, shading="auto")

cs3 = axs3.pcolormesh(rlon, rlat, tmp_ctrl_MAM, cmap='RdYlBu_r', norm=divnorm, shading="auto")
cs4 = axs4.pcolormesh(rlon, rlat, tmp_merit_MAM, cmap='RdYlBu_r', norm=divnorm, shading="auto")
cs5 = axs5.pcolormesh(rlon, rlat, tmp_diff_MAM, cmap='RdYlBu_r', vmin=-5, vmax=5, shading="auto")

cs6 = axs6.pcolormesh(rlon, rlat, tmp_ctrl_JJA, cmap='RdYlBu_r', norm=divnorm, shading="auto")
cs7 = axs7.pcolormesh(rlon, rlat, tmp_merit_JJA, cmap='RdYlBu_r', norm=divnorm, shading="auto")
cs8 = axs8.pcolormesh(rlon, rlat, tmp_diff_JJA, cmap='RdYlBu_r', vmin=-5, vmax=5, shading="auto")

cs9 = axs9.pcolormesh(rlon, rlat, tmp_ctrl_SON, cmap='RdYlBu_r', norm=divnorm, shading="auto")
cs10 = axs10.pcolormesh(rlon, rlat, tmp_merit_SON, cmap='RdYlBu_r', norm=divnorm, shading="auto")
cs11 = axs11.pcolormesh(rlon, rlat, tmp_diff_SON, cmap='RdYlBu_r', vmin=-5, vmax=5, shading="auto")

# add wind component
axs0.set_title("GLOBE", fontweight='bold', pad=10)
axs1.set_title("MERIT", fontweight='bold', pad=10)
axs2.set_title("Difference", fontweight='bold', pad=10)

axs0.text(0.03, 0.96, 'a', ha='center', va='center', transform=axs0.transAxes,
          fontsize=12, family='sans-serif')
axs1.text(0.03, 0.96, 'b', ha='center', va='center', transform=axs1.transAxes,
          fontsize=12, family='sans-serif')
axs2.text(0.03, 0.96, 'c', ha='center', va='center', transform=axs2.transAxes,
          fontsize=12, family='sans-serif')

axs = [axs0, axs1, axs2, axs3, axs4, axs5, axs6, axs7, axs8, axs9, axs10, axs11]
gl = []
for ax in axs:
    ax.set_extent([65, 175, 7, 62], crs=proj)
    ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

gl0 = axs[0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                       linewidth=1, color='grey', alpha=0.5, linestyle='--')
gl1 = axs[1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                       linewidth=1, color='grey', alpha=0.5, linestyle='--')
gl2 = axs[2].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                       linewidth=1, color='grey', alpha=0.5, linestyle='--')
gl3 = axs[3].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                       linewidth=1, color='grey', alpha=0.5, linestyle='--')
gl4 = axs[4].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                       linewidth=1, color='grey', alpha=0.5, linestyle='--')
gl5 = axs[5].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                       linewidth=1, color='grey', alpha=0.5, linestyle='--')
gl6 = axs[6].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                       linewidth=1, color='grey', alpha=0.5, linestyle='--')
gl7 = axs[7].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                       linewidth=1, color='grey', alpha=0.5, linestyle='--')
gl8 = axs[8].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                       linewidth=1, color='grey', alpha=0.5, linestyle='--')
gl9 = axs[9].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                       linewidth=1, color='grey', alpha=0.5, linestyle='--')
gl10 = axs[10].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                         linewidth=1, color='grey', alpha=0.5, linestyle='--')
gl11 = axs[11].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                         linewidth=1, color='grey', alpha=0.5, linestyle='--')

gls = [gl0, gl1, gl2, gl3, gl4, gl5, gl6, gl7, gl8, gl9, gl10, gl11]
for gl in gls:
    gl.right_labels = False
    gl.bottom_labels = False

#plt.tight_layout()
plt.subplots_adjust(left=0.07, bottom=None, right=None, top=0.93, wspace=0.12, hspace=0.15)


cax = fig.add_axes([axs9.get_position().x0 + 0.018, axs9.get_position().y0 - 0.04, axs9.get_position().width*2, 0.01])
cb1 = fig.colorbar(cs9, cax=cax, orientation='horizontal')
cax = fig.add_axes([axs11.get_position().x0 + 0.006, axs11.get_position().y0 - 0.04, axs11.get_position().width*0.95, 0.01])
cb2 = fig.colorbar(cs11, cax=cax, orientation='horizontal')

cb1.set_label('$^{o}C$')
# cb1.set_ticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000])
cb2.set_label('$^{o}C$')

xmin, xmax = axs[0].get_xbound()
ymin, ymax = axs[0].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol

fig.set_figheight(wi * y2x_ratio)
plt.show()

fig.savefig('figure_tmp.png', dpi=300)
