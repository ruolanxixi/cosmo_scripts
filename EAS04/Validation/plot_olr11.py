# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo04, pole04, pole, colorbar, custom_div_cmap
import matplotlib.colors as colors
from numpy import inf
import matplotlib

font = {'size': 14}
matplotlib.rc('font', **font)
# -------------------------------------------------------------------------------
# import data
#
seasons = ["DJF", "MAM", "JJA", "SON"]
mdvname = 'ATHB_T'  # edit here
year = '2001-2005'
mdpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn/ATHB_T/"
erapath = "/project/pr133/rxiang/data/era5/ot/remap/"
cerespath = "/project/pr133/rxiang/data/obs/rd/CERES/remap/"

# -------------------------------------------------------------------------------
# read model data
#
mddata = []
for seas in range(len(seasons)):
    season = seasons[seas]
    filename = f'{year}.{mdvname}.{season}.nc'
    data = xr.open_dataset(f'{mdpath}{filename}')[mdvname].values[0, :, :]
    mddata.append(data)

# -------------------------------------------------------------------------------
# read era5 data
#
otdata = []
for seas in range(len(seasons)):
    season = seasons[seas]
    filename = f'era5.mo.2001-2005.{season}.remap.nc'
    data = xr.open_dataset(f'{erapath}{filename}')['mtnlwrf'].values[0, :, :]
    otdata.append(data)

# -------------------------------------------------------------------------------
# read observation data
#
for seas in range(len(seasons)):
    season = seasons[seas]
    filename = f'CERES.2001-2005.1.{season}.remap.nc'
    data = xr.open_dataset(f'{cerespath}{filename}')['toa_lw_all_mon'].values[0, :, :]
    data = - data
    otdata.append(data)

# -------------------------------------------------------------------------------
# compute difference
#
diffdata = []
for i in range(len(otdata)):
    j = i % 4
    data = mddata[j] - otdata[i]
    diffdata.append(data)
np.seterr(divide='warn', invalid='warn')

bias = np.arange(0, len(diffdata), 1.0)
for i in range(len(otdata)):
    bias[i] = np.nanmean(diffdata[i][108:346, 239:476])

# -------------------------------------------------------------------------------
# plot
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()
#
ar = 1.0  # initial aspect ratio for first trial
hi = 14  # height in inches
wi = hi / ar  # width in inches
# fig = plt.figure(figsize=(wi, hi))
#
ncol = 3  # edit here
nrow = 4
# gs = gridspec.GridSpec(nrow, ncol, figure=fig)
fig, axs = plt.subplots(nrow, ncol, figsize=(wi, hi), subplot_kw={'projection': rot_pole_crs})
cs = np.empty(shape=(nrow, ncol), dtype='object')
# -------------------------
# panel plot
for i in range(nrow):
    cs[i % 4, i // 4] = axs[i % 4, i // 4].pcolormesh(rlon, rlat, mddata[i], cmap='RdYlBu_r', shading="auto")
    ax = plotcosmo04(axs[i % 4, i // 4])
divnorm = colors.TwoSlopeNorm(vmin=-30., vcenter=0., vmax=60.)
for i in np.arange(nrow, ncol * nrow, 1):
    cs[i % 4, i // 4] = axs[i % 4, i // 4].pcolormesh(rlon, rlat, diffdata[i-4], cmap='RdBu_r', norm=divnorm, shading="auto")
    ax = plotcosmo04(axs[i % 4, i // 4])

    n = round(bias[i - 4], 2)
    t = ax.text(0.97, 0.91, f'bias = {n}', horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes, fontsize=14, zorder=4)
    rect = plt.Rectangle((0.55, 0.91), width=0.45, height=0.08,
                         transform=ax.transAxes, zorder=3,
                         fill=True, facecolor="white", alpha=0.7, clip_on=False)
    ax.add_patch(rect)

# -------------------------
# add title
axs[0, 0].set_title("COSMO", fontweight='bold', pad=10)
axs[0, 1].set_title("COSMO-ERA5", fontweight='bold', pad=10)
axs[0, 2].set_title("COSMO-CERES", fontweight='bold', pad=10)

# -------------------------
# add label
axs[0, 0].text(-0.25, 0.55, 'DJF', ha='center', va='center', rotation='vertical',
               transform=axs[0, 0].transAxes, fontsize=14, fontweight='bold')
axs[1, 0].text(-0.25, 0.55, 'MAM', ha='center', va='center', rotation='vertical',
               transform=axs[1, 0].transAxes, fontsize=14, fontweight='bold')
axs[2, 0].text(-0.25, 0.55, 'JJA', ha='center', va='center', rotation='vertical',
               transform=axs[2, 0].transAxes, fontsize=14, fontweight='bold')
axs[3, 0].text(-0.25, 0.55, 'SON', ha='center', va='center', rotation='vertical',
               transform=axs[3, 0].transAxes, fontsize=14, fontweight='bold')

# -------------------------
# adjust figure
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None)
xmin, xmax = axs[0, 0].get_xbound()
ymin, ymax = axs[0, 0].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol * 1 * 0.98
fig.set_figwidth(hi / y2x_ratio)
plt.subplots_adjust(left=0.07, bottom=0.08, right=0.98, top=0.95, wspace=0.17, hspace=0.12)

# -------------------------
# add colorbar
wspace=0.057
cax = colorbar(fig, axs[3, 0], 1, wspace)  # edit here
cb1 = fig.colorbar(cs[3, 0], cax=cax, orientation='horizontal', extend='both')
cb1.set_label('$W/m^2$', fontsize=11)
cax = colorbar(fig, axs[3, 1], 2, wspace)  # edit here
cb1 = fig.colorbar(cs[3, 1], cax=cax, orientation='horizontal', extend='both')
cb1.set_label('$W/m^2$', fontsize=11)
# cax = colorbar(fig, axs[3, 1], 1)
# cb2 = fig.colorbar(cs[3, 1], cax=cax, orientation='horizontal', extend='both')
# # # cb1.set_ticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000])
# cb2.set_label('%')

plt.show()
# -------------------------
# save figure
plotpath = "/project/pr133/rxiang/figure/val04/"
fig.savefig(plotpath + 'olr11.png', dpi=500)
plt.close(fig)
