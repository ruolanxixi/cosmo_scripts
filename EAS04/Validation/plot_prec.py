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
mdvname = 'TOT_PREC'  # edit here
year = '2001-2005'
mdpath = "/scratch/snx3000/rxiang/data/cosmo/EAS04_ctrl/szn/TOT_PREC/"
erapath = "/project/pr133/rxiang/data/era5/pr/remap/"
crupath = "/project/pr133/rxiang/data/obs/pr/cru/remap/"
imergpath = "/project/pr133/rxiang/data/obs/pr/IMERG/remap/"

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
    filename = f'era5.mo.2001-2005.{season}.remap.04.nc'
    data = xr.open_dataset(f'{erapath}{filename}')['tp'].values[0, :, :] * 1000
    otdata.append(data)

# -------------------------------------------------------------------------------
# read observation data
#
for seas in range(len(seasons)):
    season = seasons[seas]
    filename = f'cru.2001-2005.05.{season}.remap.04.nc'
    data = xr.open_dataset(f'{crupath}{filename}')['pre'].values[0, :, :]
    otdata.append(data)

for seas in range(len(seasons)):
    season = seasons[seas]
    filename = f'IMERG.ydaymean.2001-2005.{season}.remap.04.nc4'
    data = xr.open_dataset(f'{imergpath}{filename}')['pr'].values[0, :, :]
    otdata.append(data)

# -------------------------------------------------------------------------------
# compute difference
#
diffdata = []
for i in range(len(otdata)):
    j = i % 4
    data = mddata[j] - otdata[i]
    diffdata.append(data)

bias = np.arange(0, len(diffdata), 1.0)
for i in range(len(otdata)):
    bias[i] = np.nanmean(diffdata[i])

# -------------------------------------------------------------------------------
# plot
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole04()
#
ar = 1.0  # initial aspect ratio for first trial
hi = 14  # height in inches
wi = hi / ar  # width in inches
# fig = plt.figure(figsize=(wi, hi))
#
ncol = 4  # edit here
nrow = 4
fig, axs = plt.subplots(nrow, ncol, figsize=(wi, hi), subplot_kw={'projection': rot_pole_crs})
cs = np.empty(shape=(nrow, ncol), dtype='object')
# -------------------------
# panel plot
divnorm = colors.TwoSlopeNorm(vmin=-15., vcenter=0., vmax=15)
for i in range(nrow):
    axs[i % 4, i // 4] = plotcosmo04(axs[i % 4, i // 4])
    cs[i % 4, i // 4] = axs[i % 4, i // 4].pcolormesh(rlon, rlat, mddata[i], cmap='YlGnBu', vmin=0, vmax=20, shading="auto")
for i in np.arange(nrow, ncol * nrow, 1):
    axs[i % 4, i // 4] = plotcosmo04(axs[i % 4, i // 4])
    cs[i % 4, i // 4] = axs[i % 4, i // 4].pcolormesh(rlon, rlat, diffdata[i-4], cmap='RdBu', norm=divnorm, shading="auto")

    n = round(bias[i - 4], 2)
    t = axs[i % 4, i // 4].text(0.97, 0.91, f'bias = {n}', horizontalalignment='right', verticalalignment='bottom',
                transform=axs[i % 4, i // 4].transAxes, fontsize=14, zorder=4)
    rect = plt.Rectangle((0.55, 0.91), width=0.45, height=0.08,
                         transform=axs[i % 4, i // 4].transAxes, zorder=3,
                         fill=True, facecolor="white", alpha=0.7, clip_on=False)
    axs[i % 4, i // 4].add_patch(rect)
# -------------------------
# add title
axs[0, 0].set_title("COSMO", fontweight='bold', pad=10)
axs[0, 1].set_title("COSMO-ERA5", fontweight='bold', pad=10)
axs[0, 2].set_title("COSMO-CRU", fontweight='bold', pad=10)
axs[0, 3].set_title("COSMO-IMERG", fontweight='bold', pad=10)

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
xmin, xmax = axs[0, 0].get_xbound()
ymin, ymax = axs[0, 0].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol * 1 * 0.98
fig.set_figwidth(hi / y2x_ratio)
plt.subplots_adjust(left=0.07, bottom=0.08, right=0.98, top=0.95, wspace=0.17, hspace=0.12)

# -------------------------
# add colorbar
wspace=0.04
cax = colorbar(fig, axs[3, 0], 1, wspace)  # edit here
cb1 = fig.colorbar(cs[3, 0], cax=cax, orientation='horizontal', extend='max')
cb1.set_label('mm/day')
cax = colorbar(fig, axs[3, 1], 3, wspace)  # edit here
cb1 = fig.colorbar(cs[3, 1], cax=cax, orientation='horizontal', extend='both')
cb1.set_label('mm/day')
# cax = colorbar(fig, axs[3, 1], 1)
# cb2 = fig.colorbar(cs[3, 1], cax=cax, orientation='horizontal', extend='both')
# # # cb1.set_ticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000])
# cb2.set_label('%')

plt.show()
# -------------------------
# save figure
plotpath = "/project/pr133/rxiang/figure/val04/"
fig.savefig(plotpath + 'pr.png', dpi=300)
plt.close(fig)
