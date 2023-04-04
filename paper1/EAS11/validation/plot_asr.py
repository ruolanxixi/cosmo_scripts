# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo, pole, colorbar, plotcosmo04
import matplotlib.colors as colors
from numpy import inf

# -------------------------------------------------------------------------------
# import data
#
seasons = ["DJF", "MAM", "JJA", "SON"]
mdvname1 = 'ASOB_T'  # edit here
mdvname2 = 'ASOD_T'  # edit here
year = '2001-2005'
mdpath1 = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn/ASOB_T/"
mdpath2 = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn/ASOD_T/"
erapath = "/project/pr133/rxiang/data/era5/ot/remap/"
cerespath = "/project/pr133/rxiang/data/obs/rd/CERES/remap/"

# -------------------------------------------------------------------------------
# read model data
#
mddata1, mddata2 = [], []
for seas in range(len(seasons)):
    season = seasons[seas]
    filename = f'{year}.{mdvname1}.{season}.nc'
    data = xr.open_dataset(f'{mdpath1}{filename}')[mdvname1].values[0, :, :]
    mddata1.append(data)
    filename = f'{year}.{mdvname2}.{season}.nc'
    data = xr.open_dataset(f'{mdpath2}{filename}')[mdvname2].values[0, :, :]
    mddata2.append(data)

# -------------------------------------------------------------------------------
# read era5 data
#
otdata = []
for seas in range(len(seasons)):
    season = seasons[seas]
    filename = f'era5.mo.2001-2005.{season}.remap.nc'
    data1 = xr.open_dataset(f'{erapath}{filename}')['mtnswrf'].values[0, :, :]
    data2 = xr.open_dataset(f'{erapath}{filename}')['mtdwswrf'].values[0, :, :]
    data = data2 - data1
    otdata.append(data)

# -------------------------------------------------------------------------------
# read observation data
#
for seas in range(len(seasons)):
    season = seasons[seas]
    filename = f'CERES.2001-2005.1.{season}.remap.nc'
    data = xr.open_dataset(f'{cerespath}{filename}')['toa_sw_all_mon'].values[0, :, :]
    otdata.append(data)

# -------------------------------------------------------------------------------
# compute difference
#
diffdata = []
for i in range(len(otdata)):
    j = i % 4
    data = mddata2[j] - mddata1[j] - otdata[i]
    diffdata.append(data)

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
    cs[i % 4, i // 4] = axs[i % 4, i // 4].pcolormesh(rlon, rlat, mddata2[i] - mddata1[i], cmap='RdYlBu_r', shading="auto")
    ax = plotcosmo04(axs[i % 4, i // 4])
divnorm = colors.TwoSlopeNorm(vmin=-100., vcenter=0., vmax=100.)
for i in np.arange(nrow, ncol * nrow, 1):
    cs[i % 4, i // 4] = axs[i % 4, i // 4].pcolormesh(rlon, rlat, diffdata[i-4], cmap='RdBu_r', norm=divnorm, shading="auto")
    ax = plotcosmo04(axs[i % 4, i // 4])

# -------------------------
# add title
axs[0, 0].set_title("COSMO", fontweight='bold', pad=10)
axs[0, 1].set_title("COSMO-ERA5", fontweight='bold', pad=10)
axs[0, 2].set_title("COSMO-CERES", fontweight='bold', pad=10)

# -------------------------
# add label
axs[0, 0].text(-0.14, 0.55, 'DJF', ha='center', va='center', rotation='vertical',
               transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')
axs[1, 0].text(-0.14, 0.55, 'MAM', ha='center', va='center', rotation='vertical',
               transform=axs[1, 0].transAxes, fontsize=13, fontweight='bold')
axs[2, 0].text(-0.14, 0.55, 'JJA', ha='center', va='center', rotation='vertical',
               transform=axs[2, 0].transAxes, fontsize=13, fontweight='bold')
axs[3, 0].text(-0.14, 0.55, 'SON', ha='center', va='center', rotation='vertical',
               transform=axs[3, 0].transAxes, fontsize=13, fontweight='bold')

# -------------------------
# adjust figure
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None)
xmin, xmax = axs[0, 0].get_xbound()
ymin, ymax = axs[0, 0].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol * 1.05
fig.set_figwidth(hi / y2x_ratio)
plt.subplots_adjust(left=0.05, bottom=0.08, right=0.98, top=0.95, wspace=0.08, hspace=0.13)

# -------------------------
# add colorbar
wspace=0.023
cax = colorbar(fig, axs[3, 0], 1, wspace)  # edit here
cb1 = fig.colorbar(cs[3, 0], cax=cax, orientation='horizontal')
cb1.set_label('$W/m^2$')
cax = colorbar(fig, axs[3, 1], 2, wspace)  # edit here
cb1 = fig.colorbar(cs[3, 1], cax=cax, orientation='horizontal')
cb1.set_label('$W/m^2$')
# cax = colorbar(fig, axs[3, 1], 1)
# cb2 = fig.colorbar(cs[3, 1], cax=cax, orientation='horizontal', extend='both')
# # # cb1.set_ticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000])
# cb2.set_label('%')

plt.show()
# -------------------------
# save figure
plotpath = "/project/pr133/rxiang/figure/validation/"
fig.savefig(plotpath + 'asr04.png', dpi=300)
plt.close(fig)
