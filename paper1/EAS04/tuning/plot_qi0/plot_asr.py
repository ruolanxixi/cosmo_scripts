# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo04, pole04, pole, colorbar, custom_div_cmap
import matplotlib.colors as colors
from numpy import inf
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# -------------------------------------------------------------------------------
# import data
#
mdnames = ['5e-6', '2.5e-6', '1e-6', '0']
mdvname1 = 'ASOB_T'  # edit here
mdvname2 = 'ASOD_T'  # edit here
year = '200106'
mdpath = "/project/pr133/rxiang/data/cosmo/"
md11path = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/200106/"
erapath = "/project/pr133/rxiang/data/era5/ot/remap/"
cerespath = "/project/pr133/rxiang/data/obs/rd/CERES/remap/"

# -------------------------------------------------------------------------------
# read model data
#
mddata1, mddata2 = [], []
for mds in range(len(mdnames)):
    md = mdnames[mds]
    filename = f'{mdvname1}.nc'
    data = xr.open_dataset(f'{mdpath}EAS04_qi0/{year}/{md}/{filename}')[mdvname1].values[0, :, :]
    mddata1.append(data)
    filename = f'{mdvname2}.nc'
    data = xr.open_dataset(f'{mdpath}EAS04_qi0/{year}/{md}/{filename}')[mdvname2].values[0, :, :]
    mddata2.append(data)

# -------------------------------------------------------------------------------
# read era5 data
#
filename = f'era5.mo.{year}.remap.04.nc'
data1 = xr.open_dataset(f'{erapath}{filename}')['mtnswrf'].values[0, :, :]
data2 = xr.open_dataset(f'{erapath}{filename}')['mtdwswrf'].values[0, :, :]
eradata = data1 - data2

# -------------------------------------------------------------------------------
# read observation data
#

filename = f'CERES.{year}.1.remap.04.nc'
otdata = - xr.open_dataset(f'{cerespath}{filename}')['toa_sw_all_mon'].values[0, :, :]

# -------------------------------------------------------------------------------
# read EAS11 data
#
md11data1 = xr.open_dataset(f'{md11path}/{mdvname1}.nc')[mdvname1].values[0, :, :]
md11data2 = xr.open_dataset(f'{md11path}/{mdvname2}.nc')[mdvname2].values[0, :, :]
md11data = md11data1 - md11data2
filename = f'era5.mo.{year}.remap.11.nc'
data1 = xr.open_dataset(f'{erapath}{filename}')['mtnswrf'].values[0, :, :]
data2 = xr.open_dataset(f'{erapath}{filename}')['mtdwswrf'].values[0, :, :]
era11data = data1 - data2
filename = f'CERES.{year}.1.remap.11.nc'
ot11data = - xr.open_dataset(f'{cerespath}{filename}')['toa_sw_all_mon'].values[0, :, :]

diff11data = []
data = md11data - era11data
diff11data.append(data)
data = md11data - ot11data
diff11data.append(data)

bias11 = np.arange(1.0, 3.0, 1.0)
for i in range(2):
    bias11[i] = np.nanmean(diff11data[i][108:346, 239:476])

# -------------------------------------------------------------------------------
# compute difference
#
diffdata = []
for i in range(len(mdnames)):
    data = mddata1[i] - mddata2[i] - eradata
    diffdata.append(data)
for i in range(len(mdnames)):
    data = mddata1[i] - mddata2[i] - otdata
    diffdata.append(data)

bias = np.arange(1.0, 9.0, 1.0)
for i in range(8):
    bias[i] = np.mean(diffdata[i])

# -------------------------------------------------------------------------------
# plot
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole04()
[pole_lat11, pole_lon11, lat11, lon11, rlat11, rlon11, rot_pole_crs11] = pole()
#
ar = 1.0  # initial aspect ratio for first trial
hi = 14  # height in inches
wi = hi / ar  # width in inches
# fig = plt.figure(figsize=(wi, hi))
#
ncol = 3  # edit here
nrow = 5
# gs = gridspec.GridSpec(nrow, ncol, figure=fig)
fig, axs = plt.subplots(nrow, ncol, figsize=(wi, hi), subplot_kw={'projection': rot_pole_crs})
cs = np.empty(shape=(nrow, ncol), dtype='object')
# -------------------------
# panel plot
for i in range(nrow-1):
    cs[i % nrow, i // nrow] = axs[i % nrow, i // nrow].pcolormesh(rlon, rlat, mddata1[i] - mddata2[i], cmap=cmc.roma_r, shading="auto")
    ax = plotcosmo04(axs[i % nrow, i // nrow])

cs[4, 0] = axs[4, 0].pcolormesh(rlon11, rlat11, md11data, cmap=cmc.roma_r, shading="auto")
ax = plotcosmo04(axs[4, 0])

cmap = custom_div_cmap(27, cmc.vik)
levels = MaxNLocator(nbins=26).tick_values(-40, 100)
norm = colors.TwoSlopeNorm(vmin=-100., vcenter=0., vmax=60.)
# cs[1, 0] = axs[1, 0].pcolormesh(rlon, rlat, mddata1[0] - mddata2[0] - mddata1[1] + mddata2[1], cmap=cmap, norm=norm, shading="auto")
for i in np.arange(nrow, ncol * nrow, 1):
    if (i % nrow) != 4:
        cs[i % nrow, i // nrow] = axs[i % nrow, i // nrow].pcolormesh(rlon, rlat, diffdata[i%nrow+(i // nrow-1)*4], cmap=cmap, norm=norm, shading="auto")
        ax = plotcosmo04(axs[i % nrow, i // nrow])
        n = round(bias[i%nrow+(i // nrow-1)*4], 2)
        t = ax.text(0.97, 0.90, f'bias = {n}', horizontalalignment='right', verticalalignment='bottom',
                    transform=ax.transAxes, fontsize=13, zorder=4)
        rect = plt.Rectangle((0.425, 0.91), width=0.56, height=0.08,
                             transform=ax.transAxes, zorder=3,
                             fill=True, facecolor="white", alpha=0.7, clip_on=False)
        ax.add_patch(rect)
    else:
        cs[i % nrow, i // nrow] = axs[i % nrow, i // nrow].pcolormesh(rlon11, rlat11, diff11data[i // nrow - 1], cmap=cmap,
                                                                      norm=norm, shading="auto")
        ax = plotcosmo04(axs[i % nrow, i // nrow])
        n = round(bias11[i // nrow - 1], 2)
        t = ax.text(0.97, 0.90, f'bias = {n}', horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes, fontsize=13, zorder=4)
        rect = plt.Rectangle((0.425, 0.91), width=0.56, height=0.08,
                             transform=ax.transAxes, zorder=3,
                             fill=True, facecolor="white", alpha=0.7, clip_on=False)
        ax.add_patch(rect)

# -------------------------
# add title
axs[0, 0].set_title("COSMO", fontweight='bold', pad=10, fontsize=14)
axs[0, 1].set_title("COSMO-ERA5", fontweight='bold', pad=10, fontsize=14)
axs[0, 2].set_title("COSMO-CERES", fontweight='bold', pad=10, fontsize=14)

# -------------------------
# add label
axs[0, 0].text(-0.24, 0.55, 'qi0 = 5E-6', ha='center', va='center', rotation='vertical',
               transform=axs[0, 0].transAxes, fontsize=14, fontweight='bold')
axs[1, 0].text(-0.24, 0.55, 'qi0 = 2.5E-6', ha='center', va='center', rotation='vertical',
               transform=axs[1, 0].transAxes, fontsize=14, fontweight='bold')
axs[2, 0].text(-0.24, 0.55, 'qi0 = 1E-6', ha='center', va='center', rotation='vertical',
               transform=axs[2, 0].transAxes, fontsize=14, fontweight='bold')
axs[3, 0].text(-0.24, 0.55, 'qi0 = 0', ha='center', va='center', rotation='vertical',
               transform=axs[3, 0].transAxes, fontsize=14, fontweight='bold')
axs[4, 0].text(-0.24, 0.55, 'EAS11', ha='center', va='center', rotation='vertical',
               transform=axs[4, 0].transAxes, fontsize=14, fontweight='bold')
# -------------------------
# adjust figure
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None)
xmin, xmax = axs[0, 0].get_xbound()
ymin, ymax = axs[0, 0].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol * 0.94
fig.set_figwidth(hi / y2x_ratio)
plt.subplots_adjust(left=0.05, bottom=0.08, right=0.99, top=0.95, wspace=0.08, hspace=0.12)

# -------------------------
# add colorbar
wspace=0.053
cax = colorbar(fig, axs[nrow-1, 0], 1, wspace)  # edit here
cb1 = fig.colorbar(cs[0, 0], cax=cax, orientation='horizontal', extend='both')
cb1.set_label('$W/m^2$', fontsize=12)
cb1.ax.tick_params(labelsize=12)
cax = colorbar(fig, axs[nrow-1, 1], 2, wspace)  # edit here
cb2 = fig.colorbar(cs[1, 1], cax=cax, orientation='horizontal', ticks=np.append(np.linspace(-100, 0, 6), np.linspace(10, 60, 6)), extend='both')
cb2.set_label('$W/m^2$', fontsize=12)
cb2.ax.tick_params(labelsize=12)
# cax = colorbar(fig, axs[3, 1], 1)
# cb2 = fig.colorbar(cs[3, 1], cax=cax, orientation='horizontal', extend='both')
# # # cb1.set_ticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000])
# cb2.set_label('%')

plt.show()
# -------------------------
# save figure
plotpath = "/project/pr133/rxiang/figure/qi0/"
fig.savefig(plotpath + 'asr.png', dpi=300)
plt.close(fig)

