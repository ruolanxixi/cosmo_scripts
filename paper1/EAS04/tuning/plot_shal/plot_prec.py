# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo04, pole04, colorbar, custom_div_cmap
import matplotlib.colors as colors
from numpy import inf
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# -------------------------------------------------------------------------------
# import data
#
mdnames = ['ctrl', 'SHAL']
mdvname = 'TOT_PREC'  # edit here
year = '200106'
mdpath = "/project/pr133/rxiang/data/cosmo/"
erapath = "/project/pr133/rxiang/data/era5/pr/remap/"
crupath = "/project/pr133/rxiang/data/obs/pr/cru/remap/"
imergpath = "/project/pr133/rxiang/data/obs/pr/IMERG/remap/"

# -------------------------------------------------------------------------------
# read model data
#
mddata = []
for mds in range(len(mdnames)):
    md = mdnames[mds]
    filename = f'{mdvname}.nc'
    data = xr.open_dataset(f'{mdpath}EAS04_{md}/{year}/{filename}')[mdvname].values[0, :, :]
    mddata.append(data)

# -------------------------------------------------------------------------------
# read era5 data
#
filename = f'era5.mo.{year}.remap.04.nc'
eradata = xr.open_dataset(f'{erapath}{filename}')['tp'].values[0, :, :] * 1000

# -------------------------------------------------------------------------------
# read observation data
#
filename = f'cru.{year}.05.remap.04.nc'
crudata = xr.open_dataset(f'{crupath}{filename}')['pre'].values[0, :, :]

filename = f'{year}.remap.04.nc4'
imergdata = xr.open_dataset(f'{imergpath}{filename}')['pr'].values[0, :, :]

# -------------------------------------------------------------------------------
# compute difference
#
diffdata = []
for i in range(len(mdnames)):
    data = mddata[i] - eradata
    diffdata.append(data)
for i in range(len(mdnames)):
    data = mddata[i] - crudata
    diffdata.append(data)
for i in range(len(mdnames)):
    data = mddata[i] - imergdata
    diffdata.append(data)

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
nrow = 2
# gs = gridspec.GridSpec(nrow, ncol, figure=fig)
fig, axs = plt.subplots(nrow, ncol, figsize=(wi, hi), subplot_kw={'projection': rot_pole_crs})
cs = np.empty(shape=(nrow, ncol), dtype='object')
# -------------------------
# panel plot
for i in range(nrow):
    cs[i % 2, i // 2] = axs[i % 2, i // 2].pcolormesh(rlon, rlat, mddata[i], cmap=cmc.davos_r, clim=(0, 30), shading="auto")
    ax = plotcosmo04(axs[i % 2, i // 2])

cmap = custom_div_cmap(27, cmc.broc_r)
norm = colors.TwoSlopeNorm(vmin=-15., vcenter=0., vmax=15.)
cs[1, 0] = axs[1, 0].pcolormesh(rlon, rlat, mddata[0] - mddata[1], cmap=cmap, norm=norm, shading="auto")
for i in np.arange(nrow, ncol * nrow, 1):
    cs[i % 2, i // 2] = axs[i % 2, i // 2].pcolormesh(rlon, rlat, diffdata[i-2], cmap=cmap, norm=norm, shading="auto")
    ax = plotcosmo04(axs[i % 2, i // 2])

# -------------------------
# add title
axs[0, 0].set_title("COSMO", fontweight='bold', pad=10, fontsize=14)
axs[0, 1].set_title("COSMO-ERA5", fontweight='bold', pad=10, fontsize=14)
axs[0, 2].set_title("COSMO-CRU", fontweight='bold', pad=10, fontsize=14)
axs[0, 3].set_title("COSMO-IMERG", fontweight='bold', pad=10, fontsize=14)

# -------------------------
# add label
axs[0, 0].text(-0.14, 0.55, 'Explicit', ha='center', va='center', rotation='vertical',
               transform=axs[0, 0].transAxes, fontsize=14, fontweight='bold')
axs[1, 0].text(-0.14, 0.55, 'Explicit - Shallow', ha='center', va='center', rotation='vertical',
               transform=axs[1, 0].transAxes, fontsize=14, fontweight='bold')
# -------------------------
# adjust figure
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None)
xmin, xmax = axs[0, 0].get_xbound()
ymin, ymax = axs[0, 0].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol * 1.05
fig.set_figwidth(hi / y2x_ratio)
plt.subplots_adjust(left=0.05, bottom=0.08, right=0.98, top=0.95, wspace=0.1, hspace=0.09)

# -------------------------
# add colorbar
wspace=0.021
cax = colorbar(fig, axs[1, 0], 1, wspace)  # edit here
cb1 = fig.colorbar(cs[0, 0], cax=cax, orientation='horizontal', extend='max')
cb1.set_label('mm/day', fontsize=12)
cb1.ax.tick_params(labelsize=12)
cax = colorbar(fig, axs[1, 1], 3, wspace)  # edit here
cb2 = fig.colorbar(cs[1, 1], cax=cax, orientation='horizontal', ticks=np.linspace(-15, 15, 11), extend='both')
cb2.set_label('mm/day', fontsize=12)
cb2.ax.tick_params(labelsize=12)
# cax = colorbar(fig, axs[3, 1], 1)
# cb2 = fig.colorbar(cs[3, 1], cax=cax, orientation='horizontal', extend='both')
# # # cb1.set_ticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000])
# cb2.set_label('%')

plt.show()
# -------------------------
# save figure
plotpath = "/project/pr133/rxiang/figure/shal/"
fig.savefig(plotpath + 'pr.png', dpi=300)
plt.close(fig)

for i in range(6):
    print(np.nanmean(diffdata[i]))
