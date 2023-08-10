# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo, plotcosmo_notick, pole, colorbar
from mycolor import custom_div_cmap
import matplotlib.colors as colors
import cmcrameri.cm as cmc
from numpy import inf

# -------------------------------------------------------------------------------
# import data
#
seasons = ["DJF", "MAM", "JJA", "SON"]
mdvname = 'T_2M'  # edit here
year = '2001-2005'
pdpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn/T_2M/"
lgmpath = "/project/pr133/rxiang/data/cosmo/EAS11_lgm/szn/T_2M/"

# -------------------------------------------------------------------------------
# read PD data
#
pddata = []
for seas in range(len(seasons)):
    season = seasons[seas]
    filename = f'{year}.{mdvname}.{season}.nc'
    data = xr.open_dataset(f'{pdpath}{filename}')[mdvname].values[0, :, :] - 273.15
    pddata.append(data)

# -------------------------------------------------------------------------------
lgmdata = []
for seas in range(len(seasons)):
    season = seasons[seas]
    filename = f'{year}.{mdvname}.{season}.nc'
    data = xr.open_dataset(f'{lgmpath}{filename}')[mdvname].values[0, :, :] - 273.15
    lgmdata.append(data)

# -------------------------------------------------------------------------------
# compute difference
diffdata = {}
for seas in range(len(seasons)):
    diffdata[seas] = lgmdata[seas] - pddata[seas]


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
divnorm = colors.TwoSlopeNorm(vmin=np.amin(pddata), vcenter=0., vmax=np.amax(pddata))
for i in range(nrow):
    cs[i, 0] = axs[i, 0].pcolormesh(rlon, rlat, pddata[i], cmap='RdYlBu_r', norm=divnorm, shading="auto")
    axs[i, 0] = plotcosmo_notick(axs[i, 0])
    cs[i, 1] = axs[i, 1].pcolormesh(rlon, rlat, lgmdata[i], cmap='RdYlBu_r', norm=divnorm, shading="auto")
    axs[i, 1] = plotcosmo_notick(axs[i, 1])
divnorm = colors.TwoSlopeNorm(vmin=-6., vcenter=0., vmax=6)
cmap2 = custom_div_cmap(27, cmc.vik)
for i in range(nrow):
    cs[i, 2] = axs[i, 2].pcolormesh(rlon, rlat, diffdata[i], cmap=cmap2, norm=divnorm, shading="auto")
    axs[i, 2] = plotcosmo_notick(axs[i, 2])

# annual = np.nanmean(np.stack((diffdata[0], diffdata[1], diffdata[2], diffdata[3])), axis=0)
# cs[3, 2] = axs[3, 2].pcolormesh(rlon, rlat, annual, cmap=cmap2, norm=divnorm, shading="auto")

# -------------------------
# add title
axs[0, 0].set_title("PD", fontweight='bold', pad=10)
axs[0, 1].set_title("LGM", fontweight='bold', pad=10)
axs[0, 2].set_title("LGM - PD", fontweight='bold', pad=10)

# -------------------------
# add label
axs[0, 0].text(-0.16, 0.55, 'DJF', ha='center', va='center', rotation='vertical',
               transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')
axs[1, 0].text(-0.16, 0.55, 'MAM', ha='center', va='center', rotation='vertical',
               transform=axs[1, 0].transAxes, fontsize=13, fontweight='bold')
axs[2, 0].text(-0.16, 0.55, 'JJA', ha='center', va='center', rotation='vertical',
               transform=axs[2, 0].transAxes, fontsize=13, fontweight='bold')
axs[3, 0].text(-0.16, 0.55, 'SON', ha='center', va='center', rotation='vertical',
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
cax = colorbar(fig, axs[3, 0], 2, wspace)  # edit here
cb1 = fig.colorbar(cs[3, 0], cax=cax, orientation='horizontal')
cb1.set_label('$^{o}C$', fontsize=11)
cax = colorbar(fig, axs[3, 2], 1, wspace)  # edit here
cb1 = fig.colorbar(cs[3, 2], cax=cax, orientation='horizontal')
cb1.set_label('$^{o}C$', fontsize=11)
# cax = colorbar(fig, axs[3, 1], 1)
# cb2 = fig.colorbar(cs[3, 1], cax=cax, orientation='horizontal', extend='both')
# # # cb1.set_ticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000])
# cb2.set_label('%')

plt.show()
# -------------------------
# save figure
# plotpath = "/project/pr133/rxiang/figure/validation/"
# fig.savefig(plotpath + 'tmp04.png', dpi=300)
# plt.close(fig)
