# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo, pole, colorbar
import matplotlib.colors as colors
from numpy import inf

# -------------------------------------------------------------------------------
# import data
#
seasons = ["DJF", "MAM", "JJA", "SON"]
mdvname = 'ALHFL_S'  # edit here
year = '01'
sim = ["GLOBE_ex_nofilt", "MERIT", "MERIT_raw","GLOBE_ex"]
datapath = "/Users/kaktus/Documents/ETH/BECCY/myscripts/data/"
erapath = "/Users/kaktus/Documents/ETH/BECCY/myscripts/data/ERA5/"

# -------------------------------------------------------------------------------
# read model data
#
mddata = []
for sims in range(len(sim)):
    simulation = sim[sims]
    mdpath = f'{datapath}' + f'{simulation}' + "/"
    for seas in range(len(seasons)):
        season = seasons[seas]
        filename = f'{year}_{mdvname}_{season}.nc'
        data = xr.open_dataset(f'{mdpath}{filename}')[mdvname].values[0, :, :]
        mddata.append(data)

# -------------------------------------------------------------------------------
# compute model data difference
#
np.seterr(divide='ignore', invalid='ignore')
for i in range(len(seasons)*len(sim)):
    if i // 4 == 0:
        mddata[i] = mddata[i]
    else:
        mddata[i] = (mddata[i % 4] - mddata[i]) / mddata[i % 4] * 100
np.seterr(divide='warn', invalid='warn')

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
ncol = len(sim)
nrow = 4
# gs = gridspec.GridSpec(nrow, ncol, figure=fig)
fig, axs = plt.subplots(nrow, ncol, figsize=(wi, hi), subplot_kw={'projection': rot_pole_crs})
cs = np.empty(shape=(nrow, ncol), dtype='object')
# -------------------------
# panel plot
divnorm = colors.TwoSlopeNorm(vmin=-50., vcenter=0., vmax=50)
for i in range(ncol * nrow):
    if i // 4 == 0:
        cs[i % 4, i // 4] = axs[i % 4, i // 4].pcolormesh(rlon, rlat, mddata[i], cmap='YlOrRd', shading="auto")
        ax = plotcosmo(axs[i % 4, i // 4])
    else:
        cs[i % 4, i // 4] = axs[i % 4, i // 4].pcolormesh(rlon, rlat, mddata[i], cmap='RdYlBu', norm=divnorm, shading="auto")
        ax = plotcosmo(axs[i % 4, i // 4])

# -------------------------
# add title
axs[0, 0].set_title("MERIT_raw", fontweight='bold', pad=10)
axs[0, 1].set_title("MERIT", fontweight='bold', pad=10)
axs[0, 2].set_title("GLOBE", fontweight='bold', pad=10)
axs[0, 3].set_title("GLOBE_filt", fontweight='bold', pad=10)
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
# add colorbar
cax = colorbar(fig, axs[3, 0], 1)  # edit here
cb1 = fig.colorbar(cs[3, 0], cax=cax, orientation='horizontal')
cb1.set_label('$W m^{-2}$')
cax = colorbar(fig, axs[3, 1], 3)
cb2 = fig.colorbar(cs[3, 1], cax=cax, orientation='horizontal')
# # # cb1.set_ticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000])
cb2.set_label('$W m^{-2}$')
# -------------------------
# adjust figure
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None)
xmin, xmax = axs[0, 0].get_xbound()
ymin, ymax = axs[0, 0].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol
fig.set_figwidth(hi / y2x_ratio)

plt.show()
# -------------------------
# save figure
plotpath = "/Users/kaktus/Documents/ETH/BECCY/myscripts/figure/"
fig.savefig(plotpath + 'compare_topo_lhf.png', dpi=300)
plt.close(fig)
