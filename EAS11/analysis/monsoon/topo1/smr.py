# -------------------------------------------------------------------------------
# modules
#
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo, pole, colorbar
import cartopy.crs as ccrs
from numpy import inf
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap
from pyproj import Transformer
import scipy.ndimage as ndimage


# -------------------------------------------------------------------------------
# read data
#
sims = ['ctrl', 'topo1']
all_smr, all_smr_sms = [], []

for s in range(len(sims)):
    sim = sims[s]
    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/TOT_PREC/smr'
    data = Dataset(f'{path}' + '/' + '01-05.TOT_PREC.smr.nc')
    smr = data.variables['TOT_PREC'][...]
    smr = np.nanmean(smr, axis=0)
    smr_sms = ndimage.gaussian_filter(smr, sigma=10, order=0)
    all_smr.append(smr)
    all_smr_sms.append(smr_sms)

# -------------------------------------------------------------------------------
# plot
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

ar = 1.0  # initial aspect ratio for first trial
wi = 11  # height in inches
hi = wi * ar  # width in inches
ncol = 2  # edit here
nrow = 2
axs, cs, ct = np.empty(3, dtype='object'), np.empty(3, dtype='object'), np.empty(3, dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.05, 0.11, 0.99, 0.93
gs = gridspec.GridSpec(2, 2, left=left, bottom=bottom, right=right, top=top, wspace=0.12, hspace=0.1)
axs[0] = fig.add_subplot(gs[0], projection=rot_pole_crs)
axs[1] = fig.add_subplot(gs[2], projection=rot_pole_crs)
axs[2] = fig.add_subplot(gs[3], projection=rot_pole_crs)

levels = MaxNLocator(nbins=15).tick_values(0, 15)
cmap = cmc.davos_r
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

sims = ['Control', 'Reduced topography']

for i in range(2):
    sim = sims[i]
    axs[i] = plotcosmo(axs[i])
    cs[i] = axs[i].pcolormesh(rlon, rlat, all_smr[i], cmap=cmap, norm=norm, shading="auto")
    ct[i] = axs[i].contour(rlon, rlat, all_smr_sms[i], levels=np.linspace(2, 11, 4, endpoint=True), colors='maroon', linewidths=1)
    axs[i].text(0.02, 1.02, f'{sim}', ha='left', va='bottom', transform=axs[i].transAxes, fontsize=14)

    # trans = Transformer.from_proj(ccrs.PlateCarree(), rot_pole_crs, always_xy=True)
    # x = np.array([90, 102, 152, 140, 105, 147, 70])
    # y = np.array([35, 33, 29, 21, 23, 10, 9])
    # loc_lon, loc_lat = trans.transform(x, y)
    # manual_locations = [i for i in zip(loc_lon, loc_lat)]
    clabel = axs[i].clabel(ct[i], [2., 5., 8., 11.], inline=True, fontsize=13, use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

cax = fig.add_axes([axs[1].get_position().x0, axs[1].get_position().y0 - bottom, axs[1].get_position().width, 0.02])
cbar = fig.colorbar(cs[1], cax=cax, orientation='horizontal', extend='max')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('mm/day', fontsize=13)

# plot difference
levels = MaxNLocator(nbins=15).tick_values(-5, 5)
cmap = custom_div_cmap(25, cmc.broc_r)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

axs[2] = plotcosmo(axs[2])
cs[2] = axs[2].pcolormesh(rlon, rlat, all_smr[0] - all_smr[1], cmap=cmap, clim=(-5, 5), shading="auto")
ct[2] = axs[2].contour(rlon, rlat, all_smr_sms[0] - all_smr_sms[1], levels=np.linspace(-2, 2, 3, endpoint=True), colors='maroon',
                       linewidths=1)
axs[2].text(0.02, 1.02, 'Control - Reduced topography', ha='left', va='bottom', transform=axs[2].transAxes, fontsize=14)

# trans = Transformer.from_proj(ccrs.PlateCarree(), rot_pole_crs, always_xy=True)
# x = np.array([90, 90, 119, 113, 140, 138, 160, 170, 90])
# y = np.array([35, 27, 28, 22, 29, 13, 15, 42, 57])
# loc_lon, loc_lat = trans.transform(x, y)
# manual_locations = [i for i in zip(loc_lon, loc_lat)]
# clabel = axs[2].clabel(ct[2], [-2., 0, 2], inline=True, fontsize=13, manual=manual_locations)
clabel = axs[2].clabel(ct[2], [-2., 0, 2], inline=True, use_clabeltext=True, fontsize=13)
for l in clabel:
    l.set_rotation(0)
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.7)) for txt in clabel]

cax = fig.add_axes([axs[2].get_position().x0, axs[2].get_position().y0 - bottom, axs[2].get_position().width, 0.02])
cbar = fig.colorbar(cs[2], cax=cax, orientation='horizontal', extend='both')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('mm/day', fontsize=13)

fig.suptitle('Total Rainfall May to Sep', fontsize=16, fontweight='bold')

xmin, xmax = axs[1].get_xbound()
ymin, ymax = axs[1].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol + 0.15
fig.set_figheight(wi * y2x_ratio)

# fig.show()
plotpath = "/project/pr133/rxiang/figure/monsoon/topo1/"
fig.savefig(plotpath + 'smr.png', dpi=500)
plt.close(fig)






