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
sims = ['ctrl', 'topo1', 'topo2']
all_rr_max, all_rr_max_sms = [], []

for s in range(len(sims)):
    sim = sims[s]
    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/TOT_PREC/smr'
    data = Dataset(f'{path}' + '/' + '01-05.TOT_PREC.smr.cpm.nc')
    ri = data.variables['TOT_PREC'][...]
    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/TOT_PREC/jan'
    data = Dataset(f'{path}' + '/' + '01-05.TOT_PREC.jan.tm.nc')
    jan = data.variables['TOT_PREC'][...]
    rr = ri - jan
    rr_max = np.nanmax(rr, axis=0)
    rr_max_sms = ndimage.gaussian_filter(rr_max, sigma=5, order=0)
    all_rr_max.append(rr_max)
    all_rr_max_sms.append(rr_max_sms)

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
left, bottom, right, top = 0.05, 0.05, 0.99, 0.90
gs = gridspec.GridSpec(2, 2, left=left, bottom=bottom, right=right, top=top, wspace=0.12, hspace=0.2)
axs[0] = fig.add_subplot(gs[0], projection=rot_pole_crs)
axs[1] = fig.add_subplot(gs[1], projection=rot_pole_crs)
axs[2] = fig.add_subplot(gs[3], projection=rot_pole_crs)

levels = MaxNLocator(nbins=15).tick_values(0, 30)
cmap = cmc.davos_r
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

sims = ['Control', 'Reduced topography', 'Envelope topography']

for i in range(3):
    sim = sims[i]
    axs[i] = plotcosmo(axs[i])
    cs[i] = axs[i].pcolormesh(rlon, rlat, all_rr_max[i], cmap=cmap, norm=norm, shading="auto")
    ct[i] = axs[i].contour(rlon, rlat, all_rr_max_sms[i], levels=np.linspace(5, 25, 3, endpoint=True), colors='maroon', linewidths=1)
    axs[i].text(0.02, 1.02, f'{sim}', ha='left', va='bottom', transform=axs[i].transAxes, fontsize=14)

cax = fig.add_axes([axs[0].get_position().x0, axs[0].get_position().y0 - 0.13, axs[0].get_position().width, 0.02])
cbar = fig.colorbar(cs[0], cax=cax, orientation='horizontal', extend='both')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('mm/day', fontsize=13)

fig.suptitle('Monsoon Annual Range', fontsize=16, fontweight='bold')

xmin, xmax = axs[1].get_xbound()
ymin, ymax = axs[1].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol + (1-top)
fig.set_figheight(wi * y2x_ratio)

plt.show()






