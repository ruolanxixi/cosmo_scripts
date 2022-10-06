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
all_smr, all_rt, all_rg = [], [], []
all_smr_sms, all_rt_sms, all_rg_sms = [], [], []

for s in range(len(sims)):
    sim = sims[s]
    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/TOT_PREC/smr'
    data = Dataset(f'{path}' + '/' + '01-05.TOT_PREC.smr.nc')
    smr = data.variables['TOT_PREC'][...]
    smr = np.nansum(smr, axis=0)
    smr_sms = ndimage.gaussian_filter(smr, sigma=5, order=0)
    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/day/TOT_PREC'
    data = Dataset(f'{path}' + '/' + '01-05.TOT_PREC.nc')
    yr = data.variables['TOT_PREC'][...]
    yr = np.nansum(yr, axis=0)
    rt = smr/yr*100
    rt_sms = ndimage.gaussian_filter(rt, sigma=5, order=0)
    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{sim}/monsoon/TOT_PREC/'
    data = Dataset(f'{path}' + '/' + '01-05.TOT_PREC.cpm.nc')
    cpm = data.variables['TOT_PREC'][...]
    rg = np.nanmax(cpm, axis=0) - np.nanmin(cpm, axis=0)
    rg_sms = ndimage.gaussian_filter(rg, sigma=5, order=0)
    all_smr.append(smr)
    all_rt.append(rt)
    all_rg.append(rg)
    all_smr_sms.append(smr_sms)
    all_rt_sms.append(rt_sms)
    all_rg_sms.append(rg_sms)

# -------------------------------------------------------------------------------
# plot
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

ar = 1.0  # initial aspect ratio for first trial
wi = 5.5 * 3  # height in inches # 5.5 for 1 column
hi = wi * ar  # width in inches
nrow = 3
ncol = 3

axs, cs, ct = np.empty(9, dtype='object'), np.empty(9, dtype='object'), np.empty(9, dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.05, 0.05, 0.99, 0.90
wspace, hspace = 0.12, 0,2
gs = gridspec.GridSpec(3, 3, left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

for i in range(9):
    axs[i] = fig.add_subplot(gs[i], projection=rot_pole_crs)
    axs[i] = plotcosmo(axs[i])

levels = MaxNLocator(nbins=15).tick_values(0, 30)
cmap = cmc.davos_r
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

sims = ['Control', 'Reduced topography', 'Envelope topography']

for i in range(3):
    sim = sims[i]
    cs[i] = axs[i].pcolormesh(rlon, rlat, all_smr[i], cmap=cmap, norm=norm, shading="auto")
    ct[i] = axs[i].contour(rlon, rlat, all_smr_sms[i], levels=np.linspace(50, 300, 3, endpoint=True), colors='maroon', linewidths=1)
    axs[i].text(0.02, 1.02, f'{sim}', ha='left', va='bottom', transform=axs[i].transAxes, fontsize=14)

for i in 3, 4, 5:
    cs[i] = axs[i].pcolormesh(rlon, rlat, all_rg[i], cmap=cmap, norm=norm, shading="auto")
    ct[i] = axs[i].contour(rlon, rlat, all_rg_sms[i], levels=np.linspace(5, 20, 4, endpoint=True), colors='maroon', linewidths=1)

for i in 6, 7, 8:
    cs[i] = axs[i].pcolormesh(rlon, rlat, all_rt[i], cmap=cmap, norm=norm, shading="auto")
    ct[i] = axs[i].contour(rlon, rlat, all_rt_sms[i], levels=np.linspace(55, 80, 4, endpoint=True), colors='maroon', linewidths=1)

for i in 2, 5:
    cax = fig.add_axes([axs[i].get_position().x0, axs[i].get_position().y0 - 0.13, axs[i].get_position().width, 0.02])
    cbar = fig.colorbar(cs[i], cax=cax, orientation='horizontal', extend='both')
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.set_xlabel('mm/day', fontsize=13)

cax = fig.add_axes([axs[8].get_position().x0, axs[i].get_position().y0 - 0.13, axs[8].get_position().width, 0.02])
cbar = fig.colorbar(cs[i], cax=cax, orientation='horizontal', extend='both')
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_xlabel('%', fontsize=13)

fig.suptitle('Monsoon Annual Range', fontsize=16, fontweight='bold')

xmin, xmax = axs[1].get_xbound()
ymin, ymax = axs[1].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol + (1-top) + hspace
fig.set_figheight(wi * y2x_ratio)

plt.show()






