# -------------------------------------------------------------------------------
# modules
# %%
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
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_, wind
from pyproj import Transformer
import scipy.ndimage as ndimage
import matplotlib

font = {'size': 11}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
# %%
sims = ['PI', 'PD', 'LGM', 'PLIO']
mdpath = "/project/pr133/rxiang/data/echam5"
tsurf = {}
fname = {'01': 'jan.nc', '07': 'jul.nc'}
labels = {'PI': 'Pre-industrial', 'PD': 'Present day (1970-1995)', 'LGM': 'Last glacial maximum', 'PLIO': 'Mid-Pliocene'}
month = {'01': 'JAN', '07': 'JUL'}
colors = {'PI': '#f46d43', 'PD': 'steelblue', 'LGM': '#4393c3', 'PLIO': 'darkslateblue'}

lat = xr.open_dataset(f'{mdpath}/LGM/analysis/tsurf/mon/jan.nc')['lat'].values[:]
lon = xr.open_dataset(f'{mdpath}/LGM/analysis/tsurf/mon/jan.nc')['lon'].values[:]
lon_, lat_ = np.meshgrid(lon, lat)

data = xr.open_dataset(f'{mdpath}/LGM/analysis/tsurf/yr/tsurf_ydaymean.nc')['tsurf'].values[:, :, :] - 273.15

mask_na = (lat_ > 40) & (lat_ < 60) & (lon_ > 315) & (lon_ < 345)
mask_np = (lat_ > 10) & (lat_ < 40) & (lon_ > 160) & (lon_ < 220)
mask_na_3d = np.broadcast_to(mask_na, data.shape)
mask_np_3d = np.broadcast_to(mask_np, data.shape)

sims = ['PI', 'LGM']
for s in range(len(sims)):
    sim = sims[s]
    tsurf[sim] = {}
    tsurf[sim]['tsurf'] = {}
    tsurf[sim]['label'] = labels[sim]
    tsurf[sim]['color'] = colors[sim]
    na_, np_ = [], []
    # for mon in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']:
    #     data = xr.open_dataset(f'{mdpath}/{sim}/analysis/tsurf/mon/{mon}.nc')
    #     ts = data['tsurf'].values[0, :, :] - 273.15
    #     ts_ = np.mean(data[mask_na].flatten())
    #     na_.append(ts_)
    #     ts_ = np.mean(data[mask_np].flatten())
    #     np_.append(ts_)
    data = xr.open_dataset(f'{mdpath}/{sim}/analysis/tsurf/yr/tsurf_ydaymean.nc')['tsurf'].values[:, :, :] - 273.15
    mid = data*mask_na_3d
    mid[mid == 0] = np.nan
    na_ = np.nanmean(np.nanmean(mid, axis=1), axis=1)
    mid = data * mask_np_3d
    mid[mid == 0] = np.nan
    np_ = np.nanmean(np.nanmean(mid, axis=1), axis=1)

    tsurf[sim]['tsurf']['na'] = na_
    tsurf[sim]['tsurf']['np'] = np_
# -------------------------------------------------------------------------------
# plot
# %%
fig = plt.figure()
fig.set_size_inches(12, 7)
ax = fig.add_subplot(111)

cf_lines = []
cf_labels = []

lw = 2.
textsize = 20.
labelsize = 22.
titlesize = 28.
handlelength=2.

x = np.arange(0, 365, 1)

for sim in sims:
    color = tsurf[sim]['color']
    ax.plot(x, tsurf[sim]['tsurf']['np'][1:], lw=lw, color=color, label=tsurf[sim]['label'])
    cf_labels.append(tsurf[sim]['label'])

# Labels, legend, scale, limits
# ax.set_ylim([10**(-8), 1])
# ax.set_xlim([0, 20])
ax.set_xticks(np.linspace(0, 365, 12, endpoint=True))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.legend(loc='best', frameon=False,  prop={'size': textsize}, handlelength=handlelength)
ax.tick_params(axis='both', which='major', labelsize=22)
# ax.set_xlabel('Month', size=labelsize, labelpad=5)
ax.set_ylabel('Sea surface temperature ($^{o}$C)', size=labelsize)

# Remove some lines
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)
plt.grid(False)

# ax = plt.gca()
# ax.set_xlim([0, 7])
# ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
# ax1.set_yticks([0, 2, 4, 6, 8])
# ax.set_xticklabels(['0:00', '3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00'])
# ax.tick_params(axis='both', which='major', labelsize=10)
# plt.title('Summer wind speed at 500 hPa', fontsize=11)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper2/results/gm/"
fig.savefig(plotpath + 'sst.png', dpi=500, transparent='True')
plt.close(fig)







