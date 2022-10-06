# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns

font = {'size': 12}
matplotlib.rc('font', **font)
# -------------------------------------------------------------------------------
# import data
#
mdpath = "/project/pr133/rxiang/data/cosmo/EAS04_ctrl/mon/TOT_PREC/"
erapath = "/project/pr133/rxiang/data/era5/pr/remap/"
crupath = "/project/pr133/rxiang/data/obs/pr/cru/remap/"
imergpath = "/project/pr133/rxiang/data/obs/pr/IMERG/remap/"

md = xr.open_dataset(f'{mdpath}' + '2001-2005.TOT_PREC.nc')['TOT_PREC'].values[:, :, :]
lat = xr.open_dataset(f'{mdpath}' + '2001-2005.TOT_PREC.nc')['lat'].values[...]
lon = xr.open_dataset(f'{mdpath}' + '2001-2005.TOT_PREC.nc')['lon'].values[...]

era = xr.open_dataset(f'{erapath}' + 'era5.mo.2001-2005.mon.remap.04.nc')['tp'].values[:, :, :] * 1000
imerg = xr.open_dataset(f'{imergpath}' + 'IMERG.ydaymean.2001-2005.mon.remap.04.nc4')['pr'].values[:, :, :]
cru = xr.open_dataset(f'{crupath}' + 'cru.2001-2005.05.mon.remap.04.nc')['pre'].values[:, :, :]

mdpr = np.nanmean(np.nanmean(md[:, 130:-10, 10:-10], axis=1), axis=1)
erapr = np.nanmean(np.nanmean(era[:, 130:-10, 10:-10], axis=1), axis=1)
imergpr = np.nanmean(np.nanmean(imerg[:, 130:-10, 10:-10], axis=1), axis=1)
crupr = np.nanmean(np.nanmean(cru[:, 130:-10, 10:-10], axis=1), axis=1)

pr = [mdpr, erapr, imergpr, crupr]

mdpath = "/project/pr133/rxiang/data/cosmo/EAS04_ctrl/mon/T_2M/"
crupath = "/project/pr133/rxiang/data/obs/tmp/cru/remap/"

md = xr.open_dataset(f'{mdpath}' + '2001-2005.T_2M.nc')['T_2M'].values[:, :, :] - 273.15
era = xr.open_dataset(f'{erapath}' + 'era5.mo.2001-2005.mon.remap.04.nc')['t2m'].values[:, :, :] - 273.15
cru = xr.open_dataset(f'{crupath}' + 'cru.2001-2005.05.mon.remap.04.nc')['tmp'].values[:, :, :]

mdt2m = np.nanmean(np.nanmean(md[:, 130:-10, 10:-10], axis=1), axis=1)
erat2m = np.nanmean(np.nanmean(era[:, 130:-10, 10:-10], axis=1), axis=1)
crut2m = np.nanmean(np.nanmean(cru[:, 130:-10, 10:-10], axis=1), axis=1)

t2m = [mdt2m, erat2m, crut2m]

# -------------------------------------------------------------------------------
# plot
colors = {
    'green': '#ccebc5',
    'orange': '#fdcdac',
    'yellow': '#ffffbf',
    'blue': '#a6cee3',
    'purple': '#bebada'
}
color = [colors['orange'], colors['blue'], colors['purple']]

lists = sns.color_palette('Set3', 12)
color = [lists[4], lists[5], lists[6], lists[9]]
marker = ['o', '^', '*']

label = ['COSMO', 'ERA5', 'IMERG', 'CRU']
x = np.arange(0, 12, 1)
fig = plt.figure(figsize=(4, 6), constrained_layout=True)
plt.subplot(211)
for i in range(4):
    plt.plot(x, pr[i], color=color[i], label=label[i], linewidth=2)
plt.legend(loc='upper left', fontsize=10, frameon=False, ncol=1, labelspacing=0.35)
# plt.title('Annual Precipitation Cycle')
plt.ylabel("Precipitation (mm/day)")
ax1 = plt.gca()
ax1.set_xlim([0, 11])
ax1.set_xticks([])
ax1.set_yticks([0, 2, 4, 6, 8])
# ax1.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])

plt.subplot(212)
label = ['COSMO', 'ERA5', 'CRU']
for i in range(3):
    plt.plot(x, t2m[i], color=color[i], label=label[i], linewidth=2)
plt.legend(loc='upper left', fontsize=10, frameon=False, ncol=1, labelspacing=0.35)
# plt.title('Annual Surface Temperature Cycle')
plt.ylabel("Temperature ($^{o}C$)")
ax2 = plt.gca()
# ax2.grid(color='lightgray', linestyle='--', alpha=0.5)
ax2.set_xlim([0, 11])
ax2.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
ax2.set_yticks([0, 5, 10, 15, 20])
ax2.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
plt.show()
plotpath = "/project/pr133/rxiang/figure/EAS04/validation/"
fig.savefig(plotpath + 'seasonal.png', dpi=500)
plt.close()
