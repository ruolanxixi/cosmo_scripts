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
mdpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/mon/TOT_PREC/"
erapath = "/project/pr133/rxiang/data/era5/pr/remap/"
crupath = "/project/pr133/rxiang/data/obs/pr/cru/remap/"
imergpath = "/project/pr133/rxiang/data/obs/pr/IMERG/remap/"

md = xr.open_dataset(f'{mdpath}' + '2001-2005.TOT_PREC.nc')['TOT_PREC'].values[:, :, :]
lat = xr.open_dataset(f'{mdpath}' + '2001-2005.TOT_PREC.nc')['lat'].values[...]
lon = xr.open_dataset(f'{mdpath}' + '2001-2005.TOT_PREC.nc')['lon'].values[...]
mask_bc = (lat > 20) & (lat < 40) & (lon > 90) & (lon < 110)
mask3d_bc = np.broadcast_to(mask_bc, md.shape)

era = xr.open_dataset(f'{erapath}' + 'era5.mo.2001-2005.mon.remap.nc')['tp'].values[:, :, :] * 1000
imerg = xr.open_dataset(f'{imergpath}' + 'IMERG.ydaymean.2001-2005.mon.remap.nc4')['pr'].values[:, :, :]

md_bc = [d[m] for d, m in zip(md, mask3d_bc)]
era_bc = [d[m] for d, m in zip(era, mask3d_bc)]
imerg_bc = [d[m] for d, m in zip(imerg, mask3d_bc)]

mdpr_bc, erapr_bc, imergpr_bc = [], [], []
for j in range(len(md_bc)):
    mdpr_bc.append(np.nanmean(md_bc[j]))
    erapr_bc.append(np.nanmean(era_bc[j]))
    imergpr_bc.append(np.mean(imerg_bc[j]))

mdpr = np.nanmean(np.nanmean(md[:, 10:-10, 10:-10], axis=1), axis=1)
erapr = np.nanmean(np.nanmean(era[:, 10:-10, 10:-10], axis=1), axis=1)
imergpr = np.nanmean(np.nanmean(imerg[:, 10:-10, 10:-10], axis=1), axis=1)

pr = [mdpr, erapr, imergpr]
pr_bc = [mdpr_bc, erapr_bc, imergpr_bc]

mdpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/mon/T_2M/"
crupath = "/project/pr133/rxiang/data/obs/tmp/cru/remap/"

md = xr.open_dataset(f'{mdpath}' + '2001-2005.T_2M.nc')['T_2M'].values[:, :, :] - 273.15
era = xr.open_dataset(f'{erapath}' + 'era5.mo.2001-2005.mon.remap.nc')['t2m'].values[:, :, :] - 273.15
cru = xr.open_dataset(f'{crupath}' + 'cru.2001-2005.05.mon.remap.nc')['tmp'].values[:, :, :]

md_bc = [d[m] for d, m in zip(md, mask3d_bc)]
era_bc = [d[m] for d, m in zip(era, mask3d_bc)]
cru_bc = [d[m] for d, m in zip(cru, mask3d_bc)]

mdt2m_bc, erat2m_bc, crut2m_bc = [], [], []
for j in range(len(md_bc)):
    mdt2m_bc.append(np.nanmean(md_bc[j]))
    erat2m_bc.append(np.nanmean(era_bc[j]))
    crut2m_bc.append(np.nanmean(cru_bc[j]))

mdt2m = np.nanmean(np.nanmean(md[:, 10:-10, 10:-10], axis=1), axis=1)
erat2m = np.nanmean(np.nanmean(era[:, 10:-10, 10:-10], axis=1), axis=1)
crut2m = np.nanmean(np.nanmean(cru[:, 10:-10, 10:-10], axis=1), axis=1)

t2m = [mdt2m, erat2m, crut2m]
t2m_bc = [mdt2m_bc, erat2m_bc, crut2m_bc]

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

color = sns.color_palette('Set3', 7)[4:]
marker = ['o', '^', '*']

label = ['COSMO', 'ERA5', 'IMERG']
x = np.arange(0, 12, 1)
fig = plt.figure(figsize=(7, 3), constrained_layout=True)
plt.subplot(121)
for i in range(3):
    plt.plot(x, pr[i], color=color[i], label=label[i], linewidth=2)
plt.legend(loc='upper left', fontsize=10, frameon=False, ncol=1, labelspacing=0.35)
# plt.title('Annual Precipitation Cycle')
plt.ylabel("Precipitation (mm/day)")
ax1 = plt.gca()
ax1.set_xlim([0, 11])
ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
ax1.set_yticks([1, 2, 3, 4, 5, 6])
ax1.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])

plt.subplot(122)
label = ['COSMO', 'ERA5']
for i in range(2):
    plt.plot(x, t2m[i], color=color[i], label=label[i], linewidth=2)
plt.legend(loc='upper left', fontsize=10, frameon=False, ncol=1, labelspacing=0.35)
# plt.title('Annual Surface Temperature Cycle')
plt.ylabel("Temperature ($^{o}C$)")
ax2 = plt.gca()
# ax2.grid(color='lightgray', linestyle='--', alpha=0.5)
ax2.set_xlim([0, 11])
ax2.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
ax2.set_yticks([5, 10, 15, 20, 25])
ax2.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
plt.show()
plotpath = "/project/pr133/rxiang/figure/EAS11/validation/"
fig.savefig(plotpath + 'seasonal.png', dpi=500)
plt.close()

label = ['COSMO', 'ERA5', 'IMERG']
x = np.arange(0, 12, 1)
fig = plt.figure(figsize=(7, 3), constrained_layout=True)
plt.subplot(121)
for i in range(3):
    plt.plot(x, pr_bc[i], color=color[i], label=label[i], linewidth=2)
plt.legend(loc='upper left', fontsize=10, frameon=False, ncol=1, labelspacing=0.35)
plt.ylabel("Precipitation (mm/day)")
# plt.title('Annual Precipitation Cycle (BECCY)')
ax1 = plt.gca()
ax1.set_xlim([0, 11])
ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
ax1.set_yticks([0, 2, 4, 6, 8, 10])
ax1.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
ax1.text(0.97, 0.97, 'BECCY', ha='right', va='top', transform=ax1.transAxes)
plt.subplot(122)
label = ['COSMO', 'ERA5', 'CRU']
for i in range(3):
    plt.plot(x, t2m_bc[i], color=color[i], label=label[i], linewidth=2)
plt.legend(loc='upper left', fontsize=10, frameon=False, ncol=1, labelspacing=0.35)
# plt.title('Annual Surface Temperature Cycle (BECCY)')
plt.ylabel("Temperature ($^{o}C$)")
ax2 = plt.gca()
ax2.text(0.97, 0.97, 'BECCY', ha='right', va='top', transform=ax2.transAxes)
ax2.set_xlim([-0, 11])
ax2.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
ax2.set_yticks([0, 5, 10, 15, 20])
ax2.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
plt.show()
plotpath = "/project/pr133/rxiang/figure/EAS11/validation/"
fig.savefig(plotpath + 'seasonal_bc.png', dpi=500)
plt.close()
