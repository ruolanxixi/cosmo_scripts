# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns

font = {'size': 11}
matplotlib.rc('font', **font)
# -------------------------------------------------------------------------------
# import data
#
sims = ['ctrl', 'topo1']
mdpath = "/project/pr133/rxiang/data/cosmo/"

# -------------------------------------------------------------------------------
# read model data
# subregions
# topo1 (26°N-32°N, 98°E-104°E)

wind, di_data = [], []
fname_u = '01-05.U.50000.smr.yhourmean.nc'
fname_v = '01-05.V.50000.smr.yhourmean.nc'

lon = xr.open_dataset(f'{mdpath}EAS04_ctrl/diurnal/U/{fname_u}')['lon'].values[:, :]
lat = xr.open_dataset(f'{mdpath}EAS04_ctrl/diurnal/U/{fname_u}')['lat'].values[:, :]
data = xr.open_dataset(f'{mdpath}EAS04_ctrl/diurnal/U/{fname_u}')['U'].values[:, 0, :, :]

# mask = (lat > 26) & (lat < 32) & (lon > 98) & (lon < 104)
# mask3d = np.broadcast_to(mask, data.shape)

for i in range(len(sims)):
    sim = sims[i]
    data_u = xr.open_dataset(f'{mdpath}EAS04_{sim}/diurnal/U/{fname_u}')['U'].values[:, 0, :, :]
    data_v = xr.open_dataset(f'{mdpath}EAS04_{sim}/diurnal/V/{fname_v}')['V'].values[:, 0, :, :]
    # data_u = [d[m] for d, m in zip(data_u, mask3d)]
    # data_v = [d[m] for d, m in zip(data_v, mask3d)]
    for j in range(len(data_u)):
        di_data.append(np.nanmean(np.sqrt(data_v[j]**2+data_u[j]**2)))
    wind.append(di_data)
    di_data = []

lists = sns.color_palette('Set3', 12)
color = [lists[4], lists[5], lists[6], lists[9]]
marker = ['o', '^', '*']

label = ['Control', 'Reduced topography']
x = np.arange(0, 8, 1)
fig = plt.figure(figsize=(4, 3), constrained_layout=True)
for i in range(2):
    plt.plot(x, wind[i], color=color[i], label=label[i], linewidth=2)
plt.legend(loc='lower left', fontsize=10, frameon=False, ncol=1, labelspacing=0.35)
# plt.title('Annual Precipitation Cycle')
plt.ylabel("Wind speed (m/s)")
ax = plt.gca()
ax.set_xlim([0, 7])
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
# ax1.set_yticks([0, 2, 4, 6, 8])
ax.set_xticklabels(['0:00', '3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00'])
ax.tick_params(axis='both', which='major', labelsize=10)
plt.title('Summer wind speed at 500 hPa', fontsize=11)

plt.show()
plotpath = "/project/pr133/rxiang/figure/EAS04/analysis/monsoon/topo1/"
fig.savefig(plotpath + 'smr_wind.png', dpi=500)
plt.close(fig)
