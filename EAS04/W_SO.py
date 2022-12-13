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
sims = ['ctrl', 'topo1', 'topo2']
mdpath = "/project/pr133/rxiang/data/cosmo/"

# -------------------------------------------------------------------------------
# read model data
# subregions
# topo2 (26°N-32°N, 98°E-104°E)
# /scratch/snx3000/rxiang/data/cosmo/EAS04_ctrl/24h/W_SO

wso = {}
wsoo = []
fname = 'W_SO.nc'

lon = xr.open_dataset(f'{mdpath}EAS04_ctrl/24h/W_SO/{fname}')['lon'].values[:, :]
lat = xr.open_dataset(f'{mdpath}EAS04_ctrl/24h/W_SO/{fname}')['lat'].values[:, :]
data = xr.open_dataset(f'{mdpath}EAS04_ctrl/24h/W_SO/{fname}')['W_SO'].values[:, 0, :, :]


# mask = (lat > 19) & (lat < 23) & (lon > 90) & (lon < 95)
mask = (lat > 28) & (lat < 31) & (lon > 99) & (lon < 102)
# mask = (lat > 25) & (lat < 32) & (lon > 99) & (lon < 105)
# mask = (lat > 28) & (lat < 31) & (lon > 104) & (lon < 108) #sichuan basin
mask3d = np.broadcast_to(mask, data.shape)

colors = {'ctrl': 'darkorange', 'topo1': 'olivegreen', 'topo2': 'steelblue'}
labels = {'ctrl': 'CTRL', 'topo1': 'TRED', 'topo2': 'TENV'}

for i in range(len(sims)):
    sim = sims[i]
    data = xr.open_dataset(f'{mdpath}EAS04_{sim}/24h/W_SO/{fname}')['W_SO'].values[:, 0, :, :]
    data = [d[m] for d, m in zip(data, mask3d)]
    for j in range(len(data)):
        wsoo.append(np.nanmean(data[j]))
    wso[sim] = {}
    wso[sim]['pr'] = wsoo
    wso[sim]['color'] = colors[sim]
    wso[sim]['label'] = labels[sim]
    prr = []

fig = plt.figure()
fig.set_size_inches(12, 7)
ax = fig.add_subplot(111)

cf_lines = []
cf_labels = []

lw = 2.
textsize = 20.
labelsize = 24.
titlesize = 28.
handlelength=2.

x = np.arange(0, 1886, 1)

for sim in sims:
    color = wso[sim]['color']
    ax.plot(x, wso[sim]['pr'], lw=lw, color=color, label=wso[sim]['label'])
    cf_labels.append(wso[sim]['label'])

# Labels, legend, scale, limits
# ax.set_ylim([10**(-8), 1])
# ax.set_xlim([0, 20])
ax.set_xticks(np.linspace(0, 1886, 12, endpoint=True))
# ax.set_xticklabels(['0', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22'])
ax.legend(loc='best', frameon=False,  prop={'size': textsize}, handlelength=handlelength)
ax.tick_params(axis='both', which='major', labelsize=labelsize)
# ax.set_xlabel('Hour (UTC)', size=labelsize)
# ax.set_ylabel('mm h$^{-1}$', size=labelsize)

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
# plotpath = "/project/pr133/rxiang/figure/analysis/EAS04/topo2/"
# fig.savefig(plotpath + '24h_bob.png', dpi=500)
# plt.close(fig)
