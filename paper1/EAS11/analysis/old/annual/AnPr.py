# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

# -------------------------------------------------------------------------------
# import data
#
sims = ['ctrl', 'topo1', 'topo2']
mdvname = 'TOT_PREC'
mdpath = "/project/pr133/rxiang/data/cosmo/"

# -------------------------------------------------------------------------------
# read model data
# Monsoon subregions
# ISM (5°N-30°N, 70°E-100°E), EASM (20°N-45°N, 100°E-150°E), WNPSM (5°N-20°N, 100°E-170°E)

ism, easm, wnpsm = [], [], []
an_ism, an_easm, an_wnpsm = [], [], []
filename = f'01-05_{mdvname}.nc'

lon = xr.open_dataset(f'{mdpath}EAS11_ctrl/mon/{mdvname}/{filename}')['lon'].values[:, :]
lat = xr.open_dataset(f'{mdpath}EAS11_ctrl/mon/{mdvname}/{filename}')['lat'].values[:, :]
data = xr.open_dataset(f'{mdpath}EAS11_ctrl/mon/{mdvname}/{filename}')[mdvname].values[:, :, :]

mask_ism = (lat > 5) & (lat < 30) & (lon > 70) & (lon < 100)
mask3d_ism = np.broadcast_to(mask_ism, data.shape)
mask_easm = (lat > 20) & (lat < 45) & (lon > 100) & (lon < 150)
mask3d_easm = np.broadcast_to(mask_easm, data.shape)
mask_wnpsm = (lat > 5) & (lat < 20) & (lon > 100) & (lon < 170)
mask3d_wnpsm = np.broadcast_to(mask_wnpsm, data.shape)

for i in range(len(sims)):
    sim = sims[i]
    data = xr.open_dataset(f'{mdpath}EAS11_{sim}/mon/{mdvname}/{filename}')[mdvname].values[:, :, :]
    data_ism = [d[m] for d, m in zip(data, mask3d_ism)]
    data_easm = [d[m] for d, m in zip(data, mask3d_easm)]
    data_wnpsm = [d[m] for d, m in zip(data, mask3d_wnpsm)]
    for j in range(len(data_wnpsm)):
        an_ism.append(np.mean(data_ism[j]))
        an_easm.append(np.mean(data_easm[j]))
        an_wnpsm.append(np.mean(data_wnpsm[j]))
    ism.append(an_ism)
    easm.append(an_easm)
    wnpsm.append(an_wnpsm)
    an_ism, an_easm, an_wnpsm = [], [], []

# -------------------------------------------------------------------------------
# plot
colors = {
    'green': '#ccebc5',
    'orange': '#fdcdac',
    'yellow': '#ffffbf',
    'blue': '#a6cee3',
    'purple': '#bebada'
}
label = ['Control', 'Reduced topography', 'Envelope topography']
color = [colors['orange'], colors['blue'], colors['purple']]
marker = ['o', '^', '*']

x = np.arange(0, 12, 1)
plt.figure(figsize=(6, 6), constrained_layout=True)
plt.subplot(311)
for i in range(3):
    plt.plot(x, ism[i], color=color[i], label=label[i], marker=marker[i])
plt.legend(loc='upper left', fontsize=8, frameon=False, ncol=1, columnspacing=0.35)
plt.title('ISM')
ax1 = plt.gca()
ax1.set_xlim([0, 11])
ax1.set_xticks([])
ax1.set_yticks([2, 4, 6, 8, 10])
ax1.set_xticklabels([])
ax1.text(-0.05, 1.07, '[mm/day]', transform=ax1.transAxes)
plt.subplot(312)
for i in range(3):
    plt.plot(x, easm[i], color=color[i], label=label[i], marker=marker[i])
plt.title('EASM')
ax2 = plt.gca()
ax2.set_xlim([0, 11])
ax2.set_xticks([])
ax2.set_yticks([2, 3, 4, 5, 6])
ax2.set_xticklabels([])
plt.subplot(313)
for i in range(3):
    plt.plot(x, wnpsm[i], color=color[i], label=label[i], marker=marker[i])
plt.title('WNPSM')
ax3 = plt.gca()
ax3.set_xlim([0, 11])
ax3.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
ax3.set_yticks([4, 6, 8, 10, 12])
ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()



