import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def get_cumfreq(val, nvals=None):
    if nvals == None:
        nvals = len(val)
#     weights = np.ones_like(val)/float(len(val))
    weights = np.ones_like(val)/nvals
    max_int = np.max(val)
    num_bins = 10
    freq, base = np.histogram(val, bins=num_bins, range=(0, max_int),
                              weights=weights)
    freq = freq[::-1]
    cumfreq = np.cumsum(freq)
    cumfreq = cumfreq[::-1]
    base = base[0:-1]
    return base, cumfreq

# Get histogram
# def get_freq(val):
#     weights = np.ones_like(val)/float(len(val))
#     max_int = int(np.max(val)) + 1
#     num_bins = 10 * max_int
#     freq, base = np.histogram(val, bins=num_bins, range=(0, max_int),
#                               weights=weights)
#     base = base[0:-1]
#     return base, freq

# read data
sims = ['ctrl', 'topo2']
w = {}
row = []
# mdpath = "/project/pr133/rxiang/data/cosmo/"
# fname_w = '01-05.W.50000.smr.yhourmean.nc'

mdpath = '/project/pr133/rxiang/data/cosmo/'
fname_w = '01_W_50000.nc'

colors = {'ctrl': 'darkorange', 'topo2': 'steelblue'}
labels = {'ctrl': 'CTRL', 'topo2': 'TENV'}

# data = xr.open_dataset(f'{mdpath}EAS04_ctrl/diurnal/W/{fname_w}')
data = xr.open_dataset(f'{mdpath}EAS04_ctrl/3h3D/W/{fname_w}')
lon = data['lon'].values[:, :]
lat = data['lat'].values[:, :]

mask = (lat > 26) & (lat < 32) & (lon > 98) & (lon < 104)
mask3d = np.broadcast_to(mask, data['W'].values[:, 0, :, :].shape)

for s in range(len(sims)):
    sim = sims[s]
    data = xr.open_dataset(f'{mdpath}EAS04_{sim}/3h3D/W/{fname_w}')
    # data = np.nanmean(data['W'].values[:, 0, :, :], axis=0)
    data = data['W'].values[:, 0, :, :]
    # row = data[(lat > 26) & (lat < 32) & (lon > 98) & (lon < 104)]
    row = data[mask3d].flatten()
    del data
    w[sim] = {}
    w[sim]['w'] = row
    row = []

    w_pos = w[sim]['w'][w[sim]['w'] > 0.]
    w_neg = w[sim]['w'][w[sim]['w'] < 0.]

    nvals_w = len(w[sim]['w'])

    base_w_pos, cf_w_pos = get_cumfreq(w_pos, nvals_w)
    base_w_neg, cf_w_neg = get_cumfreq(-w_neg, nvals_w)
    base_w_neg *= -1.

    w[sim]['base_w_pos_850'] = base_w_pos
    w[sim]['base_w_neg_850'] = base_w_neg
    w[sim]['cf_w_pos_850'] = cf_w_pos
    w[sim]['cf_w_neg_850'] = cf_w_neg

    w[sim]['color'] = colors[sim]
    w[sim]['label'] = labels[sim]

del mask, mask3d, lon, lat

fig = plt.figure()
fig.set_size_inches(10, 8)
ax = fig.add_subplot(111)

cf_lines = []
cf_labels = []

loc_x = 0.5
incr = .3

lw = 2.
textsize = 20.
labelsize = 24.
titlesize = 28.
handlelength=2.

for sim in sims:
    color = w[sim]['color']
    ax.plot(w[sim]['base_w_pos_850'], w[sim]['cf_w_pos_850'], lw=lw, color=color, label=w[sim]['label'])
    cf_lines += ax.plot(w[sim]['base_w_neg_850'], w[sim]['cf_w_neg_850'], lw=lw)
    cf_labels.append(w[sim]['label'])

    # ax.plot(loc_x, w[sim]['cf_w_pos_850'][0], '<', ms=7)
    # ax.plot(-loc_x, w[sim]['cf_w_neg_850'][0], '>', ms=7)
    loc_x += incr

# Labels, legend, scale, limits
ax.set_yscale('log')
ax.set_ylim([10**(-8), 1])
ax.set_xlim([-10, 15])
ax.legend(loc='best', frameon=False,  prop={'size': textsize}, handlelength=handlelength)
ax.tick_params(axis='both', which='major', labelsize=labelsize)
ax.set_xlabel('w @ 500 hPa [m/s]', size=labelsize)
ax.set_ylabel('Cumulative Frequency', size=labelsize)

# Remove some lines
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)
plt.grid(False)

plt.show()
plotpath = "/project/pr133/rxiang/figure/analysis/EAS04/topo2/smr/"
fig.savefig(plotpath + 'cumwind500.png', dpi=500)
plt.close(fig)
# Save and close
# with plt.rc_context({'savefig.format': 'pdf'}):
#     plt.savefig('plots/w_new/w_' + str(p_hPa) + '_' + str(res) + '.pdf', bbox_inches='tight')

